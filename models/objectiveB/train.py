"""
Objective B - calibrate and evaluate the outbreak detector.

There is nothing to fit by gradient: the detector is statistical. What is learned are its thresholds
(p-value, minimum count, minimum excess ratio, symptom-signature agreement). They are calibrated on the EARLIER part
of the timeline and scored on a LATER held-out part (a time split; a random split would leak inside an outbreak).

    calibration   first day the seasonal guard has a year of history (2025-01-20) to the day before --test-start
    test          --test-start (default 2025-10-01) to the end
The first year is warm-up: with no year-ago reference the detector would announce every seasonal surge as news.

Ground truth comes from the generator's labels (outbreak_active / outbreak_id in encounters.csv), read here for
scoring only. The detector itself sees node summaries only.

Calibration picks the setting with the best recall of "linked" outbreaks (the same disease active at another hospital in
the same region, which is how the generator spreads an outbreak), among settings that send at most --max-false
false notifications per year (consecutive alert days for one disease count as one notification, as in the store).
An alert is correct when at least two of its hospitals are inside (or within a week of) a labelled outbreak of that
disease.

Writes model/outbreak_model.json (thresholds) and model/meta.json (metrics).
Run from the project root (builds the node summaries first when the store is empty):
    python models/objectiveB/train.py
    python models/objectiveB/train.py --rebuild --test-start 2025-09-01 --max-false 3
"""

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / "models" / "objectiveE").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objectiveB.detect import DEFAULTS, MODEL_DIR, MODEL_FILE, META_FILE, OutbreakDetector, Panel, Stats  # noqa: E402
from models.objectiveB.node import DEFAULT_DATA, build  # noqa: E402
from models.objectiveB.store import MERGE_WITHIN_DAYS, Store  # noqa: E402

LAG_TOLERANCE = (6, 7)         # a site counts as "in an outbreak" from 6 days after it ended to 7 days before it starts
DETECT_GRACE = 3               # days after an outbreak ends in which a detection still counts
GRID = {"p_thresh": [0.02, 0.01, 0.005, 0.001, 0.0001], "min_ratio": [1.3, 1.5, 2.0, 2.5],
        "min_sim": [0.3, 0.5, 0.7], "min_count": [4, 6, 8]}
ATYP_P_GRID = [0.01, 0.005, 0.001, 0.0005, 0.0001]
MAX_NOVEL_DAYS_PER_YEAR = 2.0  # the generator injects no novel pathogen, so every novel alert here is a false alarm


# ------------------------------------------------------------------ ground truth
class Truth:
    """Labelled outbreaks as arrays on the panel's hospital x disease x day grid."""

    def __init__(self, panel, data_path):
        df = pd.read_csv(data_path, usecols=["hospital_id", "disease_id", "encounter_date", "outbreak_active", "outbreak_id"])
        df = df[df["outbreak_active"] == 1]
        ep = df.groupby("outbreak_id").agg(h=("hospital_id", "first"), d=("disease_id", "first"),
                                           s=("encounter_date", "min"), e=("encounter_date", "max"))
        hmap = {h: i for i, h in enumerate(panel.hospitals)}
        dmap = {d: i for i, d in enumerate(panel.diseases)}
        T0 = panel.days[0]
        self.episodes = [(hmap[r.h], dmap[r.d], (pd.Timestamp(r.s) - T0).days, (pd.Timestamp(r.e) - T0).days)
                         for r in ep.itertuples() if r.h in hmap and r.d in dmap]
        shape = (len(panel.hospitals), len(panel.diseases), len(panel.days))
        self.active = np.zeros(shape, dtype=bool)
        for h, k, s, e in self.episodes:
            self.active[h, k, s:e + 1] = True
        T = shape[-1]
        c = np.concatenate([np.zeros(shape[:2] + (1,)), np.cumsum(self.active, axis=-1)], axis=-1)
        lo = np.maximum(np.arange(T) - LAG_TOLERANCE[0], 0)
        hi = np.minimum(np.arange(T) + LAG_TOLERANCE[1], T - 1)
        self.near = (c[..., hi + 1] - c[..., lo]) > 0                # any outbreak within the tolerance window
        # outbreaks with the same disease active at another hospital of the same region: what cross-site detection is for
        region = [panel.region[h] for h in panel.hospitals]
        self.linked = {i for i, (h, k, s, e) in enumerate(self.episodes)
                       if any(k2 == k and h2 != h and region[h2] == region[h] and s2 <= e and s <= e2
                              for h2, k2, s2, e2 in self.episodes)}


# ------------------------------------------------------------------ scoring
def count_notifications(flags):
    """What a user receives: the store merges alert days of one disease that are <= MERGE_WITHIN_DAYS apart into one
    alert. flags = {disease: {day: alert_was_correct}}. Returns (correct notifications, false notifications); a
    notification is correct when any of its days was."""
    good = bad = 0
    for days in flags.values():
        ts = sorted(days)
        start = 0
        for i, t in enumerate(ts):
            if i + 1 == len(ts) or ts[i + 1] - t > MERGE_WITHIN_DAYS:
                if any(days[x] for x in ts[start:i + 1]):
                    good += 1
                else:
                    bad += 1
                start = i + 1
    return good, bad


def evaluate(det, stats, truth, params, t_lo, t_hi):
    """Score the detector over days t_lo..t_hi (inclusive) with thresholds `params`."""
    panel = stats.panel
    sig = stats.signals(params)
    hmap = {h: i for i, h in enumerate(panel.hospitals)}
    covered, tp, fp, novel_days, alert_days, flags = set(), 0, 0, 0, 0, {}
    for t in range(t_lo, t_hi + 1):
        alerts = det.alerts_at(stats, sig, t, params)
        alert_days += bool(alerts)
        for a in alerts:
            if a["kind"] == "possible_novel":
                novel_days += 1
                continue
            k = panel.diseases.index(a["disease_id"])
            hs = [hmap[s["hospital_id"]] for s in a["sites"]]
            covered.update((h, k, t) for h in hs)
            ok = sum(bool(truth.near[h, k, t]) for h in hs) >= 2
            tp, fp = tp + ok, fp + (not ok)
            flags.setdefault(k, {})[t] = ok
    good, bad = count_notifications(flags)
    years = (t_hi - t_lo + 1) / 365.0

    eps = [(i, *e) for i, e in enumerate(truth.episodes) if t_lo <= e[2] <= t_hi]
    site_hit, alert_hit, delays, linked_n, linked_hit = 0, 0, [], 0, 0
    sig_known = sig[0]
    for i, h, k, s, e in eps:
        end = min(e + DETECT_GRACE, len(panel.days) - 1)
        site_hit += bool(sig_known[h, k, s:end + 1].any())
        first = next((t for t in range(s, end + 1) if (h, k, t) in covered), None)
        alert_hit += first is not None
        if first is not None:
            delays.append(first - s)
        if i in truth.linked:
            linked_n += 1
            linked_hit += first is not None
    n = max(len(eps), 1)
    return {
        "days": t_hi - t_lo + 1, "episodes": len(eps), "linked_episodes": linked_n,
        "site_signal_recall": round(site_hit / n, 3),
        "alert_recall": round(alert_hit / n, 3),
        "linked_alert_recall": round(linked_hit / max(linked_n, 1), 3),
        "median_delay_days": float(np.median(delays)) if delays else None,
        "alert_days": alert_days, "notifications": good + bad,
        "notification_precision": round(good / max(good + bad, 1), 3),
        "false_notifications_per_year": round(bad / years, 1),
        "novel_alert_days_per_year": round(novel_days / years, 1),
    }


def first_eval_day(p):
    """First day with a year-ago reference for the seasonal guard (plus two weeks of margin)."""
    return max(p["season_shifts"]) + 14


def calibrate(det, stats, truth, t_lo, t_hi, max_false):
    """Grid search the known-disease thresholds, then the atypical-share threshold."""
    keys = list(GRID)
    results = []
    for values in itertools.product(*GRID.values()):
        params = {**DEFAULTS, **dict(zip(keys, values)), "atyp_p_thresh": 1e-12}   # novel path off while tuning
        results.append((params, evaluate(det, stats, truth, params, t_lo, t_hi)))
    feasible = [r for r in results if r[1]["false_notifications_per_year"] <= max_false]
    if feasible:
        best = max(feasible, key=lambda r: (r[1]["linked_alert_recall"], r[1]["notification_precision"],
                                            -r[1]["false_notifications_per_year"]))
    else:
        print(f"  warning: no setting stays under {max_false} false notifications/year; using the quietest one")
        best = min(results, key=lambda r: r[1]["false_notifications_per_year"])
    params = dict(best[0])
    for p in ATYP_P_GRID:                       # most sensitive atypical threshold that stays quiet
        params["atyp_p_thresh"] = p
        if evaluate(det, stats, truth, params, t_lo, t_hi)["novel_alert_days_per_year"] <= MAX_NOVEL_DAYS_PER_YEAR:
            break
    else:
        params["atyp_p_thresh"] = ATYP_P_GRID[-1] / 10
    return params, len(results), len(feasible)


def show(name, m):
    delay = "n/a" if m["median_delay_days"] is None else f"{m['median_delay_days']:g} d"
    print(f"  {name:11s} {m['episodes']:3d} outbreaks ({m['linked_episodes']} linked) | recall: site-signal {m['site_signal_recall']:.0%}, "
          f"alerted {m['alert_recall']:.0%}, linked {m['linked_alert_recall']:.0%} | median delay {delay} | "
          f"{m['notifications']} notifications, precision {m['notification_precision']:.0%}, false/yr {m['false_notifications_per_year']:g}, "
          f"novel alert days/yr {m['novel_alert_days_per_year']:g}")


def main():
    ap = argparse.ArgumentParser(description="Calibrate and evaluate the outbreak detector")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="encounters CSV (labels for scoring; summaries are built from it)")
    ap.add_argument("--db", help="SQLite file (default: data/outbreak.db or $OUTBREAK_DB)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the node summaries first")
    ap.add_argument("--test-start", default="2025-10-01", help="first held-out day")
    ap.add_argument("--max-false", type=float, default=6.0, help="false notifications per year allowed when calibrating")
    args = ap.parse_args()

    store = Store(args.db)
    if args.rebuild or store.day_range()[0] is None:
        print("Building node summaries ...")
        build(args.data, store, verbose=False)
    panel = Panel(store.load_summaries())
    stats = Stats(panel, DEFAULTS)
    truth = Truth(panel, args.data)
    det = OutbreakDetector(params={})
    test_t = int((pd.Timestamp(args.test_start) - panel.days[0]).days)
    last = len(panel.days) - 1
    first = first_eval_day(DEFAULTS)
    if not first + 30 < test_t < last:
        sys.exit(f"--test-start must fall between {panel.days[min(first + 31, last)].date()} and {panel.days[last].date()}")
    print(f"{len(panel.hospitals)} hospitals, {len(panel.days)} days, {len(truth.episodes)} labelled outbreaks "
          f"({len(truth.linked)} linked to another hospital in their region)")
    print(f"Calibrating on {panel.days[first].date()} .. {panel.days[test_t - 1].date()}, "
          f"testing on {panel.days[test_t].date()} .. {panel.days[last].date()}")

    params, n_grid, n_ok = calibrate(det, stats, truth, first, test_t - 1, args.max_false)
    print(f"Searched {n_grid} settings ({n_ok} within the false-alert budget). Chosen:")
    print("  " + ", ".join(f"{k}={params[k]}" for k in [*GRID, "atyp_p_thresh"]))

    metrics = {"calibration": evaluate(det, stats, truth, params, first, test_t - 1),
               "test": evaluate(det, stats, truth, params, test_t, last),
               "test_with_default_thresholds": evaluate(det, stats, truth, DEFAULTS, test_t, last)}
    print("\nResults")
    show("calibration", metrics["calibration"])
    show("TEST", metrics["test"])
    show("defaults", metrics["test_with_default_thresholds"])

    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_FILE.write_text(json.dumps({"params": params}, indent=2))
    meta = {"model": "poisson-baseline + cross-site symptom-signature agreement", "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "calibration_days": [str(panel.days[first].date()), str(panel.days[test_t - 1].date())],
            "test_days": [str(panel.days[test_t].date()), str(panel.days[last].date())],
            "max_false_notifications_per_year": args.max_false, "metrics": metrics,
            "note": "the generator injects no novel pathogen, so the possible-novel path is only checked for false alarms"}
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {MODEL_FILE}\nWrote {META_FILE}")


if __name__ == "__main__":
    main()
