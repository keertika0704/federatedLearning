"""
Objective B, central layer - detect an outbreak that is rising at several hospitals at once.

Works ONLY on the node summaries (node.py / store.py), never on patient rows.

How it works
    1. Per hospital and disease, compare the last 7 days of cases with that site's own recent normal: the median of
       eight earlier 7-day totals, skipping a one-week gap so a growing outbreak does not pollute its own baseline.
       The p-value is the Poisson upper tail P(X >= cases). A site "signals" when p is small, the count is at least
       `min_count` and at least `min_ratio` times the expectation.
    2. Cross-site confirmation. An alert needs `min_sites` hospitals signalling for the SAME disease, and their
       excess-symptom signatures must agree. The signature is (symptom counts this week) - (usual symptom counts per
       week): the symptoms the extra cases bring. Sites whose signature disagrees with the rest are dropped.
    3. Possible new disease. Independently of diagnoses, a site signals when the SHARE of its cases that objective A
       flags atypical / novel rises (binomial tail; a normal outbreak raises cases but not that share). Sites whose
       atypical-case signatures agree are grouped into a "possible novel outbreak" alert.
    Scope is regional (one region) or multi-region (pandemic risk). Severity is watch / warning / critical from the
    number of sites and regions. Thresholds come from model/outbreak_model.json (written by train.py).

Use from the terminal (from the project root, after `node.py --build`):
    python models/objectiveB/detect.py                       # scan as of the latest day in the store
    python models/objectiveB/detect.py --as-of 2025-11-10    # replay a past day (uses only data up to that day)
    python models/objectiveB/detect.py --as-of 2025-11-10 --save   # also store the alerts and notify
"""

import argparse
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / "models" / "objectiveE").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objectiveB.store import ATYPICAL_ID, SYMPTOMS, Store  # noqa: E402

MODEL_DIR = HERE / "model"
MODEL_FILE = MODEL_DIR / "outbreak_model.json"
META_FILE = MODEL_DIR / "meta.json"
REFERENCE = PROJECT_ROOT / "data" / "reference" / "master"
UNKNOWN_KEY = "UNKNOWN"

DEFAULTS = {
    # windows (days)
    "window": 7, "gap": 7, "n_blocks": 8, "min_blocks": 4,
    "season_shifts": [357, 364, 371],        # same 7-day window last year (+/- a week), for the seasonal guard
    # known-disease signal: Poisson upper tail vs the site's own baseline
    "min_base": 1.0, "p_thresh": 0.01, "min_count": 5, "min_ratio": 1.5,
    # cross-site confirmation
    "min_sites": 2, "min_sim": 0.5,
    # atypical-share signal (possible new disease): binomial upper tail
    "atyp_min_base_rate": 0.02, "atyp_p_thresh": 0.01, "atyp_min_count": 4, "atyp_min_ratio": 2.0,
    # lifecycle
    "resolve_after_days": 14,
}


@lru_cache(maxsize=1)
def reference_names():
    """(hospital names, disease names, symptom names) from the reference tables; ids are the fallback."""
    def load(file, key, name):
        try:
            df = pd.read_csv(REFERENCE / file, encoding="utf-8-sig")
            return dict(zip(df[key], df[name]))
        except (OSError, KeyError, ValueError):
            return {}
    return (load("hospital_master.csv", "hospital_id", "hospital_name"),
            load("disease_master.csv", "disease_id", "disease_name"),
            load("symptom_master.csv", "symptom_id", "symptom_name"))


# ------------------------------------------------------------------ array helpers
def _shift(a, s):
    """out[..., t] = a[..., t - s]; NaN where that day does not exist."""
    out = np.full(a.shape, np.nan)
    if s < a.shape[-1]:
        out[..., s:] = a[..., : a.shape[-1] - s]
    return out


def _roll(a, w):
    """Trailing w-day sum along the last axis; NaN until a full window exists."""
    c = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, axis=-1)], axis=-1)
    out = np.full(a.shape, np.nan)
    out[..., w - 1:] = c[..., w:] - c[..., :-w]
    return out


def _clean(x, fill=0.0):
    return np.nan_to_num(x, nan=fill)


# ------------------------------------------------------------------ summaries -> dense arrays
class Panel:
    """Node summaries as arrays: n[hospital, disease, day], sym[hospital, disease, day, symptom], and the same
    for the atypical pseudo-disease. A day a node never reported is marked in `reported` (and never read as 0)."""

    def __init__(self, df):
        if df.empty:
            raise ValueError("no node summaries to analyse")
        self.hospitals = sorted(df["hospital_id"].unique())
        self.region = df.drop_duplicates("hospital_id").set_index("hospital_id")["region"].to_dict()
        self.diseases = sorted(d for d in df["disease_id"].unique() if d != ATYPICAL_ID)
        self.days = pd.date_range(df["day"].min(), df["day"].max(), freq="D")
        H, K, T = len(self.hospitals), len(self.diseases), len(self.days)
        hmap = {h: i for i, h in enumerate(self.hospitals)}
        dmap = {d: i for i, d in enumerate(self.diseases)}
        h = df["hospital_id"].map(hmap).to_numpy()
        t = (df["day"] - self.days[0]).dt.days.to_numpy()
        n = df["n"].to_numpy(dtype=float)
        sym = df.reindex(columns=SYMPTOMS).to_numpy(dtype=float)
        k = df["disease_id"].map(dmap)
        known = k.notna().to_numpy()
        kk = k[known].astype(int).to_numpy()

        self.n = np.zeros((H, K, T))
        self.sym = np.zeros((H, K, T, len(SYMPTOMS)))
        self.n[h[known], kk, t[known]] = n[known]
        self.sym[h[known], kk, t[known]] = sym[known]
        self.n_atyp = np.zeros((H, T))
        self.sym_atyp = np.zeros((H, T, len(SYMPTOMS)))
        a = (df["disease_id"] == ATYPICAL_ID).to_numpy()
        self.n_atyp[h[a], t[a]] = n[a]
        self.sym_atyp[h[a], t[a]] = sym[a]
        self.reported = np.zeros((H, T), dtype=bool)
        self.reported[h, t] = True
        self.total = self.n.sum(axis=1)                                      # (H, T) all encounters


class Stats:
    """Every hospital x disease x day test statistic (independent of the alert thresholds)."""

    def __init__(self, panel, p):
        self.panel, self.p = panel, p
        W, G, NB, MB = p["window"], p["gap"], p["n_blocks"], p["min_blocks"]
        self.full = _roll(panel.reported.astype(float), W) == W              # (H, T) last W days all reported

        # known diseases: Poisson tail against the median of earlier 7-day totals
        r = np.where(self.full[:, None, :], _roll(panel.n, W), np.nan)       # (H, K, T)
        blocks = np.stack([_shift(r, W + G + W * j) for j in range(NB)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            base = np.nanmedian(blocks, axis=0)
        base = np.where((~np.isnan(blocks)).sum(0) >= MB, base, np.nan)
        # seasonal guard (once there is a year of history): never expect less than the same weeks last year, so a
        # predictable seasonal surge is not announced as news every year
        last_year = np.stack([_shift(r, s) for s in p["season_shifts"]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            season = np.nanmedian(last_year, axis=0)
        season = np.where((~np.isnan(last_year)).sum(0) >= 2, season, 0.0)
        self.season = season
        self.k = r
        self.lam = np.maximum(np.maximum(base, season), p["min_base"])       # NaN while there is too little history
        ok = ~np.isnan(self.k) & ~np.isnan(self.lam)
        self.pval = np.where(ok, poisson.sf(_clean(self.k) - 1, _clean(self.lam, 1.0)), 1.0)
        self.ratio = np.where(ok, self.k / np.where(ok, self.lam, 1.0), 0.0)

        # atypical share: binomial tail against the site's pooled earlier share
        a7 = np.where(self.full, _roll(panel.n_atyp, W), np.nan)             # (H, T)
        t7 = np.where(self.full, _roll(panel.total, W), np.nan)
        ab = np.stack([_shift(a7, W + G + W * j) for j in range(NB)])
        tb = np.stack([_shift(t7, W + G + W * j) for j in range(NB)])
        rate = np.nansum(ab, 0) / np.maximum(np.nansum(tb, 0), 1.0)
        rate = np.where((~np.isnan(tb)).sum(0) >= MB, rate, np.nan)
        self.a7, self.t7 = a7, t7
        self.p0 = np.clip(np.maximum(rate, p["atyp_min_base_rate"]), 1e-6, 1 - 1e-6)   # NaN while no history
        oka = ~np.isnan(a7) & ~np.isnan(self.p0) & ~np.isnan(t7)
        self.pval_a = np.where(oka, binom.sf(_clean(a7) - 1, np.rint(_clean(t7)).astype(int), _clean(self.p0, 0.5)), 1.0)
        self.ratio_a = np.where(oka, a7 / np.maximum(t7, 1.0) / np.where(oka, self.p0, 1.0), 0.0)

        # prefix sums over days, for symptom-signature windows
        self.csym = np.concatenate([np.zeros_like(panel.sym[:, :, :1]), np.cumsum(panel.sym, axis=2)], axis=2)
        self.csym_a = np.concatenate([np.zeros_like(panel.sym_atyp[:, :1]), np.cumsum(panel.sym_atyp, axis=1)], axis=1)

    def signals(self, p):
        """Boolean masks for these thresholds: known (H, K, T) and atypical (H, T)."""
        known = (self.pval <= p["p_thresh"]) & (self.k >= p["min_count"]) & (self.ratio >= p["min_ratio"])
        atyp = (self.pval_a <= p["atyp_p_thresh"]) & (self.a7 >= p["atyp_min_count"]) & (self.ratio_a >= p["atyp_min_ratio"])
        return known, atyp

    def signature(self, h, k, t):
        """Excess symptom counts at (hospital h, disease k) on day t: this week minus the usual week.
        k=None uses the atypical cases."""
        W, G, NB, MB = (self.p[x] for x in ("window", "gap", "n_blocks", "min_blocks"))
        C = self.csym_a[h] if k is None else self.csym[h, k]                # (T + 1, symptoms)

        def window(end):
            return C[end + 1] - C[end - W + 1]
        base = []
        for j in range(NB):
            end = t - W - G - W * j
            if end >= W - 1 and self.full[h, end]:
                base.append(window(end))
        usual = np.mean(base, axis=0) if len(base) >= MB else np.zeros(C.shape[-1])
        return window(t) - usual


# ------------------------------------------------------------------ grouping
def _consensus(sigs, min_sites, min_sim):
    """Largest set of sites (>= min_sites) whose excess-symptom signatures agree (mean pairwise cosine >= min_sim).
    Drops the least similar site until they do. Returns (site list, mean similarity)."""
    sites = list(sigs)
    while len(sites) >= min_sites:
        M = np.array([sigs[s] for s in sites], dtype=float)
        norm = np.linalg.norm(M, axis=1, keepdims=True)
        M = np.divide(M, norm, out=np.zeros_like(M), where=norm > 0)
        S = M @ M.T
        n = len(sites)
        mean_sim = float((S.sum() - np.trace(S)) / (n * (n - 1)))
        if mean_sim >= min_sim:
            return sites, mean_sim
        sites.pop(int(np.argmin((S.sum(1) - np.diag(S)) / (n - 1))))
    return [], 0.0


def severity_of(n_sites, n_regions):
    if n_regions >= 3 or n_sites >= 5:
        return "critical"
    if n_regions >= 2 or n_sites >= 3:
        return "warning"
    return "watch"


def describe(alert):
    """Regions, scope, severity, title and message derived from an alert's sites. Also used by the store to refresh an
    alert when it merges later days (more hospitals) into it."""
    rows = alert["sites"]
    regions = sorted({r["region"] for r in rows})
    where = f"{len(rows)} hospitals" + (f" across {len(regions)} regions" if len(regions) > 1 else f" in {regions[0]}")
    title = (f"Possible novel outbreak: unusual presentations rising at {where}" if alert["kind"] == "possible_novel"
             else f"{alert['disease_name']} rising at {where}")
    evidence = "; ".join(f"{r['hospital_name']} {r['recent_count']} vs ~{r['expected']:g} expected" for r in rows)
    symptoms = ", ".join(s["symptom"] for s in alert["signature"]) or "none distinct"
    return {"regions": regions, "scope": "multi-region" if len(regions) > 1 else "regional",
            "severity": severity_of(len(rows), len(regions)), "title": title,
            "message": f"{title}. {evidence}. Excess symptoms: {symptoms}."}


class OutbreakDetector:
    def __init__(self, path=MODEL_FILE, params=None):
        self.trained = False
        self.meta = {}
        if params is None:
            params = {}
            if Path(path).is_file():
                params = json.loads(Path(path).read_text())["params"]
                self.trained = True
            if META_FILE.is_file():
                self.meta = json.loads(META_FILE.read_text())
        self.p = {**DEFAULTS, **params}

    # ---------------------------------------------------------------- one day
    def alerts_at(self, stats, signals, t, p=None):
        """Alerts for day index t of the panel. `signals` = stats.signals(p); p defaults to this detector's
        thresholds (train.py passes candidate thresholds while calibrating)."""
        panel, p = stats.panel, p or self.p
        sig_known, sig_atyp = signals
        out = []
        for k, disease in enumerate(panel.diseases):
            hs = np.flatnonzero(sig_known[:, k, t])
            if len(hs) < p["min_sites"]:
                continue
            sigs = {int(h): stats.signature(h, k, t) for h in hs}
            sites, sim = _consensus(sigs, p["min_sites"], p["min_sim"])
            if sites:
                out.append(self._alert(stats, t, disease, sites, sim, sigs, kind="known_disease"))
        hs = np.flatnonzero(sig_atyp[:, t])
        if len(hs) >= p["min_sites"]:
            sigs = {int(h): stats.signature(h, None, t) for h in hs}
            sites, sim = _consensus(sigs, p["min_sites"], p["min_sim"])
            if sites:
                out.append(self._alert(stats, t, None, sites, sim, sigs, kind="possible_novel"))
        return out

    def _alert(self, stats, t, disease, sites, sim, sigs, kind):
        panel = stats.panel
        hosp_names, dis_names, sym_names = reference_names()
        k = panel.diseases.index(disease) if disease else None
        rows = []
        for h in sorted(sites):
            hid = panel.hospitals[h]
            if k is None:
                got, exp, ratio, pv = stats.a7[h, t], stats.p0[h, t] * stats.t7[h, t], stats.ratio_a[h, t], stats.pval_a[h, t]
            else:
                got, exp, ratio, pv = stats.k[h, k, t], stats.lam[h, k, t], stats.ratio[h, k, t], stats.pval[h, k, t]
            rows.append({"hospital_id": hid, "hospital_name": hosp_names.get(hid, hid), "region": panel.region[hid],
                         "recent_count": int(got), "expected": round(float(exp), 1), "ratio": round(float(ratio), 2),
                         "p_value": float(f"{pv:.3g}")})
        total = np.sum([sigs[h] for h in sites], axis=0)
        top = [i for i in np.argsort(total)[::-1][:5] if total[i] > 0]
        signature = [{"symptom_id": SYMPTOMS[i], "symptom": sym_names.get(SYMPTOMS[i], SYMPTOMS[i]),
                      "excess": round(float(total[i]), 1)} for i in top]
        alert = {"disease_key": disease or UNKNOWN_KEY, "disease_id": disease, "kind": kind,
                 "disease_name": "Atypical presentations" if disease is None else dis_names.get(disease, disease),
                 "day": panel.days[t].strftime("%Y-%m-%d"), "sites": rows, "similarity": round(sim, 3),
                 "signature": signature}
        alert.update(describe(alert))
        return alert

    # ---------------------------------------------------------------- scan / trends
    def _locate(self, panel, as_of):
        day = pd.Timestamp(as_of).normalize()
        if day < panel.days[0] or day > panel.days[-1]:
            raise ValueError(f"as_of {day.date()} is outside the summaries ({panel.days[0].date()} to {panel.days[-1].date()})")
        return int((day - panel.days[0]).days)

    def scan_panel(self, panel, as_of=None):
        """Alerts for one day; `panel` should hold only data up to that day. Default day: the latest."""
        stats = Stats(panel, self.p)
        t = len(panel.days) - 1 if as_of is None else self._locate(panel, as_of)
        return self.alerts_at(stats, stats.signals(self.p), t)

    def trends(self, panel, as_of=None):
        """Hospital x disease grid of this week's count vs expectation, for the dashboard."""
        stats = Stats(panel, self.p)
        t = len(panel.days) - 1 if as_of is None else self._locate(panel, as_of)
        sig_known, sig_atyp = stats.signals(self.p)
        hosp_names, dis_names, _ = reference_names()
        num = lambda x, nd=2: None if x is None or np.isnan(x) else round(float(x), nd)    # noqa: E731
        rows = []
        for h, hid in enumerate(panel.hospitals):
            cells = {}
            for k, d in enumerate(panel.diseases):
                cells[d] = {"count": num(stats.k[h, k, t], 0), "expected": num(stats.lam[h, k, t], 1),
                            "ratio": num(stats.ratio[h, k, t]), "p_value": num(stats.pval[h, k, t], 4),
                            "signal": bool(sig_known[h, k, t])}
            rows.append({"hospital_id": hid, "hospital_name": hosp_names.get(hid, hid), "region": panel.region[hid],
                         "reported": bool(stats.full[h, t]), "cells": cells,
                         "atypical": {"count": num(stats.a7[h, t], 0), "of": num(stats.t7[h, t], 0),
                                      "expected_share": num(stats.p0[h, t], 3),
                                      "ratio": num(stats.ratio_a[h, t]), "p_value": num(stats.pval_a[h, t], 4),
                                      "signal": bool(sig_atyp[h, t])}})
        return {"day": panel.days[t].strftime("%Y-%m-%d"),
                "diseases": [{"disease_id": d, "disease_name": dis_names.get(d, d)} for d in panel.diseases],
                "hospitals": rows}


def run_scan(store, detector, as_of=None, notify=True):
    """Detect as of `as_of` (default: latest day), store the alerts, notify on new / escalated ones, and close
    stale alerts. Uses only summaries up to as_of, so replaying a past day never sees the future."""
    df = store.load_summaries(as_of)
    panel = Panel(df)
    day = panel.days[-1] if as_of is None else pd.Timestamp(as_of).normalize()
    alerts = detector.scan_panel(panel, day)
    saved = []
    for a in alerts:
        rec, change = store.upsert_alert(a, rebuild=describe)
        if notify and change in ("new", "escalated"):
            from models.objectiveB.notify import notify as send
            send(rec, change)
        saved.append({**rec, "change": change})
    resolved = store.resolve_stale(day, detector.p["resolve_after_days"])
    return {"as_of": day.strftime("%Y-%m-%d"), "history_days": len(panel.days), "alerts": saved,
            "new": sum(a["change"] == "new" for a in saved), "resolved": resolved}


def main():
    ap = argparse.ArgumentParser(description="Detect outbreaks across hospitals")
    ap.add_argument("--as-of", help="day to analyse, YYYY-MM-DD (default: latest day in the store)")
    ap.add_argument("--db", help="SQLite file (default: data/outbreak.db or $OUTBREAK_DB)")
    ap.add_argument("--save", action="store_true", help="store the alerts and send notifications")
    args = ap.parse_args()

    store = Store(args.db)
    det = OutbreakDetector()
    if not det.trained:
        print("Note: no trained model/outbreak_model.json - using default thresholds. Run train.py to calibrate.")
    if args.save:
        res = run_scan(store, det, args.as_of)
        alerts = res["alerts"]
        print(f"As of {res['as_of']}: {len(alerts)} alert(s), {res['new']} new, {res['resolved']} resolved")
    else:
        alerts = det.scan_panel(Panel(store.load_summaries(args.as_of)), args.as_of)
        print(f"{len(alerts)} alert(s)")
    for a in alerts:
        print(f"\n[{a['severity'].upper()}] {a['title']}  (scope {a['scope']}, similarity {a['similarity']})")
        for s in a["sites"]:
            print(f"    {s['hospital_id']} {s['hospital_name']:32s} {s['region']:8s} {s['recent_count']:3d} vs ~{s['expected']:g} (p={s['p_value']:g})")
        print("    excess symptoms:", ", ".join(f"{s['symptom']} (+{s['excess']:g})" for s in a["signature"]))


if __name__ == "__main__":
    main()
