"""
Federated outbreak detector (Objective B).

B's detector is statistical and cross-site (it compares hospitals with each other), so there is no
per-hospital model to evaluate locally. What is federated is the pipeline that feeds its calibration:

  hospital (one Flower round)
      summaries   B's own NodeSummarizer, with the FEDERATED atypical detector (Objective A, this
                  package) as the local flagger: per day and disease, the case count and the 28 symptom
                  counts, plus the "ATYP" pseudo-disease. Counts only, as in the centralised design.
      episodes    the hospital's own labelled outbreaks (disease, first day, last day), the ground truth
                  B's calibration scores against. No patient rows or identifiers.
  server
      assembles B's Panel / Stats / Truth from those and runs B's unmodified calibrate() and evaluate()
      (models/objectiveB/train.py), time-split exactly as B does; it never sees an encounter row.

B's modules are imported unmodified. Run the `atypical` task first: its output is the local detector.
Nothing is written to B's SQLite store.
"""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from carepath_fl.common.partition import PROJECT_ROOT
from carepath_fl.tasks import atypical

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objectiveB import detect as b_detect  # noqa: E402
from models.objectiveB import node as b_node  # noqa: E402
from models.objectiveB import train as b_train  # noqa: E402
from models.objectiveB.store import ATYPICAL_ID, SUMMARY_COLUMNS, SYMPTOMS  # noqa: E402

COLUMNS = list(b_node.NEEDED) + atypical.VITALS + ["outbreak_active", "outbreak_id"]
STAGE_SPLIT = "all"                    # B's timeline is a hospital's whole record, not a train/test split
EVAL_SPLIT = "all"
OUT_DIR = PROJECT_ROOT / "models" / "objectiveC" / "model" / "outbreak"
ATYPICAL_MODEL = atypical.OUT_DIR / "atypical_model.json"
EPOCH = pd.Timestamp("2000-01-01")
TEST_START = "2025-10-01"              # B's default held-out start
MAX_FALSE = 6.0                        # false notifications per year allowed when calibrating (B's default)
STAGES = ("summaries",)


def _text(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8)


def _str(a: np.ndarray) -> str:
    return a.tobytes().decode()


def _days(dates) -> np.ndarray:
    return (pd.DatetimeIndex(pd.to_datetime(dates)).normalize() - EPOCH).days.to_numpy(dtype=np.int32)


# ------------------------------------------------------------------ client side
def client_stage(stage: str, state: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """One hospital's day x disease count summaries and its labelled outbreak episodes."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    a_params = state["a_params"]
    hospital, region = str(df["hospital_id"].iat[0]), str(df["region"].iat[0])
    node = b_node.NodeSummarizer(hospital, region, detector=atypical.a.AtypicalDetector(params=a_params))
    rows = node.summarize(df)

    names = [*a_params["diseases"], ATYPICAL_ID]
    index = {d: i for i, d in enumerate(names)}
    out = {"hospital": _text(hospital), "region": _text(region),
           "day": _days(rows["day"]), "disease": rows["disease_id"].map(index).to_numpy(dtype=np.int16),
           "n": rows["n"].to_numpy(dtype=np.int32), "sym": rows[SYMPTOMS].to_numpy(dtype=np.int32)}

    ep = (df[df["outbreak_active"] == 1].groupby("outbreak_id")
          .agg(d=("disease_id", "first"), s=("encounter_date", "min"), e=("encounter_date", "max")))
    out["ep_disease"] = ep["d"].map(index).to_numpy(dtype=np.int16)
    out["ep_start"], out["ep_end"] = _days(ep["s"]), _days(ep["e"])
    return out


# ------------------------------------------------------------------ server side
def assemble(parts: list[dict], diseases: list[str]):
    """Replies -> (summaries frame in B's store format, labelled episodes frame)."""
    names = [*diseases, ATYPICAL_ID]
    frames, episodes = [], []
    for p in parts:
        hospital, region = _str(p["hospital"]), _str(p["region"])
        f = pd.DataFrame(p["sym"], columns=SYMPTOMS)
        f.insert(0, "n", p["n"])
        f.insert(0, "disease_id", [names[i] for i in p["disease"]])
        f.insert(0, "day", EPOCH + pd.to_timedelta(p["day"], unit="D"))
        f.insert(0, "region", region)
        f.insert(0, "hospital_id", hospital)
        frames.append(f[SUMMARY_COLUMNS])
        episodes += [{"hospital_id": hospital, "disease_id": names[d], "start": EPOCH + pd.Timedelta(days=int(s)),
                      "end": EPOCH + pd.Timedelta(days=int(e)), "outbreak_id": f"{hospital}-{i}"}
                     for i, (d, s, e) in enumerate(zip(p["ep_disease"], p["ep_start"], p["ep_end"]))]
    return pd.concat(frames, ignore_index=True), pd.DataFrame(episodes)


def truth_from_episodes(panel, episodes: pd.DataFrame, workdir: Path):
    """B's own Truth, fed the hospitals' episode reports. Truth reads a labelled-encounters file, so write
    each episode's first and last day in that format (Truth only takes each outbreak's min / max date)."""
    rows = [{"hospital_id": r.hospital_id, "disease_id": r.disease_id, "encounter_date": day.strftime("%Y-%m-%d"),
             "outbreak_active": 1, "outbreak_id": r.outbreak_id}
            for r in episodes.itertuples() for day in (r.start, r.end)]
    path = Path(workdir) / "episodes.csv"
    pd.DataFrame(rows, columns=["hospital_id", "disease_id", "encounter_date", "outbreak_active", "outbreak_id"]
                 ).to_csv(path, index=False)
    return b_train.Truth(panel, path)


def load_atypical_params() -> dict:
    if not ATYPICAL_MODEL.is_file():
        raise FileNotFoundError(f"{ATYPICAL_MODEL} not found: run the 'atypical' task first, its model is the "
                                f"local detector each hospital uses")
    return json.loads(ATYPICAL_MODEL.read_text())


def run_protocol(call, test_start: str = TEST_START, max_false: float = MAX_FALSE, log=print):
    """Collect summaries + episodes from every hospital, then calibrate and evaluate with B's own code."""
    a_params = load_atypical_params()
    parts = call("summaries", {"a_params": a_params})
    summaries, episodes = assemble(parts, a_params["diseases"])
    log(f"[outbreak] {len(parts)} hospitals sent {len(summaries):,} summary rows and {len(episodes)} labelled outbreaks")

    panel = b_detect.Panel(summaries)
    stats = b_detect.Stats(panel, b_detect.DEFAULTS)
    first = b_train.first_eval_day(b_detect.DEFAULTS)
    last = len(panel.days) - 1
    test_t = int((pd.Timestamp(test_start) - panel.days[0]).days)
    if not first + 30 < test_t < last:
        raise ValueError(f"test start must fall between {panel.days[min(first + 31, last)].date()} "
                         f"and {panel.days[last].date()}")
    with tempfile.TemporaryDirectory() as tmp:
        truth = truth_from_episodes(panel, episodes, tmp)
        det = b_detect.OutbreakDetector(params={})
        params, n_grid, n_ok = b_train.calibrate(det, stats, truth, first, test_t - 1, max_false)
        metrics = {"calibration": b_train.evaluate(det, stats, truth, params, first, test_t - 1),
                   "test": b_train.evaluate(det, stats, truth, params, test_t, last),
                   "test_with_default_thresholds": b_train.evaluate(det, stats, truth, b_detect.DEFAULTS, test_t, last)}
    log(f"[outbreak] searched {n_grid} settings ({n_ok} within the false-alert budget); chosen "
        + ", ".join(f"{k}={params[k]}" for k in [*b_train.GRID, "atyp_p_thresh"]))
    info = {"calibration_days": [str(panel.days[first].date()), str(panel.days[test_t - 1].date())],
            "test_days": [str(panel.days[test_t].date()), str(panel.days[last].date())],
            "max_false_notifications_per_year": max_false, "metrics": metrics,
            "hospitals": len(panel.hospitals), "labelled_outbreaks": len(truth.episodes),
            "linked_outbreaks": len(truth.linked)}
    return {"params": params}, info


# ------------------------------------------------------------------ export + report
def write_model(params: dict, extra_meta: dict | None = None, reference=None, out_dir: Path = OUT_DIR) -> Path:
    """outbreak_model.json ({"params": thresholds}, what OutbreakDetector(path=...) loads) and meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outbreak_model.json").write_text(json.dumps(params, indent=2))
    meta = {"model": "federated: poisson-baseline + cross-site symptom-signature agreement; thresholds "
                     "calibrated on hospital-computed summaries and hospital-reported episodes",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra_meta or {}),
            "note": "the generator injects no novel pathogen, so the possible-novel path is only checked for false alarms"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def report(model_dir: Path | None = None) -> None:
    """Federated pipeline vs the centralised one (B's recorded results with the production atypical model)."""
    model_dir = Path(model_dir or OUT_DIR)
    fed = json.loads((model_dir / "meta.json").read_text())
    central_path = PROJECT_ROOT / "models" / "objectiveB" / "model" / "meta.json"
    central = json.loads(central_path.read_text()) if central_path.is_file() else None

    keys = ["episodes", "linked_episodes", "site_signal_recall", "alert_recall", "linked_alert_recall",
            "median_delay_days", "notifications", "notification_precision", "false_notifications_per_year",
            "novel_alert_days_per_year"]
    rows = {k: {} for k in keys}
    for name, meta in (("federated", fed), ("centralised", central)):
        if meta:
            for k in keys:
                rows[k][name] = meta["metrics"]["test"].get(k)
    table = pd.DataFrame(rows).T
    print(f"held-out test, {fed['test_days'][0]} .. {fed['test_days'][1]}")
    print(table.to_string())
    fp, cp = fed["metrics"]["test"], (central or {}).get("metrics", {}).get("test")
    print("\nchosen thresholds (federated):", json.loads((model_dir / "outbreak_model.json").read_text())["params"])
    if cp is None:
        print("(no centralised metrics found to compare with)")
    (model_dir / "report.json").write_text(json.dumps({"federated_test": fp, "centralised_test": cp}, indent=2))
    print(f"[report] written to {model_dir / 'report.json'}")
