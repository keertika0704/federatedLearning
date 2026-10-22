"""
Objective C: read-only API behind the web app's "Federated learning" tab.

It reads what the federated training already wrote (model/<task>/report.json and meta.json) and adds the
plain-language description of what each hospital shares. It never imports Flower and never trains or
exposes any model, patient data, node id or key material.

    GET /api/federated/overview

Exposes a FastAPI `router`, mounted in backend/main.py.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "models" / "objectiveE").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.security import require_api_key  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
MASTER = PROJECT_ROOT / "data" / "reference" / "master" / "hospital_master.csv"
CONFIG = PROJECT_ROOT / "data" / "config" / "hospital_generation_config.csv"

router = APIRouter()

# metric id -> (label, higher is better, display format). "pct" 0..1 shown as %, "num3" three decimals, "days" one decimal.
TASKS = [
    {
        "id": "disease", "title": "Disease classifier", "question": "Which disease does this patient most likely have?",
        "model": "XGBoost, 8 diseases (Objective E1)", "kind": "turns",
        "method": "Hospitals take turns. Each receives the shared model, adds trees learned from its own patients, "
                  "and passes it on.",
        "travels": ["The shared model after the hospital's turn (the trees it just added)",
                    "Accuracy figures on the hospital's own held-out patients"],
        "metrics": [("accuracy", "Accuracy", True, "pct"),
                    ("D003_recall", "Chikungunya recall (the rarest disease)", True, "pct")],
        "replay": ["accuracy"],
        "notes": ["Turns and iterations were chosen after looking at test-split results, so the accuracy is "
                  "slightly optimistic.",
                  "The federated model has 300 boosting iterations; the comparison models use 100.",
                  "Results move a little between runs because the hospital order is random.",
                  "Per-hospital rare-disease figures rest on 30 to 60 test cases."],
    },
    {
        "id": "treatment", "title": "Treatment recommender",
        "question": "Which treatment is most likely to work, and how long will recovery take?",
        "model": "XGBoost success and recovery-time models (Objective E2)", "kind": "turns",
        "method": "Same turn-taking, for two models at once. One extra round first collects which "
                  "disease / severity / treatment combinations exist.",
        "travels": ["The two shared models after the hospital's turn",
                    "Which disease, severity and treatment combinations the hospital has seen "
                    "(no patient rows, no counts)",
                    "Validation figures on the hospital's own held-out patients"],
        "metrics": [("auc", "Success prediction AUC", True, "num3"),
                    ("recovery_mae", "Recovery-time error (days)", False, "days")],
        "replay": ["auc", "recovery_mae"],
        "notes": ["The success-prediction gain is small and only some hospitals improve: the signal in this data "
                  "is weak (AUC around 0.61).",
                  "Settings were chosen on each hospital's own validation slice, never the test split."],
    },
    {
        "id": "atypical", "title": "Atypical-case detector",
        "question": "Does this patient look unlike any known disease?",
        "model": "Statistical detector (Objective A)", "kind": "stages",
        "method": "Nothing to average: every part of the detector is a statistic. Hospitals send counts, sums "
                  "and histograms and the server assembles the same model a pooled fit would give.",
        "travels": ["Per-disease patient counts and symptom counts",
                    "Sums used to fit each disease's vitals model (about 45 shared rounds)",
                    "Histograms of one-number scores, used only to set the alert thresholds"],
        "metrics": [("auc", "Detection AUC", True, "num3"),
                    ("recall", "Share of atypical cases caught", True, "pct"),
                    ("false_alarm_rate", "False alarms on ordinary patients (a calibration target)", None, "pct")],
        "replay": [],
        "stages": ["Counts and sums", "Vitals model (distributed EM, ~45 rounds)", "Score distributions",
                   "Alert thresholds"],
        "notes": ["Counts and sums match a pooled fit exactly; threshold values agree to about 1%."],
    },
    {
        "id": "outbreak", "title": "Outbreak detector",
        "question": "Is a disease rising at several hospitals at the same time?",
        "model": "Cross-site statistical detector (Objective B)", "kind": "stages",
        "method": "Detection compares hospitals with each other, so hospitals send daily counts and their own "
                  "labelled outbreaks, and the server tunes the alert thresholds on those alone.",
        "travels": ["Daily case counts per disease, with symptom counts",
                    "The disease and first and last day of each outbreak the hospital labelled"],
        "metrics": [], "replay": [],
        "stages": ["Daily summaries from each hospital", "Threshold search on the summaries", "Held-out check"],
        "notes": ["Matches the centralised pipeline exactly. Only 21 labelled outbreaks fall in the held-out "
                  "period, so there are few events to compare."],
    },
]

PROTECTIONS = [
    {"label": "Raw patient rows stay in the hospital",
     "detail": "Clients read only whitelisted columns; tests check that no identifier reaches a shared model."},
    {"label": "Encrypted connections (TLS)", "detail": "Hospitals verify the server's certificate before connecting."},
    {"label": "Each hospital has its own key",
     "detail": "The server refuses a key nobody registered, and a node that skips TLS."},
    {"label": "No impersonation",
     "detail": "A reply that names a different hospital than the node it came from is rejected."},
]
LIMITS = [
    {"label": "No differential privacy", "detail": "Nothing adds noise to what a hospital shares."},
    {"label": "No secure aggregation",
     "detail": "The server sees each hospital's contribution on its own, not only the combined result."},
    {"label": "Hospitals see the shared model",
     "detail": "In the turn-taking models, it contains splits learned from earlier hospitals' data."},
    {"label": "Small counts can still say something",
     "detail": "73% of daily hospital-by-disease counts are below 3; hiding them would remove the outbreak signal."},
]
SCOPE_NOTE = ("These results come from a simulation on synthetic data. The real-deployment demo runs every "
              "hospital on one machine. The web app's own models are not yet replaced by the federated ones.")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _updated(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


def _csv_rows(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def hospitals() -> list[dict]:
    region = {r["hospital_id"]: r.get("region", "") for r in _csv_rows(CONFIG)}
    return [{"id": r["hospital_id"], "name": r["hospital_name"], "city": r.get("city", ""),
             "region": region.get(r["hospital_id"], "")} for r in _csv_rows(MASTER)]


def _metric(report: dict, spec: tuple) -> dict:
    mid, label, higher, fmt = spec
    wm, wins = report.get("weighted_mean", {}), report.get("hospital_wins_vs_local", {})
    per = [{"hospital": r["hospital"], "rows": r.get("test_rows"),
            "alone": r.get(f"local_only.{mid}"), "federated": r.get(f"federated.{mid}"),
            "centralized": r.get(f"centralized_sim.{mid}")} for r in report.get("per_hospital", [])]
    return {"id": mid, "label": label, "higher_is_better": higher, "format": fmt,
            "alone": wm.get(f"local_only.{mid}"), "federated": wm.get(f"federated.{mid}"),
            "centralized": wm.get(f"centralized_sim.{mid}"),
            "hospitals_improved": wins.get(mid), "hospitals": len(per), "per_hospital": per}


OUTBREAK_ROWS = [("episodes", "Held-out outbreaks", "int"), ("linked_episodes", "…that spread between hospitals", "int"),
                 ("linked_alert_recall", "Spreading outbreaks alerted", "pct"),
                 ("alert_recall", "All outbreaks alerted", "pct"), ("median_delay_days", "Median days to alert", "days"),
                 ("notifications", "Notifications sent", "int"),
                 ("notification_precision", "Notifications that were correct", "pct"),
                 ("false_notifications_per_year", "False notifications per year", "num1")]


def _task(spec: dict) -> dict:
    d = MODEL_DIR / spec["id"]
    out = {k: spec[k] for k in ("id", "title", "question", "model", "kind", "method", "travels", "notes")}
    out["stages"] = spec.get("stages", [])
    report = _read_json(d / "report.json")
    meta = _read_json(d / "meta.json")
    if meta is None and spec["id"] == "atypical":      # the atypical model file carries its own metadata
        meta = (_read_json(d / "atypical_model.json") or {}).get("meta")
    out["available"] = bool(report and meta)
    if not out["available"]:
        out["hint"] = f'Not trained yet. Run: flwr run models/objectiveC --run-config \'task="{spec["id"]}"\''
        out.update(metrics=[], comparison=None, history=None, settings={}, updated=None)
        return out
    out["updated"] = _updated(d / "report.json")
    fed = meta.get("federated", meta)

    if spec["id"] == "outbreak":
        fed_t, cen_t = report.get("federated_test") or {}, report.get("centralised_test") or {}
        out["metrics"] = []
        out["comparison"] = {"title": "Held-out period", "days": meta.get("test_days"),
                             "rows": [{"label": lab, "format": fmt, "federated": fed_t.get(k), "centralized": cen_t.get(k)}
                                      for k, lab, fmt in OUTBREAK_ROWS]}
        out["settings"] = {"hospitals": meta.get("hospitals"), "labelled_outbreaks": meta.get("labelled_outbreaks"),
                           "linked_outbreaks": meta.get("linked_outbreaks"),
                           "test_days": meta.get("test_days")}
        out["history"] = None
        return out

    out["comparison"] = None
    out["metrics"] = [_metric(report, m) for m in spec["metrics"]]
    out["replay_metrics"] = spec["replay"]
    history = fed.get("history")
    out["history"] = history if history and any(h.get("hospital") for h in history) else None
    if spec["id"] == "atypical":
        rows = meta.get("rows", {})
        out["settings"] = {"fit_rows": rows.get("fit"), "calibration_rows": rows.get("calibration"),
                           "em_rounds": meta.get("em_rounds"), "false_alarm_target": meta.get("fpr_target")}
    else:
        out["settings"] = {"turns": fed.get("turns"), "local_iters": fed.get("local_iters"),
                           "iterations": fed.get("iterations")}
    return out


@router.get("/api/federated/overview", dependencies=[Depends(require_api_key)])
def overview():
    return {"hospitals": hospitals(), "tasks": [_task(t) for t in TASKS], "protections": PROTECTIONS,
            "limits": LIMITS, "scope_note": SCOPE_NOTE}
