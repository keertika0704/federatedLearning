"""
HTTP API for objective B: detect and announce outbreaks that are rising at several hospitals at once.

Exposes a FastAPI `router`, mounted in backend/main.py. Or run it on its own (from the project root, with
OBJECTIVE_E_API_KEY set in .env and node summaries built by `python models/objectiveB/node.py --build`):
    uvicorn models.objectiveB.api:app --reload --port 8004

Endpoints (all but /health need the X-API-Key header or the UI session cookie):
    GET  /api/outbreak/health              thresholds, test metrics, the day range held in the store
    POST /api/outbreak/scan                {"as_of": "2025-11-10"} -> detect as of that day (default: latest), store the
                                           alerts, notify on new / escalated ones. Replaying a past day sees no future data.
    GET  /api/outbreak/alerts?status=open  the alert feed (status: open | acknowledged | resolved; omit for all)
    POST /api/outbreak/alerts/{id}/ack     acknowledge an alert
    GET  /api/outbreak/trends?as_of=...    hospital x disease grid: this week's cases vs what is normal
    POST /api/outbreak/summaries           a hospital node pushes its daily summaries (counts only, no patient rows)

Example POST /api/outbreak/summaries body (one row per disease with cases; diseases left out count as zero on a day you
report; "ATYP" = the cases objective A flagged atypical / novel):
    {"hospital_id": "H001", "region": "North",
     "rows": [{"day": "2025-11-10", "disease_id": "D004", "n": 7, "symptoms": {"S001": 6, "S006": 5}},
              {"day": "2025-11-10", "disease_id": "ATYP", "n": 1, "symptoms": {"S012": 1}}]}

Alert fields: disease_name, kind (known_disease | possible_novel), scope (regional | multi-region), severity
(watch | warning | critical), sites (cases this week vs expected, per hospital), similarity (how closely the sites'
excess-symptom signatures agree), signature (the symptoms the extra cases bring), status, first_seen, last_seen.
"""

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "models" / "objectiveE").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.security import require_api_key  # noqa: E402
from models.objectiveB.detect import OutbreakDetector, Panel, run_scan  # noqa: E402
from models.objectiveB.store import SUMMARY_COLUMNS, SYMPTOMS, Store  # noqa: E402

MAX_SUMMARY_ROWS = 2000
NO_DATA = "No node summaries yet. Build them first:  python models/objectiveB/node.py --build"

store = Store()
detector = OutbreakDetector()              # loads model/outbreak_model.json; falls back to defaults when untrained
router = APIRouter()


class ScanRequest(BaseModel):
    as_of: Optional[date] = Field(None, description="day to analyse; default: the latest day in the store")
    notify: bool = Field(True, description="send notifications for new / escalated alerts")


class SummaryRow(BaseModel):
    day: date
    disease_id: str = Field(..., pattern=r"^(D\d{3}|ATYP)$", examples=["D004"])
    n: int = Field(..., ge=0, description="cases this day (for ATYP: cases flagged atypical / novel)")
    symptoms: Dict[str, int] = Field(default_factory=dict, description="symptom id (S001..S028) -> how many of those cases had it")


class SummaryBatch(BaseModel):
    hospital_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,16}$", examples=["H001"])
    region: str = Field(..., min_length=1, max_length=40, examples=["North"])
    rows: List[SummaryRow] = Field(..., min_length=1, max_length=MAX_SUMMARY_ROWS)


def _panel(as_of=None):
    df = store.load_summaries(as_of)
    if df.empty:
        raise HTTPException(409, NO_DATA)
    return Panel(df)


def _bad_date(e):
    return HTTPException(422, str(e))


@router.get("/api/outbreak/health")
def health():
    lo, hi, hospitals = store.day_range()
    m = detector.meta
    return {"status": "ok", "trained": detector.trained, "model": m.get("model"), "trained_at": m.get("trained_at"),
            "thresholds": detector.p, "metrics": m.get("metrics"), "note": m.get("note"),
            "summaries": {"from": lo, "to": hi, "hospitals": hospitals},
            "open_alerts": len(store.list_alerts("open"))}


@router.post("/api/outbreak/scan", dependencies=[Depends(require_api_key)])
def scan(req: ScanRequest):
    if store.day_range()[0] is None:
        raise HTTPException(409, NO_DATA)
    try:
        return run_scan(store, detector, req.as_of.isoformat() if req.as_of else None, req.notify)
    except ValueError as e:
        raise _bad_date(e)


@router.get("/api/outbreak/alerts", dependencies=[Depends(require_api_key)])
def alerts(status: Optional[str] = Query(None, pattern="^(open|acknowledged|resolved)$"), limit: int = Query(100, ge=1, le=500)):
    found = store.list_alerts(status, limit)
    return {"count": len(found), "alerts": found}


@router.post("/api/outbreak/alerts/{alert_id}/ack", dependencies=[Depends(require_api_key)])
def acknowledge(alert_id: int):
    alert = store.acknowledge(alert_id)
    if alert is None:
        raise HTTPException(404, f"No alert with id {alert_id}")
    return alert


@router.get("/api/outbreak/trends", dependencies=[Depends(require_api_key)])
def trends(as_of: Optional[date] = None):
    panel = _panel(as_of.isoformat() if as_of else None)
    try:
        return detector.trends(panel, as_of.isoformat() if as_of else None)
    except ValueError as e:
        raise _bad_date(e)


@router.post("/api/outbreak/summaries", dependencies=[Depends(require_api_key)])
def ingest(batch: SummaryBatch):
    """A node pushes its daily counts. Replaces any rows already stored for the same hospital, day and disease."""
    import pandas as pd
    records = []
    for r in batch.rows:
        unknown = sorted(set(r.symptoms) - set(SYMPTOMS))
        if unknown:
            raise HTTPException(422, f"Unknown symptom id(s): {unknown}")
        if any(v < 0 or v > r.n for v in r.symptoms.values()):
            raise HTTPException(422, f"{r.day} {r.disease_id}: a symptom count must be between 0 and n ({r.n})")
        records.append({"hospital_id": batch.hospital_id, "region": batch.region, "day": r.day.isoformat(),
                        "disease_id": r.disease_id, "n": r.n, **{s: r.symptoms.get(s, 0) for s in SYMPTOMS}})
    stored = store.put_summaries(pd.DataFrame(records)[SUMMARY_COLUMNS])
    lo, hi, hospitals = store.day_range()
    return {"stored": stored, "summaries": {"from": lo, "to": hi, "hospitals": hospitals}}


# standalone app for `uvicorn models.objectiveB.api:app`
app = FastAPI(title="Outbreak detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])
app.include_router(router)
