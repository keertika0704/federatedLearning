"""
Assessment log: every prediction made in the web app, kept so clinicians can look back at past assessments and so
this clinic can act as a live node for outbreak surveillance (objective B).

Privacy: the patient name never reaches this API (the form keeps it in the browser). Only age, gender, vitals,
symptom ids and the model outputs are stored. What goes to the outbreak store is counts only, exactly like
models/objectiveB/node.py produces: one row per (day, disease) with the number of cases and symptom counts, plus the
"ATYP" row for cases flagged atypical / novel.

Endpoints (all need the X-API-Key header or the UI session cookie):
    POST /api/assessments                    log a finished assessment; refreshes this clinic's summary for that day
    POST /api/assessments/{id}/severity      change the severity / top treatment when the clinician re-runs the plan
    GET  /api/assessments                    past assessments, newest first
    GET  /api/assessments/clinic-summary     this clinic's last 7 days, as the outbreak detector sees them

Settings (environment): ASSESSMENTS_DB (default data/assessments.db), CLINIC_HOSPITAL_ID (default LIVE),
CLINIC_REGION (default North).
"""

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "models" / "objectiveE").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.security import require_api_key  # noqa: E402

DB_PATH = Path(os.getenv("ASSESSMENTS_DB") or PROJECT_ROOT / "data" / "assessments.db")
HOSPITAL_ID = os.getenv("CLINIC_HOSPITAL_ID", "LIVE")
REGION = os.getenv("CLINIC_REGION", "North")
VITALS = ("temperature", "heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp")
WINDOW_DAYS = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, day TEXT NOT NULL,
    age INTEGER NOT NULL, gender TEXT NOT NULL,
    vitals TEXT NOT NULL, symptoms TEXT NOT NULL,
    disease_id TEXT NOT NULL, confidence REAL NOT NULL,
    atypical INTEGER NOT NULL, novelty INTEGER NOT NULL, reasons TEXT NOT NULL,
    severity_id TEXT, treatment TEXT
);
CREATE INDEX IF NOT EXISTS ix_assessments_day ON assessments(day);
"""

router = APIRouter()


class AssessmentIn(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^[FM]$")
    vitals: Dict[str, Optional[float]] = Field(default_factory=dict)
    symptoms: List[str] = Field(default_factory=list, max_length=28)
    disease_id: str = Field(..., pattern=r"^D\d{3}$")
    confidence: float = Field(..., ge=0, le=1)
    atypical_flag: bool = False
    novelty_flag: bool = False
    reasons: List[str] = Field(default_factory=list, max_length=20)
    severity_id: Optional[str] = Field(None, pattern="^SV00[123]$")
    treatment: Optional[str] = Field(None, max_length=120)


class SeverityIn(BaseModel):
    severity_id: str = Field(..., pattern="^SV00[123]$")
    treatment: Optional[str] = Field(None, max_length=120)


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        with c:
            yield c
    finally:
        c.close()


def _public(r):
    return {"id": r["id"], "created_at": r["created_at"], "day": r["day"], "age": r["age"], "gender": r["gender"],
            "vitals": json.loads(r["vitals"]), "symptoms": json.loads(r["symptoms"]), "disease_id": r["disease_id"],
            "confidence": r["confidence"], "atypical_flag": bool(r["atypical"]), "novelty_flag": bool(r["novelty"]),
            "reasons": json.loads(r["reasons"]), "severity_id": r["severity_id"], "treatment": r["treatment"]}


def _day_counts(c, day):
    """This clinic's node summary for one day: {disease_id | 'ATYP': (cases, {symptom id: cases with it})}."""
    out = {}
    for r in c.execute("SELECT disease_id, atypical, novelty, symptoms FROM assessments WHERE day=?", (day,)):
        symptoms = json.loads(r["symptoms"])
        keys = [r["disease_id"]] + (["ATYP"] if r["atypical"] or r["novelty"] else [])
        for key in keys:
            n, per = out.get(key, (0, {}))
            for s in symptoms:
                per[s] = per.get(s, 0) + 1
            out[key] = (n + 1, per)
    return out


def _push_summary(counts, day):
    """Writes the day's counts to the outbreak store as this clinic's node. Replaces that day's rows for the node."""
    import pandas as pd
    from models.objectiveB.store import SUMMARY_COLUMNS, SYMPTOMS, Store

    records = [{"hospital_id": HOSPITAL_ID, "region": REGION, "day": day, "disease_id": key, "n": n,
                **{s: per.get(s, 0) for s in SYMPTOMS}} for key, (n, per) in counts.items()]
    Store().put_summaries(pd.DataFrame(records)[SUMMARY_COLUMNS])


@router.post("/api/assessments", dependencies=[Depends(require_api_key)])
def log_assessment(body: AssessmentIn):
    from models.objectiveB.store import SYMPTOMS
    unknown = sorted(set(body.symptoms) - set(SYMPTOMS))
    if unknown:
        raise HTTPException(422, f"Unknown symptom id(s): {unknown}")
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    vitals = {k: body.vitals.get(k) for k in VITALS}
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO assessments (created_at, day, age, gender, vitals, symptoms, disease_id, confidence, atypical, novelty, "
            "reasons, severity_id, treatment) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), day, body.age, body.gender, json.dumps(vitals),
             json.dumps(sorted(set(body.symptoms))), body.disease_id, body.confidence, int(body.atypical_flag),
             int(body.novelty_flag), json.dumps([r[:200] for r in body.reasons]), body.severity_id, body.treatment))
        assessment_id, counts = cur.lastrowid, _day_counts(c, day)
    try:
        _push_summary(counts, day)
        shared = True
    except Exception:                       # the log is the record; a surveillance hiccup must not lose it
        shared = False
    return {"id": assessment_id, "day": day, "surveillance": {"hospital_id": HOSPITAL_ID, "shared": shared}}


@router.post("/api/assessments/{assessment_id}/severity", dependencies=[Depends(require_api_key)])
def set_severity(assessment_id: int, body: SeverityIn):
    with _conn() as c:
        c.execute("UPDATE assessments SET severity_id=?, treatment=? WHERE id=?",
                  (body.severity_id, body.treatment, assessment_id))
        r = c.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
    if r is None:
        raise HTTPException(404, f"No assessment with id {assessment_id}")
    return _public(r)


@router.get("/api/assessments", dependencies=[Depends(require_api_key)])
def list_assessments(limit: int = Query(200, ge=1, le=1000)):
    with _conn() as c:
        rows = c.execute("SELECT * FROM assessments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        total = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    return {"total": total, "assessments": [_public(r) for r in rows]}


@router.get("/api/assessments/clinic-summary", dependencies=[Depends(require_api_key)])
def clinic_summary():
    """The last WINDOW_DAYS days of this clinic's assessments, in the terms the outbreak detector uses."""
    today = datetime.now(timezone.utc).date()
    since = (today - timedelta(days=WINDOW_DAYS - 1)).isoformat()
    with _conn() as c:
        rows = c.execute("SELECT disease_id, atypical, novelty FROM assessments WHERE day>=?", (since,)).fetchall()
        first = c.execute("SELECT MIN(day) FROM assessments").fetchone()[0]
    by_disease: Dict[str, int] = {}
    for r in rows:
        by_disease[r["disease_id"]] = by_disease.get(r["disease_id"], 0) + 1
    return {"hospital_id": HOSPITAL_ID, "region": REGION, "since": since, "until": today.isoformat(),
            "cases": len(rows), "atypical": sum(1 for r in rows if r["atypical"] or r["novelty"]),
            "by_disease": [{"disease_id": d, "cases": n} for d, n in sorted(by_disease.items(), key=lambda x: -x[1])],
            "first_assessment": first}
