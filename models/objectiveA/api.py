"""
HTTP API for objective A: flag atypical presentations (possible new disease / variant).

Exposes a FastAPI `router`; it is not mounted anywhere yet. To serve it, add two lines to backend/main.py:
    from models.objectiveA.api import router as atypical_router
    app.include_router(atypical_router)
Or run it on its own (from the project root, OBJECTIVE_E_API_KEY set in .env):
    uvicorn models.objectiveA.api:app --reload --port 8003

Endpoints:
    GET  /api/atypical/health   -> model loaded? known diseases, thresholds, test metrics
    POST /api/atypical/assess   -> one patient
    POST /api/atypical/batch    -> up to 500 patients

Example request body (same patient fields the disease / treatment APIs take):
    {
      "disease_id": "D004",
      "patient": {"age": 34, "gender": "F", "temperature": 38.4, "heart_rate": 88, "respiratory_rate": 18,
                  "spo2": 98, "systolic_bp": 115, "diastolic_bp": 75,
                  "S001": 1, "S002": 1, "S003": 1, "S006": 1, "S012": 1, "S013": 1}
    }
disease_id is the diagnosis (e.g. the disease model's top prediction). Leave it out and the detector
uses its own best-fitting disease. Symptoms not listed are treated as absent; vitals not listed as unknown.

Response fields
    atypical_flag / atypical_score   does this patient not fit the diagnosed disease? (flag when score >= 1)
    novelty_flag  / novelty_score    does NO known disease fit? (the possible-new-disease signal)
    reasons                          unexpected / missing symptoms and unusual vitals, in plain words
    *_ratio                          the four sub-scores behind atypical_score
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "models" / "objectiveE").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.security import require_api_key  # noqa: E402
from models.objectiveA.predict import SYMPTOMS, VITALS, AtypicalDetector  # noqa: E402

PATIENT_FIELDS = set(SYMPTOMS) | set(VITALS) | {"age", "gender"}   # age/gender accepted but not used by this model
MAX_BATCH = 500

try:
    # loads the federated model trained by Objective C once at startup
    detector = AtypicalDetector(path=PROJECT_ROOT / "models" / "objectiveC" / "model" / "atypical" / "atypical_model.json")
except SystemExit as e:                    # predict.py exits with a message when the model file is missing
    raise RuntimeError(str(e)) from None

router = APIRouter()


class AssessRequest(BaseModel):
    patient: Dict[str, Any] = Field(..., description="symptom flags (S001..S028) and vitals")
    disease_id: Optional[str] = Field(None, examples=["D004"], description="diagnosed disease; default: best fit")


class BatchRequest(BaseModel):
    patients: List[AssessRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


def _check_disease(disease_id):
    if disease_id is not None and disease_id not in detector.idx:
        raise HTTPException(404, f"Unknown disease_id {disease_id}. Known: {detector.diseases}")


def _clean(patient):
    """Keep known fields only; symptoms must be 0/1, vitals numeric. Returns (clean dict, ignored field names)."""
    ignored = sorted(k for k in patient if k not in PATIENT_FIELDS)
    clean = {}
    for k, v in patient.items():
        if k in PATIENT_FIELDS and k != "gender":
            if v is None:
                continue
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                raise HTTPException(422, f"Field {k} must be numeric, got {v!r}")
    for s in SYMPTOMS:
        if s in clean and clean[s] not in (0.0, 1.0):
            raise HTTPException(422, f"Symptom {s} must be 0 or 1, got {clean[s]:g}")
    return clean, ignored


def _assess(item: AssessRequest):
    _check_disease(item.disease_id)
    patient, ignored = _clean(item.patient)
    out = detector.assess(patient, item.disease_id)
    out["diagnosed_disease_name"] = detector.p["disease_names"].get(out["diagnosed_disease"])
    out["disease_was_given"] = item.disease_id is not None
    out["missing_vitals"] = [v for v in VITALS if v not in patient]
    out["ignored_fields"] = ignored
    return out


@router.get("/api/atypical/health")
def health():
    m = detector.p["meta"]
    return {"status": "ok", "model": m["model"], "trained_at": m["trained_at"],
            "diseases": detector.diseases, "thresholds": detector.p["threshold"],
            "false_alarm_target": m["fpr_target"], "metrics": m.get("metrics")}


@router.post("/api/atypical/assess", dependencies=[Depends(require_api_key)])
def assess(req: AssessRequest):
    return _assess(req)


@router.post("/api/atypical/batch", dependencies=[Depends(require_api_key)])
def batch(req: BatchRequest):
    results = [_assess(item) for item in req.patients]
    return {"count": len(results), "atypical": sum(r["atypical_flag"] for r in results),
            "novel": sum(r["novelty_flag"] for r in results), "results": results}


# standalone app for `uvicorn models.objectiveA.api:app`
app = FastAPI(title="Atypical presentation API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])
app.include_router(router)
