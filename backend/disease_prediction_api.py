"""
Disease prediction API.

Run the combined backend from the project root (put OBJECTIVE_E_API_KEY in .env first):
    uvicorn backend.main:app --reload --port 8000

Docs + a built-in test form: http://localhost:8000/docs

Env vars:
    OBJECTIVE_E_API_KEY REQUIRED (min 16 chars). Clients send it in the X-API-Key header.
                       The server refuses to start without it.
    ALLOWED_ORIGINS  comma-separated web origins allowed to call the API
                     (default: http://localhost:3000,http://localhost:5173)

Model: always models/objectiveE/disease_prediction/model. Not configurable on purpose.
"""
import json
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, create_model
from xgboost import XGBClassifier

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "models" / "objectiveE").is_dir()
)
DISEASE_SOURCE = PROJECT_ROOT / "models" / "objectiveE" / "disease_prediction"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objectiveE.disease_prediction.train import build_features  # noqa: E402
from backend.treatment_prediction_api import recommend_for_patient  # noqa: E402

log = logging.getLogger("uvicorn.error")

# ---------- Step 0: load .env ----------
# Looks next to this file first, then walks up (so a .env in the project root works too).
# Real environment variables always win over .env values (override=False).
from dotenv import load_dotenv  # noqa: E402

for _d in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (_d / ".env").is_file():
        load_dotenv(_d / ".env", override=False)
        log.info(f"Loaded environment from {_d / '.env'}")
        break

# ---------- Step 1: load model once at startup ----------
# Pinned to the model next to this file (models/objectiveE/model). Deliberately NOT
# configurable via env var, so another objective's model can never be loaded by accident.
MODEL_DIR = (DISEASE_SOURCE / "model").resolve()
if not MODEL_DIR.is_relative_to(DISEASE_SOURCE.resolve()):
    raise RuntimeError(f"Model dir {MODEL_DIR} is outside {DISEASE_SOURCE}. Refusing to load.")
for _f in ("model.json", "meta.json"):
    if not (MODEL_DIR / _f).is_file():
        raise RuntimeError(
            f"Missing {MODEL_DIR / _f}. Train first:\n"
            f"  python {DISEASE_SOURCE / 'train.py'} --data <train.csv> --test <test.csv>"
        )
META = json.loads((MODEL_DIR / "meta.json").read_text())
MODEL = XGBClassifier()
MODEL.load_model(MODEL_DIR / "model.json")
CLASSES = META["classes"]
SYMPTOMS = [f"S{i:03d}" for i in range(1, 29)]
log.info(f"Loaded model from {MODEL_DIR} ({len(CLASSES)} classes)")

# ---------- Step 2: API key auth ----------
# No key = no server. Fail fast at startup instead of silently running without auth.
API_KEY = (os.getenv("OBJECTIVE_E_API_KEY") or "").strip()
MIN_KEY_LEN = 16
if len(API_KEY) < MIN_KEY_LEN:
    raise RuntimeError(
        f"API_KEY is missing or shorter than {MIN_KEY_LEN} characters. Refusing to start.\n"
        "  Generate one:  python -c \"import secrets; print(secrets.token_urlsafe(32))\"\n"
        "  Put it in .env as a single line:  API_KEY=<key>   (and remove any empty API_KEY= line)\n"
        "  If you exported an empty API_KEY in this shell, run:  unset API_KEY"
    )
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)):
    if not key or not secrets.compare_digest(key, API_KEY):  # constant-time compare
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------- Step 3: request / response schemas ----------
# Symptoms are 0/1 flags, default 0, so the frontend only needs to send the ones present.
SymptomFields = {s: (Literal[0, 1], 0) for s in SYMPTOMS}

BasePatient = create_model(
    "BasePatient",
    age=(int, Field(..., ge=0, le=120)),
    gender=(Literal["M", "F"], ...),
    temperature=(float | None, Field(None, ge=30, le=45, description="°C")),
    heart_rate=(int | None, Field(None, ge=20, le=250)),
    respiratory_rate=(int | None, Field(None, ge=4, le=80)),
    spo2=(int | None, Field(None, ge=50, le=100)),
    systolic_bp=(int | None, Field(None, ge=50, le=260)),
    diastolic_bp=(int | None, Field(None, ge=20, le=180)),
    hospital_id=(int | None, None),
    **SymptomFields,
)


class PatientIn(BasePatient):
    severity_id: Literal["SV001", "SV002", "SV003"] = Field(
        ..., description="Clinically assessed severity; severity is not predicted by this model."
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52, "gender": "F", "temperature": 38.4, "heart_rate": 96,
                "respiratory_rate": 22, "spo2": 94, "systolic_bp": 128, "diastolic_bp": 82,
                "S001": 1, "S006": 1, "S017": 1, "severity_id": "SV002",
            }
        }
    }


class Prediction(BaseModel):
    disease_id: int | str
    probability: float


class PredictOut(BaseModel):
    predictions: list[Prediction]
    predicted_disease: Prediction
    treatment_recommendation: dict[str, Any]
    model_classes: int


router = APIRouter()


@router.get("/health")
def health():
    """No auth, so load balancers and uptime checks can call it."""
    return {"status": "ok", "classes": len(CLASSES), "features": len(META["features"])}


@router.post("/predict", response_model=PredictOut, dependencies=[Depends(require_api_key)])
def predict(patient: PatientIn, top: int = 3):
    top = max(1, min(top, len(CLASSES)))
    row = patient.model_dump(exclude={"severity_id"})
    if META.get("use_hospital") and row.get("hospital_id") is None:
        raise HTTPException(422, "This model was trained with hospital_id; please send it.")

    df = pd.DataFrame([row])
    df["disease_id"] = 0  # placeholder, dropped inside build_features
    # astype(float): omitted vitals arrive as None (object dtype); this turns them into NaN, which XGBoost handles
    X = build_features(df)[META["features"]].astype(float)

    proba = MODEL.predict_proba(X)[0]
    idx = proba.argsort()[::-1][:top]
    predictions = [
        {"disease_id": CLASSES[i], "probability": round(float(proba[i]), 4)} for i in idx
    ]
    treatment_patient = row.copy()
    treatment_patient["gender"] = {"M": "Male", "F": "Female"}[treatment_patient["gender"]]
    treatment_recommendation = recommend_for_patient(
        treatment_patient,
        predictions[0]["disease_id"],
        patient.severity_id,
        top_k=3,
    )

    return {
        "predictions": predictions,
        "predicted_disease": predictions[0],
        "treatment_recommendation": treatment_recommendation,
        "model_classes": len(CLASSES),
    }