import json
import logging
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, create_model
from xgboost import XGBClassifier

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "models" / "objectiveE").is_dir()
)
FEDERATED_MODELS = PROJECT_ROOT / "models" / "objectiveC" / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objectiveE.disease_prediction.train import build_features  # noqa: E402
from backend.security import require_api_key  # noqa: E402

log = logging.getLogger("uvicorn.error")

# ---------- Step 1: load model once at startup ----------
# Pinned to the federated disease model trained by Objective C. Deliberately NOT configurable via env var,
# so another objective's model can never be loaded by accident.
MODEL_DIR = (FEDERATED_MODELS / "disease").resolve()
if not MODEL_DIR.is_relative_to(FEDERATED_MODELS.resolve()):
    raise RuntimeError(f"Model dir {MODEL_DIR} is outside {FEDERATED_MODELS}. Refusing to load.")
for _f in ("model.json", "meta.json"):
    if not (MODEL_DIR / _f).is_file():
        raise RuntimeError(
            f"Missing {MODEL_DIR / _f}. Train the federated model first (from the project root):\n"
            f"  flwr run models/objectiveC --run-config 'task=\"disease\"'"
        )
META = json.loads((MODEL_DIR / "meta.json").read_text())
MODEL = XGBClassifier()
MODEL.load_model(MODEL_DIR / "model.json")
CLASSES = META["classes"]
SYMPTOMS = [f"S{i:03d}" for i in range(1, 29)]
log.info(f"Loaded model from {MODEL_DIR} ({len(CLASSES)} classes)")

# ---------- Step 2: request / response schemas ----------
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
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52, "gender": "F", "temperature": 38.4, "heart_rate": 96,
                "respiratory_rate": 22, "spo2": 94, "systolic_bp": 128, "diastolic_bp": 82,
                "S001": 1, "S006": 1, "S017": 1,
            }
        }
    }


class Prediction(BaseModel):
    disease_id: int | str
    probability: float


class PredictOut(BaseModel):
    predicted_disease: Prediction
    other_predictions: list[Prediction]
    model_classes: int


# ---------- Step 3: routes ----------
router = APIRouter(tags=["disease"])


@router.get("/health")
def health():
    """No auth, so load balancers and uptime checks can call it."""
    return {"status": "ok", "classes": len(CLASSES), "features": len(META["features"])}


@router.post("/predict", response_model=PredictOut, dependencies=[Depends(require_api_key)])
def predict(patient: PatientIn):
    row = patient.model_dump()
    if META.get("use_hospital") and row.get("hospital_id") is None:
        raise HTTPException(422, "This model was trained with hospital_id; please send it.")

    df = pd.DataFrame([row])
    df["disease_id"] = 0  # placeholder, dropped inside build_features
    # astype(float): omitted vitals arrive as None (object dtype); this turns them into NaN
    X = build_features(df)[META["features"]].astype(float)

    proba = MODEL.predict_proba(X)[0]
    top_indices = proba.argsort()[::-1][:3]
    ranked = [
        {"disease_id": CLASSES[int(i)], "probability": round(float(proba[i]), 4)}
        for i in top_indices
    ]
    return {
        "predicted_disease": ranked[0],
        "other_predictions": ranked[1:],
        "model_classes": len(CLASSES),
    }