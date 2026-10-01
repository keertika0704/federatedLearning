"""
Disease prediction API.

Run (put API_KEY in a .env file first, see .env.example):
    cd models/objectiveE/disease_prediction && uvicorn api:app --reload --port 8000

Docs + a built-in test form: http://localhost:8000/docs

Env vars:
    API_KEY          REQUIRED (min 16 chars). Clients send it in the X-API-Key header.
                     The server refuses to start without it.
    ALLOWED_ORIGINS  comma-separated web origins allowed to call the API
                     (default: http://localhost:3000,http://localhost:5173)

Model: always models/objectiveE/model (next to this file). Not configurable on purpose.
"""
import json
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, create_model
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent

from train import build_features  # noqa: E402  same feature logic as training

log = logging.getLogger("uvicorn.error")

# ---------- Step 0: load .env ----------
# Looks next to this file first, then walks up (so a .env in the project root works too).
# Real environment variables always win over .env values (override=False).
from dotenv import load_dotenv  # noqa: E402

for _d in [HERE, *HERE.parents]:
    if (_d / ".env").is_file():
        load_dotenv(_d / ".env", override=False)
        log.info(f"Loaded environment from {_d / '.env'}")
        break

# ---------- Step 1: load model once at startup ----------
# Pinned to the model next to this file (models/objectiveE/model). Deliberately NOT
# configurable via env var, so another objective's model can never be loaded by accident.
MODEL_DIR = (HERE / "model").resolve()
if not MODEL_DIR.is_relative_to(HERE):  # guards against model/ being a symlink to elsewhere
    raise RuntimeError(f"Model dir {MODEL_DIR} is outside {HERE}. Refusing to load.")
for _f in ("model.json", "meta.json"):
    if not (MODEL_DIR / _f).is_file():
        raise RuntimeError(
            f"Missing {MODEL_DIR / _f}. Train first:\n"
            f"  python {HERE / 'train.py'} --data <train.csv> --test <test.csv>"
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
    predictions: list[Prediction]
    model_classes: int


# ---------- Step 4: app ----------
# docs_url=None: we serve our own /docs below so it can pre-fill the key in dev.
app = FastAPI(title="Disease Classifier API", version="1.0.0", docs_url=None)

# DOCS_AUTOFILL_KEY=1 -> /docs auto-authorizes with API_KEY, but ONLY for requests from this machine.
# Never enable it on a deployed server: it would hand the key to anyone who opens /docs.
DOCS_AUTOFILL = os.getenv("DOCS_AUTOFILL_KEY", "0") == "1"
LOCALHOST = {"127.0.0.1", "::1", "localhost"}


@app.get("/docs", include_in_schema=False)
def docs(request: Request):
    autofill = DOCS_AUTOFILL and API_KEY and request.client and request.client.host in LOCALHOST
    preauth = f"ui.preauthorizeApiKey('APIKeyHeader', {json.dumps(API_KEY)});" if autofill else ""
    return HTMLResponse(f"""<!doctype html>
<html><head><title>{app.title} - docs</title><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  const ui = SwaggerUIBundle({{
    url: "{app.openapi_url}", dom_id: "#swagger-ui",
    persistAuthorization: true,   // if you paste the key manually, it survives page reloads
    onComplete: () => {{ {preauth} }}
  }});
</script></body></html>""")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/health")
def health():
    """No auth, so load balancers and uptime checks can call it."""
    return {"status": "ok", "classes": len(CLASSES), "features": len(META["features"])}


@app.post("/predict", response_model=PredictOut, dependencies=[Depends(require_api_key)])
def predict(patient: PatientIn, top: int = 3):
    top = max(1, min(top, len(CLASSES)))
    row = patient.model_dump()
    if META.get("use_hospital") and row.get("hospital_id") is None:
        raise HTTPException(422, "This model was trained with hospital_id; please send it.")

    df = pd.DataFrame([row])
    df["disease_id"] = 0  # placeholder, dropped inside build_features
    # astype(float): omitted vitals arrive as None (object dtype); this turns them into NaN, which XGBoost handles
    X = build_features(df)[META["features"]].astype(float)

    proba = MODEL.predict_proba(X)[0]
    idx = proba.argsort()[::-1][:top]
    return {
        "predictions": [{"disease_id": CLASSES[i], "probability": round(float(proba[i]), 4)} for i in idx],
        "model_classes": len(CLASSES),
    }