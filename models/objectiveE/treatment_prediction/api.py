"""
HTTP API for the treatment recommender.

Run:
    uvicorn backend.treatment_prediction_api:app --reload --port 8002
    (run from the project root)

Endpoints:
    GET  /api/treatment/health      -> model loaded? version, metrics
    POST /api/treatment/recommend   -> ranked treatments for one patient

Example request body:
    {
      "disease_id": "D001",
      "severity_id": "SV002",
      "top_k": 3,
      "patient": {"age": 67, "gender": "M", "temperature": 39.2, "heart_rate": 118,
                  "systolic_bp": 92, "spo2": 86, "S001": 1, "S004": 1}
    }
Symptoms not listed are treated as absent. Other missing fields are treated as unknown
and listed back in "missing_fields".
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "models" / "objectiveE").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objectiveE.treatment_prediction.predict import TreatmentRecommender

app = FastAPI(title="Treatment Recommender", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

recommender = TreatmentRecommender()   # loads model/ once at startup


class RecommendRequest(BaseModel):
    disease_id: str = Field(..., examples=["D001"])
    severity_id: str = Field(..., examples=["SV002"])
    patient: Dict[str, Any] = Field(..., description="age, gender, vitals, symptom flags (S001...)")
    top_k: Optional[int] = Field(None, ge=1)


@app.get("/api/treatment/health")
def health():
    m = recommender.meta
    return {"status": "ok", "model": m["model"], "trained_at": m["trained_at"],
            "xgboost_version": m["xgboost_version"], "patient_fields": recommender.patient_cols,
            "metrics": m.get("metrics")}


@app.post("/api/treatment/recommend")
def recommend(req: RecommendRequest):
    known = set(recommender.cand_dis.disease_id)
    if req.disease_id not in known:
        raise HTTPException(404, f"Unknown disease_id {req.disease_id}. Known: {sorted(known)}")
    return recommender.recommend(req.patient, req.disease_id, req.severity_id, req.top_k)