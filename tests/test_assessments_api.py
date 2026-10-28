"""The assessment log (backend/assessments.py) and its feed into outbreak surveillance."""
import os

import pytest

os.environ.setdefault("OBJECTIVE_E_API_KEY", "x" * 32)       # backend.security refuses to import without one

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import assessments  # noqa: E402
from backend.security import require_api_key  # noqa: E402
from models.objectiveB import store as store_module  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(assessments, "DB_PATH", tmp_path / "assessments.db")
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "outbreak.db")
    app = FastAPI()
    app.include_router(assessments.router)
    return app


@pytest.fixture
def client(app):
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app)


def body(disease="D004", symptoms=("S001", "S006"), atypical=False, **extra):
    return {"age": 34, "gender": "F", "vitals": {"temperature": 38.4, "spo2": None}, "symptoms": list(symptoms),
            "disease_id": disease, "confidence": 0.82, "atypical_flag": atypical,
            "reasons": ["unexpected symptom: Rash (seen in 1% of cases)"] if atypical else [], **extra}


def test_requires_the_api_key(app):
    assert TestClient(app).get("/api/assessments").status_code == 401
    assert TestClient(app).post("/api/assessments", json=body()).status_code == 401


def test_logged_assessment_is_listed_with_its_atypical_findings(client):
    made = client.post("/api/assessments", json=body(atypical=True, severity_id="SV002", treatment="Oseltamivir")).json()
    listed = client.get("/api/assessments").json()
    assert listed["total"] == 1
    row = listed["assessments"][0]
    assert row["id"] == made["id"] and row["atypical_flag"] and row["severity_id"] == "SV002"
    assert row["symptoms"] == ["S001", "S006"] and row["vitals"]["temperature"] == 38.4 and row["vitals"]["spo2"] is None
    assert "name" not in row and "patient_name" not in row


def test_severity_can_be_changed_after_logging(client):
    made = client.post("/api/assessments", json=body()).json()
    r = client.post(f"/api/assessments/{made['id']}/severity", json={"severity_id": "SV003", "treatment": "Oseltamivir"})
    assert r.status_code == 200 and r.json()["severity_id"] == "SV003"
    assert client.post("/api/assessments/999/severity", json={"severity_id": "SV001"}).status_code == 404


def test_rejects_unknown_symptoms_and_bad_values(client):
    assert client.post("/api/assessments", json=body(symptoms=["S999"])).status_code == 422
    assert client.post("/api/assessments", json={**body(), "age": 400}).status_code == 422
    assert client.post("/api/assessments", json={**body(), "disease_id": "flu"}).status_code == 422


def test_assessments_become_node_summaries_counts_only(client):
    client.post("/api/assessments", json=body("D004", ("S001", "S006")))
    client.post("/api/assessments", json=body("D004", ("S001",), atypical=True))
    client.post("/api/assessments", json=body("D001", ("S002",)))
    df = store_module.Store().load_summaries().set_index("disease_id")
    assert set(df.index) == {"D004", "D001", "ATYP"}
    assert set(df["hospital_id"]) == {assessments.HOSPITAL_ID}
    assert df.loc["D004", "n"] == 2 and df.loc["D004", "S001"] == 2 and df.loc["D004", "S006"] == 1
    assert df.loc["ATYP", "n"] == 1 and df.loc["ATYP", "S001"] == 1
    assert df.loc["D001", "n"] == 1


def test_clinic_summary_counts_the_last_week(client):
    for _ in range(3):
        client.post("/api/assessments", json=body("D004"))
    client.post("/api/assessments", json=body("D001", atypical=True))
    s = client.get("/api/assessments/clinic-summary").json()
    assert s["cases"] == 4 and s["atypical"] == 1
    assert s["by_disease"][0] == {"disease_id": "D004", "cases": 3}


def test_one_clinic_node_does_not_move_the_default_scan_day(tmp_path):
    store = store_module.Store(tmp_path / "x.db")
    import pandas as pd
    row = lambda h, d: {"hospital_id": h, "region": "North", "day": d, "disease_id": "D004", "n": 1,   # noqa: E731
                        **{s: 0 for s in store_module.SYMPTOMS}}
    store.put_summaries(pd.DataFrame([row("H001", "2025-12-30"), row("H002", "2025-12-30"), row("LIVE", "2026-10-27")])[store_module.SUMMARY_COLUMNS])
    assert store.day_range()[1] == "2026-10-27"
    assert store.last_shared_day() == "2025-12-30"
