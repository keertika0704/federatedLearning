"""The read-only API behind the web app's Federated learning tab (models/objectiveC/api.py)."""
import json
import os

import pytest

os.environ.setdefault("OBJECTIVE_E_API_KEY", "x" * 32)       # backend.security refuses to import without one

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.security import require_api_key  # noqa: E402
from models.objectiveC import api  # noqa: E402

TASK_IDS = ["disease", "treatment", "atypical", "outbreak"]


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(api.router)
    return app


@pytest.fixture
def client(app):
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app)


def test_overview_requires_the_api_key(app):
    assert TestClient(app).get("/api/federated/overview").status_code == 401


def test_overview_lists_every_task_and_hospital(client):
    body = client.get("/api/federated/overview").json()
    assert [t["id"] for t in body["tasks"]] == TASK_IDS
    assert len(body["hospitals"]) == 10 and body["hospitals"][0]["id"] == "H001"
    assert body["protections"] and body["limits"] and body["scope_note"]
    for t in body["tasks"]:
        assert t["travels"] and t["notes"] and t["method"] and t["question"]


@pytest.mark.skipif(not (api.MODEL_DIR / "disease" / "report.json").is_file(), reason="disease model not trained")
def test_trained_tasks_carry_comparable_numbers(client):
    tasks = {t["id"]: t for t in client.get("/api/federated/overview").json()["tasks"]}
    for tid in ("disease", "treatment", "atypical"):
        t = tasks[tid]
        assert t["available"] and t["metrics"]
        for m in t["metrics"]:
            assert len(m["per_hospital"]) == 10
            assert all(m[k] is not None for k in ("alone", "federated", "centralized"))
    assert tasks["outbreak"]["comparison"]["rows"]
    # the replay needs the recorded turns, each naming a real hospital
    for tid, turns in (("disease", 30), ("treatment", 80)):
        history = tasks[tid]["history"]
        assert history and len(history) == turns
        assert {h["hospital"] for h in history} <= {f"H{i:03d}" for i in range(1, 11)}
        assert all(m in history[0]["metrics"] for m in tasks[tid]["replay_metrics"])
    assert tasks["atypical"]["history"] is None and tasks["outbreak"]["history"] is None


def test_untrained_models_say_how_to_train_them(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    tasks = client.get("/api/federated/overview").json()["tasks"]
    assert [t["available"] for t in tasks] == [False] * 4
    assert all("flwr run" in t["hint"] and t["id"] in t["hint"] for t in tasks)
    assert all(t["metrics"] == [] and t["history"] is None for t in tasks)


def test_a_report_without_history_still_renders(client, monkeypatch, tmp_path):
    """Models trained before turn history was recorded must not break the page."""
    d = tmp_path / "disease"
    d.mkdir()
    (d / "report.json").write_text(json.dumps({"per_hospital": [], "weighted_mean": {}, "hospital_wins_vs_local": {}}))
    (d / "meta.json").write_text(json.dumps({"federated": {"turns": 3, "local_iters": 1, "iterations": 3}}))
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    disease = client.get("/api/federated/overview").json()["tasks"][0]
    assert disease["available"] and disease["history"] is None


def test_response_exposes_no_node_ids_keys_or_paths(client):
    text = client.get("/api/federated/overview").text.lower()
    for banned in ("node_id", "registry", "private key", "secrets", "begin openssh", "/users/", "api_key"):
        assert banned not in text


def test_comparison_requires_the_api_key(app):
    assert TestClient(app).get("/api/federated/comparison").status_code == 401


def test_comparison_without_a_file_says_how_to_make_it(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    body = client.get("/api/federated/comparison").json()
    assert body["available"] is False and "carepath_fl.compare" in body["hint"] and body["tasks"] == []


@pytest.mark.skipif(not (api.MODEL_DIR / "comparison.json").is_file(), reason="comparison not generated")
def test_comparison_adds_each_models_question_and_names_the_hospitals(client):
    body = client.get("/api/federated/comparison").json()
    assert body["available"] and len(body["hospitals"]) == 10
    assert [t["id"] for t in body["tasks"]] == TASK_IDS
    assert all(t["question"] and t["model"] for t in body["tasks"])
    assert body["tasks"][0]["metrics"][0]["per_hospital"][0]["hospital"] == "H001"
