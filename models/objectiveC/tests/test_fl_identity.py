"""Hospital identity binding (carepath_fl.common.identity and the server's stage rounds)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from carepath_fl import server_app
from carepath_fl.common.arrays import pack_arrays
from carepath_fl.common.identity import ENV, IdentityError, NodeRegistry

REGISTRY = {11: "H001", 12: "H002", 13: "H003"}


def test_registry_is_read_from_the_environment(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({str(k): v for k, v in REGISTRY.items()}))
    monkeypatch.setenv(ENV, str(path))
    assert NodeRegistry.from_env().hospital_by_node == REGISTRY


def test_no_registry_means_no_enforcement(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert NodeRegistry.from_env() is None


def test_missing_registry_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "nope.json"))
    with pytest.raises(IdentityError, match="register"):
        NodeRegistry.from_env()


def test_one_hospital_cannot_have_two_nodes():
    with pytest.raises(ValueError):
        NodeRegistry({1: "H001", 2: "H001"})


def test_unregistered_and_missing_nodes_are_refused():
    reg = NodeRegistry(REGISTRY)
    reg.check_complete([11, 12, 13])
    with pytest.raises(IdentityError, match="not registered"):
        reg.check_complete([11, 12, 13, 99])
    with pytest.raises(IdentityError, match="H003"):
        reg.check_complete([11, 12])
    reg.check_connected([11, 12])          # boosting turns may use a subset, but only registered nodes


def test_a_reply_cannot_claim_to_be_another_hospital():
    reg = NodeRegistry(REGISTRY)
    assert reg.check_reply(11, "H001") == "H001"
    assert reg.check_reply(11, None) == "H001"
    with pytest.raises(IdentityError, match="claims to be H002"):
        reg.check_reply(11, "H002")
    with pytest.raises(IdentityError, match="unregistered"):
        reg.check_reply(99, "H001")


def test_a_hospital_cannot_answer_twice():
    with pytest.raises(IdentityError, match="more than once"):
        NodeRegistry(REGISTRY).check_replies([11, 11, 12], [None, None, None])


# ---- through the server's real stage round, with a fake grid
def fake_grid(replies):
    return SimpleNamespace(get_node_ids=lambda: [sender for sender, _ in replies],
                           send_and_receive=lambda messages: [
                               SimpleNamespace(has_error=lambda: False, metadata=SimpleNamespace(src_node_id=sender),
                                               content={"arrays": pack_arrays(arrays)}) for sender, arrays in replies])


def payload(hospital: str | None):
    arrays = {"n": np.array([1, 2, 3])}
    if hospital:
        arrays["hospital"] = np.frombuffer(hospital.encode(), dtype=np.uint8)
    return arrays


@pytest.fixture(autouse=True)
def plain_messages(monkeypatch):
    """Flower's Message needs an active run context; only the reply handling is under test here."""
    monkeypatch.setattr(server_app, "Message", lambda **kw: SimpleNamespace(**kw))


@pytest.fixture
def enforced(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({str(k): v for k, v in REGISTRY.items()}))
    monkeypatch.setenv(ENV, str(path))


def test_stage_round_accepts_honest_replies(enforced):
    grid = fake_grid([(11, payload("H001")), (12, payload("H002")), (13, payload(None))])
    assert len(server_app.stage_caller(grid, "outbreak")("summaries", {})) == 3


def test_stage_round_rejects_a_forged_hospital_claim(enforced):
    grid = fake_grid([(11, payload("H001")), (12, payload("H001")), (13, payload("H003"))])   # node 12 claims H001
    with pytest.raises(IdentityError, match="claims to be H001"):
        server_app.stage_caller(grid, "outbreak")("summaries", {})


def test_stage_round_rejects_a_missing_hospital(enforced):
    grid = fake_grid([(11, payload("H001")), (12, payload("H002"))])
    with pytest.raises(IdentityError, match="H003"):
        server_app.stage_caller(grid, "outbreak")("summaries", {})


def test_stage_round_without_registry_is_unchanged(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    grid = fake_grid([(1, payload("H001")), (2, payload("H009"))])         # nothing to check against
    assert len(server_app.stage_caller(grid, "outbreak")("summaries", {})) == 2
