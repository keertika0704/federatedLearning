"""Replaying a scan day must not add a second copy of an outbreak that is already in the history."""
import os

os.environ.setdefault("OBJECTIVE_E_API_KEY", "x" * 32)

from models.objectiveB.detect import describe  # noqa: E402
from models.objectiveB.store import Store  # noqa: E402


def alert(day, disease="D004"):
    sites = [{"hospital_id": "H004", "hospital_name": "A", "region": "South", "recent_count": 31, "expected": 19},
             {"hospital_id": "H008", "hospital_name": "B", "region": "Central", "recent_count": 30, "expected": 14}]
    a = {"disease_key": disease, "disease_id": disease, "kind": "known_disease", "disease_name": "Influenza", "day": day,
         "sites": sites, "similarity": 0.9, "signature": []}
    a.update(describe(a))
    return a


def test_replaying_a_resolved_day_adds_nothing(tmp_path):
    store = Store(tmp_path / "o.db")
    rec, change = store.upsert_alert(alert("2025-12-18"), rebuild=describe)
    assert change == "new"
    store.resolve_stale("2026-01-20", 14)                       # a later scan closes it
    assert [a["status"] for a in store.list_alerts()] == ["resolved"]
    for _ in range(3):                                           # the same day scanned again and again
        again, change = store.upsert_alert(alert("2025-12-18"), rebuild=describe)
        assert change == "known" and again["id"] == rec["id"] and again["status"] == "resolved"
    assert len(store.list_alerts()) == 1


def test_a_new_episode_after_resolution_is_still_recorded(tmp_path):
    store = Store(tmp_path / "o.db")
    store.upsert_alert(alert("2025-12-18"), rebuild=describe)
    store.resolve_stale("2026-01-20", 14)
    _, change = store.upsert_alert(alert("2026-03-01"), rebuild=describe)     # a genuinely later outbreak
    assert change == "new" and len(store.list_alerts()) == 2


def test_other_diseases_are_not_treated_as_known(tmp_path):
    store = Store(tmp_path / "o.db")
    store.upsert_alert(alert("2025-12-18"), rebuild=describe)
    store.resolve_stale("2026-01-20", 14)
    _, change = store.upsert_alert(alert("2025-12-18", "D001"), rebuild=describe)
    assert change == "new"
