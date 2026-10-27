"""Original-vs-federated comparison (carepath_fl/compare.py): the statistics, and the file the web page reads."""
import json

import numpy as np
import pytest

from carepath_fl import compare


def test_verdict_needs_the_interval_to_leave_zero():
    assert compare._verdict([0.001, 0.01], True) == "better"
    assert compare._verdict([-0.01, -0.001], True) == "worse"
    assert compare._verdict([-0.01, 0.01], True) == "no clear difference"
    assert compare._verdict([0.0, 0.01], True) == "no clear difference"          # touching zero is not clear
    assert compare._verdict([0.001, 0.01], False) == "worse"                     # lower is better: an increase is worse
    assert compare._verdict([-0.01, -0.001], False) == "better"
    assert compare._verdict([0.001, 0.01], None) is None                         # shown, not ranked
    assert compare._verdict(None, True) is None


def test_patient_clusters_are_resampled_whole():
    clusters = np.array([7, 7, 3, 3, 3, 9])                                      # three patients, 2 + 3 + 1 rows
    for idx in compare._resamples(len(clusters), clusters, np.random.default_rng(0))[:50]:
        counts = np.bincount(idx, minlength=len(clusters))
        assert counts[0] == counts[1] and counts[2] == counts[3] == counts[4]    # a patient is in or out as a whole


def test_rows_are_resampled_individually_without_clusters():
    resamples = compare._resamples(100, None, np.random.default_rng(0))
    assert len(resamples) == compare.RESAMPLES and all(len(r) == 100 for r in resamples)


def test_paired_metric_reports_federated_minus_original():
    values = {"original": np.full(20, 0.5), "federated": np.full(20, 0.6)}
    hospital = np.repeat([f"H{i:03d}" for i in range(1, 11)], 2)          # two rows at each of the ten hospitals
    spec = dict(id="m", label="M", format="num3", higher=True, ci=True, per_hospital=True,
                fn=lambda idx, which: float(values[which][idx].mean()))
    resamples = compare._resamples(20, None, np.random.default_rng(1))
    m = compare._metric(spec, hospital, resamples)
    assert m["original"] == pytest.approx(0.5) and m["federated"] == pytest.approx(0.6)
    assert m["difference"] == pytest.approx(0.1) and m["verdict"] == "better"
    assert [r["hospital"] for r in m["per_hospital"]] == [f"H{i:03d}" for i in range(1, 11)]
    assert m["hospitals_better"] == 10


SAVED = compare.OUT_FILE


@pytest.mark.skipif(not SAVED.is_file(), reason="run `python -m carepath_fl.compare` first")
def test_saved_comparison_is_internally_consistent():
    data = json.loads(SAVED.read_text())
    assert [t["id"] for t in data["tasks"]] == ["disease", "treatment", "atypical", "outbreak"]
    for t in data["tasks"]:
        assert t["facts"] and t["notes"]
        for m in t.get("metrics", []):
            assert m["difference"] == pytest.approx(m["federated"] - m["original"], abs=2e-5)
            assert m["verdict"] == compare._verdict(m["ci"], m["higher_is_better"])
            if m["ci"]:
                assert m["ci"][0] <= m["difference"] + 1e-4 and m["difference"] - 1e-4 <= m["ci"][1]
            if m["per_hospital"]:
                assert len(m["per_hospital"]) == 10
                assert sum(r["rows"] for r in m["per_hospital"]) == t["test_rows"]
    outbreak = data["tasks"][3]
    assert outbreak["kind"] == "recorded" and outbreak["rows"]
