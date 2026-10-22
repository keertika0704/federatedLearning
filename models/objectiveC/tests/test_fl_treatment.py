"""Checks for the federated treatment recommender (carepath_fl.tasks.treatment)."""
import json

import pandas as pd
import pytest

from carepath_fl.common.partition import PROJECT_ROOT, load_partition
from carepath_fl.tasks import treatment

E2_META = PROJECT_ROOT / "models" / "objectiveE" / "treatment_prediction" / "model" / "meta.json"


@pytest.fixture(scope="module")
def prod_meta():
    return json.loads(E2_META.read_text())


def test_frozen_schema_matches_production_model(prod_meta):
    assert treatment.FEATURE_COLS == prod_meta["feature_cols"]
    assert treatment.patient_cols() == prod_meta["patient_cols"]
    assert treatment.categories() == prod_meta["categories"]
    ours = json.loads(treatment.treatment_attributes().to_json(orient="records"))
    assert sorted(ours, key=lambda r: r["treatment_id"]) == sorted(prod_meta["treatment_attributes"],
                                                                   key=lambda r: r["treatment_id"])


def test_hyperparameters_match_production_model(prod_meta):
    for name, ours in (("success", treatment.SUCCESS_PARAMS), ("recovery", treatment.RECOVERY_PARAMS)):
        theirs = prod_meta["params"][name]
        for key, value in ours.items():
            assert theirs[key] == value, f"{name}.{key}: {theirs[key]!r} != {value!r}"


def test_client_reads_only_whitelisted_columns():
    df = load_partition("H004", "train", columns=treatment.COLUMNS)
    assert set(df.columns) == set(treatment.COLUMNS)
    for banned in ("hospital_id", "_oracle_success_prob", "treatment_name", "encounter_id"):
        assert banned not in df.columns


@pytest.fixture(scope="module")
def rows():
    return load_partition("H009", "train", columns=treatment.COLUMNS)


def test_validation_slice_keeps_patients_together_and_disjoint(rows):
    fit, val = treatment.fit_rows(rows), treatment.eval_rows(rows)
    assert len(fit) + len(val) == len(rows)
    assert not set(fit["patient_token"]) & set(val["patient_token"])
    assert 0.08 < len(val) / len(rows) < 0.12


@pytest.fixture(scope="module")
def trained(rows):
    sample = treatment.fit_rows(rows).sample(900, random_state=0)
    first, n = treatment.local_train({m: b"" for m in treatment.MODELS}, sample, local_iters=2, seed=1)
    second, _ = treatment.local_train(first, sample, local_iters=1, seed=2)
    return sample, first, second, n


def test_both_models_gain_iterations_each_turn(trained):
    sample, first, second, n = trained
    assert n == len(sample)
    for name in treatment.MODELS:
        assert treatment.load_booster(first[name]).num_boosted_rounds() == 2
        assert treatment.load_booster(second[name]).num_boosted_rounds() == 3


def test_success_model_stops_growing_at_its_cap(trained, monkeypatch):
    sample, *_ = trained
    monkeypatch.setattr(treatment, "SUCCESS_MAX_ITERS", 3)
    models = {m: b"" for m in treatment.MODELS}
    counts = []
    for turn in range(1, 4):
        models, _ = treatment.local_train(models, sample, local_iters=2, seed=turn)
        counts.append({m: treatment.load_booster(models[m]).num_boosted_rounds() for m in treatment.MODELS})
    assert [c["success"] for c in counts] == [2, 3, 3]      # 2, then capped at 3, then unchanged
    assert [c["recovery"] for c in counts] == [2, 4, 6]     # recovery keeps growing


def test_evaluate_reports_all_metrics(trained):
    sample, _, second, _ = trained
    m = treatment.local_evaluate(second, sample)
    assert {"auc", "brier", "log_loss", "recovery_mae", "num-examples"} <= set(m)
    assert 0 <= m["auc"] <= 1


def test_query_returns_combinations_only(rows):
    q = treatment.local_query(treatment.fit_rows(rows))
    assert set(q) == {"candidates"}
    assert set(q["candidates"][0]) == {"disease_id", "severity_id", "treatment_id", "priority", "first_line"}


def test_merge_takes_union_and_minimum():
    a = {"candidates": [{"disease_id": "D001", "severity_id": "SV001", "treatment_id": "T001", "priority": 2, "first_line": 1}]}
    b = {"candidates": [{"disease_id": "D001", "severity_id": "SV001", "treatment_id": "T001", "priority": 1, "first_line": 0},
                        {"disease_id": "D002", "severity_id": "SV002", "treatment_id": "T003", "priority": 1, "first_line": 1}]}
    merged = pd.DataFrame(treatment.merge_queries([a, b])["candidates"]).set_index(["disease_id", "severity_id", "treatment_id"])
    assert len(merged) == 2
    assert merged.loc[("D001", "SV001", "T001")].tolist() == [1, 0]


def test_export_is_a_drop_in_for_the_production_recommender(trained, rows, tmp_path):
    sample, _, second, _ = trained
    reference = treatment.merge_queries([treatment.local_query(treatment.fit_rows(rows))])
    out = treatment.write_model(second, {"turns": 3}, reference, out_dir=tmp_path)

    rec = treatment.e2.TreatmentRecommender(model_dir=out)      # the unmodified production class
    patient = {"age": 40, "gender": "Male", "temperature": 39.0, "spo2": 96, "S001": 1}
    disease, severity = rows.iloc[0][["disease_id", "severity_id"]]
    result = rec.recommend(patient, disease, severity)
    assert result["treatments"], "expected at least one candidate treatment"
    top = result["treatments"][0]
    assert 0 < top["success_probability"] < 1 and top["expected_recovery_days"] >= 1

    picks = rec.top_picks(sample.assign(patient_token=sample["patient_token"]))
    assert picks["model_pick"].notna().all()


def test_export_without_candidates_is_refused(trained, tmp_path):
    _, _, second, _ = trained
    with pytest.raises(ValueError, match="candidate"):
        treatment.write_model(second, {}, None, out_dir=tmp_path)


def test_shared_models_carry_no_identifiers(trained):
    sample, _, second, _ = trained
    for name in treatment.MODELS:
        text = second[name].decode()
        for banned in ("patient_token", "hospital_id", "_oracle_success_prob", sample["patient_token"].iloc[0]):
            assert banned not in text
