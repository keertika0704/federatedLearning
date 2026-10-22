"""Checks for the federated disease classifier (carepath_fl.tasks.disease). Run: pytest models/objectiveC/tests"""
import ast
import json

import pytest

from carepath_fl.common.partition import PROJECT_ROOT, hospital_ids, load_partition
from carepath_fl.tasks import disease

E1_DIR = PROJECT_ROOT / "models" / "objectiveE" / "disease_prediction"


def test_frozen_schema_matches_production_model():
    meta = json.loads((E1_DIR / "model" / "meta.json").read_text())
    assert disease.CLASSES == meta["classes"]
    assert disease.FEATURE_COLUMNS == meta["features"]


def test_hyperparameters_match_the_centralised_trainer():
    """Our copy of XGB_PARAMS must equal the keyword arguments of the XGBClassifier in train.py."""
    tree = ast.parse((E1_DIR / "train.py").read_text())
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "XGBClassifier")
    literal = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
               if kw.arg and isinstance(kw.value, ast.Constant)}   # skips e.g. num_class=len(classes)
    assert literal, "expected literal keyword arguments in train.py's XGBClassifier(...) call"
    assert {k: literal.get(k) for k in disease.XGB_PARAMS} == disease.XGB_PARAMS


def test_partition_returns_only_that_hospital():
    df = load_partition("H003", "train")
    assert set(df["hospital_id"]) == {"H003"}
    assert len(hospital_ids()) == 10


def test_partition_rejects_unknown_hospital_and_split():
    with pytest.raises(ValueError):
        load_partition("H999", "train")
    with pytest.raises(ValueError):
        load_partition("H001", "validation")


def test_partition_columns_are_whitelisted():
    df = load_partition("H003", "train", columns=disease.COLUMNS)
    assert set(df.columns) == set(disease.COLUMNS)   # no hospital_id, patient_token, outcomes...


@pytest.fixture(scope="module")
def sample():
    return load_partition("H009", "train").sample(800, random_state=0)


def test_cyclic_turns_add_iterations_and_stay_8_class(sample):
    # A hospital that has never seen the rare disease must still emit an 8-class model.
    no_rare = sample[sample["disease_id"] != "D003"]
    assert "D003" not in set(no_rare["disease_id"])

    first, n = disease.local_train({"model": b""}, no_rare, local_iters=2, seed=1)
    assert n == len(no_rare)
    second, _ = disease.local_train(first, sample, local_iters=1, seed=2)

    assert disease.load_booster(first["model"]).num_boosted_rounds() == 2
    assert disease.load_booster(second["model"]).num_boosted_rounds() == 3
    metrics = disease.local_evaluate(second, sample)
    assert {"accuracy", "macro_f1", "top3_accuracy", "D003_recall", "num-examples"} <= set(metrics)
    d = disease.to_dmatrix(sample)
    assert disease.load_booster(second["model"]).predict(d).shape == (len(sample), len(disease.CLASSES))


def test_unknown_disease_id_is_rejected(sample):
    bad = sample.copy()
    bad.iloc[0, bad.columns.get_loc("disease_id")] = "D999"
    with pytest.raises(ValueError, match="unexpected disease ids"):
        disease.to_dmatrix(bad)


def test_shared_model_carries_no_identifier_columns(sample):
    models, _ = disease.local_train({"model": b""}, sample, local_iters=2, seed=0)
    model_bytes = models["model"]
    learner = json.loads(model_bytes)["learner"]
    assert set(learner["feature_names"]) == set(disease.FEATURE_COLUMNS)
    text = model_bytes.decode()
    for banned in ("patient_token", "encounter_id", "hospital_id", sample["patient_token"].iloc[0]):
        assert banned not in text
