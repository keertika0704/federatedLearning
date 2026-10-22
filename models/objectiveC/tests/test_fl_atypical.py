"""Checks for the federated atypical detector (carepath_fl.tasks.atypical)."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from carepath_fl.common.partition import hospital_ids, load_partition
from carepath_fl.tasks import atypical as T


@pytest.fixture(scope="module")
def train():
    return {h: load_partition(h, "train", columns=T.EVAL_COLUMNS) for h in hospital_ids()}


@pytest.fixture(scope="module")
def test_rows():
    return pd.concat([load_partition(h, "test", columns=T.EVAL_COLUMNS) for h in hospital_ids()])


def make_call(train, seen=None):
    def call(stage, state):
        state = json.loads(json.dumps(state))      # as it travels in a Flower message
        out = [T.client_stage(stage, state, df) for df in train.values()]
        if seen is not None:
            seen.append((stage, out))
        return out
    return call


@pytest.fixture(scope="module")
def run(train):
    seen = []
    params, info = T.run_protocol(make_call(train, seen), log=lambda *_: None)
    return params, info, seen


@pytest.fixture(scope="module")
def centralised(train):
    """Objective A's own trainer on exactly the union of the hospitals' fit / calibration slices."""
    fits, calibs = zip(*[T.split(df) for df in train.values()])
    fit, calib = pd.concat(fits), pd.concat(calibs)
    a_train = T._a_train()
    return fit, calib, a_train.fit_params(fit), a_train.build(fit, calib, T.FPR, T.SEED)


def test_counts_and_means_equal_the_centralised_fit(run, centralised):
    params, _, _ = run
    _, _, cent_fit, _ = centralised
    assert params["diseases"] == cent_fit["diseases"]
    np.testing.assert_allclose(params["prior"], cent_fit["prior"], atol=1e-12)
    np.testing.assert_allclose(params["theta"], cent_fit["theta"], atol=1e-12)
    np.testing.assert_allclose(params["vit_mean"], cent_fit["vit_mean"], atol=1e-8)
    np.testing.assert_allclose(params["vit_std"], cent_fit["vit_std"], atol=1e-6)


def test_distributed_em_fits_as_well_as_sklearn(run, centralised):
    params, _, _ = run
    fit, _, cent_fit, _ = centralised
    S, V = T.a.to_arrays(fit)
    k = fit["disease_id"].map({d: i for i, d in enumerate(T.DISEASES)}).to_numpy()

    def mean_loglik(p):
        total = n = 0
        for d in range(T.D):
            Vd = V[k == d]
            Vd = Vd[~np.isnan(Vd).any(axis=1)]
            g = p["gmm"][d]
            lp = np.stack([np.log(w) + multivariate_normal.logpdf(Vd, mean=m, cov=c, allow_singular=True)
                           for w, m, c in zip(g["weights"], g["means"], g["covs"])], axis=1)
            total += logsumexp(lp, axis=1).sum()
            n += len(Vd)
        return total / n

    assert mean_loglik(params) > mean_loglik(cent_fit) - 0.05


def test_thresholds_are_close_to_the_centralised_calibration(run, centralised):
    params, _, _ = run
    cent = centralised[3]
    for key, value in cent["threshold"].items():
        assert params["threshold"][key] == pytest.approx(value, rel=0.15), key


def test_detector_quality_on_held_out_rows(run, test_rows):
    params, _, _ = run
    m = T.score_models(params, test_rows)
    assert m["auc"] > 0.82
    assert 0.005 < m["false_alarm_rate"] < 0.06
    assert m["self_diagnosis_accuracy"] > 0.9


def test_export_loads_in_the_production_detector(run, tmp_path):
    params, info, _ = run
    out = T.write_model(params, info, out_dir=tmp_path)
    det = T.a.AtypicalDetector(path=out / "atypical_model.json")      # Objective A's own loader
    r = det.assess({"S001": 1, "S012": 1, "S013": 1, "temperature": 38.5, "heart_rate": 95,
                    "respiratory_rate": 18, "spo2": 96, "systolic_bp": 120, "diastolic_bp": 80}, disease_id="D004")
    assert {"atypical_score", "atypical_flag", "novelty_score", "reasons"} <= set(r)
    meta = json.loads((out / "atypical_model.json").read_text())["meta"]
    assert meta["rows"]["fit"] > 40000 and meta["fpr_target"] == T.FPR


def test_clients_return_only_aggregates(train, run):
    _, _, seen = run
    n_rows = {len(df) for df in train.values()}
    allowed = {T.D, len(T.SYMPTOMS), len(T.VITALS), T.N_COMP, 4, 2, T.BINS}
    assert {stage for stage, _ in seen} >= {"moments", "em", "norm1", "norm2", "cal1", "cal2", "cal3"}
    for stage, replies in seen:
        for reply in replies:
            for name, arr in reply.items():
                assert set(arr.shape) <= allowed, f"{stage}.{name} has shape {arr.shape}"
                assert not n_rows & set(arr.shape), f"{stage}.{name} looks per-row"


def test_histogram_quantiles_are_accurate():
    x = np.random.default_rng(0).lognormal(mean=1.5, sigma=0.8, size=20000)
    for q in (0.05, 0.5, 0.95, 0.99):
        assert T._quantile(T._hist(x), q) == pytest.approx(np.quantile(x, q), rel=0.01)
    z = np.random.default_rng(1).normal(size=20000)                 # robust-z scores can be negative
    assert T._quantile(T._hist(z), 0.5) == pytest.approx(np.median(z), abs=0.02)


def test_unknown_stage_and_disease_are_rejected(train):
    df = next(iter(train.values()))
    with pytest.raises(ValueError, match="unknown stage"):
        T.client_stage("cal9", {}, df)
    bad = df.copy()
    bad.iloc[0, bad.columns.get_loc("disease_id")] = "D999"
    with pytest.raises(ValueError, match="unexpected disease ids"):
        T.client_stage("moments", {}, bad)


def test_a_hospital_with_a_single_row_of_a_disease_still_splits(train):
    df = next(iter(train.values()))
    one = pd.concat([df[df["disease_id"] != "D003"], df[df["disease_id"] == "D003"].head(1)])
    fit, calib = T.split(one)
    assert len(fit) + len(calib) == len(one)
    out = T.client_stage("moments", {}, one)
    assert out["n"][T.DISEASES.index("D003")] <= 1
