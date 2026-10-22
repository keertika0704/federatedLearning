"""Checks for the federated outbreak pipeline (carepath_fl.tasks.outbreak).

The tests point the local detector at Objective A's production model, so they do not depend on the
federated atypical output, and so the federated pipeline can be compared exactly with B's own centralised
one on identical inputs."""
import json

import pandas as pd
import pytest

from carepath_fl.common.partition import DATA_DIR, hospital_ids, load_partition
from carepath_fl.tasks import outbreak as T

PROD_A = T.PROJECT_ROOT / "models" / "objectiveA" / "model" / "atypical_model.json"
ENCOUNTERS = DATA_DIR / "encounters.csv"
SMALL_GRID = {"p_thresh": [0.01, 0.001], "min_ratio": [1.5], "min_sim": [0.5], "min_count": [5]}


@pytest.fixture(scope="module")
def a_params():
    return json.loads(PROD_A.read_text())


@pytest.fixture(scope="module")
def hospital_rows():
    df = pd.read_csv(ENCOUNTERS, usecols=sorted(set(T.COLUMNS) | {"hospital_id"}))
    return {h: g.reset_index(drop=True) for h, g in df.groupby("hospital_id")}


@pytest.fixture(scope="module")
def parts(a_params, hospital_rows):
    return [T.client_stage("summaries", {"a_params": a_params}, hospital_rows[h]) for h in sorted(hospital_rows)]


@pytest.fixture(scope="module")
def central(a_params, hospital_rows):
    """B's own centralised inputs: NodeSummarizer on every hospital, Truth read from encounters.csv."""
    det = T.atypical.a.AtypicalDetector(params=a_params)
    frames = [T.b_node.NodeSummarizer(h, g["region"].iat[0], det).summarize(g) for h, g in sorted(hospital_rows.items())]
    summaries = pd.concat(frames, ignore_index=True)
    summaries["day"] = pd.to_datetime(summaries["day"])
    panel = T.b_detect.Panel(summaries)
    return summaries, panel, T.b_train.Truth(panel, ENCOUNTERS)


def test_partition_all_returns_the_whole_record_of_one_hospital():
    df = load_partition("H009", "all", columns=T.COLUMNS)
    assert set(df.columns) == set(T.COLUMNS)
    assert len(df) > load_partition("H009", "train", columns=["age"]).shape[0]


def test_summaries_equal_the_centralised_node_summaries(parts, central, a_params):
    summaries, _, _ = central
    got, _ = T.assemble(parts, a_params["diseases"])
    key = ["hospital_id", "day", "disease_id"]
    a = got.sort_values(key).reset_index(drop=True)
    b = summaries[a.columns].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_episodes_give_the_same_truth_as_the_labelled_encounters(parts, central, a_params, tmp_path):
    _, panel, truth = central
    _, episodes = T.assemble(parts, a_params["diseases"])
    fed = T.truth_from_episodes(panel, episodes, tmp_path)
    assert sorted(fed.episodes) == sorted(truth.episodes)
    assert (fed.active == truth.active).all() and (fed.near == truth.near).all()
    assert len(fed.linked) == len(truth.linked)


def test_only_counts_and_episode_dates_leave_the_hospital(parts):
    for p in parts:
        assert set(p) == {"hospital", "region", "day", "disease", "n", "sym", "ep_disease", "ep_start", "ep_end"}
        assert p["sym"].shape[1] == len(T.SYMPTOMS) and p["sym"].dtype.kind == "i"
        assert (p["sym"] <= p["n"][:, None]).all()          # symptom counts are among that day's cases


def test_federated_calibration_equals_the_centralised_one(parts, central, a_params, monkeypatch):
    summaries, panel, truth = central
    monkeypatch.setattr(T.b_train, "GRID", SMALL_GRID)
    monkeypatch.setattr(T, "load_atypical_params", lambda: a_params)
    (fed_params, info) = T.run_protocol(lambda stage, state: parts, log=lambda *_: None)

    stats = T.b_detect.Stats(panel, T.b_detect.DEFAULTS)
    first = T.b_train.first_eval_day(T.b_detect.DEFAULTS)
    test_t = int((pd.Timestamp(T.TEST_START) - panel.days[0]).days)
    cen_params, _, _ = T.b_train.calibrate(T.b_detect.OutbreakDetector(params={}), stats, truth, first,
                                           test_t - 1, T.MAX_FALSE)
    assert fed_params["params"] == cen_params
    assert info["labelled_outbreaks"] == len(truth.episodes) and info["hospitals"] == 10
    assert {"calibration", "test", "test_with_default_thresholds"} == set(info["metrics"])


def test_export_loads_in_the_production_detector(parts, a_params, tmp_path, monkeypatch):
    monkeypatch.setattr(T.b_train, "GRID", SMALL_GRID)
    monkeypatch.setattr(T, "load_atypical_params", lambda: a_params)
    params, info = T.run_protocol(lambda stage, state: parts, log=lambda *_: None)
    out = T.write_model(params, info, out_dir=tmp_path)
    det = T.b_detect.OutbreakDetector(path=out / "outbreak_model.json")     # B's own loader
    assert det.trained and det.p["p_thresh"] in SMALL_GRID["p_thresh"]


def test_missing_atypical_model_is_reported_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "ATYPICAL_MODEL", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="atypical"):
        T.load_atypical_params()


def test_unknown_stage_is_rejected(hospital_rows, a_params):
    with pytest.raises(ValueError, match="unknown stage"):
        T.client_stage("nope", {"a_params": a_params}, hospital_rows[hospital_ids()[0]])
