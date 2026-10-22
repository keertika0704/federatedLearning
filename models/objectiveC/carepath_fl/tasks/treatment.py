"""
Federated treatment recommender (Objective E2), client-side logic.

Two XGBoost models share every hospital turn: P(success) (binary:logistic) and recovery days
(reg:absoluteerror). The centralised trainer builds its category lists and candidate treatments from
all rows; here the category lists are frozen from public reference tables, and the candidate table is
the union of what each hospital reports having seen (a Flower query round: combinations only, no
patient rows, no counts).

The encoder and recommender of models/objectiveE/treatment_prediction/predict.py are loaded unmodified
by path, so the exported files are a drop-in for TreatmentRecommender. Feature lists and
hyper-parameters are copied here and checked against the production meta.json in the tests.
"""
import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from carepath_fl.common.partition import PROJECT_ROOT

_E2_PATH = PROJECT_ROOT / "models" / "objectiveE" / "treatment_prediction" / "predict.py"
_spec = importlib.util.spec_from_file_location("e2_treatment_predict", _E2_PATH)
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference" / "master"

MODELS = ("success", "recovery")
EVAL_SPLIT = "train"            # rounds are validated on a hospital's own held-out slice of train
# Chosen from pooled validation loss only (each hospital's own held-out slice, never the test split):
# the recovery model plateaus at ~100-150 trees, while the success model overfits past ~60.
DEFAULT_TURNS = 80
DEFAULT_LOCAL_ITERS = 2
SUCCESS_MAX_ITERS = 60
OUT_DIR = PROJECT_ROOT / "models" / "objectiveC" / "model" / "treatment"
HEADLINE = [("auc", True), ("recovery_mae", False)]

PATIENT_FEATURES = (["age", "gender", "temperature", "heart_rate", "respiratory_rate", "spo2",
                     "systolic_bp", "diastolic_bp"] + [f"S{i:03d}" for i in range(1, 29)])
CONTEXT_FEATURES = ["disease_id", "severity_id"]
TREATMENT_FEATURES = ["treatment_id", "treatment_category", "treatment_type", "route",
                      "inpatient_required", "priority", "first_line"]
FEATURE_COLS = CONTEXT_FEATURES + PATIENT_FEATURES + TREATMENT_FEATURES
CAT_COLS = ["gender", "disease_id", "severity_id", "treatment_id", "treatment_category",
            "treatment_type", "route"]
TX_COLS = ["treatment_id", "treatment_name", "treatment_category", "treatment_type", "route",
           "inpatient_required"]
LABELS = ["success", "recovery_days"]
# The only columns a client reads. patient_token is only used to keep one patient's rows together
# when carving the validation slice; it is never a feature and never leaves the hospital.
COLUMNS = ["patient_token"] + FEATURE_COLS + LABELS

_BASE = dict(learning_rate=0.05, max_depth=5, min_child_weight=5, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist")
SUCCESS_PARAMS = {**_BASE, "objective": "binary:logistic", "eval_metric": "logloss"}
RECOVERY_PARAMS = {**_BASE, "objective": "reg:absoluteerror", "eval_metric": "mae"}
_PARAMS = {"success": SUCCESS_PARAMS, "recovery": RECOVERY_PARAMS}


# ------------------------------------------------------------------ frozen schema (public reference)
@functools.cache
def _master(name: str) -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / f"{name}_master.csv", encoding="utf-8-sig")


@functools.cache
def treatment_attributes() -> pd.DataFrame:
    tx = _master("treatment")[TX_COLS].copy()
    tx["inpatient_required"] = (tx["inpatient_required"] == "Yes").astype(int)
    return tx


@functools.cache
def categories() -> dict[str, list[str]]:
    tx = treatment_attributes()
    cats = {"gender": ["Female", "Male"],
            "disease_id": sorted(_master("disease")["disease_id"].astype(str)),
            "severity_id": sorted(_master("severity")["severity_id"].astype(str))}
    for col in ("treatment_id", "treatment_category", "treatment_type", "route"):
        cats[col] = sorted(set(e2.as_text(tx[col])))
    return {c: cats[c] for c in CAT_COLS}


def patient_cols() -> list[str]:
    not_patient = {"disease_id", "severity_id", "priority", "first_line", *treatment_attributes().columns}
    return [c for c in FEATURE_COLS if c not in not_patient]


# ------------------------------------------------------------------ local data
def _split_idx(df: pd.DataFrame):
    # one patient never lands on both sides; same seed as the centralised trainer's early-stopping slice
    return next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
                .split(df, groups=df["patient_token"]))


def fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[_split_idx(df)[0]]


def eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[_split_idx(df)[1]]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return e2.encode(df, FEATURE_COLS, categories())


def _dmatrix(X: pd.DataFrame, label=None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=label, enable_categorical=True)


def _labels(df: pd.DataFrame) -> dict:
    return {"success": df["success"].astype(int).to_numpy(),
            "recovery": df["recovery_days"].astype(float).to_numpy()}


def load_booster(raw: bytes) -> xgb.Booster | None:
    if not raw:
        return None
    b = xgb.Booster()
    b.load_model(bytearray(raw))
    return b


def _params(name: str, seed: int, nthread: int = 4) -> dict:
    return {**_PARAMS[name], "nthread": nthread, "seed": seed}


# ------------------------------------------------------------------ client side
def local_train(models: dict[str, bytes], df, local_iters: int, seed: int) -> tuple[dict[str, bytes], int]:
    """Add up to `local_iters` boosting iterations to each shared model on this hospital's rows.
    The success model stops growing at SUCCESS_MAX_ITERS (it overfits beyond that)."""
    X, y = _features(df), _labels(df)
    out = {}
    for name in MODELS:
        prev = load_booster(models[name])
        rounds = local_iters
        if name == "success":
            rounds = min(local_iters, SUCCESS_MAX_ITERS - (prev.num_boosted_rounds() if prev else 0))
        if rounds <= 0:
            out[name] = models[name]
            continue
        b = xgb.train(_params(name, seed), _dmatrix(X, y[name]), num_boost_round=rounds, xgb_model=prev)
        out[name] = bytes(b.save_raw("json"))
    return out, len(df)


def _predict(booster: xgb.Booster, d: xgb.DMatrix) -> np.ndarray:
    """Early-stopped baselines use their best iteration (as XGBClassifier does); federated ones all trees."""
    try:
        rng = (0, booster.best_iteration + 1)
    except AttributeError:
        rng = (0, 0)
    return booster.predict(d, iteration_range=rng)


def _metrics(boosters: dict[str, xgb.Booster], df) -> dict:
    X, y = _features(df), _labels(df)
    d = _dmatrix(X)
    p = np.clip(_predict(boosters["success"], d), 1e-6, 1 - 1e-6)
    rec = _predict(boosters["recovery"], d)
    return {"auc": float(roc_auc_score(y["success"], p)),
            "brier": float(brier_score_loss(y["success"], p)),
            "log_loss": float(log_loss(y["success"], p)),
            "recovery_mae": float(mean_absolute_error(y["recovery"], rec))}


def local_evaluate(models: dict[str, bytes], df) -> dict:
    boosters = {n: load_booster(models[n]) for n in MODELS}
    if any(b is None for b in boosters.values()):
        raise ValueError("cannot evaluate an empty model")
    return {**_metrics(boosters, df), "num-examples": int(len(df))}


def local_query(df) -> dict:
    """Which disease/severity/treatment combinations this hospital has seen, with the guideline
    priority / first_line of each (a property of the combination, not of any patient)."""
    cand = (df.groupby(["disease_id", "severity_id", "treatment_id"], as_index=False)
              [["priority", "first_line"]].min())
    return {"candidates": json.loads(cand.to_json(orient="records"))}


def merge_queries(parts: list[dict]) -> dict:
    rows = pd.DataFrame([r for p in parts for r in p["candidates"]])
    cand = rows.groupby(["disease_id", "severity_id", "treatment_id"], as_index=False)[["priority", "first_line"]].min()
    return {"candidates": json.loads(cand.to_json(orient="records"))}


# ------------------------------------------------------------------ export (TreatmentRecommender format)
def write_model(models: dict[str, bytes], extra_meta: dict | None = None, reference: dict | None = None,
                out_dir: Path = OUT_DIR) -> Path:
    """success_model.json, recovery_model.json and meta.json as models/objectiveE/treatment_prediction
    loads them. Never writes over the production model directory; promoting is an explicit copy."""
    if not reference:
        raise ValueError("the treatment export needs the merged candidate table (reference)")
    records = lambda d: json.loads(d.to_json(orient="records"))
    cand = pd.DataFrame(reference["candidates"])
    cand_dis = cand.groupby(["disease_id", "treatment_id"], as_index=False)[["priority", "first_line"]].min()
    boosters = {n: load_booster(models[n]) for n in MODELS}

    out_dir.mkdir(parents=True, exist_ok=True)
    boosters["success"].save_model(out_dir / "success_model.json")
    boosters["recovery"].save_model(out_dir / "recovery_model.json")
    meta = {
        "model": "Federated XGBoost S-learner (FedXgbCyclic): P(success) classifier + recovery-days regressor",
        "xgboost_version": xgb.__version__,
        "feature_cols": FEATURE_COLS, "categories": categories(), "patient_cols": patient_cols(),
        "candidates": records(cand), "candidates_by_disease": records(cand_dis),
        "treatment_attributes": records(treatment_attributes()),
        "params": {"success": SUCCESS_PARAMS, "recovery": RECOVERY_PARAMS},
        "federated": {"strategy": "FedXgbCyclic",
                      "iterations": {n: b.num_boosted_rounds() for n, b in boosters.items()},
                      **(extra_meta or {})},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


# ------------------------------------------------------------------ used by carepath_fl.report
def load_models(model_dir: Path) -> dict[str, xgb.Booster]:
    out = {}
    for name, fname in (("success", "success_model.json"), ("recovery", "recovery_model.json")):
        b = xgb.Booster()
        b.load_model(Path(model_dir) / fname)
        out[name] = b
    return out


def fit_baseline(df, max_iters: int = 1000) -> dict[str, xgb.Booster]:
    """What the centralised trainer does, on `df` (one hospital alone, or every hospital pooled):
    early stopping on a patient-grouped 10% slice, same hyper-parameters."""
    fit, val = fit_rows(df), eval_rows(df)
    Xf, yf, Xv, yv = _features(fit), _labels(fit), _features(val), _labels(val)
    out = {}
    for name in MODELS:
        out[name] = xgb.train({**_params(name, seed=42)}, _dmatrix(Xf, yf[name]), num_boost_round=max_iters,
                              evals=[(_dmatrix(Xv, yv[name]), "val")], early_stopping_rounds=50,
                              verbose_eval=False)
    return out


def score_models(models: dict[str, xgb.Booster], df) -> dict:
    return _metrics(models, df)
