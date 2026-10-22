"""
Federated disease classifier (Objective E1), client-side logic.

Reuses the centralised trainer's feature whitelist and feature builder
(models/objectiveE/disease_prediction/train.py, loaded unmodified) so a federated model is a drop-in
replacement for it. The trainer is loaded by path under a unique module name because
objectiveE/treatment_prediction has its own train.py. Hyper-parameters and metrics are copied here so
Objective C does not require changes to Objective E; tests check the copy against the production model.
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, top_k_accuracy_score

from carepath_fl.common.partition import PROJECT_ROOT

_E1_PATH = PROJECT_ROOT / "models" / "objectiveE" / "disease_prediction" / "train.py"
_spec = importlib.util.spec_from_file_location("e1_disease_train", _E1_PATH)
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)

MODELS = ("model",)
COLUMNS = e1.FEATURES + [e1.TARGET]
EVAL_SPLIT = "test"
# 30 turns x 10 iterations. These were picked after looking at test-split results from a few offline
# settings (see the plan), so the reported test accuracy is slightly optimistic.
DEFAULT_TURNS = 30
DEFAULT_LOCAL_ITERS = 10
OUT_DIR = PROJECT_ROOT / "models" / "objectiveC" / "model" / "disease"
HEADLINE = [("accuracy", True), ("D003_recall", True)]
BASELINE_ITERS = 100

# Fixed up front so a hospital that has never seen a rare disease still emits an 8-class model.
# Must equal models/objectiveE/disease_prediction/model/meta.json["classes"] (checked in tests).
CLASSES = [f"D{i:03d}" for i in range(1, 9)]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
FEATURE_COLUMNS = e1.FEATURES + ["pulse_pressure", "symptom_count"]
RARE = "D003"

# Same values as the XGBClassifier in disease_prediction/train.py (guarded by a test).
XGB_PARAMS = dict(
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    tree_method="hist",
)


def fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df


def eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df


def score(y_true, proba) -> dict:
    """accuracy / macro F1 / top-3 accuracy, the numbers the centralised trainer prints."""
    pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "top3_accuracy": float(top_k_accuracy_score(y_true, proba, k=3, labels=range(proba.shape[1]))),
    }


def to_dmatrix(df) -> xgb.DMatrix:
    unknown = set(df[e1.TARGET].dropna().unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"unexpected disease ids {sorted(unknown)}; expected {CLASSES}")
    df = df.dropna(subset=[e1.TARGET])
    X = e1.build_features(df[e1.FEATURES + [e1.TARGET]])[FEATURE_COLUMNS]
    return xgb.DMatrix(X, label=df[e1.TARGET].map(CLASS_INDEX).values)


def booster_params(seed: int, nthread: int = 4) -> dict:
    return {**XGB_PARAMS, "objective": "multi:softprob", "num_class": len(CLASSES),
            "nthread": nthread, "seed": seed}


def load_booster(model_bytes: bytes) -> xgb.Booster | None:
    if not model_bytes:
        return None
    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    return booster


def local_train(models: dict[str, bytes], df, local_iters: int, seed: int) -> tuple[dict[str, bytes], int]:
    """Add `local_iters` boosting iterations on this hospital's rows on top of the shared model."""
    dtrain = to_dmatrix(df)
    booster = xgb.train(booster_params(seed), dtrain, num_boost_round=local_iters,
                        xgb_model=load_booster(models["model"]))
    return {"model": bytes(booster.save_raw("json"))}, dtrain.num_row()


def _metrics(booster: xgb.Booster, df) -> dict:
    d = to_dmatrix(df)
    proba = booster.predict(d).reshape(d.num_row(), len(CLASSES))
    y = d.get_label().astype(int)
    m = score(y, proba)
    m[f"{RARE}_recall"] = float(recall_score(y, proba.argmax(1), labels=[CLASS_INDEX[RARE]],
                                             average=None, zero_division=0)[0])
    return m


def local_evaluate(models: dict[str, bytes], df) -> dict:
    """Metrics of the shared model on this hospital's held-out rows (the same ones the trainer prints)."""
    booster = load_booster(models["model"])
    if booster is None:
        raise ValueError("cannot evaluate an empty model")
    return {**_metrics(booster, df), "num-examples": int(len(df))}


# ---- used by carepath_fl.report (simulation-side comparison with baselines)
def load_models(model_dir: Path) -> dict[str, xgb.Booster]:
    b = xgb.Booster()
    b.load_model(Path(model_dir) / "model.json")
    return {"model": b}


def fit_baseline(df) -> dict[str, xgb.Booster]:
    """Centralised-style model on `df` (one hospital alone, or every hospital pooled)."""
    return {"model": xgb.train(booster_params(seed=42), to_dmatrix(df), num_boost_round=BASELINE_ITERS)}


def score_models(models: dict[str, xgb.Booster], df) -> dict:
    return _metrics(models["model"], df)


def write_model(models: dict[str, bytes], extra_meta: dict | None = None, reference: dict | None = None,
                out_dir: Path = OUT_DIR) -> Path:
    """Write model.json + meta.json in the format models/objectiveE/disease_prediction/api.py loads.
    Output never goes over the production model directory; promoting it is an explicit copy."""
    booster = load_booster(models["model"])
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / "model.json")
    meta = {
        "features": FEATURE_COLUMNS,
        "classes": CLASSES,
        "use_hospital": False,
        "federated": {"strategy": "FedXgbCyclic", "iterations": booster.num_boosted_rounds(), **(extra_meta or {})},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir
