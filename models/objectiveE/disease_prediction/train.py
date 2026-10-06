"""
Disease classifier — XGBoost (gradient-boosted trees).

Usage:
    python train.py --data patients.csv
    python train.py --data patients.csv --use-hospital   # include hospital_id as a feature

Outputs (in ./model/):
    model.json      XGBoost model
    meta.json       feature order + label mapping (needed at inference time)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET = "disease_id"
# Whitelist of features, so the wider encounters.csv (severity, treatment, outcomes, hospital,
# labs, outbreak/atypical labels...) can never leak into the model. These are exactly the inputs
# the API / frontend send at triage time.
FEATURES = (["age", "gender", "temperature", "heart_rate", "respiratory_rate", "spo2",
             "systolic_bp", "diastolic_bp"] + [f"S{i:03d}" for i in range(1, 29)])


# ---------- Step 1: load + clean ----------
def load(path: str, use_hospital: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]} rows, {df.shape[1]} cols")

    keep = FEATURES + (["hospital_id"] if use_hospital else []) + [TARGET]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df[keep]

    df = df.dropna(subset=[TARGET])
    print(f"[load] missing values per column (top 5):\n{df.isna().sum().sort_values(ascending=False).head()}")
    return df


# ---------- Step 2: features ----------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[TARGET]).copy()

    # gender -> 0/1/NaN. XGBoost handles NaN natively, so no imputation needed.
    g = X["gender"].astype(str).str.strip().str.upper().str[0]
    X["gender"] = g.map({"M": 0, "F": 1})

    # Simple clinically meaningful derived features
    X["pulse_pressure"] = X["systolic_bp"] - X["diastolic_bp"]
    X["symptom_count"] = X.filter(regex=r"^S\d{3}$").sum(axis=1)
    return X


# ---------- Step 3: labels ----------
def encode_labels(y: pd.Series):
    # XGBoost needs classes as 0..K-1; disease_id may be arbitrary ints
    classes = sorted(y.unique())
    to_idx = {c: i for i, c in enumerate(classes)}
    return y.map(to_idx).values, classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="patients.csv", help="training CSV (or full CSV if --test not given)")
    p.add_argument("--test", default=None, help="separate test CSV, e.g. from split_data.py")
    p.add_argument("--use-hospital", action="store_true")
    p.add_argument("--out", default="model")
    args = p.parse_args()

    df = load(args.data, args.use_hospital)
    X = build_features(df)
    y, classes = encode_labels(df[TARGET])
    print(f"[labels] {len(classes)} diseases; class counts:\n{df[TARGET].value_counts().sort_index().to_string()}")

    # ---------- Step 4: split ----------
    if args.test:
        # Test set comes from its own file; carve validation (for early stopping) out of train only
        df_te = load(args.test, args.use_hospital)
        unseen = set(df_te[TARGET]) - set(classes)
        if unseen:
            print(f"[warn] test has diseases not in train, dropping them: {sorted(unseen)}")
            df_te = df_te[~df_te[TARGET].isin(unseen)]
        X_te = build_features(df_te)[X.columns]
        y_te = df_te[TARGET].map({c: i for i, c in enumerate(classes)}).values
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    else:
        # Single file: 70 / 15 / 15, stratified
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    print(f"[split] train {len(X_tr)} / val {len(X_val)} / test {len(X_te)}")

    # ---------- Step 5: train with early stopping on validation ----------
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=2000,          # upper bound; early stopping picks the real number
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
    print(f"[train] best iteration: {model.best_iteration}")

    # ---------- Step 6: evaluate on held-out test set ----------
    proba = model.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    names = [str(c) for c in classes]
    print("\n=== Test set ===")
    print(f"accuracy     : {accuracy_score(y_te, pred):.3f}")
    print(f"macro F1     : {f1_score(y_te, pred, average='macro'):.3f}")
    if len(classes) > 3:
        print(f"top-3 acc    : {top_k_accuracy_score(y_te, proba, k=3, labels=range(len(classes))):.3f}")
    print(f"baseline acc : {np.bincount(y_tr).max() / len(y_tr):.3f}  (always guess most common disease)")
    print("\n" + classification_report(y_te, pred, target_names=names, digits=3))

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top 10 features (gain):\n" + imp.head(10).round(4).to_string())

    # ---------- Step 7: save artifacts ----------
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    model.save_model(out / "model.json")
    (out / "meta.json").write_text(json.dumps({
        "features": list(X.columns),
        "classes": [int(c) if isinstance(c, (np.integer, int)) else c for c in classes],
        "use_hospital": args.use_hospital,
    }, indent=2))
    print(f"\n[save] model + meta written to {out}/")


if __name__ == "__main__":
    main()