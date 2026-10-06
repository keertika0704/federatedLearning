"""
Train the treatment recommender (XGBoost).

Two models, one row = one (patient, disease, severity, treatment) with its outcome:
    success_model  : XGBClassifier  -> P(success)
    recovery_model : XGBRegressor   -> recovery days
At prediction time every treatment mapped to the disease + severity is scored and ranked
(an "S-learner": treatment is a feature, and we ask the model about each candidate).

Reads ONLY
    train.csv, test.csv  - your split of encounters.csv (or patient_treatments.csv): patient features +
    treatment details + outcome in every row. Only the whitelisted columns below are used, so the wider
    encounters table (hospital, labs, admission/ICU/death, outbreak labels...) cannot leak in.
    The candidate treatments per disease/severity are taken from these rows too.

Writes (model/)
    success_model.json, recovery_model.json, meta.json

Run
    pip install pandas numpy xgboost scikit-learn
    python models/objectiveE/treatment_prediction/train.py --train data/generated/generated_data/train.csv \
        --test data/generated/generated_data/test.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predict import MODEL_DIR, TreatmentRecommender, as_text, encode

HERE = Path(__file__).resolve().parent
LABELS = ["success", "recovery_days"]
DROP = ["patient_token", "hospital_id", "_oracle_success_prob", "treatment_name"]
CAT_CANDIDATES = ["gender", "disease_id", "severity_id", "treatment_id",
                  "treatment_category", "treatment_type", "route"]

SUCCESS_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=5, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", enable_categorical=True, early_stopping_rounds=50,
                      random_state=42)
RECOVERY_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=5, min_child_weight=5,
                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                       objective="reg:absoluteerror", eval_metric="mae",
                       tree_method="hist", enable_categorical=True, early_stopping_rounds=50,
                       random_state=42)


# Whitelist: only these columns are read. Everything else in encounters.csv is either a consequence of
# the outcome (admitted, icu, died, length_of_stay_days, has_complication...), not available at
# recommendation time (labs, imaging, hospital, dates) or a label for another objective.
PATIENT_FEATURES = (["age", "gender", "temperature", "heart_rate", "respiratory_rate", "spo2",
                     "systolic_bp", "diastolic_bp"] + [f"S{i:03d}" for i in range(1, 29)])
CONTEXT_FEATURES = ["disease_id", "severity_id"]
TREATMENT_FEATURES = ["treatment_id", "treatment_category", "treatment_type", "route",
                      "inpatient_required", "priority", "first_line"]
KEEP = (["patient_token", "treatment_name"] + CONTEXT_FEATURES + PATIENT_FEATURES + TREATMENT_FEATURES
        + LABELS + ["_oracle_success_prob"])   # patient_token / treatment_name / oracle: not features (see DROP)

TX_COLS = ["treatment_id", "treatment_name", "treatment_category", "treatment_type", "route",
           "inpatient_required"]
REQUIRED = ["patient_token", "disease_id", "severity_id", "treatment_id", "priority", "first_line",
            "age", "success", "recovery_days"]


# ------------------------------------------------------------------ data
def load_split(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}.\n"
                         f"Regenerate patient_treatments.csv with the latest generate_treatments.py "
                         f"(it now includes patient features and treatment details), then re-split it.")
    return df[[c for c in KEEP if c in df.columns]]


def reference_from_data(df):
    """Candidate treatments and treatment attributes, taken from the data itself.
    The generator draws every mapped treatment for every disease+severity, so with enough
    patients every mapping row shows up here."""
    cand = (df.groupby(["disease_id", "severity_id", "treatment_id"], as_index=False)
              [["priority", "first_line"]].min())
    cand_dis = df.groupby(["disease_id", "treatment_id"], as_index=False)[["priority", "first_line"]].min()
    tx_attr = df[[c for c in TX_COLS if c in df.columns]].drop_duplicates("treatment_id")
    return cand, cand_dis, tx_attr


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(HERE / "train.csv"))
    ap.add_argument("--test", default=str(HERE / "test.csv"))
    args = ap.parse_args()

    train, test = load_split(args.train), load_split(args.test)
    cand, cand_dis, tx_attr = reference_from_data(pd.concat([train, test], ignore_index=True))
    print(f"Loaded {args.train} and {args.test}")
    print(f"Candidate table: {len(cand)} disease/severity/treatment combos, "
          f"{cand.disease_id.nunique()} diseases, {len(tx_attr)} treatments")
    overlap = set(train.patient_token) & set(test.patient_token)
    if overlap:
        print(f"WARNING: {len(overlap)} patients appear in both train and test - scores will be optimistic")

    cat_cols = [c for c in CAT_CANDIDATES if c in train.columns]
    feature_cols = [c for c in train.columns if c not in LABELS + DROP]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    not_patient = {"disease_id", "severity_id", "priority", "first_line", *tx_attr.columns}
    patient_cols = [c for c in feature_cols if c not in not_patient]   # age, gender, vitals, symptoms
    # report gaps so you can fix them at the source
    gaps = {c: int(train[c].isna().sum() + test[c].isna().sum()) for c in feature_cols}
    gaps = {c: n for c, n in gaps.items() if n}
    if gaps:
        print(f"\nNote: empty values (treated as 'missing'/unknown): {gaps}")

    categories = {c: sorted(set(as_text(train[c])) | set(as_text(test[c]))
                            | (set(as_text(cand[c])) if c in cand else set())
                            | (set(as_text(tx_attr[c])) if c in tx_attr else set()))
                  for c in cat_cols}

    Xtr, Xte = encode(train, feature_cols, categories), encode(test, feature_cols, categories)
    ytr, yte = train["success"].astype(int), test["success"].astype(int)
    dtr, dte = train["recovery_days"].astype(float), test["recovery_days"].astype(float)
    print(f"\ntrain {len(train):,} rows | test {len(test):,} rows | {len(feature_cols)} features")
    print(f"success rate: train {ytr.mean():.1%}, test {yte.mean():.1%}")

    # early-stopping slice carved out of train (test stays untouched until the end)
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
                            .split(Xtr, ytr, groups=train["patient_token"]))

    # ---- success model
    success_model = xgb.XGBClassifier(**SUCCESS_PARAMS)
    success_model.fit(Xtr.iloc[fit_idx], ytr.iloc[fit_idx],
                      eval_set=[(Xtr.iloc[val_idx], ytr.iloc[val_idx])], verbose=False)

    results = {}
    def score(name, p):
        results[name] = {"auc": roc_auc_score(yte, p), "brier": brier_score_loss(yte, p),
                         "log_loss": log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))}

    score("constant", np.full(len(test), ytr.mean()))
    rate = ytr.groupby(train["treatment_id"]).mean()
    score("treatment_only", test["treatment_id"].map(rate).fillna(ytr.mean()).to_numpy())
    logreg = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)])),
        ("clf", LogisticRegression(max_iter=2000))])
    lr_in = lambda d: d[feature_cols].assign(**{c: as_text(d[c]) for c in cat_cols})
    score("logistic_regression", logreg.fit(lr_in(train), ytr).predict_proba(lr_in(test))[:, 1])
    p_te = success_model.predict_proba(Xte)[:, 1]
    score("xgboost", p_te)
    has_oracle = "_oracle_success_prob" in test.columns
    if has_oracle:
        score("ceiling_true_probs", test["_oracle_success_prob"].to_numpy())

    print(f"\nSuccess model on test (trees used: {success_model.best_iteration + 1}):")
    print(pd.DataFrame(results).T.round(4).to_string())
    corr = float(np.corrcoef(p_te, test["_oracle_success_prob"])[0, 1]) if has_oracle else None
    if corr is not None:
        print(f"Correlation with the simulator's true probability: {corr:.3f}")

    # ---- recovery model
    recovery_model = xgb.XGBRegressor(**RECOVERY_PARAMS)
    recovery_model.fit(Xtr.iloc[fit_idx], dtr.iloc[fit_idx],
                       eval_set=[(Xtr.iloc[val_idx], dtr.iloc[val_idx])], verbose=False)
    mae = mean_absolute_error(dte, recovery_model.predict(Xte))
    mae_naive = mean_absolute_error(dte, np.full(len(test), dtr.median()))
    print(f"\nRecovery days MAE on test: xgboost {mae:.2f} | naive median {mae_naive:.2f}")

    # ---- feature importance (top 10, by gain)
    imp = pd.Series(success_model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print("\nTop features for success (gain):")
    print(imp.head(10).round(2).to_string())

    # ---- meta + policy check
    records = lambda d: json.loads(d.to_json(orient="records"))
    meta = {
        "model": "XGBoost S-learner: P(success) classifier + recovery-days regressor",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__, "sklearn_version": sklearn.__version__,
        "feature_cols": feature_cols, "categories": categories, "patient_cols": patient_cols,
        "candidates": records(cand), "candidates_by_disease": records(cand_dis),
        "treatment_attributes": records(tx_attr),
        "params": {"success": {k: v for k, v in SUCCESS_PARAMS.items()},
                   "recovery": {k: v for k, v in RECOVERY_PARAMS.items()}},
        "best_iteration": {"success": int(success_model.best_iteration),
                           "recovery": int(recovery_model.best_iteration)},
        "rows": {"train": len(train), "test": len(test)},
    }
    rec = TreatmentRecommender(success_model=success_model, recovery_model=recovery_model, meta=meta)
    picks = rec.top_picks(test)
    agree = float((picks.model_pick == picks.guide_pick).mean())
    policy = {}
    print(f"\nPolicy check on test (model agrees with guideline for {agree:.0%} of patients):")
    for col, label in [("guide_pick", "guideline (priority 1)"), ("model_pick", "model top pick")]:
        got = test["treatment_id"].values == picks[col].values
        policy[col] = {"followed": int(got.sum()),
                       "success_when_followed": float(test.loc[got, "success"].mean()),
                       "success_otherwise": float(test.loc[~got, "success"].mean())}
        print(f"  {label:24s} followed for {got.sum():5,} | success {policy[col]['success_when_followed']:.1%} "
              f"when followed vs {policy[col]['success_otherwise']:.1%} otherwise")

    meta["metrics"] = {
        "success": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in results.items()},
        "oracle_correlation": round(corr, 4) if corr is not None else None,
        "recovery_mae": {"xgboost": round(float(mae), 3), "naive_median": round(float(mae_naive), 3)},
        "policy": {"model_guideline_agreement": round(agree, 4),
                   **{k: {m: round(v, 4) if isinstance(v, float) else v for m, v in d.items()}
                      for k, d in policy.items()}},
    }

    # ---- save
    MODEL_DIR.mkdir(exist_ok=True)
    success_model.save_model(MODEL_DIR / "success_model.json")
    recovery_model.save_model(MODEL_DIR / "recovery_model.json")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {MODEL_DIR}/: success_model.json, recovery_model.json, meta.json")
    print("Patient fields the API expects:", patient_cols)


if __name__ == "__main__":
    main()