"""
Rank treatments for a patient using the trained XGBoost treatment model.

Loads (from ./model):
    success_model.json    XGBoost classifier: P(success | patient, disease, severity, treatment)
    recovery_model.json   XGBoost regressor:  expected recovery days
    meta.json             feature lists, category values, candidate treatments, metrics

Use from code:
    from predict import TreatmentRecommender
    rec = TreatmentRecommender()                       # loads ./model
    rec.recommend(patient, "D001", "SV002", top_k=3)   # list of dicts, best first

Use from the terminal:
    python predict.py                             # score the default test data (data/generated/generated_data/test.csv)
    python predict.py --csv test.csv              # score every row of any split -> test_predictions.csv + metrics
    python predict.py --disease D001 --severity SV002 --patient '{"age": 67, "gender": "M", "spo2": 86, "S001": 1}'
    python predict.py --disease D001 --severity SV002 --patient patient.json

`patient`: age, gender, vitals, and symptom flags (S001...). Symptoms not sent = absent.
Other missing fields are treated as unknown (XGBoost handles missing values).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
DEFAULT_TEST = next(p for p in HERE.parents if (p / "models" / "objectiveE").is_dir()) \
    / "data" / "generated" / "generated_data" / "test.csv"


def is_symptom(col):
    return col[:1].upper() == "S" and col[1:].isdigit()


def as_text(s):
    """Category values as plain strings; empty cells become "missing" (pandas 3 keeps NaN in astype(str))."""
    return s.astype(object).where(s.notna(), "missing").astype(str)


def encode(df, feature_cols, categories):
    """Same encoding at train and predict time: fixed category lists from meta.json.
    Unseen categories become missing rather than crashing."""
    X = df.reindex(columns=feature_cols).copy()
    for col, cats in categories.items():
        X[col] = pd.Categorical(as_text(X[col]), categories=cats)
    for col in feature_cols:
        if col not in categories:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    return X


class TreatmentRecommender:
    def __init__(self, model_dir=MODEL_DIR, success_model=None, recovery_model=None, meta=None):
        model_dir = Path(model_dir)
        if meta is None:
            meta = json.loads((model_dir / "meta.json").read_text())
        if success_model is None:
            success_model = xgb.XGBClassifier()
            success_model.load_model(model_dir / "success_model.json")
        if recovery_model is None:
            recovery_model = xgb.XGBRegressor()
            recovery_model.load_model(model_dir / "recovery_model.json")
        self.success_model, self.recovery_model, self.meta = success_model, recovery_model, meta
        self.feature_cols = meta["feature_cols"]
        self.categories = meta["categories"]
        self.patient_cols = meta["patient_cols"]
        self.cand = pd.DataFrame(meta["candidates"])
        self.cand_dis = pd.DataFrame(meta["candidates_by_disease"])
        self.tx_attr = pd.DataFrame(meta["treatment_attributes"])

    # ---- treatments mapped to this disease + severity (fallback: the disease at any severity)
    def candidates(self, disease_id, severity_id):
        c = self.cand[(self.cand.disease_id == disease_id) & (self.cand.severity_id == severity_id)]
        if c.empty:
            c = self.cand_dis[self.cand_dis.disease_id == disease_id].assign(severity_id=severity_id)
        return c

    def _score(self, rows):
        X = encode(rows, self.feature_cols, self.categories)
        rows = rows.copy()
        rows["success_prob"] = self.success_model.predict_proba(X)[:, 1]
        rows["expected_recovery_days"] = np.clip(self.recovery_model.predict(X), 1, None)
        return rows

    # ---- one patient -> ranked treatments
    def recommend(self, patient: dict, disease_id: str, severity_id: str, top_k=None):
        patient = {**{c: 0 for c in self.patient_cols if is_symptom(c)}, **patient}
        missing = [c for c in self.patient_cols if c not in patient]

        c = self.candidates(disease_id, severity_id)
        if c.empty:
            return {"disease_id": disease_id, "severity_id": severity_id, "treatments": [],
                    "missing_fields": missing, "message": "No treatments mapped for this disease"}

        rows = c.merge(self.tx_attr, on="treatment_id", how="left")
        for k in self.patient_cols:
            rows[k] = patient.get(k, np.nan)
        rows["disease_id"], rows["severity_id"] = disease_id, severity_id
        ranked = (self._score(rows)
                  .sort_values(["success_prob", "expected_recovery_days"], ascending=[False, True]))
        if top_k:
            ranked = ranked.head(top_k)

        treatments = [{
            "rank": i + 1,
            "treatment_id": r.treatment_id,
            "treatment_name": r.treatment_name,
            "success_probability": round(float(r.success_prob), 4),
            "expected_recovery_days": round(float(r.expected_recovery_days), 1),
            "guideline_priority": int(r.priority),
            "first_line": bool(r.first_line),
            "inpatient_required": bool(r.inpatient_required),
        } for i, r in enumerate(ranked.itertuples())]
        return {"disease_id": disease_id, "severity_id": severity_id, "treatments": treatments,
                "missing_fields": missing, "disclaimer": "Clinical decision support only - not a prescription."}

    # ---- many patients at once (train.py's policy check and predict.py --csv)
    def top_picks(self, df):
        base = df[["patient_token", "disease_id", "severity_id"] + self.patient_cols]
        exact = base.merge(self.cand, on=["disease_id", "severity_id"])
        rest = base[~base.patient_token.isin(exact.patient_token)]
        rows = pd.concat([exact, rest.merge(self.cand_dis, on="disease_id")], ignore_index=True)
        rows = self._score(rows.merge(self.tx_attr, on="treatment_id", how="left"))
        best = (rows.sort_values(["success_prob", "expected_recovery_days"], ascending=[False, True])
                    .drop_duplicates("patient_token").set_index("patient_token"))
        guide = (rows.sort_values(["priority", "first_line"], ascending=[True, False])
                     .drop_duplicates("patient_token").set_index("patient_token"))
        t = df["patient_token"]
        return pd.DataFrame({
            "model_pick": best["treatment_id"].reindex(t).values,
            "model_pick_name": best["treatment_name"].reindex(t).values,
            "model_pick_success_prob": best["success_prob"].reindex(t).round(4).values,
            "model_pick_recovery_days": best["expected_recovery_days"].reindex(t).round(1).values,
            "guide_pick": guide["treatment_id"].reindex(t).values,
        }, index=df.index)

    def score_given(self, df):
        """Predictions for the treatment each row actually received."""
        scored = self._score(df)
        return scored["success_prob"].round(4), scored["expected_recovery_days"].round(1)


def predict_csv(rec, path, out_path):
    """Score every row of a split (e.g. test.csv): predictions for the treatment given,
    the model's recommended treatment, and metrics if the outcome columns are present."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in ["patient_token", "disease_id", "severity_id", "treatment_id"] if c not in df]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")

    df["pred_success_prob"], df["pred_recovery_days"] = rec.score_given(df)
    picks = rec.top_picks(df)
    out = pd.concat([df, picks], axis=1)
    out["model_agrees_with_guideline"] = (out.model_pick == out.guide_pick).astype(int)

    keep = ["patient_token", "disease_id", "severity_id", "treatment_id",
            "pred_success_prob", "pred_recovery_days",
            "model_pick", "model_pick_name", "model_pick_success_prob", "model_pick_recovery_days",
            "guide_pick", "model_agrees_with_guideline"]
    keep += [c for c in ["success", "recovery_days", "_oracle_success_prob"] if c in out]
    out[keep].to_csv(out_path, index=False)
    print(f"Scored {len(out):,} rows from {path}\nWrote {out_path}")

    if "success" not in out:      # unlabeled data: predictions only
        return
    from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
    y, p = out["success"].astype(int), out["pred_success_prob"]
    print(f"\nSuccess   AUC {roc_auc_score(y, p):.4f} | Brier {brier_score_loss(y, p):.4f}")
    if "_oracle_success_prob" in out:
        print(f"          correlation with true probability {np.corrcoef(p, out['_oracle_success_prob'])[0, 1]:.3f}")
    if "recovery_days" in out:
        print(f"Recovery  MAE {mean_absolute_error(out['recovery_days'], out['pred_recovery_days']):.2f} days")
    print(f"\nModel agrees with guideline for {out.model_agrees_with_guideline.mean():.0%} of patients")
    for col, label in [("guide_pick", "guideline (priority 1)"), ("model_pick", "model top pick")]:
        got = out["treatment_id"] == out[col]
        print(f"  {label:24s} given to {got.sum():5,} | success {out.loc[got, 'success'].mean():.1%} "
              f"vs {out.loc[~got, 'success'].mean():.1%} otherwise")


def main():
    ap = argparse.ArgumentParser(description="Single patient: --disease --severity --patient.  "
                                             "Whole file: --csv test.csv")
    ap.add_argument("--csv", help=f"score every row of this file (default when no patient is given: {DEFAULT_TEST})")
    ap.add_argument("--out", default=str(HERE / "test_predictions.csv"))
    ap.add_argument("--disease")
    ap.add_argument("--severity")
    ap.add_argument("--patient", help="JSON string or path to a .json file")
    ap.add_argument("--top-k", type=int)
    args = ap.parse_args()

    rec = TreatmentRecommender()
    single = args.disease and args.severity and args.patient
    if args.csv or not single:       # no patient given -> score the test data
        predict_csv(rec, args.csv or DEFAULT_TEST, args.out)
        return
    p = args.patient
    patient = json.loads(Path(p).read_text()) if Path(p).is_file() else json.loads(p)
    print(json.dumps(rec.recommend(patient, args.disease, args.severity, args.top_k), indent=2))


if __name__ == "__main__":
    main()