"""
Predict diseases for new patients with the trained model.

Usage:
    python predict.py --data new_patients.csv            # writes predictions.csv
    python predict.py --data new_patients.csv --top 3    # top-3 diseases with probabilities
"""
import argparse
import json

import pandas as pd
from xgboost import XGBClassifier

from models.objectiveE.disease_prediction.train import build_features  # same feature logic as training — never duplicate it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-dir", default="model")
    p.add_argument("--top", type=int, default=1)
    p.add_argument("--out", default="predictions.csv")
    args = p.parse_args()

    meta = json.loads(open(f"{args.model_dir}/meta.json").read())
    model = XGBClassifier()
    model.load_model(f"{args.model_dir}/model.json")

    raw = pd.read_csv(args.data)
    ids = raw.get("patient_token", pd.Series(range(len(raw))))
    raw = raw.drop(columns=[c for c in ["disease_id"] if c in raw.columns])
    raw["disease_id"] = 0  # placeholder so build_features can drop it uniformly
    X = build_features(raw)[meta["features"]]  # enforce training column order

    proba = model.predict_proba(X)
    classes = meta["classes"]
    rows = []
    for pid, pr in zip(ids, proba):
        top = pr.argsort()[::-1][: args.top]
        row = {"patient_token": pid}
        for rank, i in enumerate(top, 1):
            row[f"disease_{rank}"] = classes[i]
            row[f"prob_{rank}"] = round(float(pr[i]), 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.head(10).to_string(index=False))
    print(f"\nWrote {len(out)} predictions to {args.out}")


if __name__ == "__main__":
    main()