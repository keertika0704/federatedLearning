"""
Split the patient CSV into train.csv and test.csv.

Usage:
    python split_data.py --data path/to/synthetic_patients.csv
    python split_data.py --data path/to/synthetic_patients.csv --test-size 0.2 --out-dir data/splits

Stratified on disease_id, so every disease appears in both files in the same proportion.
Fixed random seed, so rerunning gives the exact same split.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "disease_id"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out-dir", default=None, help="default: same folder as --data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=[TARGET])

    # Stratify needs at least 2 rows per class
    counts = df[TARGET].value_counts()
    rare = counts[counts < 2]
    if len(rare):
        print(f"[warn] diseases with <2 rows kept in train only: {list(rare.index)}")
    strat_df = df[~df[TARGET].isin(rare.index)]

    train, test = train_test_split(
        strat_df, test_size=args.test_size, stratify=strat_df[TARGET], random_state=args.seed
    )
    train = pd.concat([train, df[df[TARGET].isin(rare.index)]])

    # Sanity check: no patient in both files
    if "patient_token" in df.columns:
        overlap = set(train["patient_token"]) & set(test["patient_token"])
        assert not overlap, f"{len(overlap)} patients in both train and test"

    out = Path(args.out_dir) if args.out_dir else Path(args.data).parent
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print(f"[split] total {len(df)} -> train {len(train)} / test {len(test)}")
    print(f"[split] wrote {out / 'train.csv'} and {out / 'test.csv'}")
    print("\nDisease distribution (% of rows):")
    print(pd.DataFrame({
        "train": train[TARGET].value_counts(normalize=True).sort_index().mul(100).round(1),
        "test": test[TARGET].value_counts(normalize=True).sort_index().mul(100).round(1),
    }).to_string())


if __name__ == "__main__":
    main()