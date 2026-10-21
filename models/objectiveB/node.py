"""
Objective B, node layer: what ONE hospital computes locally before sharing anything.

A node reads only its own encounters and produces one summary row per (day, disease):
    n            cases diagnosed with that disease that day
    S001..S028   how many of those cases had each symptom
plus a pseudo-disease row "ATYP" for the cases objective A flags as atypical or novel (atypical_flag or novelty_flag).
Counts only - no patient identifiers, no individual rows. The central detector (detect.py) sees these summaries
and never the encounters, which is the privacy boundary of this objective.

Use from code:
    from node import NodeSummarizer
    rows = NodeSummarizer("H001", "North").summarize(df_for_H001)

Build summaries for every hospital in the encounters file and write them to the store (from the project root):
    python models/objectiveB/node.py --build
    python models/objectiveB/node.py --build --data data/generated/generated_data/encounters.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / "models" / "objectiveE").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objectiveB.store import ATYPICAL_ID, SUMMARY_COLUMNS, SYMPTOMS, Store  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "generated" / "generated_data" / "encounters.csv"
NEEDED = ["hospital_id", "region", "encounter_date", "disease_id", *SYMPTOMS]


class NodeSummarizer:
    def __init__(self, hospital_id, region, detector=None):
        self.hospital_id, self.region = hospital_id, region
        if detector is None:
            from models.objectiveA.predict import AtypicalDetector      # loads A's model once
            detector = AtypicalDetector()
        self.detector = detector

    def summarize(self, df):
        """df: this hospital's encounters. Returns a DataFrame with SUMMARY_COLUMNS, dense over its date range."""
        if df.empty:
            return pd.DataFrame(columns=SUMMARY_COLUMNS)
        if set(df["hospital_id"]) != {self.hospital_id}:
            raise ValueError(f"node {self.hospital_id} was given rows from other hospitals")
        diseases = list(self.detector.diseases)
        day = pd.to_datetime(df["encounter_date"]).dt.normalize()
        sym = df.reindex(columns=SYMPTOMS).fillna(0).astype(int)

        res = self.detector.score(df)[0]                                    # A's flags, computed locally
        flagged = ((res["atypical_flag"] == 1) | (res["novelty_flag"] == 1)).to_numpy()

        known = sym.assign(day=day.to_numpy(), disease_id=df["disease_id"].to_numpy())
        known = known[known["disease_id"].isin(diseases)]
        by_disease = known.groupby(["day", "disease_id"]).agg(n=(SYMPTOMS[0], "size"), **{s: (s, "sum") for s in SYMPTOMS})
        atyp = sym[flagged].assign(day=day[flagged].to_numpy())
        by_atyp = atyp.groupby("day").agg(n=(SYMPTOMS[0], "size"), **{s: (s, "sum") for s in SYMPTOMS})
        by_atyp.index = pd.MultiIndex.from_product([by_atyp.index, [ATYPICAL_ID]], names=["day", "disease_id"]) \
            if len(by_atyp) else pd.MultiIndex.from_arrays([[], []], names=["day", "disease_id"])
        both = pd.concat([by_disease, by_atyp])

        days = pd.date_range(day.min(), day.max(), freq="D")
        full = pd.MultiIndex.from_product([days, [*diseases, ATYPICAL_ID]], names=["day", "disease_id"])
        out = both.reindex(full, fill_value=0).reset_index()                # quiet days are explicit zero rows
        out["hospital_id"], out["region"] = self.hospital_id, self.region
        out["day"] = out["day"].dt.strftime("%Y-%m-%d")
        return out[SUMMARY_COLUMNS]


def build(data_path=DEFAULT_DATA, store=None, detector=None, verbose=True):
    """Summarize every hospital in the encounters file (one node at a time) and write the store."""
    store = store or Store()
    if detector is None:
        from models.objectiveA.predict import AtypicalDetector
        detector = AtypicalDetector()
    df = pd.read_csv(data_path, usecols=NEEDED)
    store.clear_summaries()
    total = 0
    for hospital_id, rows in df.groupby("hospital_id"):                    # each iteration = one node's private data
        node = NodeSummarizer(hospital_id, rows["region"].iat[0], detector)
        n = store.put_summaries(node.summarize(rows))
        total += n
        if verbose:
            print(f"  node {hospital_id} ({rows['region'].iat[0]:8s}) {len(rows):6,d} encounters -> {n:,d} summary rows")
    return total


def main():
    ap = argparse.ArgumentParser(description="Build per-hospital summaries for objective B")
    ap.add_argument("--build", action="store_true", help="summarize every hospital and write the store")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="encounters CSV")
    ap.add_argument("--db", help="SQLite file (default: data/outbreak.db or $OUTBREAK_DB)")
    args = ap.parse_args()
    if not args.build:
        ap.print_help()
        return
    store = Store(args.db)
    total = build(args.data, store)
    lo, hi, hospitals = store.day_range()
    print(f"Wrote {total:,} summary rows for {hospitals} hospitals ({lo} to {hi}) to {store.path}")


if __name__ == "__main__":
    main()
