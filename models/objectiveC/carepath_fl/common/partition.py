"""
Per-hospital data access for federated clients.

In a real deployment each hospital's SuperNode reads only its own extract. In simulation the shared
CSVs are filtered by hospital_id here, so a client can never see another hospital's rows.
"""
import os
from pathlib import Path

import pandas as pd

# models/objectiveC/carepath_fl/common/partition.py -> repo root. CAREPATH_ROOT overrides it when the
# app is installed somewhere other than the checkout (e.g. a hospital's own SuperNode host).
PROJECT_ROOT = Path(os.environ.get("CAREPATH_ROOT") or Path(__file__).resolve().parents[4])
DATA_DIR = PROJECT_ROOT / "data" / "generated" / "generated_data"
HOSPITAL_MASTER = PROJECT_ROOT / "data" / "reference" / "master" / "hospital_master.csv"


def hospital_ids() -> list[str]:
    """H001..H010, in the order used as Flower partition ids."""
    return sorted(pd.read_csv(HOSPITAL_MASTER, usecols=["hospital_id"])["hospital_id"].astype(str))


def load_partition(hospital_id: str, split: str = "train", columns: list[str] | None = None) -> pd.DataFrame:
    """Rows of one hospital from train.csv / test.csv, or its whole record from encounters.csv ('all')."""
    if split not in ("train", "test", "all"):
        raise ValueError(f"split must be 'train', 'test' or 'all', got {split!r}")
    path = DATA_DIR / ("encounters.csv" if split == "all" else f"{split}.csv")
    usecols = None if columns is None else sorted(set(columns) | {"hospital_id"})
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["hospital_id"].astype(str) == hospital_id]
    if df.empty:
        raise ValueError(f"no rows for hospital {hospital_id!r} in {path}")
    return df.drop(columns=["hospital_id"]) if columns is not None and "hospital_id" not in columns else df
