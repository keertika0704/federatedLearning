"""
Step 1 of the treatment model: give every synthetic patient a treatment and an outcome.

Why this exists
    synthetic_patients.csv has no treatments or outcomes, so there is nothing to train a
    treatment model on. This script simulates "what the doctor gave" and "did it work",
    using your reference tables plus a few transparent clinical rules (below).

How the simulation works
    1. Candidates: rows in disease_treatment_mapping for the patient's (disease_id, severity_id).
       If that exact combo has no rows, fall back to any row for the disease.
    2. Doctor's choice: mostly follows priority / first_line, but with some variation
       (EXPLORE share picked at random). Variation matters: without it the model never
       sees how the *other* treatments would have done, and can't rank them.
    3. Outcome: success probability depends on severity, age and vitals, and on whether
       the treatment fits the patient (e.g. oxygen helps a low-SpO2 patient, IV fluids help
       a tachycardic/hypotensive one, inpatient treatments suit severe cases).
       Then success ~ Bernoulli(p), and recovery_days follows from severity and success.

    These rules are made up. Say so in the review: the model learns these rules,
    which proves the pipeline, not clinical truth.

Reads
    data/generated/generated_data/synthetic_patients.csv
    disease_treatment_mapping, treatment_master, severity_master   (searched under data/config and data/reference)

Writes
    data/generated/generated_data/patient_treatments.csv   - one self-contained row per patient:
        patient_token, hospital_id, disease_id, severity_id,
        treatment_id, treatment_name, treatment_category, treatment_type, route,
        inpatient_required (0/1), priority, first_line (0/1),
        age, gender, vitals..., S001...S028,
        success (0/1), recovery_days, _oracle_success_prob
    Split this file into train.csv / test.csv; train.py needs nothing else.

    _oracle_success_prob is the true simulated probability. It is ONLY for evaluating the
    model later (does it rank the truly best treatment first?). Never use it as a feature.

Run (from anywhere)
    python data/generated/generators/generate_treatments.py
    python data/generated/generators/generate_treatments.py --seed 7 --explore 0.3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---- paths (same layout as generate_synthetic_patients.py) ----
SCRIPT_DIR = Path(__file__).resolve().parent      # data/generated/generators
DATA_ROOT = SCRIPT_DIR.parents[1]                 # data
INPUT_DIRS = [DATA_ROOT / "config", DATA_ROOT / "reference"]
OUTPUT_DIR = DATA_ROOT / "generated" / "generated_data"

# ---- simulation settings ----
EXPLORE = 0.25          # share of patients whose treatment is picked uniformly from the candidates
BASE_DAYS = [4, 7, 12, 18, 25]   # typical recovery days by severity level (mild -> critical)


# ---------------------------------------------------------------- loading
def load(name):
    """Find <name>.csv (or '<name> copy.csv') anywhere under data/config or data/reference."""
    for d in INPUT_DIRS:
        hits = sorted(d.rglob(f"{name}*.csv")) if d.exists() else []
        if hits:
            df = pd.read_csv(hits[0])
            df.columns = [c.strip() for c in df.columns]
            return df
    raise FileNotFoundError(f"Couldn't find {name}.csv under {', '.join(map(str, INPUT_DIRS))}")


def yes(v):
    return str(v).strip().lower() in {"yes", "y", "true", "1"}


def find_col(df, *keywords):
    """First column whose lowercase name contains any keyword."""
    for c in df.columns:
        if any(k in c.lower() for k in keywords):
            return c
    return None


def severity_levels(sev_master, ids_in_data):
    """Map severity_id -> 0..1 (0 = mildest). Uses severity_name words, else ID order."""
    order = {"mild": 0, "moderate": 1, "severe": 2, "critical": 3}
    level = {}
    if sev_master is not None and "severity_name" in sev_master:
        for sid, name in zip(sev_master["severity_id"], sev_master["severity_name"].astype(str).str.lower()):
            for word, lv in order.items():
                if word in name:
                    level[sid] = lv
    missing = sorted(set(ids_in_data) - set(level))
    if missing:                                   # fall back to sorted ID order
        all_ids = sorted(set(ids_in_data) | set(level))
        level = {sid: i for i, sid in enumerate(all_ids)}
    top = max(level.values()) or 1
    return {sid: lv / top for sid, lv in level.items()}, {sid: lv for sid, lv in level.items()}


# ---------------------------------------------------------------- patient risk factors (each 0..1)
def risk_factors(p):
    temp_c = find_col(p, "temp")
    hr_c = find_col(p, "heart", "pulse", "hr")
    spo2_c = find_col(p, "spo2", "oxygen", "saturation")
    sbp_c = find_col(p, "systolic", "sbp")
    print("Vitals used ->", {"temp": temp_c, "heart_rate": hr_c, "spo2": spo2_c, "systolic": sbp_c})

    z = pd.Series(0.0, index=p.index)
    f = pd.DataFrame(index=p.index)
    if temp_c:
        t = p[temp_c].astype(float)
        if t.median() > 50:                       # Fahrenheit -> Celsius
            t = (t - 32) * 5 / 9
        f["fever"] = ((t - 38.5) / 1.5).clip(0, 1)
    else:
        f["fever"] = z
    f["tachy"] = ((p[hr_c] - 100) / 30).clip(0, 1) if hr_c else z
    f["hypoxia"] = ((95 - p[spo2_c]) / 10).clip(0, 1) if spo2_c else z
    f["hypotension"] = ((100 - p[sbp_c]) / 20).clip(0, 1) if sbp_c else z
    age = p["age"].astype(float)
    f["age_risk"] = ((age - 60) / 25).clip(0, 1) + ((5 - age) / 5).clip(0, 1)
    return f


# ---------------------------------------------------------------- outcome model (the "truth")
def success_prob(row, rf, sev, tx):
    """True P(success). Patient risk lowers it; a treatment that fits the patient raises it."""
    base = 0.82 if yes(tx["first_line"]) else 0.68
    base -= 0.03 * (max(int(tx["priority"]), 1) - 1)
    z = np.log(base / (1 - base))

    # patient risk
    z -= 1.3 * sev + 0.7 * rf["age_risk"] + 0.9 * rf["hypoxia"] + 0.4 * rf["fever"] \
         + 0.4 * rf["tachy"] + 0.5 * rf["hypotension"]

    # treatment-patient fit
    ttype = str(tx.get("treatment_type", "")).lower()
    inpatient = yes(tx.get("inpatient_required", "No"))
    if "respiratory" in ttype or "oxygen" in str(tx.get("treatment_name", "")).lower():
        z += 2.0 * rf["hypoxia"] - 0.3 * (1 - rf["hypoxia"])     # great if hypoxic, mostly useless if not
    if "fluid" in ttype or "rehydration" in ttype:
        z += 1.0 * rf["tachy"] + 0.9 * rf["hypotension"]
    if "antipyretic" in ttype:
        z += 0.5 * rf["fever"]
    z += (1.0 * sev) if inpatient else (-0.7 * sev)               # severe cases do better admitted
    if inpatient and sev == 0:
        z -= 0.2                                                  # over-treating mild cases: small penalty

    return float(1 / (1 + np.exp(-z)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default=str(OUTPUT_DIR / "synthetic_patients.csv"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "patient_treatments.csv"))
    ap.add_argument("--explore", type=float, default=EXPLORE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    patients = pd.read_csv(args.patients)
    mapping = load("disease_treatment_mapping")
    tmaster = load("treatment_master")
    try:
        smaster = load("severity_master")
    except FileNotFoundError:
        smaster = None
        print("severity_master not found - ordering severities by ID")

    mapping["priority"] = pd.to_numeric(mapping["priority"], errors="coerce").fillna(5).astype(int)
    mapping = mapping.merge(tmaster, on="treatment_id", how="left")
    unknown_tx = mapping.loc[mapping["treatment_name"].isna(), "treatment_id"].unique()
    if len(unknown_tx):
        print("WARNING: treatment IDs in mapping but not in treatment_master:", list(unknown_tx))

    sev01, sev_lvl = severity_levels(smaster, patients["severity_id"].unique())
    rf_all = risk_factors(patients)

    by_combo = {k: g for k, g in mapping.groupby(["disease_id", "severity_id"])}
    by_disease = {k: g for k, g in mapping.groupby("disease_id")}

    rows, fallback, skipped = [], set(), set()
    for i, pt in patients.iterrows():
        d, s = pt["disease_id"], pt["severity_id"]
        cands = by_combo.get((d, s))
        if cands is None:
            cands = by_disease.get(d)
            if cands is None:
                skipped.add(d)
                continue
            cands = cands.drop_duplicates("treatment_id")
            fallback.add((d, s))

        # doctor's choice: mostly guideline-driven, sometimes not
        if rng.random() < args.explore:
            w = np.ones(len(cands))
        else:
            w = (1 / cands["priority"].clip(lower=1)).to_numpy() * \
                np.where(cands["first_line"].map(yes), 2.0, 1.0)
        tx = cands.iloc[rng.choice(len(cands), p=w / w.sum())]

        sev = sev01.get(s, 0.5)
        p = success_prob(pt, rf_all.loc[i], sev, tx)
        ok = int(rng.random() < p)
        days = BASE_DAYS[min(sev_lvl.get(s, 1), len(BASE_DAYS) - 1)] * (1 + 0.6 * (1 - p))
        days *= rng.lognormal(0, 0.2) * (1.0 if ok else 1.5)

        rows.append({
            "patient_token": pt["patient_token"], "disease_id": d, "severity_id": s,
            "treatment_id": tx["treatment_id"],
            "treatment_name": tx.get("treatment_name"),
            "treatment_category": tx.get("treatment_category"),
            "treatment_type": tx.get("treatment_type"),
            "route": tx.get("route"),
            "inpatient_required": int(yes(tx.get("inpatient_required", "No"))),
            "priority": int(tx["priority"]),
            "first_line": int(yes(tx["first_line"])),
            "success": ok,
            "recovery_days": int(round(max(days, 1))), "_oracle_success_prob": round(p, 4),
        })

    # one self-contained file: patient features + treatment + outcome
    out = pd.DataFrame(rows)
    patient_feats = patients.drop(columns=["disease_id", "severity_id"], errors="ignore")
    out = out.merge(patient_feats, on="patient_token", how="left")
    first = ["patient_token", "hospital_id", "disease_id", "severity_id"]
    labels = ["success", "recovery_days", "_oracle_success_prob"]
    out = out[[c for c in first if c in out] + [c for c in out if c not in first + labels] + labels]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # ---- summary ----
    print(f"\nWrote {len(out):,} rows -> {args.out}")
    if skipped:
        print(f"WARNING: no treatments mapped for diseases {sorted(skipped)} - those patients were skipped")
    if fallback:
        print(f"Note: {len(fallback)} disease/severity combos had no mapping rows; used the disease's "
              f"treatments for any severity: {sorted(fallback)[:10]}{' ...' if len(fallback) > 10 else ''}")
    print(f"\nOverall success rate: {out['success'].mean():.1%}")
    gaps = out.isna().sum()
    if gaps.any():
        print(f"WARNING: empty values in output: {gaps[gaps > 0].to_dict()}")
    summary = (out.groupby(["treatment_id", "treatment_name"], dropna=False)
                  .agg(patients=("success", "size"), success_rate=("success", "mean"),
                       median_days=("recovery_days", "median")))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()