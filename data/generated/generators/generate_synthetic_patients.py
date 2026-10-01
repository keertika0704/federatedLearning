import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent      # data/generated/generators
DATA_ROOT = SCRIPT_DIR.parents[1]                 # data
INPUT_DIRS = [DATA_ROOT / "config", DATA_ROOT / "reference"]
OUTPUT_DIR = DATA_ROOT / "generated" / "generated_data"
 
# ---- generation settings ----
NOISE_SYMPTOM_P = 0.03   # chance of an unrelated symptom showing up (real-world noise)
VITAL_WIDEN = 0.15       # vitals can fall 15% outside the configured range
PREF_AGE_SHARE = 0.75    # share of patients inside the disease's preferred age range
PREF_GENDER_SHARE = 0.75 # share of patients matching the disease's preferred gender
MANDATORY_P = 0.97       # probability a "mandatory" symptom is present
 
FREQ_P = {
    "very common": 0.85, "common": 0.60, "frequent": 0.60, "occasional": 0.35,
    "uncommon": 0.20, "rare": 0.10, "very rare": 0.05,
}
DEFAULT_FREQ_P = 0.30
SEV_IDS = {"mild_percentage": "SV001", "moderate_percentage": "SV002", "severe_percentage": "SV003"}
VITALS = ["temperature", "heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp"]
 
 
def load(name):
    """Find <name>.csv (or '<name> copy.csv') anywhere under data/config or data/reference."""
    paths = sorted(p for d in INPUT_DIRS for p in d.rglob(f"{name}*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing {name}.csv under {[str(d) for d in INPUT_DIRS]}")
    print(f"  loaded {paths[0].relative_to(DATA_ROOT)}")
    return pd.read_csv(paths[0])
 
 
def build_symptom_profiles(dsm, sym_idx):
    """disease_id -> list of (symptom column index, probability)."""
    unknown = set(dsm["frequency"].astype(str).str.strip().str.lower()) - set(FREQ_P)
    if unknown:
        print(f"WARNING: unknown frequency labels, using {DEFAULT_FREQ_P} for: {unknown}")
 
    profiles = {}
    for disease, group in dsm.groupby("disease_id"):
        prof = []
        for sid, freq, mand in group[["symptom_id", "frequency", "mandatory"]].astype(str).itertuples(index=False):
            if sid not in sym_idx:
                print(f"WARNING: {sid} in mapping but not in symptom_master, skipping")
                continue
            if mand.strip().lower() == "yes":
                p = MANDATORY_P
            else:
                p = FREQ_P.get(freq.strip().lower(), DEFAULT_FREQ_P)
            prof.append((sym_idx[sid], p))
        profiles[disease] = prof
    return profiles
 
 
def sample_vitals(rng, vit, disease, severity):
    row = vit.get((disease, severity))
    if row is None:  # fall back to any severity row for this disease
        row = next(v for (d, _), v in vit.items() if d == disease)
    out = {}
    for v in VITALS:
        lo, hi = getattr(row, f"{v}_min"), getattr(row, f"{v}_max")
        pad = (hi - lo) * VITAL_WIDEN
        x = rng.uniform(lo - pad, hi + pad)
        out[v] = round(x, 1) if v == "temperature" else int(round(x))
    out["spo2"] = min(out["spo2"], 100)
    return out
 
 
def sample_gender(rng, preferred):
    pg = str(preferred).strip().lower()
    if pg.startswith("m"):
        return "Male" if rng.random() < PREF_GENDER_SHARE else "Female"
    if pg.startswith("f"):
        return "Female" if rng.random() < PREF_GENDER_SHARE else "Male"
    return "Male" if rng.random() < 0.5 else "Female"
 
 
def generate(n_patients, seed):
    rng = np.random.default_rng(seed)
 
    gen_cfg = load("disease_generation_config")
    dsm = load("disease_symptom_mapping")
    vitals_cfg = load("vital_range_config")
    sym_master = load("symptom_master")
    hosp_cfg = load("hospital_generation_config")
 
    all_symptoms = sym_master["symptom_id"].astype(str).tolist()
    sym_idx = {s: i for i, s in enumerate(all_symptoms)}
    profiles = build_symptom_profiles(dsm, sym_idx)
 
    vit = {(r.disease_id, r.severity_id): r for r in vitals_cfg.itertuples(index=False)}
    cfg = gen_cfg.set_index("disease_id")
 
    diseases = [d for d in cfg.index if d in profiles]
    missing = set(cfg.index) - set(diseases)
    if missing:
        print(f"WARNING: no symptom mapping for {missing}, these diseases are skipped")
    dis_w = cfg.loc[diseases, "prevalence_weight"].astype(float).values
    dis_w = dis_w / dis_w.sum()
 
    hosp_ids = hosp_cfg["hospital_id"].astype(str).tolist()
    hosp_w = hosp_cfg["expected_daily_patients"].astype(float).values
    hosp_w = hosp_w / hosp_w.sum()
 
    rows = []
    for i in range(n_patients):
        disease = rng.choice(diseases, p=dis_w)
        c = cfg.loc[disease]
 
        sev_p = np.array([c[k] for k in SEV_IDS], dtype=float)
        severity = rng.choice(list(SEV_IDS.values()), p=sev_p / sev_p.sum())
 
        if rng.random() < PREF_AGE_SHARE:
            age = int(rng.integers(c["preferred_age_min"], c["preferred_age_max"] + 1))
        else:
            age = int(rng.integers(1, 91))
 
        symptoms = (rng.random(len(all_symptoms)) < NOISE_SYMPTOM_P).astype(int)
        for j, p in profiles[disease]:
            if rng.random() < p:
                symptoms[j] = 1
        if symptoms.sum() == 0:  # every patient presents with at least one symptom
            symptoms[max(profiles[disease], key=lambda t: t[1])[0]] = 1
 
        row = {
            "patient_token": f"P{i:06d}",
            "hospital_id": str(rng.choice(hosp_ids, p=hosp_w)),
            "age": age,
            "gender": sample_gender(rng, c["preferred_gender"]),
            **sample_vitals(rng, vit, disease, severity),
        }
        row.update(dict(zip(all_symptoms, symptoms.tolist())))
        row["severity_id"] = severity   # not a model feature (post-diagnosis)
        row["disease_id"] = disease     # target
        rows.append(row)
 
    return pd.DataFrame(rows)
 
 
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patients from reference tables")
    parser.add_argument("--n", type=int, default=20000, help="number of patients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="synthetic_patients.csv", help="file name inside generated_data/")
    args = parser.parse_args()
 
    print("Reading reference tables:")
    df = generate(args.n, args.seed)
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / args.out
    df.to_csv(out_path, index=False)
 
    print(f"\nSaved {out_path.relative_to(DATA_ROOT.parent)}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nPatients per disease:")
    print(df["disease_id"].value_counts().to_string())
    print("\nPatients per hospital:")
    print(df["hospital_id"].value_counts().sort_index().to_string())
 
 
if __name__ == "__main__":
    main()
 
