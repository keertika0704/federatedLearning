"""
One table for every model: encounters.csv

Each row is one hospital encounter. The table carries everything needed for
    - disease detection            (features: demographics, vitals, symptoms, labs, imaging  ->  disease_id)
    - treatment recommendation     (features + disease/severity  ->  treatment_id, success, recovery_days)
    - atypical symptom detection   (objective A: is_atypical / atypical_type)
    - pandemic / outbreak detection(objective B: encounter_date + hospital_id + disease_id, label outbreak_active)
    - federated learning           (hospital_id / node_name = the client; hospitals differ in climate and
                                    disease mix, so the data is non-IID across nodes)

How it works
    1. Outbreaks: a daily simulation decides, per (hospital, disease), when an outbreak episode starts,
       how long it lasts and how strongly it multiplies cases. Episodes can spill over to other hospitals
       in the same region a few days later.
    2. Volume and disease mix: per hospital per day, cases of each disease ~ Poisson(
       hospital daily volume * scale * disease prevalence * seasonality * climate factor * outbreak factor).
    3. Each encounter reuses the logic of the two existing generators (generate_synthetic_patients.py and
       generate_treatments.py): age/gender/vitals/symptoms, then treatment choice + outcome.
    4. On top: onset days, lab results, imaging, complications, atypical presentations, admission/ICU/death.

Reads  (data/config and data/reference)
    disease_generation_config, hospital_generation_config, vital_range_config,
    disease_{symptom,lab,imaging,complication,treatment}_mapping,
    {symptom,lab,imaging,severity,treatment,hospital,disease,complication}_master

Writes (data/generated/generated_data)
    encounters.csv                   the table
    encounters_data_dictionary.csv   every column with its group and role (feature / target / leakage warning)

Leakage guide (details in the dictionary)
    - Disease detection: do NOT use severity, treatment, outcome, complication, outbreak or atypical columns.
    - Treatment model: features may include disease_id and severity_id; never `_oracle_success_prob`, outcome columns.
    - Split by time or by hospital for objectives B / federated experiments, not randomly.

Run
    python data/generated/generators/generate_encounters.py
    python data/generated/generators/generate_encounters.py --days 365 --scale 0.1 --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent      # data/generated/generators
sys.path.insert(0, str(SCRIPT_DIR))
import generate_synthetic_patients as gp          # noqa: E402  (symptom profiles, vitals, gender, loader)
import generate_treatments as gt                  # noqa: E402  (risk factors, success model)

DATA_ROOT, OUTPUT_DIR = gp.DATA_ROOT, gp.OUTPUT_DIR
load = gp.load

# ---- generation settings ----
PEAK_BOOST = 2.0              # in-season case multiplier (scaled by hospital seasonal_variation_factor)
OUTBREAK_MIN_DAYS, OUTBREAK_MAX_DAYS = 14, 35
OUTBREAK_ACTIVE_AT = 1.3      # outbreak_active = 1 when the episode multiplier is at least this
SPILLOVER_P = 0.5             # chance an outbreak reaches each other hospital in the same region
CLIMATE_FACTOR = {            # disease_category -> climate -> relative case rate (makes hospitals non-IID)
    "Vector-borne": {"Tropical": 1.5, "Coastal": 1.5, "Continental": 0.6, "Semi-arid": 0.8},
    "Respiratory": {"Continental": 1.3, "Semi-arid": 1.1, "Tropical": 0.9, "Coastal": 0.9},
    "Food & Water-borne": {"Tropical": 1.2, "Coastal": 1.2, "Continental": 0.9, "Semi-arid": 1.0},
}
ATYPICAL_P = 0.06             # base share of atypical presentations (doubled for children <5 and adults >65)
ATYPICAL_TYPES = ["missing_hallmark", "extra_symptoms", "vitals_discordant"]
LOW_WHEN_ABNORMAL = {"L001", "L003", "L009"}           # labs that go low (others go high)
LOW_WBC_DISEASES = {"D001", "D003"}                    # leukopenia in dengue / chikungunya
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"])}


def snake(s):
    return "".join(c.lower() if c.isalnum() else "_" for c in str(s)).strip("_")


def in_season(month, start, end):
    return start <= month <= end if start <= end else (month >= start or month <= end)


# ---------------------------------------------------------------- step 1: outbreaks
def simulate_outbreaks(rng, dates, hosp_ids, hosp_region, hosp_mult, dis_ids, ob_p, peak):
    """Returns multiplier[day, hosp, disease] and episode_id[day, hosp, disease] (0 = no episode)."""
    nd, H, K = len(dates), len(hosp_ids), len(dis_ids)
    mult = np.ones((nd, H, K))
    ep_id = np.zeros((nd, H, K), dtype=int)
    counter = [0]
    months = dates.month.to_numpy()

    def launch(t, h, k, amp, spill):
        dur = int(rng.integers(OUTBREAK_MIN_DAYS, OUTBREAK_MAX_DAYS + 1))
        end = min(t + dur, nd)
        if ep_id[t:end, h, k].any():
            return
        counter[0] += 1
        bump = np.sin(np.pi * (np.arange(end - t) + 0.5) / dur)
        mult[t:end, h, k] = 1 + (amp - 1) * bump
        ep_id[t:end, h, k] = counter[0]
        if spill:
            for h2 in range(H):
                if h2 != h and hosp_region[h2] == hosp_region[h] and rng.random() < SPILLOVER_P:
                    lag = int(rng.integers(3, 11))
                    if t + lag < nd:
                        launch(t + lag, h2, k, 1 + (hosp_mult[h2] - 1) * rng.uniform(0.5, 0.9), False)

    for t in range(nd):
        for k in range(K):
            rate = ob_p[k] / (90 if in_season(months[t], *peak[k]) else 400)
            for h in range(H):
                if ep_id[t, h, k] == 0 and rng.random() < rate:
                    launch(t, h, k, 1 + (hosp_mult[h] - 1) * rng.uniform(0.6, 1.0), True)
    return mult, ep_id


# ---------------------------------------------------------------- step 2: clinical details
def build_lab_profiles(dlm, lab_master):
    """disease_id -> list of (lab_id, p_ordered, p_abnormal)."""
    prof = {}
    for r in dlm.itertuples(index=False):
        p_order = gp.FREQ_P.get(str(r.frequency).strip().lower(), gp.DEFAULT_FREQ_P)
        prof.setdefault(r.disease_id, []).append((r.lab_id, p_order, float(r.abnormal_probability)))
    return prof


def sample_lab(rng, lab, disease, abnormal, sev01):
    if lab.result_type == "Binary":
        return int(abnormal)
    lo, hi = float(lab.reference_range_low), float(lab.reference_range_high)
    span = hi - lo
    if not abnormal:
        return round(rng.uniform(lo + 0.1 * span, hi - 0.1 * span), 2)
    low = lab.lab_id in LOW_WHEN_ABNORMAL or (lab.lab_id == "L002" and disease in LOW_WBC_DISEASES)
    if low and lo > 0:
        return round(max(lo - span * rng.uniform(0.1, 0.6), 0.1 * lo), 2)
    return round(hi + span * rng.uniform(0.1, 1.0 + 1.5 * sev01), 2)


def make_atypical(rng, symptoms, disease, mandatory, profile_idx, n_sym, vit, severity, vitals):
    """Distort a typical presentation. Returns (symptoms, vitals, atypical_type)."""
    kind = str(rng.choice(ATYPICAL_TYPES))
    if kind == "missing_hallmark":
        for j in mandatory[disease]:
            symptoms[j] = 0
    elif kind == "extra_symptoms":
        outside = [j for j in range(n_sym) if j not in profile_idx[disease]]
        for j in rng.choice(outside, size=int(rng.integers(2, 5)), replace=False):
            symptoms[j] = 1
    else:  # vitals look like a different disease
        other = str(rng.choice([d for d in mandatory if d != disease]))
        vitals = gp.sample_vitals(rng, vit, other, severity)
    if symptoms.sum() == 0:
        symptoms[int(rng.choice(sorted(profile_idx[disease])))] = 1
    return symptoms, vitals, kind


# ---------------------------------------------------------------- data dictionary
def describe_columns(cols, symptom_names):
    rows = []
    for c in cols:
        if c in {"encounter_id", "patient_token"}:
            g, role, d = "identity", "id", "row / patient identifier"
        elif c in {"encounter_date", "year", "month", "day_of_week", "week_of_year"}:
            g, role, d = "time", "feature", "encounter date and calendar parts (objective B: use for time series / splits)"
        elif c in {"hospital_id", "hospital_name", "city", "state", "hospital_type", "level", "beds",
                   "node_name", "region", "climate", "urban_rural", "catchment_population"}:
            g, role, d = "hospital", "feature", "hospital attributes; hospital_id / node_name = federated client"
        elif c == "daily_encounters_at_hospital":
            g, role, d = "hospital", "feature", "encounters at this hospital on this day (load signal for objective B)"
        elif c in {"age", "gender"}:
            g, role, d = "demographics", "feature", c
        elif c in gp.VITALS:
            g, role, d = "vitals", "feature", c
        elif c == "days_since_onset":
            g, role, d = "history", "feature", "days between symptom onset and the encounter"
        elif c in symptom_names:
            g, role, d = "symptoms", "feature", f"1 = {symptom_names[c]} present"
        elif c.startswith("lab_"):
            g, role, d = "labs", "feature", "result, or empty if not ordered (binary tests: 1 = positive)"
        elif c.startswith("img_"):
            g, role, d = "imaging", "feature", "1 = abnormal, 0 = normal, empty if not performed"
        elif c == "disease_id":
            g, role, d = "diagnosis", "target", "target of disease detection"
        elif c in {"disease_name", "disease_category", "communicable"}:
            g, role, d = "diagnosis", "leakage_for_disease_model", "label description"
        elif c in {"severity_id", "severity_name"}:
            g, role, d = "severity", "post_diagnosis", "severity (feature for treatment model, not for disease detection)"
        elif c in {"has_complication", "n_complications", "complication_ids"}:
            g, role, d = "complications", "post_diagnosis", "complications for this encounter"
        elif c.startswith("treatment_") or c in {"route", "inpatient_required", "priority", "first_line"}:
            g, role, d = "treatment", "target_treatment", "treatment given (target of treatment recommender)"
        elif c in {"success", "recovery_days"}:
            g, role, d = "outcome", "target_treatment", "treatment outcome targets"
        elif c in {"admitted", "icu", "length_of_stay_days", "died"}:
            g, role, d = "outcome", "post_treatment", "hospital course / outcome"
        elif c in {"is_atypical", "atypical_type"}:
            g, role, d = "objective_A", "target_A", "atypical presentation label (none / missing_hallmark / extra_symptoms / vitals_discordant)"
        elif c in {"outbreak_active", "outbreak_id"}:
            g, role, d = "objective_B", "target_B", "outbreak episode label for this hospital + disease + day"
        elif c == "_oracle_success_prob":
            g, role, d = "oracle", "evaluation_only", "true simulated success probability; never a feature"
        else:
            g, role, d = "other", "feature", c
        rows.append({"column": c, "group": g, "role": role, "description": d})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generate the unified encounters table")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--scale", type=float, default=0.05, help="fraction of expected_daily_patients to generate")
    ap.add_argument("--explore", type=float, default=gt.EXPLORE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="encounters.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Reading reference tables:")
    gen_cfg, hosp_cfg = load("disease_generation_config"), load("hospital_generation_config")
    vitals_cfg, dsm = load("vital_range_config"), load("disease_symptom_mapping")
    dlm, dim_ = load("disease_lab_mapping"), load("disease_imaging_mapping")
    dcm, dtm = load("disease_complication_mapping"), load("disease_treatment_mapping")
    sym_m, lab_m, img_m = load("symptom_master"), load("lab_master"), load("imaging_master")
    sev_m, trt_m, hosp_m = load("severity_master"), load("treatment_master"), load("hospital_master")
    dis_m, comp_m = load("disease_master"), load("complication_master")
    for df in (dtm, trt_m):
        df.columns = [c.strip() for c in df.columns]

    # ---- lookups
    symptoms = sym_m["symptom_id"].astype(str).tolist()
    sym_idx = {s: i for i, s in enumerate(symptoms)}
    profiles = gp.build_symptom_profiles(dsm, sym_idx)
    profile_idx = {d: {j for j, _ in p} for d, p in profiles.items()}
    mandatory = {d: [sym_idx[s] for s in g.loc[g["mandatory"].astype(str).str.lower() == "yes", "symptom_id"]
                     if s in sym_idx] for d, g in dsm.groupby("disease_id")}
    vit = {(r.disease_id, r.severity_id): r for r in vitals_cfg.itertuples(index=False)}
    cfg = gen_cfg.set_index("disease_id")
    dis_ids = [d for d in cfg.index if d in profiles]
    dis_info = dis_m.set_index("disease_id")
    labs = lab_m.set_index("lab_id")
    lab_cols = {l: "lab_" + snake(labs.loc[l, "test_code"] if pd.notna(labs.loc[l, "test_code"]) else "NA")  # sodium's code "NA" reads as NaN
                for l in labs.index if l != "L014"}  # L014 = SpO2 (vital)
    lab_rows = {r.lab_id: r for r in lab_m.itertuples(index=False)}
    lab_profiles = build_lab_profiles(dlm, lab_m)
    imgs = img_m.set_index("imaging_id")
    img_cols = {i: f"img_{snake(imgs.loc[i, 'imaging_name'])}" for i in imgs.index}
    img_profiles = {}
    for r in dim_.itertuples(index=False):
        p = gp.FREQ_P.get(str(r.frequency).strip().lower(), gp.DEFAULT_FREQ_P)
        img_profiles.setdefault(r.disease_id, []).append((r.imaging_id, p, float(r.abnormal_probability)))
    comp_profiles = {}
    for r in dcm.itertuples(index=False):
        p = 0.5 * gp.FREQ_P.get(str(r.frequency).strip().lower(), gp.DEFAULT_FREQ_P)
        comp_profiles.setdefault(r.disease_id, []).append((r.complication_id, r.severity_id, p))

    sev_ids = list(gp.SEV_IDS.values())
    sev01, sev_lvl = gt.severity_levels(sev_m, sev_ids)
    sev_name = sev_m.set_index("severity_id")["severity_name"].to_dict()

    dtm["priority"] = pd.to_numeric(dtm["priority"], errors="coerce").fillna(5).astype(int)
    dtm = dtm.merge(trt_m, on="treatment_id", how="left")
    by_combo = {k: g for k, g in dtm.groupby(["disease_id", "severity_id"])}
    by_disease = {k: g for k, g in dtm.groupby("disease_id")}

    hosp_ids = hosp_cfg["hospital_id"].astype(str).tolist()
    hosp_region = hosp_cfg["region"].tolist()
    hosp_climate = hosp_cfg["climate"].tolist()
    hosp_svf = hosp_cfg["seasonal_variation_factor"].astype(float).to_numpy()
    hosp_ob = hosp_cfg["outbreak_multiplier"].astype(float).to_numpy()
    hosp_vol = hosp_cfg["expected_daily_patients"].astype(float).to_numpy()
    H, K = len(hosp_ids), len(dis_ids)

    share = cfg.loc[dis_ids, "prevalence_weight"].astype(float).to_numpy()
    share = share / share.sum()
    ob_p = cfg.loc[dis_ids, "outbreak_probability"].astype(float).to_numpy()
    peak = [(MONTHS[str(cfg.loc[d, "peak_start_month"]).strip().lower()],
             MONTHS[str(cfg.loc[d, "peak_end_month"]).strip().lower()]) for d in dis_ids]

    # ---- step 1: calendar, seasonality, outbreaks, daily case counts
    dates = pd.date_range(args.start, periods=args.days, freq="D")
    months = dates.month.to_numpy()
    season = np.ones((len(dates), H, K))
    for k in range(K):
        flag = np.array([in_season(m, *peak[k]) for m in months])
        season[:, :, k] = np.where(flag[:, None], 1 + (PEAK_BOOST - 1) * hosp_svf[None, :], 1.0)
    climate = np.array([[CLIMATE_FACTOR.get(dis_info.loc[d, "disease_category"], {}).get(hosp_climate[h], 1.0)
                         for d in dis_ids] for h in range(H)])
    mult, ep_id = simulate_outbreaks(rng, dates, hosp_ids, hosp_region, hosp_ob, dis_ids, ob_p, peak)
    lam = (hosp_vol[None, :, None] * args.scale * share[None, None, :]) * season * climate[None] * mult
    counts = rng.poisson(lam)
    print(f"\nOutbreak episodes simulated: {ep_id.max()}, encounters to generate: {counts.sum():,}")

    # ---- step 2: build encounters (patient, labs, imaging, complications, atypical flag)
    rows = []
    for t, h, k in zip(*np.nonzero(counts)):
        disease = dis_ids[k]
        c = cfg.loc[disease]
        for _ in range(counts[t, h, k]):
            sev_p = np.array([c[x] for x in gp.SEV_IDS], dtype=float)
            severity = str(rng.choice(sev_ids, p=sev_p / sev_p.sum()))
            lvl = sev01[severity]
            age = int(rng.integers(c["preferred_age_min"], c["preferred_age_max"] + 1)) \
                if rng.random() < gp.PREF_AGE_SHARE else int(rng.integers(1, 91))

            sym = (rng.random(len(symptoms)) < gp.NOISE_SYMPTOM_P).astype(int)
            for j, p in profiles[disease]:
                if rng.random() < p:
                    sym[j] = 1
            if sym.sum() == 0:
                sym[max(profiles[disease], key=lambda x: x[1])[0]] = 1
            vitals = gp.sample_vitals(rng, vit, disease, severity)

            atyp_p = ATYPICAL_P * (2 if age < 5 or age > 65 else 1)
            atypical = "none"
            if rng.random() < atyp_p:
                sym, vitals, atypical = make_atypical(rng, sym, disease, mandatory, profile_idx,
                                                      len(symptoms), vit, severity, vitals)

            row = {
                "encounter_date": dates[t].date().isoformat(),
                "hospital_id": hosp_ids[h], "age": age,
                "gender": gp.sample_gender(rng, c["preferred_gender"]),
                **vitals,
                "days_since_onset": int(np.clip(1 + rng.poisson(2 + 1.5 * sev_lvl[severity]), 1, 21)),
            }
            row.update(dict(zip(symptoms, sym.tolist())))

            for lab_id, p_order, p_abn in lab_profiles.get(disease, []):
                if lab_id in lab_cols and rng.random() < p_order:
                    abn = rng.random() < min(0.98, p_abn * (1 + 0.35 * lvl))
                    row[lab_cols[lab_id]] = sample_lab(rng, lab_rows[lab_id], disease, abn, lvl)
            for img_id, p_order, p_abn in img_profiles.get(disease, []):
                if rng.random() < min(0.95, p_order + 0.25 * lvl):
                    row[img_cols[img_id]] = int(rng.random() < min(0.98, p_abn * (1 + 0.4 * lvl)))

            comps = [cid for cid, csev, p in comp_profiles.get(disease, [])
                     if sev_lvl[severity] >= sev_lvl[csev] and rng.random() < p]
            row["n_complications"] = len(comps)
            row["has_complication"] = int(bool(comps))
            row["complication_ids"] = "|".join(comps)

            row.update({"disease_id": disease, "severity_id": severity,
                        "severity_name": sev_name.get(severity, severity),
                        "is_atypical": int(atypical != "none"), "atypical_type": atypical,
                        "outbreak_active": int(mult[t, h, k] >= OUTBREAK_ACTIVE_AT),
                        "outbreak_id": int(ep_id[t, h, k]) if mult[t, h, k] >= OUTBREAK_ACTIVE_AT else 0})
            rows.append(row)

    df = pd.DataFrame(rows)
    n = len(df)
    df.insert(0, "patient_token", [f"P{i:06d}" for i in range(n)])
    df.insert(0, "encounter_id", [f"E{i:06d}" for i in range(n)])

    # ---- step 3: treatment + outcome (same simulation as generate_treatments.py)
    rf = gt.risk_factors(df)
    tx_cols = {k: [] for k in ["treatment_id", "treatment_name", "treatment_category", "treatment_type", "route",
                               "inpatient_required", "priority", "first_line", "success", "recovery_days",
                               "_oracle_success_prob", "admitted", "icu", "length_of_stay_days", "died"]}
    for i, (d, s) in enumerate(zip(df["disease_id"], df["severity_id"])):
        cands = by_combo.get((d, s))
        if cands is None:
            cands = by_disease[d].drop_duplicates("treatment_id")
        if rng.random() < args.explore:
            w = np.ones(len(cands))
        else:
            w = (1 / cands["priority"].clip(lower=1)).to_numpy() * np.where(cands["first_line"].map(gt.yes), 2.0, 1.0)
        tx = cands.iloc[rng.choice(len(cands), p=w / w.sum())]
        sev = sev01[s]
        p = gt.success_prob(None, rf.iloc[i], sev, tx)
        ok = int(rng.random() < p)
        days = gt.BASE_DAYS[min(sev_lvl[s], len(gt.BASE_DAYS) - 1)] * (1 + 0.6 * (1 - p))
        days = int(round(max(days * rng.lognormal(0, 0.2) * (1.0 if ok else 1.5), 1)))

        inpatient = gt.yes(tx.get("inpatient_required", "No"))
        admitted = int(inpatient or s == "SV003" or (s == "SV002" and rng.random() < 0.3))
        icu = int(s == "SV003" and rng.random() < np.clip(0.25 + 0.5 * rf.iloc[i]["hypoxia"], 0, 0.9))
        died = int(not ok and ((icu and rng.random() < 0.35) or (s == "SV003" and rng.random() < 0.08)))
        for key, val in [("treatment_id", tx["treatment_id"]), ("treatment_name", tx.get("treatment_name")),
                         ("treatment_category", tx.get("treatment_category")),
                         ("treatment_type", tx.get("treatment_type")), ("route", tx.get("route")),
                         ("inpatient_required", int(inpatient)), ("priority", int(tx["priority"])),
                         ("first_line", int(gt.yes(tx["first_line"]))), ("success", ok),
                         ("recovery_days", days), ("_oracle_success_prob", round(p, 4)),
                         ("admitted", admitted), ("icu", icu),
                         ("length_of_stay_days", max(1, round(days * 0.6)) if admitted else 0), ("died", died)]:
            tx_cols[key].append(val)
    for key, vals in tx_cols.items():
        df[key] = vals

    # ---- step 4: hospital + calendar + disease columns
    hosp_info = hosp_m.merge(hosp_cfg[["hospital_id", "region", "climate", "urban_rural", "catchment_population"]],
                             on="hospital_id", how="left")
    df = df.merge(hosp_info, on="hospital_id", how="left")
    d_ts = pd.to_datetime(df["encounter_date"])
    df["year"], df["month"], df["day_of_week"] = d_ts.dt.year, d_ts.dt.month, d_ts.dt.dayofweek
    df["week_of_year"] = d_ts.dt.isocalendar().week.astype(int)
    df["daily_encounters_at_hospital"] = df.groupby(["hospital_id", "encounter_date"])["encounter_id"].transform("size")
    df["disease_name"] = df["disease_id"].map(dis_info["disease_name"])
    df["disease_category"] = df["disease_id"].map(dis_info["disease_category"])
    df["communicable"] = (df["disease_id"].map(dis_info["communicable"]).astype(str).str.lower() == "yes").astype(int)

    order = (["encounter_id", "patient_token", "encounter_date", "year", "month", "day_of_week", "week_of_year",
              "hospital_id", "hospital_name", "city", "state", "hospital_type", "level", "beds", "node_name",
              "region", "climate", "urban_rural", "catchment_population", "daily_encounters_at_hospital",
              "age", "gender"] + gp.VITALS + ["days_since_onset"] + symptoms
             + sorted(c for c in df if c.startswith("lab_")) + sorted(c for c in df if c.startswith("img_"))
             + ["disease_id", "disease_name", "disease_category", "communicable",
                "severity_id", "severity_name", "has_complication", "n_complications", "complication_ids",
                "treatment_id", "treatment_name", "treatment_category", "treatment_type", "route",
                "inpatient_required", "priority", "first_line", "success", "recovery_days",
                "admitted", "icu", "length_of_stay_days", "died",
                "is_atypical", "atypical_type", "outbreak_active", "outbreak_id", "_oracle_success_prob"])
    df = df[order]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / args.out
    df.to_csv(out_path, index=False)
    dd_path = out_path.with_name(out_path.stem + "_data_dictionary.csv")
    describe_columns(list(df.columns), dict(zip(sym_m["symptom_id"], sym_m["symptom_name"]))).to_csv(dd_path, index=False)

    # ---- summary
    print(f"\nSaved {out_path.relative_to(DATA_ROOT.parent)}: {len(df):,} rows x {df.shape[1]} columns")
    print(f"Saved {dd_path.relative_to(DATA_ROOT.parent)}")
    print(f"\nDate range: {df['encounter_date'].min()} -> {df['encounter_date'].max()}")
    print(f"Success rate: {df['success'].mean():.1%} | admitted: {df['admitted'].mean():.1%} | "
          f"ICU: {df['icu'].mean():.1%} | died: {df['died'].mean():.2%}")
    print(f"Atypical share: {df['is_atypical'].mean():.1%} | outbreak_active share: {df['outbreak_active'].mean():.1%} "
          f"({df.loc[df['outbreak_active'] == 1, 'outbreak_id'].nunique()} episodes with encounters)")
    print("\nEncounters per disease:")
    print(df["disease_name"].value_counts().to_string())
    print("\nEncounters per hospital:")
    print(df.groupby(["hospital_id", "node_name"]).size().to_string())


if __name__ == "__main__":
    main()
