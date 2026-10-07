"""
Objective A - train the atypical-presentation detector.

Fits "what a typical patient looks like" for each known disease (symptom probabilities + a Gaussian
mixture over vitals), then calibrates how far from typical counts as atypical. The fit does NOT use
the is_atypical label, because a genuinely new disease has no label. The label (and atypical_type)
are only used afterwards to measure how well the detector works.

Reads (only these columns of train.csv / test.csv, a split of encounters.csv):
    age/gender are NOT used; symptoms S001..S028, the 6 vitals, disease_id,
    and (evaluation only) is_atypical, atypical_type.
    symptom_master / disease_master for readable names in the output.

Writes
    model/atypical_model.json     everything predict.py needs (plain numbers, no pickle)

Reports
    1. Detection quality on test.csv vs the is_atypical label (AUC, precision, recall by type).
    2. "New disease" experiment: each disease is hidden from the model in turn, then its patients are
       scored. A good detector flags them as "no known disease explains this" (novelty_flag), much more
       often than it flags patients of diseases it does know. This is the COVID-looks-like-flu situation.

Run (from the project root)
    python models/objectiveA/train.py
    python models/objectiveA/train.py --train path/train.csv --test path/test.csv --fpr 0.05 --clean-fit
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from predict import (COMPONENTS, DEFAULT_TEST, MODEL_DIR, MODEL_FILE, PROJECT_ROOT, SYMPTOMS, VITALS,
                     AtypicalDetector, component_scores, hallmark_weights, ratio_matrix, symptom_surprise,
                     to_arrays, vital_surprise)

DEFAULT_TRAIN = DEFAULT_TEST.with_name("train.csv")
EVAL_ONLY = ["is_atypical", "atypical_type"]
KEEP = SYMPTOMS + VITALS + ["disease_id"] + EVAL_ONLY
N_SEVERITY_LEVELS = 3        # mild / moderate / severe -> one mixture component each
SMOOTHING = 1.0              # Laplace smoothing for symptom probabilities


# ------------------------------------------------------------------ data
def load(path, label):
    df = pd.read_csv(path)
    missing = [c for c in SYMPTOMS + VITALS + ["disease_id"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    df = df[[c for c in KEEP if c in df.columns]].dropna(subset=["disease_id"]).reset_index(drop=True)
    print(f"[load] {label}: {len(df):,} rows" + (f", {df.is_atypical.mean():.1%} labelled atypical" if "is_atypical" in df else ""))
    return df


def readable_names():
    master = PROJECT_ROOT / "data" / "reference" / "master"
    out = {"symptom_names": {s: s for s in SYMPTOMS}, "disease_names": {}}
    try:
        sym = pd.read_csv(master / "symptom_master.csv")
        out["symptom_names"].update(dict(zip(sym["symptom_id"], sym["symptom_name"])))
        dis = pd.read_csv(master / "disease_master.csv")
        out["disease_names"] = dict(zip(dis["disease_id"], dis["disease_name"]))
    except FileNotFoundError:
        print("[warn] reference master tables not found, using IDs instead of names")
    return out


# ------------------------------------------------------------------ fitting
def fit_params(df, seed=42):
    """Learn the typical presentation of every disease in df."""
    diseases = sorted(df["disease_id"].unique())
    S, V = to_arrays(df)
    prior, theta, gmm, vmean, vstd = [], [], [], [], []
    for d in diseases:
        m = (df["disease_id"] == d).to_numpy()
        prior.append(m.mean())
        theta.append(((S[m].sum(axis=0) + SMOOTHING) / (m.sum() + 2 * SMOOTHING)).tolist())
        Vd = V[m][~np.isnan(V[m]).any(axis=1)]
        g = GaussianMixture(N_SEVERITY_LEVELS, covariance_type="full", reg_covar=1e-2, random_state=seed).fit(Vd)
        gmm.append({"weights": g.weights_.tolist(), "means": g.means_.tolist(), "covs": g.covariances_.tolist()})
        vmean.append(np.nanmean(V[m], axis=0).tolist())
        vstd.append(np.nanstd(V[m], axis=0).tolist())
    params = {"diseases": diseases, "prior": prior, "theta": theta, "gmm": gmm,
              "vit_mean": vmean, "vit_std": vstd,
              "threshold": {**{c: 1.0 for c in COMPONENTS}, "novelty": 1.0}}

    # robust normalisation: each disease's own typical surprise level (median / MAD)
    raw_s, raw_v = symptom_surprise(S, params["theta"]), vital_surprise(V, gmm, vmean)
    wm, wu = hallmark_weights(theta)
    raw_m, raw_u = (1 - S) @ wm.T, S @ wu.T
    norm = {"sym": {"median": [], "mad": []}, "vit": {"median": [], "mad": []},
            "miss": {"scale": []}, "unexp": {"scale": []}}
    for k, d in enumerate(diseases):
        m = (df["disease_id"] == d).to_numpy()
        for key, raw in (("sym", raw_s), ("vit", raw_v)):
            x = raw[m, k]
            med = float(np.median(x))
            norm[key]["median"].append(med)
            norm[key]["mad"].append(float(1.4826 * np.median(np.abs(x - med))))
        for key, raw in (("miss", raw_m), ("unexp", raw_u)):   # mostly zeros, so scale by the 95th percentile (min 1 nat)
            norm[key]["scale"].append(float(max(np.quantile(raw[m, k], 0.95), 1.0)))
    params["norm"] = norm
    return params


def calibrate(params, calib, fpr):
    """Pick thresholds so about `fpr` of ordinary patients get flagged (no atypical labels needed)."""
    S, V = to_arrays(calib)
    comps = component_scores(params, S, V)
    k = calib["disease_id"].map({d: i for i, d in enumerate(params["diseases"])}).to_numpy()
    rows = np.arange(len(calib))
    for c in COMPONENTS:                                       # the components share the false-alarm budget
        params["threshold"][c] = float(max(np.quantile(comps[c][rows, k], 1 - fpr / len(COMPONENTS)), 1e-3))
    _, ratio = ratio_matrix(params, S, V)
    scale = float(np.quantile(ratio[rows, k], 1 - fpr))        # components overlap, so rescale to hit the target exactly
    for c in COMPONENTS:
        params["threshold"][c] *= scale
    _, ratio = ratio_matrix(params, S, V)
    params["threshold"]["novelty"] = float(np.quantile(ratio.min(axis=1), 1 - fpr))
    return params


def build(fit_df, calib_df, fpr, seed=42):
    return calibrate(fit_params(fit_df, seed), calib_df, fpr)


# ------------------------------------------------------------------ evaluation
def evaluate(params, test):
    from sklearn.metrics import roc_auc_score
    det = AtypicalDetector(params=params)
    res, _, _, diag = det.score(test)
    out = {}
    if "is_atypical" in test:
        y = test["is_atypical"].astype(int).to_numpy()
        f = res.atypical_flag.to_numpy()
        tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
        out = {"auc": float(roc_auc_score(y, res.atypical_score)),
               "precision": tp / max(int(f.sum()), 1), "recall": tp / max(int(y.sum()), 1),
               "false_alarm_rate": fp / max(int((y == 0).sum()), 1)}
        print("\n=== Atypical detection on test (diagnosis given) ===")
        print(f"AUC {out['auc']:.3f} | flagged {f.mean():.1%} | precision {out['precision']:.1%} | "
              f"recall {out['recall']:.1%} | false-alarm rate {out['false_alarm_rate']:.1%}")
        if "atypical_type" in test:
            out["recall_by_type"] = {}
            for t, g in res.assign(t=test["atypical_type"]).groupby("t"):
                if t != "none":
                    out["recall_by_type"][t] = float(g.atypical_flag.mean())
                    print(f"  recall on {t:20s}: {g.atypical_flag.mean():.1%} (n={len(g):,})")
        print("Flag rate by disease on ordinary (non-atypical) patients:")
        typical = res[y == 0].assign(d=test["disease_id"][y == 0].to_numpy())
        print("  " + ", ".join(f"{d} {g.atypical_flag.mean():.1%}" for d, g in typical.groupby("d")))

    # same score but the disease is NOT told: the detector uses its own best guess
    res2 = det.score(test.drop(columns=["disease_id"]))[0]
    acc = float((res2.diagnosed_disease.to_numpy() == test["disease_id"].to_numpy()).mean())
    out["self_diagnosis_accuracy"] = acc
    print(f"\nWithout a given diagnosis, the detector's own best-fit disease is right for {acc:.1%} of patients"
          + (f"; AUC {roc_auc_score(test['is_atypical'], res2.atypical_score):.3f}" if "is_atypical" in test else ""))
    return out


def new_disease_experiment(fit_df, calib_df, test, fpr, seed):
    """Hide one disease at a time, then check whether its patients get the 'no known disease fits' flag."""
    print("\n=== New-disease experiment (leave one disease out) ===")
    print("Each disease is removed from training, then its test patients are scored. novelty_flag = no remaining")
    print(f"disease explains the patient. Threshold set for ~{fpr:.0%} false alarms on known diseases.\n")
    rows = []
    for d in sorted(test["disease_id"].unique()):
        params = build(fit_df[fit_df.disease_id != d], calib_df[calib_df.disease_id != d], fpr, seed)
        det = AtypicalDetector(params=params)
        res = det.score(test.drop(columns=["disease_id"]))[0]
        hidden = (test["disease_id"] == d).to_numpy()
        rows.append({"hidden_disease": d, "n": int(hidden.sum()),
                     "flagged_when_unknown": res.novelty_flag[hidden].mean(),
                     "flagged_when_known": res.novelty_flag[~hidden].mean(),
                     "score_unknown": res.novelty_score[hidden].median(),
                     "score_known": res.novelty_score[~hidden].median()})
    t = pd.DataFrame(rows)
    print(t.assign(**{c: t[c].map("{:.1%}".format) for c in ["flagged_when_unknown", "flagged_when_known"]})
           .round(2).to_string(index=False))
    return t


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--fpr", type=float, default=0.05, help="target share of ordinary patients flagged (default 5%%)")
    ap.add_argument("--clean-fit", action="store_true",
                    help="fit only on rows labelled not atypical (needs labels; default fits on everything, like real life)")
    ap.add_argument("--skip-new-disease", action="store_true", help="skip the leave-one-disease-out experiment")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, test = load(args.train, "train"), load(args.test, "test")
    fit_df, calib_df = train_test_split(train, test_size=0.15, stratify=train["disease_id"], random_state=args.seed)
    if args.clean_fit:
        fit_df = fit_df[fit_df["is_atypical"] == 0]
        print(f"[fit] --clean-fit: {len(fit_df):,} typical rows")
    print(f"[split] fit {len(fit_df):,} / calibration {len(calib_df):,} / test {len(test):,}")

    params = build(fit_df, calib_df, args.fpr, args.seed)
    print("[calibrate] thresholds: " + ", ".join(f"{k} {v:.2f}" for k, v in params["threshold"].items()))
    metrics = evaluate(params, test)
    if not args.skip_new_disease:
        exp = new_disease_experiment(fit_df, calib_df, test, args.fpr, args.seed)
        metrics["new_disease"] = exp.round(4).to_dict(orient="records")

    params.update(readable_names())
    params["meta"] = {"model": "per-disease symptom Bernoulli + vitals Gaussian mixture, robust-z calibrated",
                      "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "fpr_target": args.fpr, "clean_fit": args.clean_fit,
                      "rows": {"fit": len(fit_df), "calibration": len(calib_df), "test": len(test)},
                      "metrics": metrics}
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_FILE.write_text(json.dumps(params, indent=2, default=float))
    print(f"\nSaved {MODEL_FILE}")


if __name__ == "__main__":
    main()
