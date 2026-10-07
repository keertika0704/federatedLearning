"""
Objective A - flag atypical presentations (a possible new disease or variant).

How it works (no labels needed to fit):
    For every known disease we learn what a typical patient looks like:
        symptoms : P(symptom present | disease), as independent yes/no probabilities
        vitals   : a small Gaussian mixture over the 6 vitals (one component per severity level)
    A patient's "surprise" under a disease is measured four ways: overall symptom pattern, vitals,
    expected-but-missing symptoms, and unexpected-but-present symptoms. Each is scaled against that
    disease's own typical patients and divided by a threshold calibrated in train.py, so the
    atypical_score (the largest of the four) >= 1.0 means "atypical".

    atypical_score  surprise relative to the DIAGNOSED disease (given, or the best-fitting disease).
                    Catches e.g. loss of taste in a patient diagnosed with influenza.
    novelty_score   the best fit over ALL known diseases (lowest score). High = no known disease
                    explains this patient, the signal for a brand-new disease.
    reasons         which symptoms are unexpected / missing, and which vitals are out of range.

Loads  model/atypical_model.json   (written by train.py)

Use from code:
    from predict import AtypicalDetector
    det = AtypicalDetector()
    det.assess({"S001": 1, "S012": 1, "S013": 1, "temperature": 38.5, ...}, disease_id="D004")

Use from the terminal (from the project root):
    python models/objectiveA/predict.py                         # scores data/generated/generated_data/test.csv
    python models/objectiveA/predict.py --csv some_file.csv --out flags.csv
    python models/objectiveA/predict.py --disease D004 --patient '{"S001": 1, "S012": 1, "S013": 1}'
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
MODEL_FILE = MODEL_DIR / "atypical_model.json"
PROJECT_ROOT = next(p for p in HERE.parents if (p / "models").is_dir())
DEFAULT_TEST = PROJECT_ROOT / "data" / "generated" / "generated_data" / "test.csv"

SYMPTOMS = [f"S{i:03d}" for i in range(1, 29)]
VITALS = ["temperature", "heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp"]
UNEXPECTED_P = 0.10    # a present symptom rarer than this in the diagnosed disease is "unexpected"
EXPECTED_P = 0.60      # an absent symptom more common than this is "missing"
VITAL_Z = 2.5          # vitals further than this many SDs from the disease average are listed


# ------------------------------------------------------------------ scoring maths (shared with train.py)
def to_arrays(df):
    """Symptom matrix (absent/missing = 0) and vitals matrix (missing = NaN)."""
    S = df.reindex(columns=SYMPTOMS).fillna(0).to_numpy(dtype=float)
    V = df.reindex(columns=VITALS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return S, V


def symptom_surprise(S, theta):
    """(n, K) negative log-likelihood of each patient's symptoms under each disease."""
    th = np.clip(np.asarray(theta), 1e-6, 1 - 1e-6)
    return -(S @ np.log(th).T + (1 - S) @ np.log(1 - th).T)


def vital_surprise(V, gmms, vit_mean):
    """(n, K) negative log-density of each patient's vitals under each disease's mixture."""
    out = np.empty((len(V), len(gmms)))
    for k, g in enumerate(gmms):
        Vk = np.where(np.isnan(V), np.asarray(vit_mean[k]), V)       # unknown vital -> disease average
        comp = [np.log(w) + multivariate_normal.logpdf(Vk, mean=m, cov=c, allow_singular=True)
                for w, m, c in zip(g["weights"], g["means"], g["covs"])]
        out[:, k] = -logsumexp(np.vstack(comp), axis=0)
    return out


def robust_z(raw, median, mad):
    return (raw - np.asarray(median)) / np.maximum(np.asarray(mad), 1e-6)


def hallmark_weights(theta):
    """Per-disease, per-symptom surprise weights for the two rule-like components."""
    th = np.clip(np.asarray(theta), 1e-6, 1 - 1e-6)
    missing = np.where(th > EXPECTED_P, -np.log(1 - th), 0.0)      # absent although usually present
    unexpected = np.where(th < UNEXPECTED_P, -np.log(th), 0.0)     # present although usually absent
    return missing, unexpected


COMPONENTS = ["sym", "vit", "miss", "unexp"]
COMPONENT_LABEL = {"sym": "symptom_ratio", "vit": "vitals_ratio", "miss": "missing_ratio", "unexp": "unexpected_ratio"}


def component_scores(params, S, V):
    """Dict name -> (n, K) normalised surprise of each patient under each disease.
    sym / vit: overall surprise of the symptom pattern / the vitals (robust z-score).
    miss / unexp: total surprise of expected-but-absent / unexpected-but-present symptoms."""
    nm = params["norm"]
    wm, wu = hallmark_weights(params["theta"])
    return {
        "sym": robust_z(symptom_surprise(S, params["theta"]), nm["sym"]["median"], nm["sym"]["mad"]),
        "vit": robust_z(vital_surprise(V, params["gmm"], params["vit_mean"]), nm["vit"]["median"], nm["vit"]["mad"]),
        "miss": ((1 - S) @ wm.T) / np.asarray(nm["miss"]["scale"]),
        "unexp": (S @ wu.T) / np.asarray(nm["unexp"]["scale"]),
    }


def ratio_matrix(params, S, V):
    """Per-component ratios (score / threshold) and their max, each (n, K). Ratio >= 1 means atypical."""
    comps = component_scores(params, S, V)
    ratios = {c: comps[c] / params["threshold"][c] for c in COMPONENTS}
    return ratios, np.max(np.stack(list(ratios.values())), axis=0)


# ------------------------------------------------------------------ detector
class AtypicalDetector:
    def __init__(self, path=MODEL_FILE, params=None):
        if params is None:
            if not Path(path).is_file():
                raise SystemExit(f"Missing {path}. Train first:  python models/objectiveA/train.py")
            params = json.loads(Path(path).read_text())
        self.p = params
        self.diseases = params["diseases"]
        self.idx = {d: i for i, d in enumerate(self.diseases)}

    # best-fitting disease when none is given (naive Bayes over symptoms + vitals)
    def infer_disease(self, S, V):
        post = (-symptom_surprise(S, self.p["theta"]) - vital_surprise(V, self.p["gmm"], self.p["vit_mean"])
                + np.log(self.p["prior"]))
        return post.argmax(axis=1)

    def score(self, df, disease_col="disease_id"):
        """Scores for every row of a DataFrame. Uses df[disease_col] as the diagnosis when present."""
        S, V = to_arrays(df)
        ratios, ratio = ratio_matrix(self.p, S, V)
        best = self.infer_disease(S, V)
        if disease_col in df.columns:
            known = df[disease_col].map(self.idx)
            diag = np.where(known.isna(), best, known.fillna(0).astype(int).to_numpy())
        else:
            diag = best
        rows = np.arange(len(df))
        score = ratio[rows, diag]
        novelty = ratio.min(axis=1)
        return pd.DataFrame({
            "diagnosed_disease": [self.diseases[i] for i in diag],
            "atypical_score": score.round(3),
            "atypical_flag": (score >= 1.0).astype(int),
            "novelty_score": novelty.round(3),
            "novelty_flag": (novelty >= self.p["threshold"]["novelty"]).astype(int),
            **{COMPONENT_LABEL[c]: ratios[c][rows, diag].round(3) for c in COMPONENTS},
        }, index=df.index), S, V, diag

    def reasons(self, s_row, v_row, k):
        """Human-readable reasons for one patient under disease index k."""
        names, th = self.p["symptom_names"], np.asarray(self.p["theta"][k])
        dname = self.p.get("disease_names", {}).get(self.diseases[k], self.diseases[k])
        out = []
        for j, sid in enumerate(SYMPTOMS):
            if s_row[j] == 1 and th[j] < UNEXPECTED_P:
                out.append((-np.log(th[j]), f"unexpected symptom: {names[sid]} (seen in {th[j]:.0%} of {dname})"))
            elif s_row[j] == 0 and th[j] > EXPECTED_P:
                out.append((-np.log(1 - th[j]), f"missing expected symptom: {names[sid]} (seen in {th[j]:.0%} of {dname})"))
        mean, sd = np.asarray(self.p["vit_mean"][k]), np.asarray(self.p["vit_std"][k])
        for j, v in enumerate(VITALS):
            if not np.isnan(v_row[j]) and abs(v_row[j] - mean[j]) > VITAL_Z * sd[j]:
                out.append((abs(v_row[j] - mean[j]) / sd[j], f"unusual {v}: {v_row[j]:g} (typical {mean[j]:.1f} +/- {sd[j]:.1f})"))
        return [t for _, t in sorted(out, reverse=True)]

    def assess_df(self, df, disease_col="disease_id", with_reasons=True):
        res, S, V, diag = self.score(df, disease_col)
        if with_reasons:
            res["reasons"] = [" | ".join(self.reasons(S[i], V[i], diag[i])) if res["atypical_flag"].iat[i]
                              or res["novelty_flag"].iat[i] else "" for i in range(len(df))]
        return res

    def assess(self, patient: dict, disease_id=None):
        """One patient (dict of symptom flags S001.. and vitals). Returns a plain dict."""
        df = pd.DataFrame([patient])
        if disease_id:
            df["disease_id"] = disease_id
        r = self.assess_df(df, with_reasons=True).iloc[0].to_dict()
        r = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        r["reasons"] = [x for x in r["reasons"].split(" | ") if x]
        return r


# ------------------------------------------------------------------ evaluation on a labelled file
def print_metrics(res, df):
    from sklearn.metrics import roc_auc_score
    y = df["is_atypical"].astype(int).to_numpy()
    print(f"\nAtypical presentations in file: {y.sum():,} of {len(y):,} ({y.mean():.1%})")
    print(f"AUC (atypical_score vs is_atypical): {roc_auc_score(y, res.atypical_score):.3f}")
    f = res.atypical_flag.to_numpy()
    tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
    print(f"Flagged {f.sum():,} | precision {tp / max(f.sum(), 1):.1%} | recall {tp / max(y.sum(), 1):.1%} "
          f"| false-alarm rate {fp / max((y == 0).sum(), 1):.1%}")
    if "atypical_type" in df:
        print("Recall by type:")
        for t, g in res.assign(t=df["atypical_type"]).groupby("t"):
            if t != "none":
                print(f"  {t:20s} {g.atypical_flag.mean():.1%} of {len(g):,}")


def main():
    ap = argparse.ArgumentParser(description="Flag atypical presentations")
    ap.add_argument("--csv", help=f"file to score (default when no patient is given: {DEFAULT_TEST})")
    ap.add_argument("--out", default=str(HERE / "atypical_predictions.csv"))
    ap.add_argument("--disease", help="diagnosed disease_id for --patient (default: best-fitting disease)")
    ap.add_argument("--patient", help="JSON string or path to .json: symptom flags (S001..) and vitals")
    args = ap.parse_args()

    det = AtypicalDetector()
    if args.patient:
        p = args.patient
        patient = json.loads(Path(p).read_text()) if Path(p).is_file() else json.loads(p)
        print(json.dumps(det.assess(patient, args.disease), indent=2))
        return

    path = args.csv or DEFAULT_TEST
    df = pd.read_csv(path)
    res = det.assess_df(df)
    ids = [c for c in ["encounter_id", "patient_token"] if c in df.columns]
    out = pd.concat([df[ids], res], axis=1)
    if "is_atypical" in df:
        out["is_atypical"] = df["is_atypical"]
    out.to_csv(args.out, index=False)
    print(f"Scored {len(df):,} rows from {path}\nWrote {args.out}")
    print(f"Flagged atypical: {res.atypical_flag.sum():,} ({res.atypical_flag.mean():.1%}) | "
          f"possible new disease (novelty): {res.novelty_flag.sum():,} ({res.novelty_flag.mean():.1%})")
    if "is_atypical" in df:
        print_metrics(res, df)


if __name__ == "__main__":
    main()
