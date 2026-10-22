"""
Federated atypical-presentation detector (Objective A).

The detector has no weights to average; every part of it is a statistic of the training rows, so each
hospital sends only summary statistics and the server assembles the same atypical_model.json that
models/objectiveA/predict.py:AtypicalDetector loads. Stages (one Flower round each):

  moments   per disease: row count, symptom sums, per-vital count / sum / sum of squares, and
            complete-case moments for the mixture's starting point        -> prior, theta, vit_mean, vit_std
  em        distributed EM for each disease's 3-component vitals mixture: per-component
            responsibilities-weighted sums only, M-step on the server (identical maths to fitting on
            pooled data)                                                  -> gmm
  norm1     histograms of each hospital's raw surprise scores per disease -> medians, 95th percentiles
  norm2     histograms of absolute deviations from those medians          -> MADs
  cal1-3    histograms of the three score families on each hospital's own 15% calibration slice, to set
            the thresholds (about `fpr` of ordinary patients flagged)     -> thresholds, novelty threshold

Histograms are over arcsinh(score) on a fixed grid, so quantiles are accurate to about 0.5% relative
and no individual score or range extreme is revealed. Nothing here reads is_atypical / atypical_type
during fitting (as in the centralised trainer); they are only used to evaluate.

The detector (predict.py) and, for baselines only, the trainer (train.py) of Objective A are loaded
unmodified by path.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from carepath_fl.common.partition import PROJECT_ROOT

_A_DIR = PROJECT_ROOT / "models" / "objectiveA"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a = _load("objA_predict", _A_DIR / "predict.py")

SYMPTOMS, VITALS, COMPONENTS = a.SYMPTOMS, a.VITALS, a.COMPONENTS
EVAL_ONLY = ["is_atypical", "atypical_type"]
COLUMNS = SYMPTOMS + VITALS + ["disease_id"]        # all a client reads while fitting
EVAL_COLUMNS = COLUMNS + EVAL_ONLY                  # evaluation labels: test rows only
EVAL_SPLIT = "test"
OUT_DIR = PROJECT_ROOT / "models" / "objectiveC" / "model" / "atypical"
HEADLINE = [("auc", True), ("recall", True), ("false_alarm_rate", None)]   # None: shown, not ranked

FPR = 0.05              # target share of ordinary patients flagged (centralised default)
N_COMP = 3              # one mixture component per severity level
SMOOTHING = 1.0         # Laplace smoothing, as in train.py
REG_COVAR = 1e-2
EM_MAX_ITERS = 50
EM_TOL = 1e-3           # on the mean per-row log-likelihood, as sklearn's GaussianMixture
CALIB_SHARE = 0.15
SEED = 42

STAGES = ("moments", "em", "norm1", "norm2", "cal1", "cal2", "cal3")
BINS, LIM = 8000, 20.0
_WIDTH = 2 * LIM / BINS


def diseases() -> list[str]:
    return sorted(pd.read_csv(PROJECT_ROOT / "data" / "reference" / "master" / "disease_master.csv",
                              encoding="utf-8-sig")["disease_id"].astype(str))


DISEASES = diseases()
D = len(DISEASES)


def fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df


def eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _check_diseases(df: pd.DataFrame) -> None:
    unknown = set(df["disease_id"].dropna().unique()) - set(DISEASES)
    if unknown:
        raise ValueError(f"unexpected disease ids {sorted(unknown)}; expected {DISEASES}")


def split(df: pd.DataFrame):
    """A hospital's fit slice and its own calibration slice (stratified by disease, like train.py;
    unstratified only if some disease has a single row here, which stratifying cannot handle)."""
    _check_diseases(df)
    stratify = df["disease_id"] if df["disease_id"].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=CALIB_SHARE, stratify=stratify, random_state=SEED)


# ------------------------------------------------------------------ histogram quantiles
def _hist(x) -> np.ndarray:
    u = np.arcsinh(np.asarray(x, dtype=float))
    idx = np.clip(((u + LIM) / _WIDTH).astype(int), 0, BINS - 1)
    return np.bincount(idx, minlength=BINS).astype(np.int64)


def _quantile(counts: np.ndarray, q: float) -> float:
    cum = np.cumsum(counts)
    if cum[-1] == 0:
        raise ValueError("cannot take a quantile of an empty histogram")
    target = q * cum[-1]
    i = min(int(np.searchsorted(cum, target, side="left")), BINS - 1)
    before = cum[i - 1] if i else 0
    frac = (target - before) / max(counts[i], 1)
    return float(np.sinh(-LIM + (i + frac) * _WIDTH))


# ------------------------------------------------------------------ client side
def _arrays(df: pd.DataFrame):
    S, V = a.to_arrays(df)
    k = df["disease_id"].map({d: i for i, d in enumerate(DISEASES)})
    if k.isna().any():
        raise ValueError(f"unexpected disease ids {sorted(set(df['disease_id']) - set(DISEASES))}")
    return S, V, k.to_numpy(dtype=int)


def _moments(df: pd.DataFrame) -> dict[str, np.ndarray]:
    S, V, k = _arrays(df)
    out = {"n": np.zeros(D), "sumS": np.zeros((D, len(SYMPTOMS))),
           "vcnt": np.zeros((D, len(VITALS))), "vsum": np.zeros((D, len(VITALS))), "vsq": np.zeros((D, len(VITALS))),
           "ccn": np.zeros(D), "ccsum": np.zeros((D, len(VITALS))), "ccouter": np.zeros((D, len(VITALS), len(VITALS)))}
    for d in range(D):
        m = k == d
        Sd, Vd = S[m], V[m]
        ok = ~np.isnan(Vd)
        cc = Vd[ok.all(axis=1)]
        out["n"][d] = m.sum()
        out["sumS"][d] = Sd.sum(axis=0)
        out["vcnt"][d], out["vsum"][d] = ok.sum(axis=0), np.nansum(Vd, axis=0)
        out["vsq"][d] = np.nansum(Vd ** 2, axis=0)
        out["ccn"][d], out["ccsum"][d], out["ccouter"][d] = len(cc), cc.sum(axis=0), cc.T @ cc
    return out


def _em(df: pd.DataFrame, gmm: dict) -> dict[str, np.ndarray]:
    S, V, k = _arrays(df)
    w, mu, cov = (np.asarray(gmm[x]) for x in ("weights", "means", "covs"))
    p = len(VITALS)
    out = {"nk": np.zeros((D, N_COMP)), "sx": np.zeros((D, N_COMP, p)), "sxx": np.zeros((D, N_COMP, p, p)),
           "ll": np.zeros(D), "n": np.zeros(D)}
    for d in range(D):
        Vd = V[k == d]
        Vd = Vd[~np.isnan(Vd).any(axis=1)]
        if not len(Vd):
            continue
        logp = np.stack([np.log(w[d, j]) + multivariate_normal.logpdf(Vd, mean=mu[d, j], cov=cov[d, j], allow_singular=True)
                         for j in range(N_COMP)], axis=1)
        ll = logsumexp(logp, axis=1)
        resp = np.exp(logp - ll[:, None])
        out["nk"][d], out["sx"][d] = resp.sum(axis=0), resp.T @ Vd
        out["sxx"][d] = np.einsum("nk,ni,nj->kij", resp, Vd, Vd)
        out["ll"][d], out["n"][d] = ll.sum(), len(Vd)
    return out


def client_stage(stage: str, state: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """What one hospital returns for one stage. Only counts, sums and histograms leave the hospital."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    fit, calib = split(df)
    if stage == "moments":
        return _moments(fit)
    if stage == "em":
        return _em(fit, state["gmm"])

    params = state["params"]
    if stage in ("norm1", "norm2"):
        S, V, k = _arrays(fit)
        raw_s = a.symptom_surprise(S, params["theta"])
        raw_v = a.vital_surprise(V, params["gmm"], params["vit_mean"])
        if stage == "norm1":
            wm, wu = a.hallmark_weights(params["theta"])
            raws = (raw_s, raw_v, (1 - S) @ wm.T, S @ wu.T)
            return {"hist": np.stack([np.stack([_hist(raw[k == d, d]) for raw in raws]) for d in range(D)])}
        med_s, med_v = np.asarray(state["med_sym"]), np.asarray(state["med_vit"])
        return {"hist": np.stack([np.stack([_hist(np.abs(raw_s[k == d, d] - med_s[d])),
                                            _hist(np.abs(raw_v[k == d, d] - med_v[d]))]) for d in range(D)])}

    S, V, k = _arrays(calib)
    rows = np.arange(len(calib))
    if stage == "cal1":
        comps = a.component_scores(params, S, V)
        return {"hist": np.stack([_hist(comps[c][rows, k]) for c in COMPONENTS])}
    _, ratio = a.ratio_matrix(params, S, V)
    if stage == "cal2":
        return {"hist": _hist(ratio[rows, k])}
    return {"hist": _hist(ratio.min(axis=1))}                                 # cal3


# ------------------------------------------------------------------ server side
def _sum(parts: list[dict], key: str) -> np.ndarray:
    return sum(p[key] for p in parts)


def _listify(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def run_protocol(call, fpr: float = FPR, log=print) -> tuple[dict, dict]:
    """Drive every stage. `call(stage, state) -> list of per-hospital {name: ndarray}` hides the transport
    (Flower in the app, direct function calls in tests). Returns (params, info)."""
    p = len(VITALS)
    parts = call("moments", {})
    n, sumS = _sum(parts, "n"), _sum(parts, "sumS")
    if (n == 0).any():
        raise ValueError(f"no training rows for diseases {[DISEASES[i] for i in np.flatnonzero(n == 0)]}")
    vcnt, vsum, vsq = (_sum(parts, k) for k in ("vcnt", "vsum", "vsq"))
    vit_mean = vsum / vcnt
    vit_std = np.sqrt(np.maximum(vsq / vcnt - vit_mean ** 2, 0.0))         # population SD, like np.nanstd
    theta = (sumS + SMOOTHING) / (n[:, None] + 2 * SMOOTHING)
    prior = n / n.sum()

    # mixture starting point: components spread along the disease's own mean +/- 1 SD (mild/moderate/severe)
    ccn, ccsum, ccouter = _sum(parts, "ccn"), _sum(parts, "ccsum"), _sum(parts, "ccouter")
    mean0 = ccsum / ccn[:, None]
    cov0 = ccouter / ccn[:, None, None] - np.einsum("di,dj->dij", mean0, mean0)
    sd0 = np.sqrt(np.einsum("dii->di", cov0))
    z = np.linspace(-1.0, 1.0, N_COMP)
    w = np.full((D, N_COMP), 1.0 / N_COMP)
    mu = mean0[:, None, :] + z[None, :, None] * sd0[:, None, :]
    cov = np.broadcast_to(cov0[:, None], (D, N_COMP, p, p)) + REG_COVAR * np.eye(p)

    prev_ll, iters = None, 0
    for iters in range(1, EM_MAX_ITERS + 1):
        parts = call("em", {"gmm": _listify({"weights": w, "means": mu, "covs": cov})})
        nk = _sum(parts, "nk") + 10 * np.finfo(float).eps
        sx, sxx = _sum(parts, "sx"), _sum(parts, "sxx")
        w = nk / nk.sum(axis=1, keepdims=True)
        mu = sx / nk[..., None]
        cov = sxx / nk[..., None, None] - np.einsum("dki,dkj->dkij", mu, mu) + REG_COVAR * np.eye(p)
        ll = _sum(parts, "ll") / _sum(parts, "n")
        if prev_ll is not None and np.abs(ll - prev_ll).max() < EM_TOL:
            break
        prev_ll = ll
    log(f"[atypical] vitals mixtures: {iters} EM rounds, mean log-likelihood/row {ll.mean():.3f}")

    params = {"diseases": list(DISEASES), "prior": prior.tolist(), "theta": theta.tolist(),
              "gmm": [{"weights": w[d].tolist(), "means": mu[d].tolist(), "covs": cov[d].tolist()} for d in range(D)],
              "vit_mean": vit_mean.tolist(), "vit_std": vit_std.tolist(),
              "threshold": {**{c: 1.0 for c in COMPONENTS}, "novelty": 1.0}}

    h = _sum(call("norm1", {"params": params}), "hist")                      # (D, 4, BINS)
    med_s = [_quantile(h[d, 0], 0.5) for d in range(D)]
    med_v = [_quantile(h[d, 1], 0.5) for d in range(D)]
    h2 = _sum(call("norm2", {"params": params, "med_sym": med_s, "med_vit": med_v}), "hist")
    params["norm"] = {
        "sym": {"median": med_s, "mad": [1.4826 * _quantile(h2[d, 0], 0.5) for d in range(D)]},
        "vit": {"median": med_v, "mad": [1.4826 * _quantile(h2[d, 1], 0.5) for d in range(D)]},
        "miss": {"scale": [max(_quantile(h[d, 2], 0.95), 1.0) for d in range(D)]},
        "unexp": {"scale": [max(_quantile(h[d, 3], 0.95), 1.0) for d in range(D)]},
    }

    hc = _sum(call("cal1", {"params": params}), "hist")                      # (4, BINS)
    for i, c in enumerate(COMPONENTS):                                        # the components share the false-alarm budget
        params["threshold"][c] = float(max(_quantile(hc[i], 1 - fpr / len(COMPONENTS)), 1e-3))
    h_ratio = _sum(call("cal2", {"params": params}), "hist")                  # components overlap: rescale to hit fpr
    scale = _quantile(h_ratio, 1 - fpr)
    for c in COMPONENTS:
        params["threshold"][c] *= scale
    h_nov = _sum(call("cal3", {"params": params}), "hist")
    params["threshold"]["novelty"] = float(_quantile(h_nov, 1 - fpr))
    log("[atypical] thresholds: " + ", ".join(f"{k} {v:.2f}" for k, v in params["threshold"].items()))

    info = {"rows": {"fit": int(n.sum()), "calibration": int(h_nov.sum())}, "em_rounds": iters, "fpr_target": fpr}
    return _listify(params), info


# ------------------------------------------------------------------ export
def _readable_names() -> dict:
    master = PROJECT_ROOT / "data" / "reference" / "master"
    out = {"symptom_names": {s: s for s in SYMPTOMS}, "disease_names": {}}
    sym = pd.read_csv(master / "symptom_master.csv", encoding="utf-8-sig")
    out["symptom_names"].update(dict(zip(sym["symptom_id"], sym["symptom_name"])))
    dis = pd.read_csv(master / "disease_master.csv", encoding="utf-8-sig")
    out["disease_names"] = dict(zip(dis["disease_id"], dis["disease_name"]))
    return out


def write_model(params: dict, extra_meta: dict | None = None, reference=None, out_dir: Path = OUT_DIR) -> Path:
    """atypical_model.json, loadable by models/objectiveA/predict.py:AtypicalDetector(path=...)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = extra_meta or {}
    params = {**params, **_readable_names(), "meta": {
        "model": "federated: per-disease symptom Bernoulli + vitals Gaussian mixture (distributed EM), "
                 "robust-z calibrated from merged histograms",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}}
    path = out_dir / "atypical_model.json"
    path.write_text(json.dumps(params, indent=2))
    return out_dir


# ------------------------------------------------------------------ used by carepath_fl.report
def load_models(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "atypical_model.json").read_text())


def _a_train():
    """Objective A's own trainer, unmodified, for the single-hospital / pooled baselines."""
    prev = sys.modules.get("predict")
    sys.modules["predict"] = a                       # its `from predict import ...` resolves to A's predict.py
    try:
        return _load("objA_train", _A_DIR / "train.py")
    finally:
        if prev is None:
            sys.modules.pop("predict", None)
        else:
            sys.modules["predict"] = prev


def fit_baseline(df: pd.DataFrame, fpr: float = FPR) -> dict:
    """The centralised algorithm on `df` (one hospital alone, or every hospital pooled)."""
    fit, calib = split(df)
    params = _a_train().build(fit, calib, fpr, SEED)
    return json.loads(json.dumps(params, default=float))


def score_models(params: dict, df: pd.DataFrame) -> dict:
    det = a.AtypicalDetector(params=params)
    res = det.score(df)[0]
    y = df["is_atypical"].astype(int).to_numpy()
    f = res.atypical_flag.to_numpy()
    tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
    own = det.score(df.drop(columns=["disease_id"]))[0].diagnosed_disease.to_numpy() == df["disease_id"].to_numpy()
    return {"auc": float(roc_auc_score(y, res.atypical_score)),
            "precision": tp / max(int(f.sum()), 1), "recall": tp / max(int(y.sum()), 1),
            "false_alarm_rate": fp / max(int((y == 0).sum()), 1),
            "self_diagnosis_accuracy": float(own.mean())}
