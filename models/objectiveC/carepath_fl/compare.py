"""
Original (centrally trained) models against the federated ones, on the same held-out patients.

    python -m carepath_fl.compare        # from models/objectiveC, CAREPATH_ROOT set; writes model/comparison.json

The original models are read from the folders of Objectives A, B and E exactly as they are on disk, and the
federated ones from model/<task>/. Both are loaded and scored the way the web app's own code loads them (the
same XGBoost classes, the same feature builders), on every hospital's rows of test.csv. Nothing is trained.

Every figure of a model comes with a paired bootstrap interval for the difference (federated minus original):
the same resampled patients score both models, so luck in which patients were drawn cancels out. Treatment rows
are resampled by patient, since one patient has several rows. A difference is only called better or worse when
that 95% interval excludes zero. The outbreak detector has no per-patient score, so its two sets of recorded
held-out numbers are shown side by side without an interval.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, recall_score, roc_auc_score, top_k_accuracy_score
from xgboost import XGBClassifier

from carepath_fl.common.partition import PROJECT_ROOT, hospital_ids, load_partition
from carepath_fl.tasks import atypical, disease, outbreak, treatment

MODELS = PROJECT_ROOT / "models"
OUT_FILE = MODELS / "objectiveC" / "model" / "comparison.json"
ORIGINAL = {"disease": MODELS / "objectiveE" / "disease_prediction" / "model",
            "treatment": MODELS / "objectiveE" / "treatment_prediction" / "model",
            "atypical": MODELS / "objectiveA" / "model",
            "outbreak": MODELS / "objectiveB" / "model"}
FEDERATED = {"disease": disease.OUT_DIR, "treatment": treatment.OUT_DIR,
             "atypical": atypical.OUT_DIR, "outbreak": outbreak.OUT_DIR}
RESAMPLES = 1000
SEED = 20261027

OUTBREAK_ROWS = [("episodes", "Held-out outbreaks", "int", None),
                 ("linked_episodes", "…that spread between hospitals", "int", None),
                 ("linked_alert_recall", "Spreading outbreaks alerted", "pct", True),
                 ("alert_recall", "All outbreaks alerted", "pct", True),
                 ("median_delay_days", "Median days to alert", "days", False),
                 ("notifications", "Notifications sent", "int", None),
                 ("notification_precision", "Notifications that were correct", "pct", True),
                 ("false_notifications_per_year", "False notifications per year", "num1", False)]


def _mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in ([path] if path.is_file() else path.glob("*.json"))
                     if p.name not in ("report.json", "meta.json", "comparison.json")) / 1e6, 2)


def _read(path: Path):
    return json.loads(path.read_text())


def test_rows(columns: list[str]) -> pd.DataFrame:
    """Every hospital's rows of test.csv, with a `hospital` column."""
    cols = list(dict.fromkeys([*columns, "hospital_id"]))
    return pd.concat([load_partition(h, "test", columns=cols) for h in hospital_ids()],
                     ignore_index=True).rename(columns={"hospital_id": "hospital"})


# ------------------------------------------------------------------ statistics
def _resamples(n: int, clusters: np.ndarray | None, rng) -> list[np.ndarray]:
    """Index sets for the paired bootstrap; with `clusters`, whole clusters (patients) are drawn."""
    if clusters is None:
        return [rng.integers(0, n, n) for _ in range(RESAMPLES)]
    order = np.argsort(clusters, kind="stable")
    bounds = np.flatnonzero(np.r_[True, clusters[order][1:] != clusters[order][:-1], True])
    members = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    return [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            for _ in range(RESAMPLES)]


def _verdict(ci, higher):
    if ci is None or higher is None:
        return None
    if ci[0] > 0:
        return "better" if higher else "worse"
    if ci[1] < 0:
        return "worse" if higher else "better"
    return "no clear difference"


def _metric(spec: dict, hospital: np.ndarray, resamples: list[np.ndarray]) -> dict:
    """spec: id, label, format, higher (None = shown, not ranked), fn(idx, which) -> float, ci, per_hospital."""
    fn, every = spec["fn"], np.arange(len(hospital))
    original, federated = fn(every, "original"), fn(every, "federated")
    ci = None
    if spec["ci"]:
        diff = [fn(i, "federated") - fn(i, "original") for i in resamples]
        ci = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    per = []
    if spec["per_hospital"]:
        for h in hospital_ids():
            idx = np.flatnonzero(hospital == h)
            per.append({"hospital": h, "rows": int(len(idx)), "original": fn(idx, "original"),
                        "federated": fn(idx, "federated")})
    return {"id": spec["id"], "label": spec["label"], "format": spec["format"], "higher_is_better": spec["higher"],
            "original": original, "federated": federated, "difference": federated - original, "ci": ci,
            "verdict": _verdict(ci, spec["higher"]), "per_hospital": per,
            "hospitals_better": (sum(1 for r in per if (r["federated"] > r["original"]) == bool(spec["higher"])
                                     and r["federated"] != r["original"]) if per and spec["higher"] is not None else None)}


def _rounded(x):
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    return round(float(x), 5) if isinstance(x, (float, np.floating)) else x


# ------------------------------------------------------------------ disease
def _disease_probabilities(model_dir: Path, df: pd.DataFrame) -> np.ndarray:
    """What the web app computes: XGBClassifier.predict_proba on the features named in meta.json."""
    meta = _read(model_dir / "meta.json")
    if meta["classes"] != disease.CLASSES:
        raise ValueError(f"{model_dir}: classes {meta['classes']} differ from {disease.CLASSES}")
    model = XGBClassifier()
    model.load_model(model_dir / "model.json")
    X = disease.e1.build_features(df[disease.e1.FEATURES + [disease.e1.TARGET]])[meta["features"]].astype(float)
    return model.predict_proba(X)


def compare_disease() -> dict:
    df = test_rows(disease.COLUMNS)
    y = df[disease.e1.TARGET].map(disease.CLASS_INDEX).to_numpy()
    proba = {w: _disease_probabilities(d, df) for w, d in (("original", ORIGINAL["disease"]),
                                                           ("federated", FEDERATED["disease"]))}
    rare = disease.CLASS_INDEX[disease.RARE]

    def acc(i, w): return float((proba[w][i].argmax(1) == y[i]).mean())
    def f1(i, w): return float(f1_score(y[i], proba[w][i].argmax(1), average="macro"))
    def top3(i, w): return float(top_k_accuracy_score(y[i], proba[w][i], k=3, labels=range(len(disease.CLASSES))))
    def rec(i, w): return float(recall_score(y[i], proba[w][i].argmax(1), labels=[rare], average=None, zero_division=0)[0])

    specs = [dict(id="accuracy", label="Accuracy", format="pct", higher=True, fn=acc, ci=True, per_hospital=True),
             dict(id="macro_f1", label="Macro F1 (every disease counts equally)", format="pct", higher=True, fn=f1,
                  ci=True, per_hospital=False),
             dict(id="top3_accuracy", label="Right disease among the top 3", format="pct", higher=True, fn=top3,
                  ci=True, per_hospital=False),
             dict(id="D003_recall", label="Chikungunya recall (the rarest disease)", format="pct", higher=True,
                  fn=rec, ci=True, per_hospital=True)]
    fed_meta, orig_meta = _read(FEDERATED["disease"] / "meta.json"), _read(ORIGINAL["disease"] / "meta.json")
    orig_best = json.loads((ORIGINAL["disease"] / "model.json").read_text())["learner"]["attributes"].get("best_iteration")
    facts = [("Trained on", "Every hospital's patients pooled in one place", "Each hospital's own patients, which never left it"),
             ("Boosting iterations", str(int(orig_best) + 1) if orig_best is not None else "–",
              str(fed_meta["federated"]["iterations"])),
             ("Model file", f"{_mb(ORIGINAL['disease'] / 'model.json')} MB", f"{_mb(FEDERATED['disease'] / 'model.json')} MB")]
    notes = ["Both models are scored on the same held-out patients, with the same features.",
             "The federated model's turns and iterations were picked after looking at test results, so its score is "
             "slightly optimistic. The original model's size came from early stopping on a validation slice, not the test set.",
             "The two have a different number of iterations, so a difference may come from model size as well as from how they were trained.",
             "Rare-disease figures per hospital rest on 30 to 60 patients and are noisy."]
    return {"df": df, "specs": specs, "clusters": None, "facts": facts, "notes": notes, "orig_meta": orig_meta}


# ------------------------------------------------------------------ treatment
def compare_treatment() -> dict:
    df = test_rows(treatment.COLUMNS)
    X = treatment._features(df)
    y_s, y_r = df["success"].astype(int).to_numpy(), df["recovery_days"].astype(float).to_numpy()
    p, r = {}, {}
    for w in ("original", "federated"):
        base = ORIGINAL["treatment"] if w == "original" else FEDERATED["treatment"]
        rec = treatment.e2.TreatmentRecommender(model_dir=base)         # the production class, unmodified
        p[w] = rec.success_model.predict_proba(X)[:, 1]
        r[w] = np.clip(rec.recovery_model.predict(X), 1, None)         # as the recommender reports it

    def auc(i, w): return float(roc_auc_score(y_s[i], p[w][i]))
    def mae(i, w): return float(mean_absolute_error(y_r[i], r[w][i]))

    specs = [dict(id="auc", label="Success prediction AUC", format="num3", higher=True, fn=auc, ci=True, per_hospital=True),
             dict(id="recovery_mae", label="Recovery-time error (days)", format="days", higher=False, fn=mae, ci=True,
                  per_hospital=True)]
    om = _read(ORIGINAL["treatment"] / "meta.json")
    fm = _read(FEDERATED["treatment"] / "meta.json")["federated"]["iterations"]
    facts = [("Trained on", "Every hospital's patients pooled in one place", "Each hospital's own patients, which never left it"),
             ("Boosting iterations (success / recovery)",
              f"{om['best_iteration']['success'] + 1} / {om['best_iteration']['recovery'] + 1}",
              f"{fm['success']} / {fm['recovery']}"),
             ("Model files", f"{_mb(ORIGINAL['treatment'] / 'success_model.json') + _mb(ORIGINAL['treatment'] / 'recovery_model.json'):.2f} MB",
              f"{_mb(FEDERATED['treatment'] / 'success_model.json') + _mb(FEDERATED['treatment'] / 'recovery_model.json'):.2f} MB")]
    notes = ["Both models score the same held-out rows. Intervals resample whole patients, because one patient has several rows.",
             "The signal in this data is weak (AUC around 0.61 for either model), so small differences are easy to over-read.",
             "The federated settings were chosen on each hospital's own validation slice, never the test split."]
    return {"df": df, "specs": specs, "clusters": df["patient_token"].to_numpy(), "facts": facts, "notes": notes,
            "orig_meta": om, "recorded": {"auc": om["metrics"]["success"]["xgboost"]["auc"],
                                          "recovery_mae": om["metrics"]["recovery_mae"]["xgboost"]}}


# ------------------------------------------------------------------ atypical
def compare_atypical() -> dict:
    df = test_rows(atypical.EVAL_COLUMNS)
    y = df["is_atypical"].astype(int).to_numpy()
    score, flag = {}, {}
    for w, base in (("original", ORIGINAL["atypical"]), ("federated", FEDERATED["atypical"])):
        det = atypical.a.AtypicalDetector(path=base / "atypical_model.json")      # the production class, unmodified
        res = det.score(df)[0]
        score[w], flag[w] = res.atypical_score.to_numpy(), res.atypical_flag.to_numpy().astype(int)

    def auc(i, w): return float(roc_auc_score(y[i], score[w][i]))
    def recall(i, w): return float(flag[w][i][y[i] == 1].mean())
    def alarms(i, w): return float(flag[w][i][y[i] == 0].mean())

    specs = [dict(id="auc", label="Detection AUC", format="num3", higher=True, fn=auc, ci=True, per_hospital=True),
             dict(id="recall", label="Share of atypical cases caught", format="pct", higher=True, fn=recall, ci=True,
                  per_hospital=True),
             dict(id="false_alarm_rate", label="False alarms on ordinary patients (a calibration target)", format="pct",
                  higher=None, fn=alarms, ci=False, per_hospital=True)]
    om = _read(ORIGINAL["atypical"] / "atypical_model.json")["meta"]
    fm = _read(FEDERATED["atypical"] / "atypical_model.json")["meta"]
    facts = [("Trained on", "Every hospital's patients pooled in one place",
              "Counts, sums and histograms from each hospital; never patient rows"),
             ("Patients behind the fit", f"{om['rows']['fit']:,}", f"{fm['rows']['fit']:,}"),
             ("Model file", f"{_mb(ORIGINAL['atypical'] / 'atypical_model.json')} MB",
              f"{_mb(FEDERATED['atypical'] / 'atypical_model.json')} MB")]
    notes = ["Both detectors are asked about the same held-out patients and are told each patient's real diagnosis.",
             "Each detector sets its own alert threshold to flag about 5% of ordinary patients, so the false-alarm rate "
             "is a target, not a result, and it is not ranked. Compare the share caught at that rate.",
             "The federated version matches pooled statistics exactly where they are sums and to about 1% where they are histogram quantiles."]
    return {"df": df, "specs": specs, "clusters": None, "facts": facts, "notes": notes, "orig_meta": om,
            "recorded": {"auc": om["metrics"]["auc"], "recall": om["metrics"]["recall"]}}


# ------------------------------------------------------------------ outbreak
def compare_outbreak() -> dict:
    om, fm = _read(ORIGINAL["outbreak"] / "meta.json"), _read(FEDERATED["outbreak"] / "meta.json")
    o, f = om["metrics"]["test"], fm["metrics"]["test"]
    rows = [{"id": k, "label": lab, "format": fmt, "higher_is_better": higher, "original": o.get(k), "federated": f.get(k)}
            for k, lab, fmt, higher in OUTBREAK_ROWS]
    return {"rows": rows, "test_days": fm.get("test_days"),
            "facts": [{"label": "Trained on", "original": "Every hospital's daily counts pooled in one place",
                       "federated": "Daily counts and labelled outbreaks sent by each hospital"},
                      {"label": "Local flagger used for the counts", "original": "The original atypical detector",
                       "federated": "The federated atypical detector"}],
            "notes": ["These are the held-out numbers each pipeline recorded when it was calibrated, not recomputed here. "
                      "They cover the same days.",
                      f"Only {o.get('episodes', '21')} labelled outbreaks fall in the held-out period, so there are few events to compare and no "
                      "interval can be given.",
                      "The detector works on daily counts, so federating its calibration changes little."]}


# ------------------------------------------------------------------ run
TITLES = {"disease": "Disease classifier", "treatment": "Treatment recommender",
          "atypical": "Atypical-case detector", "outbreak": "Outbreak detector"}


def build() -> dict:
    rng = np.random.default_rng(SEED)
    tasks = []
    for name, fn in (("disease", compare_disease), ("treatment", compare_treatment), ("atypical", compare_atypical)):
        print(f"[compare] {name}")
        part = fn()
        hospital = part["df"]["hospital"].to_numpy()
        resamples = _resamples(len(hospital), part["clusters"], rng)
        metrics = [_metric(s, hospital, resamples) for s in part["specs"]]
        # sanity: the originals recorded their own held-out numbers; ours must agree with them
        for key, recorded in (part.get("recorded") or {}).items():
            ours = next(m["original"] for m in metrics if m["id"] == key)
            print(f"  original {key}: scored here {ours:.4f}, recorded by its own training {recorded:.4f}")
            if abs(ours - recorded) > 0.005:
                raise RuntimeError(f"{name}.{key}: the original scores {ours:.4f} here but recorded {recorded:.4f}")
        tasks.append({"id": name, "title": TITLES[name], "kind": "scored", "test_rows": int(len(hospital)),
                      "metrics": metrics, "facts": [dict(zip(("label", "original", "federated"), f)) for f in part["facts"]],
                      "notes": part["notes"]})
    print("[compare] outbreak")
    tasks.append({"id": "outbreak", "title": TITLES["outbreak"], "kind": "recorded", **compare_outbreak()})
    return _rounded({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "resamples": RESAMPLES, "tasks": tasks})


def main() -> None:
    result = build()
    OUT_FILE.write_text(json.dumps(result, indent=2))
    for t in result["tasks"]:
        for m in t.get("metrics", []):
            ci = f"  95% CI [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}] {m['verdict']}" if m["ci"] else ""
            print(f"{t['id']:10s} {m['id']:18s} original {m['original']:.4f}  federated {m['federated']:.4f}{ci}")
    print(f"[compare] written to {OUT_FILE}")


if __name__ == "__main__":
    main()
