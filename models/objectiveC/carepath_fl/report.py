"""
Per-hospital report for a federated model.

    python -m carepath_fl.report --task disease|treatment [--model <dir>]

Scores the federated model on each hospital's own held-out test rows and compares it with that
hospital training alone. The centralised baseline pools every hospital's rows, which is only possible
in simulation; it is an upper bound for reference, not something a real deployment can compute.
Writes report.json next to the model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from carepath_fl.common.partition import hospital_ids, load_partition
from carepath_fl.tasks import NAMES, get_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=NAMES, default="disease")
    ap.add_argument("--model", default=None, help="federated model directory (default: the task's OUT_DIR)")
    args = ap.parse_args()
    task = get_task(args.task)
    model_dir = Path(args.model) if args.model else task.OUT_DIR
    if hasattr(task, "report"):      # cross-site tasks (outbreak) have no per-hospital comparison
        task.report(model_dir)
        return

    federated = task.load_models(model_dir)
    hosp = hospital_ids()
    train = {h: load_partition(h, "train", columns=task.COLUMNS) for h in hosp}
    test = {h: load_partition(h, "test", columns=getattr(task, "EVAL_COLUMNS", task.COLUMNS)) for h in hosp}
    central = task.fit_baseline(pd.concat(train.values()))

    rows = []
    for h in hosp:
        local = task.fit_baseline(train[h])
        row = {"hospital": h, "test_rows": len(test[h])}
        for name, models in (("federated", federated), ("local_only", local), ("centralized_sim", central)):
            for k, v in task.score_models(models, test[h]).items():
                row[f"{name}.{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows).set_index("hospital")

    show = [f"{v}.{m}" for m, _ in task.HEADLINE for v in ("federated", "local_only", "centralized_sim")]
    print(df[show].round(4).to_string())
    w = df["test_rows"]
    summary = {c: float(np.average(df[c], weights=w)) for c in df.columns if c != "test_rows"}
    print("\nweighted mean over hospitals:")
    for c in show:
        print(f"  {c:32s} {summary[c]:.4f}")

    wins = {}
    for metric, higher in task.HEADLINE:
        if higher is None:          # shown for context (e.g. a calibration target), not ranked
            continue
        better = df[f"federated.{metric}"] > df[f"local_only.{metric}"] if higher \
            else df[f"federated.{metric}"] < df[f"local_only.{metric}"]
        wins[metric] = int(better.sum())
        print(f"federated beats local-only on {metric} at {wins[metric]}/{len(hosp)} hospitals")

    (model_dir / "report.json").write_text(json.dumps(
        {"task": args.task, "per_hospital": df.round(5).reset_index().to_dict("records"),
         "weighted_mean": {k: round(v, 5) for k, v in summary.items()}, "hospital_wins_vs_local": wins}, indent=2))
    print(f"[report] written to {model_dir / 'report.json'}")


if __name__ == "__main__":
    main()
