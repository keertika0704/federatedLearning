"""
Federated tasks. Each task module provides the client-side logic for one model family:

    MODELS                names of the boosters it trains (keys of the shared ArrayRecord)
    COLUMNS               the only columns a client reads from its data
    EVAL_SPLIT            which split `local_evaluate` runs on during training rounds
    DEFAULT_TURNS / DEFAULT_LOCAL_ITERS
    OUT_DIR               where the federated model is written
    fit_rows(df), eval_rows(df)
    local_train(models, df, local_iters, seed) -> (models, n_rows)
    local_evaluate(models, df) -> metrics
    write_model(models, extra_meta, reference) -> out_dir
    optional: local_query(df) / merge_queries(parts) -> reference   (non-patient aggregates the
              exported model needs, collected in one extra round before training)
    for carepath_fl.report: load_models, fit_baseline, score_models, HEADLINE

Statistics-based tasks (no boosters) instead provide client_stage(stage, state, df) -> {name: ndarray}
and run_protocol(call) -> (params, info); the server drives their stages one Flower round each.
"""
import importlib

NAMES = ("disease", "treatment", "atypical", "outbreak")


def get_task(name: str):
    if name not in NAMES:
        raise ValueError(f"unknown task {name!r}; expected one of {NAMES}")
    return importlib.import_module(f"carepath_fl.tasks.{name}")
