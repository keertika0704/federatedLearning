"""Flower ClientApp: one instance per hospital. Raw rows never leave this process."""
import json

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from carepath_fl.common.arrays import pack, pack_arrays, unpack
from carepath_fl.common.partition import hospital_ids, load_partition
from carepath_fl.tasks import get_task

app = ClientApp()


def _hospital(context: Context) -> str:
    return hospital_ids()[int(context.node_config["partition-id"])]


def _rows(context: Context, task, split: str, evaluation: bool = False):
    hospital = _hospital(context)
    columns = getattr(task, "EVAL_COLUMNS", task.COLUMNS) if evaluation else task.COLUMNS
    return load_partition(hospital, split, columns=columns)


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = msg.content["config"]
    task = get_task(cfg["task"])
    if hasattr(task, "client_stage"):        # statistics-based task: return summaries for this stage
        rows = _rows(context, task, getattr(task, "STAGE_SPLIT", "train"))
        stats = task.client_stage(str(cfg["stage"]), json.loads(cfg["state"]), rows)
        return Message(content=RecordDict({"arrays": pack_arrays(stats)}), reply_to=msg)
    df = task.fit_rows(_rows(context, task, "train"))

    # Different seed per turn, so subsampling differs between a hospital's turns.
    models, n = task.local_train(unpack(msg.content["arrays"]), df, int(cfg["local-iters"]),
                                 seed=int(cfg["server-round"]))

    # hospital-number only labels whose turn it was in the recorded training history (an id, not data)
    metrics = MetricRecord({"num-examples": n, "hospital-number": int(_hospital(context)[1:])})
    return Message(content=RecordDict({"arrays": pack(models), "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    task = get_task(msg.content["config"]["task"])
    df = task.eval_rows(_rows(context, task, task.EVAL_SPLIT, evaluation=True))
    metrics = task.local_evaluate(unpack(msg.content["arrays"]), df)
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)


@app.query()
def query(msg: Message, context: Context) -> Message:
    """Non-patient aggregates the exported model needs (e.g. which treatment combinations exist)."""
    task = get_task(msg.content["config"]["task"])
    df = task.fit_rows(_rows(context, task, "train"))
    payload = json.dumps(task.local_query(df))
    return Message(content=RecordDict({"reply": ConfigRecord({"payload": payload})}), reply_to=msg)
