"""Flower ServerApp: coordinates the hospitals; sees only the shared models and metric numbers."""
import json

from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbCyclic

from carepath_fl.common.arrays import pack, unpack, unpack_arrays
from carepath_fl.common.identity import NodeRegistry
from carepath_fl.tasks import get_task

app = ServerApp()


def collect_reference(grid: Grid, task, task_name: str) -> dict:
    """One query round to every hospital, merged by the task."""
    registry = NodeRegistry.from_env()
    node_ids = list(grid.get_node_ids())
    if registry:
        registry.check_complete(node_ids)
    content = RecordDict({"config": ConfigRecord({"task": task_name})})
    messages = [Message(content=content, message_type=MessageType.QUERY, dst_node_id=n) for n in node_ids]
    parts, senders = [], []
    for reply in grid.send_and_receive(messages):
        if reply.has_error():
            raise RuntimeError(f"a hospital failed the query round: {reply.error}")
        parts.append(json.loads(reply.content["reply"]["payload"]))
        senders.append(reply.metadata.src_node_id)
    if len(parts) != len(node_ids):
        raise RuntimeError(f"expected {len(node_ids)} query replies, got {len(parts)}")
    if registry:
        registry.check_replies(senders, [None] * len(senders))
    return task.merge_queries(parts)


def stage_caller(grid: Grid, task_name: str):
    """`call(stage, state)` for statistics-based tasks: one TRAIN message per hospital, replies unpacked."""
    registry = NodeRegistry.from_env()
    if registry is None:
        print("[identity] no node registry: hospital identities are not enforced (simulation)")

    def call(stage: str, state: dict) -> list[dict]:
        node_ids = list(grid.get_node_ids())
        if registry:
            registry.check_complete(node_ids)      # every stage pools all hospitals; none may be missing
        content = RecordDict({"config": ConfigRecord(
            {"task": task_name, "stage": stage, "state": json.dumps(state)})})
        messages = [Message(content=content, message_type=MessageType.TRAIN, dst_node_id=n) for n in node_ids]
        parts, senders = [], []
        for reply in grid.send_and_receive(messages):
            if reply.has_error():
                raise RuntimeError(f"a hospital failed stage {stage!r}: {reply.error}")
            parts.append(unpack_arrays(reply.content["arrays"]))
            senders.append(reply.metadata.src_node_id)
        if len(parts) != len(node_ids):
            raise RuntimeError(f"stage {stage!r}: expected {len(node_ids)} replies, got {len(parts)}")
        if registry:                                # a payload that names its hospital must match its sender
            registry.check_replies(senders, [p["hospital"].tobytes().decode() if "hospital" in p else None
                                             for p in parts])
        return parts
    return call


def turn_history(result) -> list[dict]:
    """One entry per turn: which hospital took it and the shared model's metrics on that hospital's held-out
    rows right after (the cyclic strategy evaluates the same hospital it just trained)."""
    history = []
    for turn in sorted(result.train_metrics_clientapp):
        trained = result.train_metrics_clientapp[turn]
        evaluated = result.evaluate_metrics_clientapp.get(turn, {})
        history.append({
            "turn": int(turn),
            "hospital": f"H{int(trained['hospital-number']):03d}" if "hospital-number" in trained else None,
            "metrics": {k: float(v) for k, v in evaluated.items() if k != "num-examples"}})
    return history


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    task_name = str(cfg["task"])
    task = get_task(task_name)

    if hasattr(task, "run_protocol"):        # statistics-based task, no boosting turns
        params, info = task.run_protocol(stage_caller(grid, task_name))
        if cfg.get("save-model", True):
            print(f"[export] federated {task_name} model written to {task.write_model(params, info)}")
        return

    # 0 = use the task's own default
    turns = int(cfg["num-server-rounds"]) or task.DEFAULT_TURNS
    local_iters = int(cfg["local-iters"]) or task.DEFAULT_LOCAL_ITERS

    registry = NodeRegistry.from_env()
    if registry:
        registry.check_connected(grid.get_node_ids())
    reference = collect_reference(grid, task, task_name) if hasattr(task, "local_query") else None

    # Each model is created by the first hospital's turn; the server starts from empty ones.
    strategy = FedXgbCyclic(fraction_train=1.0, fraction_evaluate=1.0)
    result = strategy.start(
        grid=grid, initial_arrays=pack({name: b"" for name in task.MODELS}), num_rounds=turns,
        train_config=ConfigRecord({"task": task_name, "local-iters": local_iters}),
        evaluate_config=ConfigRecord({"task": task_name}))

    if cfg.get("save-model", True):
        out = task.write_model(unpack(result.arrays), {"turns": turns, "local_iters": local_iters,
                                                       "history": turn_history(result)}, reference)
        print(f"[export] federated {task_name} model written to {out}")
