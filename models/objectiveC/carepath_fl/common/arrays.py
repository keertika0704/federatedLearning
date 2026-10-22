"""Flower ArrayRecord helpers: boosters (one uint8 array of XGBoost JSON bytes each) and named numeric arrays."""
import numpy as np
from flwr.app import Array, ArrayRecord


def pack(models: dict[str, bytes]) -> ArrayRecord:
    return ArrayRecord(array_dict={
        name: Array.from_numpy_ndarray(np.frombuffer(raw, dtype=np.uint8)) for name, raw in models.items()})


def unpack(record: ArrayRecord) -> dict[str, bytes]:
    return {name: record[name].numpy().tobytes() for name in record.keys()}


def pack_arrays(arrays: dict[str, np.ndarray]) -> ArrayRecord:
    """Named numeric arrays (sums, counts, histograms) -> ArrayRecord."""
    return ArrayRecord(array_dict={name: Array.from_numpy_ndarray(np.asarray(a)) for name, a in arrays.items()})


def unpack_arrays(record: ArrayRecord) -> dict[str, np.ndarray]:
    return {name: record[name].numpy() for name in record.keys()}
