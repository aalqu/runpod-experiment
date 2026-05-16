import csv
from pathlib import Path

import numpy as np


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialise_value(value):
    if isinstance(value, str):
        return np.array(value)
    return np.asarray(value)


def save_run_result(path, result):
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **{k: _serialise_value(v) for k, v in result.items()})
    return path


def load_run_result(path):
    loaded = np.load(Path(path), allow_pickle=True)
    result = {}
    for key in loaded.files:
        value = loaded[key]
        if value.ndim == 0:
            result[key] = value.item()
        else:
            result[key] = value
    return result


def save_summary_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", newline="") as handle:
            handle.write("")
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
