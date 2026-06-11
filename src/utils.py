from __future__ import annotations

import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np


N_VALUES = [100, 316, 1000, 3162, 10000]
Z_DIMS = [0, 1, 5, 10]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_seed(*parts: Any, base_seed: int = 1729) -> int:
    text = "|".join(str(part) for part in parts)
    total = base_seed
    for char in text:
        total = (total * 131 + ord(char)) % (2**32 - 1)
    return int(total)


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(dict(out[key]), value)
        elif value is not None:
            out[key] = value
    return out


def support_metadata(support: str) -> dict[str, Any]:
    if support == "euclidean1d":
        return {
            "support": support,
            "support_type": "euclidean",
            "event_dim": 1,
            "coord_dim": 1,
            "embedding_dim": 1,
            "manifold_type": None,
            "manifold_dim": None,
            "volume": 1.0,
        }
    if support == "euclidean2d":
        return {
            "support": support,
            "support_type": "euclidean",
            "event_dim": 2,
            "coord_dim": 2,
            "embedding_dim": 2,
            "manifold_type": None,
            "manifold_dim": None,
            "volume": 1.0,
        }
    if support == "circle":
        return {
            "support": support,
            "support_type": "manifold",
            "event_dim": 2,
            "coord_dim": 1,
            "embedding_dim": 2,
            "manifold_type": "circle",
            "manifold_dim": 1,
            "volume": 2.0 * math.pi,
        }
    if support == "sphere":
        return {
            "support": support,
            "support_type": "manifold",
            "event_dim": 3,
            "coord_dim": 2,
            "embedding_dim": 3,
            "manifold_type": "sphere",
            "manifold_dim": 2,
            "volume": 4.0 * math.pi,
        }
    raise ValueError(f"Unknown support: {support}")


@contextmanager
def timer() -> Any:
    start = time.perf_counter()
    box = {"seconds": None}
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start


def inverse_softplus(y: float) -> float:
    if y <= 0:
        return -30.0
    if y > 20:
        return y
    return math.log(math.expm1(y))


def chunk_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))
