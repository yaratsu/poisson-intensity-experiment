from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .utils import ensure_dir


def array_hash(array: np.ndarray, max_bytes: int = 1_000_000) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    view = arr.view(np.uint8)
    if view.nbytes <= max_bytes:
        h.update(view)
    else:
        h.update(view[: max_bytes // 2])
        h.update(view[-max_bytes // 2 :])
        h.update(str(view.nbytes).encode("utf-8"))
    return h.hexdigest()


def stable_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


class DistanceCache:
    """Small ndarray cache for bandwidth-independent distance computations."""

    def __init__(self, cache_dir: str | Path = "cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir) / "distances"
        self.enabled = bool(enabled)
        self.memory: dict[str, np.ndarray] = {}
        self.stats = {"hits": 0, "misses": 0, "seconds": 0.0}

    def get_or_compute(
        self,
        key: dict[str, Any] | str,
        compute_fn: Callable[[], np.ndarray],
        persist: bool = False,
    ) -> np.ndarray:
        if not self.enabled:
            start = time.perf_counter()
            value = compute_fn()
            self.stats["seconds"] += time.perf_counter() - start
            return value

        key_str = key if isinstance(key, str) else stable_key(key)
        if key_str in self.memory:
            self.stats["hits"] += 1
            return self.memory[key_str]

        path = self.cache_dir / f"{key_str}.npz"
        if persist and path.exists():
            self.stats["hits"] += 1
            value = np.load(path)["value"]
            self.memory[key_str] = value
            return value

        self.stats["misses"] += 1
        start = time.perf_counter()
        value = compute_fn()
        self.stats["seconds"] += time.perf_counter() - start
        self.memory[key_str] = value
        if persist:
            ensure_dir(path.parent)
            np.savez_compressed(path, value=value)
        return value
