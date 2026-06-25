from __future__ import annotations

import json
import math
import random
import time
import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np


N_VALUES = [100, 316, 1000, 3162, 10000]
Z_DIMS = [0, 1, 5, 10]
CSV_EXCLUDE_KEYS = {"config", "kernel_profile"}
METRIC_DEDUP_KEYS = ["scenario", "support", "z_dim", "n", "repetition", "method", "seed"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int, seed_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if not seed_torch:
        return
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


def csv_cell(value: Any) -> Any:
    value = to_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_dicts_csv(rows: list[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            key = str(key)
            if key in CSV_EXCLUDE_KEYS:
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def update_dicts_csv_dedup(path: str | Path, row: Mapping[str, Any], key_fields: list[str]) -> None:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    row_csv = {str(key): csv_cell(value) for key, value in row.items() if str(key) not in CSV_EXCLUDE_KEYS}
    new_key = tuple(str(row_csv.get(key, "")) for key in key_fields)
    kept = []
    for existing in rows:
        existing_key = tuple(str(existing.get(key, "")) for key in key_fields)
        if existing_key != new_key:
            kept.append(existing)
    kept.append(row_csv)
    write_dicts_csv(kept, path)


def iter_metric_json_paths(results_dir: str | Path, glob_pattern: str = "*.json"):
    results_dir = Path(results_dir)
    metrics_dirs = []
    top_metrics_dir = results_dir / "metrics"
    if top_metrics_dir.exists():
        metrics_dirs.append(top_metrics_dir)
    if results_dir.exists():
        metrics_dirs.extend(
            path
            for path in results_dir.rglob("metrics")
            if path.is_dir() and path != top_metrics_dir
        )
    seen_dirs: set[Path] = set()
    for metrics_dir in sorted(metrics_dirs, key=lambda path: (len(path.parts), str(path))):
        resolved = metrics_dir.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        for path in sorted(metrics_dir.glob(glob_pattern)):
            if path.name.startswith("history_"):
                continue
            yield path


def collect_metric_json_csv(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    metrics_dir = ensure_dir(results_dir / "metrics")
    deduped_rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in iter_metric_json_paths(results_dir):
        try:
            row = load_json(path)
        except Exception:
            continue
        if "squared_hellinger" in row:
            row = dict(row)
            row["metric_source_path"] = str(path)
            key = tuple(str(csv_cell(row.get(field))) for field in METRIC_DEDUP_KEYS)
            deduped_rows.setdefault(key, row)
    rows = list(deduped_rows.values())
    if rows:
        write_dicts_csv(rows, metrics_dir / "all_metrics.csv")


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
