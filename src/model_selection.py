from __future__ import annotations

import shutil
import math
from pathlib import Path
from typing import Any

from .cache_utils import stable_key
from .utils import ensure_dir, iter_metric_json_paths, load_json, save_json


SETTING_KEYS = [
    "scenario",
    "support",
    "support_type",
    "event_dim",
    "manifold_type",
    "z_dim",
    "n",
    "method",
    "output_activation",
    "dnn_architecture_tag",
    "manifold_learning",
    "manifold_input",
    "expected_count",
    "epsilon",
    "alpha",
    "beta",
    "covariate_sampler",
    "bandwidth_selection",
    "bandwidth_theory_mode",
    "bandwidth_cv_folds",
    "bandwidth_search",
    "bandwidth_scale_x",
    "bandwidth_scale_z",
    "bandwidth_grid_size",
    "bandwidth_fine_grid_size",
    "kernel_cv_max_replicates",
    "kernel_cv_max_events_per_replicate",
    "validation_z_chunk_size",
    "validation_quadrature_method",
    "kernel",
    "boundary_correction",
    "gaussian_cutoff",
    "use_gaussian_cutoff",
    "mesh_resolution",
    "correction_type",
]


def setting_hash_from_row(row: dict[str, Any]) -> str:
    payload = {key: row.get(key) for key in SETTING_KEYS if key in row}
    return stable_key(payload)


def collect_metric_rows(results_dir: str | Path, glob_pattern: str = "*.json") -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in iter_metric_json_paths(results_dir, glob_pattern=glob_pattern):
        try:
            row = load_json(path)
        except Exception:
            continue
        if "squared_hellinger" in row:
            rows.append((path, row))
    return rows


def select_best_models(
    results_dir: str | Path = "results",
    keep_all_models: bool = False,
    required_repetitions: set[int] | None = None,
    setting_hashes: set[str] | None = None,
    metric_glob: str = "*.json",
) -> list[dict[str, Any]]:
    results_dir = Path(results_dir)
    rows = collect_metric_rows(results_dir, glob_pattern=metric_glob)
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, row in rows:
        policy = str(row.get("save_model_policy", "best_repeat"))
        if policy != "best_repeat":
            continue
        setting_hash = row.get("setting_hash") or setting_hash_from_row(row)
        row["setting_hash"] = setting_hash
        if setting_hashes is not None and setting_hash not in setting_hashes:
            continue
        groups.setdefault(setting_hash, []).append((path, row))

    selected_metadata = []
    for setting_hash, members in groups.items():
        valid = [item for item in members if _is_number(item[1].get("squared_hellinger"))]
        if not valid:
            continue
        best_path, best_row = min(valid, key=lambda item: float(item[1]["squared_hellinger"]))
        best_rep = int(best_row.get("repetition", -1))
        suffix = ".pt" if str(best_row.get("model_suffix", "")).endswith(".pt") else ".pkl"
        filename = "best_model.pt" if suffix == ".pt" else "best_estimator.pkl"
        best_dir = ensure_dir(results_dir / "models" / "best" / setting_hash)
        best_model_path = best_dir / filename
        source_results_dir = best_path.parent.parent
        tmp_path = _resolve_saved_model_path(best_row.get("model_path_tmp"), source_results_dir, results_dir)
        if tmp_path.exists():
            shutil.copy2(tmp_path, best_model_path)
        else:
            model_path = _resolve_saved_model_path(best_row.get("model_path"), source_results_dir, results_dir)
            if model_path.exists():
                shutil.copy2(model_path, best_model_path)

        all_metrics = []
        for _, row in sorted(members, key=lambda item: int(item[1].get("repetition", -1))):
            all_metrics.append(
                {
                    "repetition": int(row.get("repetition", -1)),
                    "squared_hellinger": float(row.get("squared_hellinger", "nan")),
                    "model_path_tmp": row.get("model_path_tmp"),
                }
            )
        metadata = {
            "setting_hash": setting_hash,
            "best_repetition": best_rep,
            "best_squared_hellinger": float(best_row["squared_hellinger"]),
            "all_repetition_metrics": all_metrics,
            "method": best_row.get("method"),
            "scenario": best_row.get("scenario"),
            "support": best_row.get("support"),
            "z_dim": best_row.get("z_dim"),
            "n": best_row.get("n"),
            "output_activation": best_row.get("output_activation"),
            "selected_bandwidths": {
                "h_x": best_row.get("bandwidth_x"),
                "h_z": best_row.get("bandwidth_z"),
                "h_m": best_row.get("bandwidth_manifold"),
            },
            "config": best_row.get("config", {}),
            "best_model_path": str(best_model_path),
            "metric_paths": [str(path) for path, _ in members],
        }
        save_json(metadata, best_dir / "best_model_metadata.json")
        selected_metadata.append(metadata)

        present_reps = {int(row.get("repetition", -999)) for _, row in members}
        cleanup_allowed = required_repetitions is None or required_repetitions.issubset(present_reps)
        for path, row in members:
            rep = int(row.get("repetition", -1))
            row["setting_hash"] = setting_hash
            row["model_saved"] = bool(rep == best_rep and best_model_path.exists())
            row["best_repeat"] = best_rep
            row["best_model_path"] = str(best_model_path) if best_model_path.exists() else None
            save_json(row, path)
            tmp = _resolve_saved_model_path(row.get("model_path_tmp"), path.parent.parent, results_dir)
            if cleanup_allowed and not keep_all_models and tmp.exists():
                tmp.unlink()
    return selected_metadata


def _resolve_saved_model_path(value: Any, source_results_dir: Path, target_results_dir: Path) -> Path:
    if value in {None, ""}:
        return Path("__missing_model_path__")
    raw = Path(str(value))
    if raw.exists():
        return raw

    candidates: list[Path] = []
    parts = raw.parts
    if "results" in parts:
        rel_after_results = Path(*parts[parts.index("results") + 1 :])
        candidates.extend([source_results_dir / rel_after_results, target_results_dir / rel_after_results])
    if not raw.is_absolute():
        candidates.extend([source_results_dir / raw, target_results_dir / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
