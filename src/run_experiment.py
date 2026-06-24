from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .covariates import make_covariate_sampler
from .evaluation import squared_hellinger
from .intensities import make_true_intensity
from .model_selection import select_best_models, setting_hash_from_row
from .simulate import simulate_dataset
from .utils import deep_update, ensure_dir, load_json, load_yaml, save_json, set_random_seed, stable_seed, update_dicts_csv_dedup, write_dicts_csv


def build_estimator(method: str, config: dict[str, Any]) -> Any:
    method = method.lower()
    if method in {"dnn", "dnn_npmle"}:
        from .methods.dnn_npmle import DNNNPMLEEstimator

        return DNNNPMLEEstimator(config.get("dnn", {}))
    if method in {"kernel_euclidean", "euclidean_kernel"}:
        from .methods.kernel_euclidean import EuclideanKernelEstimator

        estimator_config = dict(config.get("kernel_euclidean", {}))
        estimator_config["mode"] = "spatial_only"
        return EuclideanKernelEstimator(estimator_config)
    if method in {"kernel_covariate", "covariate_kernel"}:
        from .methods.kernel_covariate import CovariateKernelEstimator

        return CovariateKernelEstimator(config.get("kernel_covariate", {}))
    if method in {"kernel_manifold", "manifold_kernel"}:
        from .methods.kernel_manifold import ManifoldKernelEstimator

        return ManifoldKernelEstimator(config.get("kernel_manifold", {}))
    raise ValueError(f"Unknown method: {method}")


def method_label(
    method: str,
    output_activation: str | None = None,
    architecture: str | None = None,
    manifold_learning: str | None = None,
) -> str:
    method = method.lower()
    if method in {"dnn", "dnn_npmle"}:
        label = f"dnn_npmle_{output_activation or 'softplus'}"
        if architecture:
            label = f"{label}_{architecture}"
        if manifold_learning:
            label = f"{label}_{manifold_learning}"
        return label
    return method


def _tag_number(value: Any) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def dnn_architecture_label(config: dict[str, Any]) -> str:
    architecture = str(config.get("architecture", "theory")).lower()
    if architecture in {"theory", "adaptive", "auto"}:
        return (
            f"{architecture}"
            f"_ref{_tag_number(config.get('architecture_reference_n', 1000))}"
            f"_ds{_tag_number(config.get('depth_scale', 1.0))}"
            f"_ws{_tag_number(config.get('width_scale', 64.0))}"
        )
    widths = config.get("hidden_layers") or []
    width_tag = "x".join(str(int(width)) for width in widths) if widths else "custom"
    return f"{architecture}_{width_tag}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one conditional Poisson intensity experiment.")
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--scenario", default=None, choices=["compositional", "near_zero", "manifold"])
    parser.add_argument("--support", default=None, choices=["euclidean1d", "euclidean2d", "circle", "sphere"])
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--output-activation", choices=["softplus", "relu"], default=None)
    parser.add_argument("--dnn-architecture", choices=["theory", "adaptive", "auto", "fixed", "manual"], default=None)
    parser.add_argument("--hidden-layers", default=None, help="Comma- or space-separated hidden widths for fixed/manual DNN architecture.")
    parser.add_argument("--architecture-reference-n", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--width-scale", type=float, default=None)
    parser.add_argument("--width-multiple", type=int, default=None)
    parser.add_argument("--architecture-rate-exponent", type=float, default=None)
    parser.add_argument("--manifold-learning", choices=["agnostic", "oracle"], default=None)
    parser.add_argument("--manifold-input", choices=["intrinsic", "embedded"], default=None)
    parser.add_argument("--repetition", type=int, default=None)
    parser.add_argument("--expected-count", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, choices=["float32", "float64"])
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--integration-points", type=int, default=None)
    parser.add_argument("--eval-z", type=int, default=None)
    parser.add_argument("--eval-points", type=int, default=None)
    parser.add_argument("--kernel-chunk-size", type=int, default=None)
    parser.add_argument("--eval-chunk-size", type=int, default=None)
    parser.add_argument("--gaussian-cutoff", type=float, default=None)
    parser.add_argument("--no-gaussian-cutoff", action="store_true")
    parser.add_argument("--kernel-type", choices=["gaussian", "epanechnikov"], default=None)
    parser.add_argument("--boundary-correction", choices=["none", "renormalize"], default=None)
    parser.add_argument("--bandwidth-search", choices=["coarse_to_fine", "full"], default=None)
    parser.add_argument("--bandwidth-selection", choices=["theory_guided_5fold_cv", "poisson_5fold_cv", "validation_nll", "ward_cv", "ward_np"], default=None)
    parser.add_argument("--bandwidth-cv-folds", type=int, default=None)
    parser.add_argument("--bandwidth-theory-mode", choices=["separate_spatial_covariate", "klutchnikoff_joint", "ward_geodesic_grid"], default=None)
    parser.add_argument("--bandwidth-scale-x", type=float, default=None)
    parser.add_argument("--bandwidth-scale-z", type=float, default=None)
    parser.add_argument("--bandwidth-grid-size", type=int, default=None)
    parser.add_argument("--bandwidth-fine-grid-size", type=int, default=None)
    parser.add_argument("--kernel-cv-max-replicates", type=int, default=None)
    parser.add_argument("--kernel-cv-max-events-per-replicate", type=int, default=None)
    parser.add_argument("--validation-z-chunk-size", type=int, default=None)
    parser.add_argument("--validation-quadrature-method", choices=["uniform", "sobol"], default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--use-distance-cache", dest="use_distance_cache", action="store_true", default=None)
    parser.add_argument("--no-distance-cache", dest="use_distance_cache", action="store_false")
    parser.add_argument("--single-realization-batch-size", type=int, default=None)
    parser.add_argument("--mesh-resolution", type=int, default=None)
    parser.add_argument("--correction-type", choices=["none", "global", "local"], default=None)
    parser.add_argument("--save-model-policy", choices=["best_repeat", "all", "none"], default=None)
    parser.add_argument("--keep-all-models", action="store_true")
    parser.add_argument(
        "--plot-after-repetitions",
        type=int,
        default=None,
        help="For best_repeat model saving, wait until this many repetitions are present before drawing intensity plots.",
    )
    parser.add_argument("--covariate-sampler", default=None, choices=["uniform", "beta", "gaussian_copula"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def parse_hidden_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    pieces = value.replace(",", " ").split()
    if not pieces:
        return None
    widths = [int(piece) for piece in pieces]
    if any(width <= 0 for width in widths):
        raise ValueError("--hidden-layers must contain positive integers")
    return widths


def assemble_config(args: argparse.Namespace) -> dict[str, Any]:
    base = load_yaml("configs/default.yaml") if Path("configs/default.yaml").exists() else {}
    cfg = deep_update(base, load_yaml(args.config))
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    cli = {
        "experiment": {
            "scenario": args.scenario,
            "support": args.support,
            "z_dim": args.z_dim,
            "n": args.n,
            "method": args.method,
            "repetition": args.repetition,
            "expected_count": args.expected_count,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "beta": args.beta,
            "seed": args.seed,
            "eval_z": args.eval_z,
            "eval_points": args.eval_points,
            "eval_chunk_size": args.eval_chunk_size,
            "cache_dir": args.cache_dir,
            "mesh_resolution": args.mesh_resolution,
            "save_model_policy": args.save_model_policy,
            "keep_all_models": args.keep_all_models if args.keep_all_models else None,
            "plot_after_repetitions": args.plot_after_repetitions,
            "covariate_sampler": args.covariate_sampler,
            "results_dir": args.results_dir,
        },
        "dnn": {
            "output_activation": args.output_activation,
            "architecture": args.dnn_architecture,
            "hidden_layers": hidden_layers,
            "architecture_reference_n": args.architecture_reference_n,
            "depth_scale": args.depth_scale,
            "width_scale": args.width_scale,
            "width_multiple": args.width_multiple,
            "architecture_rate_exponent": args.architecture_rate_exponent,
            "manifold_learning": args.manifold_learning,
            "manifold_input": args.manifold_input,
            "device": args.device,
            "dtype": args.dtype,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "integration_points": args.integration_points,
        },
    }
    kernel_cli = {
        "kernel": args.kernel_type,
        "boundary_correction": args.boundary_correction,
        "kernel_chunk_size": args.kernel_chunk_size,
        "gaussian_cutoff": args.gaussian_cutoff,
        "use_gaussian_cutoff": False if args.no_gaussian_cutoff else None,
        "bandwidth_selection": args.bandwidth_selection,
        "bandwidth_cv_folds": args.bandwidth_cv_folds,
        "bandwidth_theory_mode": args.bandwidth_theory_mode,
        "bandwidth_scale_x": args.bandwidth_scale_x,
        "bandwidth_scale_z": args.bandwidth_scale_z,
        "bandwidth_search": args.bandwidth_search,
        "bandwidth_grid_size": args.bandwidth_grid_size,
        "bandwidth_fine_grid_size": args.bandwidth_fine_grid_size,
        "kernel_cv_max_replicates": args.kernel_cv_max_replicates,
        "kernel_cv_max_events_per_replicate": args.kernel_cv_max_events_per_replicate,
        "validation_z_chunk_size": args.validation_z_chunk_size,
        "validation_quadrature_method": args.validation_quadrature_method,
        "cache_dir": args.cache_dir,
        "use_distance_cache": args.use_distance_cache,
        "single_realization_batch_size": args.single_realization_batch_size,
        "mesh_resolution": args.mesh_resolution,
        "correction_type": args.correction_type,
    }
    cli["kernel_euclidean"] = kernel_cli
    cli["kernel_covariate"] = kernel_cli
    cli["kernel_manifold"] = kernel_cli
    cfg = deep_update(cfg, cli)
    cfg.setdefault("experiment", {})
    cfg.setdefault("dnn", {})
    cfg.setdefault("kernel_euclidean", {})
    cfg.setdefault("kernel_covariate", {})
    cfg.setdefault("kernel_manifold", {})
    exp = cfg["experiment"]
    defaults = {
        "scenario": "compositional",
        "support": "euclidean1d",
        "z_dim": 0,
        "n": 100,
        "method": "dnn_npmle",
        "repetition": 0,
        "expected_count": 30.0,
        "epsilon": 1e-4,
        "beta": 2.0,
        "seed": None,
        "eval_z": 64,
        "eval_points": 512,
        "eval_chunk_size": 4096,
        "cache_dir": "cache",
        "mesh_resolution": 64,
        "save_model_policy": "best_repeat",
        "keep_all_models": False,
        "plot_after_repetitions": 1,
        "covariate_sampler": "uniform",
        "results_dir": "results",
    }
    for key, value in defaults.items():
        exp.setdefault(key, value)
    if hidden_layers is not None and args.dnn_architecture is None:
        cfg["dnn"]["architecture"] = "fixed"
    if args.manifold_learning == "oracle" and args.manifold_input is None:
        cfg["dnn"]["manifold_input"] = "intrinsic"
    cfg["dnn"].setdefault("expected_count", exp["expected_count"])
    cfg["dnn"].setdefault("seed", exp.get("seed") or 0)
    return cfg


def main(argv: list[str] | None = None) -> None:
    main_start = time.perf_counter()
    args = parse_args(argv)
    cfg = assemble_config(args)
    exp = cfg["experiment"]
    scenario = exp["scenario"]
    support = exp["support"]
    z_dim = int(exp["z_dim"])
    n = int(exp["n"])
    method = exp["method"]
    repetition = int(exp["repetition"])
    expected_count = float(exp["expected_count"])
    seed = exp.get("seed")
    if seed is None:
        seed = stable_seed(scenario, support, z_dim, n, method, repetition)
    seed = int(seed)
    is_dnn_method = method.lower() in {"dnn", "dnn_npmle"}
    set_random_seed(seed, seed_torch=is_dnn_method)
    for section in ["dnn", "kernel_euclidean", "kernel_covariate", "kernel_manifold"]:
        cfg[section]["seed"] = seed
    cfg["dnn"]["expected_count"] = expected_count

    rng = np.random.default_rng(seed)
    simulate_start = time.perf_counter()
    cov_sampler = make_covariate_sampler(exp.get("covariate_sampler", "uniform"))
    true_intensity = make_true_intensity(
        scenario=scenario,
        support=support,
        z_dim=z_dim,
        expected_count=expected_count,
        epsilon=float(exp.get("epsilon", 1e-4)),
        beta=float(exp.get("beta", 2.0)),
        alpha=exp.get("alpha"),
    )
    true_intensity.metadata.update({"scenario": scenario})
    data = simulate_dataset(true_intensity, n=n, z_dim=z_dim, rng=rng, covariate_sampler=cov_sampler)
    simulate_runtime = time.perf_counter() - simulate_start
    data["metadata"].update(
        {
            "scenario": scenario,
            "beta": float(true_intensity.beta),
            "alpha": float(true_intensity.alpha),
            "theory_rate_exponent": float(true_intensity.theory_rate_exponent(z_dim)),
        }
    )
    cfg["dnn"].update(
        {
            "scenario": scenario,
            "support": support,
            "beta": float(true_intensity.beta),
            "alpha": float(true_intensity.alpha),
        }
    )
    estimator = build_estimator(method, cfg)
    if method.lower() in {"dnn", "dnn_npmle"}:
        cfg["dnn"].setdefault("output_activation", "softplus")
        estimator.config.update(cfg["dnn"])

    start = time.perf_counter()
    estimator.fit(data)
    fit_runtime = time.perf_counter() - start

    metric_rng = np.random.default_rng(seed + 2024)
    hellinger_start = time.perf_counter()
    h2 = squared_hellinger(
        estimator,
        true_intensity,
        z_dim=z_dim,
        rng=metric_rng,
        n_eval_z=int(exp.get("eval_z", 64)),
        n_eval_points=int(exp.get("eval_points", 512)),
        covariate_sampler=cov_sampler,
        eval_chunk_size=int(exp.get("eval_chunk_size", 4096)),
        cache_dir=exp.get("cache_dir"),
        cache_seed=seed + 2024,
        mesh_resolution=exp.get("mesh_resolution"),
    )
    hellinger_runtime = time.perf_counter() - hellinger_start

    results_dir = Path(exp.get("results_dir", "results"))
    metrics_dir = ensure_dir(results_dir / "metrics")
    models_dir = ensure_dir(results_dir / "models")
    metadata = data["metadata"]
    is_dnn = is_dnn_method
    is_manifold = metadata["support_type"] == "manifold"
    architecture_tag = dnn_architecture_label(estimator.config) if is_dnn else None
    label = method_label(
        method,
        cfg["dnn"].get("output_activation") if is_dnn else None,
        architecture_tag,
        cfg["dnn"].get("manifold_learning") if is_dnn and is_manifold else None,
    )
    run_id = f"{scenario}_{support}_z{z_dim}_n{n}_{label}_rep{repetition}_seed{seed}"
    model_suffix = ".pt" if method.lower() in {"dnn", "dnn_npmle"} else ".pkl"
    save_model_policy = str(exp.get("save_model_policy", "best_repeat"))
    keep_all_models = bool(exp.get("keep_all_models", False))
    if hasattr(estimator, "history"):
        save_json({"history": estimator.history}, metrics_dir / f"history_{run_id}.json")

    cv_path = None
    cv_save_runtime = 0.0
    if getattr(estimator, "cv_results", None):
        cv_save_start = time.perf_counter()
        cv_dir = ensure_dir(results_dir / "bandwidth_cv")
        cv_path = cv_dir / f"{scenario}_{support}_zdim{z_dim}_n{n}_{label}_rep{repetition}.csv"
        write_dicts_csv(estimator.cv_results, cv_path)
        cv_save_runtime = time.perf_counter() - cv_save_start

    row = {
        "scenario": scenario,
        "support": support,
        "support_type": metadata["support_type"],
        "event_dim": metadata["event_dim"],
        "manifold_type": metadata["manifold_type"],
        "z_dim": z_dim,
        "n": n,
        "repetition": repetition,
        "method": label,
        "output_activation": cfg["dnn"].get("output_activation") if is_dnn else None,
        "dnn_architecture": estimator.config.get("architecture") if is_dnn else None,
        "dnn_architecture_tag": architecture_tag,
        "dnn_hidden_layers": estimator.config.get("resolved_hidden_layers", estimator.config.get("hidden_layers")) if is_dnn else None,
        "dnn_depth": estimator.config.get("architecture_depth_used") if is_dnn else None,
        "dnn_width": estimator.config.get("architecture_width_used") if is_dnn else None,
        "dnn_architecture_rate_exponent": estimator.config.get("architecture_rate_exponent_used") if is_dnn else None,
        "manifold_learning": cfg["dnn"].get("manifold_learning") if is_dnn and is_manifold else None,
        "manifold_input": cfg["dnn"].get("manifold_input") if is_dnn and is_manifold else None,
        "expected_count": expected_count,
        "epsilon": exp.get("epsilon"),
        "alpha": exp.get("alpha"),
        "beta": exp.get("beta"),
        "covariate_sampler": exp.get("covariate_sampler"),
        "bandwidth_selection": estimator.config.get("bandwidth_selection") if hasattr(estimator, "config") else None,
        "bandwidth_theory_mode": estimator.config.get("bandwidth_theory_mode") if hasattr(estimator, "config") else None,
        "bandwidth_cv_folds": estimator.config.get("bandwidth_cv_folds") if hasattr(estimator, "config") else None,
        "bandwidth_search": estimator.config.get("bandwidth_search") if hasattr(estimator, "config") else None,
        "bandwidth_scale_x": estimator.config.get("bandwidth_scale_x") if hasattr(estimator, "config") else None,
        "bandwidth_scale_z": estimator.config.get("bandwidth_scale_z") if hasattr(estimator, "config") else None,
        "bandwidth_grid_size": estimator.config.get("bandwidth_grid_size") if hasattr(estimator, "config") else None,
        "bandwidth_fine_grid_size": estimator.config.get("bandwidth_fine_grid_size") if hasattr(estimator, "config") else None,
        "kernel_cv_max_replicates": estimator.config.get("kernel_cv_max_replicates") if hasattr(estimator, "config") else None,
        "kernel_cv_max_events_per_replicate": estimator.config.get("kernel_cv_max_events_per_replicate") if hasattr(estimator, "config") else None,
        "validation_z_chunk_size": estimator.config.get("validation_z_chunk_size") if hasattr(estimator, "config") else None,
        "validation_quadrature_method": estimator.config.get("validation_quadrature_method") if hasattr(estimator, "config") else None,
        "bandwidth_x": getattr(estimator, "bandwidth_x", None),
        "bandwidth_z": getattr(estimator, "bandwidth_z", None),
        "bandwidth_manifold": getattr(estimator, "bandwidth_m", None),
        "bandwidth_cv_score": getattr(estimator, "selected_bandwidth_cv_score", None),
        "bandwidth_cv_path": str(cv_path) if cv_path is not None else None,
        "kernel": estimator.config.get("kernel") if hasattr(estimator, "config") else None,
        "boundary_correction": estimator.config.get("boundary_correction") if hasattr(estimator, "config") else None,
        "gaussian_cutoff": estimator.config.get("gaussian_cutoff") if hasattr(estimator, "config") else None,
        "use_gaussian_cutoff": estimator.config.get("use_gaussian_cutoff") if hasattr(estimator, "config") else None,
        "mesh_resolution": estimator.config.get("mesh_resolution") if hasattr(estimator, "config") else exp.get("mesh_resolution"),
        "correction_type": estimator.config.get("correction_type") if hasattr(estimator, "config") else None,
        "save_model_policy": save_model_policy,
        "plot_after_repetitions": exp.get("plot_after_repetitions"),
        "model_saved": False,
        "best_repeat": None,
        "best_model_path": None,
        "model_suffix": model_suffix,
        "config": cfg,
        "squared_hellinger": h2,
        "theory_rate_exponent": true_intensity.theory_rate_exponent(z_dim),
        "seed": seed,
        "runtime_seconds": None,
        "fit_runtime_seconds": fit_runtime,
        "simulate_runtime_seconds": simulate_runtime,
        "hellinger_runtime_seconds": hellinger_runtime,
        "plot_runtime_seconds": 0.0,
        "best_intensity_plot_dir": None,
        "best_intensity_plot_repetition": None,
        "best_intensity_plot_metadata": None,
        "best_intensity_plot_files": None,
        "intensity_plot_created_this_run": False,
        "cv_save_runtime_seconds": cv_save_runtime,
        "model_save_runtime_seconds": 0.0,
        "metrics_write_runtime_seconds": 0.0,
        "model_selection_runtime_seconds": 0.0,
        "metrics_update_runtime_seconds": 0.0,
        "postprocess_runtime_seconds": 0.0,
        "kernel_profile": getattr(estimator, "profile", None),
        "mean_observed_count": metadata["mean_observed_count"],
        "total_events": metadata["total_events"],
    }
    setting_hash = setting_hash_from_row(row)
    row["setting_hash"] = setting_hash
    if save_model_policy == "all":
        model_save_start = time.perf_counter()
        model_path = models_dir / f"{run_id}{model_suffix}"
        estimator.save(model_path)
        row["model_save_runtime_seconds"] = time.perf_counter() - model_save_start
        row["model_path"] = str(model_path)
        row["model_saved"] = True
    elif save_model_policy == "best_repeat":
        model_save_start = time.perf_counter()
        tmp_dir = ensure_dir(models_dir / "tmp")
        tmp_path = tmp_dir / f"{setting_hash}_rep{repetition}{model_suffix}"
        estimator.save(tmp_path)
        row["model_save_runtime_seconds"] = time.perf_counter() - model_save_start
        row["model_path_tmp"] = str(tmp_path)
    elif save_model_policy == "none":
        row["model_path"] = None
    else:
        raise ValueError(f"Unknown save_model_policy: {save_model_policy}")

    if save_model_policy != "best_repeat" and not args.no_plots:
        from .plots import plot_intensity

        plot_start = time.perf_counter()
        plot_dir = (
            results_dir
            / "plots"
            / "intensity"
            / scenario
            / support
            / f"zdim_{z_dim}"
            / f"n_{n}"
            / label
        )
        plot_intensity(true_intensity, estimator, plot_dir, z_dim=z_dim)
        row["plot_runtime_seconds"] = time.perf_counter() - plot_start

    row["runtime_seconds"] = time.perf_counter() - main_start
    row["postprocess_runtime_seconds"] = row["runtime_seconds"] - simulate_runtime - fit_runtime - hellinger_runtime - row["plot_runtime_seconds"]
    metrics_path = metrics_dir / f"{run_id}.json"
    metrics_write_start = time.perf_counter()
    save_json(row, metrics_path)
    all_csv = metrics_dir / "all_metrics.csv"
    update_dicts_csv_dedup(all_csv, row, ["scenario", "support", "z_dim", "n", "repetition", "method", "seed"])
    row["metrics_write_runtime_seconds"] = time.perf_counter() - metrics_write_start
    row["runtime_seconds"] = time.perf_counter() - main_start
    row["postprocess_runtime_seconds"] = row["runtime_seconds"] - simulate_runtime - fit_runtime - hellinger_runtime - row["plot_runtime_seconds"]
    save_json(row, metrics_path)
    update_dicts_csv_dedup(all_csv, row, ["scenario", "support", "z_dim", "n", "repetition", "method", "seed"])

    if save_model_policy == "best_repeat":
        model_selection_start = time.perf_counter()
        selected_metadata = select_best_models(
            results_dir,
            keep_all_models=keep_all_models,
            required_repetitions={0, 1, 2},
            setting_hashes={setting_hash},
            metric_glob=f"{scenario}_{support}_z{z_dim}_n{n}_{label}_rep*.json",
        )
        model_selection_runtime = time.perf_counter() - model_selection_start
        metrics_update_start = time.perf_counter()
        for metadata_item in selected_metadata:
            for metric_path in metadata_item.get("metric_paths", []):
                path = Path(metric_path)
                if path.exists():
                    update_dicts_csv_dedup(all_csv, load_json(path), ["scenario", "support", "z_dim", "n", "repetition", "method", "seed"])
        metrics_update_runtime = time.perf_counter() - metrics_update_start

        plot_update: dict[str, Any] = {}
        plot_runtime = 0.0
        if not args.no_plots:
            best_metadata = next(
                (item for item in selected_metadata if item.get("setting_hash") == setting_hash),
                None,
            )
            plot_after_repetitions = max(1, int(exp.get("plot_after_repetitions", 1)))
            present_repetitions = {
                int(item.get("repetition", -999))
                for item in (best_metadata or {}).get("all_repetition_metrics", [])
            }
            if best_metadata is not None and len(present_repetitions) >= plot_after_repetitions:
                from .plots import plot_best_repeat_intensity

                plot_start = time.perf_counter()
                plot_update = plot_best_repeat_intensity(
                    best_metadata,
                    results_dir=results_dir,
                    current_estimator=estimator,
                    current_repetition=repetition,
                    true_intensity=true_intensity,
                )
                if plot_update.get("intensity_plot_created_this_run"):
                    plot_runtime = time.perf_counter() - plot_start

        row = load_json(metrics_path)
        row["model_selection_runtime_seconds"] = model_selection_runtime
        row["metrics_update_runtime_seconds"] = metrics_update_runtime
        if plot_update:
            row.update(plot_update)
            row["plot_runtime_seconds"] = float(row.get("plot_runtime_seconds") or 0.0) + plot_runtime
        row["runtime_seconds"] = time.perf_counter() - main_start
        row["fit_runtime_seconds"] = fit_runtime
        row["simulate_runtime_seconds"] = simulate_runtime
        row["hellinger_runtime_seconds"] = hellinger_runtime
        row["plot_runtime_seconds"] = row.get("plot_runtime_seconds", 0.0)
        row["postprocess_runtime_seconds"] = row["runtime_seconds"] - simulate_runtime - fit_runtime - hellinger_runtime - row["plot_runtime_seconds"]
        save_json(row, metrics_path)
        update_dicts_csv_dedup(all_csv, row, ["scenario", "support", "z_dim", "n", "repetition", "method", "seed"])

    print(f"run_id={run_id}")
    print(f"squared_hellinger={h2:.8g}")
    print(f"runtime_seconds={row['runtime_seconds']:.2f}")
    print(f"fit_runtime_seconds={fit_runtime:.2f}")
    print(f"hellinger_runtime_seconds={hellinger_runtime:.2f}")
    print(f"plot_runtime_seconds={row['plot_runtime_seconds']:.2f}")
    print(f"postprocess_runtime_seconds={row['postprocess_runtime_seconds']:.2f}")
    print(f"model_selection_runtime_seconds={row.get('model_selection_runtime_seconds', 0.0):.2f}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
