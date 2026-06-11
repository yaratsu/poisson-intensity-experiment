from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .covariates import make_covariate_sampler
from .evaluation import squared_hellinger
from .intensities import make_true_intensity
from .methods.dnn_npmle import DNNNPMLEEstimator
from .methods.kernel_covariate import CovariateKernelEstimator
from .methods.kernel_euclidean import EuclideanKernelEstimator
from .methods.kernel_manifold import ManifoldKernelEstimator
from .plots import plot_intensity
from .simulate import simulate_dataset
from .utils import deep_update, ensure_dir, load_yaml, save_json, set_random_seed, stable_seed


def build_estimator(method: str, config: dict[str, Any]) -> Any:
    method = method.lower()
    if method in {"dnn", "dnn_npmle"}:
        return DNNNPMLEEstimator(config.get("dnn", {}))
    if method in {"kernel_euclidean", "euclidean_kernel"}:
        return EuclideanKernelEstimator(config.get("kernel_euclidean", {}))
    if method in {"kernel_covariate", "covariate_kernel"}:
        return CovariateKernelEstimator(config.get("kernel_covariate", {}))
    if method in {"kernel_manifold", "manifold_kernel"}:
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
            f"_ds{_tag_number(config.get('depth_scale', 1.0))}"
            f"_ws{_tag_number(config.get('width_scale', 8.0))}"
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
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--width-scale", type=float, default=None)
    parser.add_argument("--min-depth", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-width", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
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
            "covariate_sampler": args.covariate_sampler,
            "results_dir": args.results_dir,
        },
        "dnn": {
            "output_activation": args.output_activation,
            "architecture": args.dnn_architecture,
            "hidden_layers": hidden_layers,
            "depth_scale": args.depth_scale,
            "width_scale": args.width_scale,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "min_width": args.min_width,
            "max_width": args.max_width,
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
    set_random_seed(seed)
    for section in ["dnn", "kernel_euclidean", "kernel_covariate", "kernel_manifold"]:
        cfg[section]["seed"] = seed
    cfg["dnn"]["expected_count"] = expected_count

    rng = np.random.default_rng(seed)
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
    data = simulate_dataset(true_intensity, n=n, z_dim=z_dim, rng=rng, covariate_sampler=cov_sampler)
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
    runtime = time.perf_counter() - start

    metric_rng = np.random.default_rng(seed + 2024)
    h2 = squared_hellinger(
        estimator,
        true_intensity,
        z_dim=z_dim,
        rng=metric_rng,
        n_eval_z=int(exp.get("eval_z", 64)),
        n_eval_points=int(exp.get("eval_points", 512)),
        covariate_sampler=cov_sampler,
    )

    results_dir = Path(exp.get("results_dir", "results"))
    metrics_dir = ensure_dir(results_dir / "metrics")
    models_dir = ensure_dir(results_dir / "models")
    metadata = data["metadata"]
    is_dnn = method.lower() in {"dnn", "dnn_npmle"}
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
    estimator.save(models_dir / f"{run_id}{model_suffix}")
    if hasattr(estimator, "history"):
        save_json({"history": estimator.history}, metrics_dir / f"history_{run_id}.json")

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
        "squared_hellinger": h2,
        "theory_rate_exponent": true_intensity.theory_rate_exponent(z_dim),
        "seed": seed,
        "runtime_seconds": runtime,
        "mean_observed_count": metadata["mean_observed_count"],
        "total_events": metadata["total_events"],
    }
    save_json(row, metrics_dir / f"{run_id}.json")
    all_csv = metrics_dir / "all_metrics.csv"
    if all_csv.exists():
        df = pd.read_csv(all_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.drop_duplicates(subset=["scenario", "support", "z_dim", "n", "repetition", "method", "seed"], keep="last")
    else:
        df = pd.DataFrame([row])
    df.to_csv(all_csv, index=False)

    if not args.no_plots:
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

    print(f"run_id={run_id}")
    print(f"squared_hellinger={h2:.8g}")
    print(f"runtime_seconds={runtime:.2f}")
    print(f"metrics={metrics_dir / f'{run_id}.json'}")


if __name__ == "__main__":
    main()
