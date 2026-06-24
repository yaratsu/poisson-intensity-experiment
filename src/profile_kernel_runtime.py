from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .covariates import make_covariate_sampler
from .evaluation import squared_hellinger
from .intensities import make_true_intensity
from .methods.kernel_covariate import CovariateKernelEstimator
from .methods.kernel_euclidean import EuclideanKernelEstimator
from .methods.kernel_manifold import ManifoldKernelEstimator
from .simulate import simulate_dataset
from .utils import ensure_dir, save_json, set_random_seed, stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile kernel baseline runtime.")
    parser.add_argument("--scenario", default="compositional", choices=["compositional", "near_zero", "manifold"])
    parser.add_argument("--support", default="euclidean2d", choices=["euclidean1d", "euclidean2d", "circle", "sphere"])
    parser.add_argument("--z-dim", type=int, default=1)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--method", default=None, choices=[None, "kernel_euclidean", "kernel_covariate", "kernel_manifold", "euclidean_kernel", "covariate_kernel", "manifold_kernel"])
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--expected-count", type=float, default=30.0)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--eval-z", type=int, default=8)
    parser.add_argument("--eval-points", type=int, default=128)
    parser.add_argument("--prediction-points", type=int, default=256)
    parser.add_argument("--kernel-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument("--gaussian-cutoff", type=float, default=4.0)
    parser.add_argument("--no-gaussian-cutoff", action="store_true")
    parser.add_argument("--kernel-type", choices=["gaussian", "epanechnikov"], default="gaussian")
    parser.add_argument("--boundary-correction", choices=["none", "renormalize"], default="renormalize")
    parser.add_argument("--bandwidth-search", choices=["coarse_to_fine", "full"], default="coarse_to_fine")
    parser.add_argument("--bandwidth-grid-size", type=int, default=5)
    parser.add_argument("--bandwidth-fine-grid-size", type=int, default=3)
    parser.add_argument("--kernel-cv-max-replicates", type=int, default=512)
    parser.add_argument("--kernel-cv-max-events-per-replicate", type=int, default=128)
    parser.add_argument("--validation-z-chunk-size", type=int, default=256)
    parser.add_argument("--validation-quadrature-method", choices=["uniform", "sobol"], default="uniform")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--use-distance-cache", dest="use_distance_cache", action="store_true", default=True)
    parser.add_argument("--no-distance-cache", dest="use_distance_cache", action="store_false")
    parser.add_argument("--mesh-resolution", type=int, default=64)
    parser.add_argument("--correction-type", choices=["none", "global", "local"], default="none")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def build_estimator(method: str, config: dict[str, Any]):
    if method in {"kernel_euclidean", "euclidean_kernel"}:
        config = dict(config)
        config["mode"] = "spatial_only"
        return EuclideanKernelEstimator(config)
    if method in {"kernel_covariate", "covariate_kernel"}:
        return CovariateKernelEstimator(config)
    if method in {"kernel_manifold", "manifold_kernel"}:
        return ManifoldKernelEstimator(config)
    raise ValueError(f"Unknown method: {method}")


def default_method(scenario: str, support: str, z_dim: int) -> str:
    if scenario == "manifold" or support in {"circle", "sphere"}:
        return "kernel_manifold"
    if int(z_dim) == 0:
        return "kernel_euclidean"
    return "kernel_covariate"


def max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return float(rss / (1024 * 1024) if rss > 10_000_000 else rss / 1024)


def main() -> None:
    args = parse_args()
    method = args.method or default_method(args.scenario, args.support, args.z_dim)
    seed = stable_seed("profile", args.scenario, args.support, args.z_dim, args.n, method, args.repetition)
    set_random_seed(seed, seed_torch=False)
    rng = np.random.default_rng(seed)
    cov_sampler = make_covariate_sampler("uniform")
    true_intensity = make_true_intensity(
        scenario=args.scenario,
        support=args.support,
        z_dim=args.z_dim,
        expected_count=args.expected_count,
        epsilon=args.epsilon,
        beta=args.beta,
        alpha=args.alpha,
    )
    true_intensity.metadata.update({"scenario": args.scenario})
    data = simulate_dataset(true_intensity, n=args.n, z_dim=args.z_dim, rng=rng, covariate_sampler=cov_sampler)
    data["metadata"].update({"scenario": args.scenario, "beta": true_intensity.beta, "alpha": true_intensity.alpha})

    config = {
        "kernel": args.kernel_type,
        "boundary_correction": args.boundary_correction,
        "kernel_chunk_size": args.kernel_chunk_size,
        "gaussian_cutoff": args.gaussian_cutoff,
        "use_gaussian_cutoff": not args.no_gaussian_cutoff,
        "bandwidth_search": args.bandwidth_search,
        "bandwidth_grid_size": args.bandwidth_grid_size,
        "bandwidth_fine_grid_size": args.bandwidth_fine_grid_size,
        "kernel_cv_max_replicates": args.kernel_cv_max_replicates,
        "kernel_cv_max_events_per_replicate": args.kernel_cv_max_events_per_replicate,
        "validation_z_chunk_size": args.validation_z_chunk_size,
        "validation_quadrature_method": args.validation_quadrature_method,
        "cache_dir": args.cache_dir,
        "use_distance_cache": args.use_distance_cache,
        "mesh_resolution": args.mesh_resolution,
        "correction_type": args.correction_type,
        "seed": seed,
    }
    estimator = build_estimator(method, config)

    fit_start = time.perf_counter()
    estimator.fit(data)
    fit_time = time.perf_counter() - fit_start

    q_points = true_intensity.sample_integration_points(args.prediction_points, rng)
    z_pred = np.empty((1, 0), dtype=np.float64) if args.z_dim == 0 else np.full((1, args.z_dim), 0.5)
    pred_start = time.perf_counter()
    pred = estimator.predict(q_points, z_pred)
    prediction_time = time.perf_counter() - pred_start

    h_start = time.perf_counter()
    h2 = squared_hellinger(
        estimator,
        true_intensity,
        z_dim=args.z_dim,
        rng=np.random.default_rng(seed + 2024),
        n_eval_z=args.eval_z,
        n_eval_points=args.eval_points,
        covariate_sampler=cov_sampler,
        eval_chunk_size=args.eval_chunk_size,
        cache_dir=args.cache_dir,
        cache_seed=seed + 2024,
        mesh_resolution=args.mesh_resolution,
    )
    hellinger_time = time.perf_counter() - h_start

    profile = dict(getattr(estimator, "profile", {}))
    out = {
        "scenario": args.scenario,
        "support": args.support,
        "z_dim": args.z_dim,
        "n": args.n,
        "method": method,
        "seed": seed,
        "fit_time_seconds": fit_time,
        "bandwidth_selection_time_seconds": profile.get("bandwidth_selection_time_seconds"),
        "validation_nll_time_seconds": profile.get("validation_nll_time_seconds"),
        "prediction_time_seconds": prediction_time,
        "hellinger_evaluation_time_seconds": hellinger_time,
        "distance_computation_time_seconds": profile.get("distance_computation_time_seconds"),
        "distance_cache_hits": profile.get("distance_cache_hits"),
        "distance_cache_misses": profile.get("distance_cache_misses"),
        "memory_max_rss_mb": max_rss_mb(),
        "squared_hellinger": h2,
        "prediction_mean": float(np.mean(pred)),
        "bandwidth_x": getattr(estimator, "bandwidth_x", None),
        "bandwidth_m": getattr(estimator, "bandwidth_m", None),
        "bandwidth_z": getattr(estimator, "bandwidth_z", None),
        "kernel_cv_max_replicates": args.kernel_cv_max_replicates,
        "kernel_cv_max_events_per_replicate": args.kernel_cv_max_events_per_replicate,
        "validation_z_chunk_size": args.validation_z_chunk_size,
        "validation_quadrature_method": args.validation_quadrature_method,
        "validation_scores": getattr(estimator, "validation_scores", []),
    }
    out_dir = ensure_dir(Path(args.results_dir) / "profiles")
    path = out_dir / f"kernel_runtime_{args.scenario}_{args.support}_zdim{args.z_dim}_n{args.n}.json"
    save_json(out, path)
    print(f"profile={path}")
    print(f"fit_time_seconds={fit_time:.4g}")
    print(f"hellinger_evaluation_time_seconds={hellinger_time:.4g}")


if __name__ == "__main__":
    main()
