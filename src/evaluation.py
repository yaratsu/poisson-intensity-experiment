from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .covariates import CovariateSampler, UniformCovariateSampler
from .integration import make_quadrature
from .intensities import TrueIntensity
from .utils import ensure_dir


def squared_hellinger(
    estimator: Any,
    true_intensity: TrueIntensity,
    z_dim: int,
    rng: np.random.Generator,
    n_eval_z: int = 128,
    n_eval_points: int = 1024,
    covariate_sampler: CovariateSampler | None = None,
    eval_chunk_size: int = 4096,
    cache_dir: str | None = None,
    cache_seed: int | None = None,
    mesh_resolution: int | None = None,
) -> float:
    sampler = covariate_sampler or UniformCovariateSampler()
    Z_eval, points, weights = evaluation_grid(
        true_intensity=true_intensity,
        z_dim=z_dim,
        rng=rng,
        n_eval_z=n_eval_z,
        n_eval_points=n_eval_points,
        covariate_sampler=sampler,
        cache_dir=cache_dir,
        cache_seed=cache_seed,
        mesh_resolution=mesh_resolution,
    )
    values = []
    for z in Z_eval:
        z_one = z.reshape(1, -1)
        total = 0.0
        for sl in _chunk_slices(points.shape[0], int(eval_chunk_size)):
            pts = points[sl]
            w = weights[sl]
            true_vals = true_intensity.evaluate(pts, np.repeat(z_one, pts.shape[0], axis=0))
            pred_vals = estimator.predict(pts, z_one)
            diff = np.sqrt(np.maximum(pred_vals, 0.0)) - np.sqrt(np.maximum(true_vals, 0.0))
            total += float(np.sum(diff**2 * w))
        values.append(0.5 * total)
    return float(np.mean(values))


def evaluation_grid(
    true_intensity: TrueIntensity,
    z_dim: int,
    rng: np.random.Generator,
    n_eval_z: int,
    n_eval_points: int,
    covariate_sampler: CovariateSampler,
    cache_dir: str | None = None,
    cache_seed: int | None = None,
    mesh_resolution: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario = true_intensity.metadata.get("scenario", true_intensity.__class__.__name__.lower())
    support = true_intensity.metadata["support"]
    seed_tag = "none" if cache_seed is None else str(int(cache_seed))
    path = None
    if cache_dir:
        path = Path(cache_dir) / "evaluation_grid" / f"{scenario}_{support}_zdim{z_dim}_seed{seed_tag}.npz"
        if path.exists():
            loaded = np.load(path)
            return loaded["Z_eval"], loaded["points"], loaded["weights"]

    if z_dim == 0:
        Z_eval = np.empty((1, 0), dtype=np.float64)
    else:
        Z_eval = covariate_sampler.sample(n_eval_z, z_dim, rng)
    metadata = dict(true_intensity.metadata)
    if cache_dir:
        metadata["cache_dir"] = cache_dir
    if mesh_resolution is not None:
        metadata["mesh_resolution"] = mesh_resolution
    points, weights, _ = make_quadrature(metadata, n_eval_points, rng)
    if path is not None:
        ensure_dir(path.parent)
        np.savez_compressed(path, Z_eval=Z_eval, points=points, weights=weights)
    return Z_eval, points, weights


def poisson_validation_nll(
    estimator: Any,
    data: dict[str, Any],
    indices: np.ndarray,
    rng: np.random.Generator,
    n_integration_points: int = 256,
) -> float:
    points, weights, _ = make_quadrature(data["metadata"], n_integration_points, rng)
    losses = []
    Z = data["Z"]
    for idx in indices:
        z = Z[idx].reshape(1, -1)
        obs = data["event_coords"][idx]
        loss = 0.0
        if obs.shape[0]:
            pred_obs = estimator.predict(obs, np.repeat(z, obs.shape[0], axis=0))
            loss -= float(np.log(np.maximum(pred_obs, 1e-12)).sum())
        pred_q = estimator.predict(points, z)
        loss += float(np.sum(pred_q * weights))
        losses.append(loss)
    return float(np.mean(losses)) if losses else float("nan")


def _chunk_slices(n: int, chunk_size: int):
    for start in range(0, n, max(int(chunk_size), 1)):
        yield slice(start, min(start + max(int(chunk_size), 1), n))
