from __future__ import annotations

from typing import Any

import numpy as np

from .covariates import CovariateSampler, UniformCovariateSampler
from .integration import make_quadrature
from .intensities import TrueIntensity


def squared_hellinger(
    estimator: Any,
    true_intensity: TrueIntensity,
    z_dim: int,
    rng: np.random.Generator,
    n_eval_z: int = 128,
    n_eval_points: int = 1024,
    covariate_sampler: CovariateSampler | None = None,
) -> float:
    sampler = covariate_sampler or UniformCovariateSampler()
    if z_dim == 0:
        Z_eval = np.empty((1, 0), dtype=np.float64)
    else:
        Z_eval = sampler.sample(n_eval_z, z_dim, rng)
    points, weights, _ = make_quadrature(true_intensity.metadata, n_eval_points, rng)
    values = []
    for z in Z_eval:
        z_one = z.reshape(1, -1)
        true_vals = true_intensity.evaluate(points, np.repeat(z_one, points.shape[0], axis=0))
        pred_vals = estimator.predict(points, z_one)
        diff = np.sqrt(np.maximum(pred_vals, 0.0)) - np.sqrt(np.maximum(true_vals, 0.0))
        values.append(0.5 * float(np.sum(diff**2 * weights)))
    return float(np.mean(values))


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
