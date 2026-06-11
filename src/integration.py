from __future__ import annotations

import math

import numpy as np

from .geometry import make_circle_mesh, make_sphere_mesh


def euclidean_quadrature(
    event_dim: int,
    n_points: int,
    rng: np.random.Generator,
    method: str = "sobol",
) -> tuple[np.ndarray, np.ndarray]:
    if method == "sobol":
        try:
            from scipy.stats import qmc

            m = int(math.ceil(math.log2(max(n_points, 2))))
            sampler = qmc.Sobol(d=event_dim, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
            points = sampler.random_base2(m)[:n_points]
        except Exception:
            points = rng.uniform(0.0, 1.0, size=(n_points, event_dim))
    else:
        points = rng.uniform(0.0, 1.0, size=(n_points, event_dim))
    weights = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    return points.astype(np.float64), weights


def manifold_quadrature(
    manifold_type: str,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if manifold_type == "circle":
        mesh = make_circle_mesh(max(n_points, 16))
    elif manifold_type == "sphere":
        n_theta = max(8, int(np.sqrt(n_points / 2)))
        n_phi = max(16, int(np.ceil(n_points / n_theta)))
        mesh = make_sphere_mesh(n_theta=n_theta, n_phi=n_phi)
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    return (
        mesh.quadrature_points.astype(np.float64),
        mesh.quadrature_weights.astype(np.float64),
        mesh.embedded_quadrature_points.astype(np.float64),
    )


def make_quadrature(
    metadata: dict,
    n_points: int,
    rng: np.random.Generator,
    method: str = "sobol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if metadata["support_type"] == "euclidean":
        points, weights = euclidean_quadrature(metadata["event_dim"], n_points, rng, method)
        return points, weights, points
    coords, weights, embedded = manifold_quadrature(metadata["manifold_type"], n_points)
    return coords, weights, embedded
