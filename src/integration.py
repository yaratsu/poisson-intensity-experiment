from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .geometry import make_circle_mesh, make_sphere_mesh
from .utils import ensure_dir


_MANIFOLD_QUADRATURE_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


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
    cache_dir: str | None = None,
    mesh_resolution: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = int(mesh_resolution or n_points)
    key = (manifold_type, resolution)
    if key in _MANIFOLD_QUADRATURE_CACHE:
        return _MANIFOLD_QUADRATURE_CACHE[key]
    path = None
    if cache_dir:
        path = Path(cache_dir) / "mesh" / f"{manifold_type}_res{resolution}.npz"
        if path.exists():
            loaded = np.load(path)
            out = (loaded["coords"].astype(np.float64), loaded["weights"].astype(np.float64), loaded["embedded"].astype(np.float64))
            _MANIFOLD_QUADRATURE_CACHE[key] = out
            return out
    if manifold_type == "circle":
        mesh = make_circle_mesh(max(resolution, 16))
    elif manifold_type == "sphere":
        n_theta = max(8, int(np.sqrt(resolution / 2)))
        n_phi = max(16, int(np.ceil(resolution / n_theta)))
        mesh = make_sphere_mesh(n_theta=n_theta, n_phi=n_phi)
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    out = (
        mesh.quadrature_points.astype(np.float64),
        mesh.quadrature_weights.astype(np.float64),
        mesh.embedded_quadrature_points.astype(np.float64),
    )
    _MANIFOLD_QUADRATURE_CACHE[key] = out
    if path is not None:
        ensure_dir(path.parent)
        np.savez_compressed(path, coords=out[0], weights=out[1], embedded=out[2])
    return out


def make_quadrature(
    metadata: dict,
    n_points: int,
    rng: np.random.Generator,
    method: str = "sobol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if metadata["support_type"] == "euclidean":
        points, weights = euclidean_quadrature(metadata["event_dim"], n_points, rng, method)
        return points, weights, points
    coords, weights, embedded = manifold_quadrature(
        metadata["manifold_type"],
        n_points,
        cache_dir=metadata.get("cache_dir"),
        mesh_resolution=metadata.get("mesh_resolution"),
    )
    return coords, weights, embedded
