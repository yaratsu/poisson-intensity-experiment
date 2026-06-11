from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import (
    circle_to_embedding,
    embedding_to_intrinsic,
    intrinsic_to_embedding,
    sample_uniform_circle,
    sample_uniform_sphere,
)
from .integration import make_quadrature
from .utils import support_metadata


def _z_col(Z: np.ndarray, j: int) -> np.ndarray:
    if Z.shape[1] <= j:
        return np.zeros(Z.shape[0], dtype=np.float64)
    return Z[:, j]


def _broadcast_z(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    if Z.shape[1] == 0:
        return np.empty((X.shape[0], 0), dtype=np.float64)
    if Z.shape[0] == X.shape[0]:
        return Z
    if Z.shape[0] == 1:
        return np.repeat(Z, X.shape[0], axis=0)
    raise ValueError(f"Cannot broadcast Z with shape {Z.shape} to X with shape {X.shape}")


@dataclass
class TrueIntensity:
    support: str
    z_dim: int
    expected_count: float = 30.0
    beta: float = 2.0
    alpha: float = 0.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.metadata = support_metadata(self.support)
        self.scale = self._calibrate_scale()

    @property
    def volume(self) -> float:
        return float(self.metadata["volume"])

    def base_intensity(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X = self._coerce_intrinsic_points(X)
        Z = _broadcast_z(X, Z)
        vals = self.scale * self.base_intensity(X, Z)
        return np.maximum(vals, 0.0).astype(np.float64)

    def sample_integration_points(
        self, n_points: int, rng: np.random.Generator
    ) -> np.ndarray:
        points, _, _ = make_quadrature(self.metadata, n_points, rng, method="uniform")
        return points

    def integration_weights(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        return np.full(points.shape[0], self.volume / points.shape[0], dtype=np.float64)

    def theory_rate_exponent(self, z_dim: int) -> float:
        raise NotImplementedError

    def sample_events(
        self,
        Z: np.ndarray,
        rng: np.random.Generator,
        upper_grid_size: int = 1024,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        events: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        for z in Z:
            s_max = self.safe_upper_bound(z.reshape(1, -1), rng, upper_grid_size)
            n_candidates = int(rng.poisson(max(s_max * self.volume, 0.0)))
            cand_coords, cand_embedded = self._sample_uniform_support(n_candidates, rng)
            if n_candidates == 0:
                coords.append(np.empty((0, self.metadata["coord_dim"]), dtype=np.float64))
                events.append(np.empty((0, self.metadata["embedding_dim"]), dtype=np.float64))
                continue
            vals = self.evaluate(cand_coords, np.repeat(z.reshape(1, -1), n_candidates, axis=0))
            accept_prob = np.clip(vals / max(s_max, 1e-12), 0.0, 1.0)
            keep = rng.uniform(0.0, 1.0, size=n_candidates) <= accept_prob
            coords.append(cand_coords[keep].astype(np.float64))
            events.append(cand_embedded[keep].astype(np.float64))
        return events, coords

    def safe_upper_bound(
        self, z: np.ndarray, rng: np.random.Generator, n_points: int = 2048
    ) -> float:
        points = self.sample_integration_points(n_points, rng)
        vals = self.evaluate(points, np.repeat(z, points.shape[0], axis=0))
        return float(np.max(vals) * 1.25 + 1e-8)

    def _sample_uniform_support(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.metadata["support_type"] == "euclidean":
            dim = self.metadata["event_dim"]
            x = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float64)
            return x, x.copy()
        if self.metadata["manifold_type"] == "circle":
            return sample_uniform_circle(n, rng)
        if self.metadata["manifold_type"] == "sphere":
            return sample_uniform_sphere(n, rng)
        raise ValueError(f"Unsupported support: {self.support}")

    def _coerce_intrinsic_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.metadata["support_type"] == "euclidean":
            return X
        coord_dim = self.metadata["coord_dim"]
        embedding_dim = self.metadata["embedding_dim"]
        if X.shape[1] == coord_dim:
            return X
        if X.shape[1] == embedding_dim:
            return embedding_to_intrinsic(X, self.metadata["manifold_type"])
        raise ValueError(f"Unexpected point shape {X.shape} for {self.support}")

    def _calibrate_scale(self) -> float:
        rng = np.random.default_rng(314159)
        n_z = 128 if self.z_dim > 0 else 1
        Z = rng.uniform(0.0, 1.0, size=(n_z, self.z_dim))
        if self.z_dim == 0:
            Z = np.empty((1, 0), dtype=np.float64)
        points, weights, _ = make_quadrature(self.metadata, 1024, rng, method="uniform")
        integrals = []
        for z in Z:
            zz = np.repeat(z.reshape(1, -1), points.shape[0], axis=0)
            integrals.append(float(np.sum(self.base_intensity(points, zz) * weights)))
        mean_integral = max(float(np.mean(integrals)), 1e-12)
        return float(self.expected_count / mean_integral)


@dataclass
class CompositionalIntensity(TrueIntensity):
    """Low-dimensional compositional intensity on [0, 1]^d."""

    def base_intensity(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = _broadcast_z(X, Z)
        x1 = X[:, 0]
        x2 = X[:, 1] if X.shape[1] > 1 else 0.5 * X[:, 0]
        z1 = _z_col(Z, 0)
        z2 = _z_col(Z, 1)
        h1 = x1 + 0.3 * z1 + 0.2 * z2
        h2 = x2 + 0.5 * np.sin(2.0 * np.pi * z1)
        vals = 0.2 + (0.6 + 0.4 * np.sin(2.0 * np.pi * h1)) ** 2
        vals += 0.5 * np.exp(-10.0 * (h2 - 0.5) ** 2)
        return vals

    def theory_rate_exponent(self, z_dim: int) -> float:
        alpha = 0.0
        beta_eff = self.beta
        t_eff = min(int(self.metadata["event_dim"]) + int(z_dim), 4)
        numerator = (1.0 + min(alpha, 1.0)) * beta_eff
        return float(numerator / (numerator + t_eff))


@dataclass
class NearZeroIntensity(TrueIntensity):
    epsilon: float = 1e-4
    p: float = 2.0
    alpha: float = 1.0

    def base_intensity(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = _broadcast_z(X, Z)
        z1 = _z_col(Z, 0)
        z2 = _z_col(Z, 1)
        center1 = 0.25 + 0.5 * z1
        if X.shape[1] == 1:
            dist2 = (X[:, 0] - center1) ** 2
        else:
            center2 = 0.5 + 0.25 * np.sin(2.0 * np.pi * z2)
            dist2 = (X[:, 0] - center1) ** 2 + (X[:, 1] - center2) ** 2
        valley = (1.0 - np.exp(-18.0 * dist2)) ** self.p
        return self.epsilon + valley

    def theory_rate_exponent(self, z_dim: int) -> float:
        beta = self.beta
        numerator = (1.0 + min(self.alpha, 1.0)) * beta
        dim = int(self.metadata["event_dim"]) + int(z_dim)
        return float(numerator / (numerator + dim))


@dataclass
class ManifoldIntensity(TrueIntensity):
    epsilon: float = 1e-4

    def base_intensity(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X = self._coerce_intrinsic_points(X)
        Z = _broadcast_z(X, Z)
        z1 = _z_col(Z, 0)
        z2 = _z_col(Z, 1)
        if self.metadata["manifold_type"] == "circle":
            theta = X[:, 0]
            raw = 1.0 + 0.5 * np.sin(3.0 * theta + z1) + 0.3 * np.cos(2.0 * theta - z2)
        elif self.metadata["manifold_type"] == "sphere":
            theta = X[:, 0]
            phi = X[:, 1]
            raw = 1.0 + 0.5 * np.sin(2.0 * theta) * np.cos(3.0 * phi + z1)
            raw += 0.3 * np.sin(phi + z2)
        else:
            raise ValueError("ManifoldIntensity supports circle and sphere only")
        return np.maximum(raw, 0.0) + self.epsilon

    def theory_rate_exponent(self, z_dim: int) -> float:
        beta = self.beta
        alpha = self.alpha
        d_m = int(self.metadata["manifold_dim"])
        numerator = (1.0 + min(alpha, 1.0)) * beta
        return float(numerator / (numerator + d_m + int(z_dim)))

    def _sample_uniform_support(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.metadata["manifold_type"] == "circle":
            return sample_uniform_circle(n, rng)
        return sample_uniform_sphere(n, rng)


def make_true_intensity(
    scenario: str,
    support: str,
    z_dim: int,
    expected_count: float = 30.0,
    epsilon: float = 1e-4,
    beta: float = 2.0,
    alpha: float | None = None,
) -> TrueIntensity:
    scenario = scenario.lower()
    if scenario == "compositional":
        if support not in {"euclidean1d", "euclidean2d"}:
            raise ValueError("Compositional scenario is implemented for Euclidean supports")
        return CompositionalIntensity(
            support=support,
            z_dim=z_dim,
            expected_count=expected_count,
            beta=beta,
            alpha=0.0 if alpha is None else alpha,
        )
    if scenario == "near_zero":
        if support not in {"euclidean1d", "euclidean2d"}:
            raise ValueError("Near-zero scenario is implemented for Euclidean supports")
        return NearZeroIntensity(
            support=support,
            z_dim=z_dim,
            expected_count=expected_count,
            epsilon=epsilon,
            beta=beta,
            alpha=1.0 if alpha is None else alpha,
        )
    if scenario == "manifold":
        if support not in {"circle", "sphere"}:
            raise ValueError("Manifold scenario requires circle or sphere support")
        return ManifoldIntensity(
            support=support,
            z_dim=z_dim,
            expected_count=expected_count,
            epsilon=epsilon,
            beta=beta,
            alpha=0.0 if alpha is None else alpha,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def points_for_model(points: np.ndarray, metadata: dict, mode: str = "intrinsic") -> np.ndarray:
    if metadata["support_type"] == "euclidean":
        return points
    if mode == "intrinsic":
        if points.shape[1] == metadata["coord_dim"]:
            return points
        return embedding_to_intrinsic(points, metadata["manifold_type"])
    if mode == "embedded":
        if points.shape[1] == metadata["embedding_dim"]:
            return points
        return intrinsic_to_embedding(points, metadata["manifold_type"])
    raise ValueError(f"Unknown manifold input mode: {mode}")
