from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class CovariateSampler:
    """Base interface for covariate samplers on [0, 1]^z_dim."""

    def sample(self, n: int, z_dim: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class UniformCovariateSampler(CovariateSampler):
    def sample(self, n: int, z_dim: int, rng: np.random.Generator) -> np.ndarray:
        if z_dim == 0:
            return np.empty((n, 0), dtype=np.float64)
        return rng.uniform(0.0, 1.0, size=(n, z_dim)).astype(np.float64)


@dataclass
class BetaCovariateSampler(CovariateSampler):
    a: float = 2.0
    b: float = 2.0

    def sample(self, n: int, z_dim: int, rng: np.random.Generator) -> np.ndarray:
        if z_dim == 0:
            return np.empty((n, 0), dtype=np.float64)
        return rng.beta(self.a, self.b, size=(n, z_dim)).astype(np.float64)


@dataclass
class GaussianCopulaCovariateSampler(CovariateSampler):
    rho: float = 0.4

    def sample(self, n: int, z_dim: int, rng: np.random.Generator) -> np.ndarray:
        if z_dim == 0:
            return np.empty((n, 0), dtype=np.float64)
        cov = np.full((z_dim, z_dim), self.rho, dtype=np.float64)
        np.fill_diagonal(cov, 1.0)
        normals = rng.multivariate_normal(np.zeros(z_dim), cov, size=n)
        return normal_cdf(normals)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(x)
    except Exception:
        erf = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def make_covariate_sampler(name: str, **kwargs) -> CovariateSampler:
    name = (name or "uniform").lower()
    if name == "uniform":
        return UniformCovariateSampler()
    if name == "beta":
        return BetaCovariateSampler(
            a=float(kwargs.get("a", kwargs.get("beta_a", 2.0))),
            b=float(kwargs.get("b", kwargs.get("beta_b", 2.0))),
        )
    if name in {"gaussian_copula", "copula"}:
        return GaussianCopulaCovariateSampler(rho=float(kwargs.get("rho", 0.4)))
    raise ValueError(f"Unknown covariate sampler: {name}")
