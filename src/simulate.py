from __future__ import annotations

from typing import Any

import numpy as np

from .covariates import CovariateSampler, UniformCovariateSampler
from .intensities import TrueIntensity


def simulate_dataset(
    intensity: TrueIntensity,
    n: int,
    z_dim: int,
    rng: np.random.Generator,
    covariate_sampler: CovariateSampler | None = None,
) -> dict[str, Any]:
    sampler = covariate_sampler or UniformCovariateSampler()
    Z = sampler.sample(n, z_dim, rng)
    events, event_coords = intensity.sample_events(Z, rng)
    counts = np.asarray([arr.shape[0] for arr in events], dtype=np.int64)
    metadata = dict(intensity.metadata)
    metadata.update(
        {
            "z_dim": z_dim,
            "n": n,
            "expected_count": intensity.expected_count,
            "mean_observed_count": float(np.mean(counts)) if n else 0.0,
            "total_events": int(np.sum(counts)),
        }
    )
    return {
        "Z": Z.astype(np.float64),
        "events": events,
        "event_coords": event_coords,
        "counts": counts,
        "metadata": metadata,
    }


def flatten_events(data: dict[str, Any], use_coords: bool = True) -> tuple[np.ndarray, np.ndarray]:
    key = "event_coords" if use_coords else "events"
    arrays = data[key]
    counts = [arr.shape[0] for arr in arrays]
    dim = data["metadata"]["coord_dim" if use_coords else "embedding_dim"]
    if sum(counts) == 0:
        return np.empty((0, dim), dtype=np.float64), np.empty((0,), dtype=np.int64)
    flat = np.concatenate(arrays, axis=0).astype(np.float64)
    owner = np.repeat(np.arange(len(arrays), dtype=np.int64), counts)
    return flat, owner
