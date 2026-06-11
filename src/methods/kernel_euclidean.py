from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from ..integration import make_quadrature
from .base import Estimator


class EuclideanKernelEstimator(Estimator):
    method_name = "kernel_euclidean"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {
            "mode": "joint_kernel",
            "kernel": "gaussian",
            "a_n": 1e-6,
            "bandwidth_x": None,
            "bandwidth_z": None,
            "bandwidth_x_candidates": None,
            "bandwidth_z_candidates": None,
            "validation_fraction": 0.2,
            "validation_integration_points": 128,
            "seed": 0,
        }
        if config:
            self.config.update({k: v for k, v in config.items() if v is not None})
        self.metadata: dict[str, Any] = {}
        self.Z = np.empty((0, 0), dtype=np.float64)
        self.event_points = np.empty((0, 0), dtype=np.float64)
        self.event_owner = np.empty((0,), dtype=np.int64)
        self.bandwidth_x = None
        self.bandwidth_z = None
        self.validation_scores: list[dict[str, float]] = []

    def fit(self, data: dict[str, Any]) -> "EuclideanKernelEstimator":
        self.metadata = dict(data["metadata"])
        if self.metadata["support_type"] != "euclidean":
            raise ValueError("EuclideanKernelEstimator supports Euclidean event spaces only")
        Z = np.asarray(data["Z"], dtype=np.float64)
        arrays = [np.asarray(arr, dtype=np.float64) for arr in data["event_coords"]]
        event_dim = int(self.metadata["event_dim"])
        if sum(arr.shape[0] for arr in arrays) == 0:
            flat = np.empty((0, event_dim), dtype=np.float64)
            owner = np.empty((0,), dtype=np.int64)
        else:
            flat = np.concatenate(arrays, axis=0)
            owner = np.repeat(np.arange(len(arrays), dtype=np.int64), [arr.shape[0] for arr in arrays])

        self.bandwidth_x, self.bandwidth_z = self._select_bandwidths(Z, arrays)
        self.Z = Z
        self.event_points = flat
        self.event_owner = owner
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Zb = self._broadcast_z(X.shape[0], np.asarray(Z, dtype=np.float64))
        if self.event_points.shape[0] == 0:
            return np.full(X.shape[0], 1e-8, dtype=np.float64)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for sl in _chunk_slices(X.shape[0], 256):
            out[sl] = self._predict_chunk(X[sl], Zb[sl])
        return np.maximum(out, 1e-12)

    def _predict_chunk(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        mode = self.config["mode"]
        if mode == "spatial_only" or self.Z.shape[1] == 0:
            denom = np.full(X.shape[0], max(self.Z.shape[0], 1), dtype=np.float64)
            z_owner_weight = np.ones((X.shape[0], self.event_points.shape[0]), dtype=np.float64)
        else:
            Kz_reps = self._product_kernel(Z[:, None, :] - self.Z[None, :, :], self.bandwidth_z)
            denom = np.maximum(Kz_reps.sum(axis=1), float(self.config["a_n"]))
            z_owner_weight = Kz_reps[:, self.event_owner]
        numerator = np.zeros(X.shape[0], dtype=np.float64)
        for ev_sl in _chunk_slices(self.event_points.shape[0], 20000):
            Kx = self._product_kernel(X[:, None, :] - self.event_points[None, ev_sl, :], self.bandwidth_x)
            numerator += np.sum(Kx * z_owner_weight[:, ev_sl], axis=1)
        return numerator / np.maximum(denom, float(self.config["a_n"]))

    def _select_bandwidths(self, Z: np.ndarray, arrays: list[np.ndarray]) -> tuple[float, float]:
        hx_fixed = self.config.get("bandwidth_x")
        hz_fixed = self.config.get("bandwidth_z")
        hx_candidates = [float(hx_fixed)] if hx_fixed else self._default_hx_candidates()
        hz_candidates = [float(hz_fixed)] if hz_fixed else self._default_hz_candidates(Z.shape[1])
        if len(hx_candidates) == 1 and len(hz_candidates) == 1:
            return hx_candidates[0], hz_candidates[0]

        rng = np.random.default_rng(int(self.config["seed"]))
        n = Z.shape[0]
        indices = np.arange(n)
        rng.shuffle(indices)
        n_val = int(round(float(self.config["validation_fraction"]) * n))
        n_val = min(max(n_val, 1 if n > 5 else 0), max(n - 1, 0))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:] if n_val else indices
        if len(val_idx) == 0:
            return hx_candidates[len(hx_candidates) // 2], hz_candidates[len(hz_candidates) // 2]

        best = (math.inf, hx_candidates[0], hz_candidates[0])
        for hx, hz in itertools.product(hx_candidates, hz_candidates):
            self.bandwidth_x, self.bandwidth_z = float(hx), float(hz)
            self.Z = Z[train_idx]
            train_arrays = [arrays[i] for i in train_idx]
            if sum(arr.shape[0] for arr in train_arrays):
                self.event_points = np.concatenate(train_arrays, axis=0)
                self.event_owner = np.repeat(
                    np.arange(len(train_arrays), dtype=np.int64),
                    [arr.shape[0] for arr in train_arrays],
                )
            else:
                self.event_points = np.empty((0, int(self.metadata["event_dim"])), dtype=np.float64)
                self.event_owner = np.empty((0,), dtype=np.int64)
            score = self._validation_nll(Z, arrays, val_idx)
            self.validation_scores.append({"bandwidth_x": float(hx), "bandwidth_z": float(hz), "val_nll": score})
            if score < best[0]:
                best = (score, float(hx), float(hz))
        return best[1], best[2]

    def _validation_nll(self, Z: np.ndarray, arrays: list[np.ndarray], val_idx: np.ndarray) -> float:
        rng = np.random.default_rng(int(self.config["seed"]) + 991)
        q_points, q_weights, _ = make_quadrature(
            self.metadata,
            int(self.config["validation_integration_points"]),
            rng,
        )
        losses = []
        for idx in val_idx:
            z = Z[idx].reshape(1, -1)
            obs = arrays[idx]
            loss = 0.0
            if obs.shape[0]:
                pred_obs = self.predict(obs, np.repeat(z, obs.shape[0], axis=0))
                loss -= float(np.log(np.maximum(pred_obs, 1e-12)).sum())
            pred_q = self.predict(q_points, z)
            loss += float(np.sum(pred_q * q_weights))
            losses.append(loss)
        return float(np.mean(losses)) if losses else math.inf

    def _default_hx_candidates(self) -> list[float]:
        event_dim = int(self.metadata.get("event_dim", 1))
        base = np.array([0.03, 0.06, 0.10, 0.18, 0.32]) if event_dim == 1 else np.array([0.05, 0.09, 0.16, 0.28])
        cfg = self.config.get("bandwidth_x_candidates")
        return [float(x) for x in (cfg if cfg is not None else base)]

    def _default_hz_candidates(self, z_dim: int) -> list[float]:
        if z_dim == 0:
            return [1.0]
        cfg = self.config.get("bandwidth_z_candidates")
        return [float(x) for x in (cfg if cfg is not None else [0.12, 0.25, 0.5, 1.0])]

    def _product_kernel(self, diff: np.ndarray, bandwidth: float) -> np.ndarray:
        return product_kernel(diff, bandwidth, self.config.get("kernel", "gaussian"))

    def _broadcast_z(self, n: int, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        z_dim = int(self.metadata.get("z_dim", Z.shape[1]))
        if z_dim == 0:
            return np.empty((n, 0), dtype=np.float64)
        if Z.shape[0] == n:
            return Z
        if Z.shape[0] == 1:
            return np.repeat(Z, n, axis=0)
        raise ValueError(f"Cannot broadcast Z shape {Z.shape} to {n} rows")


def product_kernel(diff: np.ndarray, bandwidth: float, kernel: str = "gaussian") -> np.ndarray:
    if diff.shape[-1] == 0:
        return np.ones(diff.shape[:-1], dtype=np.float64)
    h = max(float(bandwidth), 1e-12)
    u = diff / h
    if kernel == "epanechnikov":
        vals = 0.75 * np.maximum(1.0 - u**2, 0.0) / h
        vals[np.abs(u) > 1.0] = 0.0
        return np.prod(vals, axis=-1)
    norm_const = (math.sqrt(2.0 * math.pi) * h) ** diff.shape[-1]
    return np.exp(-0.5 * np.sum(u**2, axis=-1)) / norm_const


def _chunk_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))
