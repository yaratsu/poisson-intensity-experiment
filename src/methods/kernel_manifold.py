from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from ..geometry import (
    circle_geodesic,
    embedding_to_intrinsic,
    intrinsic_to_embedding,
    sphere_geodesic,
)
from ..integration import make_quadrature
from .base import Estimator
from .kernel_euclidean import product_kernel


class ManifoldKernelEstimator(Estimator):
    method_name = "kernel_manifold"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {
            "a_n": 1e-6,
            "bandwidth_m": None,
            "bandwidth_z": None,
            "bandwidth_m_candidates": None,
            "bandwidth_z_candidates": None,
            "validation_fraction": 0.2,
            "validation_integration_points": 128,
            "seed": 0,
        }
        if config:
            self.config.update({k: v for k, v in config.items() if v is not None})
        self.metadata: dict[str, Any] = {}
        self.Z = np.empty((0, 0), dtype=np.float64)
        self.event_coords = np.empty((0, 0), dtype=np.float64)
        self.event_embedded = np.empty((0, 0), dtype=np.float64)
        self.event_owner = np.empty((0,), dtype=np.int64)
        self.bandwidth_m = None
        self.bandwidth_z = None
        self.validation_scores: list[dict[str, float]] = []

    def fit(self, data: dict[str, Any]) -> "ManifoldKernelEstimator":
        self.metadata = dict(data["metadata"])
        if self.metadata["support_type"] != "manifold":
            raise ValueError("ManifoldKernelEstimator supports manifold event spaces only")
        Z = np.asarray(data["Z"], dtype=np.float64)
        coord_arrays = [np.asarray(arr, dtype=np.float64) for arr in data["event_coords"]]
        emb_arrays = [np.asarray(arr, dtype=np.float64) for arr in data["events"]]
        self.bandwidth_m, self.bandwidth_z = self._select_bandwidths(Z, coord_arrays, emb_arrays)
        self.Z = Z
        self._store_events(coord_arrays, emb_arrays, np.arange(Z.shape[0]))
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        coords, embedded = self._coerce_points(X)
        Zb = self._broadcast_z(coords.shape[0], np.asarray(Z, dtype=np.float64))
        if self.event_owner.shape[0] == 0:
            return np.full(coords.shape[0], 1e-8, dtype=np.float64)
        out = np.zeros(coords.shape[0], dtype=np.float64)
        for sl in _chunk_slices(coords.shape[0], 128):
            out[sl] = self._predict_chunk(coords[sl], embedded[sl], Zb[sl])
        return np.maximum(out, 1e-12)

    def _predict_chunk(self, coords: np.ndarray, embedded: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.Z.shape[1] == 0:
            denom = np.full(coords.shape[0], max(self.Z.shape[0], 1), dtype=np.float64)
            z_owner_weight = np.ones((coords.shape[0], self.event_owner.shape[0]), dtype=np.float64)
        else:
            Kz_reps = product_kernel(Z[:, None, :] - self.Z[None, :, :], self.bandwidth_z, "gaussian")
            denom = np.maximum(Kz_reps.sum(axis=1), float(self.config["a_n"]))
            z_owner_weight = Kz_reps[:, self.event_owner]
        numerator = np.zeros(coords.shape[0], dtype=np.float64)
        for ev_sl in _chunk_slices(self.event_owner.shape[0], 20000):
            d = self._geodesic_distance(coords, embedded, ev_sl)
            Km = self._manifold_kernel(d, self.bandwidth_m)
            numerator += np.sum(Km * z_owner_weight[:, ev_sl], axis=1)
        return numerator / np.maximum(denom, float(self.config["a_n"]))

    def _geodesic_distance(self, coords: np.ndarray, embedded: np.ndarray, ev_sl: slice) -> np.ndarray:
        if self.metadata["manifold_type"] == "circle":
            return circle_geodesic(coords[:, 0], self.event_coords[ev_sl, 0])
        return sphere_geodesic(embedded, self.event_embedded[ev_sl])

    def _manifold_kernel(self, d: np.ndarray, bandwidth: float) -> np.ndarray:
        h = max(float(bandwidth), 1e-12)
        dim = int(self.metadata["manifold_dim"])
        norm_const = (math.sqrt(2.0 * math.pi) * h) ** dim
        return np.exp(-0.5 * (d / h) ** 2) / norm_const

    def _select_bandwidths(
        self,
        Z: np.ndarray,
        coord_arrays: list[np.ndarray],
        emb_arrays: list[np.ndarray],
    ) -> tuple[float, float]:
        hm_fixed = self.config.get("bandwidth_m")
        hz_fixed = self.config.get("bandwidth_z")
        hm_candidates = [float(hm_fixed)] if hm_fixed else self._default_hm_candidates()
        hz_candidates = [float(hz_fixed)] if hz_fixed else self._default_hz_candidates(Z.shape[1])
        if len(hm_candidates) == 1 and len(hz_candidates) == 1:
            return hm_candidates[0], hz_candidates[0]
        rng = np.random.default_rng(int(self.config["seed"]))
        n = Z.shape[0]
        indices = np.arange(n)
        rng.shuffle(indices)
        n_val = int(round(float(self.config["validation_fraction"]) * n))
        n_val = min(max(n_val, 1 if n > 5 else 0), max(n - 1, 0))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:] if n_val else indices
        if len(val_idx) == 0:
            return hm_candidates[len(hm_candidates) // 2], hz_candidates[len(hz_candidates) // 2]
        best = (math.inf, hm_candidates[0], hz_candidates[0])
        for hm, hz in itertools.product(hm_candidates, hz_candidates):
            self.bandwidth_m, self.bandwidth_z = float(hm), float(hz)
            self.Z = Z[train_idx]
            self._store_events(coord_arrays, emb_arrays, train_idx)
            score = self._validation_nll(Z, coord_arrays, val_idx)
            self.validation_scores.append({"bandwidth_m": float(hm), "bandwidth_z": float(hz), "val_nll": score})
            if score < best[0]:
                best = (score, float(hm), float(hz))
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

    def _store_events(self, coord_arrays: list[np.ndarray], emb_arrays: list[np.ndarray], indices: np.ndarray) -> None:
        selected_coords = [coord_arrays[i] for i in indices]
        selected_emb = [emb_arrays[i] for i in indices]
        if sum(arr.shape[0] for arr in selected_coords):
            self.event_coords = np.concatenate(selected_coords, axis=0)
            self.event_embedded = np.concatenate(selected_emb, axis=0)
            self.event_owner = np.repeat(
                np.arange(len(selected_coords), dtype=np.int64),
                [arr.shape[0] for arr in selected_coords],
            )
        else:
            self.event_coords = np.empty((0, int(self.metadata["coord_dim"])), dtype=np.float64)
            self.event_embedded = np.empty((0, int(self.metadata["embedding_dim"])), dtype=np.float64)
            self.event_owner = np.empty((0,), dtype=np.int64)

    def _default_hm_candidates(self) -> list[float]:
        cfg = self.config.get("bandwidth_m_candidates")
        if cfg is not None:
            return [float(x) for x in cfg]
        return [0.08, 0.16, 0.32, 0.64] if self.metadata.get("manifold_type") == "circle" else [0.12, 0.25, 0.5, 0.9]

    def _default_hz_candidates(self, z_dim: int) -> list[float]:
        if z_dim == 0:
            return [1.0]
        cfg = self.config.get("bandwidth_z_candidates")
        return [float(x) for x in (cfg if cfg is not None else [0.12, 0.25, 0.5, 1.0])]

    def _coerce_points(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coord_dim = int(self.metadata["coord_dim"])
        embedding_dim = int(self.metadata["embedding_dim"])
        manifold_type = self.metadata["manifold_type"]
        if X.shape[1] == coord_dim:
            coords = X
            embedded = intrinsic_to_embedding(coords, manifold_type)
            return coords, embedded
        if X.shape[1] == embedding_dim:
            embedded = X
            coords = embedding_to_intrinsic(embedded, manifold_type)
            return coords, embedded
        raise ValueError(f"Unexpected point shape {X.shape} for {manifold_type}")

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


def _chunk_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))
