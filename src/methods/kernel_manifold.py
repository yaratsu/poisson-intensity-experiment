from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np

from ..cache_utils import DistanceCache, array_hash
from ..geometry import circle_geodesic, embedding_to_intrinsic, intrinsic_to_embedding
from ..integration import make_quadrature
from .base import Estimator
from .kernel_euclidean import gaussian_from_scaled_sqdist, product_kernel, raw_sqdist


class ManifoldKernelEstimator(Estimator):
    method_name = "kernel_manifold"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {
            "a_n": 1e-6,
            "bandwidth_m": None,
            "bandwidth_z": None,
            "bandwidth_m_candidates": None,
            "bandwidth_z_candidates": None,
            "bandwidth_selection": "ward_cv",
            "bandwidth_cv_folds": 5,
            "bandwidth_theory_mode": "ward_geodesic_grid",
            "bandwidth_coarse_multipliers": [0.5, 0.75, 1.0, 1.5, 2.0],
            "bandwidth_fine_multipliers": [0.8, 1.0, 1.25],
            "bandwidth_min": 0.03,
            "bandwidth_max": 1.25,
            "bandwidth_search": "coarse_to_fine",
            "bandwidth_grid_size": 5,
            "bandwidth_fine_grid_size": 3,
            "validation_fraction": 0.2,
            "validation_integration_points": 128,
            "validation_quadrature_method": "uniform",
            "kernel_cv_max_replicates": 512,
            "kernel_cv_max_events_per_replicate": 128,
            "validation_z_chunk_size": 256,
            "kernel_chunk_size": 4096,
            "max_dense_entries": 20_000_000,
            "gaussian_cutoff": 4.0,
            "use_gaussian_cutoff": True,
            "mesh_resolution": 64,
            "correction_type": "none",
            "cache_dir": "cache",
            "use_distance_cache": True,
            "seed": 0,
        }
        if config:
            self.config.update({k: v for k, v in config.items() if v is not None})
        self.metadata: dict[str, Any] = {}
        self.Z = np.empty((0, 0), dtype=np.float64)
        self.event_coords = np.empty((0, 0), dtype=np.float64)
        self.event_embedded = np.empty((0, 0), dtype=np.float64)
        self.event_z = np.empty((0, 0), dtype=np.float64)
        self.event_owner = np.empty((0,), dtype=np.int64)
        self.bandwidth_m = None
        self.bandwidth_z = None
        self.validation_scores: list[dict[str, float]] = []
        self.cv_results: list[dict[str, Any]] = []
        self.selected_bandwidth_cv_score: float | None = None
        self.profile = {
            "fit_time_seconds": 0.0,
            "bandwidth_selection_time_seconds": 0.0,
            "validation_nll_time_seconds": 0.0,
            "prediction_time_seconds": 0.0,
            "distance_computation_time_seconds": 0.0,
            "distance_cache_hits": 0,
            "distance_cache_misses": 0,
        }
        self.distance_cache = DistanceCache(
            self.config.get("cache_dir", "cache"),
            enabled=bool(self.config.get("use_distance_cache", True)),
        )
        self._dataset_hash = "unfit"
        self._training_token = 0
        self._last_training_indices: tuple[int, ...] | None = None
        self._fold_event_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["distance_cache"] = None
        state["_fold_event_cache"] = {}
        state["cv_results"] = []
        state["validation_scores"] = []
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.distance_cache = DistanceCache(
            self.config.get("cache_dir", "cache"),
            enabled=bool(self.config.get("use_distance_cache", True)),
        )
        self._fold_event_cache = {}

    def fit(self, data: dict[str, Any]) -> "ManifoldKernelEstimator":
        start_fit = time.perf_counter()
        self.metadata = dict(data["metadata"])
        if self.metadata["support_type"] != "manifold":
            raise ValueError("ManifoldKernelEstimator supports manifold event spaces only")
        if str(self.config.get("correction_type", "none")).lower() != "none":
            raise NotImplementedError(
                "correction_type='global'/'local' is reserved for future Ward-style shape corrections; "
                "the existing implementation uses the normalized geodesic kernel with correction_type='none'."
            )
        Z = np.asarray(data["Z"], dtype=np.float64)
        coord_arrays = [np.asarray(arr, dtype=np.float64) for arr in data["event_coords"]]
        emb_arrays = [np.asarray(arr, dtype=np.float64) for arr in data["events"]]
        self._dataset_hash = self._hash_dataset(Z, coord_arrays)
        start_bw = time.perf_counter()
        self.bandwidth_m, self.bandwidth_z = self._select_bandwidths(Z, coord_arrays, emb_arrays)
        self.profile["bandwidth_selection_time_seconds"] += time.perf_counter() - start_bw
        self.Z = Z
        self._store_events(coord_arrays, emb_arrays, np.arange(Z.shape[0]))
        self.profile["fit_time_seconds"] += time.perf_counter() - start_fit
        self._sync_cache_stats()
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        coords, embedded = self._coerce_points(X)
        Zb = self._broadcast_z(coords.shape[0], np.asarray(Z, dtype=np.float64))
        if self.event_owner.shape[0] == 0:
            return np.full(coords.shape[0], 1e-8, dtype=np.float64)
        out = np.zeros(coords.shape[0], dtype=np.float64)
        q_chunk = self._query_chunk_size(coords.shape[0])
        for sl in _chunk_slices(coords.shape[0], q_chunk):
            out[sl] = self._predict_chunk(coords[sl], embedded[sl], Zb[sl])
        self.profile["prediction_time_seconds"] += time.perf_counter() - start
        self._sync_cache_stats()
        return np.maximum(out, 1e-12)

    def _predict_chunk(self, coords: np.ndarray, embedded: np.ndarray, Z: np.ndarray) -> np.ndarray:
        z_dim = self.Z.shape[1]
        if z_dim == 0:
            denom = np.full(coords.shape[0], max(self.Z.shape[0], 1), dtype=np.float64)
            Kz_reps = None
        else:
            dz2 = self._raw_sqdist(Z, self.Z, "z_replicate") / max(float(self.bandwidth_z) ** 2, 1e-24)
            Kz_reps = gaussian_from_scaled_sqdist(dz2, z_dim, float(self.bandwidth_z), self._gaussian_cutoff())
            denom = np.maximum(Kz_reps.sum(axis=1), float(self.config["a_n"]))

        numerator = np.zeros(coords.shape[0], dtype=np.float64)
        ev_chunk = self._event_chunk_size(coords.shape[0])
        for ev_sl in _chunk_slices(self.event_owner.shape[0], ev_chunk):
            Km = self._manifold_kernel_chunk(coords, embedded, ev_sl)
            if Kz_reps is not None:
                numerator += np.sum(Km * Kz_reps[:, self.event_owner[ev_sl]], axis=1)
            else:
                numerator += np.sum(Km, axis=1)
        return numerator / np.maximum(denom, float(self.config["a_n"]))

    def _manifold_kernel_chunk(self, coords: np.ndarray, embedded: np.ndarray, ev_sl: slice) -> np.ndarray:
        h = max(float(self.bandwidth_m), 1e-12)
        cutoff = self._gaussian_cutoff()
        dim = int(self.metadata["manifold_dim"])
        norm_const = (math.sqrt(2.0 * math.pi) * h) ** dim
        if self.metadata["manifold_type"] == "circle":
            d = circle_geodesic(coords[:, 0], self.event_coords[ev_sl, 0])
            scaled2 = (d / h) ** 2
            return gaussian_from_scaled_sqdist(scaled2, dim, h, cutoff)

        # Sphere: use dot products directly. If cutoff is active, avoid arccos on points outside the cap.
        dots = self._sphere_dots(embedded, self.event_embedded[ev_sl])
        dots = np.clip(dots, -1.0, 1.0)
        if cutoff is None:
            d = np.arccos(dots)
            return np.exp(-0.5 * (d / h) ** 2) / norm_const
        radius = min(float(cutoff) * h, math.pi)
        mask = dots >= math.cos(radius)
        out = np.zeros_like(dots, dtype=np.float64)
        if np.any(mask):
            d = np.arccos(dots[mask])
            out[mask] = np.exp(-0.5 * (d / h) ** 2) / norm_const
        return out

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
            self.selected_bandwidth_cv_score = None
            return hm_candidates[0], hz_candidates[0]

        selection = str(self.config.get("bandwidth_selection", "ward_cv")).lower()
        if selection == "ward_np":
            best = self._select_bandwidths_ward_np(Z, coord_arrays, emb_arrays, hm_candidates, hz_candidates)
            self.selected_bandwidth_cv_score = best[0]
            return best[1], best[2]
        if selection not in {"ward_cv", "poisson_5fold_cv", "theory_guided_5fold_cv"}:
            raise ValueError(f"Unsupported manifold bandwidth_selection: {selection}")

        Z_cv, coord_cv, emb_cv = self._prepare_cv_data(Z, coord_arrays, emb_arrays)
        folds = self._make_cv_folds(Z_cv.shape[0])
        if not folds:
            return hm_candidates[len(hm_candidates) // 2], hz_candidates[len(hz_candidates) // 2]

        q_points, q_weights, _ = self._validation_quadrature()
        search = str(self.config.get("bandwidth_search", "coarse_to_fine")).lower()
        if search == "full":
            best = self._evaluate_bandwidth_pairs(hm_candidates, hz_candidates, Z_cv, coord_cv, emb_cv, folds, q_points, q_weights, stage="full")
            self.selected_bandwidth_cv_score = best[0]
            return best[1], best[2]
        if search != "coarse_to_fine":
            raise ValueError(f"Unknown bandwidth_search: {search}")
        coarse_hm = self._coarse_candidates(hm_candidates, Z_cv.shape[0])
        coarse_hz = [1.0] if Z_cv.shape[1] == 0 else self._coarse_candidates(hz_candidates, Z_cv.shape[0])
        best = self._evaluate_bandwidth_pairs(coarse_hm, coarse_hz, Z_cv, coord_cv, emb_cv, folds, q_points, q_weights, stage="coarse")
        fine_hm = self._fine_candidates_around(best[1])
        fine_hz = [1.0] if Z_cv.shape[1] == 0 else self._fine_candidates_around(best[2])
        best = self._evaluate_bandwidth_pairs(fine_hm, fine_hz, Z_cv, coord_cv, emb_cv, folds, q_points, q_weights, current_best=best, stage="fine")
        self.selected_bandwidth_cv_score = best[0]
        return best[1], best[2]

    def _evaluate_bandwidth_pairs(
        self,
        hm_candidates: list[float],
        hz_candidates: list[float],
        Z: np.ndarray,
        arrays: list[np.ndarray],
        emb_arrays: list[np.ndarray],
        folds: list[tuple[np.ndarray, np.ndarray]],
        q_points: np.ndarray,
        q_weights: np.ndarray,
        current_best: tuple[float, float, float] | None = None,
        stage: str = "coarse",
    ) -> tuple[float, float, float]:
        best = current_best if current_best is not None else (math.inf, hm_candidates[0], hz_candidates[0])
        seen = {(score["bandwidth_m"], score["bandwidth_z"]) for score in self.validation_scores}
        for hm, hz in itertools.product(hm_candidates, hz_candidates):
            if (float(hm), float(hz)) in seen:
                continue
            self.bandwidth_m, self.bandwidth_z = float(hm), float(hz)
            start = time.perf_counter()
            fold_rows = []
            for fold_id, (train_idx, val_idx) in enumerate(folds):
                self.Z = Z[train_idx]
                self._store_events(arrays, emb_arrays, train_idx)
                fold_nll = self._validation_nll(Z, arrays, val_idx, q_points, q_weights)
                fold_rows.append(
                    {
                        "candidate_id": len(self.validation_scores),
                        "stage": stage,
                        "fold": fold_id,
                        "h_x": None,
                        "h_z": None if Z.shape[1] == 0 else float(hz),
                        "h_m": float(hm),
                        "criterion": "ward_poisson_cv",
                        "fold_nll": float(fold_nll),
                        "bandwidth_selection": self.config.get("bandwidth_selection", "ward_cv"),
                        "bandwidth_theory_mode": self.config.get("bandwidth_theory_mode", "ward_geodesic_grid"),
                        "n_train": int(train_idx.shape[0]),
                        "N_total_train": int(sum(arrays[i].shape[0] for i in train_idx)),
                        "event_dim": int(self.metadata["event_dim"]),
                        "z_dim": int(Z.shape[1]),
                    }
                )
            fold_vals = np.asarray([row["fold_nll"] for row in fold_rows], dtype=np.float64)
            score = float(np.mean(fold_vals))
            std = float(np.std(fold_vals, ddof=1)) if fold_vals.size > 1 else 0.0
            for row in fold_rows:
                row["mean_nll"] = score
                row["std_nll"] = std
                self.cv_results.append(row)
            self.profile["validation_nll_time_seconds"] += time.perf_counter() - start
            self.validation_scores.append({"bandwidth_m": float(hm), "bandwidth_z": float(hz), "val_nll": score})
            if score < best[0]:
                best = (score, float(hm), float(hz))
        return best

    def _validation_nll(self, Z: np.ndarray, arrays: list[np.ndarray], val_idx: np.ndarray, q_points: np.ndarray, q_weights: np.ndarray) -> float:
        if val_idx.size == 0:
            return math.inf
        Z_val = Z[val_idx]
        losses = self._integral_terms_for_covariates(q_points, q_weights, Z_val)

        counts = np.asarray([arrays[int(idx)].shape[0] for idx in val_idx], dtype=np.int64)
        total_events = int(np.sum(counts))
        if total_events:
            obs = np.concatenate([arrays[int(idx)] for idx in val_idx if arrays[int(idx)].shape[0]], axis=0)
            local_owner = np.repeat(np.arange(val_idx.shape[0], dtype=np.int64), counts)
            if Z.shape[1] == 0:
                event_z = np.empty((obs.shape[0], 0), dtype=np.float64)
            else:
                event_z = Z_val[local_owner]
            pred_obs = self.predict(obs, event_z)
            np.add.at(losses, local_owner, -np.log(np.maximum(pred_obs, 1e-12)))
        return float(np.mean(losses))

    def _integral_terms_for_covariates(self, q_points: np.ndarray, q_weights: np.ndarray, Z_val: np.ndarray) -> np.ndarray:
        if Z_val.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        if self.event_owner.shape[0] == 0:
            return np.full(Z_val.shape[0], float(np.sum(q_weights)) * 1e-8, dtype=np.float64)
        q_coords, q_embedded = self._coerce_points(np.asarray(q_points, dtype=np.float64))
        z_dim = self.Z.shape[1]
        out = np.zeros(Z_val.shape[0], dtype=np.float64)
        z_chunk = self._z_chunk_size(Z_val.shape[0])
        for z_sl in _chunk_slices(Z_val.shape[0], z_chunk):
            Z_chunk = Z_val[z_sl]
            if z_dim == 0:
                denom = np.full(Z_chunk.shape[0], max(self.Z.shape[0], 1), dtype=np.float64)
                Kz_reps = None
            else:
                dz2 = self._raw_sqdist(Z_chunk, self.Z, "z_replicate") / max(float(self.bandwidth_z) ** 2, 1e-24)
                Kz_reps = gaussian_from_scaled_sqdist(dz2, z_dim, float(self.bandwidth_z), self._gaussian_cutoff())
                denom = np.maximum(Kz_reps.sum(axis=1), float(self.config["a_n"]))

            integral = np.zeros(Z_chunk.shape[0], dtype=np.float64)
            q_chunk = self._query_chunk_size(q_coords.shape[0])
            for q_sl in _chunk_slices(q_coords.shape[0], q_chunk):
                coords = q_coords[q_sl]
                embedded = q_embedded[q_sl]
                w = q_weights[q_sl]
                numerator = np.zeros((Z_chunk.shape[0], coords.shape[0]), dtype=np.float64)
                ev_chunk = self._event_chunk_size_multi(coords.shape[0], Z_chunk.shape[0])
                for ev_sl in _chunk_slices(self.event_owner.shape[0], ev_chunk):
                    Km = self._manifold_kernel_chunk(coords, embedded, ev_sl)
                    if Kz_reps is None:
                        numerator += Km.sum(axis=1, keepdims=True).T
                    else:
                        numerator += Kz_reps[:, self.event_owner[ev_sl]] @ Km.T
                integral += (numerator / np.maximum(denom[:, None], float(self.config["a_n"]))) @ w
            out[z_sl] = integral
        return out

    def _store_events(self, coord_arrays: list[np.ndarray], emb_arrays: list[np.ndarray], indices: np.ndarray) -> None:
        index_key = tuple(int(i) for i in indices.tolist())
        changed_training_set = index_key != self._last_training_indices
        cached = self._fold_event_cache.get(index_key)
        if cached is not None:
            self.event_coords, self.event_embedded, self.event_owner, self.event_z = cached
            if changed_training_set:
                self._training_token += 1
                self._last_training_indices = index_key
            return
        selected_coords = [coord_arrays[i] for i in indices]
        selected_emb = [emb_arrays[i] for i in indices]
        if sum(arr.shape[0] for arr in selected_coords):
            self.event_coords = np.concatenate(selected_coords, axis=0).astype(np.float64)
            self.event_embedded = np.concatenate(selected_emb, axis=0).astype(np.float64)
            self.event_owner = np.repeat(np.arange(len(selected_coords), dtype=np.int64), [arr.shape[0] for arr in selected_coords])
        else:
            self.event_coords = np.empty((0, int(self.metadata["coord_dim"])), dtype=np.float64)
            self.event_embedded = np.empty((0, int(self.metadata["embedding_dim"])), dtype=np.float64)
            self.event_owner = np.empty((0,), dtype=np.int64)
        if self.Z.shape[1] == 0:
            self.event_z = np.empty((self.event_owner.shape[0], 0), dtype=np.float64)
        else:
            self.event_z = self.Z[self.event_owner]
        if self.metadata.get("manifold_type") == "sphere" and self.event_embedded.shape[0]:
            self.event_embedded = self.event_embedded / np.maximum(np.linalg.norm(self.event_embedded, axis=1, keepdims=True), 1e-12)
        self._fold_event_cache[index_key] = (self.event_coords, self.event_embedded, self.event_owner, self.event_z)
        if changed_training_set:
            self._training_token += 1
            self._last_training_indices = index_key

    def _prepare_cv_data(
        self,
        Z: np.ndarray,
        coord_arrays: list[np.ndarray],
        emb_arrays: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        rng = np.random.default_rng(int(self.config.get("seed", 0)) + 2718)
        n = Z.shape[0]
        max_reps = int(self.config.get("kernel_cv_max_replicates") or 0)
        if max_reps > 0 and n > max_reps:
            keep = np.sort(rng.choice(n, size=max_reps, replace=False))
        else:
            keep = np.arange(n, dtype=np.int64)

        max_events = int(self.config.get("kernel_cv_max_events_per_replicate") or 0)
        cv_coords: list[np.ndarray] = []
        cv_emb: list[np.ndarray] = []
        for idx in keep:
            coords = coord_arrays[int(idx)]
            emb = emb_arrays[int(idx)]
            if max_events > 0 and coords.shape[0] > max_events:
                event_idx = np.sort(rng.choice(coords.shape[0], size=max_events, replace=False))
                cv_coords.append(coords[event_idx])
                cv_emb.append(emb[event_idx])
            else:
                cv_coords.append(coords)
                cv_emb.append(emb)
        return Z[keep], cv_coords, cv_emb

    def _validation_quadrature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        rng = np.random.default_rng(int(self.config["seed"]) + 991)
        metadata = dict(self.metadata)
        metadata["cache_dir"] = self.config.get("cache_dir", "cache")
        metadata["mesh_resolution"] = self.config.get("mesh_resolution", 64)
        return make_quadrature(metadata, int(self.config["validation_integration_points"]), rng)

    def _raw_sqdist(self, A: np.ndarray, B: np.ndarray, distance_type: str) -> np.ndarray:
        entries = int(A.shape[0]) * int(B.shape[0])
        key = {
            "method": self.method_name,
            "support_type": self.metadata.get("support_type"),
            "dataset_hash": self._dataset_hash,
            "query_grid_hash": array_hash(A),
            "target_hash": array_hash(B),
            "distance_type": distance_type,
            "z_dim": self.metadata.get("z_dim"),
            "event_dim": self.metadata.get("event_dim"),
            "manifold_type": self.metadata.get("manifold_type"),
        }
        cache_limit = min(int(self.config.get("max_dense_entries", 20_000_000)), 5_000_000)
        use_cache = entries <= cache_limit and A.shape[0] >= 64
        return self.distance_cache.get_or_compute(key, lambda: raw_sqdist(A, B), persist=False) if use_cache else raw_sqdist(A, B)

    def _sphere_dots(self, embedded: np.ndarray, event_embedded: np.ndarray) -> np.ndarray:
        embedded = embedded / np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-12)
        start = time.perf_counter()
        dots = embedded @ event_embedded.T
        self.distance_cache.stats["seconds"] += time.perf_counter() - start
        return dots

    def _default_hm_candidates(self) -> list[float]:
        cfg = self.config.get("bandwidth_m_candidates")
        if cfg is not None:
            return [float(x) for x in cfg]
        return [0.08, 0.16, 0.32, 0.64, 0.90] if self.metadata.get("manifold_type") == "circle" else [0.12, 0.20, 0.32, 0.50, 0.80]

    def _default_hz_candidates(self, z_dim: int) -> list[float]:
        if z_dim == 0:
            return [1.0]
        cfg = self.config.get("bandwidth_z_candidates")
        return [float(x) for x in (cfg if cfg is not None else [0.12, 0.20, 0.32, 0.50, 0.80])]

    def _coarse_candidates(self, candidates: list[float], n: int) -> list[float]:
        size = int(self.config.get("bandwidth_grid_size", 5))
        if n >= 10_000:
            size = min(size, 4)
        elif n >= 3_000:
            size = min(size, 5)
        if len(candidates) <= size:
            return list(candidates)
        idx = np.linspace(0, len(candidates) - 1, size).round().astype(int)
        return [float(candidates[i]) for i in np.unique(idx)]

    def _fine_candidates(self, candidates: list[float], best: float) -> list[float]:
        size = int(self.config.get("bandwidth_fine_grid_size", 3))
        if size <= 1 or len(candidates) <= 1:
            return [float(best)]
        arr = np.asarray(sorted(set(float(x) for x in candidates)), dtype=np.float64)
        pos = int(np.argmin(np.abs(arr - float(best))))
        lo = arr[max(0, pos - 1)]
        hi = arr[min(len(arr) - 1, pos + 1)]
        if lo <= 0 or hi <= 0 or lo == hi:
            return [float(best)]
        return [float(x) for x in np.geomspace(lo, hi, size)]

    def _select_bandwidths_ward_np(
        self,
        Z: np.ndarray,
        coord_arrays: list[np.ndarray],
        emb_arrays: list[np.ndarray],
        hm_candidates: list[float],
        hz_candidates: list[float],
    ) -> tuple[float, float, float]:
        self.Z = Z
        self._store_events(coord_arrays, emb_arrays, np.arange(Z.shape[0]))
        if self.event_coords.shape[0] == 0:
            return (math.inf, hm_candidates[len(hm_candidates) // 2], hz_candidates[len(hz_candidates) // 2])
        search = str(self.config.get("bandwidth_search", "coarse_to_fine")).lower()
        if search == "coarse_to_fine":
            coarse_hm = self._coarse_candidates(hm_candidates, Z.shape[0])
            coarse_hz = [1.0] if Z.shape[1] == 0 else self._coarse_candidates(hz_candidates, Z.shape[0])
            best = self._evaluate_ward_np_pairs(coarse_hm, coarse_hz, Z, stage="coarse")
        else:
            best = self._evaluate_ward_np_pairs(hm_candidates, hz_candidates, Z, stage="full")
        if search == "coarse_to_fine":
            fine_hm = self._fine_candidates_around(best[1])
            fine_hz = [1.0] if Z.shape[1] == 0 else self._fine_candidates_around(best[2])
            best = self._evaluate_ward_np_pairs(fine_hm, fine_hz, Z, current_best=best, stage="fine")
        elif search != "full":
            raise ValueError(f"Unknown bandwidth_search: {search}")
        return best

    def _evaluate_ward_np_pairs(
        self,
        hm_candidates: list[float],
        hz_candidates: list[float],
        Z: np.ndarray,
        current_best: tuple[float, float, float] | None = None,
        stage: str = "coarse",
    ) -> tuple[float, float, float]:
        best = current_best if current_best is not None else (math.inf, hm_candidates[0], hz_candidates[0])
        seen = {(score["bandwidth_m"], score["bandwidth_z"]) for score in self.validation_scores}
        query_z = self.event_z if Z.shape[1] else np.empty((self.event_coords.shape[0], 0), dtype=np.float64)
        target = max(float(Z.shape[0]) * float(self.metadata.get("volume", 1.0)), 1e-12)
        for hm, hz in itertools.product(hm_candidates, hz_candidates):
            if (float(hm), float(hz)) in seen:
                continue
            self.bandwidth_m, self.bandwidth_z = float(hm), float(hz)
            preds = self.predict(self.event_coords, query_z)
            T = float(np.sum(1.0 / np.maximum(preds, 1e-12)))
            score = float((T - target) ** 2)
            row = {
                "candidate_id": len(self.validation_scores),
                "stage": stage,
                "fold": 0,
                "h_x": None,
                "h_z": None if Z.shape[1] == 0 else float(hz),
                "h_m": float(hm),
                "criterion": "ward_np",
                "fold_nll": score,
                "mean_nll": score,
                "std_nll": 0.0,
                "bandwidth_selection": "ward_np",
                "bandwidth_theory_mode": self.config.get("bandwidth_theory_mode", "ward_geodesic_grid"),
                "n_train": int(Z.shape[0]),
                "N_total_train": int(self.event_coords.shape[0]),
                "event_dim": int(self.metadata["event_dim"]),
                "z_dim": int(Z.shape[1]),
            }
            self.cv_results.append(row)
            self.validation_scores.append({"bandwidth_m": float(hm), "bandwidth_z": float(hz), "val_nll": score})
            if score < best[0]:
                best = (score, float(hm), float(hz))
        return best

    def _make_cv_folds(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if n <= 1:
            return []
        k = min(max(int(self.config.get("bandwidth_cv_folds", 5)), 2), n)
        rng = np.random.default_rng(int(self.config.get("seed", 0)) + 1701)
        indices = np.arange(n, dtype=np.int64)
        rng.shuffle(indices)
        val_splits = np.array_split(indices, k)
        folds = []
        all_idx = np.arange(n, dtype=np.int64)
        for val_idx in val_splits:
            if val_idx.size == 0:
                continue
            mask = np.ones(n, dtype=bool)
            mask[val_idx] = False
            train_idx = all_idx[mask]
            if train_idx.size:
                folds.append((train_idx, np.asarray(val_idx, dtype=np.int64)))
        return folds

    def _fine_candidates_around(self, best: float) -> list[float]:
        multipliers = np.asarray(self.config.get("bandwidth_fine_multipliers", [0.8, 1.0, 1.25]), dtype=np.float64)
        return self._clip_bandwidths(float(best) * multipliers)

    def _clip_bandwidths(self, values: np.ndarray) -> list[float]:
        lo = float(self.config.get("bandwidth_min", 0.03))
        hi = float(self.config.get("bandwidth_max", 1.25))
        clipped = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
        return [float(x) for x in np.unique(np.round(clipped, 12))]

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
            return Z.astype(np.float64)
        if Z.shape[0] == 1:
            return np.repeat(Z.astype(np.float64), n, axis=0)
        raise ValueError(f"Cannot broadcast Z shape {Z.shape} to {n} rows")

    def _query_chunk_size(self, n_query: int) -> int:
        requested = int(self.config.get("kernel_chunk_size", 4096))
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        n_reps = max(self.Z.shape[0], 1)
        by_reps = max(1, max_entries // n_reps)
        return max(1, min(requested, n_query, by_reps))

    def _event_chunk_size(self, n_query_chunk: int) -> int:
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        return max(1, min(self.event_owner.shape[0], max_entries // max(int(n_query_chunk), 1)))

    def _event_chunk_size_multi(self, n_query_chunk: int, n_z_chunk: int) -> int:
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        denom = max(int(n_query_chunk) + int(n_z_chunk), 1)
        return max(1, min(self.event_owner.shape[0], max_entries // denom))

    def _z_chunk_size(self, n_z: int) -> int:
        requested = int(self.config.get("validation_z_chunk_size", 256))
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        by_reps = max(1, max_entries // max(self.Z.shape[0], 1))
        return max(1, min(int(n_z), requested, by_reps))

    def _gaussian_cutoff(self) -> float | None:
        if not bool(self.config.get("use_gaussian_cutoff", True)):
            return None
        cutoff = self.config.get("gaussian_cutoff")
        return None if cutoff is None else float(cutoff)

    def _hash_dataset(self, Z: np.ndarray, arrays: list[np.ndarray]) -> str:
        counts = np.asarray([arr.shape[0] for arr in arrays], dtype=np.int64)
        pieces = [array_hash(Z), array_hash(counts)]
        if sum(counts):
            pieces.append(array_hash(np.concatenate(arrays, axis=0)))
        return "_".join(pieces)

    def _sync_cache_stats(self) -> None:
        self.profile["distance_computation_time_seconds"] = float(self.distance_cache.stats["seconds"])
        self.profile["distance_cache_hits"] = int(self.distance_cache.stats["hits"])
        self.profile["distance_cache_misses"] = int(self.distance_cache.stats["misses"])


def _chunk_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))
