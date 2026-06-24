from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np

from ..cache_utils import DistanceCache, array_hash
from ..integration import make_quadrature
from .base import Estimator


class EuclideanKernelEstimator(Estimator):
    method_name = "kernel_euclidean"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {
            "mode": "spatial_only",
            "kernel": "gaussian",
            "a_n": 1e-6,
            "boundary_correction": "renormalize",
            "boundary_correction_eps": 1e-8,
            "bandwidth_x": None,
            "bandwidth_z": None,
            "bandwidth_x_candidates": None,
            "bandwidth_z_candidates": None,
            "bandwidth_selection": "theory_guided_5fold_cv",
            "bandwidth_cv_folds": 5,
            "bandwidth_theory_mode": "separate_spatial_covariate",
            "bandwidth_scale_x": 1.0,
            "bandwidth_scale_z": 1.0,
            "bandwidth_min": 0.02,
            "bandwidth_max": 1.25,
            "bandwidth_coarse_multipliers": [0.5, 0.75, 1.0, 1.5, 2.0],
            "bandwidth_highdim_multipliers": [0.75, 1.0, 1.5, 2.0, 3.0],
            "bandwidth_fine_multipliers": [0.8, 1.0, 1.25],
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
            "cache_dir": "cache",
            "use_distance_cache": True,
            "seed": 0,
        }
        if config:
            self.config.update({k: v for k, v in config.items() if v is not None})
        self.metadata: dict[str, Any] = {}
        self.Z = np.empty((0, 0), dtype=np.float64)
        self.event_points = np.empty((0, 0), dtype=np.float64)
        self.event_z = np.empty((0, 0), dtype=np.float64)
        self.event_owner = np.empty((0,), dtype=np.int64)
        self.bandwidth_x = None
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
        self._tree_cache: dict[tuple[Any, ...], Any] = {}
        self._fold_event_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["distance_cache"] = None
        state["_tree_cache"] = {}
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
        self._tree_cache = {}
        self._fold_event_cache = {}

    def fit(self, data: dict[str, Any]) -> "EuclideanKernelEstimator":
        start_fit = time.perf_counter()
        self.metadata = dict(data["metadata"])
        if self.metadata["support_type"] != "euclidean":
            raise ValueError("EuclideanKernelEstimator supports Euclidean event spaces only")
        Z = np.asarray(data["Z"], dtype=np.float64)
        arrays = [np.asarray(arr, dtype=np.float64) for arr in data["event_coords"]]
        self._dataset_hash = self._hash_dataset(Z, arrays)

        start_bw = time.perf_counter()
        self.bandwidth_x, self.bandwidth_z = self._select_bandwidths(Z, arrays)
        self.profile["bandwidth_selection_time_seconds"] += time.perf_counter() - start_bw
        self.Z = Z
        self._store_events(arrays, np.arange(Z.shape[0]))
        self.profile["fit_time_seconds"] += time.perf_counter() - start_fit
        self._sync_cache_stats()
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Zb = self._broadcast_z(X.shape[0], np.asarray(Z, dtype=np.float64))
        if self.event_points.shape[0] == 0:
            return np.full(X.shape[0], 1e-8, dtype=np.float64)
        out = np.zeros(X.shape[0], dtype=np.float64)
        q_chunk = self._query_chunk_size(X.shape[0])
        for sl in _chunk_slices(X.shape[0], q_chunk):
            out[sl] = self._predict_chunk(X[sl], Zb[sl])
        self.profile["prediction_time_seconds"] += time.perf_counter() - start
        self._sync_cache_stats()
        return np.maximum(out, 1e-12)

    def _predict_chunk(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.config.get("kernel", "gaussian") == "epanechnikov":
            return self._predict_epanechnikov_chunk(X, Z)
        return self._predict_gaussian_chunk(X, Z)

    def _predict_gaussian_chunk(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        mode = self.config["mode"]
        z_dim = self.Z.shape[1]
        if mode == "spatial_only" or z_dim == 0:
            denom = np.full(X.shape[0], max(self.Z.shape[0], 1), dtype=np.float64)
            Kz_reps = None
        else:
            dz2 = self._raw_sqdist(Z, self.Z, "z_replicate") / max(float(self.bandwidth_z) ** 2, 1e-24)
            Kz_reps = gaussian_from_scaled_sqdist(
                dz2,
                z_dim,
                float(self.bandwidth_z),
                self._gaussian_cutoff(),
            )
            denom = np.maximum(Kz_reps.sum(axis=1), float(self.config["a_n"]))

        numerator = np.zeros(X.shape[0], dtype=np.float64)
        boundary_mass = self._event_boundary_mass(X)[:, None]
        ev_chunk = self._event_chunk_size(X.shape[0])
        for ev_sl in _chunk_slices(self.event_points.shape[0], ev_chunk):
            dx2 = self._raw_sqdist(X, self.event_points[ev_sl], "x_event") / max(float(self.bandwidth_x) ** 2, 1e-24)
            Kx = gaussian_from_scaled_sqdist(
                dx2,
                int(self.metadata["event_dim"]),
                float(self.bandwidth_x),
                self._gaussian_cutoff(),
            )
            Kx = Kx / boundary_mass
            if Kz_reps is not None:
                numerator += np.sum(Kx * Kz_reps[:, self.event_owner[ev_sl]], axis=1)
            else:
                numerator += np.sum(Kx, axis=1)
        return numerator / np.maximum(denom, float(self.config["a_n"]))

    def _predict_epanechnikov_chunk(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        trees = self._epanechnikov_trees()
        mode = self.config["mode"]
        z_dim = self.Z.shape[1]
        out = np.zeros(X.shape[0], dtype=np.float64)
        for i, (x, z) in enumerate(zip(X, Z)):
            if mode == "spatial_only" or z_dim == 0:
                denom = float(max(self.Z.shape[0], 1))
            else:
                z_scaled = z / max(float(self.bandwidth_z), 1e-12)
                cand_reps = trees["z_tree"].query_ball_point(z_scaled, math.sqrt(z_dim))
                if cand_reps:
                    dz = z.reshape(1, -1) - self.Z[np.asarray(cand_reps)]
                    denom = float(np.sum(product_kernel(dz, float(self.bandwidth_z), "epanechnikov")))
                else:
                    denom = 0.0
                denom = max(denom, float(self.config["a_n"]))

            if mode == "spatial_only" or z_dim == 0:
                query = x / max(float(self.bandwidth_x), 1e-12)
            else:
                query = np.concatenate([x / max(float(self.bandwidth_x), 1e-12), z / max(float(self.bandwidth_z), 1e-12)])
            candidates = trees["joint_tree"].query_ball_point(query, trees["joint_radius"])
            if not candidates:
                out[i] = 0.0
                continue
            idx = np.asarray(candidates, dtype=np.int64)
            kx = product_kernel(x.reshape(1, -1) - self.event_points[idx], float(self.bandwidth_x), "epanechnikov")
            kx = kx / self._event_boundary_mass(x.reshape(1, -1))[0]
            if mode == "spatial_only" or z_dim == 0:
                kz = np.ones(idx.shape[0], dtype=np.float64)
            else:
                kz = product_kernel(z.reshape(1, -1) - self.event_z[idx], float(self.bandwidth_z), "epanechnikov")
            out[i] = float(np.sum(kx * kz)) / max(denom, float(self.config["a_n"]))
        return out

    def _select_bandwidths(self, Z: np.ndarray, arrays: list[np.ndarray]) -> tuple[float, float]:
        hx_fixed = self.config.get("bandwidth_x")
        hz_fixed = self.config.get("bandwidth_z")
        hx_candidates, hz_candidates = self._bandwidth_candidates(Z, arrays)
        if hx_fixed:
            hx_candidates = [float(hx_fixed)]
        if hz_fixed:
            hz_candidates = [float(hz_fixed)]
        if len(hx_candidates) == 1 and len(hz_candidates) == 1:
            self.selected_bandwidth_cv_score = None
            return hx_candidates[0], hz_candidates[0]

        selection = str(self.config.get("bandwidth_selection", "theory_guided_5fold_cv")).lower()
        if selection not in {"theory_guided_5fold_cv", "poisson_5fold_cv", "validation_nll"}:
            raise ValueError(f"Unsupported bandwidth_selection for Euclidean kernels: {selection}")
        Z_cv, arrays_cv = self._prepare_cv_data(Z, arrays)
        folds = self._make_cv_folds(Z_cv.shape[0])
        if not folds:
            return hx_candidates[len(hx_candidates) // 2], hz_candidates[len(hz_candidates) // 2]

        q_points, q_weights, _ = self._validation_quadrature()
        search = str(self.config.get("bandwidth_search", "coarse_to_fine")).lower()
        if search == "full":
            best = self._evaluate_bandwidth_pairs(hx_candidates, hz_candidates, Z_cv, arrays_cv, folds, q_points, q_weights, stage="full")
            self.selected_bandwidth_cv_score = best[0]
            return best[1], best[2]
        elif search == "coarse_to_fine":
            coarse_hx = self._coarse_candidates(hx_candidates, Z_cv.shape[0])
            coarse_hz = [1.0] if Z_cv.shape[1] == 0 else self._coarse_candidates(hz_candidates, Z_cv.shape[0])
            best = self._evaluate_bandwidth_pairs(coarse_hx, coarse_hz, Z_cv, arrays_cv, folds, q_points, q_weights, stage="coarse")
            fine_hx = self._fine_candidates_around(best[1])
            fine_hz = [1.0] if Z_cv.shape[1] == 0 else self._fine_candidates_around(best[2])
            best = self._evaluate_bandwidth_pairs(fine_hx, fine_hz, Z_cv, arrays_cv, folds, q_points, q_weights, current_best=best, stage="fine")
            self.selected_bandwidth_cv_score = best[0]
            return best[1], best[2]
        else:
            raise ValueError(f"Unknown bandwidth_search: {search}")

    def _evaluate_bandwidth_pairs(
        self,
        hx_candidates: list[float],
        hz_candidates: list[float],
        Z: np.ndarray,
        arrays: list[np.ndarray],
        folds: list[tuple[np.ndarray, np.ndarray]],
        q_points: np.ndarray,
        q_weights: np.ndarray,
        current_best: tuple[float, float, float] | None = None,
        stage: str = "coarse",
    ) -> tuple[float, float, float]:
        best = current_best if current_best is not None else (math.inf, hx_candidates[0], hz_candidates[0])
        seen = {(score["bandwidth_x"], score["bandwidth_z"]) for score in self.validation_scores}
        for hx, hz in itertools.product(hx_candidates, hz_candidates):
            if (float(hx), float(hz)) in seen:
                continue
            self.bandwidth_x, self.bandwidth_z = float(hx), float(hz)
            self._tree_cache.clear()
            start = time.perf_counter()
            fold_rows = []
            for fold_id, (train_idx, val_idx) in enumerate(folds):
                self.Z = Z[train_idx]
                self._store_events(arrays, train_idx)
                fold_nll = self._validation_nll(Z, arrays, val_idx, q_points, q_weights)
                fold_rows.append(
                    {
                        "candidate_id": len(self.validation_scores),
                        "stage": stage,
                        "fold": fold_id,
                        "h_x": float(hx),
                        "h_z": None if Z.shape[1] == 0 else float(hz),
                        "h_m": None,
                        "criterion": "poisson_nll",
                        "fold_nll": float(fold_nll),
                        "bandwidth_selection": self.config.get("bandwidth_selection", "theory_guided_5fold_cv"),
                        "bandwidth_theory_mode": self.config.get("bandwidth_theory_mode", "separate_spatial_covariate"),
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
            self.validation_scores.append({"bandwidth_x": float(hx), "bandwidth_z": float(hz), "val_nll": score})
            if score < best[0]:
                best = (score, float(hx), float(hz))
        return best

    def _validation_nll(
        self,
        Z: np.ndarray,
        arrays: list[np.ndarray],
        val_idx: np.ndarray,
        q_points: np.ndarray,
        q_weights: np.ndarray,
    ) -> float:
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
        if self.event_points.shape[0] == 0:
            return np.full(Z_val.shape[0], float(np.sum(q_weights)) * 1e-8, dtype=np.float64)

        z_dim = self.Z.shape[1]
        out = np.zeros(Z_val.shape[0], dtype=np.float64)
        z_chunk = self._z_chunk_size(Z_val.shape[0])
        for z_sl in _chunk_slices(Z_val.shape[0], z_chunk):
            Z_chunk = Z_val[z_sl]
            if self.config["mode"] == "spatial_only" or z_dim == 0:
                denom = np.full(Z_chunk.shape[0], max(self.Z.shape[0], 1), dtype=np.float64)
                Kz_reps = None
            else:
                dz2 = self._raw_sqdist(Z_chunk, self.Z, "z_replicate") / max(float(self.bandwidth_z) ** 2, 1e-24)
                Kz_reps = gaussian_from_scaled_sqdist(dz2, z_dim, float(self.bandwidth_z), self._gaussian_cutoff())
                denom = np.maximum(Kz_reps.sum(axis=1), float(self.config["a_n"]))

            integral = np.zeros(Z_chunk.shape[0], dtype=np.float64)
            q_chunk = self._query_chunk_size(q_points.shape[0])
            for q_sl in _chunk_slices(q_points.shape[0], q_chunk):
                q = q_points[q_sl]
                w = q_weights[q_sl]
                numerator = np.zeros((Z_chunk.shape[0], q.shape[0]), dtype=np.float64)
                ev_chunk = self._event_chunk_size_multi(q.shape[0], Z_chunk.shape[0])
                for ev_sl in _chunk_slices(self.event_points.shape[0], ev_chunk):
                    Kx = self._event_kernel_matrix(q, self.event_points[ev_sl])
                    if Kz_reps is None:
                        numerator += Kx.sum(axis=1, keepdims=True).T
                    else:
                        numerator += Kz_reps[:, self.event_owner[ev_sl]] @ Kx.T
                integral += (numerator / np.maximum(denom[:, None], float(self.config["a_n"]))) @ w
            out[z_sl] = integral
        return out

    def _validation_quadrature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        rng = np.random.default_rng(int(self.config["seed"]) + 991)
        return make_quadrature(
            self.metadata,
            int(self.config["validation_integration_points"]),
            rng,
            method=str(self.config.get("validation_quadrature_method", "uniform")),
        )

    def _store_events(self, arrays: list[np.ndarray], indices: np.ndarray) -> None:
        index_key = tuple(int(i) for i in indices.tolist())
        changed_training_set = index_key != self._last_training_indices
        cached = self._fold_event_cache.get(index_key)
        if cached is not None:
            self.event_points, self.event_owner, self.event_z = cached
            if changed_training_set:
                self._training_token += 1
                self._last_training_indices = index_key
                self._tree_cache.clear()
            return
        selected = [arrays[i] for i in indices]
        event_dim = int(self.metadata["event_dim"])
        if sum(arr.shape[0] for arr in selected):
            self.event_points = np.concatenate(selected, axis=0).astype(np.float64)
            self.event_owner = np.repeat(np.arange(len(selected), dtype=np.int64), [arr.shape[0] for arr in selected])
        else:
            self.event_points = np.empty((0, event_dim), dtype=np.float64)
            self.event_owner = np.empty((0,), dtype=np.int64)
        if self.Z.shape[1] == 0:
            self.event_z = np.empty((self.event_owner.shape[0], 0), dtype=np.float64)
        else:
            self.event_z = self.Z[self.event_owner]
        self._fold_event_cache[index_key] = (self.event_points, self.event_owner, self.event_z)
        if changed_training_set:
            self._training_token += 1
            self._last_training_indices = index_key
            self._tree_cache.clear()

    def _prepare_cv_data(self, Z: np.ndarray, arrays: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        rng = np.random.default_rng(int(self.config.get("seed", 0)) + 2718)
        n = Z.shape[0]
        max_reps = int(self.config.get("kernel_cv_max_replicates") or 0)
        if max_reps > 0 and n > max_reps:
            keep = np.sort(rng.choice(n, size=max_reps, replace=False))
        else:
            keep = np.arange(n, dtype=np.int64)

        max_events = int(self.config.get("kernel_cv_max_events_per_replicate") or 0)
        cv_arrays: list[np.ndarray] = []
        for idx in keep:
            arr = arrays[int(idx)]
            if max_events > 0 and arr.shape[0] > max_events:
                event_idx = np.sort(rng.choice(arr.shape[0], size=max_events, replace=False))
                cv_arrays.append(arr[event_idx])
            else:
                cv_arrays.append(arr)
        return Z[keep], cv_arrays

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
        # Event-loglikelihood queries are usually tiny and rarely reused; caching them
        # can consume much more memory than it saves. Integration/evaluation grids are
        # larger and are reused across bandwidth candidates, so cache those.
        use_cache = entries <= cache_limit and A.shape[0] >= 64
        return self.distance_cache.get_or_compute(key, lambda: raw_sqdist(A, B), persist=False) if use_cache else raw_sqdist(A, B)

    def _event_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        kernel = str(self.config.get("kernel", "gaussian")).lower()
        if kernel == "epanechnikov":
            diff = X[:, None, :] - Y[None, :, :]
            Kx = product_kernel(diff, float(self.bandwidth_x), "epanechnikov")
        else:
            dx2 = self._raw_sqdist(X, Y, "x_event") / max(float(self.bandwidth_x) ** 2, 1e-24)
            Kx = gaussian_from_scaled_sqdist(
                dx2,
                int(self.metadata["event_dim"]),
                float(self.bandwidth_x),
                self._gaussian_cutoff(),
            )
        return Kx / self._event_boundary_mass(X)[:, None]

    def _event_boundary_mass(self, X: np.ndarray) -> np.ndarray:
        correction = str(self.config.get("boundary_correction", "none")).lower()
        if correction in {"none", "false", "off", "0"}:
            return np.ones(X.shape[0], dtype=np.float64)
        if correction not in {"renormalize", "renormalise", "normalize", "normalise"}:
            raise ValueError(f"Unknown boundary_correction: {correction}")
        eps = float(self.config.get("boundary_correction_eps", 1e-8))
        mass = unit_cube_boundary_mass(
            np.asarray(X, dtype=np.float64),
            float(self.bandwidth_x),
            str(self.config.get("kernel", "gaussian")).lower(),
        )
        return np.maximum(mass, eps)

    def _epanechnikov_trees(self):
        try:
            from scipy.spatial import cKDTree
        except Exception as exc:
            raise RuntimeError("Epanechnikov fast path requires scipy.spatial.cKDTree") from exc
        key = (self._training_token, float(self.bandwidth_x), float(self.bandwidth_z), self.config["mode"], self.Z.shape[1])
        if key in self._tree_cache:
            return self._tree_cache[key]
        z_dim = self.Z.shape[1]
        if self.config["mode"] == "spatial_only" or z_dim == 0:
            joint = self.event_points / max(float(self.bandwidth_x), 1e-12)
            radius = math.sqrt(int(self.metadata["event_dim"]))
            payload = {"joint_tree": cKDTree(joint), "joint_radius": radius, "z_tree": None}
        else:
            joint = np.column_stack(
                [
                    self.event_points / max(float(self.bandwidth_x), 1e-12),
                    self.event_z / max(float(self.bandwidth_z), 1e-12),
                ]
            )
            radius = math.sqrt(int(self.metadata["event_dim"]) + z_dim)
            payload = {"joint_tree": cKDTree(joint), "joint_radius": radius, "z_tree": cKDTree(self.Z / max(float(self.bandwidth_z), 1e-12))}
        self._tree_cache[key] = payload
        return payload

    def _default_hx_candidates(self) -> list[float]:
        event_dim = int(self.metadata.get("event_dim", 1))
        base = np.array([0.03, 0.06, 0.10, 0.18, 0.32]) if event_dim == 1 else np.array([0.05, 0.09, 0.16, 0.28, 0.40])
        cfg = self.config.get("bandwidth_x_candidates")
        return [float(x) for x in (cfg if cfg is not None else base)]

    def _default_hz_candidates(self, z_dim: int) -> list[float]:
        if z_dim == 0:
            return [1.0]
        cfg = self.config.get("bandwidth_z_candidates")
        return [float(x) for x in (cfg if cfg is not None else [0.12, 0.20, 0.32, 0.50, 0.80])]

    def _bandwidth_candidates(self, Z: np.ndarray, arrays: list[np.ndarray]) -> tuple[list[float], list[float]]:
        cfg_x = self.config.get("bandwidth_x_candidates")
        cfg_z = self.config.get("bandwidth_z_candidates")
        if cfg_x is not None or cfg_z is not None:
            hx = [float(x) for x in (cfg_x if cfg_x is not None else self._default_hx_candidates())]
            hz = [1.0] if Z.shape[1] == 0 else [float(z) for z in (cfg_z if cfg_z is not None else self._default_hz_candidates(Z.shape[1]))]
            return hx, hz

        n = max(int(Z.shape[0]), 1)
        folds = max(min(int(self.config.get("bandwidth_cv_folds", 5)), n), 1)
        n_train_eff = max(int(round(n * max(folds - 1, 1) / max(folds, 1))), 1)
        total_events = max(int(sum(arr.shape[0] for arr in arrays)), 1)
        N_total_train_eff = max(int(round(total_events * max(folds - 1, 1) / max(folds, 1))), 1)
        event_dim = int(self.metadata.get("event_dim", 1))
        z_dim = int(Z.shape[1])
        c_x = float(self.config.get("bandwidth_scale_x", 1.0))
        c_z = float(self.config.get("bandwidth_scale_z", 1.0))
        mode = str(self.config.get("bandwidth_theory_mode", "separate_spatial_covariate")).lower()

        if self.config.get("mode") == "spatial_only" or z_dim == 0:
            hx0 = c_x * (N_total_train_eff ** (-1.0 / (4.0 + event_dim)))
            hz0 = 1.0
        elif mode == "klutchnikoff_joint":
            denom_dim = 5.0 + z_dim if event_dim == 1 else 4.0 + event_dim + z_dim
            hx0 = c_x * (n_train_eff ** (-1.0 / denom_dim))
            hz0 = c_z * (n_train_eff ** (-1.0 / denom_dim))
        else:
            hx0 = c_x * (N_total_train_eff ** (-1.0 / (4.0 + event_dim)))
            hz0 = c_z * (n_train_eff ** (-1.0 / (4.0 + z_dim)))

        multipliers = self._coarse_multipliers(z_dim)
        hx = self._clip_bandwidths(hx0 * multipliers)
        hz = [1.0] if z_dim == 0 else self._clip_bandwidths(hz0 * multipliers)
        return hx, hz

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

    def _coarse_multipliers(self, z_dim: int) -> np.ndarray:
        key = "bandwidth_highdim_multipliers" if z_dim >= 5 else "bandwidth_coarse_multipliers"
        return np.asarray(self.config.get(key, [0.5, 0.75, 1.0, 1.5, 2.0]), dtype=np.float64)

    def _fine_candidates_around(self, best: float) -> list[float]:
        multipliers = np.asarray(self.config.get("bandwidth_fine_multipliers", [0.8, 1.0, 1.25]), dtype=np.float64)
        return self._clip_bandwidths(float(best) * multipliers)

    def _clip_bandwidths(self, values: np.ndarray) -> list[float]:
        lo = float(self.config.get("bandwidth_min", 0.02))
        hi = float(self.config.get("bandwidth_max", 1.25))
        clipped = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
        return [float(x) for x in np.unique(np.round(clipped, 12))]

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
            return [float(x) for x in np.linspace(lo, hi if hi > lo else best, size)]
        return [float(x) for x in np.geomspace(lo, hi, size)]

    def _gaussian_cutoff(self) -> float | None:
        if not bool(self.config.get("use_gaussian_cutoff", True)):
            return None
        cutoff = self.config.get("gaussian_cutoff")
        return None if cutoff is None else float(cutoff)

    def _query_chunk_size(self, n_query: int) -> int:
        requested = int(self.config.get("kernel_chunk_size", 4096))
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        n_reps = max(self.Z.shape[0], 1)
        by_reps = max(1, max_entries // n_reps)
        return max(1, min(requested, n_query, by_reps))

    def _event_chunk_size(self, n_query_chunk: int) -> int:
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        return max(1, min(self.event_points.shape[0], max_entries // max(int(n_query_chunk), 1)))

    def _event_chunk_size_multi(self, n_query_chunk: int, n_z_chunk: int) -> int:
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        denom = max(int(n_query_chunk) + int(n_z_chunk), 1)
        return max(1, min(self.event_points.shape[0], max_entries // denom))

    def _z_chunk_size(self, n_z: int) -> int:
        requested = int(self.config.get("validation_z_chunk_size", 256))
        max_entries = int(self.config.get("max_dense_entries", 20_000_000))
        by_reps = max(1, max_entries // max(self.Z.shape[0], 1))
        return max(1, min(int(n_z), requested, by_reps))

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


def raw_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    out = A2 + B2 - 2.0 * (A @ B.T)
    np.maximum(out, 0.0, out=out)
    return out


def gaussian_from_scaled_sqdist(
    scaled_sqdist: np.ndarray,
    dim: int,
    bandwidth: float,
    cutoff: float | None = None,
) -> np.ndarray:
    if dim == 0:
        return np.ones_like(scaled_sqdist, dtype=np.float64)
    h = max(float(bandwidth), 1e-12)
    norm_const = (math.sqrt(2.0 * math.pi) * h) ** int(dim)
    if cutoff is None:
        return np.exp(-0.5 * scaled_sqdist) / norm_const
    mask = scaled_sqdist <= float(cutoff) ** 2
    out = np.zeros_like(scaled_sqdist, dtype=np.float64)
    out[mask] = np.exp(-0.5 * scaled_sqdist[mask]) / norm_const
    return out


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


def unit_cube_boundary_mass(points: np.ndarray, bandwidth: float, kernel: str = "gaussian") -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.shape[1] == 0:
        return np.ones(points.shape[0], dtype=np.float64)
    h = max(float(bandwidth), 1e-12)
    kernel = str(kernel).lower()
    if kernel == "epanechnikov":
        lower = np.maximum(-1.0, (points - 1.0) / h)
        upper = np.minimum(1.0, points / h)
        per_dim = _epanechnikov_antiderivative(upper) - _epanechnikov_antiderivative(lower)
        per_dim = np.maximum(per_dim, 0.0)
    else:
        upper = (1.0 - points) / h
        lower = -points / h
        per_dim = _normal_cdf(upper) - _normal_cdf(lower)
        per_dim = np.maximum(per_dim, 0.0)
    return np.prod(per_dim, axis=1)


def _epanechnikov_antiderivative(u: np.ndarray) -> np.ndarray:
    return 0.75 * (u - (u**3) / 3.0)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(x)
    except Exception:
        erf = np.vectorize(math.erf, otypes=[np.float64])
        return 0.5 * (1.0 + erf(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))


def _chunk_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))
