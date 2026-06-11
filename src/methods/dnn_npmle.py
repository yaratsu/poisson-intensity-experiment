from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..integration import make_quadrature
from ..intensities import points_for_model
from ..utils import ensure_dir, inverse_softplus
from .base import Estimator


class _TorchIntensityNet:
    pass


def _load_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class DNNNPMLEEstimator(Estimator):
    method_name = "dnn_npmle"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {
            "architecture": "theory",
            "hidden_layers": [128, 128, 128],
            "depth_scale": 1.0,
            "width_scale": 8.0,
            "min_depth": 2,
            "max_depth": 12,
            "min_width": 32,
            "max_width": 512,
            "width_multiple": 8,
            "architecture_rate_exponent": None,
            "output_activation": "softplus",
            "manifold_learning": "agnostic",
            "manifold_input": "intrinsic",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "max_epochs": 150,
            "patience": 15,
            "validation_fraction": 0.2,
            "integration_points": 128,
            "grad_clip": 5.0,
            "eps": 1e-8,
            "dtype": "float32",
            "device": "auto",
            "seed": 0,
            "verbose": False,
            "expected_count": 30.0,
        }
        if config:
            self.config.update({k: v for k, v in config.items() if v is not None})
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.history: dict[str, list[float]] = {"train_nll": [], "val_nll": []}
        self.input_dim: int | None = None
        self._torch_dtype = None
        self._device = None

    def fit(self, data: dict[str, Any]) -> "DNNNPMLEEstimator":
        torch, nn, F = _load_torch()
        self.metadata = dict(data["metadata"])
        if self._uses_agnostic_manifold_learning():
            self.config["manifold_input"] = "embedded"
        seed = int(self.config.get("seed", 0))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device_cfg = self.config.get("device", "auto")
        if device_cfg in {None, "auto"}:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_cfg == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        else:
            device = str(device_cfg)
        self._device = torch.device(device)
        self._torch_dtype = torch.float64 if self.config.get("dtype") == "float64" else torch.float32

        Z = np.asarray(data["Z"], dtype=np.float64)
        n, z_dim = Z.shape
        point_arrays = self._event_point_arrays(data)
        point_dim = self._point_dim()
        self.input_dim = point_dim + z_dim
        resolved_hidden_layers = self._resolve_hidden_layers(n, z_dim, point_dim)
        self.config["resolved_hidden_layers"] = resolved_hidden_layers
        if self._uses_agnostic_manifold_learning():
            target = 1.0
        else:
            target = float(self.config.get("expected_count", self.metadata.get("expected_count", 30.0)))
            target = target / max(float(self.metadata.get("volume", 1.0)), 1e-12)
        self.model = self._build_model(self.input_dim, target).to(self._device, dtype=self._torch_dtype)

        all_indices = np.arange(n)
        rng.shuffle(all_indices)
        n_val = int(round(float(self.config["validation_fraction"]) * n))
        n_val = min(max(n_val, 1 if n > 5 else 0), max(n - 1, 0))
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:] if n_val > 0 else all_indices

        q_model, q_weights = self._training_quadrature(data, rng)
        q_points_t = torch.as_tensor(q_model, dtype=self._torch_dtype, device=self._device)
        q_weights_t = torch.as_tensor(q_weights, dtype=self._torch_dtype, device=self._device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        batch_size = int(self.config["batch_size"])
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = math.inf
        bad_epochs = 0

        for _epoch in range(int(self.config["max_epochs"])):
            rng.shuffle(train_indices)
            train_losses = []
            self.model.train()
            for start in range(0, len(train_indices), batch_size):
                batch = train_indices[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = self._batch_nll(batch, point_arrays, Z, q_points_t, q_weights_t, torch, F)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["grad_clip"]))
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            train_nll = float(np.mean(train_losses)) if train_losses else math.nan
            val_nll = self._validation_nll(val_indices, point_arrays, Z, q_points_t, q_weights_t, torch, F)
            self.history["train_nll"].append(train_nll)
            self.history["val_nll"].append(val_nll)
            monitor = val_nll if np.isfinite(val_nll) else train_nll
            if monitor < best_val - 1e-5:
                best_val = monitor
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.config["patience"]):
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        torch, _nn, _F = _load_torch()
        if self.model is None:
            raise RuntimeError("Estimator must be fit before calling predict")
        X_model = self._prepare_points(np.asarray(X, dtype=np.float64))
        Zb = self._broadcast_z(X_model.shape[0], np.asarray(Z, dtype=np.float64))
        inputs = np.concatenate([X_model, Zb], axis=1).astype(np.float64)
        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], 8192):
                chunk = torch.as_tensor(
                    inputs[start : start + 8192],
                    dtype=self._torch_dtype,
                    device=self._device,
                )
                vals = self._model_intensity(chunk)
                out.append(vals.detach().cpu().numpy())
        return np.maximum(np.concatenate(out), float(self.config["eps"])).astype(np.float64)

    def save(self, path: str | Path) -> None:
        torch, _nn, _F = _load_torch()
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model is not None else None,
                "config": self.config,
                "metadata": self.metadata,
                "history": self.history,
                "input_dim": self.input_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DNNNPMLEEstimator":
        torch, _nn, _F = _load_torch()
        payload = torch.load(path, map_location="cpu")
        obj = cls(payload["config"])
        obj.metadata = payload["metadata"]
        obj.history = payload.get("history", {})
        obj.input_dim = payload["input_dim"]
        device_cfg = obj.config.get("device", "auto")
        device = "cuda" if device_cfg in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
        obj._device = torch.device(device)
        obj._torch_dtype = torch.float64 if obj.config.get("dtype") == "float64" else torch.float32
        obj.model = obj._build_model(obj.input_dim, 1.0).to(obj._device, dtype=obj._torch_dtype)
        obj.model.load_state_dict(payload["state_dict"])
        return obj

    def _build_model(self, input_dim: int, initial_intensity: float):
        torch, nn, _F = _load_torch()

        class Net(nn.Module):
            def __init__(self, outer: DNNNPMLEEstimator):
                super().__init__()
                hidden_layers = outer.config.get("resolved_hidden_layers") or outer.config["hidden_layers"]
                dims = [input_dim] + list(hidden_layers) + [1]
                layers = []
                for left, right in zip(dims[:-2], dims[1:-1]):
                    layer = nn.Linear(left, right)
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)
                    layers.extend([layer, nn.ReLU()])
                out = nn.Linear(dims[-2], dims[-1])
                nn.init.normal_(out.weight, mean=0.0, std=1e-3)
                if outer.config["output_activation"] == "softplus":
                    out.bias.data.fill_(inverse_softplus(float(initial_intensity)))
                else:
                    out.bias.data.fill_(float(initial_intensity))
                layers.append(out)
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return Net(self)

    def _resolve_hidden_layers(self, n: int, z_dim: int, point_dim: int) -> list[int]:
        architecture = str(self.config.get("architecture", "theory")).lower()
        if architecture in {"fixed", "manual"}:
            return [int(width) for width in self.config.get("hidden_layers", [128, 128, 128])]
        if architecture not in {"theory", "adaptive", "auto"}:
            raise ValueError(f"Unknown DNN architecture mode: {architecture}")

        gamma = self._architecture_rate_exponent(z_dim=z_dim, point_dim=point_dim)
        gamma = float(np.clip(gamma, 1e-6, 0.999999))
        n_eff = max(int(n), 3)

        depth = int(math.ceil(float(self.config.get("depth_scale", 1.0)) * math.log(n_eff)))
        depth = max(int(self.config.get("min_depth", 2)), depth)
        depth = min(int(self.config.get("max_depth", 12)), depth)

        raw_width = float(self.config.get("width_scale", 8.0)) * (n_eff ** ((1.0 - gamma) / 2.0))
        width = int(math.ceil(raw_width))
        width = max(int(self.config.get("min_width", 32)), width)
        width = min(int(self.config.get("max_width", 512)), width)
        multiple = int(self.config.get("width_multiple", 1))
        if multiple > 1:
            width = int(math.ceil(width / multiple) * multiple)

        self.config["architecture_rate_exponent_used"] = gamma
        self.config["architecture_depth_used"] = depth
        self.config["architecture_width_used"] = width
        return [width] * depth

    def _architecture_rate_exponent(self, z_dim: int, point_dim: int) -> float:
        explicit = self.config.get("architecture_rate_exponent")
        if explicit is not None:
            return float(explicit)

        scenario = str(self.config.get("scenario", self.metadata.get("scenario", ""))).lower()
        beta = float(self.config.get("beta", self.metadata.get("beta", 2.0)))
        alpha = self.config.get("alpha", self.metadata.get("alpha"))
        if alpha is None:
            alpha = 1.0 if scenario == "near_zero" else 0.0
        alpha = float(alpha)
        numerator = (1.0 + min(alpha, 1.0)) * beta

        if scenario == "compositional":
            effective_dim = self.config.get("theory_effective_dim")
            if effective_dim is None:
                effective_dim = min(int(self.metadata.get("event_dim", point_dim)) + int(z_dim), 4)
        elif self.metadata.get("support_type") == "manifold":
            if self._uses_agnostic_manifold_learning():
                effective_dim = int(point_dim) + int(z_dim)
            else:
                effective_dim = int(self.metadata.get("manifold_dim", point_dim)) + int(z_dim)
        else:
            effective_dim = int(self.metadata.get("event_dim", point_dim)) + int(z_dim)

        return float(numerator / (numerator + max(float(effective_dim), 1e-12)))

    def _model_intensity(self, inputs):
        _torch, _nn, F = _load_torch()
        raw = self.model(inputs)
        eps = float(self.config["eps"])
        if self.config["output_activation"] == "relu":
            return F.relu(raw) + eps
        if self.config["output_activation"] == "softplus":
            return F.softplus(raw) + eps
        raise ValueError(f"Unknown output activation: {self.config['output_activation']}")

    def _batch_nll(self, batch, point_arrays, Z, q_points_t, q_weights_t, torch, F):
        batch = np.asarray(batch, dtype=np.int64)
        z_dim = Z.shape[1]
        event_parts = []
        z_parts = []
        for idx in batch:
            pts = point_arrays[idx]
            if pts.shape[0] == 0:
                continue
            event_parts.append(pts)
            z_parts.append(np.repeat(Z[idx].reshape(1, z_dim), pts.shape[0], axis=0))
        loss = torch.zeros((), dtype=self._torch_dtype, device=self._device)
        batch_size = max(len(batch), 1)
        if event_parts:
            event_x = np.concatenate(event_parts, axis=0)
            event_z = np.concatenate(z_parts, axis=0) if z_dim else np.empty((event_x.shape[0], 0))
            event_inputs = np.concatenate([event_x, event_z], axis=1)
            event_inputs_t = torch.as_tensor(event_inputs, dtype=self._torch_dtype, device=self._device)
            intensity = self._model_intensity(event_inputs_t)
            loss = loss - torch.log(torch.clamp(intensity, min=float(self.config["eps"]))).sum() / batch_size

        q = q_points_t.shape[0]
        z_batch = Z[batch]
        q_rep = q_points_t.repeat(batch_size, 1)
        if z_dim:
            z_rep_np = np.repeat(z_batch, q, axis=0)
            z_rep = torch.as_tensor(z_rep_np, dtype=self._torch_dtype, device=self._device)
            q_inputs = torch.cat([q_rep, z_rep], dim=1)
        else:
            q_inputs = q_rep
        q_vals = self._model_intensity(q_inputs).reshape(batch_size, q)
        integral = (q_vals * q_weights_t.reshape(1, -1)).sum(dim=1).mean()
        return loss + integral

    def _validation_nll(self, indices, point_arrays, Z, q_points_t, q_weights_t, torch, F) -> float:
        if len(indices) == 0:
            return math.nan
        self.model.eval()
        with torch.no_grad():
            losses = []
            for start in range(0, len(indices), int(self.config["batch_size"])):
                batch = indices[start : start + int(self.config["batch_size"])]
                losses.append(float(self._batch_nll(batch, point_arrays, Z, q_points_t, q_weights_t, torch, F).cpu().item()))
        self.model.train()
        return float(np.mean(losses)) if losses else math.nan

    def _event_point_arrays(self, data: dict[str, Any]) -> list[np.ndarray]:
        if self._uses_agnostic_manifold_learning():
            return [np.asarray(arr, dtype=np.float64) for arr in data["events"]]
        if self.metadata["support_type"] == "manifold" and self.config["manifold_input"] == "embedded":
            raw_arrays = data["events"]
        else:
            raw_arrays = data["event_coords"]
        return [self._prepare_points(arr) for arr in raw_arrays]

    def _training_quadrature(
        self,
        data: dict[str, Any],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._uses_agnostic_manifold_learning():
            arrays = [np.asarray(arr, dtype=np.float64) for arr in data["events"] if arr.shape[0] > 0]
            q = int(self.config["integration_points"])
            if arrays:
                flat = np.concatenate(arrays, axis=0)
                replace = flat.shape[0] < q
                choice = rng.choice(flat.shape[0], size=q, replace=replace)
                q_model = flat[choice]
            else:
                dim = int(self.metadata["embedding_dim"])
                q_model = rng.normal(size=(q, dim)).astype(np.float64)
                q_model /= np.maximum(np.linalg.norm(q_model, axis=1, keepdims=True), 1e-12)

            counts = np.asarray(data.get("counts", []), dtype=np.float64)
            mean_count = float(np.mean(counts)) if counts.size else float(self.config.get("expected_count", 30.0))
            weights = np.full(q_model.shape[0], mean_count / max(q_model.shape[0], 1), dtype=np.float64)
            return q_model.astype(np.float64), weights

        q_points, q_weights, q_embedded = make_quadrature(
            self.metadata,
            int(self.config["integration_points"]),
            rng,
        )
        if self.metadata["support_type"] == "manifold" and self.config["manifold_input"] == "embedded":
            q_model = q_embedded
        else:
            q_model = q_points
        q_model = points_for_model(q_model, self.metadata, self.config["manifold_input"])
        return q_model.astype(np.float64), q_weights.astype(np.float64)

    def _prepare_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return points_for_model(X, self.metadata, self.config.get("manifold_input", "intrinsic")).astype(np.float64)

    def _point_dim(self) -> int:
        if self.metadata["support_type"] == "manifold" and self.config["manifold_input"] == "embedded":
            return int(self.metadata["embedding_dim"])
        return int(self.metadata["coord_dim"])

    def _uses_agnostic_manifold_learning(self) -> bool:
        return (
            self.metadata.get("support_type") == "manifold"
            and str(self.config.get("manifold_learning", "agnostic")).lower() == "agnostic"
        )

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
        raise ValueError(f"Cannot broadcast Z shape {Z.shape} to {n} prediction points")
