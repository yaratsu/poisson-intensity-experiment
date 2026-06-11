from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import ensure_dir


class Estimator:
    method_name = "base"

    def fit(self, data: dict[str, Any]) -> "Estimator":
        raise NotImplementedError

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Estimator":
        with Path(path).open("rb") as f:
            return pickle.load(f)
