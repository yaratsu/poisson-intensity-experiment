from __future__ import annotations

import numpy as np

from ..base import Estimator


class SGCPGaussianProcessEstimator(Estimator):
    """Interface stub for a Kirichenko-van Zanten-style SGCP baseline."""

    method_name = "optional_sgcp_gp"

    def fit(self, data):
        raise NotImplementedError(
            "SGCPGaussianProcessEstimator is an optional interface stub. "
            "Implement posterior inference before using it in experiments."
        )

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Optional Bayesian SGCP baseline is not implemented yet.")
