from __future__ import annotations

import numpy as np

from ..base import Estimator


class BayesianCovariateDrivenEstimator(Estimator):
    """Interface stub for a Giordano et al.-style covariate-driven Bayesian model."""

    method_name = "optional_bayesian_covariate"

    def fit(self, data):
        raise NotImplementedError(
            "BayesianCovariateDrivenEstimator is an optional interface stub. "
            "Add a concrete posterior approximation before running it."
        )

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Optional Bayesian covariate-driven baseline is not implemented yet.")
