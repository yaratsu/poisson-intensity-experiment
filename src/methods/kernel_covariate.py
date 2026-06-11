from __future__ import annotations

from typing import Any

from .kernel_euclidean import EuclideanKernelEstimator


class CovariateKernelEstimator(EuclideanKernelEstimator):
    """Klutchnikoff-Massiot-style conditional product-kernel baseline."""

    method_name = "kernel_covariate"

    def __init__(self, config: dict[str, Any] | None = None):
        base = {
            "mode": "joint_kernel",
            "kernel": "gaussian",
            "a_n": 1e-6,
        }
        if config:
            base.update(config)
        super().__init__(base)
