#!/usr/bin/env bash
set -euo pipefail

# Compare Gaussian kernel predictions with and without numerical tail cutoff.

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import numpy as np

from src.covariates import make_covariate_sampler
from src.intensities import make_true_intensity
from src.methods.kernel_covariate import CovariateKernelEstimator
from src.simulate import simulate_dataset

rng = np.random.default_rng(123)
scenario = "compositional"
support = "euclidean2d"
z_dim = 1
n = 100
true_intensity = make_true_intensity(scenario, support, z_dim)
data = simulate_dataset(true_intensity, n=n, z_dim=z_dim, rng=rng, covariate_sampler=make_covariate_sampler("uniform"))

base = {
    "bandwidth_x": 0.16,
    "bandwidth_z": 0.5,
    "bandwidth_search": "full",
    "kernel_chunk_size": 1024,
    "use_distance_cache": False,
    "seed": 123,
}
ref = CovariateKernelEstimator({**base, "use_gaussian_cutoff": False}).fit(data)
fast = CovariateKernelEstimator({**base, "use_gaussian_cutoff": True, "gaussian_cutoff": 4.0}).fit(data)

X = rng.uniform(0.0, 1.0, size=(128, 2))
Z = rng.uniform(0.0, 1.0, size=(128, z_dim))
pred_ref = ref.predict(X, Z)
pred_fast = fast.predict(X, Z)
rel = np.max(np.abs(pred_fast - pred_ref) / np.maximum(np.abs(pred_ref), 1e-12))
print(f"max_relative_difference={rel:.6g}")
if rel > 1e-3:
    raise SystemExit(1)
PY
