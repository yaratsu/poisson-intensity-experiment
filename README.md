# Nonparametric Intensity Estimation Experiments

This repository implements simulation experiments for the paper **“Nonparametric Intensity Estimation for Covariate-Driven Poisson Processes Using Deep Learning.”** It compares a DNN-based NPMLE against kernel baselines under compositional, near-zero, and manifold-supported conditional Poisson process intensities.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## VS Code Dev Container

This repo includes a Docker + `uv` development environment for VS Code.

1. Install Docker Desktop and the VS Code **Dev Containers** extension.
2. Open this folder in VS Code.
3. Run **Dev Containers: Reopen in Container**.

The container installs `uv`, then `postCreateCommand` creates `/workspace/.venv` and installs `requirements.txt`. VS Code is configured to use `/workspace/.venv/bin/python`.

You can also start the environment manually:

```bash
docker compose up -d --build
docker compose exec poisson-exp bash
uv venv .venv
uv pip install -r requirements.txt
```

## Run One Experiment

```bash
bash scripts/run_experiment.sh \
  --scenario compositional \
  --support euclidean2d \
  --z-dim 5 \
  --n 1000 \
  --method dnn_npmle \
  --output-activation softplus \
  --repetition 0 \
  --expected-count 30 \
  --device auto
```

The runner writes per-run JSON metrics to `results/metrics/`, models to `results/models/`, cumulative metrics to `results/metrics/all_metrics.csv`, and intensity visualizations to `results/plots/intensity/`.

For `z_dim=0`, covariates are always represented as `np.empty((n, 0))`.

## GPU Runs on Kaggle

Enable a GPU accelerator, install dependencies, then run:

```bash
bash scripts/run_all_gpu.sh
```

This runs the DNN-NPMLE with both `softplus` and `relu` output activations across the configured scenario grid. For a small test:

```bash
N_VALUES="100" Z_DIMS="0 1" REPETITIONS="0" EPOCHS="30" bash scripts/run_all_gpu.sh
```

## Local Kernel Baselines

```bash
bash scripts/run_all_local.sh
```

This runs Euclidean and covariate product-kernel baselines on `[0,1]` and `[0,1]^2`, and the manifold kernel baseline on `S^1` and `S^2`.

## Aggregate Results

```bash
bash scripts/aggregate_results.sh
```

This creates:

- `results/summary_metrics.csv`
- `results/summary_table.md`
- `results/summary_table.tex`
- `results/plots/boxplot_*.png`
- `results/plots/boxplot_*.pdf`

## Scenarios

- `compositional`: a smooth low-dimensional compositional intensity on Euclidean supports, designed to favor estimators that can exploit intermediate structure.
- `near_zero`: an intensity with a nontrivial low-intensity region; Hellinger remains stable where KL-style criteria can be fragile.
- `manifold`: smooth intensities on the circle `S^1` and sphere `S^2`, evaluated with manifold quadrature.

## Methods

- `dnn_npmle`: the proposed deep-learning NPMLE trained with the Poisson process negative log-likelihood.
- `kernel_covariate`: Klutchnikoff-Massiot-style conditional product-kernel estimator with denominator trimming.
- `kernel_euclidean`: Euclidean pooled or joint product-kernel estimator.
- `kernel_manifold`: Ward et al.-style geodesic kernel estimator with mesh/quadrature integration.
- `optional_bayes/SGCPGaussianProcessEstimator`: interface stub for a Kirichenko-van Zanten-style sigmoidal Gaussian Cox process.
- `optional_bayes/BayesianCovariateDrivenEstimator`: interface stub for a Giordano et al.-style Bayesian covariate-driven estimator.

## Reproducibility Notes

Each run fixes a deterministic seed based on scenario, support, `z_dim`, `n`, method, and repetition unless `--seed` is supplied. The default sample sizes are `[100, 316, 1000, 3162, 10000]`, covariate dimensions are `[0, 1, 5, 10]`, repetitions are `0, 1, 2`, and the default expected event count per replicate is `30`.
