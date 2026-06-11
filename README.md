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

For manifold experiments, training is geometry-agnostic by default. To run the oracle diagnostic version that uses the true manifold quadrature and intrinsic coordinates, add `--manifold-learning oracle`:

```bash
bash scripts/run_experiment.sh \
  --scenario manifold \
  --support circle \
  --z-dim 1 \
  --n 1000 \
  --method dnn_npmle \
  --output-activation softplus \
  --manifold-learning oracle \
  --repetition 0 \
  --device auto
```

If you want oracle quadrature while still feeding embedded coordinates to the network, add `--manifold-input embedded`.

By default, the DNN architecture follows the paper's theoretical scaling. For sample size `n` and squared-Hellinger rate exponent `gamma`, the runner uses

```text
depth ~= log(n),   width ~= n^((1 - gamma) / 2),
```

with practical constants and caps from `configs/default.yaml`. The default method label includes the architecture tag, for example `dnn_npmle_softplus_theory_ds1_ws8`. The resolved architecture is saved in each metrics JSON/CSV as `dnn_hidden_layers`, `dnn_depth`, `dnn_width`, and `dnn_architecture_rate_exponent`. Use `--dnn-architecture fixed --hidden-layers "128 128 128"` to reproduce a fixed architecture.

## GPU Runs on Kaggle

Enable a GPU accelerator, install dependencies, then run:

```bash
bash scripts/run_all_gpu.sh
```

This runs the DNN-NPMLE with both `softplus` and `relu` output activations across the configured scenario grid. For a small test:

```bash
bash scripts/run_all_gpu.sh \
  --scenario compositional \
  --support euclidean2d \
  --z-dim 5 \
  --n 100 \
  --repetitions 1 \
  --epochs 30 \
  --output-activations softplus
```

The all-run scripts accept grid arguments directly. Use singular options for one setting and plural options for space-separated grids:

```bash
bash scripts/run_all_gpu.sh \
  --scenarios "compositional near_zero" \
  --supports "euclidean1d euclidean2d" \
  --z-dims "0 1 5" \
  --n-values "100 316 1000" \
  --repetitions 3 \
  --epochs 150 \
  --dnn-architecture theory
```

For oracle manifold ablations on Kaggle:

```bash
bash scripts/run_all_gpu.sh \
  --scenario manifold \
  --support circle \
  --z-dim 1 \
  --n 100 \
  --repetitions 1 \
  --epochs 30 \
  --manifold-learning oracle
```

## Local Kernel Baselines

```bash
bash scripts/run_all_local.sh
```

This runs Euclidean and covariate product-kernel baselines on `[0,1]` and `[0,1]^2`. The manifold kernel baseline uses geodesic distances and is therefore oracle-only; run it explicitly with `RUN_ORACLE_MANIFOLD_KERNEL=1 bash scripts/run_all_local.sh`.

For one local baseline grid:

```bash
bash scripts/run_all_local.sh \
  --scenario near_zero \
  --support euclidean2d \
  --z-dim 1 \
  --n 100 \
  --repetitions 3 \
  --epsilon 1e-4
```

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

- `dnn_npmle`: the proposed deep-learning NPMLE trained with the Poisson process negative log-likelihood. For manifold scenarios, the default training mode is `manifold_learning: agnostic`: the network uses embedded event coordinates and an empirical support quadrature built only from observed events. It does not use true intrinsic coordinates, geodesic distances, manifold volume, or circle/sphere quadrature during training. True manifold quadrature is used only for simulation, oracle evaluation, and plotting. Set `--manifold-learning oracle` for diagnostic ablations; this defaults to intrinsic manifold coordinates unless `--manifold-input embedded` is supplied.
- `kernel_covariate`: Klutchnikoff-Massiot-style conditional product-kernel estimator with denominator trimming.
- `kernel_euclidean`: Euclidean pooled or joint product-kernel estimator.
- `kernel_manifold`: Ward et al.-style geodesic kernel estimator with mesh/quadrature integration. This is an oracle/diagnostic baseline because it uses manifold geometry; default local scripts skip it unless `RUN_ORACLE_MANIFOLD_KERNEL=1` is set.
- `optional_bayes/SGCPGaussianProcessEstimator`: interface stub for a Kirichenko-van Zanten-style sigmoidal Gaussian Cox process.
- `optional_bayes/BayesianCovariateDrivenEstimator`: interface stub for a Giordano et al.-style Bayesian covariate-driven estimator.

## Reproducibility Notes

Each run fixes a deterministic seed based on scenario, support, `z_dim`, `n`, method, and repetition unless `--seed` is supplied. The default sample sizes are `[100, 316, 1000, 3162, 10000]`, covariate dimensions are `[0, 1, 5, 10]`, repetitions are `0, 1, 2`, and the default expected event count per replicate is `30`.
