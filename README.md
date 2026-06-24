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

The runner writes per-run JSON metrics to `results/metrics/`, cumulative metrics to `results/metrics/all_metrics.csv`, bandwidth CV tables to `results/bandwidth_cv/`, and intensity visualizations to `results/plots/intensity/`.

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

## Bandwidth Selection

Kernel bandwidths are selected in a way that follows the reference estimators while keeping the same Nadaraya-Watson/product-kernel and geodesic-kernel forms.

- `kernel_covariate` uses Klutchnikoff-Massiot-inspired pilot scales. By default, `h_x` is centered at `N_total_train^(-1 / (4 + d_x))` and `h_z` at `n_train^(-1 / (4 + d_z))`; use `--bandwidth-theory-mode klutchnikoff_joint` for the original one-dimensional time/covariate joint scale.
- `kernel_euclidean` is fixed as a pooled spatial baseline: it ignores covariates and estimates the marginal event-space intensity. When `z_dim=0`, `run_all_local.sh` skips `kernel_covariate` because it degenerates to the same no-covariate estimator.
- `kernel_manifold` defaults to Ward-style Poisson CV log-likelihood with geodesic kernels and mesh/quadrature integration: `--bandwidth-selection ward_cv`.
- The optional Ward/Cronie-van Lieshout nonparametric criterion is available with `--bandwidth-selection ward_np`.
- Final bandwidths are selected by 5-fold CV over independent Poisson process replicates. Folds never split events from the same replicate.
- For speed, bandwidth CV can use a deterministic subset of replicates and a capped number of events per replicate. The final estimator is still fit on the full dataset. Defaults are `--kernel-cv-max-replicates 512` and `--kernel-cv-max-events-per-replicate 128`; set either value to `0` to use all available CV replicates or all events.

The detailed CV table for every kernel run is saved as:

```text
results/bandwidth_cv/{scenario}_{support}_zdim{z_dim}_n{n}_{method}_rep{rep}.csv
```

Useful overrides:

```bash
bash scripts/run_all_local.sh \
  --scenario compositional \
  --support euclidean2d \
  --z-dim 5 \
  --n 1000 \
  --bandwidth-selection theory_guided_5fold_cv \
  --bandwidth-theory-mode separate_spatial_covariate \
  --bandwidth-cv-folds 5 \
  --kernel-cv-max-replicates 512 \
  --kernel-cv-max-events-per-replicate 128
```

## Model Saving

By default, `--save-model-policy best_repeat` keeps only the best fitted model/estimator for each experimental setting among repetitions `0, 1, 2`. Best means smallest squared Hellinger distance. Metrics, DNN history, and bandwidth CV tables are saved for every repetition, while intensity plots are generated only for the selected best repeat and written under `results/plots/intensity/`. The `run_all_*.sh` scripts pass `--plot-after-repetitions` automatically, so plots are delayed until the requested repetitions for that setting are present.

During individual repetitions, temporary model files are written under `results/models/tmp/`. After all three repetitions for a setting are present, aggregation or the next run copies the best model to:

```text
results/models/best/{setting_hash}/best_model.pt
results/models/best/{setting_hash}/best_estimator.pkl
```

and writes `best_model_metadata.json`. Use `--save-model-policy all` to keep every model permanently, or `--keep-all-models` to keep temporary repeat models while still selecting the best.

## Speeding Up Kernel Baselines

Dense kernel baselines are expensive because each prediction compares all query locations with all observed events, and bandwidth validation repeats this calculation many times. The implementation keeps the same statistical estimators while changing how the computation is carried out:

- Euclidean and covariate kernels use flattened event arrays, vectorized product-kernel calculations, chunking, and trimmed denominators `f_hat(z) \vee a_n`.
- Euclidean event-space kernels use boundary renormalization by default: `K_h(x-y)` is divided by `int_[0,1]^d K_h(x-u) du`. This reduces endpoint/corner bias for both Gaussian and Epanechnikov kernels. Disable it with `--boundary-correction none`.
- Bandwidth CV reuses flattened fold data and evaluates validation-fold event likelihoods and integration terms in batches. This removes repeated kernel matrix work across validation replicates without changing the Poisson CV objective.
- Kernel bandwidth CV uses fixed uniform integration points by default to avoid repeated Sobol/scipy startup overhead in many short local runs. Use `--validation-quadrature-method sobol` to switch CV integration back to Sobol points.
- Gaussian kernels are numerically truncated at four bandwidths for computational efficiency; this does not change the estimator except for negligible tail contributions. Disable this with `--no-gaussian-cutoff`.
- Epanechnikov kernels use `scipy.spatial.cKDTree` sparse neighbor searches when possible.
- Bandwidth selection uses replicate-level 5-fold Poisson CV by default. Candidate centers are theory-guided for Euclidean/covariate kernels and Ward-style for manifold kernels; `--bandwidth-search coarse_to_fine` reduces the candidate grid while still evaluating candidates by 5-fold CV. `--bandwidth-search full` is still available.
- A comparison with the older `poisson_covariate_experiment/modern_core.py` implementation showed that its speed mainly came from CV subsampling/capping and lighter dtype choices. This code adopts the CV capping idea as a bandwidth-selection approximation, but keeps the baseline estimator as the paper-consistent product-kernel/geodesic-kernel ratio estimator. It does not switch to the older interval-only kernel implementation or change the final estimator fit.
- Distance computations, evaluation grids, and manifold mesh quadrature are cached under `cache/` by default.
- Manifold kernels still use geodesic distances and mesh/quadrature integration. Circle distances are computed by angle differences; sphere distances are chunked and use dot-product cutoffs for Gaussian tails. The current implementation is the normalized geodesic kernel with `correction_type=none`; `global` and `local` correction modes are reserved for future shape-correction extensions.

Useful options:

```bash
bash scripts/run_all_local.sh \
  --scenario compositional \
  --support euclidean2d \
  --z-dim 5 \
  --n 1000 \
  --repetitions 1 \
  --kernel-chunk-size 4096 \
  --gaussian-cutoff 4.0 \
  --bandwidth-search coarse_to_fine \
  --bandwidth-grid-size 5 \
  --bandwidth-fine-grid-size 3 \
  --kernel-cv-max-replicates 512 \
  --kernel-cv-max-events-per-replicate 128 \
  --validation-z-chunk-size 256 \
  --validation-quadrature-method uniform
```

To profile a kernel run:

```bash
bash scripts/profile_kernel_runtime.sh \
  --scenario compositional \
  --support euclidean2d \
  --z-dim 5 \
  --n 1000 \
  --method kernel_covariate
```

Profiles are saved as `results/profiles/kernel_runtime_{scenario}_{support}_zdim{z_dim}_n{n}.json`.

## Aggregate Results

```bash
bash scripts/aggregate_results.sh
```

Aggregation also performs best-repeat model selection by default. Set `SELECT_BEST_MODELS=0` to skip that step, or `KEEP_ALL_MODELS=1` to retain temporary repetition models.

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
- `kernel_euclidean`: Euclidean pooled spatial product-kernel estimator that ignores covariates.
- `kernel_manifold`: Ward et al.-style geodesic kernel estimator with mesh/quadrature integration. This is an oracle/diagnostic baseline because it uses manifold geometry; default local scripts skip it unless `RUN_ORACLE_MANIFOLD_KERNEL=1` is set.
- `optional_bayes/SGCPGaussianProcessEstimator`: interface stub for a Kirichenko-van Zanten-style sigmoidal Gaussian Cox process.
- `optional_bayes/BayesianCovariateDrivenEstimator`: interface stub for a Giordano et al.-style Bayesian covariate-driven estimator.

## Reproducibility Notes

Each run fixes a deterministic seed based on scenario, support, `z_dim`, `n`, method, and repetition unless `--seed` is supplied. The default sample sizes are `[100, 316, 1000, 3162, 10000]`, covariate dimensions are `[0, 1, 5, 10]`, repetitions are `0, 1, 2`, and the default expected event count per replicate is `30`.
