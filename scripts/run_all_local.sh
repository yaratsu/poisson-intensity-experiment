#!/usr/bin/env bash
set -euo pipefail

# Local CPU entry point for non-deep baselines.
# This script runs Euclidean/covariate kernels on Euclidean supports and the
# Ward-style manifold kernel on circle/sphere supports.
#
# Small smoke run:
#   bash scripts/run_all_local.sh --n 100 --z-dims "0 1" --repetitions 1

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all_local.sh [options]

Grid options:
  --scenario NAME              One true-intensity scenario: compositional, near_zero, manifold.
  --scenarios "A B"             Space-separated true-intensity scenarios.
  --support NAME               One support: euclidean1d, euclidean2d, circle, sphere.
  --supports "A B"              Space-separated supports.
  --z-dim K                    One covariate dimension.
  --z-dims "K ..."              Space-separated covariate dimensions.
  --n N                        One sample size.
  --n-values "N ..."            Space-separated sample sizes.
  --repetitions R              Run repetitions 0, ..., R-1.
  --repetition-values "R ..."   Explicit repetition indices.

Intensity and local-baseline options:
  --expected-count M           Expected events per replicate.
  --epsilon EPS                Near-zero/manifold epsilon.
  --alpha A                    Alpha parameter for theory-rate setting.
  --beta B                     Smoothness parameter.
  --kernel-chunk-size K        Query chunk size for kernel predictions.
  --eval-chunk-size K          Query chunk size for Hellinger evaluation.
  --gaussian-cutoff C          Truncate Gaussian tails beyond C bandwidths.
  --no-gaussian-cutoff         Disable Gaussian tail truncation.
  --kernel-type TYPE           gaussian or epanechnikov.
  --boundary-correction MODE   none or renormalize for Euclidean event-space boundaries.
  --bandwidth-selection MODE   theory_guided_5fold_cv, ward_cv, ward_np, etc.
  --bandwidth-cv-folds K       Number of replicate-level CV folds.
  --bandwidth-theory-mode MODE separate_spatial_covariate or klutchnikoff_joint.
  --bandwidth-search MODE      coarse_to_fine or full.
  --bandwidth-grid-size K      Coarse bandwidth grid size.
  --bandwidth-fine-grid-size K Fine bandwidth grid size.
  --kernel-cv-max-replicates K Max replicates used for kernel bandwidth CV; 0 means all.
  --kernel-cv-max-events-per-replicate K Max events per replicate used for kernel bandwidth CV; 0 means all.
  --validation-z-chunk-size K  Validation covariate chunk size.
  --validation-quadrature-method METHOD uniform or sobol for kernel CV integration points.
  --cache-dir DIR              Cache directory.
  --use-distance-cache         Enable distance cache.
  --no-distance-cache          Disable distance cache.
  --mesh-resolution K          Manifold mesh/quadrature resolution.
  --correction-type TYPE       none, global, or local.
  --save-model-policy POLICY   best_repeat, all, or none.
  --keep-all-models            Keep non-best temporary models.
  --no-plots                   Skip best-repeat intensity plots.
  --run-oracle-manifold-kernel Include Ward-style manifold kernel baseline.
  -h, --help                   Show this help.
EOF
}

repetition_values() {
  local count="$1"
  if ! [[ "$count" =~ ^[0-9]+$ ]]; then
    echo "--repetitions must be a nonnegative integer" >&2
    exit 2
  fi
  local values=""
  local i
  for ((i = 0; i < count; i++)); do
    values+="$i "
  done
  echo "${values% }"
}

is_euclidean_case() {
  local scenario="$1"
  local support="$2"
  case "$scenario:$support" in
    compositional:euclidean1d|compositional:euclidean2d) return 0 ;;
    near_zero:euclidean1d|near_zero:euclidean2d) return 0 ;;
    *) return 1 ;;
  esac
}

is_manifold_case() {
  local scenario="$1"
  local support="$2"
  case "$scenario:$support" in
    manifold:circle|manifold:sphere) return 0 ;;
    *) return 1 ;;
  esac
}

N_VALUES="${N_VALUES:-100 316 1000 3162 10000}"
Z_DIMS="${Z_DIMS:-0 1 5 10}"
REPETITIONS="${REPETITIONS:-0 1 2}"
SCENARIOS="${SCENARIOS:-compositional near_zero manifold}"
SUPPORTS="${SUPPORTS:-euclidean1d euclidean2d circle sphere}"
EXPECTED_COUNT="${EXPECTED_COUNT:-30}"
EPSILON="${EPSILON:-}"
ALPHA="${ALPHA:-}"
BETA="${BETA:-}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-4096}"
GAUSSIAN_CUTOFF="${GAUSSIAN_CUTOFF:-4.0}"
USE_GAUSSIAN_CUTOFF="${USE_GAUSSIAN_CUTOFF:-1}"
KERNEL_TYPE="${KERNEL_TYPE:-gaussian}"
BOUNDARY_CORRECTION="${BOUNDARY_CORRECTION:-renormalize}"
BANDWIDTH_SELECTION="${BANDWIDTH_SELECTION:-}"
BANDWIDTH_CV_FOLDS="${BANDWIDTH_CV_FOLDS:-5}"
BANDWIDTH_THEORY_MODE="${BANDWIDTH_THEORY_MODE:-}"
BANDWIDTH_SEARCH="${BANDWIDTH_SEARCH:-coarse_to_fine}"
BANDWIDTH_GRID_SIZE="${BANDWIDTH_GRID_SIZE:-5}"
BANDWIDTH_FINE_GRID_SIZE="${BANDWIDTH_FINE_GRID_SIZE:-3}"
KERNEL_CV_MAX_REPLICATES="${KERNEL_CV_MAX_REPLICATES:-512}"
KERNEL_CV_MAX_EVENTS_PER_REPLICATE="${KERNEL_CV_MAX_EVENTS_PER_REPLICATE:-128}"
VALIDATION_Z_CHUNK_SIZE="${VALIDATION_Z_CHUNK_SIZE:-256}"
VALIDATION_QUADRATURE_METHOD="${VALIDATION_QUADRATURE_METHOD:-uniform}"
CACHE_DIR="${CACHE_DIR:-cache}"
USE_DISTANCE_CACHE="${USE_DISTANCE_CACHE:-1}"
MESH_RESOLUTION="${MESH_RESOLUTION:-64}"
CORRECTION_TYPE="${CORRECTION_TYPE:-none}"
SAVE_MODEL_POLICY="${SAVE_MODEL_POLICY:-best_repeat}"
KEEP_ALL_MODELS="${KEEP_ALL_MODELS:-0}"
NO_PLOTS="${NO_PLOTS:-0}"
RUN_ORACLE_MANIFOLD_KERNEL="${RUN_ORACLE_MANIFOLD_KERNEL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIOS="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --support) SUPPORTS="$2"; shift 2 ;;
    --supports) SUPPORTS="$2"; shift 2 ;;
    --z-dim) Z_DIMS="$2"; shift 2 ;;
    --z-dims) Z_DIMS="$2"; shift 2 ;;
    --n) N_VALUES="$2"; shift 2 ;;
    --n-values) N_VALUES="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$(repetition_values "$2")"; shift 2 ;;
    --repetition-values|--reps) REPETITIONS="$2"; shift 2 ;;
    --expected-count) EXPECTED_COUNT="$2"; shift 2 ;;
    --epsilon) EPSILON="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --kernel-chunk-size) KERNEL_CHUNK_SIZE="$2"; shift 2 ;;
    --eval-chunk-size) EVAL_CHUNK_SIZE="$2"; shift 2 ;;
    --gaussian-cutoff) GAUSSIAN_CUTOFF="$2"; USE_GAUSSIAN_CUTOFF=1; shift 2 ;;
    --no-gaussian-cutoff) USE_GAUSSIAN_CUTOFF=0; shift ;;
    --kernel-type) KERNEL_TYPE="$2"; shift 2 ;;
    --boundary-correction) BOUNDARY_CORRECTION="$2"; shift 2 ;;
    --bandwidth-selection) BANDWIDTH_SELECTION="$2"; shift 2 ;;
    --bandwidth-cv-folds) BANDWIDTH_CV_FOLDS="$2"; shift 2 ;;
    --bandwidth-theory-mode) BANDWIDTH_THEORY_MODE="$2"; shift 2 ;;
    --bandwidth-search) BANDWIDTH_SEARCH="$2"; shift 2 ;;
    --bandwidth-grid-size) BANDWIDTH_GRID_SIZE="$2"; shift 2 ;;
    --bandwidth-fine-grid-size) BANDWIDTH_FINE_GRID_SIZE="$2"; shift 2 ;;
    --kernel-cv-max-replicates) KERNEL_CV_MAX_REPLICATES="$2"; shift 2 ;;
    --kernel-cv-max-events-per-replicate) KERNEL_CV_MAX_EVENTS_PER_REPLICATE="$2"; shift 2 ;;
    --validation-z-chunk-size) VALIDATION_Z_CHUNK_SIZE="$2"; shift 2 ;;
    --validation-quadrature-method) VALIDATION_QUADRATURE_METHOD="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --use-distance-cache) USE_DISTANCE_CACHE=1; shift ;;
    --no-distance-cache) USE_DISTANCE_CACHE=0; shift ;;
    --mesh-resolution) MESH_RESOLUTION="$2"; shift 2 ;;
    --correction-type) CORRECTION_TYPE="$2"; shift 2 ;;
    --save-model-policy) SAVE_MODEL_POLICY="$2"; shift 2 ;;
    --keep-all-models) KEEP_ALL_MODELS=1; shift ;;
    --no-plots) NO_PLOTS=1; shift ;;
    --run-oracle-manifold-kernel) RUN_ORACLE_MANIFOLD_KERNEL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPETITION_COUNT=0
for _rep in $REPETITIONS; do
  REPETITION_COUNT=$((REPETITION_COUNT + 1))
done
if [[ "$REPETITION_COUNT" -lt 1 ]]; then
  echo "At least one repetition is required." >&2
  exit 2
fi

count_cases() {
  local count=0
  local scenario support z_dim n rep
  for scenario in $SCENARIOS; do
    for support in $SUPPORTS; do
      for z_dim in $Z_DIMS; do
        for n in $N_VALUES; do
          for rep in $REPETITIONS; do
            if is_euclidean_case "$scenario" "$support"; then
              count=$((count + 1))
              if [[ "$z_dim" != "0" ]]; then
                count=$((count + 1))
              fi
            elif is_manifold_case "$scenario" "$support" && [[ "$RUN_ORACLE_MANIFOLD_KERNEL" == "1" ]]; then
              count=$((count + 1))
            fi
          done
        done
      done
    done
  done
  echo "$count"
}

if [[ "$NO_PLOTS" == "1" ]]; then
  PLOT_STATUS="off"
else
  PLOT_STATUS="on"
fi
echo "run_all_local: launching $(count_cases) runs (scenarios=[$SCENARIOS], supports=[$SUPPORTS], z_dims=[$Z_DIMS], n_values=[$N_VALUES], repetitions=[$REPETITIONS], plots=$PLOT_STATUS, plot_after_repetitions=$REPETITION_COUNT)"

run_case() {
  local scenario="$1"
  local support="$2"
  local z_dim="$3"
  local n="$4"
  local rep="$5"
  local method="$6"
  local intensity_args=(--expected-count "$EXPECTED_COUNT")
  if [[ -n "$EPSILON" ]]; then
    intensity_args+=(--epsilon "$EPSILON")
  fi
  if [[ -n "$ALPHA" ]]; then
    intensity_args+=(--alpha "$ALPHA")
  fi
  if [[ -n "$BETA" ]]; then
    intensity_args+=(--beta "$BETA")
  fi
  local kernel_args=(
    --kernel-chunk-size "$KERNEL_CHUNK_SIZE"
    --eval-chunk-size "$EVAL_CHUNK_SIZE"
    --kernel-type "$KERNEL_TYPE"
    --boundary-correction "$BOUNDARY_CORRECTION"
    --bandwidth-cv-folds "$BANDWIDTH_CV_FOLDS"
    --bandwidth-search "$BANDWIDTH_SEARCH"
    --bandwidth-grid-size "$BANDWIDTH_GRID_SIZE"
    --bandwidth-fine-grid-size "$BANDWIDTH_FINE_GRID_SIZE"
    --kernel-cv-max-replicates "$KERNEL_CV_MAX_REPLICATES"
    --kernel-cv-max-events-per-replicate "$KERNEL_CV_MAX_EVENTS_PER_REPLICATE"
    --validation-z-chunk-size "$VALIDATION_Z_CHUNK_SIZE"
    --validation-quadrature-method "$VALIDATION_QUADRATURE_METHOD"
    --cache-dir "$CACHE_DIR"
    --mesh-resolution "$MESH_RESOLUTION"
    --correction-type "$CORRECTION_TYPE"
    --save-model-policy "$SAVE_MODEL_POLICY"
  )
  if [[ -n "$BANDWIDTH_SELECTION" ]]; then
    kernel_args+=(--bandwidth-selection "$BANDWIDTH_SELECTION")
  fi
  if [[ -n "$BANDWIDTH_THEORY_MODE" ]]; then
    kernel_args+=(--bandwidth-theory-mode "$BANDWIDTH_THEORY_MODE")
  fi
  if [[ "$KEEP_ALL_MODELS" == "1" ]]; then
    kernel_args+=(--keep-all-models)
  fi
  if [[ "$USE_GAUSSIAN_CUTOFF" == "1" ]]; then
    kernel_args+=(--gaussian-cutoff "$GAUSSIAN_CUTOFF")
  else
    kernel_args+=(--no-gaussian-cutoff)
  fi
  if [[ "$USE_DISTANCE_CACHE" == "1" ]]; then
    kernel_args+=(--use-distance-cache)
  else
    kernel_args+=(--no-distance-cache)
  fi
  local run_args=(--plot-after-repetitions "$REPETITION_COUNT")
  if [[ "$NO_PLOTS" == "1" ]]; then
    run_args+=(--no-plots)
  fi
  bash scripts/run_experiment.sh \
    --scenario "$scenario" \
    --support "$support" \
    --z-dim "$z_dim" \
    --n "$n" \
    --repetition "$rep" \
    --method "$method" \
    "${intensity_args[@]}" \
    "${kernel_args[@]}" \
    "${run_args[@]}"
}

for scenario in $SCENARIOS; do
  for support in $SUPPORTS; do
    for z_dim in $Z_DIMS; do
      for n in $N_VALUES; do
        for rep in $REPETITIONS; do
          if is_euclidean_case "$scenario" "$support"; then
            run_case "$scenario" "$support" "$z_dim" "$n" "$rep" kernel_euclidean
            if [[ "$z_dim" != "0" ]]; then
              run_case "$scenario" "$support" "$z_dim" "$n" "$rep" kernel_covariate
            fi
          elif is_manifold_case "$scenario" "$support" && [[ "$RUN_ORACLE_MANIFOLD_KERNEL" == "1" ]]; then
            run_case "$scenario" "$support" "$z_dim" "$n" "$rep" kernel_manifold
          fi
        done
      done
    done
  done
done
