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
    --run-oracle-manifold-kernel) RUN_ORACLE_MANIFOLD_KERNEL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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
  bash scripts/run_experiment.sh \
    --scenario "$scenario" \
    --support "$support" \
    --z-dim "$z_dim" \
    --n "$n" \
    --repetition "$rep" \
    --method "$method" \
    "${intensity_args[@]}"
}

for scenario in $SCENARIOS; do
  for support in $SUPPORTS; do
    for z_dim in $Z_DIMS; do
      for n in $N_VALUES; do
        for rep in $REPETITIONS; do
          if is_euclidean_case "$scenario" "$support"; then
            run_case "$scenario" "$support" "$z_dim" "$n" "$rep" kernel_euclidean
            run_case "$scenario" "$support" "$z_dim" "$n" "$rep" kernel_covariate
          elif is_manifold_case "$scenario" "$support" && [[ "$RUN_ORACLE_MANIFOLD_KERNEL" == "1" ]]; then
            run_case "$scenario" "$support" "$z_dim" "$n" "$rep" kernel_manifold
          fi
        done
      done
    done
  done
done
