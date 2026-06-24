#!/usr/bin/env bash
set -euo pipefail

# Kaggle/GPU entry point for the proposed DNN-NPMLE.
# On Kaggle, install requirements first, enable an accelerator, then run:
#   bash scripts/run_all_gpu.sh
#
# Override the grid by arguments, e.g.
#   bash scripts/run_all_gpu.sh --scenario compositional --support euclidean2d --z-dim 5 --repetitions 3
# Environment variables such as N_VALUES="100 316" Z_DIMS="0 1" are still supported.
# Manifold DNN training defaults to geometry-agnostic mode. For oracle ablations:
#   MANIFOLD_LEARNING=oracle bash scripts/run_all_gpu.sh
# Add MANIFOLD_INPUT=embedded to use embedded coordinates with oracle quadrature.

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all_gpu.sh [options]

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

DNN and intensity options:
  --dnn-architecture MODE      theory/adaptive/auto or fixed/manual.
  --hidden-layers "W ..."      Hidden widths for fixed/manual architecture.
  --architecture-reference-n N Reference sample size for theory architecture.
  --depth-scale C              Theory depth at the reference sample size.
  --width-scale C              Theory width at the reference sample size.
  --width-multiple K           Round theory width up to this multiple.
  --epochs E                   Number of DNN epochs.
  --output-activations "A B"    softplus, relu, or both.
  --device DEVICE              cuda, cpu, or auto.
  --manifold-learning MODE     agnostic or oracle.
  --manifold-input MODE        intrinsic or embedded.
  --save-model-policy POLICY   best_repeat, all, or none.
  --keep-all-models            Keep temporary models for all repetitions.
  --no-plots                   Skip best-repeat intensity plots.
  --expected-count M           Expected events per replicate.
  --epsilon EPS                Near-zero/manifold epsilon.
  --alpha A                    Alpha parameter for theory-rate setting.
  --beta B                     Smoothness parameter.
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

is_valid_case() {
  local scenario="$1"
  local support="$2"
  case "$scenario:$support" in
    compositional:euclidean1d|compositional:euclidean2d) return 0 ;;
    near_zero:euclidean1d|near_zero:euclidean2d) return 0 ;;
    manifold:circle|manifold:sphere) return 0 ;;
    *) return 1 ;;
  esac
}

N_VALUES="${N_VALUES:-100 316 1000 3162 10000}"
Z_DIMS="${Z_DIMS:-0 1 5 10}"
REPETITIONS="${REPETITIONS:-0 1 2}"
SCENARIOS="${SCENARIOS:-compositional near_zero manifold}"
SUPPORTS="${SUPPORTS:-euclidean1d euclidean2d circle sphere}"
OUTPUT_ACTIVATIONS="${OUTPUT_ACTIVATIONS:-softplus relu}"
DNN_ARCHITECTURE="${DNN_ARCHITECTURE:-theory}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-}"
ARCHITECTURE_REFERENCE_N="${ARCHITECTURE_REFERENCE_N:-1000}"
DEPTH_SCALE="${DEPTH_SCALE:-3.0}"
WIDTH_SCALE="${WIDTH_SCALE:-64.0}"
WIDTH_MULTIPLE="${WIDTH_MULTIPLE:-8}"
EPOCHS="${EPOCHS:-150}"
DEVICE="${DEVICE:-cuda}"
EXPECTED_COUNT="${EXPECTED_COUNT:-30}"
EPSILON="${EPSILON:-}"
ALPHA="${ALPHA:-}"
BETA="${BETA:-}"
MANIFOLD_LEARNING="${MANIFOLD_LEARNING:-agnostic}"
MANIFOLD_INPUT="${MANIFOLD_INPUT:-}"
SAVE_MODEL_POLICY="${SAVE_MODEL_POLICY:-best_repeat}"
KEEP_ALL_MODELS="${KEEP_ALL_MODELS:-0}"
NO_PLOTS="${NO_PLOTS:-0}"

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
    --dnn-architecture) DNN_ARCHITECTURE="$2"; shift 2 ;;
    --hidden-layers) HIDDEN_LAYERS="$2"; shift 2 ;;
    --architecture-reference-n) ARCHITECTURE_REFERENCE_N="$2"; shift 2 ;;
    --depth-scale) DEPTH_SCALE="$2"; shift 2 ;;
    --width-scale) WIDTH_SCALE="$2"; shift 2 ;;
    --width-multiple) WIDTH_MULTIPLE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --output-activations|--activations) OUTPUT_ACTIVATIONS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --manifold-learning) MANIFOLD_LEARNING="$2"; shift 2 ;;
    --manifold-input) MANIFOLD_INPUT="$2"; shift 2 ;;
    --save-model-policy) SAVE_MODEL_POLICY="$2"; shift 2 ;;
    --keep-all-models) KEEP_ALL_MODELS=1; shift ;;
    --no-plots) NO_PLOTS=1; shift ;;
    --expected-count) EXPECTED_COUNT="$2"; shift 2 ;;
    --epsilon) EPSILON="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
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

run_case() {
  local scenario="$1"
  local support="$2"
  local z_dim="$3"
  local n="$4"
  local rep="$5"
  local activation="$6"
  local -a manifold_args=()
  local dnn_args=(
    --dnn-architecture "$DNN_ARCHITECTURE"
    --architecture-reference-n "$ARCHITECTURE_REFERENCE_N"
    --depth-scale "$DEPTH_SCALE"
    --width-scale "$WIDTH_SCALE"
    --width-multiple "$WIDTH_MULTIPLE"
  )
  if [[ -n "$HIDDEN_LAYERS" ]]; then
    dnn_args+=(--hidden-layers "$HIDDEN_LAYERS")
  fi
  local intensity_args=(--expected-count "$EXPECTED_COUNT")
  local model_save_args=(--save-model-policy "$SAVE_MODEL_POLICY" --plot-after-repetitions "$REPETITION_COUNT")
  if [[ "$KEEP_ALL_MODELS" == "1" ]]; then
    model_save_args+=(--keep-all-models)
  fi
  if [[ "$NO_PLOTS" == "1" ]]; then
    model_save_args+=(--no-plots)
  fi
  if [[ -n "$EPSILON" ]]; then
    intensity_args+=(--epsilon "$EPSILON")
  fi
  if [[ -n "$ALPHA" ]]; then
    intensity_args+=(--alpha "$ALPHA")
  fi
  if [[ -n "$BETA" ]]; then
    intensity_args+=(--beta "$BETA")
  fi
  if [[ "$scenario" == "manifold" ]]; then
    manifold_args+=(--manifold-learning "$MANIFOLD_LEARNING")
    if [[ -n "$MANIFOLD_INPUT" ]]; then
      manifold_args+=(--manifold-input "$MANIFOLD_INPUT")
    fi
  fi
  if [[ ${#manifold_args[@]} -gt 0 ]]; then
    bash scripts/run_experiment.sh \
      --scenario "$scenario" \
      --support "$support" \
      --z-dim "$z_dim" \
      --n "$n" \
      --method dnn_npmle \
      --output-activation "$activation" \
      --repetition "$rep" \
      --device "$DEVICE" \
      --max-epochs "$EPOCHS" \
      "${dnn_args[@]}" \
      "${intensity_args[@]}" \
      "${model_save_args[@]}" \
      "${manifold_args[@]}"
  else
    bash scripts/run_experiment.sh \
      --scenario "$scenario" \
      --support "$support" \
      --z-dim "$z_dim" \
      --n "$n" \
      --method dnn_npmle \
      --output-activation "$activation" \
      --repetition "$rep" \
      --device "$DEVICE" \
      --max-epochs "$EPOCHS" \
      "${dnn_args[@]}" \
      "${intensity_args[@]}" \
      "${model_save_args[@]}"
  fi
}

for scenario in $SCENARIOS; do
  for support in $SUPPORTS; do
    if ! is_valid_case "$scenario" "$support"; then
      continue
    fi
    for z_dim in $Z_DIMS; do
      for n in $N_VALUES; do
        for rep in $REPETITIONS; do
          for activation in $OUTPUT_ACTIVATIONS; do
            run_case "$scenario" "$support" "$z_dim" "$n" "$rep" "$activation"
          done
        done
      done
    done
  done
done
