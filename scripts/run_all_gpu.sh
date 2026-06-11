#!/usr/bin/env bash
set -euo pipefail

# Kaggle/GPU entry point for the proposed DNN-NPMLE.
# On Kaggle, install requirements first, enable an accelerator, then run:
#   bash scripts/run_all_gpu.sh
#
# Override the grid by setting environment variables, e.g.
#   N_VALUES="100 316" Z_DIMS="0 1" bash scripts/run_all_gpu.sh

N_VALUES="${N_VALUES:-100 316 1000 3162 10000}"
Z_DIMS="${Z_DIMS:-0 1 5 10}"
REPETITIONS="${REPETITIONS:-0 1 2}"
EPOCHS="${EPOCHS:-150}"

run_case() {
  local scenario="$1"
  local support="$2"
  local z_dim="$3"
  local n="$4"
  local rep="$5"
  local activation="$6"
  bash scripts/run_experiment.sh \
    --scenario "$scenario" \
    --support "$support" \
    --z-dim "$z_dim" \
    --n "$n" \
    --method dnn_npmle \
    --output-activation "$activation" \
    --repetition "$rep" \
    --expected-count 30 \
    --device cuda \
    --max-epochs "$EPOCHS"
}

for z_dim in $Z_DIMS; do
  for n in $N_VALUES; do
    for rep in $REPETITIONS; do
      for activation in softplus relu; do
        run_case compositional euclidean1d "$z_dim" "$n" "$rep" "$activation"
        run_case compositional euclidean2d "$z_dim" "$n" "$rep" "$activation"
        run_case near_zero euclidean1d "$z_dim" "$n" "$rep" "$activation"
        run_case near_zero euclidean2d "$z_dim" "$n" "$rep" "$activation"
        run_case manifold circle "$z_dim" "$n" "$rep" "$activation"
        run_case manifold sphere "$z_dim" "$n" "$rep" "$activation"
      done
    done
  done
done
