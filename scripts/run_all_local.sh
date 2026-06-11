#!/usr/bin/env bash
set -euo pipefail

# Local CPU entry point for non-deep baselines.
# This script runs Euclidean/covariate kernels on Euclidean supports and the
# Ward-style manifold kernel on circle/sphere supports.
#
# Small smoke run:
#   N_VALUES="100" Z_DIMS="0 1" REPETITIONS="0" bash scripts/run_all_local.sh

N_VALUES="${N_VALUES:-100 316 1000 3162 10000}"
Z_DIMS="${Z_DIMS:-0 1 5 10}"
REPETITIONS="${REPETITIONS:-0 1 2}"

for z_dim in $Z_DIMS; do
  for n in $N_VALUES; do
    for rep in $REPETITIONS; do
      for scenario in compositional near_zero; do
        for support in euclidean1d euclidean2d; do
          bash scripts/run_experiment.sh --scenario "$scenario" --support "$support" --z-dim "$z_dim" --n "$n" --repetition "$rep" --method kernel_euclidean
          bash scripts/run_experiment.sh --scenario "$scenario" --support "$support" --z-dim "$z_dim" --n "$n" --repetition "$rep" --method kernel_covariate
        done
      done
      if [[ "${RUN_ORACLE_MANIFOLD_KERNEL:-0}" == "1" ]]; then
        for support in circle sphere; do
          bash scripts/run_experiment.sh --scenario manifold --support "$support" --z-dim "$z_dim" --n "$n" --repetition "$rep" --method kernel_manifold
        done
      fi
    done
  done
done
