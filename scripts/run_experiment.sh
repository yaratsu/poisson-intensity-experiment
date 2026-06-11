#!/usr/bin/env bash
set -euo pipefail

# Run one experiment. Extra arguments are passed directly to python -m src.run_experiment.
# Example:
# bash scripts/run_experiment.sh \
#   --scenario compositional \
#   --support euclidean2d \
#   --z-dim 5 \
#   --n 1000 \
#   --method dnn_npmle \
#   --output-activation softplus \
#   --repetition 0 \
#   --expected-count 30 \
#   --device cuda
# Manifold oracle ablation example:
# bash scripts/run_experiment.sh \
#   --scenario manifold --support circle --z-dim 1 --n 1000 \
#   --method dnn_npmle --output-activation softplus \
#   --manifold-learning oracle --repetition 0 --device cuda

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
elif command -v python >/dev/null 2>&1 && python -c "import numpy" >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  PYTHON_CMD=(uv run --with-requirements requirements.txt python)
else
  echo "No Python with the required dependencies found. Install requirements.txt first." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" -m src.run_experiment "$@"
