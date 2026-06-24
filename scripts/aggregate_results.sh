#!/usr/bin/env bash
set -euo pipefail

# Rebuild results/summary_metrics.csv, results/summary_table.md,
# results/summary_table.tex, and optional theoretical-rate box plots.

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
elif command -v python >/dev/null 2>&1 && python -c "import pandas" >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1 && python3 -c "import pandas" >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  PYTHON_CMD=(uv run --with-requirements requirements.txt python)
else
  echo "No Python with the required dependencies found. Install requirements.txt first." >&2
  exit 1
fi

SELECT_BEST_MODELS="${SELECT_BEST_MODELS:-1}"
ARGS=(--results-dir "${RESULTS_DIR:-results}" --boxplots)
if [[ "$SELECT_BEST_MODELS" == "1" ]]; then
  ARGS+=(--select-best-models)
fi
if [[ "${KEEP_ALL_MODELS:-0}" == "1" ]]; then
  ARGS+=(--keep-all-models)
fi

"${PYTHON_CMD[@]}" -m src.aggregate "${ARGS[@]}"
