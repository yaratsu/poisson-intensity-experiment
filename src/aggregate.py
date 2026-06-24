from __future__ import annotations

import argparse

from .model_selection import select_best_models
from .plots import collect_metric_json, create_boxplots, create_summary_tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate Poisson intensity experiment results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--boxplots", action="store_true", help="Also regenerate theoretical-rate box plots.")
    parser.add_argument("--select-best-models", action="store_true", help="Select and save only the best repetition model for each setting.")
    parser.add_argument("--keep-all-models", action="store_true", help="Keep non-best temporary model files when selecting best models.")
    args = parser.parse_args(argv)
    if args.select_best_models:
        select_best_models(args.results_dir, keep_all_models=args.keep_all_models, required_repetitions={0, 1, 2})
    collect_metric_json(args.results_dir)
    create_summary_tables(args.results_dir)
    if args.boxplots:
        create_boxplots(args.results_dir)


if __name__ == "__main__":
    main()
