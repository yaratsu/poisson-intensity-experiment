from __future__ import annotations

import argparse

from .plots import collect_metric_json, create_boxplots, create_summary_tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate Poisson intensity experiment results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--boxplots", action="store_true", help="Also regenerate theoretical-rate box plots.")
    args = parser.parse_args(argv)
    collect_metric_json(args.results_dir)
    create_summary_tables(args.results_dir)
    if args.boxplots:
        create_boxplots(args.results_dir)


if __name__ == "__main__":
    main()
