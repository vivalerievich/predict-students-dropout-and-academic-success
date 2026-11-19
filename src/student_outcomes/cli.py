"""Command-line interface for training on a local UCI data file."""

import argparse
import json
from os import cpu_count
from pathlib import Path

import optuna

from student_outcomes.models import DEFAULT_TRIALS
from student_outcomes.training import run_benchmark


def _positive_integer(value: str) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return integer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark enrollment-time student outcomes."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/data.csv"), help="Local UCI CSV"
    )
    parser.add_argument(
        "--trials",
        type=_positive_integer,
        default=DEFAULT_TRIALS,
        help=f"Optuna trials per ensemble (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_integer,
        default=min(4, cpu_count() or 1),
        help="Maximum validation folds evaluated in parallel (default: up to 4)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include XGBoost and LightGBM (requires extra extended)",
    )
    parser.add_argument(
        "--model-out", type=Path, help="Save the fitted model selected by CV"
    )
    args = parser.parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        result = run_benchmark(
            args.data,
            trials=args.trials,
            jobs=args.jobs,
            extended=args.extended,
            model_out=args.model_out,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
