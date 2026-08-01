from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .shared import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOLUTION_ROOT,
    DEFAULT_SUMMARY_CSV,
    ModelStrengtheningOptions,
    PrecedenceOptions,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact two-bucket Gurobi relaxation from "
            "lb_bucket/manuscript.tex using summary bestBound values as "
            "the external lower bounds."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="CSV containing insName, bestBound, and optionally bestObj.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing FF2020 benchmark files named '<ins>.txt'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where search traces and the result CSV will be written.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing output directory by keeping the current result CSV "
            "and skipping instances that already have a result row."
        ),
    )
    parser.add_argument(
        "--instances",
        type=int,
        nargs="*",
        help="Optional explicit list of instance ids. If omitted, use all rows in summary-csv.",
    )
    parser.add_argument(
        "--solution-root",
        type=Path,
        default=DEFAULT_SOLUTION_ROOT,
        help=(
            "Optional root directory for persisted incumbent solution YAML files. "
            "The default points to the requested Outputs_scenarios run."
        ),
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=None,
        help=(
            "Optional fixed bucket length override. If omitted, use delta = p_max "
            "for each instance."
        ),
    )
    parser.add_argument(
        "--delta-pmax-plus-one",
        action="store_true",
        help=(
            "Set delta separately for each instance to max processing time + 1. "
            "If omitted, the script uses --delta when provided, otherwise the existing "
            "LB/UB-based automatic delta rule."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional Gurobi thread limit per model solve.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=None,
        help=(
            "Optional Gurobi time limit per T. If hit, the search is not certified and "
            "the script falls back to the input LB for that instance."
        ),
    )
    parser.add_argument(
        "--max-bucket-count",
        type=int,
        default=None,
        help=(
            "Optional absolute cap for the search. If omitted, ceil(bestObj/delta) from the "
            "summary is used whenever available. This is used only when --delta is fixed."
        ),
    )
    parser.add_argument(
        "--same-bucket-threshold",
        type=int,
        default=10,
        help=(
            "Legacy option kept for compatibility. It is ignored by the current "
            "default delta policy."
        ),
    )
    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Enable Gurobi console logs.",
    )
    parser.add_argument(
        "--display-interval-sec",
        type=int,
        default=1,
        help=(
            "Gurobi DisplayInterval parameter in seconds. "
            "Lower values print progress more frequently."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-model Gurobi log files. "
            "If omitted, only console logging is used."
        ),
    )
    parser.add_argument(
        "--disable-ub-warm-start",
        action="store_true",
        help=(
            "Disable loading the persisted incumbent UB schedule and passing it to Gurobi "
            "as a MIP start."
        ),
    )
    parser.add_argument(
        "--precedence-formulation",
        choices=("bucket", "d", "e"),
        default="bucket",
        help=(
            "Choose one precedence formulation: "
            "'bucket' uses precedence-bucket-prefix + precedence-bucket, "
            "'d' uses the cumulative d-based formulation, "
            "'e' uses the binary e-based linking formulation."
        ),
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help=(
            "Disable cumulative precedence and all four valid-inequality families, "
            "leaving only the base two-bucket model."
        ),
    )
    parser.add_argument(
        "--disable-cumulative-precedence",
        action="store_true",
        help="Legacy flag kept for compatibility; ignored by the current range-based formulation.",
    )
    parser.add_argument(
        "--disable-valid-ineq-i",
        action="store_true",
        help="Disable family (i): earliest/latest bucket fixings and residual bounds.",
    )
    parser.add_argument(
        "--disable-valid-ineq-ii",
        action="store_true",
        help="Disable family (ii): head/tail linking cuts.",
    )
    parser.add_argument(
        "--disable-valid-ineq-iii",
        action="store_true",
        help="Disable family (iii): job-chain lower-bound cuts.",
    )
    parser.add_argument(
        "--disable-valid-ineq-iv",
        action="store_true",
        help="Disable family (iv): stage-based cuts and global scalar cut.",
    )
    return parser.parse_args(argv)


def resolve_strengthening_options(
    args: argparse.Namespace,
) -> ModelStrengtheningOptions:
    if args.base_model_only:
        return ModelStrengtheningOptions(
            cumulative_precedence=False,
            valid_ineq_i=False,
            valid_ineq_ii=False,
            valid_ineq_iii=False,
            valid_ineq_iv=False,
        )
    return ModelStrengtheningOptions(
        cumulative_precedence=False,
        valid_ineq_i=not args.disable_valid_ineq_i,
        valid_ineq_ii=not args.disable_valid_ineq_ii,
        valid_ineq_iii=not args.disable_valid_ineq_iii,
        valid_ineq_iv=not args.disable_valid_ineq_iv,
    )


def resolve_precedence_options(args: argparse.Namespace) -> PrecedenceOptions:
    return PrecedenceOptions(formulation=args.precedence_formulation)
