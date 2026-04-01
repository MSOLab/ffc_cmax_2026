from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from hybridflowshop.report import export_multi_scenario_method_rpdf_comparison_html

SUMMARY_FILENAME = "summary_method_rpdf_and_norm_time_long.csv"
DEFAULT_OUTPUT_FILENAME = "selected_subroutine_flow_comparison.html"
DEFAULT_LABEL_MODE = "basename"
SCENARIO_PATH_LIST: list[str] = [
    "Outputs_scenarios/20260324T205559_877608/ff2020/20260324-03",
    "Outputs_scenarios/20260331T235223_209924/ff2020/20260331-03",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one HTML chart comparing mean norm_time vs mean RPDf "
            "for selected post-processed scenario directories."
        )
    )
    parser.add_argument(
        "scenario_dirs",
        nargs="*",
        help=(
            "Scenario directories containing "
            f"{SUMMARY_FILENAME}. If omitted, SCENARIO_PATH_LIST is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILENAME),
        help=f"Output HTML path. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--label-mode",
        choices=["basename", "path3"],
        default=DEFAULT_LABEL_MODE,
        help=(
            "How to label each selected scenario trace. "
            "'basename' uses the directory name and automatically "
            "disambiguates duplicates. 'path3' always uses the last "
            "three path components."
        ),
    )
    return parser


def _last_path_components(path: Path, depth: int) -> str:
    parts = [part for part in path.parts if part not in {path.anchor, ""}]
    tail = parts[-depth:] if len(parts) >= depth else parts
    return "/".join(tail) if tail else str(path)


def _deduplicate_labels(labels: list[str]) -> list[str]:
    duplicates = {label for label, count in Counter(labels).items() if count > 1}
    if not duplicates:
        return labels

    seen: Counter[str] = Counter()
    deduplicated: list[str] = []
    for label in labels:
        seen[label] += 1
        if label not in duplicates:
            deduplicated.append(label)
            continue
        deduplicated.append(f"{label} [{seen[label]}]")
    return deduplicated


def resolve_scenario_paths(cli_paths: Sequence[str]) -> list[Path]:
    raw_paths = list(cli_paths) if cli_paths else list(SCENARIO_PATH_LIST)
    return [Path(path).expanduser().resolve() for path in raw_paths]


def build_scenario_labels(
    scenario_paths: Sequence[Path], label_mode: str = DEFAULT_LABEL_MODE
) -> list[str]:
    if label_mode == "path3":
        return _deduplicate_labels(
            [_last_path_components(path, depth=3) for path in scenario_paths]
        )

    base_labels = [path.name for path in scenario_paths]
    counts = Counter(base_labels)
    labels = [
        _last_path_components(path, depth=3) if counts[path.name] > 1 else path.name
        for path in scenario_paths
    ]
    return _deduplicate_labels(labels)


def load_scenario_metrics(
    scenario_paths: Sequence[Path], label_mode: str = DEFAULT_LABEL_MODE
) -> list[tuple[str, pd.DataFrame]]:
    labels = build_scenario_labels(scenario_paths, label_mode=label_mode)
    scenario_metrics: list[tuple[str, pd.DataFrame]] = []

    for scenario_path, label in zip(scenario_paths, labels, strict=True):
        summary_path = scenario_path / SUMMARY_FILENAME
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing required summary CSV at {summary_path}")
        scenario_metrics.append((label, pd.read_csv(summary_path)))

    return scenario_metrics


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    scenario_paths = resolve_scenario_paths(args.scenario_dirs)
    if not scenario_paths:
        logging.error(
            "No scenario directories resolved. Provide CLI paths or populate "
            "SCENARIO_PATH_LIST in the script."
        )
        return 1
    if len(scenario_paths) < 2:
        logging.error("At least 2 scenario directories are required.")
        return 1

    try:
        scenario_metrics = load_scenario_metrics(
            scenario_paths=scenario_paths,
            label_mode=args.label_mode,
        )
    except Exception as e:
        logging.error("Failed to load selected scenario summaries: %s", e)
        return 1

    created = export_multi_scenario_method_rpdf_comparison_html(
        scenario_metrics=scenario_metrics,
        output_path=args.output.resolve(),
    )
    if not created:
        logging.error("No valid aggregated traces were available for plotting.")
        return 1

    logging.info("Selected scenario comparison HTML saved to %s", args.output.resolve())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
