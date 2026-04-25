from __future__ import annotations

import argparse
import sys
from pathlib import Path

from routix import ElapsedTimer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hfs_config import MainMetadata
from main import (
    _setup_logging,
    determine_run_mode_and_base_dir,
    read_yaml,
    run_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hybrid Flowshop experiments using a chosen metadata YAML.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to a metadata YAML file compatible with MainMetadata.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output (log is still written to file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    e_timer = ElapsedTimer()
    raw_metadata = read_yaml(args.metadata)
    config = MainMetadata.model_validate(raw_metadata)
    run_mode, base_output_dir_path, prev_flow, resume_dir = (
        determine_run_mode_and_base_dir(config, e_timer)
    )
    _setup_logging(
        base_output_dir_path / config.scenario_log_filename,
        quiet=args.quiet,
    )
    run_experiment(
        e_timer,
        config,
        run_mode,
        base_output_dir_path,
        prev_flow,
        resume_dir,
    )


if __name__ == "__main__":
    main()
