from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paper_fan2023_hea.config import load_config
from paper_fan2023_hea.runner import run_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone Fan 2023 HEA implementation."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = load_config(args.config)
    output_dir = run_from_config(config, quiet=args.quiet)
    if not args.quiet:
        print(f"Fan 2023 HEA outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
