#!/usr/bin/env python
"""CLI entry point for experiment comparison."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from exp_compare.main import CompareConfig, run_comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare experiment results across multiple runs and compute RPD metrics."
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to comparison configuration YAML file (default: exp_compare/config.yaml)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (logs still written to file)",
    )
    return parser.parse_args()


def setup_logging(quiet: bool) -> None:
    """Configure logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

    # File handler (always present)
    log_path = Path("exp_compare.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


def load_config(config_path: str) -> CompareConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
        return CompareConfig(**raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config: {e}")
    except Exception as e:
        raise ValueError(f"Invalid config: {e}")


def get_default_config_path() -> Path:
    """Get the default config path (exp_compare/config.yaml)."""
    # Get the directory where this module is located
    current_file = Path(__file__).resolve()
    return current_file.parent / "config.yaml"


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(args.quiet)

    # Use default config path if not specified
    config_path = args.config if args.config else str(get_default_config_path())

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logging.error(str(e))
        return 2
    except ValueError as e:
        logging.error(str(e))
        return 2

    exit_code = run_comparison(config)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
