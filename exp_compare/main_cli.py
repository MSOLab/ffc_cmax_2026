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
    """Configure logging handlers.

    Args:
        quiet (bool): If True, suppress console output (logs still written to file).
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_path = Path("exp_compare.log").resolve()

    has_file = False
    has_console = False

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename).resolve() == log_path:
                    has_file = True
            except Exception:
                pass
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in {
            sys.stderr,
            sys.stdout,
        }:
            has_console = True

    if not quiet and not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    if not has_file:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


def load_config(config_path: str) -> CompareConfig:
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to the comparison configuration YAML file.

    Returns:
        CompareConfig: CompareConfig object with validated configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or config validation fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CompareConfig(**raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config: {e}")
    except Exception as e:
        raise ValueError(f"Invalid config: {e}")


def get_default_config_path() -> Path:
    """Get the default config path (exp_compare/config.yaml).

    Returns:
        Path: Path to the default configuration file.
    """
    # Get the directory where this module is located
    current_file = Path(__file__).resolve()
    return current_file.parent / "config.yaml"


def main() -> int:
    """Main entry point for the experiment comparison CLI.

    Returns:
        int: Exit code: 0 for success, 2 for config/file errors, 1 for other errors.
    """
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
