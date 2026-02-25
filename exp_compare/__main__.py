"""Allow running the package as a module: python -m exp_compare.

This module provides the entry point for running exp_compare as a script.

Usage:
    python -m exp_compare
    python -m exp_compare --config path/to/config.yaml
"""

import sys

from exp_compare.main_cli import main

if __name__ == "__main__":
    sys.exit(main())
