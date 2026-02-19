"""Allow running the package as a module: python -m exp_compare."""

from exp_compare.main_cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
