"""Build all P1 (FFc||Cmax) algorithm-explainer Gantt panels (plan §5/§6).

Runs the demo-instance maker and every per-algorithm runner in sequence, so a
single command regenerates the full slide-panel set under
``analysis_outputs/20260612_p1_algo_explainer/{md,bn2d,cp_lb,isw_cp,qsr}/``.

Each runner is invoked as a subprocess via ``uv run python`` so its module-load
side effects stay isolated. The demo instance is built first because every
runner reads ``resources/demo_p1_10x4/1.txt``.

Usage::

    uv run python scripts/p1_algo_explainer/build_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Execution order: the demo instance must exist before any runner reads it.
STEPS: list[str] = [
    "make_demo_instance.py",
    "run_md.py",
    "run_bn2d.py",
    "run_cp_lb.py",
    "run_neh_cp.py",
    "run_isw_cp.py",
    "run_qsr.py",
]


def main() -> int:
    failures: list[str] = []
    for script in STEPS:
        script_path = HERE / script
        print(f"\n=== {script} ===", flush=True)
        result = subprocess.run(
            ["uv", "run", "python", str(script_path)],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            failures.append(script)
            print(f"!!! {script} failed (exit {result.returncode})", flush=True)

    print("\n=== summary ===", flush=True)
    if failures:
        print("FAILED: " + ", ".join(failures), flush=True)
        return 1
    print("All explainer panels regenerated.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
