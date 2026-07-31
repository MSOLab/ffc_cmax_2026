from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lb_bucket.mip.shared import log_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot time-series UB/LB curves from saved *_progress_trace.csv files."
    )
    parser.add_argument(
        "--progress-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing per-instance *_progress_trace.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG figures will be written. Defaults to <progress-root>/progress_plots.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        nargs="*",
        help="Optional list of instance ids. If omitted, all *_progress_trace.csv files are used.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--include-barrier",
        action="store_true",
        help="Also overlay barrier primal/dual curves when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress_root = args.progress_root
    if not progress_root.is_dir():
        raise FileNotFoundError(f"Progress root not found: {progress_root}")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else progress_root / "progress_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_paths = resolve_progress_paths(progress_root, args.instances)
    log_progress(
        f"Plotting progress curves for {len(progress_paths)} instance(s) from "
        f"{progress_root} into {output_dir}"
    )
    for progress_path in progress_paths:
        rows = load_progress_rows(progress_path)
        if not rows:
            log_progress(f"Skipping empty progress file: {progress_path}")
            continue
        plot_progress_file(
            rows=rows,
            output_path=output_dir / f"{progress_path.stem}.png",
            dpi=args.dpi,
            include_barrier=args.include_barrier,
        )


def resolve_progress_paths(progress_root: Path, instances: list[int] | None) -> list[Path]:
    if instances:
        paths = []
        for instance_id in instances:
            path = progress_root / f"{instance_id}_progress_trace.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Progress trace not found: {path}")
            paths.append(path)
        return paths

    return sorted(
        progress_root.glob("*_progress_trace.csv"),
        key=lambda path: int(path.stem.split("_")[0]),
    )


def load_progress_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_progress_file(
    *,
    rows: list[dict[str, str]],
    output_path: Path,
    dpi: int,
    include_barrier: bool,
) -> None:
    ins_name = str(rows[0]["ins_name"])
    objective_ub = extract_xy(rows, "objective_ub")
    objective_lb = extract_xy(rows, "objective_lb")
    horizon_ub = extract_xy(rows, "horizon_ub")
    horizon_lb = extract_xy(rows, "horizon_lb")
    barrier_primal = extract_xy(rows, "barrier_horizon_primal")
    barrier_dual = extract_xy(rows, "barrier_horizon_dual")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

    plot_series(
        axes[0],
        objective_ub,
        label="Objective UB",
        color="#c0392b",
        linestyle="-",
    )
    plot_series(
        axes[0],
        objective_lb,
        label="Objective LB",
        color="#1f77b4",
        linestyle="-",
    )
    axes[0].set_title(f"Instance {ins_name} - Objective Progress", fontsize=12)
    axes[0].set_ylabel("Objective", fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    plot_series(
        axes[1],
        horizon_ub,
        label="Horizon UB",
        color="#d35400",
        linestyle="-",
    )
    plot_series(
        axes[1],
        horizon_lb,
        label="Horizon LB",
        color="#2980b9",
        linestyle="-",
    )
    if include_barrier:
        plot_series(
            axes[1],
            barrier_primal,
            label="Barrier primal horizon",
            color="#8e44ad",
            linestyle="--",
        )
        plot_series(
            axes[1],
            barrier_dual,
            label="Barrier dual horizon",
            color="#16a085",
            linestyle="--",
        )
    axes[1].set_title(f"Instance {ins_name} - Horizon Progress", fontsize=12)
    axes[1].set_xlabel("Runtime (sec)", fontsize=10)
    axes[1].set_ylabel("Horizon value", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Progress Curves for Instance {ins_name}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log_progress(f"Wrote progress plot for instance {ins_name}: {output_path}")


def extract_xy(rows: list[dict[str, str]], value_key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        runtime = parse_float(row.get("runtime_sec"))
        value = parse_float(row.get(value_key))
        if runtime is None or value is None:
            continue
        points.append((runtime, value))
    return points


def plot_series(
    axis,
    points: list[tuple[float, float]],
    *,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    if not points:
        return
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    axis.step(x_values, y_values, where="post", label=label, color=color, linestyle=linestyle)
    axis.plot(x_values, y_values, "o", color=color, markersize=3)


def parse_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    stripped = str(raw_value).strip()
    if stripped == "":
        return None
    return float(stripped)


if __name__ == "__main__":
    main()
