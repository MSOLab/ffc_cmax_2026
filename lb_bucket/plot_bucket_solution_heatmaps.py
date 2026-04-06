from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lb_bucket.mip.shared import log_progress


DEFAULT_SOLUTION_ROOT = Path(__file__).resolve().parent / "results" / "solutions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stage-by-bucket heatmaps for each job from saved bucket-model solutions."
        )
    )
    parser.add_argument(
        "--solution-root",
        type=Path,
        default=DEFAULT_SOLUTION_ROOT,
        help="Directory containing per-instance solution folders with metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG figures will be written. Defaults to <solution-root>/../plots.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        nargs="*",
        help="Optional list of instance ids to plot. If omitted, plot all saved solutions.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        choices=("x", "a", "b"),
        default=("x", "a", "b"),
        help="Variables to visualize.",
    )
    parser.add_argument(
        "--jobs-per-figure",
        type=int,
        default=25,
        help="Maximum number of job subplots per figure page.",
    )
    parser.add_argument(
        "--subplot-cols",
        type=int,
        default=5,
        help="Maximum number of subplot columns per page.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--subplot-scale",
        type=float,
        default=2.6,
        help="Width/height scale per subplot in inches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solution_root = args.solution_root
    if not solution_root.is_dir():
        raise FileNotFoundError(f"Solution root not found: {solution_root}")

    output_dir = (
        args.output_dir if args.output_dir is not None else solution_root.parent / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_dirs = resolve_instance_dirs(solution_root, args.instances)
    log_progress(
        f"Plotting heatmaps for {len(instance_dirs)} instance(s) from {solution_root} "
        f"into {output_dir}"
    )
    for instance_dir in instance_dirs:
        metadata = load_metadata(instance_dir)
        for variable_name in args.variables:
            variable_rows = load_variable_rows(instance_dir / f"{variable_name}.csv")
            plot_variable_pages(
                metadata=metadata,
                variable_name=variable_name,
                variable_rows=variable_rows,
                output_dir=output_dir / str(metadata["ins_name"]),
                jobs_per_figure=args.jobs_per_figure,
                subplot_cols=args.subplot_cols,
                dpi=args.dpi,
                subplot_scale=args.subplot_scale,
            )


def resolve_instance_dirs(solution_root: Path, instances: list[int] | None) -> list[Path]:
    if instances:
        dirs = []
        for instance_id in instances:
            instance_dir = solution_root / str(instance_id)
            if not instance_dir.is_dir():
                raise FileNotFoundError(f"Saved solution folder not found: {instance_dir}")
            dirs.append(instance_dir)
        return dirs

    dirs = [path for path in solution_root.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda path: int(path.name))


def load_metadata(instance_dir: Path) -> dict[str, object]:
    metadata_path = instance_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_variable_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Variable CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_variable_pages(
    *,
    metadata: dict[str, object],
    variable_name: str,
    variable_rows: list[dict[str, str]],
    output_dir: Path,
    jobs_per_figure: int,
    subplot_cols: int,
    dpi: int,
    subplot_scale: float,
) -> None:
    ins_name = str(metadata["ins_name"])
    job_count = int(metadata["job_count"])
    stage_count = int(metadata["stage_count"])
    t_upper = int(metadata["t_upper"])

    matrices = build_job_matrices(
        variable_rows=variable_rows,
        job_count=job_count,
        stage_count=stage_count,
        t_upper=t_upper,
    )
    vmax = determine_vmax(variable_name, matrices)
    job_ids = list(range(1, job_count + 1))
    pages = [
        job_ids[start : start + jobs_per_figure]
        for start in range(0, len(job_ids), jobs_per_figure)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for page_index, page_job_ids in enumerate(pages, start=1):
        ncols = min(subplot_cols, len(page_job_ids))
        nrows = math.ceil(len(page_job_ids) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * subplot_scale, nrows * subplot_scale),
            squeeze=False,
            constrained_layout=True,
        )
        image = None
        cmap = select_cmap(variable_name)
        xticks, xticklabels = build_bucket_ticks(t_upper)
        yticks = list(range(stage_count))
        yticklabels = [str(stage_idx) for stage_idx in range(1, stage_count + 1)]

        for axis, job_idx in zip(axes.flat, page_job_ids):
            image = axis.imshow(
                matrices[job_idx],
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
            )
            axis.set_title(f"Job {job_idx}", fontsize=9)
            axis.set_xticks(xticks)
            axis.set_xticklabels(xticklabels, fontsize=7)
            axis.set_yticks(yticks)
            axis.set_yticklabels(yticklabels, fontsize=7)
            axis.set_xlabel("Bucket", fontsize=8)
            axis.set_ylabel("Stage", fontsize=8)

        for axis in axes.flat[len(page_job_ids) :]:
            axis.axis("off")

        if image is not None:
            colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.01)
            colorbar.set_label("Within (job, stage) share", fontsize=9)
        fig.suptitle(
            f"Instance {ins_name} - {variable_name} heatmaps by job (row-normalized)",
            fontsize=12,
        )

        output_path = output_dir / f"{variable_name}_page{page_index:02d}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log_progress(f"Wrote {variable_name} heatmap page for instance {ins_name}: {output_path}")


def build_job_matrices(
    *,
    variable_rows: list[dict[str, str]],
    job_count: int,
    stage_count: int,
    t_upper: int,
) -> dict[int, np.ma.MaskedArray]:
    matrices = {
        job_idx: np.zeros((stage_count, t_upper), dtype=float)
        for job_idx in range(1, job_count + 1)
    }
    for row in variable_rows:
        stage_idx = int(row["stage"])
        job_idx = int(row["job"])
        bucket_idx = int(row["bucket"])
        value = float(row["value"])
        matrices[job_idx][stage_idx - 1, bucket_idx - 1] = value
    return normalize_and_mask_matrices(matrices)


def normalize_and_mask_matrices(
    matrices: dict[int, np.ndarray],
) -> dict[int, np.ma.MaskedArray]:
    normalized: dict[int, np.ma.MaskedArray] = {}
    for job_idx, matrix in matrices.items():
        normalized_matrix = matrix.copy()
        for row_idx in range(normalized_matrix.shape[0]):
            row_sum = float(normalized_matrix[row_idx].sum())
            if row_sum > 0.0:
                normalized_matrix[row_idx] /= row_sum
        normalized[job_idx] = np.ma.masked_where(normalized_matrix <= 0.0, normalized_matrix)
    return normalized


def determine_vmax(variable_name: str, matrices: dict[int, np.ma.MaskedArray]) -> float:
    return 1.0


def build_bucket_ticks(t_upper: int) -> tuple[list[int], list[str]]:
    if t_upper <= 8:
        ticks = list(range(t_upper))
    else:
        step = max(1, math.ceil(t_upper / 6))
        ticks = sorted({0, *range(step - 1, t_upper, step), t_upper - 1})
    labels = [str(tick + 1) for tick in ticks]
    return ticks, labels


def select_cmap(variable_name: str):
    cmap_name = "YlOrRd" if variable_name == "x" else "Blues"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


if __name__ == "__main__":
    main()
