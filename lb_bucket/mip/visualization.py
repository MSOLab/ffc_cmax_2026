from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


def write_solution_payload_visualizations(
    output_dir: Path,
    solution_payload: Mapping[str, Any],
    *,
    jobs_per_figure: int = 20,
    subplot_cols: int = 4,
    dispatch_jobs_per_figure: int = 10,
    dispatch_subplot_cols: int = 2,
    dpi: int = 220,
) -> None:
    """Persist analysis-friendly plots for a saved bucket-MIP solution payload."""
    metadata = dict(solution_payload.get("metadata", {}) or {})
    ins_name = metadata.get("ins_name")
    if ins_name is None:
        logging.warning("[MIP LB] Skipping visualization export: missing ins_name.")
        return

    solution_dir = output_dir / "solutions" / str(ins_name)
    solution_dir.mkdir(parents=True, exist_ok=True)
    plot_root = solution_dir / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    try:
        _write_stage_bucket_x_heatmaps(plot_root, solution_payload, dpi=dpi)
    except Exception:
        logging.exception("[MIP LB] Failed to export stage-bucket x utilization plot.")

    try:
        _write_variable_heatmap_pages(
            plot_root / "bucket_heatmaps",
            solution_payload,
            variable_names=("x", "a", "b"),
            jobs_per_figure=jobs_per_figure,
            subplot_cols=subplot_cols,
            dpi=dpi,
        )
    except Exception:
        logging.exception("[MIP LB] Failed to export bucket heatmap pages.")

    try:
        _write_dispatch_window_pages(
            plot_root / "dispatch_windows",
            solution_payload,
            jobs_per_figure=dispatch_jobs_per_figure,
            subplot_cols=dispatch_subplot_cols,
            dpi=dpi,
        )
    except Exception:
        logging.exception("[MIP LB] Failed to export dispatch-window plots.")


def write_dispatch_schedule_window_visualizations(
    output_dir: Path,
    solution_payload: Mapping[str, Any] | None,
    dispatched_schedules: Mapping[str, HybridFlowshopLiteSchedule | None],
    *,
    variant_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    jobs_per_figure: int = 10,
    subplot_cols: int = 2,
    dpi: int = 220,
) -> None:
    """Persist per-dispatch overlays of realized operations on top of ES/LS windows."""
    if solution_payload is None or not dispatched_schedules:
        return

    dispatch_windows = list(solution_payload.get("dispatch_windows", []) or [])
    if not dispatch_windows:
        return

    plot_root = output_dir / "dispatch" / "dispatch_window_overlays"
    plot_root.mkdir(parents=True, exist_ok=True)

    metadata = dict(solution_payload.get("metadata", {}) or {})
    job_labels = _get_entity_labels(
        metadata,
        key="job_ids",
        fallback_count=int(metadata.get("job_count") or 0),
        fallback_values=[
            _normalize_identifier(job_id)
            for schedule in dispatched_schedules.values()
            if schedule is not None
            for job_id in schedule.jobs
        ],
    )
    rows_by_job: dict[str, list[dict[str, Any]]] = {}
    for row in dispatch_windows:
        job_key = _resolve_index_label(row["job"], job_labels)
        rows_by_job.setdefault(job_key, []).append(dict(row))
    for row_list in rows_by_job.values():
        row_list.sort(key=lambda row: int(row["stage"]))

    best_dispatch_makespan = min(
        (
            float(schedule.makespan)
            for schedule in dispatched_schedules.values()
            if schedule is not None
        ),
        default=None,
    )

    for variant, schedule in dispatched_schedules.items():
        if schedule is None:
            continue
        try:
            variant_meta = dict(
                variant_metadata.get(variant, {}) if variant_metadata else {}
            )
            artifact_slug = str(variant_meta.get("artifact_slug") or variant)
            display_name = str(variant_meta.get("short_name") or variant)
            title_label = str(variant_meta.get("status_summary") or variant)
            _write_dispatch_overlay_pages(
                plot_root=plot_root / artifact_slug,
                solution_payload=solution_payload,
                schedule=schedule,
                variant_name=display_name,
                variant_title_label=title_label,
                artifact_slug=artifact_slug,
                rows_by_job=rows_by_job,
                best_dispatch_makespan=best_dispatch_makespan,
                jobs_per_figure=jobs_per_figure,
                subplot_cols=subplot_cols,
                dpi=dpi,
            )
        except Exception:
            logging.exception(
                "[MIP LB] Failed to export dispatch-window overlay for variant %s.",
                variant,
            )


def _write_stage_bucket_x_heatmaps(
    plot_root: Path,
    solution_payload: Mapping[str, Any],
    *,
    dpi: int,
) -> None:
    occupied, utilization, csv_rows = _build_stage_bucket_x_matrices(solution_payload)
    if occupied is None or utilization is None:
        return

    _write_csv(plot_root / "stage_bucket_x_utilization.csv", csv_rows)
    _write_heatmap(
        output_path=plot_root / "stage_bucket_x_occupied_time.png",
        matrix=occupied,
        title="Bucket-MIP incumbent stage/bucket occupancy: sum_j x[s,j,t]",
        colorbar_label="sum_j x[s,j,t]",
        cmap_name="YlOrRd",
        dpi=dpi,
        annotation_format="{value:.0f}",
    )
    _write_heatmap(
        output_path=plot_root / "stage_bucket_x_utilization.png",
        matrix=utilization,
        title="Bucket-MIP incumbent x occupancy by stage/bucket",
        colorbar_label="sum_j x[s,j,t] / (m_s * delta)",
        cmap_name="YlOrRd",
        dpi=dpi,
        annotation_format="{value:.2f}",
    )


def _build_stage_bucket_x_matrices(
    solution_payload: Mapping[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, float | int]]]:
    metadata = dict(solution_payload.get("metadata", {}) or {})
    t_upper = int(metadata.get("t_upper") or 0)
    stage_count = int(metadata.get("stage_count") or 0)
    delta = float(metadata.get("delta") or 0.0)
    machine_counts = list(metadata.get("machine_count_per_stage") or [])
    x_rows = list(solution_payload.get("x", []) or [])
    if t_upper <= 0 or stage_count <= 0 or delta <= 0 or not machine_counts:
        return None, None, []

    occupied = np.zeros((stage_count, t_upper), dtype=float)
    for row in x_rows:
        stage_idx = int(row["stage"])
        bucket_idx = int(row["bucket"])
        if 1 <= stage_idx <= stage_count and 1 <= bucket_idx <= t_upper:
            occupied[stage_idx - 1, bucket_idx - 1] += float(row["value"])

    utilization = occupied.copy()
    csv_rows: list[dict[str, float | int]] = []
    for stage_idx in range(stage_count):
        machine_cnt = max(int(machine_counts[stage_idx]), 1)
        capacity = float(machine_cnt) * delta
        utilization[stage_idx, :] = utilization[stage_idx, :] / capacity
        for bucket_idx in range(t_upper):
            csv_rows.append(
                {
                    "stage": stage_idx + 1,
                    "bucket": bucket_idx + 1,
                    "occupied_time": float(occupied[stage_idx, bucket_idx]),
                    "capacity": capacity,
                    "utilization": float(utilization[stage_idx, bucket_idx]),
                }
            )

    return occupied, utilization, csv_rows


def _write_heatmap(
    *,
    output_path: Path,
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap_name: str,
    dpi: int,
    annotation_format: str,
) -> None:
    stage_count, t_upper = matrix.shape
    masked = np.ma.masked_where(matrix <= 0.0, matrix)
    fig, axis = plt.subplots(
        figsize=(max(7.5, t_upper * 0.45), max(3.0, stage_count * 0.6)),
        constrained_layout=True,
    )
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    vmax = max(1.0, float(np.max(matrix)) if matrix.size else 1.0)
    image = axis.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    xticks, xticklabels = _build_bucket_ticks(t_upper)
    axis.set_xticks(xticks)
    axis.set_xticklabels(xticklabels, fontsize=8)
    axis.set_yticks(list(range(stage_count)))
    axis.set_yticklabels([str(idx) for idx in range(1, stage_count + 1)], fontsize=8)
    axis.set_xlabel("Bucket", fontsize=9)
    axis.set_ylabel("Stage", fontsize=9)
    axis.set_title(title, fontsize=12)
    if matrix.size <= 160:
        _annotate_heatmap_cells(axis, matrix, fmt=annotation_format, vmax=vmax)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.94, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=9)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _annotate_heatmap_cells(
    axis: Any,
    matrix: np.ndarray,
    *,
    fmt: str,
    vmax: float,
) -> None:
    threshold = 0.45 * vmax if vmax > 0.0 else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix[row_idx, col_idx])
            if value <= 0.0:
                continue
            text_color = "white" if value >= threshold else "#222222"
            axis.text(
                col_idx,
                row_idx,
                fmt.format(value=value),
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
            )


def _write_variable_heatmap_pages(
    plot_root: Path,
    solution_payload: Mapping[str, Any],
    *,
    variable_names: Sequence[str],
    jobs_per_figure: int,
    subplot_cols: int,
    dpi: int,
) -> None:
    metadata = dict(solution_payload.get("metadata", {}) or {})
    t_upper = int(metadata.get("t_upper") or 0)
    stage_count = int(metadata.get("stage_count") or 0)
    job_count = int(metadata.get("job_count") or 0)
    if t_upper <= 0 or stage_count <= 0 or job_count <= 0:
        return

    job_labels = _get_entity_labels(
        metadata,
        key="job_ids",
        fallback_count=job_count,
    )
    stage_labels = _get_entity_labels(
        metadata,
        key="stage_ids",
        fallback_count=stage_count,
    )
    plot_root.mkdir(parents=True, exist_ok=True)
    for variable_name in variable_names:
        rows = list(solution_payload.get(variable_name, []) or [])
        matrices = _build_job_matrices(
            variable_rows=rows,
            job_count=job_count,
            stage_count=stage_count,
            t_upper=t_upper,
        )
        _plot_variable_pages(
            plot_root=plot_root,
            ins_name=str(metadata["ins_name"]),
            variable_name=variable_name,
            matrices=matrices,
            job_labels=job_labels,
            stage_labels=stage_labels,
            t_upper=t_upper,
            jobs_per_figure=jobs_per_figure,
            subplot_cols=subplot_cols,
            dpi=dpi,
        )


def _write_dispatch_window_pages(
    plot_root: Path,
    solution_payload: Mapping[str, Any],
    *,
    jobs_per_figure: int,
    subplot_cols: int,
    dpi: int,
) -> None:
    metadata = dict(solution_payload.get("metadata", {}) or {})
    dispatch_windows = list(solution_payload.get("dispatch_windows", []) or [])
    if not dispatch_windows:
        return

    delta = float(metadata.get("delta") or 0.0)
    stage_count = int(metadata.get("stage_count") or 0)
    job_count = int(metadata.get("job_count") or 0)
    dispatch_cmax = float(metadata.get("dispatch_cmax") or 0.0)
    if delta <= 0.0 or stage_count <= 0 or job_count <= 0:
        return

    job_labels = _get_entity_labels(
        metadata,
        key="job_ids",
        fallback_count=job_count,
    )
    stage_labels = _get_entity_labels(
        metadata,
        key="stage_ids",
        fallback_count=stage_count,
    )
    plot_root.mkdir(parents=True, exist_ok=True)
    rows_by_job: dict[str, list[dict[str, Any]]] = {job_id: [] for job_id in job_labels}
    for row in dispatch_windows:
        rows_by_job[_resolve_index_label(row["job"], job_labels)].append(dict(row))
    for row_list in rows_by_job.values():
        row_list.sort(key=lambda row: int(row["stage"]))

    job_ids = list(job_labels)
    pages = [
        job_ids[start : start + jobs_per_figure]
        for start in range(0, len(job_ids), jobs_per_figure)
    ]
    legend_handles = [
        Patch(
            facecolor="#4daf4a", edgecolor="none", alpha=0.55, label="Earliest interval"
        ),
        Patch(
            facecolor="#ff7f00", edgecolor="none", alpha=0.55, label="Latest interval"
        ),
        Patch(
            facecolor="#377eb8",
            edgecolor="none",
            alpha=0.28,
            label="x bucket occupancy",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="ES-LS start window",
        ),
    ]

    for page_index, page_job_ids in enumerate(pages, start=1):
        ncols = min(subplot_cols, len(page_job_ids))
        nrows = math.ceil(len(page_job_ids) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4.2, nrows * 2.8),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, job_id in zip(axes.flat, page_job_ids):
            job_rows = rows_by_job.get(job_id, [])
            for row in job_rows:
                stage_idx = int(row["stage"])
                y = stage_idx - 1
                _draw_dispatch_window_row(axis, row=row, y=y, delta=delta)
            axis.set_title(f"Job {job_id}", fontsize=9)
            axis.set_ylim(stage_count - 0.5, -0.5)
            axis.set_yticks(list(range(stage_count)))
            axis.set_yticklabels(stage_labels, fontsize=7)
            axis.set_xlabel("Time", fontsize=8)
            axis.set_ylabel("Stage", fontsize=8)
            axis.set_xlim(0.0, max(dispatch_cmax, delta))
            axis.grid(axis="x", alpha=0.15, linewidth=0.6)

        for axis in axes.flat[len(page_job_ids) :]:
            axis.axis("off")

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=4,
            fontsize=8,
            frameon=False,
        )
        fig.suptitle(
            f"Instance {metadata['ins_name']} - dispatch-window ES/LS and x occupancy",
            fontsize=12,
        )
        output_path = plot_root / f"dispatch_windows_page{page_index:02d}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _write_dispatch_overlay_pages(
    *,
    plot_root: Path,
    solution_payload: Mapping[str, Any],
    schedule: HybridFlowshopLiteSchedule,
    variant_name: str,
    variant_title_label: str,
    artifact_slug: str,
    rows_by_job: Mapping[str, Sequence[Mapping[str, Any]]],
    best_dispatch_makespan: float | None,
    jobs_per_figure: int,
    subplot_cols: int,
    dpi: int,
) -> None:
    metadata = dict(solution_payload.get("metadata", {}) or {})
    delta = float(metadata.get("delta") or 0.0)
    stage_count = int(metadata.get("stage_count") or 0)
    dispatch_cmax = float(metadata.get("dispatch_cmax") or 0.0)
    if delta <= 0.0 or stage_count <= 0:
        return

    plot_root.mkdir(parents=True, exist_ok=True)
    op_lookup = _build_schedule_operation_lookup(schedule)
    job_ids = list(schedule.jobs)
    stage_labels = list(schedule.stages)
    if not job_ids:
        return

    pages = [
        job_ids[start : start + jobs_per_figure]
        for start in range(0, len(job_ids), jobs_per_figure)
    ]
    legend_handles = [
        Patch(
            facecolor="#4daf4a", edgecolor="none", alpha=0.55, label="Earliest interval"
        ),
        Patch(
            facecolor="#ff7f00", edgecolor="none", alpha=0.55, label="Latest interval"
        ),
        Patch(
            facecolor="#377eb8",
            edgecolor="none",
            alpha=0.28,
            label="x bucket occupancy",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="ES-LS start window",
        ),
        Patch(
            facecolor="#1f78b4",
            edgecolor="black",
            alpha=0.78,
            label="Actual op start within ES-LS",
        ),
        Patch(
            facecolor="#d7301f",
            edgecolor="black",
            alpha=0.78,
            label="Actual op start after LS",
        ),
        Patch(
            facecolor="#6a3d9a",
            edgecolor="black",
            alpha=0.78,
            label="Actual op start before ES",
        ),
    ]

    x_limit = max(dispatch_cmax, float(schedule.makespan), delta)
    current_makespan = float(schedule.makespan)
    for page_index, page_job_ids in enumerate(pages, start=1):
        ncols = min(subplot_cols, len(page_job_ids))
        nrows = math.ceil(len(page_job_ids) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 5.0, nrows * 3.15),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, job_id in zip(axes.flat, page_job_ids):
            job_key = _normalize_identifier(job_id)
            job_rows = list(rows_by_job.get(job_key, []))
            for row in job_rows:
                stage_key = _resolve_index_label(row["stage"], stage_labels)
                stage_idx = int(row["stage"])
                y = stage_idx - 1
                _draw_dispatch_window_row(axis, row=row, y=y, delta=delta)
                scheduled_op = op_lookup.get((stage_key, job_key))
                if scheduled_op is not None:
                    _draw_actual_dispatch_operation(
                        axis,
                        row=row,
                        y=y,
                        delta=delta,
                        scheduled_op=scheduled_op,
                        x_limit=x_limit,
                    )
            axis.set_title(f"Job {job_id}", fontsize=9)
            axis.set_ylim(stage_count - 0.5, -0.5)
            axis.set_yticks(list(range(stage_count)))
            axis.set_yticklabels(stage_labels, fontsize=7)
            axis.set_xlabel("Time", fontsize=8)
            axis.set_ylabel("Stage", fontsize=8)
            axis.set_xlim(0.0, x_limit)
            axis.grid(axis="x", alpha=0.15, linewidth=0.6)
            if not job_rows:
                axis.text(
                    0.5,
                    0.5,
                    "No dispatch-window data",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )

        for axis in axes.flat[len(page_job_ids) :]:
            axis.axis("off")

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            fontsize=7.2,
            columnspacing=1.1,
            handletextpad=0.5,
            frameon=False,
        )
        fig.suptitle(
            (
                f"Instance {metadata['ins_name']} | {variant_name}\n"
                f"{variant_title_label} | obj={current_makespan:.0f} | "
                f"best_dispatch_obj={best_dispatch_makespan:.0f} | page {page_index}"
            ),
            fontsize=10.5,
            y=1.065,
        )
        output_path = plot_root / f"{artifact_slug}_page{page_index:02d}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _draw_dispatch_window_row(
    axis: Any,
    *,
    row: Mapping[str, Any],
    y: int,
    delta: float,
) -> None:
    early_start = float(row["early_start"])
    late_start = float(row["late_start"])
    processing_time = float(row["processing_time"])
    _draw_bucket_patch(
        axis,
        bucket_idx=row.get("x_bucket_1"),
        delta=delta,
        y=y,
        occupancy=float(row.get("x_value_1") or 0.0),
    )
    _draw_bucket_patch(
        axis,
        bucket_idx=row.get("x_bucket_2"),
        delta=delta,
        y=y,
        occupancy=float(row.get("x_value_2") or 0.0),
    )
    axis.plot(
        [early_start, late_start],
        [y, y],
        color="black",
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    axis.add_patch(
        Rectangle(
            (early_start, y - 0.26),
            processing_time,
            0.2,
            facecolor="#4daf4a",
            edgecolor="none",
            alpha=0.55,
            zorder=4,
        )
    )
    axis.add_patch(
        Rectangle(
            (late_start, y + 0.06),
            processing_time,
            0.2,
            facecolor="#ff7f00",
            edgecolor="none",
            alpha=0.55,
            zorder=4,
        )
    )


def _build_schedule_operation_lookup(
    schedule: HybridFlowshopLiteSchedule,
) -> dict[tuple[str, str], dict[str, Any]]:
    start_map = schedule.get_jik_2_start_time_map()
    end_map = schedule.get_jik_2_end_time_map()
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for (job_id, stage_id, machine_id), start_time in start_map.items():
        key = (_normalize_identifier(stage_id), _normalize_identifier(job_id))
        lookup[key] = {
            "start": float(start_time),
            "end": float(end_map[(job_id, stage_id, machine_id)]),
            "machine": str(machine_id),
        }
    return lookup


def _draw_actual_dispatch_operation(
    axis: Any,
    *,
    row: Mapping[str, Any],
    y: int,
    delta: float,
    scheduled_op: Mapping[str, Any],
    x_limit: float,
) -> None:
    early_start = float(row["early_start"])
    late_start = float(row["late_start"])
    actual_start = float(scheduled_op["start"])
    actual_end = float(scheduled_op["end"])
    actual_duration = max(actual_end - actual_start, 0.0)

    if actual_start > late_start + 1e-9:
        facecolor = "#d7301f"
        delta_label = f"+{actual_start - late_start:.0f}"
    elif actual_start < early_start - 1e-9:
        facecolor = "#6a3d9a"
        delta_label = f"-{early_start - actual_start:.0f}"
    else:
        facecolor = "#1f78b4"
        delta_label = None

    axis.add_patch(
        Rectangle(
            (actual_start, y - 0.08),
            actual_duration,
            0.16,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.78,
            zorder=6,
        )
    )
    axis.plot(
        [actual_start, actual_start],
        [y - 0.16, y + 0.16],
        color="black",
        linewidth=0.8,
        zorder=7,
    )

    machine_label = str(scheduled_op["machine"])
    min_label_width = max(10.0, 0.03 * x_limit, 0.08 * delta)
    if actual_duration >= min_label_width:
        axis.text(
            actual_start + (actual_duration / 2.0),
            y,
            machine_label,
            fontsize=5.8,
            ha="center",
            va="center",
            color="white",
            zorder=8,
        )
    if delta_label is not None:
        label_pad = max(8.0, 0.02 * delta, 0.01 * x_limit)
        if delta_label.startswith("-"):
            label_x = max(0.0, actual_start - label_pad)
            label_ha = "right"
        else:
            label_x = min(x_limit, actual_end + label_pad)
            label_ha = "left"
        axis.text(
            label_x,
            y + 0.22,
            delta_label,
            fontsize=6.2,
            ha=label_ha,
            va="bottom",
            color=facecolor,
            zorder=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.15,
            },
        )


def _draw_bucket_patch(
    axis: Any,
    *,
    bucket_idx: Any,
    delta: float,
    y: int,
    occupancy: float,
) -> None:
    if bucket_idx in (None, "") or occupancy <= 0.0:
        return
    bucket = int(bucket_idx)
    bucket_start = float(bucket - 1) * delta
    share = min(1.0, occupancy / delta) if delta > 0.0 else 0.0
    axis.add_patch(
        Rectangle(
            (bucket_start, y - 0.35),
            delta,
            0.7,
            facecolor="#377eb8",
            edgecolor="none",
            alpha=0.18 + 0.45 * share,
            zorder=1,
        )
    )
    axis.text(
        bucket_start + (delta / 2.0),
        y + 0.34,
        _format_compact_number(occupancy),
        fontsize=6.2,
        ha="center",
        va="bottom",
        color="#2f4f6f",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.55,
            "pad": 0.1,
        },
    )


def _plot_variable_pages(
    *,
    plot_root: Path,
    ins_name: str,
    variable_name: str,
    matrices: Mapping[int, np.ma.MaskedArray],
    job_labels: Sequence[str],
    stage_labels: Sequence[str],
    t_upper: int,
    jobs_per_figure: int,
    subplot_cols: int,
    dpi: int,
) -> None:
    job_ids = list(sorted(matrices))
    pages = [
        job_ids[start : start + jobs_per_figure]
        for start in range(0, len(job_ids), jobs_per_figure)
    ]
    xticks, xticklabels = _build_bucket_ticks(t_upper)
    yticks = list(range(len(stage_labels)))
    yticklabels = list(stage_labels)
    cmap = _select_cmap(variable_name)
    for page_index, page_job_ids in enumerate(pages, start=1):
        ncols = min(subplot_cols, len(page_job_ids))
        nrows = math.ceil(len(page_job_ids) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 3.1, nrows * 2.7),
            squeeze=False,
            constrained_layout=True,
        )
        image = None
        for axis, job_idx in zip(axes.flat, page_job_ids):
            image = axis.imshow(
                matrices[job_idx],
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_title(f"Job {job_labels[job_idx - 1]}", fontsize=9)
            axis.set_xticks(xticks)
            axis.set_xticklabels(xticklabels, fontsize=7)
            axis.set_yticks(yticks)
            axis.set_yticklabels(yticklabels, fontsize=7)
            axis.set_xlabel("Bucket", fontsize=8)
            axis.set_ylabel("Stage", fontsize=8)
        for axis in axes.flat[len(page_job_ids) :]:
            axis.axis("off")
        if image is not None:
            colorbar = fig.colorbar(
                image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.01
            )
            colorbar.set_label("Within (job, stage) share", fontsize=9)
        fig.suptitle(
            f"Instance {ins_name} - {variable_name} bucket heatmaps by job",
            fontsize=12,
        )
        output_path = plot_root / f"{variable_name}_page{page_index:02d}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _build_job_matrices(
    *,
    variable_rows: Sequence[Mapping[str, Any]],
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
        matrices[job_idx][stage_idx - 1, bucket_idx - 1] = float(row["value"])
    return _normalize_and_mask_matrices(matrices)


def _normalize_and_mask_matrices(
    matrices: Mapping[int, np.ndarray],
) -> dict[int, np.ma.MaskedArray]:
    normalized: dict[int, np.ma.MaskedArray] = {}
    for job_idx, matrix in matrices.items():
        normalized_matrix = matrix.copy()
        for row_idx in range(normalized_matrix.shape[0]):
            row_sum = float(normalized_matrix[row_idx].sum())
            if row_sum > 0.0:
                normalized_matrix[row_idx] /= row_sum
        normalized[job_idx] = np.ma.masked_where(
            normalized_matrix <= 0.0, normalized_matrix
        )
    return normalized


def _build_bucket_ticks(t_upper: int) -> tuple[list[int], list[str]]:
    if t_upper <= 8:
        ticks = list(range(t_upper))
    else:
        step = max(1, math.ceil(t_upper / 6))
        ticks = sorted({0, *range(step - 1, t_upper, step), t_upper - 1})
    labels = [str(tick + 1) for tick in ticks]
    return ticks, labels


def _select_cmap(variable_name: str):
    if variable_name == "x":
        cmap_name = "YlOrRd"
    elif variable_name == "a":
        cmap_name = "Greens"
    else:
        cmap_name = "Blues"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_identifier(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _format_compact_number(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) <= 1e-9:
        return str(int(rounded))
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def _get_entity_labels(
    metadata: Mapping[str, Any],
    *,
    key: str,
    fallback_count: int,
    fallback_values: Sequence[str] | None = None,
) -> list[str]:
    raw_values = list(metadata.get(key) or [])
    if raw_values:
        return [_normalize_identifier(value) for value in raw_values]

    if fallback_values:
        deduped = list(
            dict.fromkeys(_normalize_identifier(value) for value in fallback_values)
        )
        if deduped:
            return deduped[:fallback_count] if fallback_count > 0 else deduped

    return [str(idx) for idx in range(1, fallback_count + 1)]


def _resolve_index_label(raw_value: Any, labels: Sequence[str]) -> str:
    if isinstance(raw_value, (int, np.integer)):
        idx = int(raw_value)
        if 1 <= idx <= len(labels):
            return labels[idx - 1]
    if isinstance(raw_value, float) and raw_value.is_integer():
        idx = int(raw_value)
        if 1 <= idx <= len(labels):
            return labels[idx - 1]

    normalized = _normalize_identifier(raw_value)
    return normalized
