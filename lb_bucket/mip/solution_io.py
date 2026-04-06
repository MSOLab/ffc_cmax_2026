from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .shared import BucketModelVars, BucketSearchResult, SearchTraceRow, TwoBucketInstance


def extract_solution_payload(
    model_vars: BucketModelVars,
    *,
    instance: TwoBucketInstance,
    result: BucketSearchResult,
    trace_row: SearchTraceRow,
    t_lower: int,
    t_upper: int,
    binary_job_ids: tuple[int, ...] | None = None,
    value_tolerance: float = 1e-9,
) -> dict[str, Any] | None:
    if trace_row.solution_count <= 0:
        return None

    payload = {
        "metadata": {
            "ins_name": instance.ins_name,
            "job_count": instance.job_count,
            "stage_count": instance.stage_count,
            "machine_count_per_stage": instance.machine_count_per_stage,
            "delta": result.delta,
            "input_lb": result.input_lb,
            "input_ub": result.input_ub,
            "t_lower": t_lower,
            "t_upper": t_upper,
            "objective_bucket_count": t_upper - t_lower,
            "status": trace_row.status,
            "solution_count": trace_row.solution_count,
            "objective_ub": trace_row.objective_ub,
            "objective_lb": trace_row.objective_lb,
            "horizon_ub": trace_row.horizon_ub,
            "horizon_lb": trace_row.horizon_lb,
            "bucket_indexed_lb": result.bucket_indexed_lb,
            "certified_final_lb": result.certified_final_lb,
            "search_certified": result.search_certified,
            "termination_reason": result.termination_reason,
            "binary_job_ids": list(binary_job_ids or ()),
            "value_tolerance": value_tolerance,
        },
        "a": _extract_operation_bucket_rows(model_vars.a, value_tolerance),
        "b": _extract_operation_bucket_rows(model_vars.b, value_tolerance),
        "c": _extract_operation_bucket_rows(model_vars.c, value_tolerance),
        "x": _extract_operation_bucket_rows(model_vars.x, value_tolerance),
        "u": _extract_bucket_rows(model_vars.u, value_tolerance),
        "z": _extract_bucket_rows(model_vars.z, value_tolerance),
    }
    return payload


def write_solution_payload(output_dir: Path, payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    solution_dir = output_dir / "solutions" / str(metadata["ins_name"])
    solution_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = solution_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    for variable_name in ("a", "b", "c", "x"):
        rows = payload[variable_name]
        _write_csv_rows(
            solution_dir / f"{variable_name}.csv",
            ("stage", "job", "bucket", "value"),
            rows,
        )

    for variable_name in ("u", "z"):
        rows = payload[variable_name]
        _write_csv_rows(
            solution_dir / f"{variable_name}.csv",
            ("bucket", "value"),
            rows,
        )


def _extract_operation_bucket_rows(var_dict: Any, value_tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (stage_idx, job_idx, bucket_idx), var in var_dict.items():
        value = float(var.X)
        if abs(value) <= value_tolerance:
            continue
        rows.append(
            {
                "stage": stage_idx,
                "job": job_idx,
                "bucket": bucket_idx,
                "value": value,
            }
        )
    rows.sort(key=lambda row: (row["job"], row["stage"], row["bucket"]))
    return rows


def _extract_bucket_rows(var_dict: Any, value_tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket_idx, var in var_dict.items():
        value = float(var.X)
        if abs(value) <= value_tolerance:
            continue
        rows.append({"bucket": bucket_idx, "value": value})
    rows.sort(key=lambda row: row["bucket"])
    return rows


def _write_csv_rows(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
