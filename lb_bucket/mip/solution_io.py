from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .dispatch_windows import compute_dispatch_window_payload
from .shared import BucketModelVars, BucketSearchResult, SearchTraceRow, TwoBucketInstance


def extract_solution_payload(
    model_vars: BucketModelVars,
    *,
    instance: TwoBucketInstance,
    result: BucketSearchResult,
    trace_row: SearchTraceRow,
    t_lower: int,
    t_upper: int,
    precedence_formulation: str,
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
            "precedence_formulation": precedence_formulation,
            "value_tolerance": value_tolerance,
        },
        "a": _extract_operation_bucket_rows(model_vars.a, value_tolerance),
        "b": _extract_operation_bucket_rows(model_vars.b, value_tolerance),
        "c": _extract_operation_bucket_rows(model_vars.c, value_tolerance),
        "x": _extract_operation_bucket_rows(model_vars.x, value_tolerance),
        "u": _extract_bucket_rows(model_vars.u, value_tolerance),
        "z": _extract_bucket_rows(model_vars.z, value_tolerance),
    }
    dispatch_window_payload = compute_dispatch_window_payload(
        instance,
        payload,
        cmax=result.horizon_ub,
        value_tolerance=value_tolerance,
    )
    payload["metadata"].update(dispatch_window_payload["metadata"])
    payload["dispatch_window_inputs"] = dispatch_window_payload["dispatch_window_inputs"]
    payload["dispatch_windows"] = dispatch_window_payload["dispatch_windows"]
    return payload


def write_solution_payload(output_dir: Path, payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    solution_dir = output_dir / "solutions" / str(metadata["ins_name"])
    solution_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = solution_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_to_jsonable(metadata), indent=2, ensure_ascii=True),
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

    if "dispatch_window_inputs" in payload:
        _write_csv_rows(
            solution_dir / "dispatch_window_inputs.csv",
            (
                "stage",
                "job",
                "processing_time",
                "a_bucket",
                "b_bucket",
                "x_bucket_1",
                "x_value_1",
                "x_bucket_2",
                "x_value_2",
                "x_at_a_bucket",
                "x_at_b_bucket",
                "spans_two_buckets",
            ),
            payload["dispatch_window_inputs"],
        )

    if "dispatch_windows" in payload:
        _write_csv_rows(
            solution_dir / "dispatch_windows.csv",
            (
                "stage",
                "job",
                "processing_time",
                "a_bucket",
                "b_bucket",
                "x_bucket_1",
                "x_value_1",
                "x_bucket_2",
                "x_value_2",
                "x_at_a_bucket",
                "x_at_b_bucket",
                "spans_two_buckets",
                "es_candidate",
                "completion_candidate",
                "early_start",
                "late_start",
                "slack",
            ),
            payload["dispatch_windows"],
        )
        _write_csv_rows(
            solution_dir / "operation_debug.csv",
            (
                "stage",
                "job",
                "processing_time",
                "a_bucket",
                "b_bucket",
                "x_bucket_1",
                "x_value_1",
                "x_bucket_2",
                "x_value_2",
                "es_candidate",
                "completion_candidate",
                "early_start",
                "late_start",
                "slack",
            ),
            [
                {
                    "stage": row["stage"],
                    "job": row["job"],
                    "processing_time": row["processing_time"],
                    "a_bucket": row["a_bucket"],
                    "b_bucket": row["b_bucket"],
                    "x_bucket_1": row["x_bucket_1"],
                    "x_value_1": row["x_value_1"],
                    "x_bucket_2": row["x_bucket_2"],
                    "x_value_2": row["x_value_2"],
                    "es_candidate": row["es_candidate"],
                    "completion_candidate": row["completion_candidate"],
                    "early_start": row["early_start"],
                    "late_start": row["late_start"],
                    "slack": row["slack"],
                }
                for row in payload["dispatch_windows"]
            ],
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


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return value
