from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybridflowshop.io_solution import get_end_time_dict, get_start_time_dict

from .shared import BucketModelVars, TwoBucketInstance


@dataclass(frozen=True)
class ParsedUbSchedule:
    solution_path: Path
    makespan: int
    operation_intervals: dict[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class BucketWarmStart:
    z_value: int
    a_values: dict[tuple[int, int, int], int]
    b_values: dict[tuple[int, int, int], int]
    c_values: dict[tuple[int, int, int], float]
    x_values: dict[tuple[int, int, int], float]


def resolve_solution_path(
    summary_csv: Path,
    ins_name: str,
    solution_root: Path | None = None,
) -> Path | None:
    search_root = solution_root if solution_root is not None else summary_csv.parent
    candidate_paths = [
        search_root / ins_name / "results" / f"{ins_name}_solution.yaml",
        search_root / ins_name / f"{ins_name}_solution.yaml",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return candidate_path
    return None


def load_ub_schedule(
    instance: TwoBucketInstance,
    solution_path: Path,
) -> ParsedUbSchedule:
    start_time_map = get_start_time_dict(solution_path)
    end_time_map = get_end_time_dict(solution_path)

    operation_intervals: dict[tuple[int, int], tuple[int, int]] = {}
    makespan = 0

    for operation_key, start_time in start_time_map.items():
        if operation_key not in end_time_map:
            raise ValueError(
                f"Solution file {solution_path} is missing an end time for {operation_key}."
            )
        if not isinstance(operation_key, tuple) or len(operation_key) != 3:
            raise ValueError(
                f"Unexpected solution key {operation_key!r} in {solution_path}; "
                "expected (job, stage, machine)."
            )

        job_label, stage_label, _machine_label = operation_key
        job_idx = _parse_zero_based_suffix(job_label, "j") + 1
        stage_idx = _parse_zero_based_suffix(stage_label, "i") + 1
        if not (1 <= job_idx <= instance.job_count):
            raise ValueError(
                f"Solution file {solution_path} has job index {job_idx} outside "
                f"1..{instance.job_count}."
            )
        if not (1 <= stage_idx <= instance.stage_count):
            raise ValueError(
                f"Solution file {solution_path} has stage index {stage_idx} outside "
                f"1..{instance.stage_count}."
            )

        end_time = int(end_time_map[operation_key])
        start_time = int(start_time)
        if end_time <= start_time:
            raise ValueError(
                f"Operation {(job_label, stage_label)} in {solution_path} has "
                f"non-positive duration [{start_time}, {end_time})."
            )

        expected_processing_time = instance.processing_times_by_stage[stage_idx - 1][
            job_idx - 1
        ]
        if end_time - start_time != expected_processing_time:
            raise ValueError(
                f"Operation {(job_label, stage_label)} in {solution_path} has duration "
                f"{end_time - start_time}, but the instance expects "
                f"{expected_processing_time}."
            )

        operation_id = (stage_idx, job_idx)
        if operation_id in operation_intervals:
            raise ValueError(
                f"Duplicate operation {(job_label, stage_label)} detected in {solution_path}."
            )

        operation_intervals[operation_id] = (start_time, end_time)
        makespan = max(makespan, end_time)

    expected_operation_count = instance.stage_count * instance.job_count
    if len(operation_intervals) != expected_operation_count:
        raise ValueError(
            f"Solution file {solution_path} contains {len(operation_intervals)} operations, "
            f"but the instance expects {expected_operation_count}."
        )

    return ParsedUbSchedule(
        solution_path=solution_path,
        makespan=makespan,
        operation_intervals=operation_intervals,
    )


def build_bucket_warm_start(
    instance: TwoBucketInstance,
    ub_schedule: ParsedUbSchedule,
    bucket_count: int,
    delta: int,
) -> BucketWarmStart | None:
    if ub_schedule.makespan > bucket_count * delta:
        return None

    a_values: dict[tuple[int, int, int], int] = {}
    b_values: dict[tuple[int, int, int], int] = {}
    c_values: dict[tuple[int, int, int], float] = {}
    x_values: dict[tuple[int, int, int], float] = {}
    last_bucket_job_load = {job_idx: 0.0 for job_idx in range(1, instance.job_count + 1)}
    last_bucket_stage_load = {
        stage_idx: 0.0 for stage_idx in range(1, instance.stage_count + 1)
    }

    for (stage_idx, job_idx), (start_time, end_time) in ub_schedule.operation_intervals.items():
        start_bucket = start_time // delta + 1
        end_bucket = (end_time - 1) // delta + 1
        if end_bucket > bucket_count:
            raise ValueError(
                f"Warm-start schedule for instance {instance.ins_name} reaches bucket "
                f"{end_bucket}, beyond T={bucket_count}."
            )

        a_values[stage_idx, job_idx, start_bucket] = 1
        b_values[stage_idx, job_idx, end_bucket] = 1
        if start_bucket == end_bucket:
            c_values[stage_idx, job_idx, start_bucket] = 1.0

        for bucket_idx in range(start_bucket, end_bucket + 1):
            bucket_start = (bucket_idx - 1) * delta
            bucket_end = bucket_idx * delta
            overlap = max(0, min(end_time, bucket_end) - max(start_time, bucket_start))
            if overlap > 0:
                x_values[stage_idx, job_idx, bucket_idx] = float(overlap)
                if bucket_idx == bucket_count:
                    last_bucket_job_load[job_idx] += overlap
                    last_bucket_stage_load[stage_idx] += overlap

    max_job_load = max(last_bucket_job_load.values(), default=0.0)
    max_stage_average_load = max(
        (
            last_bucket_stage_load[stage_idx]
            / instance.machine_count_per_stage[stage_idx - 1]
        )
        for stage_idx in range(1, instance.stage_count + 1)
    )
    z_value = math.ceil(max(max_job_load, max_stage_average_load))
    if z_value > delta:
        raise ValueError(
            f"Warm-start schedule for instance {instance.ins_name} requires z={z_value}, "
            f"which exceeds delta={delta}."
        )

    return BucketWarmStart(
        z_value=z_value,
        a_values=a_values,
        b_values=b_values,
        c_values=c_values,
        x_values=x_values,
    )


def apply_bucket_warm_start(
    model_vars: BucketModelVars,
    warm_start: BucketWarmStart,
) -> None:
    for var in model_vars.a.values():
        var.Start = 0.0
    for var in model_vars.b.values():
        var.Start = 0.0
    for var in model_vars.c.values():
        var.Start = 0.0
    for var in model_vars.x.values():
        var.Start = 0.0
    model_vars.z.Start = float(warm_start.z_value)

    for key, value in warm_start.a_values.items():
        model_vars.a[key].Start = float(value)
    for key, value in warm_start.b_values.items():
        model_vars.b[key].Start = float(value)
    for key, value in warm_start.c_values.items():
        model_vars.c[key].Start = float(value)
    for key, value in warm_start.x_values.items():
        model_vars.x[key].Start = float(value)


def _parse_zero_based_suffix(label: Any, prefix: str) -> int:
    label_text = str(label)
    if not label_text.startswith(prefix):
        raise ValueError(
            f"Expected label starting with '{prefix}', but received {label_text!r}."
        )
    return int(label_text[len(prefix) :])
