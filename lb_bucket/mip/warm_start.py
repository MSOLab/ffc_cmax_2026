from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from routix.io.yaml import load_yaml

from .data import compute_range_bucket_bounds
from .shared import BucketModelVars, TwoBucketInstance


@dataclass(frozen=True)
class ParsedUbSchedule:
    solution_path: Path
    makespan: int
    operation_intervals: dict[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class BucketWarmStart:
    a_values: dict[tuple[int, int, int], int]
    b_values: dict[tuple[int, int, int], int]
    c_values: dict[tuple[int, int, int], float]
    x_values: dict[tuple[int, int, int], float]
    u_values: dict[int, int]
    z_values: dict[int, float]


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

    solution_index = _build_solution_index(search_root)
    return solution_index.get(ins_name)


def load_ub_schedule(
    instance: TwoBucketInstance,
    solution_path: Path,
) -> ParsedUbSchedule:
    solution_dict = _load_solution_dict(solution_path)
    start_time_map = _extract_time_map(solution_dict, "start_time_map", "start_times", solution_path)
    end_time_map = _extract_time_map(solution_dict, "end_time_map", "end_times", solution_path)

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
    input_lb: int,
    input_ub: int,
    delta: int,
) -> BucketWarmStart | None:
    t_lower, t_upper = compute_range_bucket_bounds(input_lb, input_ub, delta)
    if ub_schedule.makespan > t_upper * delta:
        return None

    a_values: dict[tuple[int, int, int], int] = {}
    b_values: dict[tuple[int, int, int], int] = {}
    c_values: dict[tuple[int, int, int], float] = {}
    x_values: dict[tuple[int, int, int], float] = {}
    u_values = {bucket_idx: 0 for bucket_idx in range(t_lower + 1, t_upper + 1)}
    z_values = {bucket_idx: 0.0 for bucket_idx in range(t_lower + 1, t_upper + 1)}

    last_occupied_bucket = 0
    for (stage_idx, job_idx), (start_time, end_time) in ub_schedule.operation_intervals.items():
        start_bucket = start_time // delta + 1
        end_bucket = (end_time - 1) // delta + 1
        if end_bucket > t_upper:
            raise ValueError(
                f"Warm-start schedule for instance {instance.ins_name} reaches bucket "
                f"{end_bucket}, beyond T_U={t_upper}."
            )

        last_occupied_bucket = max(last_occupied_bucket, end_bucket)
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

    first_range_bucket = t_lower + 1
    upper_residual = input_ub - (t_upper - 1) * delta
    for bucket_idx in range(first_range_bucket, t_upper + 1):
        if bucket_idx <= last_occupied_bucket:
            u_values[bucket_idx] = 1
            if bucket_idx < last_occupied_bucket:
                z_values[bucket_idx] = float(delta)
            else:
                z_values[bucket_idx] = float(
                    ub_schedule.makespan - (bucket_idx - 1) * delta
                )

    if first_range_bucket in z_values:
        z_values[first_range_bucket] = max(
            z_values[first_range_bucket],
            float(input_lb - t_lower * delta),
        )
    if t_upper in z_values:
        z_values[t_upper] = min(z_values[t_upper], float(upper_residual))

    return BucketWarmStart(
        a_values=a_values,
        b_values=b_values,
        c_values=c_values,
        x_values=x_values,
        u_values=u_values,
        z_values=z_values,
    )


def apply_bucket_warm_start(
    model_vars: BucketModelVars,
    warm_start: BucketWarmStart,
) -> None:
    for key, value in warm_start.a_values.items():
        if key in model_vars.a:
            model_vars.a[key].Start = float(value)
    for key, value in warm_start.b_values.items():
        if key in model_vars.b:
            model_vars.b[key].Start = float(value)
    for key, value in warm_start.c_values.items():
        if key in model_vars.c:
            model_vars.c[key].Start = float(value)
    for key, value in warm_start.x_values.items():
        if key in model_vars.x:
            model_vars.x[key].Start = float(value)
    for key, value in warm_start.u_values.items():
        if key in model_vars.u:
            model_vars.u[key].Start = float(value)
    for key, value in warm_start.z_values.items():
        if key in model_vars.z:
            model_vars.z[key].Start = float(value)


def _parse_zero_based_suffix(label: Any, prefix: str) -> int:
    label_text = str(label)
    if not label_text.startswith(prefix):
        raise ValueError(
            f"Expected label starting with '{prefix}', but received {label_text!r}."
    )
    return int(label_text[len(prefix) :])


@lru_cache(maxsize=None)
def _load_solution_dict(solution_path: Path) -> dict:
    return load_yaml(solution_path, encoding="utf-8")


def _extract_time_map(
    solution_dict: dict,
    primary_key: str,
    fallback_key: str,
    solution_path: Path,
) -> dict:
    if primary_key in solution_dict:
        return solution_dict[primary_key]
    if fallback_key in solution_dict:
        return solution_dict[fallback_key]
    raise ValueError(
        f"Neither '{primary_key}' nor '{fallback_key}' found in solution file: {solution_path}"
    )


@lru_cache(maxsize=None)
def _build_solution_index(search_root: Path) -> dict[str, Path]:
    solution_index: dict[str, Path] = {}
    for solution_path in search_root.rglob("*_solution.yaml"):
        ins_name = solution_path.stem.removesuffix("_solution")
        solution_index.setdefault(ins_name, solution_path)
    return solution_index
