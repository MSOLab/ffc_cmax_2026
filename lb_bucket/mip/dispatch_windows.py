from __future__ import annotations

from typing import Any

from .shared import TwoBucketInstance


def compute_dispatch_window_payload(
    instance: TwoBucketInstance,
    solution_payload: dict[str, Any],
    *,
    cmax: float | None = None,
    value_tolerance: float | None = None,
) -> dict[str, Any]:
    """
    Build per-operation dispatch-window inputs and ES/LS values from an MIP solution.

    The handwritten note provided by the user was partially ambiguous in the
    same-bucket terms, so this implementation uses the dimensionally consistent
    interpretation:

    - same-bucket start candidate: `(A_ij - 1) * delta`
    - two-bucket start candidate: `(B_ij - 1) * delta + x_ij,B_ij - p_ij`
    - same-bucket completion candidate: `B_ij * delta`
    - two-bucket completion candidate: `(B_ij - 1) * delta + x_ij,B_ij`

    The forward/backward passes then propagate precedence inside each job by
    taking max/min with the running values.
    """
    metadata = solution_payload["metadata"]
    tol = (
        float(value_tolerance)
        if value_tolerance is not None
        else float(metadata.get("value_tolerance", 1e-9))
    )
    delta = int(metadata["delta"])
    if cmax is not None:
        dispatch_cmax = float(cmax)
    elif metadata.get("horizon_ub") is not None:
        dispatch_cmax = float(metadata["horizon_ub"])
    elif (
        metadata.get("objective_ub") is not None and metadata.get("t_lower") is not None
    ):
        dispatch_cmax = float(metadata["t_lower"]) * float(delta) + float(
            metadata["objective_ub"]
        )
    elif metadata.get("input_ub") is not None:
        dispatch_cmax = float(metadata["input_ub"])
    else:
        raise ValueError(
            "Dispatch-window computation requires cmax, horizon_ub, objective_ub+t_lower, or input_ub."
        )

    a_bucket_map = _build_weighted_bucket_map(solution_payload["a"])
    b_bucket_map = _build_weighted_bucket_map(solution_payload["b"])
    x_bucket_map = _build_x_bucket_map(solution_payload["x"])

    op_inputs_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for job_idx in range(1, instance.job_count + 1):
        for stage_idx in range(1, instance.stage_count + 1):
            op_key = (stage_idx, job_idx)
            a_bucket = _resolve_integral_bucket(
                a_bucket_map.get(op_key),
                op_key,
                "a",
                tol,
            )
            b_bucket = _resolve_integral_bucket(
                b_bucket_map.get(op_key),
                op_key,
                "b",
                tol,
            )
            processing_time = instance.processing_times_by_stage[stage_idx - 1][
                job_idx - 1
            ]
            x_at_a_bucket = float(x_bucket_map.get((stage_idx, job_idx, a_bucket), 0.0))
            x_at_b_bucket = float(x_bucket_map.get((stage_idx, job_idx, b_bucket), 0.0))
            spans_two_buckets = a_bucket != b_bucket
            x_bucket_1 = a_bucket
            x_value_1 = x_at_a_bucket
            x_bucket_2 = b_bucket if spans_two_buckets else None
            x_value_2 = x_at_b_bucket if spans_two_buckets else None

            op_inputs_by_key[op_key] = {
                "stage": stage_idx,
                "job": job_idx,
                "processing_time": processing_time,
                "a_bucket": a_bucket,
                "b_bucket": b_bucket,
                "x_bucket_1": x_bucket_1,
                "x_value_1": x_value_1,
                "x_bucket_2": x_bucket_2,
                "x_value_2": x_value_2,
                "x_at_a_bucket": x_at_a_bucket,
                "x_at_b_bucket": x_at_b_bucket,
                "spans_two_buckets": spans_two_buckets,
            }

    dispatch_window_rows = _compute_dispatch_windows(
        instance=instance,
        delta=delta,
        dispatch_cmax=dispatch_cmax,
        op_inputs_by_key=op_inputs_by_key,
    )

    return {
        "metadata": {
            "dispatch_cmax": dispatch_cmax,
            "dispatch_window_algorithm": "handwritten_interpreted_v1",
            "dispatch_window_note": (
                "Uses A_ij, B_ij, and x_ij,B_ij derived from the final MIP solution. "
                "dispatch_cmax is taken from the final MIP horizon value when available. "
                "Same-bucket terms are interpreted as (A_ij-1)*delta and B_ij*delta."
            ),
        },
        "dispatch_window_inputs": [
            op_inputs_by_key[stage_idx, job_idx]
            for job_idx in range(1, instance.job_count + 1)
            for stage_idx in range(1, instance.stage_count + 1)
        ],
        "dispatch_windows": dispatch_window_rows,
    }


def build_dispatch_window_lookup(
    dispatch_window_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["stage"]), int(row["job"])): dict(row) for row in dispatch_window_rows
    }


def _build_weighted_bucket_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int], float]:
    weighted_map: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["stage"]), int(row["job"]))
        weighted_map[key] = weighted_map.get(key, 0.0) + float(row["bucket"]) * float(
            row["value"]
        )
    return weighted_map


def _build_x_bucket_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int, int], float]:
    x_map: dict[tuple[int, int, int], float] = {}
    for row in rows:
        key = (int(row["stage"]), int(row["job"]), int(row["bucket"]))
        x_map[key] = float(row["value"])
    return x_map


def _resolve_integral_bucket(
    weighted_value: float | None,
    op_key: tuple[int, int],
    variable_name: str,
    value_tolerance: float,
) -> int:
    if weighted_value is None:
        raise ValueError(
            f"Operation {op_key} is missing a positive {variable_name}-bucket value."
        )
    rounded = round(weighted_value)
    if abs(weighted_value - rounded) > value_tolerance:
        raise ValueError(
            f"Operation {op_key} has non-integral {variable_name}-bucket weighted sum "
            f"{weighted_value:.9f}."
        )
    return int(rounded)


def _compute_dispatch_windows(
    *,
    instance: TwoBucketInstance,
    delta: int,
    dispatch_cmax: float,
    op_inputs_by_key: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    es_by_key: dict[tuple[int, int], float] = {}
    ls_by_key: dict[tuple[int, int], float] = {}
    es_candidate_by_key: dict[tuple[int, int], float] = {}
    completion_candidate_by_key: dict[tuple[int, int], float] = {}

    for job_idx in range(1, instance.job_count + 1):
        temp_start = 0.0
        for stage_idx in range(1, instance.stage_count + 1):
            op_key = (stage_idx, job_idx)
            op_input = op_inputs_by_key[op_key]
            start_candidate = _get_start_candidate(op_input, delta)
            es_candidate_by_key[op_key] = start_candidate
            temp_start = max(temp_start, start_candidate)
            es_by_key[op_key] = temp_start
            temp_start = es_by_key[op_key] + float(op_input["processing_time"])

        temp_completion = dispatch_cmax
        for stage_idx in range(instance.stage_count, 0, -1):
            op_key = (stage_idx, job_idx)
            op_input = op_inputs_by_key[op_key]
            completion_candidate = _get_completion_candidate(op_input, delta)
            completion_candidate_by_key[op_key] = completion_candidate
            temp_completion = min(temp_completion, completion_candidate)
            ls_by_key[op_key] = temp_completion - float(op_input["processing_time"])
            temp_completion = ls_by_key[op_key]

    rows: list[dict[str, Any]] = []
    for job_idx in range(1, instance.job_count + 1):
        for stage_idx in range(1, instance.stage_count + 1):
            op_key = (stage_idx, job_idx)
            op_input = op_inputs_by_key[op_key]
            early_start = es_by_key[op_key]
            late_start = ls_by_key[op_key]
            rows.append(
                {
                    **op_input,
                    "es_candidate": es_candidate_by_key[op_key],
                    "completion_candidate": completion_candidate_by_key[op_key],
                    "early_start": early_start,
                    "late_start": late_start,
                    "slack": late_start - early_start,
                }
            )
    return rows


def _get_start_candidate(op_input: dict[str, Any], delta: int) -> float:
    if not op_input["spans_two_buckets"]:
        return float((int(op_input["a_bucket"]) - 1) * delta)
    return (
        float((int(op_input["b_bucket"]) - 1) * delta)
        + float(op_input["x_at_b_bucket"])
        - float(op_input["processing_time"])
    )


def _get_completion_candidate(op_input: dict[str, Any], delta: int) -> float:
    if not op_input["spans_two_buckets"]:
        return float(int(op_input["b_bucket"]) * delta)
    return float((int(op_input["b_bucket"]) - 1) * delta) + float(
        op_input["x_at_b_bucket"]
    )
