from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from mbls.cpsat import CpsatStatus

from hybridflowshop.schedule_lite import StageIdType


def sanitize_optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


@dataclass(frozen=True)
class RetainedStageCpResult:
    ins_name: str
    input_lb: float | None
    input_ub: float
    retained_stage_mode: str
    retained_stage_ids: tuple[str, ...]
    bottleneck_stage_id: str | None
    selected_bottleneck_stage_ids: tuple[str, ...]
    bottleneck_band_radius: int | None
    retained_stage_ratios: tuple[float, ...]
    quantile_count: int | None
    job_count: int
    stage_count: int
    machine_count_per_stage: str
    objective_ub: float | None
    objective_lb: float | None
    certified_final_lb: float | None
    status_name: str
    time_limit_sec_used: float | None
    solver_runtime_sec: float | None
    wall_runtime_sec: float | None
    model_build_wall_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["retained_stage_ids"] = list(self.retained_stage_ids)
        data["selected_bottleneck_stage_ids"] = list(self.selected_bottleneck_stage_ids)
        data["retained_stage_ratios"] = list(self.retained_stage_ratios)
        return data


def build_trace_rows(
    obj_value_records: Sequence[tuple[float, float]] | None,
    obj_bound_records: Sequence[tuple[float, float]] | None,
    *,
    final_runtime_sec: float | None,
    final_objective_ub: float | None,
    final_objective_lb: float | None,
) -> list[dict[str, float | None]]:
    timestamp_map: dict[float, dict[str, float | None]] = {}

    for runtime_sec, objective_ub in obj_value_records or ():
        runtime = sanitize_optional_float(runtime_sec)
        obj = sanitize_optional_float(objective_ub)
        if runtime is None:
            continue
        row = timestamp_map.setdefault(
            runtime,
            {"runtime_sec": runtime, "objective_ub": None, "objective_lb": None},
        )
        row["objective_ub"] = obj

    for runtime_sec, objective_lb in obj_bound_records or ():
        runtime = sanitize_optional_float(runtime_sec)
        bound = sanitize_optional_float(objective_lb)
        if runtime is None:
            continue
        row = timestamp_map.setdefault(
            runtime,
            {"runtime_sec": runtime, "objective_ub": None, "objective_lb": None},
        )
        row["objective_lb"] = bound

    final_runtime = sanitize_optional_float(final_runtime_sec)
    if final_runtime is not None:
        row = timestamp_map.setdefault(
            final_runtime,
            {
                "runtime_sec": final_runtime,
                "objective_ub": None,
                "objective_lb": None,
            },
        )
        if row["objective_ub"] is None:
            row["objective_ub"] = sanitize_optional_float(final_objective_ub)
        if row["objective_lb"] is None:
            row["objective_lb"] = sanitize_optional_float(final_objective_lb)

    return [timestamp_map[t] for t in sorted(timestamp_map)]


def build_retained_stage_cp_result(
    *,
    ins_name: str,
    input_lb: float | None,
    input_ub: float,
    retained_stage_mode: str,
    retained_stage_ids: Sequence[StageIdType],
    bottleneck_stage_id: str | None,
    selected_bottleneck_stage_ids: Sequence[StageIdType] | None,
    bottleneck_band_radius: int | None,
    retained_stage_ratios: Sequence[float] | None,
    quantile_count: int | None,
    job_count: int,
    stage_count: int,
    machine_count_per_stage: Sequence[int],
    status: CpsatStatus,
    objective_ub: float | None,
    objective_lb: float | None,
    time_limit_sec_used: float | None,
    solver_runtime_sec: float | None,
    wall_runtime_sec: float | None,
    model_build_wall_sec: float | None,
) -> RetainedStageCpResult:
    return RetainedStageCpResult(
        ins_name=ins_name,
        input_lb=sanitize_optional_float(input_lb),
        input_ub=float(input_ub),
        retained_stage_mode=retained_stage_mode,
        retained_stage_ids=tuple(str(stage_id) for stage_id in retained_stage_ids),
        bottleneck_stage_id=bottleneck_stage_id,
        selected_bottleneck_stage_ids=tuple(
            str(stage_id) for stage_id in (selected_bottleneck_stage_ids or ())
        ),
        bottleneck_band_radius=bottleneck_band_radius,
        retained_stage_ratios=tuple(
            float(ratio) for ratio in (retained_stage_ratios or ())
        ),
        quantile_count=quantile_count,
        job_count=job_count,
        stage_count=stage_count,
        machine_count_per_stage=" ".join(str(v) for v in machine_count_per_stage),
        objective_ub=sanitize_optional_float(objective_ub),
        objective_lb=sanitize_optional_float(objective_lb),
        certified_final_lb=sanitize_optional_float(objective_lb),
        status_name=status.to_solver_status_enum().value,
        time_limit_sec_used=sanitize_optional_float(time_limit_sec_used),
        solver_runtime_sec=sanitize_optional_float(solver_runtime_sec),
        wall_runtime_sec=sanitize_optional_float(wall_runtime_sec),
        model_build_wall_sec=sanitize_optional_float(model_build_wall_sec),
    )
