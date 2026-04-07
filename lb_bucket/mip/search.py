from __future__ import annotations

import math
import time
from typing import Any

from .data import compute_range_bucket_bounds
from .model import build_two_bucket_model
from .progress import GurobiProgressRecorder
from .solution_io import extract_solution_payload
from .shared import (
    BucketSearchResult,
    ModelStrengtheningOptions,
    PrecedenceOptions,
    ProgressTraceRow,
    SearchTraceRow,
    SummaryBoundRecord,
    TwoBucketInstance,
    gurobi_status_name,
    log_progress,
)
from .warm_start import (
    ParsedUbSchedule,
    apply_bucket_warm_start,
    build_bucket_warm_start,
)


def _safe_float_attr(model: Any, attr_name: str) -> float | None:
    try:
        value = getattr(model, attr_name)
    except (AttributeError, TypeError):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def run_bucket_search_for_instance(
    gp: Any,
    grb: Any,
    instance: TwoBucketInstance,
    record: SummaryBoundRecord,
    *,
    delta: int,
    threads: int | None,
    time_limit_sec: float | None,
    log_to_console: bool,
    display_interval_sec: int,
    log_dir,
    search_upper_t: int | None,
    strengthening: ModelStrengtheningOptions,
    precedence: PrecedenceOptions,
    time_limit_sec_used: float,
    ub_schedule: ParsedUbSchedule | None,
) -> tuple[
    BucketSearchResult,
    list[SearchTraceRow],
    list[ProgressTraceRow],
    dict[str, Any] | None,
]:
    if record.input_ub is None:
        raise ValueError(
            f"Instance {instance.ins_name} is missing input_ub/bestObj, which is required "
            "by the current range-based formulation."
        )

    t_lower, t_upper = compute_range_bucket_bounds(
        record.input_lb, record.input_ub, delta
    )
    if search_upper_t is not None and search_upper_t != t_upper:
        raise ValueError(
            f"The updated range-based model requires T_U=ceil(UB/delta)={t_upper}, "
            f"but received search_upper_t={search_upper_t}."
        )

    wall_start = time.perf_counter()
    range_base_time = float(t_lower * delta)
    log_progress(
        "Starting instance "
        f"{instance.ins_name}: jobs={instance.job_count}, stages={instance.stage_count}, "
        f"input_LB={record.input_lb}, input_UB={record.input_ub}, "
        f"T_L={t_lower}, T_U={t_upper}, objective_bucket_count={t_upper - t_lower}, "
        f"{strengthening.describe()}, "
        f"{precedence.describe()}, a/b type policy=all_binary"
    )

    model_build_wall_start = time.perf_counter()
    model, model_vars = build_two_bucket_model(
        gp,
        grb,
        instance,
        delta,
        record.input_lb,
        record.input_ub,
        strengthening,
        precedence,
    )
    model_build_wall_sec = time.perf_counter() - model_build_wall_start

    if ub_schedule is not None:
        if ub_schedule.makespan <= t_upper * delta:
            warm_start_wall_start = time.perf_counter()
            warm_start = build_bucket_warm_start(
                instance=instance,
                ub_schedule=ub_schedule,
                input_lb=record.input_lb,
                input_ub=record.input_ub,
                delta=delta,
            )
            if warm_start is not None:
                apply_bucket_warm_start(model_vars, warm_start)
                log_progress(
                    f"Instance {instance.ins_name}: applied UB warm start from "
                    f"{ub_schedule.solution_path.name if ub_schedule.solution_path else 'incumbent'} with makespan={ub_schedule.makespan}, "
                    f"warm_start_wall_sec={time.perf_counter() - warm_start_wall_start:.2f}"
                )
        else:
            log_progress(
                f"Instance {instance.ins_name}: UB warm start skipped because "
                f"makespan={ub_schedule.makespan} exceeds horizon={t_upper * delta}"
            )

    enable_solver_logging = log_to_console or log_dir is not None
    model.Params.OutputFlag = 1 if enable_solver_logging else 0
    model.Params.LogToConsole = 1 if log_to_console else 0
    model.Params.DisplayInterval = display_interval_sec
    model.Params.Method = 2
    model.Params.NodeMethod = 2
    model.Params.Crossover = 0
    model.Params.BarOrder = 0
    if threads is not None:
        model.Params.Threads = threads
    if time_limit_sec is not None:
        model.Params.TimeLimit = time_limit_sec
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{instance.ins_name}_range.log"
        model.Params.LogFile = str(log_path)

    progress_recorder = GurobiProgressRecorder(
        grb,
        instance.ins_name,
        range_base_time,
        hard_time_limit_sec=time_limit_sec,
    )
    model.optimize(progress_recorder.callback)

    status_name = gurobi_status_name(grb, model.Status)
    if model.Status == grb.INTERRUPTED and progress_recorder.terminated_by_time_guard:
        status_name = "TIME_LIMIT_GUARD"
    solution_count = int(model.SolCount)
    solver_runtime_sec = _safe_float_attr(model, "Runtime")
    if solver_runtime_sec is None:
        solver_runtime_sec = time.perf_counter() - wall_start
    objective_ub = _safe_float_attr(model, "ObjVal") if solution_count > 0 else None
    objective_lb = None
    if model.Status != grb.INFEASIBLE:
        objective_lb = _safe_float_attr(model, "ObjBound")
        if objective_lb is None:
            objective_lb = _safe_float_attr(model, "ObjBoundC")
    progress_recorder.append_final_row(
        runtime_sec=solver_runtime_sec,
        status=status_name,
        objective_ub=objective_ub,
        objective_lb=objective_lb,
        solution_count=solution_count,
    )

    trace_rows = [
        SearchTraceRow(
            ins_name=instance.ins_name,
            bucket_count=t_upper,
            status=status_name,
            runtime_sec=solver_runtime_sec,
            objective_ub=objective_ub,
            objective_lb=objective_lb,
            horizon_ub=range_base_time + objective_ub
            if objective_ub is not None
            else None,
            horizon_lb=range_base_time + objective_lb
            if objective_lb is not None
            else None,
            solution_count=solution_count,
        )
    ]
    log_progress(
        f"Instance {instance.ins_name}: finished range model with status={status_name}, "
        f"runtime_sec={trace_rows[-1].runtime_sec:.2f}, objective_ub={objective_ub}, "
        f"objective_lb={objective_lb}, solutions={solution_count}"
    )

    if model.Status == grb.OPTIMAL:
        z_star = float(model.ObjVal)
        bucket_lb = range_base_time + z_star
        horizon_ub = range_base_time + z_star
        total_runtime_sec = solver_runtime_sec
        result = BucketSearchResult(
            ins_name=instance.ins_name,
            input_lb=record.input_lb,
            input_ub=record.input_ub,
            delta=delta,
            job_count=instance.job_count,
            stage_count=instance.stage_count,
            machine_count_per_stage=" ".join(
                str(machine_count) for machine_count in instance.machine_count_per_stage
            ),
            t0=t_lower + 1,
            search_upper_t=t_upper,
            first_feasible_t=None,
            objective_ub=z_star,
            objective_lb=z_star,
            horizon_ub=horizon_ub,
            horizon_lb=bucket_lb,
            z_star=z_star,
            z_lower_bound_used=z_star,
            bucket_indexed_lb=bucket_lb,
            certified_final_lb=max(record.input_lb, bucket_lb),
            search_certified=True,
            searched_bucket_count=max(1, t_upper - t_lower),
            time_limit_sec_used=time_limit_sec_used,
            total_runtime_sec=total_runtime_sec,
            termination_reason="OPTIMAL_RANGE_MODEL",
        )
        log_progress(
            f"Completed instance {instance.ins_name}: W*={bucket_lb:.6f}, "
            f"certified_final_lb={result.certified_final_lb:.6f}, "
            f"total_runtime_sec={total_runtime_sec:.2f}, "
            f"wall_runtime_sec={time.perf_counter() - wall_start:.2f}, "
            f"model_build_wall_sec={model_build_wall_sec:.2f}"
        )
        solution_payload = extract_solution_payload(
            model_vars,
            instance=instance,
            result=result,
            trace_row=trace_rows[-1],
            t_lower=t_lower,
            t_upper=t_upper,
            precedence_formulation=precedence.formulation,
        )
        model.dispose()
        return result, trace_rows, progress_recorder.rows, solution_payload

    if model.Status == grb.INFEASIBLE:
        model.dispose()
        raise RuntimeError(
            f"Range model became infeasible for instance {instance.ins_name} with "
            f"LB={record.input_lb}, UB={record.input_ub}, delta={delta}. "
            "This suggests an inconsistency between the updated formulation and the "
            "provided upper bound, or a modeling bug."
        )

    total_runtime_sec = solver_runtime_sec
    bucket_lb = range_base_time + objective_lb if objective_lb is not None else None
    horizon_ub = range_base_time + objective_ub if objective_ub is not None else None
    fallback_result = BucketSearchResult(
        ins_name=instance.ins_name,
        input_lb=record.input_lb,
        input_ub=record.input_ub,
        delta=delta,
        job_count=instance.job_count,
        stage_count=instance.stage_count,
        machine_count_per_stage=" ".join(
            str(machine_count) for machine_count in instance.machine_count_per_stage
        ),
        t0=t_lower + 1,
        search_upper_t=t_upper,
        first_feasible_t=None,
        objective_ub=objective_ub,
        objective_lb=objective_lb,
        horizon_ub=horizon_ub,
        horizon_lb=bucket_lb,
        z_star=objective_ub,
        z_lower_bound_used=objective_lb,
        bucket_indexed_lb=bucket_lb,
        certified_final_lb=max(
            record.input_lb,
            bucket_lb if bucket_lb is not None else float(record.input_lb),
        ),
        search_certified=False,
        searched_bucket_count=max(1, t_upper - t_lower),
        time_limit_sec_used=time_limit_sec_used,
        total_runtime_sec=total_runtime_sec,
        termination_reason=(
            f"STOPPED_RANGE_MODEL_{status_name}_USING_OBJBND"
            if objective_lb is not None
            else f"STOPPED_RANGE_MODEL_{status_name}"
        ),
    )
    log_progress(
        f"Stopped instance {instance.ins_name}: status={status_name}, "
        f"bucket_lb={bucket_lb}, certified_final_lb={fallback_result.certified_final_lb:.6f}, "
        f"total_runtime_sec={total_runtime_sec:.2f}, "
        f"wall_runtime_sec={time.perf_counter() - wall_start:.2f}, "
        f"model_build_wall_sec={model_build_wall_sec:.2f}"
    )
    solution_payload = extract_solution_payload(
        model_vars,
        instance=instance,
        result=fallback_result,
        trace_row=trace_rows[-1],
        t_lower=t_lower,
        t_upper=t_upper,
        precedence_formulation=precedence.formulation,
    )
    model.dispose()
    return fallback_result, trace_rows, progress_recorder.rows, solution_payload
