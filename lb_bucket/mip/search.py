from __future__ import annotations

import time
from typing import Any

from .data import compute_initial_bucket_count
from .model import build_two_bucket_model
from .shared import (
    BucketSearchResult,
    ModelStrengtheningOptions,
    SearchTraceRow,
    SummaryBoundRecord,
    TwoBucketInstance,
    gurobi_status_name,
    log_progress,
)


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
) -> tuple[BucketSearchResult, list[SearchTraceRow]]:
    t0 = compute_initial_bucket_count(instance, record.input_lb, delta)
    if search_upper_t is not None and search_upper_t < t0:
        raise ValueError(
            f"Search cap {search_upper_t} is smaller than T0={t0} for instance {instance.ins_name}."
        )

    trace_rows: list[SearchTraceRow] = []
    total_start = time.perf_counter()

    if search_upper_t is None:
        raise ValueError(
            f"No finite search upper bound is available for instance {instance.ins_name}. "
            "Provide bestObj in the summary or pass --max-bucket-count."
        )

    log_progress(
        "Starting instance "
        f"{instance.ins_name}: jobs={instance.job_count}, stages={instance.stage_count}, "
        f"input_LB={record.input_lb}, input_UB={record.input_ub}, "
        f"start_bucket_count(T0)={t0}, search_upper_T={search_upper_t}, "
        f"{strengthening.describe()}"
    )

    for bucket_count in range(t0, search_upper_t + 1):
        iter_start = time.perf_counter()
        log_progress(
            f"Instance {instance.ins_name}: solving T={bucket_count} "
            f"({bucket_count - t0 + 1}/{search_upper_t - t0 + 1})"
        )
        model, _ = build_two_bucket_model(
            gp, grb, instance, bucket_count, delta, strengthening
        )

        enable_solver_logging = log_to_console or log_dir is not None
        model.Params.OutputFlag = 1 if enable_solver_logging else 0
        model.Params.LogToConsole = 1 if log_to_console else 0
        model.Params.DisplayInterval = display_interval_sec
        if threads is not None:
            model.Params.Threads = threads
        if time_limit_sec is not None:
            model.Params.TimeLimit = time_limit_sec
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{instance.ins_name}_T{bucket_count}.log"
            model.Params.LogFile = str(log_path)

        model.optimize()

        status_name = gurobi_status_name(grb, model.Status)
        solution_count = int(model.SolCount)
        objective_value = float(model.ObjVal) if solution_count > 0 else None
        trace_rows.append(
            SearchTraceRow(
                ins_name=instance.ins_name,
                bucket_count=bucket_count,
                status=status_name,
                runtime_sec=time.perf_counter() - iter_start,
                objective_value=objective_value,
                solution_count=solution_count,
            )
        )
        log_progress(
            f"Instance {instance.ins_name}: finished T={bucket_count} with "
            f"status={status_name}, runtime_sec={trace_rows[-1].runtime_sec:.2f}, "
            f"objective={objective_value}, solutions={solution_count}"
        )

        if model.Status == grb.INFEASIBLE:
            model.dispose()
            continue

        if model.Status == grb.OPTIMAL:
            z_star = float(model.ObjVal)
            bucket_lb = (bucket_count - 1) * delta + z_star
            total_runtime_sec = time.perf_counter() - total_start
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
                t0=t0,
                search_upper_t=search_upper_t,
                first_feasible_t=bucket_count,
                z_star=z_star,
                bucket_indexed_lb=bucket_lb,
                certified_final_lb=max(record.input_lb, bucket_lb),
                search_certified=True,
                searched_bucket_count=bucket_count - t0 + 1,
                total_runtime_sec=total_runtime_sec,
                termination_reason="OPTIMAL_AT_FIRST_FEASIBLE_T",
            )
            log_progress(
                f"Completed instance {instance.ins_name}: first_feasible_T={bucket_count}, "
                f"z_star={z_star:.6f}, bucket_lb={bucket_lb:.6f}, "
                f"certified_final_lb={result.certified_final_lb:.6f}, "
                f"total_runtime_sec={total_runtime_sec:.2f}"
            )
            model.dispose()
            return result, trace_rows

        total_runtime_sec = time.perf_counter() - total_start
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
            t0=t0,
            search_upper_t=search_upper_t,
            first_feasible_t=None,
            z_star=None,
            bucket_indexed_lb=None,
            certified_final_lb=float(record.input_lb),
            search_certified=False,
            searched_bucket_count=bucket_count - t0 + 1,
            total_runtime_sec=total_runtime_sec,
            termination_reason=f"STOPPED_AT_T_{bucket_count}_{status_name}",
        )
        log_progress(
            f"Stopped instance {instance.ins_name}: status={status_name} at T={bucket_count}, "
            f"falling back to input_LB={record.input_lb}, "
            f"total_runtime_sec={total_runtime_sec:.2f}"
        )
        model.dispose()
        return fallback_result, trace_rows

    total_runtime_sec = time.perf_counter() - total_start
    final_result = BucketSearchResult(
        ins_name=instance.ins_name,
        input_lb=record.input_lb,
        input_ub=record.input_ub,
        delta=delta,
        job_count=instance.job_count,
        stage_count=instance.stage_count,
        machine_count_per_stage=" ".join(
            str(machine_count) for machine_count in instance.machine_count_per_stage
        ),
        t0=t0,
        search_upper_t=search_upper_t,
        first_feasible_t=None,
        z_star=None,
        bucket_indexed_lb=None,
        certified_final_lb=float(record.input_lb),
        search_certified=False,
        searched_bucket_count=max(0, search_upper_t - t0 + 1),
        total_runtime_sec=total_runtime_sec,
        termination_reason="NO_FEASIBLE_BUCKET_COUNT_FOUND_WITHIN_CAP",
    )
    log_progress(
        f"Completed instance {instance.ins_name}: no feasible T found up to {search_upper_t}, "
        f"falling back to input_LB={record.input_lb}, total_runtime_sec={total_runtime_sec:.2f}"
    )
    return final_result, trace_rows
