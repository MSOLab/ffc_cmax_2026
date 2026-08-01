from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .cli import parse_args, resolve_precedence_options, resolve_strengthening_options
from .data import (
    compute_range_bucket_bounds,
    get_max_processing_time,
    load_ff2020_instance,
    load_summary_records,
    resolve_time_limit_sec,
    select_summary_records,
)
from .search import run_bucket_search_for_instance, trace_rows_to_csv_rows
from .shared import import_gurobi, log_progress
from .solution_io import write_solution_payload
from .warm_start import load_ub_schedule, resolve_solution_path


def write_csv_rows(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_completed_instance_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return set()
        key = "ins_name" if "ins_name" in reader.fieldnames else "insName"
        if key not in reader.fieldnames:
            return set()
        completed: set[str] = set()
        for row in reader:
            ins_name = str(row.get(key, "")).strip()
            if ins_name:
                completed.add(ins_name)
        return completed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.delta is not None and args.delta <= 0:
        raise ValueError(
            f"delta must be a positive integer. Received delta={args.delta}."
        )
    if args.delta is not None and args.delta_pmax_plus_one:
        raise ValueError("Use either --delta or --delta-pmax-plus-one, not both.")

    gp, grb = import_gurobi()
    summary_records = load_summary_records(args.summary_csv)
    selected_records = select_summary_records(summary_records, args.instances)
    strengthening = resolve_strengthening_options(args)
    precedence = resolve_precedence_options(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_progress(
        f"Loaded {len(selected_records)} instance(s) from {args.summary_csv}. "
        f"Output directory: {args.output_dir}"
    )
    log_progress(f"Model strengthening options: {strengthening.describe()}")
    log_progress(f"Precedence options: {precedence.describe()}")

    result_path = args.output_dir / "bucket_lb_results.csv"
    completed_instance_names: set[str] = set()
    if args.resume:
        completed_instance_names = load_completed_instance_names(result_path)
        if completed_instance_names:
            log_progress(
                f"Resume mode: found {len(completed_instance_names)} completed instance(s) "
                f"in {result_path}"
            )
    elif result_path.exists():
        result_path.unlink()
        log_progress(f"Removed previous result file: {result_path}")

    pending_records = [
        record
        for record in selected_records
        if record.ins_name not in completed_instance_names
    ]
    if args.resume and completed_instance_names:
        skipped_count = len(selected_records) - len(pending_records)
        log_progress(f"Resume mode: skipping {skipped_count} completed instance(s)")
    if not pending_records:
        log_progress("No pending instances remain. Nothing to do.")
        return

    for record in pending_records:
        instance = load_ff2020_instance(args.input_dir, record.ins_name)
        ub_schedule = None
        if not args.disable_ub_warm_start:
            warm_start_load_start = time.perf_counter()
            solution_path = resolve_solution_path(
                args.summary_csv, record.ins_name, args.solution_root
            )
            if solution_path is not None:
                ub_schedule = load_ub_schedule(instance, solution_path)
                log_progress(
                    f"Loaded UB warm start for instance {record.ins_name}: "
                    f"makespan={ub_schedule.makespan}, path={solution_path}, "
                    f"load_wall_sec={time.perf_counter() - warm_start_load_start:.2f}"
                )
            else:
                log_progress(
                    f"No UB warm start solution found for instance {record.ins_name}; "
                    "continuing without warm start."
                )
        max_processing_time = get_max_processing_time(instance)
        if args.delta_pmax_plus_one:
            configured_bucket_count = None
            instance_delta = max_processing_time + 1
        elif args.delta is None:
            configured_bucket_count = None
            instance_delta = max_processing_time
        else:
            configured_bucket_count = None
            instance_delta = args.delta

        if instance_delta < max_processing_time:
            raise ValueError(
                f"Instance {record.ins_name} has max processing time {max_processing_time}, "
                f"but this two-bucket implementation requires delta >= max p_ij. "
                f"Received delta={instance_delta}."
            )
        if record.input_ub is None:
            raise ValueError(
                f"Instance {record.ins_name} is missing bestObj/input UB, which is required "
                "by the current range-based formulation."
            )
        _t_lower, natural_t_upper = compute_range_bucket_bounds(
            record.input_lb, record.input_ub, instance_delta
        )
        if (
            args.max_bucket_count is not None
            and args.max_bucket_count != natural_t_upper
        ):
            raise ValueError(
                f"The updated range-based formulation requires T_U=ceil(UB/delta)="
                f"{natural_t_upper}, so --max-bucket-count={args.max_bucket_count} "
                "is not compatible."
            )
        search_upper_t = natural_t_upper
        if record.job_count is not None and record.job_count != instance.job_count:
            raise ValueError(
                f"Summary jobCount={record.job_count} but instance {record.ins_name} "
                f"has job_count={instance.job_count}."
            )
        if (
            record.stage_count is not None
            and record.stage_count != instance.stage_count
        ):
            raise ValueError(
                f"Summary stageCount={record.stage_count} but instance {record.ins_name} "
                f"has stage_count={instance.stage_count}."
            )

        instance_time_limit_sec = resolve_time_limit_sec(instance, args.time_limit_sec)
        log_progress(
            f"Instance {record.ins_name} configuration: delta={instance_delta}, "
            f"search_upper_T={search_upper_t}, auto_bucket_count={configured_bucket_count}, "
            f"delta_pmax_plus_one={args.delta_pmax_plus_one}, "
            f"time_limit_sec={instance_time_limit_sec:.2f}, "
            f"{precedence.describe()}"
        )
        (
            result,
            trace_rows,
            progress_rows,
            solution_payload,
            status_name,
            solution_count,
        ) = run_bucket_search_for_instance(
            gp,
            grb,
            instance,
            record,
            delta=instance_delta,
            threads=args.threads,
            time_limit_sec=instance_time_limit_sec,
            log_to_console=args.log_to_console,
            display_interval_sec=args.display_interval_sec,
            log_dir=args.log_dir,
            search_upper_t=search_upper_t,
            strengthening=strengthening,
            precedence=precedence,
            time_limit_sec_used=instance_time_limit_sec,
            ub_schedule=ub_schedule,
        )
        append_csv_row(result_path, asdict(result))
        log_progress(
            f"Appended result row for instance {record.ins_name} to {result_path}"
        )

        if solution_payload is not None:
            write_solution_payload(args.output_dir, solution_payload)
            log_progress(
                f"Wrote solution files for instance {record.ins_name} to "
                f"{args.output_dir / 'solutions' / record.ins_name}"
            )
        else:
            log_progress(
                f"No solution payload was available for instance {record.ins_name}; "
                "skipped solution export."
            )

        trace_path = args.output_dir / f"{record.ins_name}_search_trace.csv"
        trace_csv_rows = trace_rows_to_csv_rows(
            trace_rows, result, status_name, solution_count
        )
        write_csv_rows(trace_path, trace_csv_rows)
        log_progress(f"Wrote trace CSV for instance {record.ins_name} to {trace_path}")

        progress_path = args.output_dir / f"{record.ins_name}_progress_trace.csv"
        write_csv_rows(
            progress_path, [asdict(progress_row) for progress_row in progress_rows]
        )
        log_progress(
            f"Wrote progress CSV for instance {record.ins_name} to {progress_path}"
        )

    log_progress(f"Finished all instances. Result CSV: {result_path}")
