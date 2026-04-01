from __future__ import annotations

import csv
from dataclasses import asdict
from typing import Any

from .cli import parse_args, resolve_strengthening_options
from .data import (
    get_max_processing_time,
    load_ff2020_instance,
    load_summary_records,
    resolve_search_upper_t,
    select_summary_records,
)
from .search import run_bucket_search_for_instance
from .shared import import_gurobi, log_progress


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


def main() -> None:
    args = parse_args()
    if args.delta <= 0:
        raise ValueError(f"delta must be a positive integer. Received delta={args.delta}.")

    gp, grb = import_gurobi()
    summary_records = load_summary_records(args.summary_csv)
    selected_records = select_summary_records(summary_records, args.instances)
    strengthening = resolve_strengthening_options(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_progress(
        f"Loaded {len(selected_records)} instance(s) from {args.summary_csv}. "
        f"Output directory: {args.output_dir}"
    )
    log_progress(f"Model strengthening options: {strengthening.describe()}")

    result_path = args.output_dir / "bucket_lb_results.csv"
    if result_path.exists():
        result_path.unlink()
        log_progress(f"Removed previous result file: {result_path}")

    for record in selected_records:
        instance = load_ff2020_instance(args.input_dir, record.ins_name)
        max_processing_time = get_max_processing_time(instance)
        if args.delta <= max_processing_time:
            raise ValueError(
                f"Instance {record.ins_name} has max processing time {max_processing_time}, "
                f"but this two-bucket implementation requires delta > max p_ij. "
                f"Received delta={args.delta}."
            )
        if record.job_count is not None and record.job_count != instance.job_count:
            raise ValueError(
                f"Summary jobCount={record.job_count} but instance {record.ins_name} "
                f"has job_count={instance.job_count}."
            )
        if record.stage_count is not None and record.stage_count != instance.stage_count:
            raise ValueError(
                f"Summary stageCount={record.stage_count} but instance {record.ins_name} "
                f"has stage_count={instance.stage_count}."
            )

        search_upper_t = resolve_search_upper_t(record, args.delta, args.max_bucket_count)
        result, trace_rows = run_bucket_search_for_instance(
            gp,
            grb,
            instance,
            record,
            delta=args.delta,
            threads=args.threads,
            time_limit_sec=args.time_limit_sec,
            log_to_console=args.log_to_console,
            display_interval_sec=args.display_interval_sec,
            log_dir=args.log_dir,
            search_upper_t=search_upper_t,
            strengthening=strengthening,
        )
        append_csv_row(result_path, asdict(result))
        log_progress(f"Appended result row for instance {record.ins_name} to {result_path}")

        trace_path = args.output_dir / f"{record.ins_name}_search_trace.csv"
        write_csv_rows(trace_path, [asdict(trace_row) for trace_row in trace_rows])
        log_progress(f"Wrote trace CSV for instance {record.ins_name} to {trace_path}")

    log_progress(f"Finished all instances. Result CSV: {result_path}")
