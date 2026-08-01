from __future__ import annotations

import csv
import math
from pathlib import Path

from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from .shared import SummaryBoundRecord, TwoBucketInstance, parse_optional_int


def load_summary_records(summary_csv: Path) -> list[SummaryBoundRecord]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"insName", "bestBound"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Summary CSV {summary_csv} is missing required columns: {sorted(missing)}"
            )

        records: list[SummaryBoundRecord] = []
        for row in reader:
            ins_name = str(parse_optional_int(row.get("insName")) or "").strip()
            if not ins_name:
                continue
            input_lb = parse_optional_int(row.get("bestBound"))
            if input_lb is None:
                raise ValueError(
                    f"Row for instance {ins_name} in {summary_csv} has empty bestBound."
                )
            input_ub = parse_optional_int(row.get("bestObj"))
            if input_ub is not None and input_lb > input_ub:
                raise ValueError(
                    f"Row for instance {ins_name} in {summary_csv} has bestBound={input_lb} "
                    f"greater than bestObj={input_ub}."
                )
            records.append(
                SummaryBoundRecord(
                    ins_name=ins_name,
                    input_lb=input_lb,
                    input_ub=input_ub,
                    job_count=parse_optional_int(row.get("jobCount")),
                    stage_count=parse_optional_int(row.get("stageCount")),
                )
            )
    return records


def select_summary_records(
    all_records: list[SummaryBoundRecord], selected_instances: list[int] | None
) -> list[SummaryBoundRecord]:
    record_by_name = {record.ins_name: record for record in all_records}
    if selected_instances:
        selected: list[SummaryBoundRecord] = []
        for instance_id in selected_instances:
            key = str(instance_id)
            if key not in record_by_name:
                raise KeyError(f"Instance {instance_id} not found in the summary CSV.")
            selected.append(record_by_name[key])
        return selected
    return sorted(all_records, key=lambda record: int(record.ins_name))


def load_ff2020_instance(input_dir: Path, ins_name: str) -> TwoBucketInstance:
    instance_path = input_dir / f"{ins_name}.txt"
    if not instance_path.is_file():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")

    with instance_path.open("r", encoding="utf-8") as handle:
        params = HybridFlowshopParameters.from_ff2020_data(ins_name, handle)

    stage_ids = params.stage_id_list
    job_ids = params.job_id_list
    p_map = params.stage_2_job_2_p_map

    processing_times_by_stage = [
        [int(p_map[stage_id][job_id]) for job_id in job_ids] for stage_id in stage_ids
    ]
    machine_count_per_stage = [
        len(params.stage_2_machines_map[stage_id]) for stage_id in stage_ids
    ]

    return TwoBucketInstance(
        ins_name=ins_name,
        job_count=params.job_count,
        stage_count=params.stage_count,
        machine_count_per_stage=machine_count_per_stage,
        processing_times_by_stage=processing_times_by_stage,
        stage_ids=list(stage_ids),
        job_ids=list(job_ids),
    )


def compute_initial_bucket_count(
    instance: TwoBucketInstance, input_lb: int, delta: int
) -> int:
    stage_lower_bound = max(
        math.ceil(sum(stage_processing_times) / (machine_count * delta))
        for stage_processing_times, machine_count in zip(
            instance.processing_times_by_stage, instance.machine_count_per_stage
        )
    )
    job_lower_bound = max(
        math.ceil(
            sum(
                instance.processing_times_by_stage[stage_idx][job_idx]
                for stage_idx in range(instance.stage_count)
            )
            / delta
        )
        for job_idx in range(instance.job_count)
    )
    return max(1, math.ceil(input_lb / delta), stage_lower_bound, job_lower_bound)


def compute_range_bucket_bounds(
    input_lb: int,
    input_ub: int,
    delta: int,
) -> tuple[int, int]:
    if delta <= 0:
        raise ValueError(f"delta must be positive. Received delta={delta}.")
    if input_lb <= 0:
        raise ValueError(f"input_lb must be positive. Received input_lb={input_lb}.")
    if input_ub < input_lb:
        raise ValueError(
            f"input_ub must be at least input_lb. Received input_lb={input_lb}, "
            f"input_ub={input_ub}."
        )
    t_lower = max(0, math.ceil(input_lb / delta) - 1)
    t_upper = math.ceil(input_ub / delta)
    if t_upper < t_lower + 1:
        t_upper = t_lower + 1
    return t_lower, t_upper


def resolve_search_upper_t(
    record: SummaryBoundRecord,
    delta: int,
    cli_cap: int | None,
) -> int | None:
    ub_cap = math.ceil(record.input_ub / delta) if record.input_ub is not None else None
    if cli_cap is not None and ub_cap is not None:
        return min(cli_cap, ub_cap)
    if cli_cap is not None:
        return cli_cap
    return ub_cap


def _share_last_bucket_with_real_partition(lb: int, ub: int, bucket_count: int) -> bool:
    return lb * bucket_count > ub * (bucket_count - 1)


def _share_same_integer_bucket(lb: int, ub: int, delta: int) -> bool:
    return math.ceil(lb / delta) == math.ceil(ub / delta)


def compute_auto_bucket_configuration(
    record: SummaryBoundRecord,
    same_bucket_threshold: int,
) -> tuple[int, int]:
    if record.input_ub is None:
        raise ValueError(
            f"Instance {record.ins_name} is missing bestObj/input UB, which is required "
            "for automatic bucket-size selection."
        )
    if same_bucket_threshold < 1:
        raise ValueError(
            f"same_bucket_threshold must be at least 1. Received {same_bucket_threshold}."
        )

    chosen_bucket_count = same_bucket_threshold
    for bucket_count in range(1, same_bucket_threshold + 1):
        if not _share_last_bucket_with_real_partition(
            record.input_lb, record.input_ub, bucket_count
        ):
            chosen_bucket_count = bucket_count - 1
            break

    chosen_bucket_count = max(1, chosen_bucket_count)
    delta = math.ceil(record.input_ub / chosen_bucket_count)

    while chosen_bucket_count > 1 and not _share_same_integer_bucket(
        record.input_lb, record.input_ub, delta
    ):
        chosen_bucket_count -= 1
        delta = math.ceil(record.input_ub / chosen_bucket_count)

    return chosen_bucket_count, delta


def resolve_time_limit_sec(
    instance: TwoBucketInstance, cli_time_limit_sec: float | None
) -> float:
    if cli_time_limit_sec is not None:
        return cli_time_limit_sec
    return instance.stage_count * instance.job_count * 0.1


def get_max_processing_time(instance: TwoBucketInstance) -> int:
    return max(max(stage_times) for stage_times in instance.processing_times_by_stage)
