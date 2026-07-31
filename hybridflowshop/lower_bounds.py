"""Analytic lower bounds for hybrid flow shop makespan instances."""

from __future__ import annotations

import math
from typing import Any


def _instance_data(
    instance: Any,
) -> tuple[list[str], list[str], dict[tuple[str, str], int], dict[str, int]]:
    jobs: list[str] = list(instance.job_id_list)
    stages: list[str] = list(instance.stage_id_list)
    p: dict[tuple[str, str], int] = instance.p_manager.job_stage_2_value_map(
        jobs, stages
    )
    machine_counts = {
        stage: len(instance.stage_2_machines_map[stage]) for stage in stages
    }
    return jobs, stages, p, machine_counts


def simple_job_lower_bound(instance: Any) -> int:
    """Return LB0: the longest total processing time of any job."""
    jobs, stages, p, _ = _instance_data(instance)
    if not jobs or not stages:
        return 0
    return int(max(sum(p[job, stage] for stage in stages) for job in jobs))


def santos_stage_lower_bound(instance: Any, stage: str) -> int:
    """Return the Santos et al. stage-based lower bound for one stage.

    This is LB2(k) in Chen et al. (2023), and is the bound already used by
    ``apply_shdlb``.
    """
    jobs, stages, p, machine_counts = _instance_data(instance)
    stage_idx = stages.index(stage)
    machine_count = machine_counts[stage]
    if machine_count <= 0:
        raise ValueError(f"Stage {stage!r} must have at least one machine.")

    left_sums = sorted(sum(p[job, stages[s]] for s in range(stage_idx)) for job in jobs)
    right_sums = sorted(
        sum(p[job, stages[s]] for s in range(stage_idx + 1, len(stages)))
        for job in jobs
    )
    total_processing = sum(p[job, stage] for job in jobs)
    left_head = sum(left_sums[:machine_count])
    right_tail = sum(right_sums[:machine_count])
    return int(math.ceil((left_head + total_processing + right_tail) / machine_count))


def santos_lower_bound(instance: Any) -> int:
    """Return max(LB0, max_k LB2(k)) for an HFS instance."""
    _, stages, _, _ = _instance_data(instance)
    stage_bounds = [santos_stage_lower_bound(instance, stage) for stage in stages]
    return max([simple_job_lower_bound(instance)] + stage_bounds)


def bin_packing_body_lower_bound(
    processing_times: list[int] | tuple[int, ...], machine_count: int
) -> int:
    """Return the Dell'Amico-Martello style body bound used as B3(k).

    This implements Steps 1-9 of Chen et al.'s fourth lower-bound method for
    the P//Cmax body of a single stage.
    """
    if machine_count <= 0:
        raise ValueError("machine_count must be positive.")
    times = sorted((int(value) for value in processing_times), reverse=True)
    n = len(times)
    if n == 0:
        return 0

    blb0 = math.ceil(sum(times) / machine_count)
    blb1 = times[0]
    if machine_count > n - 1:
        blb2 = times[0]
    else:
        blb2 = times[machine_count - 1] + times[machine_count]
    lower_bound = max(blb0, blb1, blb2)

    if machine_count > n - 2:
        p_bar = times[-1]
    else:
        p_bar = times[machine_count + 1]
    if p_bar <= 0:
        raise ValueError("processing times must be positive for the bin-packing bound.")

    while True:
        large_jobs = [value for value in times if lower_bound - p_bar < value]
        medium_jobs = [
            value for value in times if lower_bound / 2 < value <= lower_bound - p_bar
        ]
        compact_jobs = [value for value in times if p_bar <= value <= lower_bound / 2]

        medium_capacity_for_compact = lower_bound * len(medium_jobs) - sum(medium_jobs)
        alpha_extra = math.ceil(
            (sum(compact_jobs) - medium_capacity_for_compact) / lower_bound
        )
        alpha_bins = len(large_jobs) + len(medium_jobs) + max(0, alpha_extra)

        denominator = lower_bound // p_bar
        beta_extra = math.ceil(
            (
                len(compact_jobs)
                - sum((lower_bound - value) // p_bar for value in medium_jobs)
            )
            / denominator
        )
        beta_bins = len(large_jobs) + len(medium_jobs) + max(0, beta_extra)

        if alpha_bins <= machine_count and beta_bins <= machine_count:
            return int(lower_bound)
        lower_bound += 1


def lb3_body_bounds(instance: Any) -> dict[str, int]:
    """Return B3(k), the bin-packing body bound, for every stage."""
    jobs, stages, p, machine_counts = _instance_data(instance)
    return {
        stage: bin_packing_body_lower_bound(
            [p[job, stage] for job in jobs], machine_counts[stage]
        )
        for stage in stages
    }


def chen_lb4_stage_lower_bound(instance: Any, stage: str) -> int:
    """Return Chen et al. (2023) LB4(k) for one stage.

    The implementation follows the appendix example and Proposition 2: T4_min
    is the minimum tail, matching the T3(k) term used in the dominance proof.
    """
    jobs, stages, p, _ = _instance_data(instance)
    stage_idx = stages.index(stage)
    body_bound = bin_packing_body_lower_bound(
        [p[job, stage] for job in jobs],
        len(instance.stage_2_machines_map[stage]),
    )

    prefix_before = [sum(p[job, stages[s]] for s in range(stage_idx)) for job in jobs]
    prefix_through = [
        sum(p[job, stages[s]] for s in range(stage_idx + 1)) for job in jobs
    ]
    suffix_after = [
        sum(p[job, stages[s]] for s in range(stage_idx + 1, len(stages)))
        for job in jobs
    ]

    h4_earliest = 0 if stage_idx == 0 else min(prefix_before)
    b4_earliest = min(prefix_through)
    b4_latest = max(h4_earliest + body_bound, max(prefix_through))
    t4_max = 0 if stage_idx == len(stages) - 1 else max(suffix_after)
    t4_min = 0 if stage_idx == len(stages) - 1 else min(suffix_after)

    return int(max(b4_earliest + t4_max, b4_latest + t4_min))


def chen_lb4_stage_lower_bounds(instance: Any) -> dict[str, int]:
    """Return Chen et al. (2023) LB4(k) for every stage."""
    _, stages, _, _ = _instance_data(instance)
    return {stage: chen_lb4_stage_lower_bound(instance, stage) for stage in stages}


def chen_lb4_lower_bound(instance: Any) -> int:
    """Return Chen et al. (2023) LB4 for an HFS instance."""
    stage_bounds = chen_lb4_stage_lower_bounds(instance)
    return int(max(stage_bounds.values(), default=0))
