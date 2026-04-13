from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    StageIdType,
)


def get_stage_job_sequences_from_dispatch_windows(
    stage_id_list: Sequence[StageIdType],
    job_id_list: Sequence[JobIdType],
    dispatch_window_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    sort_rule: str = "es_ls_p_desc",
) -> dict[StageIdType, list[JobIdType]]:
    """Build a stage-specific job sequence from ES/LS dispatch windows.

    For each stage, jobs are sorted by:
    - according to the provided sort_rule, then
    - job index ascending as the last tie-break
    """
    stage_2_job_sequence: dict[StageIdType, list[JobIdType]] = {}
    for stage_idx, stage_id in enumerate(stage_id_list, start=1):
        sortable_rows: list[tuple[tuple[float, ...], JobIdType]] = []
        for job_idx, job_id in enumerate(job_id_list, start=1):
            op_window = dispatch_window_lookup.get((stage_idx, job_idx))
            if op_window is None:
                raise ValueError(
                    f"Missing dispatch-window information for stage={stage_idx}, job={job_idx}."
                )
            processing_time = stage_2_job_2_p[stage_id][job_id]
            sortable_rows.append((
                _get_dispatch_window_sort_key(
                    op_window,
                    processing_time,
                    job_idx,
                    sort_rule=sort_rule,
                ),
                job_id,
            ))
        sortable_rows.sort(key=lambda row: row[0])
        stage_2_job_sequence[stage_id] = [job_id for _key, job_id in sortable_rows]
    return stage_2_job_sequence


def get_job_sequence_from_dispatch_windows_anchor_stage(
    stage_id_list: Sequence[StageIdType],
    job_id_list: Sequence[JobIdType],
    dispatch_window_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    *,
    anchor_stage_id: StageIdType,
    sort_rule: str = "es_ls_p_desc",
) -> list[JobIdType]:
    """Build a global job sequence from one anchor stage's ES/LS window order."""
    if anchor_stage_id not in stage_id_list:
        raise ValueError(f"Unknown anchor_stage_id: {anchor_stage_id}")

    stage_idx = stage_id_list.index(anchor_stage_id) + 1
    sortable_rows: list[tuple[tuple[float, ...], JobIdType]] = []
    for job_idx, job_id in enumerate(job_id_list, start=1):
        op_window = dispatch_window_lookup.get((stage_idx, job_idx))
        if op_window is None:
            raise ValueError(
                f"Missing dispatch-window information for stage={stage_idx}, job={job_idx}."
            )
        processing_time = stage_2_job_2_p[anchor_stage_id][job_id]
        sortable_rows.append((
            _get_dispatch_window_sort_key(
                op_window,
                processing_time,
                job_idx,
                sort_rule=sort_rule,
            ),
            job_id,
        ))
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def get_job_sequence_from_dispatch_windows_aggregate(
    stage_id_list: Sequence[StageIdType],
    job_id_list: Sequence[JobIdType],
    dispatch_window_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    *,
    aggregation_rule: str = "sum_es_slack_p_desc",
) -> list[JobIdType]:
    """Build a global job sequence by aggregating ES/LS window metrics over stages."""
    sortable_rows: list[tuple[tuple[float, ...], JobIdType]] = []
    for job_idx, job_id in enumerate(job_id_list, start=1):
        es_list: list[float] = []
        ls_list: list[float] = []
        slack_list: list[float] = []
        total_p = 0.0
        for stage_idx, stage_id in enumerate(stage_id_list, start=1):
            op_window = dispatch_window_lookup.get((stage_idx, job_idx))
            if op_window is None:
                raise ValueError(
                    f"Missing dispatch-window information for stage={stage_idx}, job={job_idx}."
                )
            processing_time = float(stage_2_job_2_p[stage_id][job_id])
            es = float(op_window["early_start"])
            ls = float(op_window["late_start"])
            es_list.append(es)
            ls_list.append(ls)
            slack_list.append(ls - es)
            total_p += processing_time
        sortable_rows.append((
            _get_dispatch_window_aggregate_key(
                es_list=es_list,
                ls_list=ls_list,
                slack_list=slack_list,
                total_p=total_p,
                job_idx=job_idx,
                aggregation_rule=aggregation_rule,
            ),
            job_id,
        ))
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def get_job_tiebreak_rank_from_job_sequence(
    job_sequence: Sequence[JobIdType],
) -> dict[JobIdType, int]:
    """Build a global rank map from a single job sequence."""
    return {job_id: idx for idx, job_id in enumerate(job_sequence)}


def _get_dispatch_window_sort_key(
    op_window: Mapping[str, Any],
    processing_time: int | float,
    job_idx: int,
    *,
    sort_rule: str,
) -> tuple[float, ...]:
    early_start = float(op_window["early_start"])
    late_start = float(op_window["late_start"])
    slack = late_start - early_start
    midpoint = early_start + late_start
    proc = float(processing_time)

    if sort_rule == "es_ls_p_desc":
        return (early_start, late_start, -proc, job_idx)
    if sort_rule == "ls_es_p_desc":
        return (late_start, early_start, -proc, job_idx)
    if sort_rule == "slack_ls_es_p_desc":
        return (slack, late_start, early_start, -proc, job_idx)
    if sort_rule == "es_slack_ls_p_desc":
        return (early_start, slack, late_start, -proc, job_idx)
    if sort_rule == "midpoint_slack_ls_p_desc":
        return (midpoint, slack, late_start, -proc, job_idx)
    raise ValueError(f"Unknown dispatch-window sort_rule: {sort_rule}")


def _get_dispatch_window_aggregate_key(
    *,
    es_list: Sequence[float],
    ls_list: Sequence[float],
    slack_list: Sequence[float],
    total_p: float,
    job_idx: int,
    aggregation_rule: str,
) -> tuple[float, ...]:
    if aggregation_rule == "sum_es_slack_p_desc":
        return (sum(es_list), sum(slack_list), sum(ls_list), -total_p, job_idx)
    if aggregation_rule == "sum_ls_slack_p_desc":
        return (sum(ls_list), sum(slack_list), sum(es_list), -total_p, job_idx)
    if aggregation_rule == "tail_ls_sum_slack_p_desc":
        return (ls_list[-1], sum(slack_list), es_list[-1], -total_p, job_idx)
    raise ValueError(
        f"Unknown dispatch-window aggregation_rule: {aggregation_rule}"
    )


def get_stage_job_release_times_from_dispatch_windows(
    stage_id_list: Sequence[StageIdType],
    job_id_list: Sequence[JobIdType],
    dispatch_window_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[StageIdType, dict[JobIdType, int]]:
    """Build stage/job release times from MIP ES values."""
    stage_2_job_2_release: dict[StageIdType, dict[JobIdType, int]] = {}
    for stage_idx, stage_id in enumerate(stage_id_list, start=1):
        job_2_release: dict[JobIdType, int] = {}
        for job_idx, job_id in enumerate(job_id_list, start=1):
            op_window = dispatch_window_lookup.get((stage_idx, job_idx))
            if op_window is None:
                raise ValueError(
                    f"Missing dispatch-window information for stage={stage_idx}, job={job_idx}."
                )
            job_2_release[job_id] = int(float(op_window["early_start"]))
        stage_2_job_2_release[stage_id] = job_2_release
    return stage_2_job_2_release


def get_stage_job_latest_start_times_from_dispatch_windows(
    stage_id_list: Sequence[StageIdType],
    job_id_list: Sequence[JobIdType],
    dispatch_window_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[StageIdType, dict[JobIdType, int]]:
    """Build stage/job latest-start targets from MIP LS values."""
    stage_2_job_2_latest_start: dict[StageIdType, dict[JobIdType, int]] = {}
    for stage_idx, stage_id in enumerate(stage_id_list, start=1):
        job_2_latest_start: dict[JobIdType, int] = {}
        for job_idx, job_id in enumerate(job_id_list, start=1):
            op_window = dispatch_window_lookup.get((stage_idx, job_idx))
            if op_window is None:
                raise ValueError(
                    f"Missing dispatch-window information for stage={stage_idx}, job={job_idx}."
                )
            job_2_latest_start[job_id] = int(float(op_window["late_start"]))
        stage_2_job_2_latest_start[stage_id] = job_2_latest_start
    return stage_2_job_2_latest_start


def get_job_tiebreak_rank_from_stage_job_sequences(
    stage_id_list: Sequence[StageIdType],
    stage_2_job_sequence: Mapping[StageIdType, Sequence[JobIdType]],
) -> dict[JobIdType, int]:
    """Build a global job rank from the first available ES/LS stage sequence."""
    for stage_id in stage_id_list:
        job_sequence = stage_2_job_sequence.get(stage_id)
        if job_sequence:
            return {job_id: idx for idx, job_id in enumerate(job_sequence)}
    return {}


def get_stage_job_sequences_from_schedule(
    schedule: HybridFlowshopLiteSchedule,
) -> dict[StageIdType, list[JobIdType]]:
    """Extract a stage-wise job order from a realized schedule."""
    stage_2_job_sequence: dict[StageIdType, list[JobIdType]] = {}
    for stage_id in schedule.stages:
        stage_jobs: list[tuple[int, int, JobIdType]] = []
        for mc_id in schedule.machines_per_stage[stage_id]:
            stage_jobs.extend(schedule.get_job_sequence(stage_id, mc_id))
        stage_jobs.sort(key=lambda row: (row[0], row[1], row[2]))
        stage_2_job_sequence[stage_id] = [job_id for _s, _e, job_id in stage_jobs]
    return stage_2_job_sequence


def get_bottleneck_anchor_stage_from_solution_payload(
    stage_id_list: Sequence[StageIdType],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    stage_2_machines: Mapping[StageIdType, Sequence[Any]],
    solution_payload: Mapping[str, Any] | None,
) -> StageIdType:
    """Select a bottleneck anchor stage using saved bucket usage when available.

    The primary signal is max stage-bucket congestion from ``x`` values:
    ``sum_j x[s,j,t] / (m_s * delta)``.
    When no payload is available, or all bucket usage is missing, it falls back
    to the classic average load proxy ``sum_j p[s,j] / m_s``.
    """
    if not stage_id_list:
        raise ValueError("stage_id_list cannot be empty.")

    delta = 0.0
    x_rows: list[dict[str, Any]] = []
    if solution_payload is not None:
        metadata = solution_payload.get("metadata", {}) or {}
        delta = float(metadata.get("delta", 0.0) or 0.0)
        x_rows = list(solution_payload.get("x", []) or [])

    stage_bucket_usage: dict[tuple[int, int], float] = {}
    for row in x_rows:
        stage_idx = int(row["stage"])
        bucket_idx = int(row["bucket"])
        stage_bucket_usage[(stage_idx, bucket_idx)] = stage_bucket_usage.get(
            (stage_idx, bucket_idx), 0.0
        ) + float(row["value"])
    stage_id_to_idx = {
        stage_id: stage_idx for stage_idx, stage_id in enumerate(stage_id_list, start=1)
    }

    def stage_key(stage_id: StageIdType) -> tuple[float, float]:
        stage_idx = stage_id_to_idx[stage_id]
        machine_cnt = max(len(stage_2_machines[stage_id]), 1)
        avg_load = (
            sum(float(proc) for proc in stage_2_job_2_p[stage_id].values()) / machine_cnt
        )
        if delta > 0:
            congestion = max(
                (
                    usage / (machine_cnt * delta)
                    for (row_stage_idx, _bucket_idx), usage in stage_bucket_usage.items()
                    if row_stage_idx == stage_idx
                ),
                default=0.0,
            )
        else:
            congestion = 0.0
        return congestion, avg_load

    return max(stage_id_list, key=stage_key)


def dispatch_stage_job_sequences_strict_call_order(
    schedule: HybridFlowshopLiteSchedule,
    stage_2_job_sequence: Mapping[StageIdType, Sequence[JobIdType]],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    stage_2_job_2_release: Mapping[StageIdType, Mapping[JobIdType, int]] | None = None,
) -> HybridFlowshopLiteSchedule:
    """Dispatch each stage using the exact provided stage-wise job order."""
    for stage_id in schedule.stages:
        if stage_id not in stage_2_job_sequence:
            raise ValueError(f"Missing job sequence for stage {stage_id}.")
        if stage_id not in stage_2_job_2_p:
            raise ValueError(f"Missing duration mapping for stage {stage_id}.")
        schedule.dispatch_stage_by_jobs_strict_sequence(
            stage_id,
            stage_2_job_sequence[stage_id],
            stage_2_job_2_p[stage_id],
            job_2_release=(
                stage_2_job_2_release.get(stage_id)
                if stage_2_job_2_release is not None
                else None
            ),
        )
    return schedule


def build_schedule_from_stage_job_sequences_strict_call_order(
    schedule_factory: Callable[[], HybridFlowshopLiteSchedule],
    stage_2_job_sequence: Mapping[StageIdType, Sequence[JobIdType]],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    stage_2_job_2_release: Mapping[StageIdType, Mapping[JobIdType, int]] | None = None,
) -> HybridFlowshopLiteSchedule:
    """Create a fresh schedule and dispatch it with the provided stage-wise order."""
    schedule = schedule_factory()
    return dispatch_stage_job_sequences_strict_call_order(
        schedule,
        stage_2_job_sequence,
        stage_2_job_2_p,
        stage_2_job_2_release=stage_2_job_2_release,
    )


def dispatch_stage_job_sequences_priority_score(
    schedule: HybridFlowshopLiteSchedule,
    stage_2_job_sequence: Mapping[StageIdType, Sequence[JobIdType]],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    stage_2_job_2_release: Mapping[StageIdType, Mapping[JobIdType, int]] | None = None,
) -> HybridFlowshopLiteSchedule:
    """Dispatch each stage with readiness-first priority and ES/LS tie-breaks.

    The caller provides a per-stage job order derived from MIP ES/LS windows.
    This function does not force that order as a hard call order. Instead, it
    builds an explicit release-aware priority queue stage by stage, using the
    provided order as the tie-break. This keeps post-MIP dispatch behavior
    readiness-aware even though the legacy initializer path keeps the original
    ``dispatch_stage_by_jobs()`` semantics.
    """
    for stage_id in schedule.stages:
        if stage_id not in stage_2_job_sequence:
            raise ValueError(f"Missing job sequence for stage {stage_id}.")
        if stage_id not in stage_2_job_2_p:
            raise ValueError(f"Missing duration mapping for stage {stage_id}.")
        job_2_release = (
            stage_2_job_2_release.get(stage_id)
            if stage_2_job_2_release is not None
            else None
        )
        stage_priority_queue = schedule.get_job_priority_queue_for_stage_dispatch(
            stage_id,
            stage_2_job_sequence[stage_id],
            job_2_release=job_2_release,
        )
        for job_id in stage_priority_queue:
            duration = stage_2_job_2_p[stage_id][job_id]
            release_t = job_2_release[job_id] if job_2_release is not None else None
            schedule.add_operation_2_stage(
                stage_id, job_id, duration, release_t=release_t
            )
    return schedule


def build_schedule_from_stage_job_sequences_priority_score(
    schedule_factory: Callable[[], HybridFlowshopLiteSchedule],
    stage_2_job_sequence: Mapping[StageIdType, Sequence[JobIdType]],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    stage_2_job_2_release: Mapping[StageIdType, Mapping[JobIdType, int]] | None = None,
) -> HybridFlowshopLiteSchedule:
    """Create a fresh schedule and dispatch it by readiness with ES/LS tie-break."""
    schedule = schedule_factory()
    return dispatch_stage_job_sequences_priority_score(
        schedule,
        stage_2_job_sequence,
        stage_2_job_2_p,
        stage_2_job_2_release=stage_2_job_2_release,
    )


def improve_schedule_by_critical_stage_sequence_insertions(
    schedule_factory: Callable[[], HybridFlowshopLiteSchedule],
    schedule: HybridFlowshopLiteSchedule,
    stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
    *,
    target_stage_ids: Sequence[StageIdType] | None = None,
    max_passes: int = 2,
    max_shift: int = 3,
) -> HybridFlowshopLiteSchedule:
    """Improve a schedule by reinserting critical jobs in stage sequences.

    This is a light-weight insertion neighborhood: it extracts the current
    stage-wise job order, shifts one critical job within a target stage, and
    rebuilds the full schedule with readiness-aware dispatch.
    """
    if max_passes <= 0:
        return schedule.deepcopy()

    best_schedule = schedule.deepcopy()
    best_schedule.make_semi_active(stage_2_job_2_duration)
    target_stage_id_set = set(target_stage_ids or [])

    for _pass_idx in range(max_passes):
        critical_blocks = best_schedule.find_critical_blocks(
            stage_2_job_2_duration,
            include_singletons=False,
        )
        if not critical_blocks:
            break

        current_stage_sequences = get_stage_job_sequences_from_schedule(best_schedule)
        stage_2_job_position = {
            stage_id: {job_id: idx for idx, job_id in enumerate(job_sequence)}
            for stage_id, job_sequence in current_stage_sequences.items()
        }
        best_neighbor: HybridFlowshopLiteSchedule | None = None
        best_neighbor_makespan = best_schedule.makespan

        for block in critical_blocks:
            for job_id, stage_id, _mc_id in block:
                if target_stage_id_set and stage_id not in target_stage_id_set:
                    continue
                current_seq = current_stage_sequences.get(stage_id, [])
                current_pos = stage_2_job_position.get(stage_id, {}).get(job_id)
                if len(current_seq) <= 1 or current_pos is None:
                    continue
                for shift in range(-max_shift, max_shift + 1):
                    if shift == 0:
                        continue
                    new_pos = current_pos + shift
                    if new_pos < 0 or new_pos >= len(current_seq):
                        continue
                    stage_seq = list(current_seq)
                    stage_seq.pop(current_pos)
                    stage_seq.insert(new_pos, job_id)
                    trial_stage_sequences = dict(current_stage_sequences)
                    trial_stage_sequences[stage_id] = stage_seq
                    candidate = build_schedule_from_stage_job_sequences_priority_score(
                        schedule_factory,
                        trial_stage_sequences,
                        stage_2_job_2_duration,
                    )
                    candidate.make_semi_active(stage_2_job_2_duration)
                    if candidate.makespan < best_neighbor_makespan:
                        best_neighbor = candidate
                        best_neighbor_makespan = candidate.makespan

        if best_neighbor is None:
            break
        best_schedule = best_neighbor

    return best_schedule


def improve_schedule_by_critical_adjacent_swaps(
    schedule: HybridFlowshopLiteSchedule,
    stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
    *,
    max_passes: int = 3,
    max_adjacent_pairs_per_pass: int | None = None,
    stage_2_job_2_release: Mapping[StageIdType, Mapping[JobIdType, int]] | None = None,
) -> HybridFlowshopLiteSchedule:
    """Greedily improve a schedule with adjacent swaps on critical blocks.

    The schedule is first normalized to a semi-active schedule. Each pass then:
    1. extracts critical blocks,
    2. enumerates adjacent job pairs inside those blocks,
    3. evaluates one-swap neighbors, and
    4. accepts the best improving move, if any.

    This is intentionally lightweight: it uses only local schedule edits and
    semi-active retiming, without invoking CP.
    """
    if max_passes <= 0:
        return schedule.deepcopy()

    def respects_release_times(candidate: HybridFlowshopLiteSchedule) -> bool:
        if stage_2_job_2_release is None:
            return True
        for stage_id, job_2_release in stage_2_job_2_release.items():
            for job_id, release_t in job_2_release.items():
                if candidate.get_job_start_time(stage_id, job_id) < release_t:
                    return False
        return True

    best_schedule = schedule.deepcopy()
    best_schedule.make_semi_active(stage_2_job_2_duration)
    if not respects_release_times(best_schedule):
        return schedule.deepcopy()

    for _pass_idx in range(max_passes):
        critical_blocks = best_schedule.find_critical_blocks(
            stage_2_job_2_duration,
            include_singletons=False,
        )
        if not critical_blocks:
            break

        adjacent_pairs: list[tuple[StageIdType, JobIdType, JobIdType]] = []
        seen_pairs: set[tuple[StageIdType, JobIdType, JobIdType]] = set()
        for block in critical_blocks:
            for left_op, right_op in zip(block, block[1:]):
                pair = (left_op[1], left_op[0], right_op[0])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                adjacent_pairs.append(pair)

        if max_adjacent_pairs_per_pass is not None:
            adjacent_pairs = adjacent_pairs[:max_adjacent_pairs_per_pass]

        incumbent_makespan = best_schedule.makespan
        best_neighbor: HybridFlowshopLiteSchedule | None = None
        best_neighbor_makespan = incumbent_makespan

        for stage_id, first_job_id, second_job_id in adjacent_pairs:
            candidate = best_schedule.deepcopy()
            try:
                candidate.swap_two_operations_within_stage(
                    stage_id,
                    first_job_id,
                    second_job_id,
                    stage_2_job_2_duration,
                    do_make_semi_active=True,
                )
            except ValueError:
                continue

            candidate_makespan = candidate.makespan
            if not respects_release_times(candidate):
                continue
            if candidate_makespan < best_neighbor_makespan:
                best_neighbor = candidate
                best_neighbor_makespan = candidate_makespan

        if best_neighbor is None:
            break

        best_schedule = best_neighbor

    return best_schedule


def dispatch_job_sequence_by_stages(
    schedule: HybridFlowshopLiteSchedule,
    job_sequence: Sequence[JobIdType],
    job_2_stage_2_p: Mapping[JobIdType, Mapping[StageIdType, int]],
    from_stage: StageIdType | None = None,
    job_2_release_t: Mapping[JobIdType, int] | None = None,
):
    """Dispatch jobs in the given sequence, stage by stage.

    Args:
        schedule (HybridFlowshopLiteSchedule): The schedule to modify in-place.
        job_sequence (Sequence[JobIdType]): The sequence of job IDs to dispatch.
        job_2_stage_2_p (Mapping[JobIdType, Mapping[StageIdType, int]]): The processing
            time for each job on each stage.
        from_stage (StageIdType | None, optional): The first stage to dispatch jobs.
            If not provided, defaults to the first stage in schedule.
        job_2_release_t (Mapping[JobIdType, int] | None, optional): The release time
            for each job. Defaults to None.
    """
    for job_id in job_sequence:
        schedule.dispatch_job_by_stages(
            job_id,
            job_2_stage_2_p[job_id],
            from_stage=from_stage,
            release_t=job_2_release_t[job_id] if job_2_release_t else None,
        )


def dispatch_stages_by_job_sequence(
    schedule: HybridFlowshopLiteSchedule,
    job_sequence: Sequence[JobIdType],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    from_stage: StageIdType | None = None,
    job_2_release_t: Mapping[JobIdType, int] | None = None,
    machine_then_job: bool = False,
) -> None:
    """Dispatch stages, one by one, with the given job sequence.

    Args:
        schedule (HybridFlowshopLiteSchedule): The schedule to modify in-place.
        job_sequence (Sequence[JobIdType]): The sequence of job IDs to dispatch.
        stage_2_job_2_p (Mapping[StageIdType, Mapping[JobIdType, int]]): The processing
            time for each stage on each job.
        from_stage (StageIdType | None, optional): The first stage to dispatch jobs.
            If not provided, defaults to the first stage in schedule.
        job_2_release_t (Mapping[JobIdType, int] | None, optional): The release time
            for each job. Defaults to None.
        machine_then_job (bool, optional): If True, dispatch each stage by machine
            first then job if the stage is not the first stage of the schedule.
            If False, dispatch by job first then machine. Defaults to False.
    """
    _stage_id_list = schedule.stages
    if from_stage is not None:
        from_stage_index = _stage_id_list.index(from_stage)
        _stage_id_list = _stage_id_list[from_stage_index:]
    if machine_then_job:
        # If first target stage is the first stage of the schedule, dispatch it by job
        # first then machine. Otherwise, dispatch it by machine first then job.
        if _stage_id_list[0] == schedule.stages[0]:
            schedule.dispatch_stage_by_jobs(
                _stage_id_list[0],
                job_sequence,
                stage_2_job_2_p[_stage_id_list[0]],
                job_2_release=job_2_release_t,
            )
        else:
            schedule.machine_centric_dispatch_4(
                _stage_id_list[0],
                job_sequence,
                stage_2_job_2_p,
                job_2_release=job_2_release_t,
            )
        # Remaining stages: Machine-centric dispatch
        for stage_id in _stage_id_list[1:]:
            schedule.machine_centric_dispatch_4(
                stage_id,
                job_sequence,
                stage_2_job_2_p,
                job_2_release=job_2_release_t,
            )
    else:
        for stage_id in _stage_id_list:
            schedule.dispatch_stage_by_jobs(
                stage_id,
                job_sequence,
                stage_2_job_2_p[stage_id],
                job_2_release=job_2_release_t,
            )


def dispatch_stages_by_job_sequence_strict(
    schedule: HybridFlowshopLiteSchedule,
    job_sequence: Sequence[JobIdType],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    from_stage: StageIdType | None = None,
    job_2_release_t: Mapping[JobIdType, int] | None = None,
) -> None:
    """Dispatch stages one by one while preserving the exact given job order.

    This is the strict-sequence counterpart of :func:`dispatch_stages_by_job_sequence`.
    Each stage uses ``dispatch_stage_by_jobs_strict_sequence()`` so the provided
    ``job_sequence`` is not reordered by readiness.
    """
    _stage_id_list = schedule.stages
    if from_stage is not None:
        from_stage_index = _stage_id_list.index(from_stage)
        _stage_id_list = _stage_id_list[from_stage_index:]

    for stage_id in _stage_id_list:
        schedule.dispatch_stage_by_jobs_strict_sequence(
            stage_id,
            job_sequence,
            stage_2_job_2_p[stage_id],
            job_2_release=job_2_release_t,
        )


def from_job_sequence_get_schedule_mixed(
    schedule: HybridFlowshopLiteSchedule,
    job_sequence: Sequence[JobIdType],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    stage_2_head: Mapping[StageIdType, int],
    from_stage: StageIdType | None = None,
    job_2_release: Mapping[JobIdType, int] | None = None,
    machine_then_job: bool = False,
    use_palmer_index: bool = False,
    draw_gantt_per_step: bool = False,
    get_file_path_for_subroutine: Callable | None = None,
) -> None:
    """Mixed dispatch strategy with per-stage k values that supports incremental
    scheduling.

    Implements a hybrid dispatch strategy that combines:
    1. dispatch_job_by_stages: dispatch k jobs through all stages from from_stage onward
    2. dispatch_stage_by_jobs: priority-based scheduling for remaining jobs

    The strategy works as follows:
    - For each stage from from_stage to the last stage (in order), determine the
      remaining jobs that have not been fully dispatched from some earlier stage,
      and sort them by priority (earliest previous stage end time first, then
      sequence order). Dispatch the first k jobs from this priority queue through
      all remaining stages using dispatch_job_by_stages.
    - Fill remaining slots at each stage using dispatch_stage_by_jobs with the
      same priority rule.
    - A job that has been dispatched through all remaining stages via
      dispatch_job_by_stages is treated as completed and skipped in subsequent stages.
      (Jobs scheduled only at a single stage via dispatch_stage_by_jobs are not
      considered completed; they will be scheduled again at later stages.)

    This allows for a flexible mix of sequence-based dispatch (for priority jobs)
    and priority-based dispatch (for remaining jobs).

    Args:
        schedule (HybridFlowshopLiteSchedule): The schedule to modify in-place.
            The schedule may be empty or partially scheduled.
        job_sequence (Sequence[JobIdType]): Sequence of job IDs to schedule. The order
            represents the tiebreaker priority when jobs have the same previous stage
            end time.
        stage_2_job_2_p (Mapping[StageIdType, Mapping[JobIdType, int]]): stage -> job
            -> processing time mapping. Must include all jobs and stages.
        stage_2_head (Mapping[StageIdType, int]): Mapping from stage_id to the number
            of jobs to dispatch via dispatch_job_by_stages at that stage. These values
            are applied as a cumulative cap across stages: once k jobs have been fully
            dispatched from earlier stages, they reduce the remaining budget for later
            stages.
            E.g., with 5 jobs and stage_2_head={"s1": 3, "s2": 3}, s2 effectively
            uses min(3, 5-3) = 2 since only 2 jobs remain after s1 dispatches 3 jobs.
        from_stage (StageIdType | None, optional): The first stage to dispatch jobs.
            If not provided, defaults to the first stage in schedule.
        job_2_release (dict[str, int] | None, optional): Mapping from job ID to release
            time, used for the first stage's priority queue (as a lower bound on when
            the job can start).
            If provided, the priority is max(prev_stage_end, release_time).
            Ignored for later stages since their priority is determined by previous
            stage completion. Defaults to None.
        machine_then_job (bool, optional): If True, dispatch each stage by machine
            first then job if the stage is not the first stage of the schedule.
            If False, dispatch by job first then machine. Defaults to False.
        draw_gantt_per_step (bool, optional): If True, draw a Gantt chart for each step
            being scheduled. Defaults to False.
        get_file_path_for_subroutine (Callable | None, optional): Callable that returns
            a file path for each subroutine call. If provided, it is called with the
            subroutine name and stage ID to generate a file path for saving Gantt
            charts. Defaults to None.

    Raises:
        ValueError: If *from_stage* is provided but is not a valid stage in the schedule.
        ValueError: If any stage_id in *stage_2_head* is not a valid stage in the schedule.
        ValueError: If any value in *stage_2_head* is negative.
        ValueError: If any required duration in *stage_2_job_2_p* is missing for the jobs and stages being scheduled.
    """
    # Validate from_stage if provided
    if from_stage is not None and from_stage not in schedule.stages:
        raise ValueError(f"from_stage '{from_stage}' is not in schedule.stages")

    # Determine target stage list
    if from_stage is None:
        _stage_id_list = schedule.stages
    else:
        _stage_id_list = schedule.stages[schedule.stages.index(from_stage) :]

    # Validation: stage_2_head keys must be valid stages, and values must be non-negative
    for stage_id, k in stage_2_head.items():
        if stage_id not in _stage_id_list:
            raise ValueError(f"Unknown stage_id in stage_2_head: {stage_id}")
        if k < 0:
            raise ValueError(
                f"stage_2_head values must be non-negative, got {k} for stage {stage_id}"
            )

    # Validation: check all required durations are provided for target stages
    for stage_id in _stage_id_list:
        for job_id in job_sequence:
            if job_id not in stage_2_job_2_p.get(stage_id, {}):
                raise ValueError(
                    f"Duration for job ID {job_id} at stage {stage_id} not provided"
                )

    if get_file_path_for_subroutine is None:
        draw_gantt_per_step = False
    if draw_gantt_per_step:
        plotter = GanttPlotter()

    # Preprocess stage_2_head with cumulative adjustment
    # If stage_2_head = {"s1": 3, "s2": 3} and total jobs = 5,
    # adjust to {"s1": 3, "s2": 2} because only 2 jobs remain after s1 dispatches 3
    _stage_2_head: dict[StageIdType, int] = {}
    remaining_jobs_for_stage = len(job_sequence)
    for stage_id in _stage_id_list:
        if stage_id in stage_2_head:
            _stage_2_head[stage_id] = min(
                stage_2_head[stage_id], remaining_jobs_for_stage
            )
            remaining_jobs_for_stage -= _stage_2_head[stage_id]
        else:
            _stage_2_head[stage_id] = 0

    completed_job_set: set[JobIdType] = set()
    # For each stage, dispatch k jobs via dispatch_job_by_stages,
    # then fill remaining slots via dispatch_stage_by_jobs
    for stage_idx, stage_id in enumerate(_stage_id_list):
        if stage_idx == 0:
            _job_2_release = job_2_release
        else:
            _job_2_release = None

        # Job priority queue for this stage: sorted by (1) earliest start time, then (2) sequence order
        remaining_job_sequence = [
            job_id for job_id in job_sequence if job_id not in completed_job_set
        ]
        job_priority_queue = schedule.get_job_priority_queue_for_stage_dispatch(
            stage_id, remaining_job_sequence, job_2_release=_job_2_release
        )

        # Phase 1: dispatch k jobs through all remaining stages (skip if stage_k is 0)
        stage_k = _stage_2_head.get(stage_id, 0)
        if stage_k > 0:
            stages_from_here = _stage_id_list[stage_idx:]
            first_k_jobs = list(job_priority_queue)[:stage_k]

            for job_id in first_k_jobs:
                job_stage_dur = {
                    s: stage_2_job_2_p[s][job_id] for s in stages_from_here
                }
                schedule.dispatch_job_by_stages(
                    job_id,
                    job_stage_dur,
                    from_stage=stage_id,
                    release_t=_job_2_release.get(job_id, 0) if _job_2_release else 0,
                )
                completed_job_set.add(job_id)
        if draw_gantt_per_step:
            if get_file_path_for_subroutine is None:
                raise ValueError(
                    "get_file_path_for_subroutine must be provided if draw_gantt_per_step is True"
                )
            output_path: Path = get_file_path_for_subroutine(
                f"{stage_id}_after_job_by_stages.png"
            )
            plotter.export_hybrid_flowshop_plot(
                output_path,
                schedule.get_jik_2_start_time_map(),
                schedule.get_jik_2_end_time_map(),
                job_sequence,
                schedule.stages,
            )
        # Phase 2: Fill remaining slots at this stage using dispatch_stage_by_jobs
        unscheduled_jobs = [
            job_id for job_id in job_priority_queue if job_id not in completed_job_set
        ]
        if unscheduled_jobs:
            if machine_then_job:
                if stage_id == schedule.stages[0]:
                    schedule.dispatch_stage_by_jobs(
                        stage_id,
                        unscheduled_jobs,
                        stage_2_job_2_p[stage_id],
                        job_2_release=_job_2_release,
                    )
                else:
                    schedule.machine_centric_dispatch_4(
                        stage_id,
                        unscheduled_jobs,
                        stage_2_job_2_p,
                        job_2_release=_job_2_release,
                        use_palmer_index=use_palmer_index,
                    )
            else:
                schedule.dispatch_stage_by_jobs(
                    stage_id,
                    unscheduled_jobs,
                    stage_2_job_2_p[stage_id],
                    job_2_release=_job_2_release,
                )
            if draw_gantt_per_step:
                if get_file_path_for_subroutine is None:
                    raise ValueError(
                        "get_file_path_for_subroutine must be provided if draw_gantt_per_step is True"
                    )
                output_path = get_file_path_for_subroutine(
                    f"{stage_id}_after_stage_by_jobs.png"
                )
                plotter.export_hybrid_flowshop_plot(
                    output_path,
                    schedule.get_jik_2_start_time_map(),
                    schedule.get_jik_2_end_time_map(),
                    job_sequence,
                    schedule.stages,
                )


def reverse_even_positions(sequence: list[Any], in_place: bool = False) -> list[Any]:
    """Reverse only even positions (1-based), keeping odd positions fixed.

    For example, [A,B,C,D,E,F,G,H] -> [A,H,C,F,E,D,G,B]

    Args:
        sequence: The input sequence to be modified.
        in_place: If True, modify the input sequence in place and return it.
            If False, return a new modified list. Defaults to False.

    Returns:
        The modified sequence with even positions reversed.
    """
    if in_place:
        result = sequence
    else:
        result = sequence.copy()
    even_position_elements = result[1::2]
    even_position_elements.reverse()
    result[1::2] = even_position_elements
    return result
