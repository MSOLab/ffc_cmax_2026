from pathlib import Path
from typing import Callable, Mapping, Sequence

from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    StageIdType,
)


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
    """
    _stage_id_list = schedule.stages
    if from_stage is not None:
        from_stage_index = _stage_id_list.index(from_stage)
        _stage_id_list = _stage_id_list[from_stage_index:]
    for stage_id in _stage_id_list:
        schedule.dispatch_stage_by_jobs(
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
        target_stage_list = schedule.stages
    else:
        target_stage_list = schedule.stages[schedule.stages.index(from_stage) :]

    # Validation: stage_2_head keys must be valid stages, and values must be non-negative
    for stage_id, k in stage_2_head.items():
        if stage_id not in target_stage_list:
            raise ValueError(f"Unknown stage_id in stage_2_head: {stage_id}")
        if k < 0:
            raise ValueError(
                f"stage_2_head values must be non-negative, got {k} for stage {stage_id}"
            )

    # Validation: check all required durations are provided for target stages
    for stage_id in target_stage_list:
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
    for stage_id in target_stage_list:
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
    for stage_idx, stage_id in enumerate(target_stage_list):
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
            stages_from_here = target_stage_list[stage_idx:]
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
            schedule.dispatch_stage_by_jobs(
                stage_id,
                unscheduled_jobs,
                stage_2_job_2_p[stage_id],
                job_2_release=_job_2_release,
            )
            if draw_gantt_per_step:
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
