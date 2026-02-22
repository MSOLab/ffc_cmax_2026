from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence, TypeAlias

from hybridflowshop.painter.gantt import GanttPlotter

JobIdType = str
StageIdType = str
McIdType = str

# Wave batch scheduling type aliases
ReadyHeapEntry: TypeAlias = tuple[int, int, JobIdType, int]
"""(end_time, tie_breaker, job_id, stage_idx) for heap entries"""


@dataclass
class WaveBatchState:
    """State for wave batch scheduling algorithm."""

    ready_heaps: list[list[ReadyHeapEntry]]
    job_2_duration: dict[int, Mapping[JobIdType, int]]
    job_2_pos: dict[JobIdType, int]
    iteration_no: int
    total_jobs: int
    stage_ids: Sequence[StageIdType]
    _prev_scheduled_last_stage: int = 0


class HybridFlowshopLiteSchedule:
    # Parameters

    jobs: Sequence[JobIdType]
    """ID of jobs"""

    stages: Sequence[StageIdType]
    """ID of stages"""

    machines_per_stage: Mapping[StageIdType, Sequence[McIdType]]
    """map(stage ID -> ID of machines)"""

    # Helper parameters

    stage_2_prev_stage: Mapping[StageIdType, StageIdType | None]
    """map(stage ID -> previous stage ID or None if first stage)"""

    # Variables

    __stage_2_mc_2_job_tuple_seq: dict[
        StageIdType, dict[McIdType, list[tuple[int, int, JobIdType]]]
    ]
    """map(stage ID -> map(machine ID -> sequence of (start times, end times, job IDs)))"""

    __stage_2_job_2_end_time: dict[StageIdType, dict[JobIdType, int]]
    """map(stage ID -> map(job ID -> end time))"""

    __wave_batch_state: WaveBatchState | None
    """Wave batch dispatching internal state (WaveBatchState or None)"""

    def __init__(
        self,
        jobs: Sequence[JobIdType],
        stages: Sequence[StageIdType],
        machines_per_stage: Mapping[StageIdType, Sequence[McIdType]],
    ):
        self.jobs = jobs
        self.stages = stages
        self.machines_per_stage = machines_per_stage
        self.stage_2_prev_stage = {
            stage: stages[i - 1] if i > 0 else None for i, stage in enumerate(stages)
        }
        self._initialize_variables()

    def _initialize_variables(self):
        self.__stage_2_mc_2_job_tuple_seq = {
            stage: {mc: [] for mc in self.machines_per_stage[stage]}
            for stage in self.stages
        }
        self.__stage_2_job_2_end_time = {stage: {} for stage in self.stages}
        # Wave batch dispatching internal state
        self.__wave_batch_state = None

    def deepcopy(
        self, job_subsequence: set[JobIdType] | None = None
    ) -> HybridFlowshopLiteSchedule:
        new_instance = HybridFlowshopLiteSchedule(
            jobs=self.jobs,
            stages=self.stages,
            machines_per_stage=self.machines_per_stage,
        )

        for stage in self.stages:
            if job_subsequence is None:
                new_instance.__stage_2_job_2_end_time[stage] = {
                    job: end_time
                    for job, end_time in self.__stage_2_job_2_end_time[stage].items()
                }
            else:
                new_instance.__stage_2_job_2_end_time[stage] = {
                    job: end_time
                    for job, end_time in self.__stage_2_job_2_end_time[stage].items()
                    if job in job_subsequence
                }
            for mc in self.machines_per_stage[stage]:
                if job_subsequence is None:
                    new_instance.__stage_2_mc_2_job_tuple_seq[stage][mc] = [
                        job_tuple
                        for job_tuple in self.__stage_2_mc_2_job_tuple_seq[stage][mc]
                    ]
                else:
                    new_instance.__stage_2_mc_2_job_tuple_seq[stage][mc] = [
                        job_tuple
                        for job_tuple in self.__stage_2_mc_2_job_tuple_seq[stage][mc]
                        if job_tuple[2] in job_subsequence
                    ]

        return new_instance

    # Getters

    def get_job_sequence(
        self, stage_id: StageIdType, mc_id: McIdType
    ) -> list[tuple[int, int, JobIdType]]:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")
        return self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id]

    def get_machine_latest_end_time(
        self, stage_id: StageIdType, mc_id: McIdType
    ) -> int:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")

        job_tuple_seq = self.get_job_sequence(stage_id, mc_id)
        if not job_tuple_seq:
            return 0
        return job_tuple_seq[-1][1]

    def get_machine_earliest_start_time(
        self,
        stage_id: StageIdType,
        mc_id: McIdType,
        duration: int,
        release_t: int | None = None,
        after_last: bool = False,
    ) -> int:
        """Return the earliest feasible start time on a machine.

        This mirrors the core behavior of `Resource.get_earliest_start_time()` in the
        full schedule implementation: the operation may be inserted into an idle gap
        between existing operations as long as no overlap occurs.
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        job_tuple_seq = self.get_job_sequence(stage_id, mc_id)
        prev_end = release_t if release_t is not None else 0

        if after_last:
            makespan = self.get_machine_latest_end_time(stage_id, mc_id)
            return makespan if makespan >= prev_end else prev_end

        if not job_tuple_seq:
            return prev_end

        # Find the first operation with start >= prev_end.
        starts = [job_tuple[0] for job_tuple in job_tuple_seq]
        start_idx = bisect.bisect_right(starts, prev_end - 1)

        # If the operation just before start_idx overlaps prev_end, push prev_end forward.
        if start_idx > 0:
            before_start, before_end, _ = job_tuple_seq[start_idx - 1]
            if before_end > prev_end:
                prev_end = before_end

        # Scan forward to find the first gap that can fit `duration`.
        for op_start, op_end, _ in job_tuple_seq[start_idx:]:
            if prev_end + duration <= op_start:
                return prev_end
            if prev_end < op_end:
                prev_end = op_end

        return prev_end

    def get_machine_and_earliest_available_time_by_start_idle_idx(
        self, stage_id: StageIdType, duration: int, release_t: int | None = None
    ) -> tuple[McIdType, int]:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if not self.machines_per_stage[stage_id]:
            raise ValueError(f"No machines available in stage {stage_id}.")
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        # Initialize with first machine's values
        first_mc = self.machines_per_stage[stage_id][0]

        best_eat = self.get_machine_earliest_start_time(
            stage_id, first_mc, duration, release_t=release_t
        )
        best_idle = best_eat - self.get_machine_latest_end_time(stage_id, first_mc)

        best_mc = first_mc

        # Check remaining machines
        for mc in self.machines_per_stage[stage_id][1:]:
            eat = self.get_machine_earliest_start_time(
                stage_id, mc, duration, release_t=release_t
            )
            idle = eat - self.get_machine_latest_end_time(stage_id, mc)

            # (1) earliest available time (2) smallest idle time
            if eat < best_eat or (eat == best_eat and idle < best_idle):
                best_mc, best_eat, best_idle = mc, eat, idle

        return best_mc, best_eat

    def get_job_end_time(
        self,
        stage_id: StageIdType,
        job_id: JobIdType,
        default_if_missing: int | None = None,
    ) -> int:
        """Return the completion time (end time) of `job_id` at `stage_id`.

        This is useful for enforcing the precedence constraint in hybrid flow shop:
        an operation at a later stage cannot start before the job completes at the
        previous stage.

        If the job is not found in the stage schedule:
        - return `default_if_missing` if it is provided
        - otherwise raise ValueError
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if job_id not in self.__stage_2_job_2_end_time[stage_id]:
            if default_if_missing is not None:
                return default_if_missing
            raise ValueError(f"Job ID {job_id} not found in stage ID {stage_id}")
        return self.__stage_2_job_2_end_time[stage_id][job_id]

    def get_prev_stage_end_time(
        self,
        stage_id: StageIdType,
        job_id: JobIdType,
        default_if_missing: int | None = None,
    ) -> int:
        """Return `job_id` completion time at the stage immediately before `stage_id`.

        For the first stage in `self.stages`, this returns 0.
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")

        prev_stage_id = self.stage_2_prev_stage[stage_id]
        if prev_stage_id is None:
            return 0
        return self.get_job_end_time(
            prev_stage_id, job_id, default_if_missing=default_if_missing
        )

    @property
    def makespan(self) -> int:
        last_stage = self.stages[-1]
        max_end_time = 0
        for mc in self.machines_per_stage[last_stage]:
            mc_latest_end_time = self.get_machine_latest_end_time(last_stage, mc)
            if mc_latest_end_time > max_end_time:
                max_end_time = mc_latest_end_time
        return max_end_time

    def iter_operations_on_stage(
        self, stage_id: StageIdType
    ) -> Iterator[tuple[McIdType, int, int, JobIdType]]:
        """Iterate over all operations on a stage yielding (mc, start_time, end_time, job_id)."""
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        for mc in self.machines_per_stage[stage_id]:
            for start_time, end_time, job_id in self.get_job_sequence(stage_id, mc):
                yield mc, start_time, end_time, job_id

    def _iter_operations(
        self,
    ) -> Iterator[tuple[StageIdType, McIdType, int, int, JobIdType]]:
        """Iterate over all operations yielding (stage, mc, start_time, end_time, job_id)."""
        for stage in self.stages:
            for mc, start_time, end_time, job_id in self.iter_operations_on_stage(
                stage
            ):
                yield stage, mc, start_time, end_time, job_id

    def get_operation_set(self) -> set[tuple[JobIdType, StageIdType, McIdType]]:
        return {
            (job_id, stage, mc) for stage, mc, _, _, job_id in self._iter_operations()
        }

    def get_jik_2_start_time_map(
        self,
    ) -> dict[tuple[JobIdType, StageIdType, McIdType], int]:
        return {
            (job_id, stage, mc): int(start)
            for stage, mc, start, _, job_id in self._iter_operations()
        }

    def get_jik_2_end_time_map(
        self,
    ) -> dict[tuple[JobIdType, StageIdType, McIdType], int]:
        return {
            (job_id, stage, mc): int(end)
            for stage, mc, _, end, job_id in self._iter_operations()
        }

    def get_ji_2_end_time_map(self) -> dict[tuple[JobIdType, StageIdType], int]:
        return {
            (job_id, stage): int(end)
            for stage, _, _, end, job_id in self._iter_operations()
        }

    def get_stage_2_mc_2_last_end_time_map(
        self,
    ) -> dict[StageIdType, dict[McIdType, int]]:
        stage_2_mc_2_last_end_time = {
            stage: {mc: 0 for mc in self.machines_per_stage[stage]}
            for stage in self.stages
        }
        for stage, mc, _, end, _ in self._iter_operations():
            if end > stage_2_mc_2_last_end_time[stage][mc]:
                stage_2_mc_2_last_end_time[stage][mc] = end
        return stage_2_mc_2_last_end_time

    # Setters

    def sort_by_start_times(self) -> None:
        for stage in self.stages:
            for mc in self.machines_per_stage[stage]:
                self.__stage_2_mc_2_job_tuple_seq[stage][mc].sort(key=lambda x: x[0])

    def add_ops_times_2_mc(
        self,
        stage_id: StageIdType,
        mc_id: McIdType,
        job_id: JobIdType,
        start_time: int,
        end_time: int,
    ) -> None:
        """Add an operation to a specific machine with explicit start and end times.

        This method directly inserts an operation into the machine's timeline at the
        specified time interval. The operation will be inserted into idle gaps if
        available, maintaining a sorted, non-overlapping schedule.

        Args:
            stage_id (StageIdType): Stage identifier
            mc_id (McIdType): Machine identifier within the stage
            job_id (JobIdType): Job identifier
            start_time (int): Start time of the operation
            end_time (int): End time of the operation

        Raises:
            ValueError: If stage_id, mc_id, or job_id is invalid
            ValueError: If job is already scheduled in the stage
            ValueError: If end_time < start_time
            ValueError: If the operation overlaps with existing operations on the machine
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")
        if job_id not in self.jobs:
            raise ValueError(f"Invalid job ID: {job_id}")
        if job_id in self.__stage_2_job_2_end_time[stage_id]:
            raise ValueError(
                f"Job ID {job_id} already scheduled in stage ID {stage_id}"
            )

        if end_time < start_time:
            raise ValueError(
                f"Invalid time interval for {job_id} in {stage_id}.{mc_id}: "
                f"start_time={start_time}, end_time={end_time}"
            )

        job_tuple_seq = self.get_job_sequence(stage_id, mc_id)
        starts = [job_tuple[0] for job_tuple in job_tuple_seq]
        insert_idx = bisect.bisect_right(starts, start_time)

        # Keep the per-machine timeline consistent (sorted, non-overlapping) with O(1) neighbor checks.
        if insert_idx > 0:
            prev_start, prev_end, prev_job = job_tuple_seq[insert_idx - 1]
            if prev_end > start_time:
                raise ValueError(
                    f"Operation overlap on {stage_id}.{mc_id}: {prev_job} "
                    f"[{prev_start}, {prev_end}) overlaps {job_id} [{start_time}, {end_time})"
                )
        if insert_idx < len(job_tuple_seq):
            next_start, next_end, next_job = job_tuple_seq[insert_idx]
            if end_time > next_start:
                raise ValueError(
                    f"Operation overlap on {stage_id}.{mc_id}: {job_id} "
                    f"[{start_time}, {end_time}) overlaps {next_job} [{next_start}, {next_end})"
                )

        job_tuple_seq.insert(insert_idx, (start_time, end_time, job_id))
        self.__stage_2_job_2_end_time[stage_id][job_id] = end_time

    # Setters - dispatching methods

    def add_operation_2_mc(
        self,
        stage_id: StageIdType,
        mc_id: McIdType,
        job_id: JobIdType,
        duration: int,
        release_t: int | None = None,
    ) -> None:
        """Add an operation to a specific machine by computing earliest start time.

        This method schedules an operation on a specified machine by finding the
        earliest feasible start time that satisfies:
        1. Previous stage precedence constraint (job cannot start before previous stage completes)
        2. Release time constraint (if provided)
        3. Machine availability (can utilize idle gaps between existing operations)

        The operation will be inserted into the earliest available gap that can
        accommodate the duration, maintaining a sorted, non-overlapping schedule.

        Args:
            stage_id (StageIdType): Stage identifier
            mc_id (McIdType): Machine identifier within the stage
            job_id (JobIdType): Job identifier
            duration (int): Duration of the operation (must be > 0)
            release_t (int | None, optional): Earliest time the operation can start.
                Defaults to None (uses previous stage end time or 0).

        Raises:
            ValueError: If stage_id, mc_id, or job_id is invalid
            ValueError: If job is already scheduled in the stage
            ValueError: If duration <= 0
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")
        if job_id not in self.jobs:
            raise ValueError(f"Invalid job ID: {job_id}")
        if job_id in self.__stage_2_job_2_end_time[stage_id]:
            raise ValueError(
                f"Job ID {job_id} already scheduled in stage ID {stage_id}"
            )

        prev_ops_end_time = self.get_prev_stage_end_time(
            stage_id, job_id, default_if_missing=0
        )
        # Adjust release time by previous operation end time
        if release_t is None or release_t < prev_ops_end_time:
            release_t = prev_ops_end_time

        # Find earliest feasible time on the specified machine (can be inside idle gaps)
        start_time = self.get_machine_earliest_start_time(
            stage_id, mc_id, duration, release_t=release_t
        )
        # Compute end time
        end_time = start_time + duration
        # Append operation
        self.add_ops_times_2_mc(stage_id, mc_id, job_id, start_time, end_time)

    def add_operation_2_stage(
        self,
        stage_id: StageIdType,
        job_id: JobIdType,
        duration: int,
        release_t: int | None = None,
    ) -> None:
        """Add an operation to a stage with automatic machine selection.

        This method schedules an operation on the best available machine in the stage.
        The machine is selected based on:
        1. Earliest available time (primary criterion)
        2. Smallest idle time if tied (secondary criterion)

        The operation will be placed in the earliest feasible time slot that satisfies:
        - Previous stage precedence constraint
        - Release time constraint (if provided)
        - Machine availability (can utilize idle gaps)

        Args:
            stage_id (StageIdType): Stage identifier
            job_id (JobIdType): Job identifier
            duration (int): Duration of the operation (must be > 0)
            release_t (int | None, optional): Earliest time the operation can start.
                Defaults to None (uses previous stage end time or 0).

        Raises:
            ValueError: If stage_id or job_id is invalid
            ValueError: If job is already scheduled in the stage
            ValueError: If duration <= 0
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if job_id not in self.jobs:
            raise ValueError(f"Invalid job ID: {job_id}")
        if job_id in self.__stage_2_job_2_end_time[stage_id]:
            raise ValueError(
                f"Job ID {job_id} already scheduled in stage ID {stage_id}"
            )

        prev_ops_end_time = self.get_prev_stage_end_time(
            stage_id, job_id, default_if_missing=0
        )
        # Adjust release time by previous operation end time
        if release_t is None or release_t < prev_ops_end_time:
            release_t = prev_ops_end_time
        # Find machine and earliest available time
        mc_id, start_time = (
            self.get_machine_and_earliest_available_time_by_start_idle_idx(
                stage_id, duration, release_t=release_t
            )
        )
        # Compute end time
        end_time = start_time + duration
        # Append operation
        self.add_ops_times_2_mc(stage_id, mc_id, job_id, start_time, end_time)

    def get_job_priority_queue_for_stage_dispatch(
        self,
        stage_id: StageIdType,
        job_id_seq: Sequence[JobIdType],
        job_2_release: Mapping[JobIdType, int] | None = None,
    ) -> list[JobIdType]:
        """Returns a priority-ordered list of job IDs for dispatching to a stage.

        The priority is determined by:
        1. Effective start time (max of previous stage end time and release time)
           - Earlier effective start time = higher priority
        2. Input sequence order (as tiebreaker)

        Args:
            stage_id (StageIdType): Stage identifier
            job_id_seq (Sequence[JobIdType]): Sequence of job identifiers to prioritize
            job_2_release (Mapping[JobIdType, int] | None, optional): Mapping from job
                ID to release time. If provided, each job's effective start time is
                calculated as max(prev_stage_end_time, release_time). Jobs not in the
                mapping use release_time=0 (can start immediately). Defaults to None.

        Returns:
            list[JobIdType]: Priority-ordered list of job identifiers (highest priority first)
        """
        job_id_2_pos = {job_id: pos for pos, job_id in enumerate(job_id_seq)}
        job_priority_queue = sorted(
            job_id_seq,
            key=lambda job_id: (
                max(
                    self.get_prev_stage_end_time(
                        stage_id, job_id, default_if_missing=0
                    ),
                    job_2_release.get(job_id, 0) if job_2_release else 0,
                ),
                job_id_2_pos[job_id],
            ),
        )
        return job_priority_queue

    def dispatch_stage_by_jobs(
        self,
        stage_id: StageIdType,
        job_id_seq: Sequence[JobIdType],
        job_2_duration: Mapping[JobIdType, int],
        job_2_release: Mapping[JobIdType, int] | None = None,
    ) -> None:
        """Dispatch multiple jobs to a stage with precedence-aware priority.

        This method schedules all jobs in the sequence to the specified stage.
        Jobs are scheduled in priority order based on:
        1. Effective start time (max of previous stage end time and release time)
        2. Input sequence order (as tiebreaker)

        This priority rule ensures that jobs ready earlier can claim earlier time slots,
        particularly important when idle gaps exist in the machine timelines.

        Args:
            stage_id (StageIdType): Stage identifier
            job_id_seq (Sequence[JobIdType]): Sequence of job identifiers to dispatch
            job_2_duration (Mapping[JobIdType, int]): Mapping from job ID to operation duration
            job_2_release (Mapping[JobIdType, int] | None, optional): Mapping from job ID to release time.
                If provided, each job's effective start time is max(prev_stage_end_time, release_time).
                Defaults to None.

        Raises:
            ValueError: If stage_id is invalid
            ValueError: If a job's duration is not provided in job_2_duration
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")

        job_priority_queue = self.get_job_priority_queue_for_stage_dispatch(
            stage_id, job_id_seq
        )

        for job_id in job_priority_queue:
            if job_id not in job_2_duration:
                raise ValueError(f"Duration for job ID {job_id} not provided")
            duration = job_2_duration[job_id]
            release_t = job_2_release[job_id] if job_2_release is not None else None
            self.add_operation_2_stage(stage_id, job_id, duration, release_t=release_t)

    def dispatch_job_by_stages(
        self,
        job_id: JobIdType,
        stage_2_duration: Mapping[StageIdType, int],
        from_stage: StageIdType | None = None,
        release_t: int | None = None,
    ) -> None:
        """Dispatch a single job through all stages in sequence.

        This method schedules a job through all stages in the order defined by
        self.stages. Each stage operation automatically respects the precedence
        constraint from the previous stage.

        Args:
            job_id (JobIdType): Job identifier
            stage_2_duration (Mapping[StageIdType, int]): Mapping from stage ID to operation duration
            from_stage (StageIdType | None, optional): Stage to start from. If provided,
                scheduling begins at this stage (skipping earlier stages). Defaults to None.
            release_t (int | None, optional): Release time for the job. All stages for
                this job will respect this release time. Defaults to None.

        Raises:
            ValueError: If job_id is invalid
            ValueError: If a stage's duration is not provided in stage_2_duration
        """
        if job_id not in self.jobs:
            raise ValueError(f"Invalid job ID: {job_id}")

        stage_iter = self.stages
        if from_stage is not None:
            from_idx = self.stages.index(from_stage)
            stage_iter = self.stages[from_idx:]

        for stage_id in stage_iter:
            if stage_id not in stage_2_duration:
                raise ValueError(f"Duration for stage ID {stage_id} not provided")
            duration = stage_2_duration[stage_id]
            self.add_operation_2_stage(stage_id, job_id, duration, release_t=release_t)

    # ============================================================================
    # Helper methods for wave batch dispatching
    # ============================================================================

    def _get_scheduled_count(self, stage_idx: int) -> int:
        """Count jobs scheduled at a specific stage index.

        Args:
            stage_idx (int): Index of the stage to count scheduled jobs.

        Returns:
            int: Number of jobs scheduled at the specified stage.
        """
        if self.__wave_batch_state is None:
            return 0
        stage_id = self.__wave_batch_state.stage_ids[stage_idx]
        return len(self.__stage_2_job_2_end_time[stage_id])

    def _push_ready(
        self,
        stage_idx: int,
        job_id: JobIdType,
        tie_breaker: int,
    ) -> None:
        """Push a job to the ready heap for a stage.

        Args:
            stage_idx (int): Index of the stage.
            job_id (JobIdType): Job ID to push.
            tie_breaker (int): Tie-breaking value (e.g., job position in original sequence).
        """
        if self.__wave_batch_state is None:
            return
        stage_id: str = self.__wave_batch_state.stage_ids[stage_idx]
        end_time: int = self.__stage_2_job_2_end_time[stage_id][job_id]
        ready_heap: list[ReadyHeapEntry] = self.__wave_batch_state.ready_heaps[
            stage_idx
        ]
        # Heap entries: (end_time, tie_breaker, job_id, stage_idx)
        heapq.heappush(ready_heap, (end_time, tie_breaker, job_id, stage_idx))

    def _promote(self, stage_idx: int, batch_size: int) -> int:
        """Promote jobs from stage i to stage i+1.

        This moves completed jobs from the current stage to the next stage
        by scheduling their operations on the next stage's machines.

        Args:
            stage_idx (int): Index of the stage to promote from.
            batch_size (int): Maximum number of jobs to promote.

        Returns:
            int: Number of jobs promoted.
        """
        if self.__wave_batch_state is None:
            return 0

        num_stages: int = len(self.__wave_batch_state.stage_ids)

        # Don't promote from the last stage
        if stage_idx >= num_stages - 1:
            return 0

        next_stage_idx = stage_idx + 1
        ready_heap: list[ReadyHeapEntry] = self.__wave_batch_state.ready_heaps[
            stage_idx
        ]
        promoted_count = 0

        while ready_heap and promoted_count < batch_size:
            entry: ReadyHeapEntry = heapq.heappop(ready_heap)
            end_time, tie_breaker, job_id, _ = entry

            # Check if this job can be promoted (has duration for next stage)
            if job_id not in self.__wave_batch_state.job_2_duration[next_stage_idx]:
                continue

            # Schedule the job on the next stage
            next_stage_id: str = self.__wave_batch_state.stage_ids[next_stage_idx]
            duration: int = self.__wave_batch_state.job_2_duration[next_stage_idx][
                job_id
            ]
            self.add_operation_2_stage(next_stage_id, job_id, duration)

            # Push to ready heap for next stage
            self._push_ready(
                next_stage_idx,
                job_id,
                self.__wave_batch_state.job_2_pos.get(job_id, 0),
            )
            promoted_count += 1

        return promoted_count

    def _feed_stage1(
        self,
        batch_jobs: list[JobIdType],
        job_2_release: Mapping[JobIdType, int] | None,
    ) -> None:
        """Inject a batch of jobs to stage 1.

        Args:
            batch_jobs: List of job IDs to inject to stage 1.
            job_2_release: Mapping from job ID to release time.
        """
        if self.__wave_batch_state is None:
            return
        stage_id: str = self.__wave_batch_state.stage_ids[0]

        for job_id in batch_jobs:
            duration: int = self.__wave_batch_state.job_2_duration[0][job_id]
            release_t: int | None = (
                job_2_release.get(job_id) if job_2_release is not None else None
            )
            self.add_operation_2_stage(stage_id, job_id, duration, release_t=release_t)
            # Push to ready heap for stage 0
            self._push_ready(
                0, job_id, self.__wave_batch_state.job_2_pos.get(job_id, 0)
            )

    # ============================================================================
    # Main dispatching method for wave batch scheduling
    # ============================================================================

    def dispatch_wave_batches(
        self,
        batch_size: int,
        job_ids: Sequence[JobIdType],
        stage_ids: Sequence[StageIdType],
        stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
        job_2_release_time: Mapping[JobIdType, int] | None = None,
        get_file_path_for_subroutine: Callable | None = None,
    ) -> None:
        """Dispatch jobs using wave batch scheduling.

        This method implements a wave batch scheduling algorithm that injects jobs
        to stage 1 in batches and allows them to propagate forward through stages
        based on completion times.

        Algorithm:
        - Iteration-based: each iteration injects B jobs to stage 1, then promotes jobs forward
        - Uses min-heap per stage to track completed jobs not yet promoted
        - Promotion depth = min(iter_no - 1, num_stages - 1)

        Args:
            job_ids: Sequence of job IDs to schedule.
            stage_ids: Sequence of stage IDs to use.
            stage_2_job_2_duration: stage -> job -> duration mapping.
            batch_size: Number of jobs to inject per iteration to stage 1.
            job_2_release_time: Optional mapping from job ID to release time.
                Defaults to None.

        Raises:
            ValueError: If batch_size <= 0.
            ValueError: If any job's duration is missing.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not job_ids:
            return
        if not stage_ids:
            return

        # Validation: Check if stage_ids is a subsequence of self.stages
        # Check elements
        stage_id_set = set(self.stages)
        for stage_id in stage_ids:
            if stage_id not in stage_id_set:
                raise ValueError(
                    f"Stage ID {stage_id} in stage_ids is not in self.stages"
                )

        # Check contiguity
        start_idx = self.stages.index(stage_ids[0])
        expected_slice = list(self.stages[start_idx : start_idx + len(stage_ids)])
        if list(stage_ids) != expected_slice:
            raise ValueError(
                "stage_ids must be a contiguous subsequence of self.stages. "
                f"Expected {expected_slice}, got {list(stage_ids)}"
            )

        # Validation: check all required durations are provided
        for stage_id in stage_ids:
            for job_id in job_ids:
                if job_id not in stage_2_job_2_duration.get(stage_id, {}):
                    raise ValueError(
                        f"Duration for job ID {job_id} at stage {stage_id} not provided"
                    )

        # Initialize wave batch state
        num_stages = len(stage_ids)
        stage_2_idx = {stage_id: i for i, stage_id in enumerate(stage_ids)}

        # Convert job_2_duration to stage_idx -> job -> duration format
        job_2_duration_by_stage_idx: dict[int, Mapping[JobIdType, int]] = {}
        for stage_id, job_2_duration in stage_2_job_2_duration.items():
            if stage_id in stage_2_idx:
                stage_idx = stage_2_idx[stage_id]
                job_2_duration_by_stage_idx[stage_idx] = job_2_duration

        self.__wave_batch_state = WaveBatchState(
            ready_heaps=[[] for _ in range(num_stages)],  # One min-heap per stage
            job_2_duration=job_2_duration_by_stage_idx,
            job_2_pos={job_id: pos for pos, job_id in enumerate(job_ids)},
            iteration_no=0,
            total_jobs=len(job_ids),
            stage_ids=stage_ids,
        )

        if get_file_path_for_subroutine is not None:
            plotter = GanttPlotter()

        # Main loop: while not all jobs in last stage
        num_jobs = len(job_ids)
        while self._get_scheduled_count(num_stages - 1) < num_jobs:
            self.__wave_batch_state.iteration_no += 1
            iter_no = self.__wave_batch_state.iteration_no

            # Feed stage 1: inject batch_size jobs
            start_idx = (iter_no - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(job_ids))
            batch_jobs = list(job_ids[start_idx:end_idx])

            # Inject batch to stage 1 (may be empty on final iterations)
            if batch_jobs:
                self._feed_stage1(batch_jobs, job_2_release_time)

            # Promote jobs forward through stages based on iteration number.
            # In iteration iter_no, jobs can propagate up to (iter_no - 1) stages forward.
            # For example:
            #   Iteration 1: jobs injected to stage 1 only (no promotion yet)
            #   Iteration 2: jobs from iter 1 can move to stage 2
            #   Iteration 3: jobs from iter 1 can move to stage 3, jobs from iter 2 can move to stage 2
            promote_depth = min(iter_no - 1, num_stages - 1)

            # Promote from stage 0 to 1, then 1 to 2, etc. up to promote_depth stages
            for stage_idx in range(promote_depth):
                self._promote(stage_idx, batch_size)

            # For debug: draw gantt chart after each iteration
            if get_file_path_for_subroutine is not None:
                output_path: Path = get_file_path_for_subroutine(f"{iter_no}_gantt.png")
                plotter.export_hybrid_flowshop_plot(
                    output_path,
                    self.get_jik_2_start_time_map(),
                    self.get_jik_2_end_time_map(),
                    job_ids,
                    stage_ids,
                )

            # Check if we made progress - if no jobs were added to last stage this iteration
            # and there are no more jobs to promote, we're done
            curr_count = self._get_scheduled_count(num_stages - 1)
            self.__wave_batch_state._prev_scheduled_last_stage = curr_count

            no_more_injection = end_idx >= num_jobs
            all_empty = all(
                len(h) == 0 for h in self.__wave_batch_state.ready_heaps[:-1]
            )
            if no_more_injection and all_empty:
                break

        # Clean up state
        self.__wave_batch_state = None

    # Setters - remove

    def remove_operations(
        self, removed_ops: set[tuple[JobIdType, StageIdType, McIdType]]
    ) -> None:
        stage_2_mc_2_job_id_set: dict[StageIdType, dict[McIdType, set[JobIdType]]] = {}
        for job_id, stage_id, mc_id in removed_ops:
            if stage_id not in stage_2_mc_2_job_id_set:
                stage_2_mc_2_job_id_set[stage_id] = {}
            if mc_id not in stage_2_mc_2_job_id_set[stage_id]:
                stage_2_mc_2_job_id_set[stage_id][mc_id] = set()
            stage_2_mc_2_job_id_set[stage_id][mc_id].add(job_id)

        for stage_id, mc_2_job_id_set in stage_2_mc_2_job_id_set.items():
            for mc_id, job_id_set in mc_2_job_id_set.items():
                job_tuple_seq: list[tuple[int, int, str]] = self.get_job_sequence(
                    stage_id, mc_id
                )
                job_seq = [job_tuple[2] for job_tuple in job_tuple_seq]
                index_to_be_removed = set()
                for job_id in job_id_set:
                    del self.__stage_2_job_2_end_time[stage_id][job_id]
                    index = job_seq.index(job_id)
                    index_to_be_removed.add(index)
                new_job_tuple_seq = [
                    job_tuple
                    for idx, job_tuple in enumerate(job_tuple_seq)
                    if idx not in index_to_be_removed
                ]
                self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id] = new_job_tuple_seq

    def make_semi_active(
        self,
        stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
        start_from_stage: StageIdType | None = None,
    ) -> None:
        """Convert to semi-active schedule by retiming operations in-place.

        A semi-active schedule is one where no operation can be started earlier
        without changing the processing order on any machine.  This method
        preserves the current machine assignments and job ordering on each
        machine, but recomputes every operation's (start, end) so that each
        operation begins at its earliest feasible time.

        For each machine with job order [j1, j2, j3, ...]:

        * `start(j1) = max(prev_stage_end(j1), 0)`
        * `start(j2) = max(prev_stage_end(j2), end(j1))`
        * `start(j3) = max(prev_stage_end(j3), end(j2))`
        * `end(jk)   = start(jk) + duration(jk)`

        Note: this is *not* an active-schedule construction that inserts
        operations into idle gaps.  It simply packs operations as tightly as
        possible while respecting the existing machine sequence.

        Args:
            stage_2_job_2_duration: stage -> job -> processing_time mapping.
            start_from_stage: If None (default), retime **all** stages from
                the first to the last.  If set to a valid stage ID, only stages
                from that stage onward are retimed; earlier stages are left
                untouched.  Precedence constraints from earlier stages are
                still respected via get_prev_stage_end_time.

        Raises:
            ValueError: If *start_from_stage* is not None and is not a
                valid stage ID.

        Post-conditions (for retimed stages):
            * Each machine's operation list is sorted by non-decreasing start
              time with no overlaps.
            * Precedence is satisfied: for every job, the start time at stage
              *k* is >= the end time at stage *k-1*.
            * ``__stage_2_job_2_end_time`` is consistent with the retimed
              tuples.
        """
        # Determine which stages to process.
        if start_from_stage is None:
            first_idx = 0
        else:
            if start_from_stage not in self.stages:
                raise ValueError(f"Invalid stage ID: {start_from_stage}")
            first_idx = self.stages.index(start_from_stage)

        for stage_idx in range(first_idx, len(self.stages)):
            stage_id = self.stages[stage_idx]
            prev_stage_id = self.stages[stage_idx - 1] if stage_idx > 0 else None
            job_2_duration = stage_2_job_2_duration[stage_id]
            job_2_prev_stage_end_time = (
                self.__stage_2_job_2_end_time[prev_stage_id]
                if prev_stage_id is not None
                else {}
            )
            mc_2_job_tuple_seq = self.__stage_2_mc_2_job_tuple_seq[stage_id]

            for mc_id in self.machines_per_stage[stage_id]:
                job_tuple_seq = mc_2_job_tuple_seq[mc_id]
                if not job_tuple_seq:
                    continue

                machine_available = 0
                new_tuple_seq: list[tuple[int, int, JobIdType]] = []

                for _, _, job_id in job_tuple_seq:
                    duration = job_2_duration[job_id]
                    release = job_2_prev_stage_end_time.get(job_id, 0)
                    start = max(release, machine_available)
                    end = start + duration

                    new_tuple_seq.append((start, end, job_id))
                    self.__stage_2_job_2_end_time[stage_id][job_id] = end
                    machine_available = end

                mc_2_job_tuple_seq[mc_id] = new_tuple_seq

    def swap_two_operations_within_stage(
        self,
        stage_id: StageIdType,
        job_id_1: JobIdType,
        job_id_2: JobIdType,
        stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
        *,
        do_make_semi_active: bool = True,
    ) -> None:
        """Swap two jobs' operations within a stage.

        Finds the operations of *job_id_1* and *job_id_2* in stage_id and
        swaps their positions:

        * **Same machine:** the two operations exchange positions in the
          machine's operation list (order swap).
        * **Different machines:** each job takes the other's slot on the
          other's machine (assignment swap).

        After the swap the (start, end) values of affected tuples are
        stale.  The `do_make_semi_active` flag controls what happens next:

        * `True` (default) -- `make_semi_active` is called starting from
          *stage_id* onward so that all (start, end) values and the
          end-time cache become consistent again.  Stages before *stage_id*
          are left untouched.
        * `False` -- only the raw element swap is performed and the
          end-time cache entries for both jobs at this stage are **removed**.
          Start/end times and the end-time map are unreliable until the
          caller retimes the schedule (e.g. by calling `make_semi_active`
          manually).

        Args:
            stage_id: Stage in which to swap the two operations.
            job_id_1: First job to swap.
            job_id_2: Second job to swap.
            stage_2_job_2_duration: stage -> job -> processing_time
                mapping, used when *do_make_semi_active* is True.
            do_make_semi_active: Whether to retime the schedule after swapping.

        Raises:
            ValueError: If *stage_id* is invalid.
            ValueError: If `job_id_1 == job_id_2`.
            ValueError: If either job is not found in the stage schedule.
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if job_id_1 == job_id_2:
            raise ValueError(
                f"Cannot swap a job with itself: job_id_1 == job_id_2 == {job_id_1}"
            )

        # Locate (machine, index) for each job in the stage.
        mc_2_job_tuple_seq = self.__stage_2_mc_2_job_tuple_seq[stage_id]

        mc1: McIdType | None = None
        idx1: int = -1
        mc2: McIdType | None = None
        idx2: int = -1

        for mc_id in self.machines_per_stage[stage_id]:
            for idx, (_, _, jid) in enumerate(mc_2_job_tuple_seq[mc_id]):
                if jid == job_id_1 and mc1 is None:
                    mc1, idx1 = mc_id, idx
                elif jid == job_id_2 and mc2 is None:
                    mc2, idx2 = mc_id, idx
            if mc1 is not None and mc2 is not None:
                break

        if mc1 is None:
            raise ValueError(f"Job ID {job_id_1} not found in stage {stage_id}")
        if mc2 is None:
            raise ValueError(f"Job ID {job_id_2} not found in stage {stage_id}")

        # Swap the job_ids in the tuples.  The (start, end) values become
        # temporary placeholders; make_semi_active will recompute them.
        seq1 = mc_2_job_tuple_seq[mc1]
        seq2 = mc_2_job_tuple_seq[mc2]
        s1, e1, _ = seq1[idx1]
        s2, e2, _ = seq2[idx2]
        seq1[idx1] = (s1, e1, job_id_2)
        seq2[idx2] = (s2, e2, job_id_1)

        if do_make_semi_active:
            self.make_semi_active(stage_2_job_2_duration, start_from_stage=stage_id)
        else:
            # Invalidate stale end-time entries for both jobs at this stage.
            self.__stage_2_job_2_end_time[stage_id].pop(job_id_1, None)
            self.__stage_2_job_2_end_time[stage_id].pop(job_id_2, None)

    # Getters - critical path

    def calculate_slack(
        self, stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]]
    ) -> dict[StageIdType, dict[JobIdType, int]]:
        """
        Calculate slack for each scheduled operation using CPM.

        Slack = Latest Start Time - Earliest Start Time
        Operations with slack = 0 are critical.

        Args:
            stage_2_job_2_duration (Mapping[StageIdType, Mapping[JobIdType, int]]):
                Stage ID -> job ID -> operation duration

        Returns:
            dict[StageIdType, dict[JobIdType, int]]: Stage ID -> job ID -> slack value
        """
        # Step 1: Forward pass (earliest times)
        earliest_start: dict[StageIdType, dict[JobIdType, int]] = {
            stage_id: {} for stage_id in self.stages
        }
        earliest_finish: dict[StageIdType, dict[JobIdType, int]] = {
            stage_id: {} for stage_id in self.stages
        }

        for stage_idx, stage_id in enumerate(self.stages):
            job_2_duration = stage_2_job_2_duration[stage_id]
            prev_stage_id = self.stages[stage_idx - 1] if stage_idx > 0 else None

            for mc_id in self.machines_per_stage[stage_id]:
                job_tuple_seq = self.get_job_sequence(stage_id, mc_id)
                prev_end_on_mc = 0

                for _start_t, _end_t, job_id in job_tuple_seq:
                    prev_stage_end = (
                        self.__stage_2_job_2_end_time[prev_stage_id].get(job_id, 0)
                        if prev_stage_id is not None
                        else 0
                    )

                    es = (
                        prev_stage_end
                        if prev_stage_end > prev_end_on_mc
                        else prev_end_on_mc
                    )
                    earliest_start[stage_id][job_id] = es
                    ef = es + job_2_duration[job_id]
                    earliest_finish[stage_id][job_id] = ef

                    # Keep machine precedence anchored to the current schedule: the next
                    # operation on this machine cannot start before this operation's
                    # scheduled completion.
                    prev_end_on_mc = ef

        all_efs = [ef for s in self.stages for ef in earliest_finish[s].values()]
        if not all_efs:
            return {}
        makespan = max(all_efs)
        if makespan == 0:
            # If the makespan is zero, all operations are critical with zero slack.
            return {
                stage_id: {job_id: 0 for job_id in earliest_start[stage_id]}
                for stage_id in self.stages
            }

        # Step 2: Backward pass (latest times)
        latest_finish: dict[StageIdType, dict[JobIdType, int]] = {
            stage_id: {} for stage_id in self.stages
        }
        latest_start: dict[StageIdType, dict[JobIdType, int]] = {
            stage_id: {} for stage_id in self.stages
        }

        for stage_idx in range(len(self.stages) - 1, -1, -1):
            stage_id = self.stages[stage_idx]
            job_2_duration = stage_2_job_2_duration[stage_id]
            next_stage_id = (
                self.stages[stage_idx + 1] if stage_idx < len(self.stages) - 1 else None
            )

            for mc_id in self.machines_per_stage[stage_id]:
                job_tuple_seq = self.get_job_sequence(stage_id, mc_id)

                # For the last operation on a machine, machine constraint is makespan.
                next_ls_on_mc = makespan

                for _start_t, _end_t, job_id in reversed(job_tuple_seq):
                    # Constraint 1: Job precedence (same job, next stage)
                    if next_stage_id is None:
                        lft_1 = makespan
                    else:
                        lft_1 = latest_start[next_stage_id].get(job_id, makespan)

                    # Constraint 2: Machine precedence (same machine, next job)
                    lft_2 = next_ls_on_mc

                    lf = lft_1 if lft_1 < lft_2 else lft_2
                    ls = lf - job_2_duration[job_id]

                    latest_finish[stage_id][job_id] = lf
                    latest_start[stage_id][job_id] = ls

                    next_ls_on_mc = ls

        # Step 3: Calculate slack
        slack: dict[StageIdType, dict[JobIdType, int]] = {
            stage_id: {} for stage_id in self.stages
        }
        for stage_id in self.stages:
            for job_id in earliest_start[stage_id]:
                slack[stage_id][job_id] = (
                    latest_start[stage_id][job_id] - earliest_start[stage_id][job_id]
                )

        return slack

    def find_critical_blocks(
        self,
        stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
        tolerance: float = 1e-9,
        include_singletons: bool = False,
    ) -> list[list[tuple[JobIdType, StageIdType, McIdType]]]:
        """
        Find all critical blocks in the schedule.

        A critical block is a maximal sequence of consecutive critical operations
        on the same machine. Critical blocks are important for neighborhood search
        algorithms in scheduling optimization.

        Args:
            stage_2_job_2_duration (Mapping[StageIdType, Mapping[JobIdType, int]]):
                Stage ID -> job ID -> operation duration
            tolerance (float, optional): Tolerance for slack comparison. Defaults to 1e-9.
            include_singletons (bool, optional): Whether to include single-operation blocks.
                Defaults to False.

        Returns:
            list[list[tuple[JobIdType, StageIdType, McIdType]]]: List of critical blocks,
            where each block is a list of operations (job_id, stage_id, machine_id)
            in execution order on the same machine.
        """
        # Calculate slack
        slack: dict[str, dict[str, int]] = self.calculate_slack(stage_2_job_2_duration)

        # Find critical operations and their machines
        # Using dict structure to avoid tuple creation

        stage_2_mc_2_jobs: dict[StageIdType, dict[McIdType, list[JobIdType]]] = {
            stage_id: {} for stage_id in self.stages
        }
        # Not for algorithm but for debugging
        stage_2_critical_job_cnt: dict[StageIdType, int] = {}

        for stage_id in self.stages:
            critical_jobs: set[JobIdType] = set()

            # Find critical jobs in this stage
            for job_id in slack.get(stage_id, {}):
                if abs(slack[stage_id][job_id]) < tolerance:
                    critical_jobs.add(job_id)
            stage_2_critical_job_cnt[stage_id] = len(critical_jobs)

            mc_2_job_tuple_seq = self.__stage_2_mc_2_job_tuple_seq[stage_id]
            job_2_end_time = self.__stage_2_job_2_end_time[stage_id]
            # Find machine for each critical job
            for mc_id in self.machines_per_stage[stage_id]:
                job_sequence: list[JobIdType] = []
                for start_t, end_t, job_id in mc_2_job_tuple_seq[
                    mc_id
                ]:  # Assumed to be already sorted by start time
                    if job_id in critical_jobs:
                        job_sequence.append(job_id)
                stage_2_mc_2_jobs[stage_id][mc_id] = job_sequence
                # if job_sequence:
                #     # Sort by end time to maintain execution order
                #     stage_2_mc_2_jobs[stage_id][mc_id] = sorted(
                #         job_sequence,
                #         key=lambda job_id: job_2_end_time[job_id],
                #     )

        # from pprint import pformat

        # logging.info(
        #     f"Critical job counts per stage:\n{pformat(stage_2_critical_job_cnt, indent=2, width=80)}"
        # )

        # Extract consecutive sequences as critical blocks
        blocks: list[list[tuple[JobIdType, StageIdType, McIdType]]] = []
        for stage_id, mc_2_job_seq in stage_2_mc_2_jobs.items():
            job_2_end_time = self.__stage_2_job_2_end_time[stage_id]
            job_2_duration = stage_2_job_2_duration[stage_id]
            for mc_id, job_seq in mc_2_job_seq.items():
                if not job_seq:
                    continue

                current_block: list[tuple[JobIdType, StageIdType, McIdType]] = []

                for i, job_id in enumerate(job_seq):
                    current_block.append((job_id, stage_id, mc_id))

                    # Check if next operation is consecutive
                    if i < len(job_seq) - 1:
                        next_job_id = job_seq[i + 1]
                        current_end_t = job_2_end_time[job_id]
                        next_start_t = (
                            job_2_end_time[next_job_id] - job_2_duration[next_job_id]
                        )

                        # If there's a gap, end current block
                        if next_start_t > current_end_t:
                            if include_singletons or len(current_block) >= 2:
                                blocks.append(current_block)
                            current_block = []
                    else:
                        # Last operation
                        if include_singletons or len(current_block) >= 2:
                            blocks.append(current_block)

        return blocks

    # Setter - shift

    def right_shift(self, shift_amount: int) -> None:
        """Right-shift the entire schedule by a specified amount.

        This method adds the shift_amount to the start and end times of all
        operations in the schedule, effectively delaying the entire schedule.

        Args:
            shift_amount (int): The amount of time to shift the schedule to the right.
                Must be non-negative.
        """
        for stage_id in self.stages:
            for mc_id in self.machines_per_stage[stage_id]:
                job_tuple_seq = self.get_job_sequence(stage_id, mc_id)
                new_job_tuple_seq = []
                for start_time, end_time, job_id in job_tuple_seq:
                    new_start = start_time + shift_amount
                    new_end = end_time + shift_amount
                    new_job_tuple_seq.append((new_start, new_end, job_id))
                    self.__stage_2_job_2_end_time[stage_id][job_id] = new_end
                self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id] = new_job_tuple_seq


# Validation functions


def validate_schedule(
    sched: HybridFlowshopLiteSchedule,
    stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
) -> None:
    """Raise ``ValueError`` if the schedule violates feasibility invariants.

    Checks three invariants in order:

    1. **Duration** -- ``end - start == duration`` for every operation.
    2. **Precedence** -- for every job, the start time at stage *k* is
       >= the end time at stage *k-1*.
    3. **No overlap** -- no two operations on the same machine overlap.

    Args:
        sched: The schedule to validate.
        stage_2_job_2_duration: ``stage -> job -> processing_time`` mapping.

    Raises:
        ValueError: If any invariant is violated.
    """
    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()
    stages = list(sched.stages)

    validate_duration(start_map, end_map, stage_2_job_2_duration)
    validate_precedence(start_map, end_map, stages)
    validate_no_overlap(start_map, end_map, stages, sched.machines_per_stage)


def validate_duration(
    start_map: Mapping[tuple[JobIdType, StageIdType, McIdType], int],
    end_map: Mapping[tuple[JobIdType, StageIdType, McIdType], int],
    stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
) -> None:
    """Raise ``ValueError`` if ``end - start != duration`` for any operation.

    Args:
        start_map: ``(job, stage, mc) -> start_time`` mapping.
        end_map: ``(job, stage, mc) -> end_time`` mapping.
        stage_2_job_2_duration: ``stage -> job -> processing_time`` mapping.

    Raises:
        ValueError: If any operation's time span does not match its duration.
    """
    for (job, stage, mc), s in start_map.items():
        e = end_map[(job, stage, mc)]
        expected = stage_2_job_2_duration[stage][job]
        if e - s != expected:
            raise ValueError(
                f"Duration mismatch: {job}@{stage}.{mc}: "
                f"end-start={e - s} != duration={expected}"
            )


def validate_precedence(
    start_map: Mapping[tuple[JobIdType, StageIdType, McIdType], int],
    end_map: Mapping[tuple[JobIdType, StageIdType, McIdType], int],
    stages: Sequence[StageIdType],
) -> None:
    """Raise ``ValueError`` if any precedence constraint is violated.

    For every job, the start time at stage *k* must be >= the end time at
    stage *k-1*.

    Args:
        start_map: ``(job, stage, mc) -> start_time`` mapping.
        end_map: ``(job, stage, mc) -> end_time`` mapping.
        stages: Ordered sequence of stage IDs.

    Raises:
        ValueError: If any job starts at a stage before completing the
            previous stage.
    """
    for idx in range(1, len(stages)):
        prev_stage = stages[idx - 1]
        cur_stage = stages[idx]
        prev_ends: dict[JobIdType, int] = {}
        for (job, st, _mc), e in end_map.items():
            if st == prev_stage:
                prev_ends[job] = e
        for (job, st, _mc), s in start_map.items():
            if st == cur_stage and job in prev_ends:
                if s < prev_ends[job]:
                    raise ValueError(
                        f"Precedence violated: {job}@{cur_stage} start={s} "
                        f"< {job}@{prev_stage} end={prev_ends[job]}"
                    )


def validate_no_overlap(
    start_map: Mapping[tuple[JobIdType, StageIdType, McIdType], int],
    end_map: Mapping[tuple[JobIdType, StageIdType, McIdType], int],
    stages: Sequence[StageIdType],
    machines_per_stage: Mapping[StageIdType, Sequence[McIdType]],
) -> None:
    """Raise ``ValueError`` if any two operations overlap on the same machine.

    Args:
        start_map: ``(job, stage, mc) -> start_time`` mapping.
        end_map: ``(job, stage, mc) -> end_time`` mapping.
        stages: Ordered sequence of stage IDs.
        machines_per_stage: ``stage -> [machine_ids]`` mapping.

    Raises:
        ValueError: If two operations on the same machine have overlapping
            time intervals.
    """
    for stage in stages:
        for mc in machines_per_stage[stage]:
            ops = sorted(
                [
                    (s, end_map[(j, st, m)])
                    for (j, st, m), s in start_map.items()
                    if st == stage and m == mc
                ],
            )
            for i in range(len(ops) - 1):
                if ops[i][1] > ops[i + 1][0]:
                    raise ValueError(
                        f"Overlap on {stage}.{mc}: {ops[i]} vs {ops[i + 1]}"
                    )


# Dispatch functions for mixed dispatch strategy


def from_job_sequence_get_schedule_mixed(
    schedule: HybridFlowshopLiteSchedule,
    job_sequence: Sequence[JobIdType],
    stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
    stage_2_head: Mapping[StageIdType, int],
    job_2_release: dict[str, int] | None = None,
    draw_gantt_per_step: bool = False,
    get_file_path_for_subroutine: Callable | None = None,
) -> HybridFlowshopLiteSchedule:
    """Mixed dispatch strategy with per-stage k values.

    Implements a hybrid dispatch strategy that combines:
    1. dispatch_job_by_stages: dispatch k jobs through all stages
    2. dispatch_stage_by_jobs: priority-based scheduling for remaining jobs

    The strategy works as follows:
    - For each stage (in order), determine the remaining jobs that have not been
      fully dispatched from some earlier stage, and sort them by priority (earliest
      previous stage end time first, then sequence order). Dispatch the first k jobs
      from this priority queue through all remaining stages using dispatch_job_by_stages.
    - Fill remaining slots at each stage using dispatch_stage_by_jobs with the
      same priority rule.
    - A job that has been dispatched through all remaining stages via
      dispatch_job_by_stages is treated as completed and skipped in subsequent stages.
      (Jobs scheduled only at a single stage via dispatch_stage_by_jobs are not
      considered completed; they will be scheduled again at later stages.)

    This allows for a flexible mix of sequence-based dispatch (for priority jobs)
    and priority-based dispatch (for remaining jobs).

    Args:
        schedule: An empty HybridFlowshopLiteSchedule to populate. The schedule
            is modified in-place and also returned.
        job_sequence: Sequence of job IDs to schedule. The order represents the
            tiebreaker priority when jobs have the same previous stage end time.
        stage_2_job_2_p: stage -> job -> processing time mapping. Must include all
            jobs and stages.
        stage_2_head: Mapping from stage_id to the number of jobs to dispatch via
            dispatch_job_by_stages at that stage. These values are applied as a
            cumulative cap across stages: once k jobs have been fully dispatched from
            earlier stages, they reduce the remaining budget for later stages.
            E.g., with 5 jobs and stage_2_head={"s1": 3, "s2": 3}, s2 effectively
            uses min(3, 5-3) = 2 since only 2 jobs remain after s1 dispatches 3 jobs.
        job_2_release: Optional mapping from job ID to release time, used for the first
            stage's priority queue (as a lower bound on when the job can start). If
            provided, the priority is max(prev_stage_end, release_time). Ignored for
            later stages since their priority is determined by previous stage completion.

    Returns:
        The populated HybridFlowshopLiteSchedule.

    Raises:
        ValueError: If schedule is not empty.
        ValueError: If stage_2_head contains an unknown stage_id or a negative value.
        ValueError: If a job's duration is missing for any stage.

    Example:
        >>> sched = HybridFlowshopLiteSchedule(
        ...     jobs=["j1", "j2", "j3"],
        ...     stages=["s1", "s2"],
        ...     machines_per_stage={"s1": ["m1"], "s2": ["m1"]}
        ... )
        >>> duration = {
        ...     "s1": {"j1": 2, "j2": 3, "j3": 2},
        ...     "s2": {"j1": 3, "j2": 2, "j3": 4}
        ... }
        >>> from_job_sequence_get_schedule_mixed(sched, ["j1", "j2", "j3"], duration, {"s1": 2})
        >>> print(sched.makespan)
    """
    # Validation: schedule must be empty (no operations already scheduled)
    if schedule.get_operation_set():
        raise ValueError(
            "schedule must be empty when calling from_job_sequence_get_schedule_mixed()"
        )

    # Validation: stage_2_head keys must be valid stages, and values must be non-negative
    for stage_id, k in stage_2_head.items():
        if stage_id not in schedule.stages:
            raise ValueError(f"Unknown stage_id in stage_2_head: {stage_id}")
        if k < 0:
            raise ValueError(
                f"stage_2_head values must be non-negative, got {k} for stage {stage_id}"
            )

    # Validation: check all required durations are provided
    for stage_id in schedule.stages:
        for job_id in job_sequence:
            if job_id not in stage_2_job_2_p.get(stage_id, {}):
                raise ValueError(
                    f"Duration for job ID {job_id} at stage {stage_id} not provided"
                )

    if get_file_path_for_subroutine is None:
        draw_gantt_per_step = False
    if draw_gantt_per_step:
        plotter = GanttPlotter()

    target_stage_list = schedule.stages

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

    return schedule
