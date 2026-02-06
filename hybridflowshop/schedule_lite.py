from __future__ import annotations

import bisect
from typing import Mapping, Sequence

JobIdType = str
StageIdType = str
McIdType = str


class HybridFlowshopLiteSchedule:
    # Parameters

    jobs: Sequence[JobIdType]
    """ID of jobs"""

    stages: Sequence[StageIdType]
    """ID of stages"""

    machines_per_stage: Mapping[StageIdType, Sequence[McIdType]]
    """map(stage ID -> ID of machines)"""

    # Variables

    __stage_2_mc_2_job_tuple_seq: dict[
        StageIdType, dict[McIdType, list[tuple[int, int, JobIdType]]]
    ]
    """map(stage ID -> map(machine ID -> sequence of (start times, end times, job IDs)))"""

    __stage_2_job_2_end_time: dict[StageIdType, dict[JobIdType, int]]
    """map(stage ID -> map(job ID -> end time))"""

    def __init__(
        self,
        jobs: Sequence[JobIdType],
        stages: Sequence[StageIdType],
        machines_per_stage: Mapping[StageIdType, Sequence[McIdType]],
    ):
        self.jobs = jobs
        self.stages = stages
        self.machines_per_stage = machines_per_stage
        self._initialize_variables()

    def _initialize_variables(self):
        self.__stage_2_mc_2_job_tuple_seq = {
            stage: {mc: [] for mc in self.machines_per_stage[stage]}
            for stage in self.stages
        }
        self.__stage_2_job_2_end_time = {stage: {} for stage in self.stages}

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

    def get_machine_latest_end_time(
        self, stage_id: StageIdType, mc_id: McIdType
    ) -> int:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")

        job_tuple_seq = self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id]
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

        job_tuple_seq = self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id]
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

        stage_index = self.stages.index(stage_id)
        if stage_index == 0:
            return 0

        prev_stage_id = self.stages[stage_index - 1]
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

    def get_start_time_map(
        self,
    ) -> dict[tuple[JobIdType, StageIdType, McIdType], int]:
        result: dict[tuple[JobIdType, StageIdType, McIdType], int] = {}
        for stage in self.stages:
            for mc in self.machines_per_stage[stage]:
                for job_tuple in self.__stage_2_mc_2_job_tuple_seq[stage][mc]:
                    result[(job_tuple[2], stage, mc)] = job_tuple[0]
        return result

    def get_end_time_map(
        self,
    ) -> dict[tuple[JobIdType, StageIdType, McIdType], int]:
        result: dict[tuple[JobIdType, StageIdType, McIdType], int] = {}
        for stage in self.stages:
            for mc in self.machines_per_stage[stage]:
                for job_tuple in self.__stage_2_mc_2_job_tuple_seq[stage][mc]:
                    result[(job_tuple[2], stage, mc)] = job_tuple[1]
        return result

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

        job_tuple_seq = self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id]
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

    def dispatch_stage_by_jobs(
        self,
        stage_id: StageIdType,
        job_id_seq: Sequence[JobIdType],
        job_2_duration: Mapping[JobIdType, int],
    ) -> None:
        """Dispatch multiple jobs to a stage with precedence-aware priority.

        This method schedules all jobs in the sequence to the specified stage.
        Jobs are scheduled in priority order based on:
        1. Previous stage completion time (earlier completion = higher priority)
        2. Input sequence order (as tiebreaker)

        This priority rule ensures that jobs ready earlier can claim earlier time slots,
        particularly important when idle gaps exist in the machine timelines.

        Args:
            stage_id (StageIdType): Stage identifier
            job_id_seq (Sequence[JobIdType]): Sequence of job identifiers to dispatch
            job_2_duration (Mapping[JobIdType, int]): Mapping from job ID to operation duration

        Raises:
            ValueError: If stage_id is invalid
            ValueError: If a job's duration is not provided in job_2_duration
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")

        # Priority rule:
        # 1) jobs with smaller previous-stage completion time are scheduled earlier
        # 2) if tied, preserve the input order of `job_id_seq`
        job_id_2_pos = {job_id: pos for pos, job_id in enumerate(job_id_seq)}
        job_priority_queue = sorted(
            job_id_seq,
            key=lambda job_id: (
                self.get_prev_stage_end_time(stage_id, job_id, default_if_missing=0),
                job_id_2_pos[job_id],
            ),
        )

        for job_id in job_priority_queue:
            if job_id not in job_2_duration:
                raise ValueError(f"Duration for job ID {job_id} not provided")
            duration = job_2_duration[job_id]
            self.add_operation_2_stage(stage_id, job_id, duration)

    def dispatch_job_by_stages(
        self, job_id: JobIdType, stage_2_duration: Mapping[StageIdType, int]
    ) -> None:
        """Dispatch a single job through all stages in sequence.

        This method schedules a job through all stages in the order defined by
        self.stages. Each stage operation automatically respects the precedence
        constraint from the previous stage.

        Args:
            job_id (JobIdType): Job identifier
            stage_2_duration (Mapping[StageIdType, int]): Mapping from stage ID to operation duration

        Raises:
            ValueError: If job_id is invalid
            ValueError: If a stage's duration is not provided in stage_2_duration
        """
        if job_id not in self.jobs:
            raise ValueError(f"Invalid job ID: {job_id}")

        for stage_id in self.stages:
            if stage_id not in stage_2_duration:
                raise ValueError(f"Duration for stage ID {stage_id} not provided")
            duration = stage_2_duration[stage_id]
            self.add_operation_2_stage(stage_id, job_id, duration)

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
                job_tuple_seq: list[tuple[int, int, str]] = (
                    self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id]
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
