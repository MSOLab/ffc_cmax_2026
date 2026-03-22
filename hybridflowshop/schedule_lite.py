from __future__ import annotations

import bisect
import logging
from typing import Iterable, Iterator, Mapping, Sequence

JobIdType = str
StageIdType = str
McIdType = str
OperationType = tuple[JobIdType, StageIdType, McIdType]


class HybridFlowshopLiteSchedule:
    # Parameters

    jobs: Sequence[JobIdType]
    """ID of all jobs (scheduled or not)"""

    stages: Sequence[StageIdType]
    """ID of stages"""

    machines_per_stage: Mapping[StageIdType, Sequence[McIdType]]
    """map(stage ID -> ID of machines)"""

    # Helper parameters

    stage_2_index: Mapping[StageIdType, int]
    """map(stage ID -> stage index in self.stages)"""

    stage_2_prev_stage: Mapping[StageIdType, StageIdType | None]
    """map(stage ID -> previous stage ID or None if first stage)"""

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
        self.stage_2_index = {stage: i for i, stage in enumerate(stages)}
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

    def deepcopy(
        self, job_subsequence: set[JobIdType] | None = None
    ) -> HybridFlowshopLiteSchedule:
        """Return a deep-copied schedule, optionally filtered to a job subset.

        When ``job_subsequence`` is ``None``, this returns a full deep copy of the
        schedule. The returned instance does not share mutable containers with the
        original, including schedule metadata such as ``jobs``, ``stages``, and
        ``machines_per_stage``.

        When ``job_subsequence`` is provided, the returned schedule keeps the full
        job metadata but only copies cached schedule state and machine operation
        tuples for jobs in that subset. Empty stage/machine containers are preserved.
        """
        new_instance = HybridFlowshopLiteSchedule(
            jobs=list(self.jobs),
            stages=list(self.stages),
            machines_per_stage={
                stage: list(self.machines_per_stage[stage]) for stage in self.stages
            },
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

    def as_reversed(self) -> HybridFlowshopLiteSchedule:
        """Return a reversed-time schedule with reversed stage order.

        This converts the current (forward) schedule into a reversed schedule:
        - The ``stages`` list order is reversed (e.g., ``["s1", "s2"]`` → ``["s2", "s1"]``).
        - For each operation with ``(start_orig, end_orig)``, times are transformed by:
          ``start_rev = makespan - end_orig``, ``end_rev = makespan - start_orig``.
        - Stage IDs and machine IDs are preserved as-is; operations are added using
          their original ``(stage_id, mc_id)`` keys.

        The current instance is not modified.
        """
        makespan = self.makespan

        new_sched = HybridFlowshopLiteSchedule(
            jobs=self.jobs,
            stages=list(reversed(self.stages)),
            machines_per_stage={
                stage: list(mcs) for stage, mcs in self.machines_per_stage.items()
            },
        )

        for stage, mc, start, end, job in self._iter_operations():
            new_sched.add_ops_times_2_mc(
                stage_id=stage,
                mc_id=mc,
                job_id=job,
                start_time=makespan - end,
                end_time=makespan - start,
            )

        return new_sched

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

        Args:
            stage_id (StageIdType): Target stage ID
            mc_id (McIdType): Target machine ID within the stage
            duration (int): Duration of the operation to be scheduled
            release_t (int | None, optional): Earliest time the operation can start.
                Defaults to None (uses previous stage end time or 0).
            after_last (bool, optional): If True, return the latest end time of the
                machine. Defaults to False.

        Raises:
            ValueError: If stage_id or mc_id is invalid,
                or if duration is not positive.

        Returns:
            int: The earliest feasible start time on the machine for the duration
                given the release time and existing scheduled operations.
                If after_last is True, ignores any gaps between scheduled operations
                and returns the later of the machine's latest end time and the
                release time.
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        prev_end = release_t if release_t is not None else 0

        if after_last:
            makespan = self.get_machine_latest_end_time(stage_id, mc_id)
            return makespan if makespan >= prev_end else prev_end

        job_tuple_seq = self.get_job_sequence(stage_id, mc_id)
        if not job_tuple_seq:
            return prev_end

        starts = [op[0] for op in job_tuple_seq]
        idx = bisect.bisect_left(starts, prev_end)

        # If prev_end falls inside the previous operation, advance past it
        if idx > 0 and prev_end < job_tuple_seq[idx - 1][1]:
            prev_end = job_tuple_seq[idx - 1][1]
            # idx now points to the next operation after the overlap

        # Scan forward for a gap that fits
        while idx < len(job_tuple_seq):
            if prev_end + duration <= job_tuple_seq[idx][0]:
                return prev_end
            prev_end = job_tuple_seq[idx][1]
            idx += 1

        return prev_end

    def get_eat_for_machine(
        self,
        stage_id: StageIdType,
        mc: McIdType,
        duration: int,
        release_t: int | None = None,
    ) -> tuple[int, int]:
        """Compute (EAT, idle) for a single machine.

        Args:
            stage_id (StageIdType): Target stage ID
            mc (McIdType): Target machine ID
            duration (int): Duration of the operation to be scheduled
            release_t (int | None, optional): Earliest time the operation can start.
                Defaults to None (uses previous stage end time or 0).

        Returns:
            tuple[int, int]: (earliest available time, idle time)
        """
        eat = self.get_machine_earliest_start_time(
            stage_id, mc, duration, release_t=release_t
        )
        idle = max(eat - self.get_machine_latest_end_time(stage_id, mc), 0)
        return eat, idle

    def select_machine_by_earliest_start_then_idle(
        self, stage_id: StageIdType, duration: int, release_t: int | None = None
    ) -> tuple[McIdType, int]:
        """Select the best machine in a stage and return its earliest available time.

        Machines are compared lexicographically by:
            1. Earliest available time (EAT): smaller is better.
            2. Idle time (EAT - machine's latest end time, clamped to 0): smaller
               is better, preferring machines that have been busy more recently.

        Args:
            stage_id (StageIdType): Target stage.
            duration (int): Processing time of the operation.
            release_t (int | None, optional): Earliest time the operation can start.
                Defaults to None (uses previous stage end time or 0).

        Raises:
            ValueError: If stage_id is invalid, the stage has no machines,
                or duration is not positive.

        Returns:
            tuple[McIdType, int]: (selected machine ID, earliest available time)
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if not self.machines_per_stage[stage_id]:
            raise ValueError(f"No machines available in stage {stage_id}.")
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        # Initialize with first machine's values
        best_mc = self.machines_per_stage[stage_id][0]
        best_eat, best_idle = self.get_eat_for_machine(
            stage_id, best_mc, duration, release_t
        )

        for mc in self.machines_per_stage[stage_id][1:]:
            eat, idle = self.get_eat_for_machine(stage_id, mc, duration, release_t)

            if (eat, idle) < (best_eat, best_idle):
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

    def get_operation_set(self) -> set[OperationType]:
        return {
            (job_id, stage, mc) for stage, mc, _, _, job_id in self._iter_operations()
        }

    def get_jik_2_start_time_map(self) -> dict[OperationType, int]:
        return {
            (job_id, stage, mc): int(start)
            for stage, mc, start, _, job_id in self._iter_operations()
        }

    def get_jik_2_end_time_map(self) -> dict[OperationType, int]:
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

    def get_stage_2_mc_2_idle_time_map(
        self, include_idle_before_first_op: bool = False
    ) -> dict[StageIdType, dict[McIdType, int]]:
        stage_2_mc_2_idle_time = {
            stage: {mc: 0 for mc in self.machines_per_stage[stage]}
            for stage in self.stages
        }
        for stage_id in self.stages:
            for mc_id in self.machines_per_stage[stage_id]:
                former_end_time: int | None = None
                for start_time, end_time, job_id in self.get_job_sequence(
                    stage_id, mc_id
                ):
                    if former_end_time is None:
                        if include_idle_before_first_op:
                            former_end_time = 0
                        else:
                            former_end_time = start_time
                    idle_time = start_time - former_end_time
                    if idle_time > 0:
                        stage_2_mc_2_idle_time[stage_id][mc_id] += idle_time
                    former_end_time = end_time

        return stage_2_mc_2_idle_time

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
        mc_id, start_time = self.select_machine_by_earliest_start_then_idle(
            stage_id, duration, release_t=release_t
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

    def _get_next_stage_start_time(
        self,
        stage_id: StageIdType,
        job_id: JobIdType,
        default_if_missing: int | None = None,
    ) -> int:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")

        stage_idx = self.stage_2_index[stage_id]
        if stage_idx >= len(self.stages) - 1:
            if default_if_missing is not None:
                return default_if_missing
            raise ValueError(f"Stage ID {stage_id} has no next stage")

        next_stage_id = self.stages[stage_idx + 1]
        for mc_id in self.machines_per_stage[next_stage_id]:
            for start_time, _end_time, scheduled_job_id in self.get_job_sequence(
                next_stage_id, mc_id
            ):
                if scheduled_job_id == job_id:
                    return start_time

        if default_if_missing is not None:
            return default_if_missing
        raise ValueError(f"Job ID {job_id} not found in next stage after {stage_id}")

    def _get_latest_feasible_slot_on_machine(
        self,
        stage_id: StageIdType,
        mc_id: McIdType,
        duration: int,
        upper_bound: int,
    ) -> tuple[int, int]:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if mc_id not in self.machines_per_stage[stage_id]:
            raise ValueError(f"Invalid machine ID: {mc_id} for stage ID: {stage_id}")
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")
        if upper_bound < duration:
            raise ValueError(
                f"No feasible slot on {stage_id}.{mc_id} before {upper_bound}"
            )

        job_tuple_seq = self.get_job_sequence(stage_id, mc_id)
        next_end = upper_bound

        for start_time, end_time, _job_id in reversed(job_tuple_seq):
            gap_end = next_end
            gap_start = end_time
            if gap_end - gap_start >= duration:
                latest_end = gap_end
                return latest_end - duration, latest_end
            next_end = min(next_end, start_time)

        if next_end >= duration:
            return next_end - duration, next_end

        raise ValueError(f"No feasible slot on {stage_id}.{mc_id} before {upper_bound}")

    def dispatch_stage_reversed_by_jobs(
        self,
        stage_id: StageIdType,
        job_id_seq: Sequence[JobIdType],
        job_2_duration: Mapping[JobIdType, int],
        mc_2_lct: Mapping[McIdType, int],
        *,
        job_2_deadline: Mapping[JobIdType, int] | None = None,
    ) -> None:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")

        mc_2_index = {
            mc_id: idx for idx, mc_id in enumerate(self.machines_per_stage[stage_id])
        }

        for job_id in job_id_seq:
            if job_id not in job_2_duration:
                raise ValueError(f"Duration for job ID {job_id} not provided")

            duration = job_2_duration[job_id]
            next_stage_start = (
                self._get_next_stage_start_time(
                    stage_id, job_id, default_if_missing=None
                )
                if self.stage_2_index[stage_id] < len(self.stages) - 1
                else None
            )
            job_deadline = (
                job_2_deadline[job_id]
                if job_2_deadline is not None and job_id in job_2_deadline
                else None
            )

            best_candidate: tuple[int, int, int, McIdType] | None = None
            machine_failure_info: dict[McIdType, dict[str, object]] = {}
            for mc_id, mc_lct in mc_2_lct.items():
                if mc_id not in mc_2_index:
                    raise ValueError(
                        f"Invalid machine ID: {mc_id} for stage ID: {stage_id}"
                    )

                upper_bound = mc_lct
                if job_deadline is not None:
                    upper_bound = min(upper_bound, job_deadline)
                if next_stage_start is not None:
                    upper_bound = min(upper_bound, next_stage_start)

                try:
                    start_time, end_time = self._get_latest_feasible_slot_on_machine(
                        stage_id=stage_id,
                        mc_id=mc_id,
                        duration=duration,
                        upper_bound=upper_bound,
                    )
                except ValueError:
                    machine_failure_info[mc_id] = {
                        "mc_lct": mc_lct,
                        "upper_bound": upper_bound,
                        "job_sequence": self.get_job_sequence(stage_id, mc_id),
                    }
                    continue

                candidate = (start_time, end_time, -mc_2_index[mc_id], mc_id)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate

            if best_candidate is None:
                logging.error(
                    "Reverse dispatch failed for %s on %s: duration=%s "
                    "job_deadline=%s next_stage_start=%s mc_2_lct=%s "
                    "machine_failure_info=%s",
                    job_id,
                    stage_id,
                    duration,
                    job_deadline,
                    next_stage_start,
                    dict(mc_2_lct),
                    machine_failure_info,
                )
                raise ValueError(
                    f"Unable to reverse-dispatch job {job_id} on stage {stage_id}"
                )

            start_time, end_time, _neg_mc_idx, target_mc_id = best_candidate
            self.add_ops_times_2_mc(
                stage_id, target_mc_id, job_id, start_time, end_time
            )

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
            from_idx = self.stage_2_index[from_stage]
            stage_iter = self.stages[from_idx:]

        for stage_id in stage_iter:
            if stage_id not in stage_2_duration:
                raise ValueError(f"Duration for stage ID {stage_id} not provided")
            duration = stage_2_duration[stage_id]
            self.add_operation_2_stage(stage_id, job_id, duration, release_t=release_t)

    def get_job_2_palmer_index(
        self,
        stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
        jobs: Iterable[JobIdType] | None = None,
        from_stage: str | None = None,
    ) -> dict[JobIdType, int]:
        """Get Palmer's slope index for each job.

        Returns:
            dict[JobIdType, int]: Job ID -> Palmer's slope index
        """
        # TODO: overlap with hybridflowshop/dispatcher/base.py's get_palmer_sequence
        if not jobs:
            _job_id_list = self.jobs
        else:
            for job_id in jobs:
                if job_id not in self.jobs:
                    raise ValueError(f"Invalid job ID: {job_id}")
            _job_id_list = list(jobs)
        if not from_stage:
            _stage_id_list = self.stages
        else:
            if from_stage not in self.stages:
                raise ValueError(f"Invalid from_stage: {from_stage}")
            from_idx = self.stage_2_index[from_stage]
            _stage_id_list = self.stages[from_idx:]
        m = len(_stage_id_list)

        job_2_index: dict[JobIdType, int] = {}
        for job_id in _job_id_list:
            stage_2_p = {
                stage_id: stage_2_job_2_p.get(stage_id, {}).get(job_id, 0)
                for stage_id in _stage_id_list
            }
            job_2_index[job_id] = sum(
                (m - 2 * (stage_idx + 1) + 1) * stage_2_p[stage_id]
                for stage_idx, stage_id in enumerate(_stage_id_list)
            )
        return job_2_index

    def get_job_2_gupta_index(
        self,
        stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
        jobs: Iterable[JobIdType] | None = None,
        from_stage: str | None = None,
    ) -> dict[JobIdType, float]:
        """Get Gupta's index for each job.

        The index is calculated as:
        sign(p_{1j} - p_{mj}) / min_{k=1,...,m-1}(p_{kj} + p_{k+1,j})

        Where 1 is the from_stage and m is the last stage.

        Returns:
            dict[JobIdType, float]: Job ID -> Gupta's index
        """
        # TODO: overlap with hybridflowshop/dispatcher/base.py's get_gupta_sequence
        if not jobs:
            _job_id_list = self.jobs
        else:
            for job_id in jobs:
                if job_id not in self.jobs:
                    raise ValueError(f"Invalid job ID: {job_id}")
            _job_id_list = list(jobs)
        if not from_stage:
            _stage_id_list = self.stages
        else:
            if from_stage not in self.stages:
                raise ValueError(f"Invalid from_stage: {from_stage}")
            from_idx = self.stage_2_index[from_stage]
            _stage_id_list = self.stages[from_idx:]
        if len(_stage_id_list) == 1:
            # Gupta's index originally requires more than two stages;
            # If single stage, return processing time of the stage
            return {
                job_id: stage_2_job_2_p.get(_stage_id_list[0], {}).get(job_id, 0)
                for job_id in _job_id_list
            }

        job_2_index: dict[JobIdType, float] = {}
        for job_id in _job_id_list:
            stage_2_p = {
                stage_id: stage_2_job_2_p.get(stage_id, {}).get(job_id, 0)
                for stage_id in _stage_id_list
            }
            p_first = stage_2_p[_stage_id_list[0]]
            p_last = stage_2_p[_stage_id_list[-1]]
            # sign: +1 if p_last <= p_first, else -1
            sign = 1 if p_last <= p_first else -1

            min_sum = min(
                stage_2_p[_stage_id_list[k]] + stage_2_p[_stage_id_list[k + 1]]
                for k in range(len(_stage_id_list) - 1)
            )
            if min_sum == 0:
                job_2_index[job_id] = float("inf")
            else:
                job_2_index[job_id] = sign / min_sum
        return job_2_index

    def machine_centric_dispatch_4(
        self,
        stage_id: StageIdType,
        job_id_seq: Sequence[JobIdType],
        stage_2_job_2_p: Mapping[StageIdType, Mapping[JobIdType, int]],
        job_2_release: Mapping[JobIdType, int] | None = None,
        use_palmer_index: bool = False,
    ) -> None:
        """Machine-centric dispatching (v4): dispatch jobs on a single stage with idle-gap awareness.

        This implements the algorithm described in `dispatch_stage_by_machines_4` doc:
        - Build per-machine idle gap lists from the *existing* schedule on this stage.
        - Maintain machine cursors (t_k) that always point inside an idle gap (or at its start).
        - Iteratively:
            1) Move released jobs into candidate set J'
            2) Pick a target machine in ascending (t_k, gap_len, machine_index) order
               that can fit at least one candidate job in its current leading gap.
            3) Select a job by sort_key = (-(tr_j + alpha*p_j), tiebreaker_sign*p_j, j)
            4) Dispatch at start=t_k and shrink the current gap start to the job end.
            5) If no machine can fit any candidate job, do a time jump:
               - Prefer jump to next release time if it exists
               - Otherwise jump to next idle gap on every machine
        """
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if not job_id_seq:
            return
        if stage_id not in stage_2_job_2_p:
            raise ValueError(f"Missing processing time map for stage_id={stage_id}")

        mc_list = list(self.machines_per_stage[stage_id])
        if not mc_list:
            return

        # -------------------------
        # Precompute job ordering J, r_j, p_j, tr_j
        # -------------------------
        job_id_2_pos = {job_id: pos for pos, job_id in enumerate(job_id_seq)}
        stage_idx = self.stage_2_index[stage_id]
        remaining_stages = self.stages[stage_idx + 1 :]
        is_last_stage = stage_id == self.stages[-1]

        # release time r(job) := max(prev_stage_end, external_release)
        job_id_2_r: dict[JobIdType, int] = {}
        for job_id in job_id_seq:
            prev_end = self.get_prev_stage_end_time(
                stage_id, job_id, default_if_missing=0
            )
            ext_r = job_2_release.get(job_id, 0) if job_2_release is not None else 0
            job_id_2_r[job_id] = max(prev_end, ext_r)

        # Sort by (r_j asc, input position asc)
        _job_id_seq = sorted(
            job_id_seq, key=lambda jid: (job_id_2_r[jid], job_id_2_pos[jid])
        )
        n = len(_job_id_seq)
        J = list(range(n))
        j_2_job_id = {j: jid for j, jid in enumerate(_job_id_seq)}

        r_j = {j: job_id_2_r[j_2_job_id[j]] for j in J}

        p_j: dict[int, int] = {}
        tr_j: dict[int, int] = {}
        for j in J:
            jid = j_2_job_id[j]
            if jid not in stage_2_job_2_p[stage_id]:
                raise ValueError(
                    f"Duration for job {jid} at stage {stage_id} not provided"
                )
            p = stage_2_job_2_p[stage_id][jid]
            if p <= 0:
                raise ValueError(
                    f"Invalid processing time p={p} for job {jid} at stage {stage_id}"
                )
            p_j[j] = p
            tr_j[j] = sum(stage_2_job_2_p[s].get(jid, 0) for s in remaining_stages)

        # -------------------------
        # Infinity end for idle gaps
        # -------------------------
        p_max: int = max(p_j.values()) if p_j else 0
        r_max: int = max(r_j.values()) if r_j else 0
        max_existing_end = 0
        for mc in mc_list:
            seq = self.get_job_sequence(stage_id, mc)
            if seq:
                max_existing_end = max(max_existing_end, max(e for _, e, _ in seq))
        inf_end = max(r_max, max_existing_end) + n * p_max + 1

        # -------------------------
        # Machine state: gaps, cursor, gap pointer
        # -------------------------
        mc_2_gaps: dict[McIdType, list[list[int]]] = {}
        mc_2_gidx: dict[McIdType, int] = {}
        t_k: dict[McIdType, int] = {}
        e_k: dict[McIdType, int] = {}

        r0 = r_j[0]
        for mc in mc_list:
            ops = self.get_job_sequence(stage_id, mc)
            gaps = _build_idle_gaps_from_ops(ops, inf_end)
            gidx = _find_gap_index(gaps, r0, start_from=0)
            # clamp gap start if r0 is inside the gap
            gaps[gidx][0] = max(gaps[gidx][0], r0)
            mc_2_gaps[mc] = gaps
            mc_2_gidx[mc] = gidx
            t_k[mc] = max(r0, gaps[gidx][0])
            e_k[mc] = gaps[gidx][1]

        tp = min(t_k.values())
        mc_2_index = {mc: i for i, mc in enumerate(mc_list)}

        # -------------------------
        # Job selection key (as in doc)
        # -------------------------
        c = len(self.stages)

        tiebreaker_sign = -1 if is_last_stage else 1  # -1 for LPT, +1 for SPT

        # Precompute job sort keys to avoid repeated calculations
        func_j: dict[int, float] = {}
        if use_palmer_index:
            job_id_2_func = self.get_job_2_palmer_index(
                stage_2_job_2_p, jobs=job_id_seq, from_stage=stage_id
            )
            func_j = {j: job_id_2_func[job_id] for j, job_id in j_2_job_id.items()}
        else:
            p_multiplier = -(c - stage_idx - 2) * c / 80
            for j in J:
                func_j[j] = -(tr_j[j] + p_multiplier * p_j[j])
        # Debug: temporary override with Gupta index
        # job_id_2_func = self.get_job_2_gupta_index(
        #     stage_2_job_2_p, jobs=job_id_seq, from_stage=stage_id
        # )
        # func_j = {j: job_id_2_func[job_id] for j, job_id in j_2_job_id.items()}

        # Debug: temporary override with max index
        # if remaining_stages:
        #     func_j = {
        #         j: -max(stage_2_job_2_p[s][j_2_job_id[j]] for s in remaining_stages)
        #         for j in J
        #     }
        # else:
        #     func_j = {j: 0 for j in J}

        def job_sort_key(j: int) -> tuple:
            return (func_j[j], tiebreaker_sign * p_j[j], j)

        # -------------------------
        # Job state
        # -------------------------
        unscheduled_jobs: list[int] = list(J)  # J''
        candid_jobs: list[int] = []  # J'
        u = 0

        # Main loop
        while unscheduled_jobs or candid_jobs:
            # 3.1 Update J'
            if unscheduled_jobs:
                if tp >= r_j[u]:
                    # v := max j in J'' with r_j <= tp (contiguous prefix due to sorting)
                    v = u
                    for j in unscheduled_jobs:
                        if r_j[j] <= tp:
                            v = j
                        else:
                            break
                    candid_jobs.extend(range(u, v + 1))
                    removed = v - u + 1
                    unscheduled_jobs = unscheduled_jobs[removed:]
                    u = v + 1
                else:
                    if not candid_jobs:
                        # Jump tp to the next release
                        tp = r_j[unscheduled_jobs[0]]
                        # Re-align each machine cursor to the idle gap that covers or follows tp
                        for mc in mc_list:
                            gaps = mc_2_gaps[mc]
                            gidx = _find_gap_index(gaps, tp, start_from=mc_2_gidx[mc])
                            mc_2_gidx[mc] = gidx
                            gaps[gidx][0] = max(gaps[gidx][0], tp)
                            t_k[mc] = max(tp, gaps[gidx][0])
                            e_k[mc] = gaps[gidx][1]
                        tp = min(t_k.values())
                        continue

            # 3.2 Dispatch: find target machine that can fit someone in its leading gap
            ordered_mcs = sorted(
                mc_list,
                key=lambda mc: (t_k[mc], e_k[mc] - t_k[mc], mc_2_index[mc]),
            )

            target_mc = None
            candid_gap: list[int] = []
            for mc in ordered_mcs:
                gap_len = e_k[mc] - t_k[mc]
                if gap_len <= 0:
                    continue
                feasible = [j for j in candid_jobs if p_j[j] <= gap_len]
                if feasible:
                    target_mc = mc
                    candid_gap = feasible
                    break

            if target_mc is None:
                # No machine can fit any candidate job in current leading gaps.
                # Prefer release jump if there exists a future release.
                future_releases = [r_j[j] for j in unscheduled_jobs if r_j[j] > tp]
                if future_releases:
                    tp = min(future_releases)
                    for mc in mc_list:
                        gaps = mc_2_gaps[mc]
                        gidx = _find_gap_index(gaps, tp, start_from=mc_2_gidx[mc])
                        mc_2_gidx[mc] = gidx
                        gaps[gidx][0] = max(gaps[gidx][0], tp)
                        t_k[mc] = max(tp, gaps[gidx][0])
                        e_k[mc] = gaps[gidx][1]
                    tp = min(t_k.values())
                    continue

                # Otherwise idle jump: move each machine to its next gap
                for mc in mc_list:
                    gaps = mc_2_gaps[mc]
                    gidx = mc_2_gidx[mc] + 1
                    if gidx >= len(gaps):
                        # Defensive: keep at last gap
                        gidx = len(gaps) - 1
                    mc_2_gidx[mc] = gidx
                    t_k[mc] = gaps[gidx][0]
                    e_k[mc] = gaps[gidx][1]
                tp = min(t_k.values())
                continue

            # 3.2.2 Select job
            jp = min(candid_gap, key=job_sort_key)

            # 3.2.3 Dispatch & update
            start_time = t_k[target_mc]
            end_time = start_time + p_j[jp]
            self.add_ops_times_2_mc(
                stage_id, target_mc, j_2_job_id[jp], start_time, end_time
            )

            # Update job state
            candid_jobs.remove(jp)

            # Shrink the current gap start to the job end
            gaps = mc_2_gaps[target_mc]
            gidx = mc_2_gidx[target_mc]
            gaps[gidx][0] = end_time

            # If gap becomes empty, advance to the next one
            if gaps[gidx][0] >= gaps[gidx][1]:
                gidx = _find_gap_index(gaps, end_time, start_from=gidx + 1)
                mc_2_gidx[target_mc] = gidx
                gaps[gidx][0] = max(gaps[gidx][0], end_time)
            else:
                # Still within same gap
                mc_2_gidx[target_mc] = gidx

            t_k[target_mc] = max(end_time, gaps[mc_2_gidx[target_mc]][0])
            e_k[target_mc] = gaps[mc_2_gidx[target_mc]][1]

            tp = min(t_k.values())

    # Setters - remove

    def remove_operations(self, removed_ops: set[OperationType]) -> None:
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
                # Remove from end time cache
                new_job_tuple_seq = [
                    job_tuple
                    for job_tuple in job_tuple_seq
                    if job_tuple[2] not in job_id_set
                ]
                for job_tuple in job_tuple_seq:
                    if job_tuple[2] in job_id_set:
                        self.__stage_2_job_2_end_time[stage_id].pop(job_tuple[2], None)
                self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id] = new_job_tuple_seq

    def remove_jobs(self, job_ids: set[JobIdType]) -> None:
        """Remove all operations for the given set of job IDs.

        This method removes all operations (across all stages and machines)
        for each job ID in the provided set. It updates both the job sequence
        data structure and the job end time cache.

        Args:
            job_ids: A set of job IDs to remove from the schedule.
                If a job ID is not found in the schedule, it is silently ignored.
                If an empty set is provided, the schedule remains unchanged.

        Example:
            >>> sched = HybridFlowshopLiteSchedule(
            ...     jobs=["j1", "j2", "j3"],
            ...     stages=["s1", "s2"],
            ...     machines_per_stage={"s1": ["m1"], "s2": ["m1"]}
            ... )
            >>> # ... schedule operations ...
            >>> sched.remove_jobs({"j1", "j3"})  # Remove j1 and j3 completely
        """
        if not job_ids:
            return

        for stage_id in self.stages:
            for mc_id in self.machines_per_stage[stage_id]:
                job_tuple_seq: list[tuple[int, int, JobIdType]] = self.get_job_sequence(
                    stage_id, mc_id
                )
                # Filter out jobs that are in the removal set
                new_job_tuple_seq = [
                    job_tuple
                    for job_tuple in job_tuple_seq
                    if job_tuple[2] not in job_ids
                ]
                # Remove from end time cache
                for job_tuple in job_tuple_seq:
                    if job_tuple[2] in job_ids:
                        self.__stage_2_job_2_end_time[stage_id].pop(job_tuple[2], None)
                self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id] = new_job_tuple_seq

    # Setters - retiming

    def _is_selected_operation(
        self,
        operation_set: set[OperationType] | frozenset[OperationType],
        stage_id: StageIdType,
        mc_id: McIdType,
        job_id: JobIdType,
    ) -> bool:
        return not operation_set or (job_id, stage_id, mc_id) in operation_set

    def _rebuild_stage_end_time_cache(self, stage_id: StageIdType) -> None:
        job_2_end_time: dict[JobIdType, int] = {}
        for mc_id in self.machines_per_stage[stage_id]:
            for _, end_time, job_id in self.__stage_2_mc_2_job_tuple_seq[stage_id][
                mc_id
            ]:
                job_2_end_time[job_id] = end_time
        self.__stage_2_job_2_end_time[stage_id] = job_2_end_time

    def _get_stage_job_2_start_time(
        self, stage_id: StageIdType
    ) -> dict[JobIdType, int]:
        job_2_start_time: dict[JobIdType, int] = {}
        for mc_id in self.machines_per_stage[stage_id]:
            for start_time, _, job_id in self.__stage_2_mc_2_job_tuple_seq[stage_id][
                mc_id
            ]:
                job_2_start_time[job_id] = start_time
        return job_2_start_time

    def make_semi_active(
        self,
        stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
        start_from_stage: StageIdType | None = None,
        *,
        operation_set: set[OperationType] | frozenset[OperationType] = frozenset(),
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
            operation_set: Optional subset of operations to retime, identified
                by ``(job_id, stage_id, mc_id)``.  If empty (default), all
                operations on the eligible stages are retimed.  Operations not
                in the set are kept fixed as anchors at their current times.

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
            first_idx = self.stage_2_index[start_from_stage]

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

                for old_start, old_end, job_id in job_tuple_seq:
                    duration = job_2_duration[job_id]
                    if self._is_selected_operation(
                        operation_set, stage_id, mc_id, job_id
                    ):
                        release = job_2_prev_stage_end_time.get(job_id, 0)
                        start = max(release, machine_available)
                        end = start + duration
                    else:
                        start = old_start
                        end = old_end
                        release = job_2_prev_stage_end_time.get(job_id, 0)
                        if start < max(release, machine_available):
                            raise ValueError(
                                f"Fixed operation {job_id}@{stage_id}.{mc_id} "
                                "violates precedence during make_semi_active"
                            )

                    new_tuple_seq.append((start, end, job_id))
                    machine_available = end

                mc_2_job_tuple_seq[mc_id] = new_tuple_seq
            self._rebuild_stage_end_time_cache(stage_id)

    def make_right_justified(
        self,
        stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
        *,
        operation_set: set[OperationType] | frozenset[OperationType] = frozenset(),
    ) -> None:
        """Right-shift operations in-place while preserving the current makespan.

        This method preserves machine assignments and machine order, and pushes
        each selected operation as far right as possible without violating:

        * inter-stage precedence for the same job,
        * machine precedence on the same machine, and
        * the current schedule makespan.

        Operations not included in ``operation_set`` are treated as fixed
        anchors and keep their current times unchanged.  If ``operation_set``
        is empty, all scheduled operations are right-shifted.
        """
        original_makespan = self.makespan
        next_stage_job_2_start_time: dict[JobIdType, int] = {}

        for stage_idx in range(len(self.stages) - 1, -1, -1):
            stage_id = self.stages[stage_idx]
            job_2_duration = stage_2_job_2_duration[stage_id]
            mc_2_job_tuple_seq = self.__stage_2_mc_2_job_tuple_seq[stage_id]

            for mc_id in self.machines_per_stage[stage_id]:
                job_tuple_seq = mc_2_job_tuple_seq[mc_id]
                if not job_tuple_seq:
                    continue

                machine_next_start = original_makespan
                new_tuple_seq_rev: list[tuple[int, int, JobIdType]] = []

                for old_start, old_end, job_id in reversed(job_tuple_seq):
                    duration = job_2_duration[job_id]
                    next_stage_start = next_stage_job_2_start_time.get(
                        job_id, original_makespan
                    )

                    if self._is_selected_operation(
                        operation_set, stage_id, mc_id, job_id
                    ):
                        end = min(next_stage_start, machine_next_start)
                        start = end - duration
                    else:
                        start = old_start
                        end = old_end
                        if end > min(next_stage_start, machine_next_start):
                            raise ValueError(
                                f"Fixed operation {job_id}@{stage_id}.{mc_id} "
                                "violates precedence during make_right_justified"
                            )

                    new_tuple_seq_rev.append((start, end, job_id))
                    machine_next_start = start

                mc_2_job_tuple_seq[mc_id] = list(reversed(new_tuple_seq_rev))

            self._rebuild_stage_end_time_cache(stage_id)
            next_stage_job_2_start_time = self._get_stage_job_2_start_time(stage_id)

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

    def collect_stage_machine_suffix_job_ids(
        self,
        stage_id: StageIdType,
        machine_id: McIdType,
        start_job_id: JobIdType,
    ) -> list[JobIdType]:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if machine_id not in self.machines_per_stage[stage_id]:
            raise ValueError(
                f"Invalid machine ID: {machine_id} for stage ID: {stage_id}"
            )

        job_tuple_seq = self.get_job_sequence(stage_id, machine_id)
        start_idx = next(
            (
                idx
                for idx, (_s, _e, job_id) in enumerate(job_tuple_seq)
                if job_id == start_job_id
            ),
            None,
        )
        if start_idx is None:
            raise ValueError(
                f"Job ID {start_job_id} not found on {stage_id}.{machine_id}"
            )
        return [job_id for _s, _e, job_id in job_tuple_seq[start_idx:]]

    def swap_stage_machine_operation_sets(
        self,
        stage_id: StageIdType,
        from_machine_id: McIdType,
        from_job_ids: Sequence[JobIdType],
        to_machine_id: McIdType,
        to_job_ids: Sequence[JobIdType],
        stage_2_job_2_duration: Mapping[StageIdType, Mapping[JobIdType, int]],
        *,
        do_make_semi_active: bool = False,
    ) -> None:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if from_machine_id not in self.machines_per_stage[stage_id]:
            raise ValueError(
                f"Invalid machine ID: {from_machine_id} for stage ID: {stage_id}"
            )
        if to_machine_id not in self.machines_per_stage[stage_id]:
            raise ValueError(
                f"Invalid machine ID: {to_machine_id} for stage ID: {stage_id}"
            )
        if from_machine_id == to_machine_id:
            if list(from_job_ids) == list(to_job_ids):
                return
            raise ValueError("swap_stage_machine_operation_sets requires two machines")

        from_job_ids = list(from_job_ids)
        to_job_ids = list(to_job_ids)
        if len(set(from_job_ids)) != len(from_job_ids):
            raise ValueError("Duplicate job IDs in from_job_ids")
        if len(set(to_job_ids)) != len(to_job_ids):
            raise ValueError("Duplicate job IDs in to_job_ids")
        overlap_job_ids = set(from_job_ids) & set(to_job_ids)
        if overlap_job_ids:
            overlap_job_id = next(iter(sorted(overlap_job_ids)))
            raise ValueError(
                f"Job ID {overlap_job_id} cannot be swapped from both machines"
            )

        from_job_id_set = set(from_job_ids)
        to_job_id_set = set(to_job_ids)
        from_seq = self.get_job_sequence(stage_id, from_machine_id)
        to_seq = self.get_job_sequence(stage_id, to_machine_id)

        existing_from_job_ids = [job_id for _s, _e, job_id in from_seq]
        existing_to_job_ids = [job_id for _s, _e, job_id in to_seq]
        for job_id in from_job_ids:
            if job_id not in existing_from_job_ids:
                raise ValueError(
                    f"Job ID {job_id} not found on {stage_id}.{from_machine_id}"
                )
        for job_id in to_job_ids:
            if job_id not in existing_to_job_ids:
                raise ValueError(
                    f"Job ID {job_id} not found on {stage_id}.{to_machine_id}"
                )

        from_selected = [
            job_tuple for job_tuple in from_seq if job_tuple[2] in from_job_id_set
        ]
        to_selected = [
            job_tuple for job_tuple in to_seq if job_tuple[2] in to_job_id_set
        ]
        from_remaining = [
            job_tuple for job_tuple in from_seq if job_tuple[2] not in from_job_id_set
        ]
        to_remaining = [
            job_tuple for job_tuple in to_seq if job_tuple[2] not in to_job_id_set
        ]
        from_insert_idx = next(
            (
                idx
                for idx, (_s, _e, job_id) in enumerate(from_seq)
                if job_id in from_job_id_set
            ),
            len(from_remaining),
        )
        to_insert_idx = next(
            (
                idx
                for idx, (_s, _e, job_id) in enumerate(to_seq)
                if job_id in to_job_id_set
            ),
            len(to_remaining),
        )

        self.__stage_2_mc_2_job_tuple_seq[stage_id][from_machine_id] = (
            from_remaining[:from_insert_idx]
            + to_selected
            + from_remaining[from_insert_idx:]
        )
        self.__stage_2_mc_2_job_tuple_seq[stage_id][to_machine_id] = (
            to_remaining[:to_insert_idx] + from_selected + to_remaining[to_insert_idx:]
        )

        affected_job_ids = set(from_job_ids) | set(to_job_ids)
        if do_make_semi_active:
            self.make_semi_active(stage_2_job_2_duration, start_from_stage=stage_id)
        else:
            for job_id in affected_job_ids:
                self.__stage_2_job_2_end_time[stage_id].pop(job_id, None)

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
    ) -> list[list[OperationType]]:
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
            list[list[OperationType]]: List of critical blocks,
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
        blocks: list[list[OperationType]] = []
        for stage_id, mc_2_job_seq in stage_2_mc_2_jobs.items():
            job_2_end_time = self.__stage_2_job_2_end_time[stage_id]
            job_2_duration = stage_2_job_2_duration[stage_id]
            for mc_id, job_seq in mc_2_job_seq.items():
                if not job_seq:
                    continue

                current_block: list[OperationType] = []

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
    start_map: Mapping[OperationType, int],
    end_map: Mapping[OperationType, int],
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
    start_map: Mapping[OperationType, int],
    end_map: Mapping[OperationType, int],
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
    start_map: Mapping[OperationType, int],
    end_map: Mapping[OperationType, int],
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


# Machine-centric dispatch helper


def _build_idle_gaps_from_ops(
    ops: list[tuple[int, int, JobIdType]],
    inf_end: int,
) -> list[list[int]]:
    """Return idle gaps as mutable [start, end] list, strictly increasing."""
    if not ops:
        return [[0, inf_end]]

    ops_sorted = sorted(ops, key=lambda x: x[0])
    gaps: list[list[int]] = []

    # gap before first op
    first_s = ops_sorted[0][0]
    if first_s > 0:
        gaps.append([0, first_s])

    # gaps between ops
    for (_, prev_e, _), (next_s, _, _) in zip(ops_sorted, ops_sorted[1:]):
        if next_s > prev_e:
            gaps.append([prev_e, next_s])

    # gap after last op
    last_e = ops_sorted[-1][1]
    if inf_end > last_e:
        gaps.append([last_e, inf_end])

    # Ensure at least one gap exists
    if not gaps:
        # machine is fully occupied until inf_end (unlikely with inf_end large)
        gaps = [[inf_end, inf_end]]

    return gaps


def _find_gap_index(gaps: list[list[int]], t: int, start_from: int = 0) -> int:
    """Smallest idx s.t. gaps[idx][1] > t. Assumes last gap end is 'inf'."""
    idx = start_from
    while idx < len(gaps) and gaps[idx][1] <= t:
        idx += 1
    if idx >= len(gaps):
        # Shouldn't happen if last gap end is inf, but be defensive.
        return len(gaps) - 1
    return idx


# Sequence extraction functions


def get_midpoint_sequence(schedule: HybridFlowshopLiteSchedule) -> list[str]:
    """Get job sequence based on midpoint criteria.

    Args:
        instance (HybridFlowshopParameters): The hybrid flowshop problem instance.
        schedule (HybridFlowshopLiteSchedule): The hybrid flowshop schedule.

    Returns:
        list[str]: A list of job names ordered by midpoint criteria.
    """
    start_map = schedule.get_jik_2_start_time_map()
    end_map = schedule.get_jik_2_end_time_map()
    jobs = schedule.jobs
    idx_map = {j: idx for idx, j in enumerate(jobs)}
    first_stage = schedule.stages[0]
    last_stage = schedule.stages[-1]

    seq_info: list[tuple[float, int, int, str]] = []
    for j in jobs:
        # find any machine k for first and last stage
        s_first = next(
            t
            for (job, stage, _), t in start_map.items()
            if job == j and stage == first_stage
        )
        e_last = next(
            t
            for (job, stage, _), t in end_map.items()
            if job == j and stage == last_stage
        )
        midpoint = (s_first + e_last) / 2
        seq_info.append((midpoint, s_first, idx_map[j], j))

    seq_info.sort(key=lambda x: (x[0], x[1], x[2]))
    return [info[3] for info in seq_info]


def get_bottleneck_stage_job_sequence(
    schedule: HybridFlowshopLiteSchedule,
) -> list[str]:
    """Get job sequence based on bottleneck stage.

    Args:
        schedule (HybridFlowshopLiteSchedule): The hybrid flowshop schedule.

    Returns:
        list[str]: A list of job names ordered by starting time at the bottleneck stage,
        with ties broken by (starting time + end time) / 2 and then by
        original job order index.
    """
    # Identify bottleneck stage as the stage with the smallest type 2 idle time
    stage_2_mc_2_idle_time_map = schedule.get_stage_2_mc_2_idle_time_map()
    stage_2_total_idle_time = {
        stage: sum(mc_2_idle_time.values())
        for stage, mc_2_idle_time in stage_2_mc_2_idle_time_map.items()
    }
    bottleneck_stage = min(
        stage_2_total_idle_time, key=lambda s: stage_2_total_idle_time[s]
    )

    start_map = schedule.get_jik_2_start_time_map()
    end_map = schedule.get_jik_2_end_time_map()
    jobs = schedule.jobs
    idx_map = {j: idx for idx, j in enumerate(jobs)}

    seq_info: list[tuple[int, float, int, str]] = []
    for j in jobs:
        s_bottleneck = next(
            t
            for (job, stage, _), t in start_map.items()
            if job == j and stage == bottleneck_stage
        )
        e_bottleneck = next(
            t
            for (job, stage, _), t in end_map.items()
            if job == j and stage == bottleneck_stage
        )
        midpoint = (s_bottleneck + e_bottleneck) / 2
        seq_info.append((s_bottleneck, midpoint, idx_map[j], j))

    seq_info.sort(key=lambda x: (x[0], x[1], x[2]))
    return [info[3] for info in seq_info]


def get_first_stage_start_sequence(schedule: HybridFlowshopLiteSchedule) -> list[str]:
    """Get job sequence based on first stage start time.

    Args:
        schedule (HybridFlowshopLiteSchedule): The hybrid flowshop schedule.

    Returns:
        list[str]: A list of job IDs ordered by first stage start time,
        with ties broken by original job order index.
    """
    start_map = schedule.get_jik_2_start_time_map()
    jobs = schedule.jobs
    idx_map = {j: idx for idx, j in enumerate(jobs)}
    first_stage = schedule.stages[0]

    seq_info: list[tuple[int, int, str]] = []
    for j in jobs:
        s_first = next(
            t
            for (job, stage, _), t in start_map.items()
            if job == j and stage == first_stage
        )
        seq_info.append((s_first, idx_map[j], j))

    seq_info.sort(key=lambda x: (x[0], x[1]))
    return [info[2] for info in seq_info]
