from __future__ import annotations

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

    def get_machine_and_earliest_available_time_by_start_idle_idx(
        self, stage_id: StageIdType, release_t: int | None = None
    ) -> tuple[McIdType, int]:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")
        if not self.machines_per_stage[stage_id]:
            raise ValueError(f"No machines available in stage {stage_id}.")

        # Initialize with first machine's values
        first_mc = self.machines_per_stage[stage_id][0]
        mc_latest_end_time = self.get_machine_latest_end_time(stage_id, first_mc)

        if release_t is not None and mc_latest_end_time < release_t:
            best_eat, best_idle = release_t, release_t - mc_latest_end_time
        else:
            best_eat, best_idle = mc_latest_end_time, 0

        best_mc = first_mc

        # Check remaining machines
        for mc in self.machines_per_stage[stage_id][1:]:
            mc_latest_end_time = self.get_machine_latest_end_time(stage_id, mc)

            if release_t is not None and mc_latest_end_time < release_t:
                eat, idle = release_t, release_t - mc_latest_end_time
            else:
                eat, idle = mc_latest_end_time, 0

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

    def append_ops_times_2_mc(
        self,
        stage_id: StageIdType,
        mc_id: McIdType,
        job_id: JobIdType,
        start_time: int,
        end_time: int,
    ) -> None:
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

        self.__stage_2_mc_2_job_tuple_seq[stage_id][mc_id].append(
            (start_time, end_time, job_id)
        )
        self.__stage_2_job_2_end_time[stage_id][job_id] = end_time

    # Setters - dispatching methods

    def append_operation_2_mc(
        self,
        stage_id: StageIdType,
        mc_id: McIdType,
        job_id: JobIdType,
        duration: int,
        release_t: int | None = None,
    ) -> None:
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
        # Find earliest available time on the specified machine
        start_time = self.get_machine_latest_end_time(stage_id, mc_id)
        if start_time < release_t:
            # Cannot start before release time
            start_time = release_t
        # Compute end time
        end_time = start_time + duration
        # Append operation
        self.append_ops_times_2_mc(stage_id, mc_id, job_id, start_time, end_time)

    def append_operation_2_stage(
        self,
        stage_id: StageIdType,
        job_id: JobIdType,
        duration: int,
        release_t: int | None = None,
    ) -> None:
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
                stage_id, release_t=release_t
            )
        )
        # Compute end time
        end_time = start_time + duration
        # Append operation
        self.append_ops_times_2_mc(stage_id, mc_id, job_id, start_time, end_time)

    def dispatch_stage_by_jobs(
        self,
        stage_id: StageIdType,
        job_id_seq: Sequence[JobIdType],
        job_2_duration: Mapping[JobIdType, int],
    ) -> None:
        if stage_id not in self.stages:
            raise ValueError(f"Invalid stage ID: {stage_id}")

        for job_id in job_id_seq:
            if job_id not in job_2_duration:
                raise ValueError(f"Duration for job ID {job_id} not provided")
            duration = job_2_duration[job_id]
            self.append_operation_2_stage(stage_id, job_id, duration)

    def dispatch_job_by_stages(
        self, job_id: JobIdType, stage_2_duration: Mapping[StageIdType, int]
    ) -> None:
        if job_id not in self.jobs:
            raise ValueError(f"Invalid job ID: {job_id}")

        for stage_id in self.stages:
            if stage_id not in stage_2_duration:
                raise ValueError(f"Duration for stage ID {stage_id} not provided")
            duration = stage_2_duration[stage_id]
            self.append_operation_2_stage(stage_id, job_id, duration)

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
