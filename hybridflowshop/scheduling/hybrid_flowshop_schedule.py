from __future__ import annotations

from collections import defaultdict

from .exception import SchedulingFailureException
from .hybrid_flowshop_operation import HybridFlowshopOperation
from .hybrid_flowshop_stage import HybridFlowshopStage


class HybridFlowshopSchedule:
    def __init__(self) -> None:
        self._stages: dict[str, HybridFlowshopStage] = {}
        """Stage name -> HybridFlowshopStage instance."""

        # Internal states

        self.job_2_last_oper_end_time_map: dict[str, int] = defaultdict(int)
        """Job name -> end time of the last operation scheduled for that job."""

        self.job_2_scheduled_oper_count_map: dict[str, int] = defaultdict(int)
        """Job name -> count of operations scheduled for that job."""

    @classmethod
    def from_stage_name_2_mc_name_list_map(
        cls, stage_name_2_mc_name_list_map: dict[str, list[str]]
    ) -> HybridFlowshopSchedule:
        """
        Create a HybridFlowshopSchedule from a mapping of stage IDs to machine IDs.

        Args:
            stage_name_2_mc_name_list_map (dict[str, list[str]]): Mapping of stage IDs to lists of machine IDs.

        Returns:
            HybridFlowshopSchedule: A new instance of HybridFlowshopSchedule.
        """
        schedule = cls()
        for stage_name, mc_name_list in stage_name_2_mc_name_list_map.items():
            schedule._stages[stage_name] = HybridFlowshopStage.from_mc_name_list(
                stage_name, mc_name_list
            )
        return schedule

    # Getters

    @property
    def makespan(self) -> int:
        """
        Calculate the makespan of the entire schedule.

        Returns:
            int: The maximum makespan across all stages.
        """
        return max((stage.makespan for stage in self._stages.values()), default=0)

    def get_stage_by_name(self, stage_name: str) -> HybridFlowshopStage:
        """
        Get a stage by its name.

        Args:
            stage_name (str): The name of the stage to retrieve.

        Raises:
            ValueError: If the stage with the given name does not exist in the schedule.

        Returns:
            HybridFlowshopStage: The stage instance with the specified name.
        """
        if stage_name not in self._stages:
            raise ValueError(f"Stage {stage_name} not found in schedule")
        return self._stages[stage_name]

    def get_start_time_map(self) -> dict[tuple[str, str, str], int]:
        """
        Get a map of (job_name, stage_name, mc_name) to start time for all operations in the schedule.

        Returns:
            dict[tuple[str, str, str], int]: A dictionary mapping (job_name, stage_name, mc_name) to start time.
        """
        return_dict: dict[tuple[str, str, str], int] = {}
        for stage in self._stages.values():
            stage_start_time_map = stage.get_start_time_map()
            for key, value in stage_start_time_map.items():
                if key in return_dict:
                    raise ValueError(
                        f"Duplicate start time entry for key {key} in stage {stage.name}."
                    )
                return_dict[key] = value
        return return_dict

    def get_end_time_map(self) -> dict[tuple[str, str, str], int]:
        """
        Get a map of (job_name, stage_name, mc_name) to end time for all operations in the schedule.

        Returns:
            dict[tuple[str, str, str], int]: A dictionary mapping (job_name, stage_name, mc_name) to end time.
        """
        return_dict: dict[tuple[str, str, str], int] = {}
        for stage in self._stages.values():
            stage_end_time_map = stage.get_end_time_map()
            for key, value in stage_end_time_map.items():
                if key in return_dict:
                    raise ValueError(
                        f"Duplicate end time entry for key {key} in stage {stage.name}."
                    )
                return_dict[key] = value
        return return_dict

    # Setters

    def schedule_operation(
        self,
        operation: HybridFlowshopOperation,
        force_add: bool = False,
    ) -> HybridFlowshopOperation | None:
        return self.get_stage_by_name(operation.stage_name).add_operation(
            operation, force_add=force_add
        )

    def dispatch_operation_earliest(
        self, job_name: str, stage_name: str, p: int, release_t: int = 0
    ) -> HybridFlowshopOperation:
        """
        Dispatch an operation to the earliest available machine in the specified stage.

        Args:
            job_name (str): The name of the job this operation belongs to.
            stage_name (str): The name of the stage this operation belongs to.
            p (int): The processing time required for this operation.
            release_t (int, optional): The earliest time the operation can start. Defaults to 0.

        Raises:
            SchedulingFailureException: If the operation is not scheduled.

        Returns:
            HybridFlowshopOperation: Scheduled operation.
        """
        stage = self.get_stage_by_name(stage_name)
        _release_t = max(release_t, self.job_2_last_oper_end_time_map[job_name])
        mc_name, start_time = stage.select_machine_by_start_slack_idx(p, _release_t)
        # integer casting to ensure start_time is an integer
        # (not np.int64 for YAML compatibility)
        end_time = int(start_time + p)
        operation = stage.add_operation(
            HybridFlowshopOperation(
                job_name=job_name,
                stage_name=stage_name,
                mc_name=mc_name,
                start=start_time,
                end=end_time,
            )
        )
        if operation is None:
            raise SchedulingFailureException(
                f"{job_name}.{stage_name}", p, mc_name, start_time
            )

        # Update internal states
        self.job_2_last_oper_end_time_map[job_name] = operation.end
        self.job_2_scheduled_oper_count_map[job_name] += 1

        return operation

    def dispatch_job_by_stages(
        self,
        job_name: str,
        stage_name_list: list[str],
        stage_name_2_p_map: dict[str, int],
        release_t: int = 0,
    ) -> list[HybridFlowshopOperation]:
        """
        Dispatch a job across multiple stages, scheduling each stage's operation
        on the earliest available machine.

        Args:
            job_name (str): The name of the job to be dispatched.
            stage_name_list (list[str]): List of stage names in the order they should be processed.
            stage_name_2_p_map (dict[str, int]): Stage name -> processing time.
            release_t (int, optional): The earliest time the job can start processing at the 1st stage.
                Defaults to 0.

        Raises:
            ValueError: If a stage name is not found in the schedule or if an operation cannot be scheduled.

        Returns:
            list[HybridFlowshopOperation]: A list of scheduled operations for the job across all stages.
        """
        operations = []
        last_end_time = max(self.job_2_last_oper_end_time_map[job_name], release_t)

        for stage_name in stage_name_list:
            operation = self.dispatch_operation_earliest(
                job_name, stage_name, stage_name_2_p_map[stage_name], last_end_time
            )
            operations.append(operation)
            last_end_time = operation.end

        return operations

    def dispatch_stage_by_jobs(
        self,
        stage_name: str,
        job_name_list: list[str],
        job_name_2_p_map: dict[str, int],
        release_t: int = 0,
    ) -> list[HybridFlowshopOperation]:
        """
        Dispatch operations for a given stage, scheduling jobs in the order
        determined by job_priority_queue.

        The job_priority_queue is sorted by:
        1. Jobs with smaller last operation end time are scheduled earlier.
        2. If two jobs have the same last operation end time,
           jobs that come earlier in job_name_list are scheduled earlier.

        Args:
            stage_name (str): Target stage name.
            job_name_list (list[str]): List of job names to be dispatched in this stage.
            job_name_2_p_map (dict[str, int]): Job name -> processing time.
            release_t (int, optional): The earliest time the stage can start.
                Defaults to 0.

        Returns:
            list[HybridFlowshopOperation]: A list of scheduled operations for the jobs
                in the specified stage, in job_priority_queue order.
        """
        operations = []

        # Make a job priority queue.
        # 1. Jobs having smaller last operation end time are scheduled earlier.
        # 2. If two jobs have the same last operation end time,
        #    jobs that comes earlier in job_name_list are scheduled earlier.
        job_priority_queue = sorted(
            job_name_list,
            key=lambda job_name: (
                self.job_2_last_oper_end_time_map[job_name],
                job_name_list.index(job_name),
            ),
        )
        # logging.info(f"Dispatching stage {stage_name} for jobs {job_priority_queue}")

        for job_name in job_priority_queue:
            operation = self.dispatch_operation_earliest(
                job_name, stage_name, job_name_2_p_map[job_name], release_t
            )
            operations.append(operation)

        return operations
