from __future__ import annotations

from .hybrid_flowshop_operation import HybridFlowshopOperation
from .hybrid_flowshop_stage import HybridFlowshopStage


class HybridFlowshopSchedule:
    def __init__(self) -> None:
        self._stages: dict[str, HybridFlowshopStage] = {}
        """Map from stage name to HybridFlowshopStage instance."""

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

    def get_earliest_start_mc_name_and_time(
        self, stage_name: str, p: int, release_t: int = 0
    ) -> tuple[str, int]:
        """Get the earliest available machine name and start time for a given stage.

        Args:
            stage_name (str): The name of the stage to check.
            p (int): The processing time required for the operation.
            release_t (int, optional): The earliest time the operation can start.
                Defaults to 0.

        Returns:
            tuple[str, int]: A tuple containing the name of the earliest available machine
                and the time it can start processing the operation.
        """
        return self.get_stage_by_name(stage_name).get_earliest_start_mc_name_and_time(
            p, release_t
        )

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
    ) -> HybridFlowshopOperation | None:
        """
        Dispatch an operation to the earliest available machine in the specified stage.

        Args:
            job_name (str): The name of the job this operation belongs to.
            stage_name (str): The name of the stage this operation belongs to.
            p (int): The processing time required for this operation.
            release_t (int, optional): The earliest time the operation can start. Defaults to 0.

        Returns:
            HybridFlowshopOperation | None: The operation if added successfully, otherwise None.
        """
        stage = self.get_stage_by_name(stage_name)
        mc_name, start_time = stage.get_earliest_start_mc_name_and_time(p, release_t)
        return stage.add_operation(
            HybridFlowshopOperation(
                job_name=job_name,
                stage_name=stage_name,
                mc_name=mc_name,
                start=start_time,
                end=start_time + p,
            )
        )

    def dispatch_job_earliest(
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
            stage_name_2_p_map (dict[str, int]): Mapping of stage names to their processing times.
            release_t (int, optional): The earliest time the job can start processing at the 1st stage.
                Defaults to 0.

        Raises:
            ValueError: If a stage name is not found in the schedule or if an operation cannot be scheduled.

        Returns:
            list[HybridFlowshopOperation]: A list of scheduled operations for the job across all stages.
        """
        stage_name_2_mc_name_map: dict[str, str] = {}
        stage_name_2_start_time_map: dict[str, int] = {}

        # calculate target machine and start time for each stage
        _release_t = max(release_t, 0)
        for stage_name in stage_name_list:
            if stage_name not in self._stages:
                raise ValueError(f"Stage {stage_name} not found in schedule")
            p = stage_name_2_p_map[stage_name]
            mc_name, start_time = self.get_earliest_start_mc_name_and_time(
                stage_name, p, _release_t
            )

            stage_name_2_mc_name_map[stage_name] = mc_name
            stage_name_2_start_time_map[stage_name] = start_time

            # Update _release_t to the end time of the last operation scheduled
            _release_t = start_time + p

        # Create operations for each stage
        operations = []
        for stage_name in stage_name_list:
            mc_name = stage_name_2_mc_name_map[stage_name]
            start_time = stage_name_2_start_time_map[stage_name]
            # integer casting to ensure start_time is an integer
            # (not np.int64 for YAML compatibility)
            end_time = int(start_time + stage_name_2_p_map[stage_name])

            operation = HybridFlowshopOperation(
                job_name=job_name,
                stage_name=stage_name,
                mc_name=mc_name,
                start=start_time,
                end=end_time,
            )
            scheduled_operation = self.schedule_operation(operation)
            if scheduled_operation is not None:
                operations.append(scheduled_operation)
            else:
                raise ValueError(
                    f"Failed to schedule operation for job {job_name} in stage {stage_name}"
                )
        return operations
