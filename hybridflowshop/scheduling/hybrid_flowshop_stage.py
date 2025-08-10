from __future__ import annotations

from schore.schedule.abstract import ParallelResourceGroup

from .hybrid_flowshop_operation import HybridFlowshopOperation
from .machine import Machine


class HybridFlowshopStage(ParallelResourceGroup[Machine]):
    def __init__(self, name: str) -> None:
        """
        Initialize a HybridFlowshopStage.

        Args:
            name (str): The name of the stage.
            mc_name_list (list[str]): List of machine IDs in this stage.
        """
        super().__init__()
        self._name: str = name
        """The name of the stage."""
        self._mc_name_2_ins_map: dict[str, Machine] = {}
        """Map from machine ID to Machine instance."""

    @classmethod
    def from_mc_name_list(
        cls, name: str, mc_name_list: list[str]
    ) -> HybridFlowshopStage:
        """Create a HybridFlowshopStage from a list of machine IDs.

        Args:
            name (str): The name of the stage.
            mc_name_list (list[str]): List of machine IDs in this stage.

        Returns:
            HybridFlowshopStage: A new instance of HybridFlowshopStage.
        """
        stage = cls(name=name)
        for mc_name in mc_name_list:
            stage.create_machine_by_name(mc_name)
        return stage

    def deepcopy(self) -> "HybridFlowshopStage":
        """
        Returns a deep copy of this HybridFlowshopStage,
        including all machines and their state.
        """
        from copy import deepcopy

        new_stage = HybridFlowshopStage(self._name)
        # Deep copy all machines
        new_stage._mc_name_2_ins_map = {
            mc_name: mc.deepcopy() if hasattr(mc, "deepcopy") else deepcopy(mc)
            for mc_name, mc in self._mc_name_2_ins_map.items()
        }
        return new_stage

    # Start required getters

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the stage.
        """
        return self._name

    @property
    def resources(self) -> list[Machine]:
        """
        Returns:
            list[Machine]: List of resources (machines) in this stage.
                The sequence of machines is determined by the order they were added.
        """
        return list(self._mc_name_2_ins_map.values())

    # End required getters

    # Start required setters

    def add_resource(self, res: Machine) -> None:
        """
        Add a resource to the stage if it does not already exist.

        Args:
            res (Resource): The resource to add.
        """
        if res.name not in self._mc_name_2_ins_map:
            self._mc_name_2_ins_map[res.name] = res

    # End required setters

    # Start getters

    def select_machine_by_start_idle_idx(
        self, duration: int, release_t: int = 0
    ) -> tuple[str, int]:
        """
        Find the machine name and earliest feasible start time for a new activity.

        1. Machines having the earliest start time are preferred.
        2. If multiple machines have the same earliest start time,
           machines with the smallest idle time (earliest start time - makespan) are preferred.
        3. If multiple machines have the same earliest start time and idle time,
           the first machine in the order they were added is chosen.

        Args:
            duration (int): Duration of the new activity (must be positive).
            release_t (int, optional): Earliest time the activity may start. Defaults to 0.

        Raises:
            ValueError: If `duration` is not positive.
            ValueError: If no resource is available to start the activity.

        Returns:
            tuple[str, int]: A tuple of (machine name, earliest feasible start time).
        """
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        candidate_info = []
        for res in self.resources:
            start_time = res.get_earliest_start_time(duration, release_t)
            idle = start_time - res.makespan
            candidate_info.append((start_time, idle, res))

        if not candidate_info:
            raise ValueError("No resource available to start the activity")

        # 1. Find machines with the earliest start time
        min_start = min(c[0] for c in candidate_info)
        earliest_start_candidates = [c for c in candidate_info if c[0] == min_start]
        # 2. Among them, find machines with the smallest idle time
        min_idle = min(c[1] for c in earliest_start_candidates)
        min_idle_candidates = [c for c in earliest_start_candidates if c[1] == min_idle]
        # 3. Pick the first one (order of addition)
        selected = min_idle_candidates[0]
        return selected[2].name, int(selected[0])

    def get_machine_by_name(self, mc_name: str) -> Machine:
        """Get a machine by its ID.

        Args:
            mc_name (str): The ID of the machine to retrieve.

        Raises:
            ValueError: If the machine with the given ID does not exist in this stage.

        Returns:
            Machine: The machine instance with the specified ID.
        """
        if mc_name not in self._mc_name_2_ins_map:
            raise ValueError(f"Machine {mc_name} not found in stage {self.name}")
        return self._mc_name_2_ins_map[mc_name]

    def get_start_time_map(self) -> dict[tuple[str, str, str], int]:
        """
        Get a map of (job_name, stage_name, mc_name) to start time
        by aggregating results from all machines in the stage.

        Returns:
            dict[tuple[str, str, str], int]: A dictionary mapping (job_name, stage_name, mc_name) to start time,
                aggregated from all machines in this stage.
        """
        return_dict: dict[tuple[str, str, str], int] = {}
        for machine in self.resources:
            machine_start_time_map = machine.get_start_time_map()
            for key, value in machine_start_time_map.items():
                if key in return_dict:
                    raise ValueError(
                        f"Duplicate start time entry for key {key} from machine {machine.name}."
                    )
                return_dict[key] = value
        return return_dict

    def get_end_time_map(self) -> dict[tuple[str, str, str], int]:
        """
        Get a map of (job_name, stage_name, mc_name) to end time
        by aggregating results from all machines in the stage.

        Returns:
            dict[tuple[str, str, str], int]: A dictionary mapping (job_name, stage_name, mc_name) to end time,
                aggregated from all machines in this stage.
        """
        return_dict: dict[tuple[str, str, str], int] = {}
        for machine in self.resources:
            machine_end_time_map = machine.get_end_time_map()
            for key, value in machine_end_time_map.items():
                if key in return_dict:
                    raise ValueError(
                        f"Duplicate end time entry for key {key} from machine {machine.name}."
                    )
                return_dict[key] = value
        return return_dict

    # End getters

    # Start setters

    def create_machine_by_name(self, mc_name: str) -> None:
        machine = Machine(name=mc_name)
        self.add_resource(machine)

    def add_operation(
        self, operation: HybridFlowshopOperation, force_add=False
    ) -> HybridFlowshopOperation | None:
        """Add an operation to the stage.

        Args:
            operation (HybridFlowshopOperation): The operation to add.
            mc_name (str): The ID of the machine to which the operation will be added.

        Raises:
            ValueError: If the operation does not belong to this stage.
            ValueError: If the operation's machine ID does not match any machine in this stage.

        Returns:
            HybridFlowshopOperation | None: The operation if added successfully, otherwise None.
        """
        if operation.stage_name != self.name:
            raise ValueError(
                f"Operation {operation.name} does not belong to this stage {self.name}"
            )
        if operation.mc_name not in self._mc_name_2_ins_map:
            raise ValueError(
                f"Machine {operation.mc_name} not found in stage {self.name}"
            )

        return self.get_machine_by_name(operation.mc_name).add_operation(
            operation, force_add=force_add
        )

    # End setters
