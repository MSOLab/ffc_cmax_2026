from __future__ import annotations

from schore.schedule.base import ParallelResourceGroup

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

    # Required getters

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

    # Required setters

    def add_resource(self, res: Machine) -> None:
        """
        Add a resource to the stage if it does not already exist.

        Args:
            res (Resource): The resource to add.
        """
        if res.name not in self._mc_name_2_ins_map:
            self._mc_name_2_ins_map[res.name] = res

    # Getters

    def get_earliest_start_mc_name_and_time(
        self, duration: int, release_t: int = 0
    ) -> tuple[str, int]:
        return self.get_earliest_start_resource_name_and_time(duration, release_t)

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

    # Setters

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
