from schore.schedule.abstract import Resource

from .hybrid_flowshop_operation import HybridFlowshopOperation


class Machine(Resource[HybridFlowshopOperation]):
    def __init__(self, name: str) -> None:
        """Initialize a Machine.

        Args:
            name (str): The name of the machine.
        """
        super().__init__()
        self._name: str = name
        """The name of the machine."""

    @property
    def name(self) -> str:
        """Returns the name of the machine."""
        return self._name

    @property
    def operations(self) -> list[HybridFlowshopOperation]:
        """Returns the list of operations assigned to this machine.

        Returns:
            list[HybridFlowshopOperation]: The list of operations.
        """
        return self.activity_list

    def add_operation(
        self, operation: HybridFlowshopOperation, force_add: bool = False
    ) -> HybridFlowshopOperation | None:
        """Add an operation to the machine.

        Args:
            operation (HybridFlowshopOperation): The operation to add.
            force_add (bool, optional): If True, force add the operation even if it conflicts with existing operations.
                Defaults to False.

        Returns:
            HybridFlowshopOperation | None: The operation if added successfully, otherwise None.
        """
        if operation.mc_name != self.name:
            raise ValueError(
                f"Operation's machine name {operation.mc_name} does not match"
                f" this machine's name {self.name}."
            )
        if self.add_activity(operation, force_add=force_add) is None:
            return None
        return operation

    def get_start_time_map(self) -> dict[tuple[str, str, str], int]:
        """Get a map of (job_name, stage_name, mc_name) to start time.

        Returns:
            dict[tuple[str, str, str], int]: A dictionary mapping (job_name, stage_name, mc_name) to start time.
        """
        return_dict: dict[tuple[str, str, str], int] = {}
        for operation in self.operations:
            key = (operation.job_name, operation.stage_name, self.name)
            if key in return_dict:
                raise ValueError(
                    f"Duplicate start time entry for key {key} in machine {self.name}."
                )
            return_dict[key] = operation.start
        return return_dict

    def get_end_time_map(self) -> dict[tuple[str, str, str], int]:
        """Get a map of (job_name, stage_name, mc_name) to end time.

        Returns:
            dict[tuple[str, str, str], int]: A dictionary mapping (job_name, stage_name, mc_name) to end time.
        """
        return_dict: dict[tuple[str, str, str], int] = {}
        for operation in self.operations:
            key = (operation.job_name, operation.stage_name, self.name)
            if key in return_dict:
                raise ValueError(
                    f"Duplicate end time entry for key {key} in machine {self.name}."
                )
            return_dict[key] = operation.end
        return return_dict
