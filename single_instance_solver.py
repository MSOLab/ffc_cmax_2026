from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from routix import ElapsedTimer, SubroutineController


class SingleInstanceSolver(ABC):
    instance: Any  # Loaded problem instance
    ctrlr: SubroutineController

    def __init__(
        self,
        instance: Any,  # Loaded problem instance
        controller_init_kwargs: dict,
        subroutine_flow: Any,
        stopping_criteria: Any,
        output_dir: Path,  # Output directory
        output_metadata: dict[str, Any],
    ):
        self.e_timer = ElapsedTimer()

        # Instance data
        self.instance = instance
        self.controller_init_kwargs = controller_init_kwargs
        # Algorithm data
        self.subroutine_flow = subroutine_flow
        self.stopping_criteria = stopping_criteria
        # Output data
        self.output_dir = output_dir
        self.output_metadata = output_metadata

        # Alias
        self.ins_name = getattr(instance, "name", None)

        self.prepare_output_directory()

    def prepare_output_directory(self):
        """Prepare the output directory for the instance run."""
        self.output_dir_instance = (
            self.output_dir / self.e_timer.get_formatted_start_dt()
        )
        if self.ins_name is not None:
            self.output_dir_instance = self.output_dir_instance / self.ins_name
        self.output_dir_instance.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def init_controller(self) -> Any:
        """
        Initialize the controller with the given instance and parameters.
        This method should be implemented by subclasses.
        """
        pass

    def run(self) -> Any:
        """
        Run the instance using the initialized controller.
        """
        self.ctrlr = self.init_controller()
        self.ctrlr.set_working_dir(self.output_dir_instance)
        self.ctrlr.run()
        return self.ctrlr

    @abstractmethod
    def post_run_process(self):
        """
        Post-run process to handle any finalization tasks.
        This method should be implemented by subclasses.
        """
        pass

    def solve(self):
        """
        Solve the instance by running the controller and performing post-run processing.
        """
        self.run()
        self.post_run_process()
