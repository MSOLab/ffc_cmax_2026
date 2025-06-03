from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from routix import ElapsedTimer, SubroutineController

ProblemT = TypeVar("ProblemT")  # Type for the problem instance
ControllerT = TypeVar("ControllerT", bound=SubroutineController)


class SingleInstanceRunner(Generic[ProblemT, ControllerT], ABC):
    """
    Abstract runner for a single problem instance.
    """

    instance: ProblemT
    ctrlr: ControllerT

    working_dir: Path
    """Working directory for the instance run."""

    def __init__(
        self,
        instance: ProblemT,
        shared_params: dict,
        subroutine_flow: Any,
        stopping_criteria: Any,
        output_dir: Path,
        output_metadata: dict[str, Any],
    ):
        # Set up the elapsed timer
        self.e_timer = ElapsedTimer()
        if dt := output_metadata.get("start_dt"):
            self.e_timer.set_start_time(dt)

        # Instance data
        self.instance = instance
        self.shared_params = shared_params
        # Algorithm data
        self.subroutine_flow = subroutine_flow
        self.stopping_criteria = stopping_criteria
        # Output data
        self.output_dir = output_dir
        self.output_metadata = output_metadata

        # Alias
        self.ins_name = getattr(instance, "name", None)

        self._init_working_dir()

    def _init_working_dir(self) -> None:
        """
        Initialize the working directory for the instance run.
        This method creates a directory structure based on the output directory,
        elapsed timer start time, and instance name if available.

        - If the output directory stem does not match the formatted start date-time,
        it creates a subdirectory with the formatted start date-time.
        - If an instance name is provided, it creates a further subdirectory for the instance.
        """
        self.working_dir = self.output_dir
        if self.output_dir.stem != self.e_timer.get_formatted_start_dt():
            self.working_dir /= self.e_timer.get_formatted_start_dt()
        if self.ins_name is not None:
            self.working_dir /= self.ins_name
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def solve(self):
        """
        Solve the instance by running the controller and performing post-run processing.
        """
        self.run()
        return self.post_run_process()

    def run(self) -> None:
        """
        Run the instance using the initialized controller.
        """
        self.ctrlr = self.init_controller()
        self.ctrlr.set_working_dir(self.working_dir)
        self.ctrlr.run()

    @abstractmethod
    def init_controller(self) -> ControllerT:
        """
        Initialize the controller with the given instance and parameters.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def post_run_process(self):
        """
        Post-run process to handle any finalization tasks.
        This method should be implemented by subclasses.
        """
        pass
