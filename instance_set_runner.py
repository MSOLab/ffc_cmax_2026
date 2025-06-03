from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

from routix import ElapsedTimer

from single_instance_runner import SingleInstanceRunner

ProblemT = TypeVar("ProblemT")
RunnerT = TypeVar("RunnerT", bound=SingleInstanceRunner)


class InstanceSetRunner(Generic[ProblemT, RunnerT], ABC):
    """
    Orchestrates solving a set of instances with a given runner class.
    """

    def __init__(
        self,
        s_i_runner_class: type[RunnerT],
        instances: Sequence[ProblemT],
        shared_param_dict: dict,
        subroutine_flow: Any,
        stopping_criteria: Any,
        output_dir: Path,
        output_metadata: dict[str, Any],
    ):
        # Set up the elapsed timer
        self.e_timer = ElapsedTimer()

        # SingleInstanceRunner
        self.s_i_runner_class = s_i_runner_class
        # Instance data
        self.instances = instances
        self.shared_param_dict = shared_param_dict
        # Algorithm data
        self.subroutine_flow = subroutine_flow
        self.stopping_criteria = stopping_criteria
        # Output data
        self.output_dir = output_dir
        self.output_metadata = output_metadata

        self.runners: list[RunnerT] = []
        self.results: list[Any] = []

        self._set_start_dt()
        self._init_working_dir()

    def _set_start_dt(self) -> None:
        """
        Sets the start date-time for the elapsed timer.
        If the start date-time is already in output_metadata, it uses that.
        Otherwise, it initializes the start date-time from the elapsed timer.
        """
        if dt := self.output_metadata.get("start_dt"):
            self.e_timer.set_start_time(dt)
        else:
            self.output_metadata["start_dt"] = self.e_timer.get_formatted_start_dt()

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
        if self.output_dir.name != self.e_timer.get_formatted_start_dt():
            self.working_dir /= self.e_timer.get_formatted_start_dt()
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self.runners.clear()
        self.results.clear()

        for idx, instance in enumerate(self.instances):
            runner = self.s_i_runner_class(
                instance=instance,
                shared_param_dict=self.shared_param_dict,
                subroutine_flow=self.subroutine_flow,
                stopping_criteria=self.stopping_criteria,
                output_dir=self.output_dir,
                output_metadata=self.output_metadata,
            )
            self.runners.append(runner)
            try:
                result = runner.run()
            except Exception as e:
                # Handle or log error, append None or an error object as appropriate
                print(f"Error in instance {idx}: {e}")  # TODO: logging
                result = None
            self.results.append(result)

        return self.post_run_process()

    @abstractmethod
    def post_run_process(self):
        """
        Post-processes the results after running all instances.
        This method should be implemented in subclasses to handle specific post-run logic.
        """
        pass

    def get_successful_results(self) -> list[Any]:
        """Returns a list of non-None results."""
        return [r for r in self.results if r is not None]
