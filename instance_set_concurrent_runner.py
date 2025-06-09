import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

from routix.runner import InstanceSetRunner

from single_instance_runner import SingleInstanceRunner

ProblemT = TypeVar("ProblemT")
RunnerT = TypeVar("RunnerT", bound=SingleInstanceRunner)


class InstanceSetConcurrentRunner(InstanceSetRunner, Generic[ProblemT, RunnerT]):
    """Orchestrates solving a set of instances concurrently using a specified runner class."""

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
        super().__init__(
            s_i_runner_class,
            instances,
            shared_param_dict,
            subroutine_flow,
            stopping_criteria,
            output_dir,
            output_metadata,
        )
        self._max_workers: int = 2  # Default value for max_workers

    def get_max_workers(self) -> int:
        """
        Returns:
            int: The maximum number of workers for concurrent execution.
                If not set, returns the default value of 2.
        """
        return self._max_workers

    def set_max_workers(self, max_workers: int) -> None:
        """
        Sets the maximum number of workers for concurrent execution.
        This method can be called to override the default value.
        """
        if max_workers < 1:
            logging.warning(
                f"Given max_workers {max_workers} is less than 1. "
                "Setting max_workers to 1."
            )
            self._max_workers = 1
        else:
            logging.info(f"Setting max_workers to {max_workers}")
            self._max_workers = max_workers

    def _run_single(self, instance: ProblemT):
        runner: RunnerT = self.s_i_runner_class(
            instance=instance,
            shared_param_dict=self.shared_param_dict,
            subroutine_flow=self.subroutine_flow,
            stopping_criteria=self.stopping_criteria,
            output_dir=self.output_dir,
            output_metadata=self.output_metadata,
        )
        self.runners.append(runner)
        try:
            return runner.run()
        except Exception as e:
            logging.error(
                f"Error in instance {getattr(instance, 'name', str(instance))}: {e}"
            )
            return None

    def run(self):
        worker_cnt = self.get_max_workers()
        if worker_cnt < 1:
            raise ValueError("max_workers must be at least 1")
        if worker_cnt == 1:
            # If max_workers is 1, run sequentially
            return super().run()

        self.runners.clear()
        self.results.clear()

        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_cnt) as executor:
            futures = [
                executor.submit(self._run_single, instance)
                for instance in self.instances
            ]
            for future in concurrent.futures.as_completed(futures):
                self.results.append(future.result())

        return self.post_run_process()
