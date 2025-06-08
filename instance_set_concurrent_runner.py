import concurrent.futures
from typing import Generic, TypeVar

from instance_set_runner import InstanceSetRunner
from single_instance_runner import SingleInstanceRunner

ProblemT = TypeVar("ProblemT")
RunnerT = TypeVar("RunnerT", bound=SingleInstanceRunner)


class InstanceSetConcurrentRunner(InstanceSetRunner, Generic[ProblemT, RunnerT]):
    def set_max_workers(self, max_workers: int) -> None:
        """
        Sets the maximum number of workers for concurrent execution.
        This method can be called to override the default value.
        """
        self._max_workers = max_workers

    def get_max_workers(self) -> int:
        if hasattr(self, "_max_workers"):
            return self._max_workers
        return 1  # Default value if not set

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
            print(f"Error in instance {getattr(instance, 'name', str(instance))}: {e}")
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
