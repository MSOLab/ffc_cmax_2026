import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Sequence

from .dynamic_data_object import DynamicDataObject
from .elapsed_timer import ElapsedTimer


class SubroutineController(ABC):
    _timer: ElapsedTimer

    _stopping_criteria: DynamicDataObject
    _subroutine_flow: DynamicDataObject

    _method_call_logs: list[dict[str, Any]]

    def __init__(
        self,
        stopping_criteria: DynamicDataObject,
        subroutine_flow: DynamicDataObject,
        start_dt: dt.datetime | None = None,
    ):
        # Set the timer first
        self._timer = ElapsedTimer()
        if start_dt is not None:
            self._timer.set_start_time(start_dt)
        else:
            self._timer.set_start_time_as_now()

        self._stopping_criteria = stopping_criteria
        self._subroutine_flow = subroutine_flow

        self._method_call_logs = []

    @property
    def timer(self) -> ElapsedTimer:
        return self._timer

    @abstractmethod
    def is_stopping_condition(self) -> bool:
        pass

    def _add_method_call_log_entry(self, **kwargs):
        self._method_call_logs.append(kwargs)

    def run(self):
        self.execute_routine(self._subroutine_flow)
        self.print_method_exec_logs()

    def print_method_exec_logs(self):
        print("\n=== Method Execution Log ===")
        for idx, entry in enumerate(self._method_call_logs, 1):
            print(
                f"[{idx}] {entry['method_name']} | start @ {entry['start_sec']:.3f} sec"
                f" | took {entry['elapsed_sec']:.3f} sec | kwargs={entry['kwargs']}"
            )
        print(28 * "=" + "\n")

    def execute_routine(self, routine_data: DynamicDataObject):
        if isinstance(routine_data, Sequence):  # is a list or tuple
            for subroutine_data in routine_data:
                self.execute_routine(subroutine_data)
        else:  # is an dict-like object
            if self.is_stopping_condition():
                return
            kwargs_dict: dict = routine_data.to_obj()
            method_name = kwargs_dict.pop("method_name")
            self.call_method(method_name, **kwargs_dict)

    def call_method(self, method_name: str, **kwargs):
        """Call a method by its name and log the execution time.

        Args:
            method_name (str): The name of the method to call.
            **kwargs: Any: Additional keyword arguments to pass to the method.

        Raises:
            AttributeError: If the method does not exist in the class.
        """
        if not hasattr(self, method_name):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {method_name}"
            )
        method_start_sec = self.timer.get_elapsed_sec()

        try:
            getattr(self, method_name)(**kwargs)
        except Exception as e:
            print(f"[Error] Method {method_name} failed: {e}")
            self._add_method_call_log_entry(
                method_name=method_name,
                start_sec=method_start_sec,
                elapsed_sec=0,
                kwargs=kwargs,
                error=str(e),
            )
            raise

        elapsed_sec = self.timer.get_elapsed_sec() - method_start_sec
        self._add_method_call_log_entry(
            method_name=method_name,
            start_sec=method_start_sec,
            elapsed_sec=elapsed_sec,
            kwargs=kwargs,
        )

    def repeat(self, n_repeats: int, routine_data: Any):
        """
        Repeat a subroutine flow n_repeats times.

        Args:
            n_repeats (int): Number of repetitions
            routine_data (Any): A single subroutine or a list of subroutines
        """
        for i in range(n_repeats):
            if self.is_stopping_condition():
                print(
                    f"[Repeat] Stopping condition met at iteration {i+1}/{n_repeats}."
                )
                break
            print(f"[Repeat] Starting repeat {i+1}/{n_repeats}")
            ddo = DynamicDataObject.from_obj(routine_data)
            self.execute_routine(ddo)
