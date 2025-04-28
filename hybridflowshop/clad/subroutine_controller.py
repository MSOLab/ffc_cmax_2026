import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Sequence

from .dynamic_data_object import DynamicDataObject
from .timer import Timer


class SubroutineController(ABC):
    _timer: Timer

    _stopping_criteria: DynamicDataObject
    _subroutine_flow: DynamicDataObject

    _called_routines: list[dict[str, Any]]
    _log_entries: list[dict[str, Any]]

    def __init__(
        self,
        stopping_criteria: DynamicDataObject,
        subroutine_flow: DynamicDataObject,
        start_dt: dt.datetime | None = None,
    ):
        # Set the timer first
        self._timer = Timer()
        if start_dt is not None:
            self._timer.set_start_time(start_dt)
        else:
            self._timer.set_start_time_as_now()

        self._stopping_criteria = stopping_criteria
        self._subroutine_flow = subroutine_flow

        self._called_routines = []
        self._log_entries = []

    @abstractmethod
    def is_stopping_condition(self) -> bool:
        pass

    def _add_called_routine(self, **kwargs):
        self._called_routines.append(kwargs)

    def _add_log_entry(self, **kwargs):
        self._log_entries.append(kwargs)

    def run(self):
        self.execute_routine(self._subroutine_flow)
        self.print_logs()

    def print_logs(self):
        print("\n==== Subroutine Execution Log ====")
        for idx, entry in enumerate(self._log_entries, 1):
            print(
                f"[{idx}] {entry['method_name']} | {entry['elapsed_sec']:.3f} sec | kwargs={entry['kwargs']}"
            )
        print("==================================\n")

    def execute_routine(self, routine_data: DynamicDataObject):
        if isinstance(routine_data, Sequence):  # is a list or tuple
            for subroutine_data in routine_data:
                self.execute_routine(subroutine_data)
        else:  # is an dict-like object
            if self.is_stopping_condition():
                return
            kwargs_dict: dict = routine_data.to_obj()
            method_name = kwargs_dict.pop("method_name")
            self.execute_method(method_name, **kwargs_dict)

    def execute_method(self, method_name: str, **kwargs):
        if not hasattr(self, method_name):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {method_name}"
            )

        method_timer = Timer()
        method_timer.set_start_time_as_now()

        method = getattr(self, method_name)
        method(**kwargs)

        elapsed_sec = method_timer.get_elapsed_sec()
        self._add_log_entry(
            method_name=method_name,
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
