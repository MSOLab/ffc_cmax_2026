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

    def __init__(
        self,
        stopping_criteria: DynamicDataObject,
        subroutine_flow: DynamicDataObject,
        start_dt: dt.datetime | None = None,
    ):
        self._stopping_criteria = stopping_criteria
        self._subroutine_flow = subroutine_flow

        self._called_routines = []
        self._timer = Timer()
        if start_dt is not None:
            self._timer.set_start_time(start_dt)
        else:
            self._timer.set_start_time_as_now()

    @abstractmethod
    def is_stopping_condition(self) -> bool:
        pass

    def _add_called_routine(self, **kwargs):
        self._called_routines.append(kwargs)

    def run(self):
        self.execute_routine(self._subroutine_flow)

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
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            method(**kwargs)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {method_name}"
            )
