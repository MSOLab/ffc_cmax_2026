import datetime as dt
from typing import Any, Sequence

from .dynamic_data_object import DynamicDataObject
from .timer import Timer


class SubroutineController:
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

    def is_stopping_condition(self) -> bool:
        raise NotImplementedError()

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
            # if the object has a function with the name of subroutine_flow.name, call it
            if hasattr(routine_data, "name"):
                try:
                    self._add_called_routine(**routine_data.to_obj())
                    getattr(self, routine_data.name)(**routine_data.to_obj())
                except AttributeError:
                    raise AttributeError(
                        f"SubroutineController has no attribute {routine_data.name}"
                    )
            else:
                raise AttributeError("Subroutine data must have a 'name' attribute")
