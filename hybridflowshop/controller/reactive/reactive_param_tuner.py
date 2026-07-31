from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from routix.util.comparison import float_a_leq_b, float_a_stl_b


@dataclass(frozen=True)
class TunerParams:
    step_size: float
    min: float
    max: float

    def decrement(self, current: float) -> float:
        new_val = current - self.step_size
        if float_a_stl_b(new_val, self.min):
            return self.min
        return new_val

    def increment(self, current: float) -> float:
        new_val = current + self.step_size
        if float_a_stl_b(self.max, new_val):
            return self.max
        return new_val


class ReactiveParamTuner:
    _method: Callable[..., Any]
    _tuner_param_dict: dict[str, TunerParams]
    current_kwargs: dict[str, Any]

    def __init__(
        self,
        method: Callable[..., Any],
        opening_kwargs: dict[str, Any],
        tuner_param_dict: dict[str, TunerParams],
    ):
        self._method = method
        self._tuner_param_dict = tuner_param_dict
        self.current_kwargs = deepcopy(opening_kwargs)

    def get_current_value(self, param_name: str) -> Any:
        if param_name not in self.current_kwargs:
            raise ValueError(f"Parameter {param_name} is not in current kwargs.")
        return self.current_kwargs[param_name]

    def current_value_hits_ub(self, param_name: str) -> bool:
        if param_name not in self._tuner_param_dict:
            raise ValueError(f"Parameter {param_name} is not tunable.")
        if param_name not in self.current_kwargs:
            raise ValueError(f"Parameter {param_name} is not in current kwargs.")
        tuner = self._tuner_param_dict[param_name]
        return float_a_leq_b(tuner.max, self.current_kwargs[param_name])

    def current_value_exceeds_ub(self, param_name: str) -> bool:
        if param_name not in self._tuner_param_dict:
            raise ValueError(f"Parameter {param_name} is not tunable.")
        if param_name not in self.current_kwargs:
            raise ValueError(f"Parameter {param_name} is not in current kwargs.")
        tuner = self._tuner_param_dict[param_name]
        return float_a_stl_b(tuner.max, self.current_kwargs[param_name])

    def call_method(
        self,
        *,
        capped_time_param_name: str | None = None,
        capped_time_param_value: float | None = None,
    ) -> Any:
        if capped_time_param_name is not None and capped_time_param_value is not None:
            if capped_time_param_name in self.current_kwargs:
                self.current_kwargs[capped_time_param_name] = min(
                    self.current_kwargs[capped_time_param_name],
                    capped_time_param_value,
                )
            else:
                self.current_kwargs[capped_time_param_name] = capped_time_param_value
        return self._method(**self.current_kwargs)

    def decrement(self, param_name: str) -> None:
        if param_name not in self._tuner_param_dict:
            raise ValueError(f"Parameter {param_name} is not tunable.")
        if param_name not in self.current_kwargs:
            raise ValueError(f"Parameter {param_name} is not in current kwargs.")
        tuner = self._tuner_param_dict[param_name]
        self.current_kwargs[param_name] = tuner.decrement(
            self.current_kwargs[param_name]
        )

    def increment(self, param_name: str) -> None:
        if param_name not in self._tuner_param_dict:
            raise ValueError(f"Parameter {param_name} is not tunable.")
        if param_name not in self.current_kwargs:
            raise ValueError(f"Parameter {param_name} is not in current kwargs.")
        tuner = self._tuner_param_dict[param_name]
        self.current_kwargs[param_name] = tuner.increment(
            self.current_kwargs[param_name]
        )
