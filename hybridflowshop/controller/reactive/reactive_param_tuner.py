from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TunerParams:
    step_size: float
    min: float
    max: float

    def decrement(self, current: float) -> float:
        new_val = current - self.step_size
        if new_val < self.min:
            return self.min
        return new_val

    def increment(self, current: float) -> float:
        new_val = current + self.step_size
        if new_val > self.max:
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

    def call_method(self, timelimit_by_global: float | None = None) -> Any:
        if timelimit_by_global is not None:
            if "computational_time" in self.current_kwargs:
                self.current_kwargs["computational_time"] = min(
                    self.current_kwargs["computational_time"],
                    timelimit_by_global,
                )
            else:
                self.current_kwargs["computational_time"] = timelimit_by_global
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
