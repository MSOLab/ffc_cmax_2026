from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from ortools.sat.python.cp_model import CpSolver


@dataclass(frozen=True)
class SolveConfig:
    log_search_progress: bool | None = None
    log_to_stdout: bool | None = None
    log_to_response: bool | None = None
    max_time_in_seconds: float | None = None
    num_workers: int | None = None
    keep_all_feasible_solutions_in_presolve: bool | None = None
    random_seed: int | None = None
    encode_cumulative_as_reservoir: bool | None = None
    expand_reservoir_constraints: bool | None = None
    expand_reservoir_using_circuit: bool | None = None
    interleave_search: bool | None = None
    use_lns_only: bool | None = None
    cp_model_probing_level: int | None = None
    cp_sat_params: Mapping[str, Any] | None = None

    def get_dict(self) -> dict[str, Any]:
        params = asdict(self)
        extra_params = params.pop("cp_sat_params", None) or {}
        if hasattr(extra_params, "to_obj"):
            extra_params = extra_params.to_obj()
        if not isinstance(extra_params, Mapping):
            raise TypeError(
                "cp_sat_params must be a mapping of CP-SAT parameter names to values."
            )
        resolved = {k: v for k, v in params.items() if v is not None}
        resolved.update({k: v for k, v in dict(extra_params).items() if v is not None})
        return resolved


def configure_solver(cfg: SolveConfig) -> CpSolver:
    s = CpSolver()
    for k, v in cfg.get_dict().items():
        if k not in s.parameters.DESCRIPTOR.fields_by_name:
            raise ValueError(f"Unknown CP-SAT parameter: {k!r}")
        current_value = getattr(s.parameters, k)
        if hasattr(current_value, "extend") and not isinstance(v, (str, bytes)):
            if not isinstance(v, Sequence):
                raise TypeError(
                    f"CP-SAT parameter {k!r} expects a sequence, got {type(v).__name__}."
                )
            del current_value[:]
            current_value.extend(v)
        else:
            setattr(s.parameters, k, v)
    return s
