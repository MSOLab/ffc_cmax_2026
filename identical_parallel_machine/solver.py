from dataclasses import dataclass
from typing import Any

from ortools.sat.python.cp_model import CpSolver


@dataclass(frozen=True)
class SolveConfig:
    log_search_progress: bool | None = None
    time_limit_s: float | None = None
    num_workers: int | None = None
    keep_all_feasible_solutions_in_presolve: bool | None = None
    random_seed: int | None = None

    def get_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = {}
        if self.log_search_progress is not None:
            return_dict["log_search_progress"] = self.log_search_progress
        if self.time_limit_s is not None:
            return_dict["max_time_in_seconds"] = float(self.time_limit_s)
        if self.num_workers is not None:
            return_dict["num_workers"] = int(self.num_workers)
        if self.keep_all_feasible_solutions_in_presolve is not None:
            return_dict["keep_all_feasible_solutions_in_presolve"] = (
                self.keep_all_feasible_solutions_in_presolve
            )
        if self.random_seed is not None:
            return_dict["random_seed"] = int(self.random_seed)
        return return_dict


def configure_solver(cfg: SolveConfig) -> CpSolver:
    s = CpSolver()
    for k, v in cfg.get_dict().items():
        setattr(s.parameters, k, v)
    return s
