from dataclasses import asdict, dataclass
from typing import Any

from ortools.sat.python.cp_model import CpSolver


@dataclass(frozen=True)
class SolveConfig:
    log_search_progress: bool | None = None
    max_time_in_seconds: float | None = None
    num_workers: int | None = None
    keep_all_feasible_solutions_in_presolve: bool | None = None
    random_seed: int | None = None

    def get_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def configure_solver(cfg: SolveConfig) -> CpSolver:
    s = CpSolver()
    for k, v in cfg.get_dict().items():
        setattr(s.parameters, k, v)
    return s
