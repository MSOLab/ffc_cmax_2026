from dataclasses import dataclass
from typing import Optional

from .solver_status import SolverStatus


@dataclass
class SolverOutputSummary:
    status: str
    elapsed_time: float
    objective_value: Optional[int]
    best_objective_bound: Optional[int]
    progress_log: Optional[list[tuple[float, float, float]]]

    def report_objective_value(self):
        print(f"Obj. value: {self.objective_value}, bound: {self.best_objective_bound}")

    def report_status(self):
        print(f"Solver status: {self.status}")
        if SolverStatus.found_feasible_solution(self.status):
            self.report_objective_value()

    def comma_separated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return (
            f"{self.status},{self.elapsed_time}"
            f",{self.objective_value},{self.best_objective_bound}"
        )
