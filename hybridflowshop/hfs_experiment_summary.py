from dataclasses import field
from typing import Any

from mbls import ExperimentSummary, SolverStatus

from .hfs_solver_output_summary import HfsSolverOutputSummary


class HfsExperimentSummary(ExperimentSummary):
    runs: list[HfsSolverOutputSummary] = field(default_factory=list)

    def get_init_summary(
        self, is_maximize: bool = False
    ) -> HfsSolverOutputSummary | None:
        # Find feasible runs
        feasible_runs = [
            run for run in self.runs if SolverStatus.found_feasible_solution(run.status)
        ]
        # If no feasible runs, return None
        if not feasible_runs:
            return None

        # Find runs with is_init=True
        init_runs = [run for run in feasible_runs if run.is_init]
        if not init_runs:
            # If no initial runs are found, return the first feasible run
            return feasible_runs[0]

        # Find the best run
        if is_maximize:
            return max(
                init_runs,
                key=lambda run: run.objective_value
                if run.objective_value is not None
                else float("-inf"),
            )
        return min(
            init_runs,
            key=lambda run: run.objective_value
            if run.objective_value is not None
            else float("inf"),
        )

    def get_improvement_ratio(self, is_maximize: bool = False) -> float | None:
        init = self.get_init_summary()
        if is_maximize:
            best = self.get_summary_maximum_obj()
        else:
            best = self.get_summary_minimum_obj()

        if not (
            init
            and best
            and init.objective_value is not None
            and best.objective_value is not None
        ):
            return None
        if init.objective_value == 0:
            return None

        if is_maximize:
            return (best.objective_value - init.objective_value) / init.objective_value
        return (init.objective_value - best.objective_value) / init.objective_value

    def to_dict(self, is_maximize: bool = False) -> dict[str, Any]:
        if is_maximize:
            best = self.get_summary_maximum_obj()
        else:
            best = self.get_summary_minimum_obj()

        init_summary = self.get_init_summary(is_maximize=is_maximize)
        if init_summary:
            init_obj = init_summary.objective_value
        else:
            init_obj = best.objective_value if best else None

        return {
            "instanceName": self.name,
            "foundFeasibleSol": self.found_feasible_solution(is_maximize),
            "totalElapsedTime": self.get_total_elapsed_time(),
            "status": best.status if best else None,
            "initObj": init_obj,
            "bestObj": best.objective_value if best else None,
            "bestBound": best.best_objective_bound if best else None,
            "improvementRatio": self.get_improvement_ratio(is_maximize),
            "methodCallCounts": f'"{self.method_call_counts}"',
            "numRuns": len(self.runs),
        }
