from collections import defaultdict
from pathlib import Path
from typing import Optional

from .solver_status import SolverStatus


class ExperimentSummary:
    """
    Summarizes the entire experimental run for one instance.
    """

    def __init__(self, name: str):
        self.name: str = name
        self.total_elapsed_time_sec: Optional[float] = None

        self.method_call_dict: dict[str, int] = defaultdict(int)

        self.initial_obj: Optional[float] = None
        self.final_obj: Optional[float] = None
        self.initial_lb: Optional[float] = None
        self.final_lb: Optional[float] = None

        self.status: str = SolverStatus.UNKNOWN

    def record_method_call(self, method_name: str):
        """Record a method call and increment its count."""
        self.method_call_dict[method_name] += 1

    def record_initial_solution(self, obj: float, lb: float):
        """Record the initial solution right after base CP solve."""
        self.initial_obj = obj
        self.initial_lb = lb

    def record_final_solution(self, obj: float, lb: float):
        """Record the final solution after all LNS search."""
        self.final_obj = obj
        self.final_lb = lb

    def record_total_elapsed_time(self, elapsed_time_sec: float):
        """Record the total elapsed time."""
        self.total_elapsed_time_sec = elapsed_time_sec

    def record_feasibility(self, feasible: bool):
        """Record whether the final solution is feasible."""
        if feasible:
            self.status = SolverStatus.FEASIBLE
        else:
            self.status = SolverStatus.INFEASIBLE

    @property
    def is_feasible(self) -> bool:
        return SolverStatus.found_feasible_solution(self.status)

    def get_improvement_ratio(self) -> Optional[float]:
        """Calculate and return the improvement ratio."""
        if self.initial_obj is None or self.final_obj is None:
            return None
        if self.initial_obj == 0:
            return None
        return (self.initial_obj - self.final_obj) / self.initial_obj

    def report(self):
        print("\n=== Experiment Summary ===")
        print(f"Instance: {self.name}")
        print(
            f"Total elapsed time: {self.total_elapsed_time_sec:.2f} sec"
            if self.total_elapsed_time_sec is not None
            else "Elapsed time: N/A"
        )
        print(f"Initial objective: {self.initial_obj}")
        print(f"Final objective: {self.final_obj}")
        improvement = self.get_improvement_ratio()
        if improvement is not None:
            print(f"Improvement ratio: {improvement:.2%}")
        else:
            print("Improvement ratio: N/A")
        print(f"Last status: {self.status}")
        print("--- Method Call Counts ---")
        for method, count in sorted(self.method_call_dict.items()):
            print(f"{method}: {count} calls")
        print("===========================\n")

    def save_as_yaml(self, file_path: Path) -> None:
        """Save the experiment summary to a YAML file."""
        import yaml

        data = {
            "instance_name": self.name,
            "total_elapsed_time_sec": self.total_elapsed_time_sec,
            "initial_obj": self.initial_obj,
            "final_obj": self.final_obj,
            "initial_lb": self.initial_lb,
            "final_lb": self.final_lb,
            "improvement_ratio": self.get_improvement_ratio(),
            "status": self.status,
            "is_feasible": self.is_feasible,
            "method_call_counts": dict(self.method_call_dict),
        }

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            raise RuntimeError(f"Error saving ExperimentSummary to {file_path}: {e}")
