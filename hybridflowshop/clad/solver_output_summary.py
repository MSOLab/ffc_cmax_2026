from dataclasses import dataclass


@dataclass
class SolverOutputSummary:
    status: str
    objective_value: int | None
    best_objective_bound: int | None
    elapsed_time: float

    def report_objective_value(self):
        print(f"LB: {self.best_objective_bound}, UB: {self.objective_value}")

    def report_status(self):
        if self.status == "OPTIMAL":
            print("Optimal solution found.")
            self.report_objective_value()
        elif self.status == "FEASIBLE":
            print("Feasible solution found.")
            self.report_objective_value()
        else:
            print("No feasible solution found.")

    def comma_seperated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return (
            f"{self.status},{self.objective_value},"
            f"{self.best_objective_bound},{self.elapsed_time}"
        )
