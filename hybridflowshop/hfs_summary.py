from dataclasses import dataclass
from pathlib import Path


@dataclass
class HFSInputSummary:
    name: str
    num_jobs: int
    num_stages: int
    computational_time: float
    n_threads: int

    def comma_seperated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return f"{self.name},{self.num_jobs},{self.num_stages},{self.computational_time},{self.n_threads}"


@dataclass
class HFSOutputSummary:
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
        return f"{self.status},{self.objective_value},{self.best_objective_bound},{self.elapsed_time}"


class HFSSummary:
    inputs: HFSInputSummary
    outputs: HFSOutputSummary

    def __init__(self, inputs: HFSInputSummary, outputs: HFSOutputSummary):
        self.inputs = inputs
        self.outputs = outputs

    def comma_seperated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return (
            f"{self.inputs.comma_seperated_values()}"
            f",{self.outputs.comma_seperated_values()}"
        )

    def save(self, output_path: Path):
        # make sure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # write data
        # if the file already exists, append to the file
        if output_path.exists():
            with open(output_path, "a") as f:
                f.write(self.comma_seperated_values())
        else:
            with open(output_path, "w") as f:
                f.write(self.comma_seperated_values())
