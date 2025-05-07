from pathlib import Path

from clad import SolverOutputSummary

from .hfs_input_summary import HFSInputSummary


class HFSSummary:
    inputs: HFSInputSummary
    outputs: SolverOutputSummary

    def __init__(self, inputs: HFSInputSummary, outputs: SolverOutputSummary):
        self.inputs = inputs
        self.outputs = outputs

    def comma_seperated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return (
            f"\n{self.inputs.comma_separated_values()}"
            f",{self.outputs.comma_separated_values()}"
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
