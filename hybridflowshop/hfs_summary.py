from pathlib import Path

from cplnx import ExperimentSummary

from .hfs_input_summary import HFSInputSummary


class HFSSummary:
    inputs: HFSInputSummary
    outputs: ExperimentSummary

    def __init__(self, inputs: HFSInputSummary, outputs: ExperimentSummary):
        self.inputs = inputs
        self.outputs = outputs

    def comma_separated_values_header(self) -> str:
        """Returns the header for the comma-separated values."""
        inputs_header_str = self.inputs.header()
        outputs_headers = self.outputs.to_dict().keys()
        outputs_header_str = ",".join(str(header) for header in outputs_headers)
        return f"{inputs_header_str},{outputs_header_str}"

    def comma_seperated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        inputs_value_str = self.inputs.comma_separated_values()
        outputs_values = self.outputs.to_dict().values()
        outputs_value_str = ",".join(str(value) for value in outputs_values)
        return f"\n{inputs_value_str},{outputs_value_str}"

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
                f.write(self.comma_separated_values_header())
                f.write(self.comma_seperated_values())
