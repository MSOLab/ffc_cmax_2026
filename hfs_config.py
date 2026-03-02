import os
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field, model_validator


class ScenarioPathConfig(BaseModel):
    """Defines the path configuration for a single experimental scenario."""

    subroutine_flow_rel_path: Path = Field(
        ..., description="Relative path to the subroutine flow YAML file."
    )
    stopping_criteria_rel_path: Path = Field(
        ..., description="Relative path to the stopping criteria YAML file."
    )
    output_dir: Path = Field(
        ..., description="Output directory for this specific scenario."
    )
    description: str | None = Field(
        default=None, description="Optional description for the scenario."
    )


class BaselineColumnMapping(BaseModel):
    """Defines the column name mapping for the baseline data file.

    This class allows flexible configuration of baseline CSV column names,
    enabling compatibility with various baseline file formats.

    Note: The `instance` field uses "Instance" as default (capital I), but this
    should not be confused with:
    - routix's INSTANCE_NAME ("insName") used internally by the algorithm
    - hybridflowshop's INPUT_NAME_COLUMN ("insName") for summary CSVs
    - exp_compare's REF_INSTANCE_ID_COLUMN ("name") for reference CSVs

    When specifying a baseline CSV, ensure the column names in the file match
    the configured mapping values.
    """

    instance: str = Field(
        "Instance", description="Column name for the instance identifier."
    )
    job_cnt: str = Field("n", description="Column name for the job count.")
    stage_cnt: str = Field("s", description="Column name for the stage count.")
    obj_val: str = Field("UB", description="Column name for the objective value (upper bound).")
    obj_bound: str = Field("LB", description="Column name for the objective bound (lower bound).")


class MainMetadata(BaseModel):
    """
    A Pydantic model to validate and manage the structure of main_metadata.yaml.
    """

    # Input data configuration
    pra_common_params_rel_path: Path = Field(
        ..., description="Path to common parameters for PRA benchmarks."
    )
    baseline_csv_path: Path = Field(
        ..., description="Path to the baseline CSV file for comparison."
    )
    input_dir: Path = Field(
        ..., description="Directory containing benchmark instance files."
    )
    benchmark_idx_list: list[int] | None = Field(
        default=None, description="List of instance IDs to run."
    )
    first: int | None = Field(default=None, description="First instance ID to run.")
    last: int | None = Field(default=None, description="Last instance ID to run.")
    benchmark_filename_format: str = Field(
        ..., description="Format string for instance filenames, e.g., '{}.txt'."
    )

    # Column mapping for baseline data
    baseline_column_mapping: BaselineColumnMapping = Field(
        default_factory=lambda: BaselineColumnMapping(
            instance="Instance",
            obj_val="UB",
            obj_bound="LB",
        ),
        description="Mapping for baseline data columns.",
    )

    # Scenario configurations
    dicts_of_i_o_data_path: List[ScenarioPathConfig] = Field(
        ...,
        alias="dicts_of_i_o_data_path",
        description="List of configurations for each scenario.",
    )

    # Output and Logging configuration
    output_dir_scenarios: Path = Field(
        default=Path("Outputs_scenarios"),
        description="Base directory for all scenario outputs.",
    )
    scenario_log_filename: str = Field(
        default="hfs_scenario_runner.log",
        description="Log file name for the multi-scenario run.",
    )
    result_dir_name: str = Field(
        default="results",
        description="Subdirectory name within each instance's output for results.",
    )

    # Execution configuration
    instance_worker_cnt: int = Field(
        default=1, description="Number of concurrent workers for multi-instance runs."
    )

    # Analysis and plotting options
    draw_gantt: bool = Field(default=False, description="Whether to draw Gantt charts.")
    result_gantt_filename_format: str = Field(
        default="{}_gantt.png", description="Filename format for Gantt charts."
    )
    draw_progress_plot: bool = Field(
        default=False, description="Whether to draw objective progress plots."
    )
    progress_plot_filename_format: str = Field(
        default="{}_progress.png", description="Filename format for progress plots."
    )
    drop_first_values_percent: float = Field(
        default=0.0,
        description="Percentage of initial data points to drop in progress plots.",
    )

    # Optional metadata for resume
    resume_dir_path: str | None = Field(
        default=None,
        description="Optional path to a resume directory or resume YAML file containing previous run data.",
    )

    # Optional metadata for post-process only
    analysis_dir_path: str | None = Field(
        default=None,
        description="If a valid directory path is provided, the runner will operate in POST_PROCESS_ONLY mode for that specific run.",
    )
    analysis_timestamp: str | None = Field(
        default=None,
        description="If a valid timestamp string is provided, the runner will operate in POST_PROCESS_ONLY mode for that specific run.",
    )

    @model_validator(mode="after")
    def _check_index_inputs(self):
        """
        Require either:
         - benchmark_idx_list (preferred), or
         - both first and last (with first <= last).
        """
        if self.benchmark_idx_list is None:
            if self.first is None or self.last is None:
                raise ValueError(
                    "Either 'benchmark_idx_list' must be provided, or both 'first' and 'last'."
                )
            if self.first > self.last:
                raise ValueError("'first' must be <= 'last'.")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation with Path objects converted to strings."""
        return self.model_dump(mode="json")

    def get_benchmark_idx_list(self) -> list[int]:
        """Generates a list of benchmark instance indices from benchmark_idx_list or first..last."""
        if self.benchmark_idx_list:
            return self.benchmark_idx_list
        if self.first is not None and self.last is not None and self.first <= self.last:
            return list(range(self.first, self.last + 1))
        raise ValueError("Invalid benchmark index configuration.")

    def get_benchmark_filename_list(self, reverse: bool = False) -> list[str]:
        """Generates a list of benchmark filenames based on the indices and filename format.

        Args:
            reverse (bool, optional): Whether to reverse the list of filenames.
                Defaults to False.

        Returns:
            list[str]: List of benchmark filenames.
        """
        filenames = [
            self.benchmark_filename_format.format(i)
            for i in self.get_benchmark_idx_list()
        ]
        if reverse:
            filenames.reverse()
        return filenames

    def get_analysis_dir_path(self) -> Path | None:
        """Returns the analysis directory path if specified, otherwise None."""
        if not self.analysis_dir_path:
            return None
        expanded = os.path.expandvars(self.analysis_dir_path)
        return Path(expanded).expanduser()
