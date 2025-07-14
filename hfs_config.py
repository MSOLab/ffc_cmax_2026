from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


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
    )  # TODO: utilize this in the code
    input_dir: Path = Field(
        ..., description="Directory containing benchmark instance files."
    )
    first: int = Field(..., description="First instance ID to run.")
    last: int = Field(..., description="Last instance ID to run.")
    benchmark_filename_format: str = Field(
        ..., description="Format string for instance filenames, e.g., '{}.txt'."
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
    worker_cnt: int = Field(
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

    # Optional metadata for post-process only
    analysis_timestamp: str | None = Field(
        default=None,
        description="If a valid timestamp string is provided, the runner will operate in POST_PROCESS_ONLY mode for that specific run.",
    )
