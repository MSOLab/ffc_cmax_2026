"""I/O utilities for loading and saving comparison data."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from pydantic import BaseModel

from exp_compare.constants import EXP_INSTANCE_ID_COLUMN


class RunConfig(BaseModel):
    """Configuration for a single run to compare."""

    path: str
    summary_csv: str = "all_scenarios_summary.csv"


class ReferenceConfig(BaseModel):
    """Configuration for reference values."""

    mode: str = "best_among_compared"
    sense: str = "min"
    ref_path: str | None = None
    ref_format: str = "csv"
    instance_key_column_in_ref: str = "name"
    reference_value_column: str | None = None


class OutputConfig(BaseModel):
    """Configuration for output files."""

    out_dir: str = "Outputs_analysis/compare_$TIMESTAMP/"
    basename: str = "rpd_compare"

    def resolve_timestamps(self) -> "OutputConfig":
        """Replace $TIMESTAMP placeholder with current datetime (YYYYMMDD_HHMMSS format).

        Returns:
            New OutputConfig with $TIMESTAMP replaced by current datetime.
        """
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_out_dir = self.out_dir.replace("$TIMESTAMP", current_timestamp)
        return OutputConfig(out_dir=resolved_out_dir, basename=self.basename)


class CompareConfig(BaseModel):
    """Main comparison configuration."""

    runs: Sequence[RunConfig]
    reference: ReferenceConfig
    output: OutputConfig


def load_run_summaries(
    runs: Sequence[RunConfig], exp_instance_id_col: str = EXP_INSTANCE_ID_COLUMN
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
    """Load summary CSVs from multiple runs.

    Args:
        runs (Sequence[RunConfig]): specifies paths and CSV filenames.
        exp_instance_id_col (str, optional): Column name for instance IDs in experiment CSVs.
            Defaults to EXP_INSTANCE_ID_COLUMN.

    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
            - {run_id: DataFrame} with all scenario data concatenated
            - {run_id: set of instance names}
    """
    dataframes: dict[str, pd.DataFrame] = {}
    instance_names: dict[str, set[str]] = {}

    for run in runs:
        run_path = Path(run.path)
        if not run_path.exists():
            logging.warning(f"Run directory not found: {run.path}")
            continue

        summary_path = run_path / run.summary_csv
        if not summary_path.exists():
            logging.warning(f"Summary CSV not found: {summary_path}")
            continue

        try:
            df = pd.read_csv(summary_path)
        except Exception as e:
            logging.warning(f"Failed to read {summary_path}: {e}")
            continue

        # Extract run_id from path
        run_id = Path(run.path).name.rstrip("/\\")
        if run_id.endswith("/"):
            run_id = run_id[:-1]
        run_id = Path(run_id).name

        logging.info(f"Loaded {len(df)} rows from {run_id}")

        # Ensure name column is string type for consistent comparison
        if exp_instance_id_col in df.columns:
            df = df.copy()
            df[exp_instance_id_col] = df[exp_instance_id_col].astype(str)

        # Store dataframe
        dataframes[run_id] = df

        # Extract instance names
        if exp_instance_id_col in df.columns:
            names = set(df[exp_instance_id_col].dropna().unique())
            instance_names[run_id] = names
            logging.info(f"  Found {len(names)} unique instances in {run_id}")
        else:
            logging.warning(
                f"  '{exp_instance_id_col}' column not found in {summary_path}"
            )
            instance_names[run_id] = set()

    return dataframes, instance_names


def compute_intersection(instance_names: Mapping[str, set[str]]) -> set[str]:
    """Compute intersection of instance names across all runs.

    Args:
        instance_names (Mapping[str, set[str]]): run_id -> set of instance names.

    Returns:
        set[str]: Set of instance names present in all runs.
    """
    if not instance_names:
        return set()

    names_iter = iter(instance_names.values())
    intersection = next(names_iter, set())
    for names in names_iter:
        intersection = intersection & names

    return intersection


def load_reference_csv(
    ref_path: Path,
    instance_key_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Load reference CSV file.

    Args:
        ref_path (Path): Path to the reference CSV file.
        instance_key_col (str): Column name for instance keys in reference.
        value_col (str): Column name for reference values.

    Raises:
        FileNotFoundError: If the reference file does not exist.

    Returns:
        pd.DataFrame: DataFrame with instance_key_col and value_col.
    """
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    df = pd.read_csv(ref_path)
    # Ensure instance_key_col is string for consistent comparison with combined_df[name]
    df = df.copy()
    df[instance_key_col] = df[instance_key_col].astype(str)
    return df[[instance_key_col, value_col]].copy()
