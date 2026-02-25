"""Main comparison logic and CSV output generation."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from exp_compare.constants import (
    ALL_RESULT_COLUMNS,
    EXP_OBJ_VALUE_COLUMN,
    RESULT_ALGO_UID_COLUMN,
    RESULT_EXP_OBJ_VALUE_COLUMN,
    RESULT_INSTANCE_ID_COLUMN,
    RESULT_RANK_COLUMN,
    RESULT_REF_OBJ_VALUE_COLUMN,
    RESULT_RPDF_COLUMN,
    RESULT_RPDV_COLUMN,
    RESULT_RUN_ID_COLUMN,
    RESULT_SCENARIO_COLUMN,
)
from exp_compare.io import (
    CompareConfig,
    ReferenceConfig,
    compute_intersection,
    load_reference_csv,
    load_run_summaries,
)
from exp_compare.metrics import compute_metrics_for_run


def build_long_format(
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build long-format DataFrame for output.

    Args:
        combined_df (pd.DataFrame): Dataframe with all combined data and computed metrics.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns: name, runId, scenario, algoUid,
                      objValue, refValue, RPDf, RPDv, rank, and metadata columns.
    """
    # Get all columns from ALL_RESULT_COLUMNS that exist in the dataframe
    result_cols = [col for col in ALL_RESULT_COLUMNS if col in combined_df.columns]

    # Add metadata columns (columns that are not in ALL_RESULT_COLUMNS and not internal columns)
    # Maintain the order from combined_df.columns
    internal_cols = {RESULT_RUN_ID_COLUMN, RESULT_SCENARIO_COLUMN}
    metadata_cols = [
        col
        for col in combined_df.columns
        if col not in result_cols
        and col not in internal_cols
        and col != RESULT_INSTANCE_ID_COLUMN
    ]

    # Build final column order: name first, then metadata, then standard result columns
    final_cols = (
        [RESULT_INSTANCE_ID_COLUMN]
        + metadata_cols
        + [col for col in result_cols if col != RESULT_INSTANCE_ID_COLUMN]
    )

    return combined_df[final_cols]


def build_wide_rpdf(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Build wide-format DataFrame with relative percentage difference values.

    Args:
        combined_df (pd.DataFrame): Dataframe with all combined data and computed metrics.

    Returns:
        pd.DataFrame: Wide-format DataFrame with columns: name, metadata columns (if any),
                      and RPDf values per algoUid.
    """
    # Get metadata columns
    non_instance_cols = {
        RESULT_INSTANCE_ID_COLUMN,
        RESULT_ALGO_UID_COLUMN,
        RESULT_RUN_ID_COLUMN,
        RESULT_SCENARIO_COLUMN,
        RESULT_RPDF_COLUMN,
        RESULT_RPDV_COLUMN,
        RESULT_EXP_OBJ_VALUE_COLUMN,
        RESULT_REF_OBJ_VALUE_COLUMN,
        RESULT_RANK_COLUMN,
    }
    metadata_cols = [col for col in combined_df.columns if col not in non_instance_cols]

    # Pivot only the RPDf values (use only relevant columns to avoid aggfunc issues)
    pivot_data = combined_df[
        [RESULT_INSTANCE_ID_COLUMN, RESULT_ALGO_UID_COLUMN, RESULT_RPDF_COLUMN]
    ]
    wide = pivot_data.pivot_table(
        index=RESULT_INSTANCE_ID_COLUMN,
        columns=RESULT_ALGO_UID_COLUMN,
        values=RESULT_RPDF_COLUMN,
    ).reset_index()

    # Merge metadata columns if present (at the end initially)
    if metadata_cols:
        metadata_df = combined_df[
            [RESULT_INSTANCE_ID_COLUMN] + metadata_cols
        ].drop_duplicates()
        wide = wide.merge(metadata_df, on=RESULT_INSTANCE_ID_COLUMN, how="left")

    # Reorder columns: name, metadata (preserving config order), then algoUid columns
    # Get all algoUid columns (all columns between name and metadata)
    algo_cols = [
        col
        for col in wide.columns
        if col not in [RESULT_INSTANCE_ID_COLUMN] + metadata_cols
    ]
    final_cols = [RESULT_INSTANCE_ID_COLUMN] + metadata_cols + algo_cols

    return wide[final_cols]


def build_wide_rpdv(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Build wide-format DataFrame with relative percentage deviation values.

    Args:
        combined_df (pd.DataFrame): Dataframe with all combined data and computed metrics.

    Returns:
        pd.DataFrame: Wide-format DataFrame with columns: name, metadata columns (if any),
                      and RPDv values per algoUid.
    """
    # Get metadata columns
    non_instance_cols = {
        RESULT_INSTANCE_ID_COLUMN,
        RESULT_ALGO_UID_COLUMN,
        RESULT_RUN_ID_COLUMN,
        RESULT_SCENARIO_COLUMN,
        RESULT_RPDV_COLUMN,
        RESULT_RPDF_COLUMN,
        RESULT_EXP_OBJ_VALUE_COLUMN,
        RESULT_REF_OBJ_VALUE_COLUMN,
        RESULT_RANK_COLUMN,
    }
    metadata_cols = [col for col in combined_df.columns if col not in non_instance_cols]

    # Pivot only the RPDv values (use only relevant columns to avoid aggfunc issues)
    pivot_data = combined_df[
        [RESULT_INSTANCE_ID_COLUMN, RESULT_ALGO_UID_COLUMN, RESULT_RPDV_COLUMN]
    ]
    wide = pivot_data.pivot_table(
        index=RESULT_INSTANCE_ID_COLUMN,
        columns=RESULT_ALGO_UID_COLUMN,
        values=RESULT_RPDV_COLUMN,
    ).reset_index()

    # Merge metadata columns if present (at the end initially)
    if metadata_cols:
        metadata_df = combined_df[
            [RESULT_INSTANCE_ID_COLUMN] + metadata_cols
        ].drop_duplicates()
        wide = wide.merge(metadata_df, on=RESULT_INSTANCE_ID_COLUMN, how="left")

    # Reorder columns: name, metadata, then algoUid columns
    algo_cols = [
        col
        for col in wide.columns
        if col not in [RESULT_INSTANCE_ID_COLUMN] + metadata_cols
    ]
    final_cols = [RESULT_INSTANCE_ID_COLUMN] + metadata_cols + algo_cols

    return wide[final_cols]


def build_summary_rpdf(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary statistics DataFrame for RPDf.

    Args:
        wide_df (pd.DataFrame): Wide-format DataFrame with RPDf values.

    Returns:
        pd.DataFrame: Summary DataFrame with count, min, max, mean, median, std per algoUid.
    """
    # Exclude non-algorithm columns (name and any metadata columns)
    algo_cols = [
        col
        for col in wide_df.columns
        if col not in [RESULT_INSTANCE_ID_COLUMN, RESULT_ALGO_UID_COLUMN]
    ]

    summary_rows = []
    for col in algo_cols:
        col_data = wide_df[col]
        # Skip non-numeric columns (like metadata)
        if not pd.api.types.is_numeric_dtype(col_data):
            continue
        summary_rows.append(
            {
                RESULT_ALGO_UID_COLUMN: col,
                "count": col_data.count(),
                "min": col_data.min(),
                "max": col_data.max(),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
            }
        )

    return pd.DataFrame(summary_rows)


def build_summary_rpdv(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary statistics DataFrame for RPDv.

    Args:
        wide_df (pd.DataFrame): Wide-format DataFrame with RPDv values.

    Returns:
        pd.DataFrame: Summary DataFrame with count, min, max, mean, median, std per algoUid.
    """
    return build_summary_rpdf(wide_df)


def determine_reference_values(
    combined_df: pd.DataFrame,
    reference_config: ReferenceConfig,
    intersection_names: set[str],
    exp_obj_value_col: str = RESULT_EXP_OBJ_VALUE_COLUMN,
) -> tuple[pd.Series, pd.DataFrame | None, list[str]]:
    """Determine reference values based on mode.

    Args:
        combined_df (pd.DataFrame): Combined DataFrame from all experiment runs.
        reference_config (ReferenceConfig): Reference configuration.
        intersection_names (set[str]): Set of instance names in intersection.
        exp_obj_value_col (str, optional): Column name for objective values in combined_df.
            Defaults to RESULT_EXP_OBJ_VALUE_COLUMN.

    Returns:
        tuple[pd.Series, pd.DataFrame | None, list[str]]: Tuple of (reference_values, metadata_df, metadata_cols).
            reference_values is a Series indexed by instance name.
            metadata_df contains instance metadata (if any).
            metadata_cols is the list of metadata column names in config order.
    """
    mode = reference_config.mode
    sense = reference_config.sense
    metadata_cols = reference_config.instance_metadata_columns

    if mode == "best_among_compared":
        # For each instance, take minimum objective value across all compared algorithms
        best_obj_df = combined_df[
            combined_df[RESULT_INSTANCE_ID_COLUMN].isin(intersection_names)
        ][[RESULT_INSTANCE_ID_COLUMN, exp_obj_value_col]]
        if sense == "max":
            ref_values = best_obj_df.groupby(RESULT_INSTANCE_ID_COLUMN)[
                exp_obj_value_col
            ].max()
        else:  # Default to "min"
            ref_values = best_obj_df.groupby(RESULT_INSTANCE_ID_COLUMN)[
                exp_obj_value_col
            ].min()

        # No metadata available in best_among_compared mode
        metadata_df: pd.DataFrame | None = None

    elif mode == "fixed_dataset":
        ref_path = Path(reference_config.ref_path)  # type: ignore[misc]
        instance_key_col = reference_config.instance_key_column_in_ref
        value_col = reference_config.reference_value_column

        if not ref_path.exists():
            logging.warning(f"Reference file not found: {ref_path}")
            return pd.Series(dtype=float), None, []

        if value_col is None:
            value_col = exp_obj_value_col  # Default to same column as experiment objective values

        ref_df, metadata_df = load_reference_csv(
            ref_path, instance_key_col, value_col, metadata_cols
        )

        # Create series indexed by name
        # ref_df still has the original instance_key_col as the column name
        ref_values = ref_df.set_index(instance_key_col)[value_col]

        # Filter to intersection names, keeping NaN for missing
        ref_values = ref_values.reindex(list(intersection_names))
        # Convert index to string for consistent merge
        ref_values.index = ref_values.index.astype(str)

        # Reindex metadata_df as well
        if metadata_df is not None:
            metadata_df = metadata_df.set_index(RESULT_INSTANCE_ID_COLUMN).reindex(
                list(intersection_names)
            )
            metadata_df = metadata_df.reset_index()
            # Ensure name column is string for consistent merge
            metadata_df[RESULT_INSTANCE_ID_COLUMN] = metadata_df[
                RESULT_INSTANCE_ID_COLUMN
            ].astype(str)

    else:
        logging.warning(f"Unknown reference mode: {mode}, using best_among_compared")
        best_obj_df = combined_df[
            combined_df[RESULT_INSTANCE_ID_COLUMN].isin(intersection_names)
        ][[RESULT_INSTANCE_ID_COLUMN, exp_obj_value_col]]
        ref_values = best_obj_df.groupby(RESULT_INSTANCE_ID_COLUMN)[
            exp_obj_value_col
        ].min()
        metadata_df = None

    return ref_values, metadata_df, metadata_cols


def run_comparison(
    config: CompareConfig, exp_obj_value_col: str = EXP_OBJ_VALUE_COLUMN
) -> int:
    """Run the comparison analysis.

    Args:
        config (CompareConfig): CompareConfig with all settings.
        exp_obj_value_col (str): Column name for objective values in input data.
            Defaults to EXP_OBJ_VALUE_COLUMN.

    Returns:
        int: Exit code: 0 for success, 2 for config/file errors, 1 for other errors.
    """
    try:
        # Resolve timestamp placeholders in output directory
        output_config = config.output.resolve_timestamps()

        # Load all run summaries
        run_id_2_df, run_id_2_ins_set = load_run_summaries(config.runs)

        if not run_id_2_df:
            logging.error("No valid run data loaded")
            return 2

        # Compute intersection
        intersection_names = compute_intersection(run_id_2_ins_set)

        if not intersection_names:
            logging.warning("Intersection of instance names is empty")
            logging.warning("No files generated, exiting normally")
            return 0

        logging.info(f"Intersection contains {len(intersection_names)} instances")

        # Combine all dataframes with runId and scenario info
        combined_rows: list[pd.DataFrame] = []
        for run_id, df in run_id_2_df.items():
            if RESULT_SCENARIO_COLUMN not in df.columns:
                # Use a placeholder scenario if not present (single scenario runs)
                df_copy = df.copy()
                df_copy[RESULT_RUN_ID_COLUMN] = run_id
                df_copy[RESULT_SCENARIO_COLUMN] = "scenario_1"
                combined_rows.append(df_copy)
            else:
                # Each row keeps its own scenario value (multi-scenario runs)
                df_copy = df.copy()
                df_copy[RESULT_RUN_ID_COLUMN] = run_id
                combined_rows.append(df_copy)

        combined_df = pd.concat(combined_rows, ignore_index=True)

        # Rename bestObj to objValue for consistency
        if (
            exp_obj_value_col in combined_df.columns
            and RESULT_EXP_OBJ_VALUE_COLUMN not in combined_df.columns
        ):
            combined_df = combined_df.rename(
                columns={exp_obj_value_col: RESULT_EXP_OBJ_VALUE_COLUMN}
            )

        # Determine reference values and metadata
        reference_values, metadata_df, metadata_cols = determine_reference_values(
            combined_df,
            config.reference,
            intersection_names,
        )

        logging.info(f"Reference values loaded for {len(reference_values)} instances")
        if metadata_df is not None:
            logging.info(f"Instance metadata loaded: {list(metadata_df.columns)}")

        # Compute metrics for each run
        all_metrics: list[pd.DataFrame] = []

        # Group by runId and scenario
        grouped = combined_df.groupby([RESULT_RUN_ID_COLUMN, RESULT_SCENARIO_COLUMN])

        for (run_id, scenario), group_df in grouped:
            metrics_df = compute_metrics_for_run(
                group_df,
                run_id,  # type: ignore[arg-type]
                scenario,  # type: ignore[arg-type]
                reference_values,
                sense=config.reference.sense,
                instance_metadata=metadata_df,
                metadata_cols=metadata_cols,
            )
            all_metrics.append(metrics_df)

        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # Drop duplicate rows if any (should not happen, but safety check)
        combined_metrics = combined_metrics.drop_duplicates()

        # Ensure intersection filter
        combined_metrics = combined_metrics[
            combined_metrics[RESULT_INSTANCE_ID_COLUMN].isin(intersection_names)
        ]

        # Create output directory
        out_dir = Path(output_config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        basename = output_config.basename

        # 0. Config
        config_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_config.yaml"
        config_path = out_dir / config_filename
        config_dict = config.model_dump(mode="yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logging.info(f"Config saved to: {config_path}")

        # 1. Long format
        long_df = build_long_format(combined_metrics)
        long_path = out_dir / f"{basename}_long.csv"
        long_df.to_csv(long_path, index=False)
        logging.info(f"Long format saved to: {long_path}")

        # 2. Wide format - rpdf
        wide_rpdf = build_wide_rpdf(combined_metrics)
        wide_rpdf_path = out_dir / f"{basename}_rpdf_wide.csv"
        wide_rpdf.to_csv(wide_rpdf_path, index=False)
        logging.info(f"Wide RPDF saved to: {wide_rpdf_path}")

        # 3. Wide format - rpdv
        wide_rpdv = build_wide_rpdv(combined_metrics)
        wide_rpdv_path = out_dir / f"{basename}_rpdv_wide.csv"
        wide_rpdv.to_csv(wide_rpdv_path, index=False)
        logging.info(f"Wide RPDV saved to: {wide_rpdv_path}")

        # 4. Summary statistics
        summary_rpdf = build_summary_rpdf(wide_rpdf)
        summary_rpdf_path = out_dir / f"{basename}_summary_rpdf.csv"
        summary_rpdf.to_csv(summary_rpdf_path, index=False)
        logging.info(f"Summary RPDF saved to: {summary_rpdf_path}")

        summary_rpdv = build_summary_rpdv(wide_rpdv)
        summary_rpdv_path = out_dir / f"{basename}_summary_rpdv.csv"
        summary_rpdv.to_csv(summary_rpdv_path, index=False)
        logging.info(f"Summary RPDV saved to: {summary_rpdv_path}")

        logging.info(f"All outputs written to {out_dir}")
        return 0

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 2
    except Exception as e:
        logging.error(f"Error during comparison: {e}", exc_info=True)
        return 1
