"""Main comparison logic and CSV output generation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

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
                      objValue, refValue, RPDf, RPDv, rank
    """
    return combined_df[
        [
            "name",
            "runId",
            "scenario",
            "algoUid",
            "objValue",
            "refValue",
            "RPDf",
            "RPDv",
            "rank",
        ]
    ]


def build_wide_rpdf(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Build wide-format DataFrame with relative percentage difference values.

    Args:
        combined_df (pd.DataFrame): Dataframe with all combined data and computed metrics.

    Returns:
        pd.DataFrame: Wide-format DataFrame with columns: name, algoUid, and RPDf values.
    """
    wide = combined_df.pivot_table(
        index="name",
        columns="algoUid",
        values="RPDf",
    ).reset_index()

    return wide


def build_wide_rpdv(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Build wide-format DataFrame with relative percentage deviation values.

    Args:
        combined_df (pd.DataFrame): Dataframe with all combined data and computed metrics.

    Returns:
        pd.DataFrame: Wide-format DataFrame with columns: name, algoUid, and RPDv values.
    """
    wide = combined_df.pivot_table(
        index="name",
        columns="algoUid",
        values="RPDv",
    ).reset_index()

    return wide


def build_summary_rpdf(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary statistics DataFrame for RPDf.

    Groupby: algoUid (columns excluding 'name')
    Stats: count, min, max, mean, median, std
    """
    algo_cols = [col for col in wide_df.columns if col != "name"]

    summary_rows = []
    for col in algo_cols:
        col_data = wide_df[col]
        summary_rows.append(
            {
                "algoUid": col,
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
    """
    Build summary statistics DataFrame for RPDv.

    Same structure as build_summary_rpdf but for RPDv values.
    """
    return build_summary_rpdf(wide_df)


def determine_reference_values(
    combined_df: pd.DataFrame,
    reference_config: ReferenceConfig,
    intersection_names: set[str],
) -> pd.Series:
    """
    Determine reference values based on mode.

    Args:
        combined_df: Combined DataFrame from all runs
        reference_config: Reference configuration
        intersection_names: Set of instance names in intersection

    Returns:
        Series indexed by name with reference values
    """
    mode = reference_config.mode
    sense = reference_config.sense

    # Determine which column contains objective values
    # Supports both "objValue" (normalized) and "bestObj" (original)
    obj_col = "objValue" if "objValue" in combined_df.columns else "bestObj"

    if mode == "best_among_compared":
        # For each instance, take minimum objective value across all compared algorithms
        best_obj_df = combined_df[combined_df["name"].isin(intersection_names)][
            ["name", obj_col]
        ]
        if sense == "max":
            ref_values = best_obj_df.groupby("name")[obj_col].max()
        else:  # Default to "min"
            ref_values = best_obj_df.groupby("name")[obj_col].min()

    elif mode == "fixed_dataset":
        ref_path = Path(reference_config.ref_path)  # type: ignore[misc]
        instance_key_col = reference_config.instance_key_column_in_ref
        value_col = reference_config.reference_value_column

        if not ref_path.exists():
            logging.warning(f"Reference file not found: {ref_path}")
            return pd.Series(dtype=float)

        ref_df = load_reference_csv(ref_path, instance_key_col, value_col)  # type: ignore[misc]

        # Create series indexed by name
        ref_values = ref_df.set_index(instance_key_col)[value_col]

        # Filter to intersection names, keeping NaN for missing
        ref_values = ref_values.reindex(list(intersection_names))

    else:
        logging.warning(f"Unknown reference mode: {mode}, using best_among_compared")
        best_obj_df = combined_df[combined_df["name"].isin(intersection_names)][
            ["name", obj_col]
        ]
        ref_values = best_obj_df.groupby("name")[obj_col].min()

    return ref_values


def run_comparison(config: CompareConfig) -> int:
    """
    Run the comparison analysis.

    Args:
        config: CompareConfig with all settings

    Returns:
        Exit code: 0 for success, 2 for config/file errors, 1 for other errors
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
            if "scenario" not in df.columns:
                # Use first scenario if not present (single scenario runs)
                scenario = "scenario_1"
            else:
                scenarios = df["scenario"].unique()
                scenario = scenarios[0] if len(scenarios) == 1 else "multi_scenario"

            df_copy = df.copy()
            df_copy["runId"] = run_id
            df_copy["scenario"] = scenario
            combined_rows.append(df_copy)

        combined_df = pd.concat(combined_rows, ignore_index=True)

        # Rename bestObj to objValue for consistency
        if "bestObj" in combined_df.columns and "objValue" not in combined_df.columns:
            combined_df = combined_df.rename(columns={"bestObj": "objValue"})

        # Determine reference values
        reference_values = determine_reference_values(
            combined_df,
            config.reference,
            intersection_names,
        )

        logging.info(f"Reference values loaded for {len(reference_values)} instances")

        # Compute metrics for each run
        all_metrics: list[pd.DataFrame] = []

        # Group by runId and scenario
        grouped = combined_df.groupby(["runId", "scenario"])

        for (run_id, scenario), group_df in grouped:
            metrics_df = compute_metrics_for_run(
                group_df,
                run_id,  # type: ignore[arg-type]
                scenario,  # type: ignore[arg-type]
                reference_values,
                sense=config.reference.sense,
            )
            all_metrics.append(metrics_df)

        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # Ensure intersection filter
        combined_metrics = combined_metrics[
            combined_metrics["name"].isin(intersection_names)
        ]

        # Create output directory
        out_dir = Path(output_config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        basename = output_config.basename

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
