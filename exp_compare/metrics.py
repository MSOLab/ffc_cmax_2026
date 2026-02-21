"""Metrics computation for experiment comparison: RPDf, RPDv, and Rank."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from exp_compare.constants import (
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


def compute_rpdf(
    obj: float | np.ndarray | pd.Series, ref: float | np.ndarray | pd.Series
) -> float | np.ndarray | pd.Series:
    """Compute Relative Percentage Difference (RPDf).

    Formula: RPDf = (obj - ref) / ((obj + ref) / 2)

    Special cases:
    - obj == 0 AND ref == 0 -> RPDf = 0
    - denominator == 0 (else) -> RPDf = NaN

    Args:
        obj (float | np.ndarray | pd.Series): Objective value(s).
        ref (float | np.ndarray | pd.Series): Reference value(s).

    Returns:
        float | np.ndarray | pd.Series: RPDf value(s).
    """
    obj = np.asarray(obj, dtype=float)
    ref = np.asarray(ref, dtype=float)

    result = np.full_like(obj, np.nan, dtype=float)

    # Case 1: both zero -> rpdf = 0
    both_zero_mask = (obj == 0) & (ref == 0)
    result[both_zero_mask] = 0.0

    # Case 2: denominator != 0 and not both zero -> compute normally
    denom = (obj + ref) / 2
    valid_mask = ~both_zero_mask & (denom != 0)
    result[valid_mask] = (obj[valid_mask] - ref[valid_mask]) / denom[valid_mask]

    if result.size == 1:
        return float(result.item())
    return result


def compute_rpdv(
    obj: float | np.ndarray | pd.Series, ref: float | np.ndarray | pd.Series
) -> float | np.ndarray | pd.Series:
    """Compute Relative Percentage Deviation (RPDv).

    Formula: RPDv = (obj - ref) / ref

    Special cases:
    - ref == 0 -> RPDv = NaN

    Args:
        obj (float | np.ndarray | pd.Series): Objective value(s).
        ref (float | np.ndarray | pd.Series): Reference value(s).

    Returns:
        float | np.ndarray | pd.Series: RPDv value(s).
    """
    obj = np.asarray(obj, dtype=float)
    ref = np.asarray(ref, dtype=float)

    result = np.full_like(obj, np.nan, dtype=float)
    valid_mask = ref != 0

    result[valid_mask] = (obj[valid_mask] - ref[valid_mask]) / ref[valid_mask]

    if result.size == 1:
        return float(result.item())
    return result


def compute_rank(
    obj_values: pd.Series,
    sense: str = "min",
) -> pd.Series:
    """Compute rank of objective values within a group.

    Args:
        obj_values (pd.Series): Series of objective values.
        sense (str): "min" for minimization (smaller is better).

    Returns:
        pd.Series: Series of ranks (dense rank, NaN for missing obj values).
    """
    if sense != "min":
        logging.warning(f"Sense '{sense}' not supported, defaulting to 'min'")
        sense = "min"

    # NaN values get NaN rank
    # Use dense rank to handle ties (1, 2, 2, 3)
    ranks = obj_values.rank(method="dense", ascending=True)

    # Convert to float to allow NaN preservation
    return ranks.astype(float)


def compute_metrics_for_run(
    df: pd.DataFrame,
    run_id: str,
    scenario: str,
    reference_values: pd.Series,
    sense: str = "min",
    exp_obj_value_col: str = RESULT_EXP_OBJ_VALUE_COLUMN,
    instance_metadata: pd.DataFrame | None = None,
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute RPDf, RPDv, and rank for a single run's data.

    Args:
        df (pd.DataFrame): DataFrame with columns: name, bestObj (or objValue).
        run_id (str): Run identifier.
        scenario (str): Scenario name.
        reference_values (pd.Series): Series indexed by name with reference values.
        sense (str): Optimization sense ("min").
        exp_obj_value_col (str): Column name for objective values.
            Defaults to RESULT_EXP_OBJ_VALUE_COLUMN.
        instance_metadata (pd.DataFrame | None): DataFrame with instance metadata
            columns. If provided, these columns will be included in the output.
        metadata_cols (list[str] | None): List of metadata column names in config order.
            If provided, metadata columns will be ordered according to this list.

    Returns:
        pd.DataFrame: DataFrame with columns: name, runId, scenario, algoUid,
                      objValue, refValue, RPDf, RPDv, rank, and metadata columns.
    """
    # Ensure name column is string for consistent merge
    df = df.copy()
    df[RESULT_INSTANCE_ID_COLUMN] = df[RESULT_INSTANCE_ID_COLUMN].astype(str)

    # Merge with reference values
    # reference_values is a Series with name as index, so we need to convert to DataFrame
    ref_df = reference_values.reset_index()
    ref_df.columns = [RESULT_INSTANCE_ID_COLUMN, RESULT_REF_OBJ_VALUE_COLUMN]
    merged = df[[RESULT_INSTANCE_ID_COLUMN, exp_obj_value_col]].copy()
    merged = merged.merge(ref_df, on=RESULT_INSTANCE_ID_COLUMN, how="left")

    # Merge instance metadata if provided
    if instance_metadata is not None:
        merged = merged.merge(
            instance_metadata, on=RESULT_INSTANCE_ID_COLUMN, how="left"
        )

    # Create algoUid
    merged[RESULT_RUN_ID_COLUMN] = run_id
    merged[RESULT_SCENARIO_COLUMN] = scenario
    merged[RESULT_ALGO_UID_COLUMN] = f"{run_id}::{scenario}"

    # Rename to objValue for consistency
    merged = merged.rename(columns={exp_obj_value_col: RESULT_EXP_OBJ_VALUE_COLUMN})

    # Compute metrics (these create RPDf, RPDv, rank columns)
    merged[RESULT_RPDF_COLUMN] = compute_rpdf(
        merged[RESULT_EXP_OBJ_VALUE_COLUMN], merged[RESULT_REF_OBJ_VALUE_COLUMN]
    )
    merged[RESULT_RPDV_COLUMN] = compute_rpdv(
        merged[RESULT_EXP_OBJ_VALUE_COLUMN], merged[RESULT_REF_OBJ_VALUE_COLUMN]
    )
    merged[RESULT_RANK_COLUMN] = (
        merged.groupby(RESULT_INSTANCE_ID_COLUMN, sort=False)[
            RESULT_EXP_OBJ_VALUE_COLUMN
        ]
        .transform(lambda g: compute_rank(g, sense=sense))
        .reset_index(level=0, drop=True)
    )

    # Add metadata columns if present
    # Get columns that are not in the standard result columns
    standard_cols_set = {
        RESULT_INSTANCE_ID_COLUMN,
        RESULT_RUN_ID_COLUMN,
        RESULT_SCENARIO_COLUMN,
        RESULT_ALGO_UID_COLUMN,
        RESULT_EXP_OBJ_VALUE_COLUMN,
        RESULT_REF_OBJ_VALUE_COLUMN,
        RESULT_RPDF_COLUMN,
        RESULT_RPDV_COLUMN,
        RESULT_RANK_COLUMN,
    }
    # Use config order if provided, otherwise get from merged columns
    if metadata_cols is not None:
        # Filter to only columns that exist in merged
        metadata_cols = [col for col in metadata_cols if col in merged.columns]
    else:
        metadata_cols = [
            col for col in merged.columns
            if col not in standard_cols_set and col != RESULT_INSTANCE_ID_COLUMN
        ]
        # Sort metadata columns for consistency
        metadata_cols.sort()

    # Build final column order: name first, then metadata, then standard columns
    final_cols = [RESULT_INSTANCE_ID_COLUMN] + metadata_cols + [
        RESULT_RUN_ID_COLUMN,
        RESULT_SCENARIO_COLUMN,
        RESULT_ALGO_UID_COLUMN,
        RESULT_EXP_OBJ_VALUE_COLUMN,
        RESULT_REF_OBJ_VALUE_COLUMN,
        RESULT_RPDF_COLUMN,
        RESULT_RPDV_COLUMN,
        RESULT_RANK_COLUMN,
    ]

    # Filter to columns that actually exist in the dataframe
    final_cols = [col for col in final_cols if col in merged.columns]

    return merged[final_cols]
