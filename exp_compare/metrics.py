"""Metrics computation for experiment comparison: RPDf, RPDv, and Rank."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def compute_rpdf(
    obj: float | np.ndarray | pd.Series, ref: float | np.ndarray | pd.Series
) -> float | np.ndarray | pd.Series:
    """
    Compute Relative Percentage Difference (RPDf).

    Formula: RPDf = (obj - ref) / ((obj + ref) / 2)

    Special cases:
    - obj == 0 AND ref == 0 -> RPDf = 0
    - denominator == 0 (else) -> RPDf = NaN

    Args:
        obj: Objective value(s).
        ref: Reference value(s).

    Returns:
        RPDf value(s).
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
    """
    Compute Relative Percentage Deviation (RPDv).

    Formula: RPDv = (obj - ref) / ref

    Special cases:
    - ref == 0 -> RPDv = NaN

    Args:
        obj: Objective value(s).
        ref: Reference value(s).

    Returns:
        RPDv value(s).
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
    """
    Compute rank of objective values within a group.

    Args:
        obj_values: Series of objective values.
        sense: "min" for minimization (smaller is better).

    Returns:
        Series of ranks (dense rank, NaN for missing obj values).
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
) -> pd.DataFrame:
    """
    Compute RPDf, RPDv, and rank for a single run's data.

    Args:
        df: DataFrame with columns: name, bestObj (or objValue)
        run_id: Run identifier
        scenario: Scenario name
        reference_values: Series indexed by name with reference values
        sense: Optimization sense ("min")

    Returns:
        DataFrame with columns: name, runId, scenario, algoUid,
                               objValue, refValue, RPDf, RPDv, rank
    """
    # Determine which column contains objective values
    # Supports both "objValue" (normalized) and "bestObj" (original)
    obj_col = "objValue" if "objValue" in df.columns else "bestObj"

    # Merge with reference values
    # reference_values is a Series with name as index, so we need to convert to DataFrame
    ref_df = reference_values.reset_index()
    ref_df.columns = ["name", "refValue"]
    merged = df[["name", obj_col]].copy()
    merged = merged.merge(ref_df, on="name", how="left")

    # Create algoUid
    merged["runId"] = run_id
    merged["scenario"] = scenario
    merged["algoUid"] = f"{run_id}::{scenario}"

    # Rename to objValue for consistency
    merged = merged.rename(columns={obj_col: "objValue"})

    # Compute metrics
    merged["RPDf"] = compute_rpdf(merged["objValue"], merged["refValue"])
    merged["RPDv"] = compute_rpdv(merged["objValue"], merged["refValue"])

    # Compute rank within each name group using transform
    merged["rank"] = (
        merged.groupby("name", sort=False)["objValue"]
        .transform(lambda g: compute_rank(g, sense=sense))
        .reset_index(level=0, drop=True)
    )

    return merged[
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
