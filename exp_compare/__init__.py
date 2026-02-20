"""Experiment comparison module for analyzing multiple HFS runs."""

from exp_compare.constants import (
    ALL_RESULT_COLUMNS,
    EXP_INSTANCE_ID_COLUMN,
    EXP_OBJ_VALUE_COLUMN,
    EXP_SCENARIO_COLUMN,
    REF_INSTANCE_ID_COLUMN,
    REF_OBJ_VALUE_COLUMN,
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
from exp_compare.io import compute_intersection, load_run_summaries
from exp_compare.main import run_comparison
from exp_compare.metrics import (
    compute_metrics_for_run,
    compute_rank,
    compute_rpdf,
    compute_rpdv,
)

__all__ = [
    "load_run_summaries",
    "compute_intersection",
    "compute_rpdf",
    "compute_rpdv",
    "compute_rank",
    "compute_metrics_for_run",
    "run_comparison",
    # Column name constants
    "ALL_RESULT_COLUMNS",
    "EXP_INSTANCE_ID_COLUMN",
    "EXP_OBJ_VALUE_COLUMN",
    "EXP_SCENARIO_COLUMN",
    "REF_INSTANCE_ID_COLUMN",
    "REF_OBJ_VALUE_COLUMN",
    "RESULT_ALGO_UID_COLUMN",
    "RESULT_EXP_OBJ_VALUE_COLUMN",
    "RESULT_INSTANCE_ID_COLUMN",
    "RESULT_RANK_COLUMN",
    "RESULT_REF_OBJ_VALUE_COLUMN",
    "RESULT_RPDF_COLUMN",
    "RESULT_RPDV_COLUMN",
    "RESULT_RUN_ID_COLUMN",
    "RESULT_SCENARIO_COLUMN",
]
