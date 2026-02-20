"""Constants for column names used in experiment comparison.

This module centralizes all column names used throughout the exp_compare module
to ensure consistency and reduce the risk of typos.
"""

from typing import Final

# Input columns (from experiment CSV files)
# These are the column names expected in input summary CSVs
EXP_INSTANCE_ID_COLUMN: Final = "name"
EXP_OBJ_VALUE_COLUMN: Final = "bestObj"
EXP_SCENARIO_COLUMN: Final = "scenario"

# Reference columns (for mode=fixed_dataset)
# Default column names when reading reference values from external files
REF_INSTANCE_ID_COLUMN: Final = "name"
REF_OBJ_VALUE_COLUMN: Final = "UB"

# Result columns (output after computing metrics)
# These are the columns in the final output CSV files (long format, wide format)
RESULT_INSTANCE_ID_COLUMN: Final = "name"
RESULT_RUN_ID_COLUMN: Final = "runId"
RESULT_SCENARIO_COLUMN: Final = "scenario"
RESULT_ALGO_UID_COLUMN: Final = "algoUid"
RESULT_EXP_OBJ_VALUE_COLUMN: Final = "objValue"
RESULT_REF_OBJ_VALUE_COLUMN: Final = "refValue"
RESULT_RPDF_COLUMN: Final = "RPDf"
RESULT_RPDV_COLUMN: Final = "RPDv"
RESULT_RANK_COLUMN: Final = "rank"

ALL_RESULT_COLUMNS = [
    RESULT_INSTANCE_ID_COLUMN,
    RESULT_RUN_ID_COLUMN,
    RESULT_SCENARIO_COLUMN,
    RESULT_ALGO_UID_COLUMN,
    RESULT_EXP_OBJ_VALUE_COLUMN,
    RESULT_REF_OBJ_VALUE_COLUMN,
    RESULT_RPDF_COLUMN,
    RESULT_RPDV_COLUMN,
    RESULT_RANK_COLUMN,
]
