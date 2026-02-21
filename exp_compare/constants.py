"""Constants for column names used in experiment comparison.

This module centralizes all column names used throughout the exp_compare module
to ensure consistency and reduce the risk of typos.

## Column Naming Convention

### Instance ID Column
- Uses `routix.constants.SubroutineReportStatisticsKeys.INSTANCE_NAME` (value: "insName")
- This is the standard instance identifier column name across the project
- Applied in: EXP_INSTANCE_ID_COLUMN, RESULT_INSTANCE_ID_COLUMN

### Objective Value Column
- Uses `routix.constants.SubroutineReportStatisticsKeys.BEST_OBJ` (value: "bestObj")
- Represents the best objective value found by the algorithm

### Reference Columns (for mode=fixed_dataset)
- REF_INSTANCE_ID_COLUMN uses "name" (not routix constant)
- This is because reference CSV files often use "name" as the instance key column
- REF_OBJ_VALUE_COLUMN uses "UB" (upper bound) as the default reference value column

### Output Columns
- RESULT_INSTANCE_ID_COLUMN: Same as EXP_INSTANCE_ID_COLUMN ("insName")
- RESULT_EXP_OBJ_VALUE_COLUMN: "objValue" (standardized output column name)
- RESULT_REF_OBJ_VALUE_COLUMN: "refValue" (reference value for RPD calculation)
- RESULT_RPDF_COLUMN: "RPDf" (Relative Percentage Difference from reference)
- RESULT_RPDV_COLUMN: "RPDv" (Relative Percentage Deviation from reference)
"""

from typing import Final

from routix.constants import SubroutineReportStatisticsKeys

# Input columns (from experiment CSV files)
# These are the column names expected in input summary CSVs
EXP_INSTANCE_ID_COLUMN: Final = SubroutineReportStatisticsKeys.INSTANCE_NAME
EXP_OBJ_VALUE_COLUMN: Final = SubroutineReportStatisticsKeys.BEST_OBJ
EXP_SCENARIO_COLUMN: Final = "scenario"

# Reference columns (for mode=fixed_dataset)
# Default column names when reading reference values from external files
REF_INSTANCE_ID_COLUMN: Final = "name"
REF_OBJ_VALUE_COLUMN: Final = "UB"

# Result columns (output after computing metrics)
# These are the columns in the final output CSV files (long format, wide format)
RESULT_INSTANCE_ID_COLUMN: Final = SubroutineReportStatisticsKeys.INSTANCE_NAME
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
