"""Constants for column names used in Hybrid Flowshop.

This module centralizes all column names used throughout the hybridflowshop
package to ensure consistency and reduce the risk of typos.
"""

from typing import Final

# Input summary columns (for hfs_input_summary.py)
INPUT_NAME_COLUMN: Final = "name"
INPUT_JOBCOUNT_COLUMN: Final = "jobCount"
INPUT_STAGECOUNT_COLUMN: Final = "stageCount"
INPUT_MACHINESPERSTAGE_COLUMN: Final = "machinesPerStage"
INPUT_TIMELIMIT_COLUMN: Final = "timelimit"

INPUT_SUMMARY_HEADER: Final = (
    f"{INPUT_NAME_COLUMN},{INPUT_JOBCOUNT_COLUMN},"
    f"{INPUT_STAGECOUNT_COLUMN},{INPUT_MACHINESPERSTAGE_COLUMN},{INPUT_TIMELIMIT_COLUMN}"
)
