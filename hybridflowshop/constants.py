"""Constants for column names used in Hybrid Flowshop.

This module centralizes all column names used throughout the hybridflowshop
package to ensure consistency and reduce the risk of typos.

## Column Naming Convention

### Instance ID Column
- `INPUT_NAME_COLUMN` uses `routix.constants.SubroutineReportStatisticsKeys.INSTANCE_NAME`
- Value: "insName"
- Used in: `hfs_input_summary.py`, `hfs_single_instance_runner.py` summary rows

### Job/Stage/Machine Columns
- `INPUT_JOBCOUNT_COLUMN`: "jobCount"
- `INPUT_STAGECOUNT_COLUMN`: "stageCount"
- `INPUT_MACHINESPERSTAGE_COLUMN`: "machinesPerStage"

### Time Limit Column
- `INPUT_TIMELIMIT_COLUMN`: "timelimit"

## Note on "insName" vs "name" vs "Instance"
The project uses multiple instance identifier column names across different contexts:

| Context | Value | Location |
|---------|-------|----------|
| Internal (routix) | "insName" | routix/constants.py |
| Input summary | "insName" | hybridflowshop/constants.py |
| Baseline CSV | "Instance" (configurable) | hfs_config.py |

This is intentional:
- "insName" (lowercase) is used for internal data structures and output CSVs
- "Instance" (capitalized) is a common convention in baseline/reference files
- All are configurable where appropriate to support different file formats
"""

from typing import Final

from routix.constants import SubroutineReportStatisticsKeys

# Input summary columns (for hfs_input_summary.py)
INPUT_NAME_COLUMN: Final = SubroutineReportStatisticsKeys.INSTANCE_NAME
INPUT_JOBCOUNT_COLUMN: Final = "jobCount"
INPUT_STAGECOUNT_COLUMN: Final = "stageCount"
INPUT_MACHINESPERSTAGE_COLUMN: Final = "machinesPerStage"
INPUT_TIMELIMIT_COLUMN: Final = "timelimit"

INPUT_SUMMARY_HEADER: Final = (
    f"{INPUT_NAME_COLUMN},{INPUT_JOBCOUNT_COLUMN},"
    f"{INPUT_STAGECOUNT_COLUMN},{INPUT_MACHINESPERSTAGE_COLUMN},{INPUT_TIMELIMIT_COLUMN}"
)
