# Experiment Comparison Tool (`exp_compare`)

Reads multiple experiment (run directory) results and generates algorithm performance comparison CSVs for the **intersection of common instance names**.

## Overview

The primary output is a wide-format CSV that can be used directly in Excel to create pivot charts. **RPDf is the primary metric**, with **RPDv as a secondary metric**.

## Installation

`exp_compare` is included in the project, so no separate installation is required.

```bash
uv sync  # Install dependencies (if needed)
```

## Usage

### CLI Mode

```bash
# Use default config (exp_compare/config.yaml)
uv run python -m exp_compare

# Use custom config
uv run python -m exp_compare --config path/to/config.yaml

# Suppress console output
uv run python -m exp_compare --quiet
```

### Options

- `--config`, `-c`: Path to YAML configuration file
  - Defaults to `exp_compare/config.yaml` if not specified
- `--quiet`, `-q`: Suppress console output (logs are still written to `exp_compare.log`)

## Configuration

### YAML Schema

```yaml
runs:
  - path: "Outputs_scenarios/20260218T012813_090868/"
    summary_csv: "all_scenarios_summary.csv"
  - path: "Outputs_scenarios/20260217T153612_195650/"
    summary_csv: "all_scenarios_summary.csv"

reference:
  mode: "best_among_compared"   # or "fixed_dataset"
  sense: "min"                  # currently only min is expected

  # only when mode=fixed_dataset
  ref_path: null                # e.g., "refs/best_known.csv"
  ref_format: "csv"             # only csv required
  instance_key_column_in_ref: "name"
  reference_value_column: null  # e.g., "UB"

output:
  out_dir: "Outputs_analysis/compare_YYYYMMDD/"
  basename: "rpd_compare"
```

### Configuration Parameters

#### `runs` (Required)
- List of run directories to compare
- `path`: Path to run directory
- `summary_csv`: Name of the summary CSV file within each run (default: `"all_scenarios_summary.csv"`)

#### `reference` (Required)
- Method for determining reference values

| Field | Description | Default |
|-------|-------------|---------|
| `mode` | `best_among_compared` (minimum among compared algorithms) or `fixed_dataset` (external CSV) | `"best_among_compared"` |
| `sense` | Optimization direction (`min` or `max`) | `"min"` |
| `ref_path` | Reference CSV file path (required when mode=fixed_dataset) | `null` |
| `ref_format` | Reference file format | `"csv"` |
| `instance_key_column_in_ref` | Column name for instance keys in reference CSV | `"name"` |
| `reference_value_column` | Column name for reference values | `null` |

#### `output` (Required)
- Output settings

| Field | Description | Default |
|-------|-------------|---------|
| `out_dir` | Output directory path | `"Outputs_analysis/compare_YYYYMMDD/"` |
| `basename` | Output file name prefix | `"rpd_compare"` |

## Input Format

Summary CSV files within each run directory must include the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `name` or `instanceName` | Instance ID | string |
| `scenario` | Algorithm ID | string |
| `bestObj` or `objValue` | Objective value | float |

> **Note:** `instanceName` column is automatically renamed to `name`.

## Output Files

All output files are generated under `output.out_dir`. Column names use camelCase.

### 1. `{basename}_long.csv`

Row per (name, algoUid):

| Column | Description |
|--------|-------------|
| `name` | Instance ID |
| `runId` | Run identifier (directory name) |
| `scenario` | Scenario name |
| `algoUid` | `{runId}::{scenario}` |
| `objValue` | Objective value |
| `refValue` | Reference value |
| `RPDf` | Relative Percentage Difference |
| `RPDv` | Relative Percentage Deviation |
| `rank` | Within-instance rank (dense rank) |

### 2. `{basename}_rpdf_wide.csv`

Row per `name`, columns: `name`, `[algoUid_1, algoUid_2, ...]`
Values: `RPDf` as floats (no percentage sign)

### 3. `{basename}_rpdv_wide.csv`

Row per `name`, columns: `name`, `[algoUid_1, algoUid_2, ...]`
Values: `RPDv` as floats

### 4. `{basename}_summary_rpdf.csv`

Grouped by: `algoUid`

| Column | Description |
|--------|-------------|
| `algoUid` | Algorithm identifier |
| `count` | Number of valid values (excluding NaN) |
| `min` | Minimum RPDf |
| `max` | Maximum RPDf |
| `mean` | Mean RPDf |
| `median` | Median RPDf |
| `std` | Standard deviation |

### 5. `{basename}_summary_rpdv.csv`

Same structure, values are `RPDv`

## Metrics Definitions

### RPDf (Relative Percentage Difference)

```
RPDf = (obj - ref) / ((obj + ref) / 2)
```

Special cases:
- `obj == 0 AND ref == 0` → `RPDf = 0`
- denominator `((obj+ref)/2) == 0` → `RPDf = NaN`

### RPDv (Relative Percentage Deviation)

```
RPDv = (obj - ref) / ref
```

Special cases:
- `ref == 0` → `RPDv = NaN`

### Rank

- Computed based on `objValue` within each instance (name)
- `sense="min"`: Smaller values are better (default)
- `sense="max"`: Larger values are better
- Missing (NaN) objective values get NaN rank
- Ties handled with dense rank (1, 2, 2, 3)

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Successful completion (including when intersection is empty) |
| 1 | Other exceptions |
| 2 | Config/input file missing or parsing failed |

## Examples

### Example 1: Basic Usage (best_among_compared)

```yaml
runs:
  - path: "Outputs_scenarios/20260218T012813_090868/"
    summary_csv: "all_scenarios_summary.csv"
  - path: "Outputs_scenarios/20260217T153612_195650/"
    summary_csv: "all_scenarios_summary.csv"

reference:
  mode: "best_among_compared"
  sense: "min"

output:
  out_dir: "Outputs_analysis/compare_20260219/"
  basename: "rpd_compare"
```

### Example 2: Fixed Reference (fan2023.csv)

```yaml
runs:
  - path: "Outputs_scenarios/20260218T012813_090868/"
    summary_csv: "all_scenarios_summary.csv"
  - path: "Outputs_scenarios/20260217T153612_195650/"
    summary_csv: "all_scenarios_summary.csv"

reference:
  mode: "fixed_dataset"
  sense: "min"
  ref_path: "resources/ff2020big_ref/fan2023.csv"
  instance_key_column_in_ref: "Instance"
  reference_value_column: "UB"

output:
  out_dir: "Outputs_analysis/compare_20260219/"
  basename: "rpd_compare"
```

Example `fan2023.csv` structure:

```csv
Instance,n,s,rep,UB,LB
1,40,5,1,976,976
2,40,10,1,1292,1292
...
```

### Example 3: Maximization Problem

```yaml
reference:
  mode: "best_among_compared"
  sense: "max"  # Larger values are better

output:
  out_dir: "Outputs_analysis/compare_20260219/"
  basename: "rpd_compare_max"
```

## Notes

- If the intersection is empty, a warning log is printed once and the program exits with code 0 (no files generated)
- Paths can be relative or absolute
- Numbers are stored without a `%` sign
- NaN values are stored as empty cells in CSV
- The `bestObj` column is internally renamed to `objValue`
