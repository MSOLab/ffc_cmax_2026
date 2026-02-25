# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

| Task | Command |
|------|---------|
| **Install dependencies** (using the recommended `uv` tool) | `uv sync` – installs the project and its development dependencies into the local `.venv`.
| **Run the full experiment** (default entry point) | `uv run main.py`
| **Run the full experiment (quiet mode)** | `uv run main.py --quiet`
| **Run a specific scenario script** (single‑instance or multi‑instance) | `uv run hfs_single_instance_runner.py`  *(or the multi‑instance equivalent)*
| **Run a single test** | `uv run pytest path/to/test_file.py::test_name`
| **Run all tests** | `uv run pytest`
| **Lint / type‑check** (project uses `ruff` and `mypy` if available) | `uv run ruff .` and `uv run mypy .`
| **Format code** | `uv run ruff --fix .` (or any formatter you prefer)

> The repository expects **Python 3.11**. All commands are intended to be executed from the project root.

## High‑Level Architecture

```
hybridflowshop/                # Core library
├── controller/               # Controllers that orchestrate the CP‑LNS algorithm
│   ├── reactive/            # Adaptive/reactive components (e.g., stopping criteria, param tuning)
│   └── hfs_cp_lns.py        # Main CP‑LNS controller implementation
├── schedule_lite.py          # Lightweight schedule representation (jobs × stages × machines)
├── painter/                 # Visualization utilities (Gantt charts, progress plots)
│   └── gantt.py
├── report/                  # Sub‑routine reporting and statistics collection
│   ├── hfs_subroutine_report.py
│   └── hfs_subroutine_report_statistics.py
├── utils.py                 # Miscellaneous helper functions used throughout the codebase
├── hfs_input_summary.py     # Summarizes problem instance input
├── hfs_summary.py           # Summarizes run results
├── solution_manager.py      # Manages solution objects, saving / loading
├── hfs_cp_lns.py            # High‑level CP‑LNS controller (exposed via __init__)
├── hfs_config.py            # Pydantic models for experiment configuration (main_metadata.yaml)
└── io_solution.py           # I/O utilities for reading/writing solutions
```

### Execution Flow (Typical Run)
1. **Configuration** – `main_metadata.yaml` is parsed into `MainMetadata` (Pydantic model in `hfs_config.py`). It defines problem instances, algorithm scenarios, output locations, and stopping criteria.
2. **Run‑mode determination** – `determine_run_mode_and_base_dir` decides between a fresh full run, a resume, or post‑process‑only based on timestamps or a resume directory.
3. **Data loading** – Common parameters (`pra_common_params.yaml`) and benchmark instances are loaded via `load_hfs_instance`.
4. **Scenario preparation** – For each scenario defined in the metadata, a sub‑routine flow (controller pipeline) and stopping‑criteria dictionary are read and validated against the controller class.
5. **Runner orchestration** – `HfsMultiScenarioRunner` (in `hfs_multi_scenario_runner.py`) creates per‑scenario runners (`HfsMultiInstanceRunner` and `HfsSingleInstanceRunner`). It distributes instances across worker processes if `instance_worker_cnt` > 1.
6. **Algorithm execution** – Each instance runs the CP‑LNS controller (`HybridFlowShopCpLnsController`). The controller repeatedly solves a CP model, applies large‑neighbourhood moves, and checks `LocalStoppingCriteria` (found in `controller/reactive/local_stopping_criteria.py`).
7. **Result handling** – Solutions, logs, Gantt charts, and summary CSVs are written to a timestamped output directory. The `output_filenames.py` module defines the naming patterns.
8. **Post‑processing** – If `analysis_timestamp` is set, the run re‑loads existing artifacts and produces additional reports without re‑executing the optimisation.

### Key Concepts
- **CP‑LNS controller** – Uses OR‑Tools CP‑SAT model (`cp_*` modules) and iteratively improves solutions via neighbourhood search.
- **Reactive components** – `controller/reactive` provides adaptive stopping criteria (`LocalStoppingCriteria`) and parameter tuning (`reactive_param_tuner.py`).
- **Schedule model** – `schedule_lite.py` provides a lightweight internal schedule representation (`HybridFlowshopLiteSchedule`) that tracks operations as `(start, end, job)` tuples per machine per stage, supporting dispatch, deepcopy, and removal operations.
- **Visualization** – `painter/gantt.py` generates PNG Gantt charts; the `report` package aggregates statistics (makespan, gaps, etc.).
- **Configuration via Pydantic** – Guarantees type‑safe experiment definitions; any validation error aborts early with a clear log message.

## Performance Comparison Metrics

The codebase includes utilities for computing **RPDf** and **RPDv** metrics to compare solution quality across methods:

| Metric | Formula | Use Case |
|--------|---------|----------|
| **RPDf** (Relative Percentage Difference from feasible) | `(obj - ref) / ((obj + ref) / 2)` | Normalized deviation, symmetric treatment |
| **RPDv** (Relative Percentage Deviation from optimal) | `(obj - ref) / ref` | Simple percentage deviation |

- **Baseline data**: Reference values (and instance metadata like job count, stage count) are loaded from a CSV file specified in `main_metadata.yaml`. The baseline CSV must include columns for instance identifier, objective value, job count, and stage count (configurable via `BaselineColumnMapping`).
- **Output files**:
  - `summary_method_end_time_and_obj_value_long.csv` – Long format with one row per method-instance
  - `summary_method_end_time_and_obj_value_wide.csv` – Wide format with one row per instance
  - `summary_method_rpdf_and_norm_time_long.csv` – Long format with RPDf, RPDv, and normalized time
  - `summary_method_rpdf_and_norm_time_wide.csv` – Wide format with RPDf, RPDv, and normalized time

Instance names are matched using `str(instance_id)` to align with baseline CSV entries.

## Column Naming Convention

The project uses standardized column names for CSV files to ensure consistency across the codebase.

### Instance Identifier Columns

| Context | Value | Location |
|---------|-------|----------|
| Internal (routix) | `"insName"` | `routix.constants.SubroutineReportStatisticsKeys.INSTANCE_NAME` |
| Input summary CSV | `"insName"` | `hybridflowshop.constants.INPUT_NAME_COLUMN` |
| Baseline/reference CSV | `"Instance"` (configurable) | `hfs_config.BaselineColumnMapping.instance` |

### Output Summary Columns (from routix)

| Column | Value | Description |
|--------|-------|-------------|
| `insName` | instance identifier | |
| `foundFeasibleSol` | boolean | Whether a feasible solution was found |
| `totalElapsedTime` | float | Total runtime in seconds |
| `firstObj` | float | Initial solution objective value |
| `firstBound` | float | Initial objective bound |
| `bestObj` | float | Best solution objective value |
| `bestBound` | float | Final objective bound |
| `improvementRatio` | float | Improvement from first to best |
| `methodCallCounts` | string | Serialized dict of method call counts |
| `reportCount` | int | Number of reports generated |

### Experiment Comparison Columns

| Column | Value | Description |
|--------|-------|-------------|
| `name` | instance identifier (for reference CSV) | Used in reference CSVs via `exp_compare.REF_INSTANCE_ID_COLUMN` |
| `objValue` | objective value | Standardized output column name |
| `refValue` | reference value | Baseline for RPD calculation |
| `RPDf` | float | Relative Percentage Difference from feasible |
| `RPDv` | float | Relative Percentage Deviation from optimal |
| `rank` | int | Ranking within instance |

---

## Notable Files and Modules
- `main.py` – Entry point, handles metadata loading and orchestrates runners.
- `hfs_config.py` – Pydantic schemas (`MainMetadata`, `IODataPath`, etc.).
- `schedule_lite.py` – Lightweight schedule representation used across all controllers.
- `controller/reactive/local_stopping_criteria.py` – Implements loop‑level stopping logic used by the LNS loop.
- `exp_compare/` – Experiment comparison utilities including RPDf/RPDv metrics computation.
- `tests/` – Unit tests for schedule_lite, stopping criteria, and reactive helpers.

## Development Tips
- **Run a single test** to iterate quickly, e.g.:
  ```bash
  uv run pytest tests/test_local_stopping_criteria.py::test_rho_no_improve_and_lb_gap
  ```
- **Explore the CP models** in files prefixed with `cp_` to understand the decision variables and constraints.
- **Add new scenarios** by extending `main_metadata.yaml` with a new entry in `dicts_of_i_o_data_path`; ensure a corresponding sub‑routine flow YAML and stopping‑criteria YAML exist.
- **Use the logging output** (`scenario_log_filename` in the metadata) to debug controller behaviour; logs are written to the timestamped output directory.

## Commit Message Guidelines

When contributing changes, follow this simple commit‑message format:

- **Title** – 50 characters or fewer, written in imperative mood (e.g., "Add quiet flag to logging").
- **Body** – Optional but recommended; use markdown bullet points (`-`) to describe what was changed, why, and any side effects.

**Example**:
```
Add quiet flag to logging

- Introduce `--quiet` CLI option to suppress console output.
- Refactor logging to a single `_setup_logging` function.
- Update documentation in `CLAUDE.md`.
```


---

*Generated for Claude Code to streamline future contributions.*