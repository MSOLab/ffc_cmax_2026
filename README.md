# Hybrid Flowshop Solver

This repository is a Python project for solving Hybrid Flowshop (HFS) problems using Constraint Programming (CP) based Large Neighborhood Search (LNS).

## Main Features

- Definition of Hybrid Flowshop problems and support for various instances
- Implementation of CP-based LNS and Pure CP algorithms
- Various stopping criteria (e.g., time limit) and experiment management
- Visualization tools for Gantt charts, time series, etc.
- Automatic saving and summarization of experiment results

## Folder Structure

```text
├── main.py                        # Main entry point
├── hfs_single_instance_runner.py  # Single instance runner script
├── hfs_multi_instance_runner.py   # Multi-instance runner script
├── main_metadata.yaml
├── LICENSE
├── pyproject.toml
├── uv.lock
├── README.md
├── hybridflowshop/                # Core modules
│   ├── hfs_cp_lns.py              # CP-LNS controller
│   ├── hfs_input_summary.py
│   ├── hfs_summary.py
│   ├── pure_cp_2023_naderi.py     # Pure CP algorithm
│   ├── solution_manager.py        # Solution management
│   ├── stopping_criteria.py       # Stopping criteria
│   ├── utils.py
│   └── painter/                   # Visualization tools
│       ├── gantt.py
│       ├── obj_value_bound_plotter.py
│       └── time_series_plotter.py
├── configs_*/                     # Various experiment settings (yaml)
├── Outputs/                       # Results (plots, csv, etc.)
│   └── pra_pure_cp/
│       ├── 0.csv, 1.csv, ...
│       ├── 0_result_gantt.png, ...
│       └── 0/, 1/, 2/, ...
├── resources/                     # Problem instance data
│   ├── pra_common_params.yaml
│   └── pra/
│       ├── 0.txt, 1.txt, ...
```

## Installation and Usage (with uv)

- Python 3.11 only
- [uv](https://github.com/astral-sh/uv) handles dependency installation automatically when running scripts.

### Example run

```bash
uv run main.py
```

## Main Dependencies

- Python 3.11 only
- `mbls`, `schore`, `ortools`, etc. (see pyproject.toml)

## Example Results

- Gantt charts (.png), solutions (.csv), and logs are saved in the Outputs/ folder.

## Notes

- Various experiment setting yaml files are in the configs_*/ folders.
- Problem instances (.txt) and common parameters are in the resources/ folder.

---
