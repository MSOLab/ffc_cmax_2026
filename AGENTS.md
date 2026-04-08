# AGENTS.md

## Project Structure & Module Organization

`hybridflowshop/` is the main solver package. Root scripts (`main.py`, `hfs_*_runner.py`, `batch_analyzer.py`) launch experiments. Keep benchmark data in `resources/`, run settings in `configs_*/` and `main_metadata*.yaml`, and comparison utilities in `exp_compare/`. Tests live in `tests/`, with `tests/controller/` and `tests/dispatcher/` mirroring the package layout. Treat `Outputs*/` as generated artifacts, not source.

```plaintext
hybridflowshop/
├── controller/
│   ├── controller_core.py        # Base class (CpSubroutineController) with CP model setup, solve helpers
│   ├── hfs_cp_lns.py             # Main controller (HybridFlowShopCpLnsController) – subroutines as methods
│   ├── pw_cp.py                  # Prefix-window CP constructor (PwCpConstructor)
│   ├── neh_cp.py                 # NEH-CP constructor
│   └── reactive/                 # Adaptive components
│       ├── local_stopping_criteria.py
│       ├── reactive_looper.py
│       ├── reactive_loop_report.py
│       └── reactive_param_tuner.py
├── cpsat_model_2/                # CP-SAT model building blocks
│   ├── cumulative.py             # BaseModelBuilder, CumulativeVars, OperationVars
│   ├── pw_cp.py                  # PwCpModelBuilder
│   └── params.py
├── dispatcher/                   # Constructive heuristics (job, machine, stage, BN2D, mixed)
├── schedule_lite.py              # HybridFlowshopLiteSchedule – (start,end,job) per machine/stage
├── solution_manager.py
├── painter/gantt.py              # Gantt chart generation (PNG)
└── report/                       # Subroutine reporting & statistics

identical_parallel_machine/       # Sub-problem CP model for single-stage IPM
exp_compare/                      # Post-run comparison and RPD metrics CLI
configs_20s/ configs_100s/ configs_600s/ configs_dispatch/   # Subroutine flow YAMLs per time limit
```

### Key Third-Party Packages

| Package | Role |
|---------|------|
| `routix` | Orchestration: `DynamicDataObject`, `ElapsedTimer`, `StoppingCriteria`, `SubroutineFlowValidator`, `RunMode` |
| `mbls` | CP-SAT helpers: `CpSubroutineController`, `CustomCpModel`, `CpsatSolverReport` |
| `schore` | Problem instance representation: `HybridFlowshopParameters` |
| `ortools` | OR-Tools CP-SAT solver (9.14) |

## Build, Test, and Development Commands

Always invoke Python tooling through `uv`: use `uv run python ...` and `uv run pytest ...`, not `python`, `python3`, or `python -m pytest`.

- `uv sync --dev` installs the locked Python 3.11 runtime and dev tools.
- `uv run python main.py` runs the main multi-scenario workflow from `main_metadata.yaml`.
- `uv run python main.py --quiet` suppresses console output (log still written to file).
- `uv run pytest` runs the full test suite.
- `uv run pytest tests/controller/test_pw_cp.py -q` runs a focused regression loop.
- `uv run ruff check .` performs linting.
- `uv run ruff format .` formats code.
- `uv run mypy hybridflowshop tests` checks type consistency.
- `uv run python -m exp_compare` compares completed experiment runs (uses `exp_compare/config.yaml` by default).

## Execution Flow

1. `main.py` loads `main_metadata.yaml` into `MainMetadata` (`hfs_config.py`) and calls `determine_run_mode_and_base_dir` → picks `FULL_RUN`, `RESUME`, or `POST_PROCESS_ONLY`.
2. Common params and benchmark instances are loaded; each scenario's `subroutine_flow_*.yaml` and `stopping_criteria.yaml` are validated against `HybridFlowShopCpLnsController` via `SubroutineFlowValidator`.
3. `HfsMultiScenarioRunner` distributes instances across worker processes (`instance_worker_cnt`) using `HfsMultiInstanceRunner` / `HfsSingleInstanceRunner`.
4. Each instance runs the CP-LNS controller: CP model → neighbourhood moves → `LocalStoppingCriteria` check → repeat.
5. Outputs (solutions, logs, Gantt PNGs, summary CSVs) land in a timestamped subdirectory under `Outputs_scenarios/`.

### Run Modes (set in `main_metadata.yaml`)

| Field | Effect |
|-------|--------|
| Neither set | `FULL_RUN` – fresh optimisation |
| `analysis_dir_path` or `analysis_timestamp` | `POST_PROCESS_ONLY` – reload artifacts, regenerate reports |
| `resume_dir_path` | `RESUME` – continue from a previous run's cached subroutine flow |

Each scenario's `subroutine_flow_*.yaml` declares which controller methods to call in order. `SubroutineFlowValidator` checks every method name against `HybridFlowShopCpLnsController` before the run starts.

## exp_compare CLI

```bash
uv run python -m exp_compare                            # default: exp_compare/config.yaml
uv run python -m exp_compare --config path/to/cfg.yaml --quiet
```

Config specifies `runs` (list of output dirs + summary CSV name), `reference` (`best_among_compared` or `fixed_dataset`), and `output` (dir with optional `$TIMESTAMP` placeholder). Primary metric is **RPDf**; **RPDv** is secondary.

| Metric | Formula |
|--------|---------|
| **RPDf** | `(obj − ref) / ((obj + ref) / 2)` |
| **RPDv** | `(obj − ref) / ref` |

## Coding Style & Naming Conventions

Use 4-space indentation, keep functions focused, and add type hints on public APIs when practical. Follow the existing Python naming style: `snake_case` for modules, functions, and variables; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Match the repository's descriptive test names and scenario-style YAML names such as `subroutine_flow_20260331-01.yaml`. Run Ruff before opening a PR, and avoid unrelated formatting churn.

## Testing Guidelines

Pytest is configured in `pyproject.toml` to discover `tests/test_*.py` files and `test_*` functions. Add regression tests next to the area you changed: controller logic in `tests/controller/`, dispatch behavior in `tests/dispatcher/`, and cross-cutting behavior in top-level `tests/`. No coverage gate is configured, so compensate with targeted tests for scheduling precedence, IO, and summary/output behavior.

## Commit & Pull Request Guidelines

Recent history mixes short imperative subjects with scoped Conventional Commit prefixes such as `fix(pw-cp): ...` and `refactor(pw-cp): ...`. Prefer that style for code changes, and reserve timestamped "run setting" commits for experiment/config snapshots. PRs should state the behavior change, list the verification commands you ran, link related issues, and include a sample output or screenshot when Gantt or progress-plot rendering changes. Do not commit `Outputs*/`, caches, or `.venv/`.
