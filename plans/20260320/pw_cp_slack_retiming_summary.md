# PW-CP Slack Retiming Summary

## Purpose

This document summarizes:

- the design context provided before implementation
- the intended behavioral contract for PW-CP slack-aware retiming
- the concrete changes currently staged in git
- the follow-up fixes made during review

The goal of this work is to improve reconstruction fidelity for non-final PW-CP batches without changing the underlying cumulative CP model.

## Original Context

### 1. Slack-aware machine reassignment after `create_schedule`

The requested flow for non-final PW-CP batches was:

1. solve CP
2. `create_schedule(...)`
3. extract solved slack intervals from CP
4. reassign machine ownership so the realized schedule reflects the solved slack on the intended machine
5. `make_semi_active(...)`
6. continue with acceptance and debug export

The key modeling choice was:

- keep the current cumulative CP model unchanged
- fix the mismatch only during schedule reconstruction
- scope successor reassignment to stage-local suffixes only

The intended successor behavior was:

- each positive solved slack interval belongs to a specific stage machine slot
- find the realized right-boundary operation at `slack_end`
- swap the stage-local machine suffixes so that the boundary operation and its following stage operations move together onto the intended slack machine
- retime only once after all stage adjustments

### 2. Reverse predecessor reassignment

The requested predecessor reassignment behavior was:

- process stages in reverse order
- keep `right_time_fixed` operations as anchors
- remove only movable predecessor operations on the affected stage machines
- re-dispatch them backward using machine latest completion targets derived from solved slack

This introduced a new schedule API:

- `dispatch_stage_reversed_by_jobs(stage_id, job_id_seq, job_2_duration, mc_2_lct, *, job_2_deadline=None)`

The reverse dispatch contract was:

- each job should be placed at the latest feasible position
- upper bound should respect machine slack start, optional job deadline, and next-stage start
- if no feasible slot exists even after checking down to time zero, raise `ValueError`

### 3. Candidate retiming refactor

The requested non-final candidate postprocessing contract was:

- `report.obj_value > 0`: do successor reassignment, predecessor reassignment, then `make_semi_active()`
- `report.obj_value == 0`: skip reassignment but still do `make_semi_active()`
- `report.obj_value < 0`: raise an explicit error

Debug export behavior was intended to:

- save retiming snapshots by stage
- render only the snapshots that actually exist
- preserve the existing suffix naming

### 4. Additional context that preceded the current staged state

The implementation context also included these updates:

- `verify/pw_cp_positive_boundary_deviation_demo.py` had to be updated to the current slack hint API
- right-boundary slack hints were redefined around actual machine slack semantics and then unified via a common slack length
- Gantt plotting was refactored so figures are created lazily and always closed after export/display
- missing right boundaries should be represented explicitly with `None` instead of fallback makespan values

## Staged Implementation Summary

### PW-CP controller changes

`hybridflowshop/controller/pw_cp.py` now includes:

- `StageBoundaryProfile = dict[str, list[int | None]]`
- initial batch Gantt export when `debug_export=True`
- non-final candidate retiming split into explicit snapshots:
  - `before_retiming`
  - `after_successor_reassign`
  - `after_predecessor_reassign`
  - `after_semi_active`
- explicit objective handling for non-final feasible solutions:
  - raise if `obj_value is None`
  - raise if `obj_value < 0`
  - run reassignment only when `obj_value > 0`
  - always call `make_semi_active()` when `obj_value >= 0`
- solved slack extraction against the current `slack_start_*` plus shared `slack_length` variables
- `_apply_successor_machine_reassignment(...)`
- `_apply_predecessor_machine_reassignment(...)`
- `_draw_schedule_gantt(...)`
- `_draw_candidate_retiming_gantts(...)`
- `_build_plot_inputs_with_slack_intervals(...)`

Behaviorally, this means:

- non-final candidates are reconstructed using solved slack intervals before final semi-active retiming
- predecessor reassignment is anchored by already-stabilized later stages
- debug visualizations can show CP slack intervals on separate plotting lanes

### Right-slack hint and extraction changes

The staged code changes the right-slack auxiliary semantics from per-machine slack extras to a shared slack length:

- old hint variables like `slack_extra_*`, `stage_slack_min_*`, `global_slack_min` are removed
- new hint variables are:
  - `slack_start_{stage}_{machine_idx}`
  - `slack_length`

The implemented hint calculation now:

- computes actual slack lengths per machine
- takes the minimum positive/zero slack length as the common shared length
- recomputes each machine start as `slack_end - common_slack_length`
- skips machines whose right boundary is `None`

The staged CP model changes in `hybridflowshop/cpsat_model_2/cumulative.py` mirror this:

- one shared `slack_length` variable
- one interval per valid machine boundary using that shared length
- objective maximizes the shared `slack_length`

### Schedule utility changes

`hybridflowshop/schedule_lite.py` now includes:

- reverse dispatch support:
  - `_get_latest_feasible_slot_on_machine(...)`
  - `dispatch_stage_reversed_by_jobs(...)`
- stage-local suffix utilities:
  - `collect_stage_machine_suffix_job_ids(...)`
  - `swap_stage_machine_operation_sets(...)`

These utilities support:

- moving successor suffixes between two machines while preserving selected order
- removing predecessor operations and re-dispatching them backward under machine and precedence limits

### Gantt plotter changes

`hybridflowshop/painter/gantt.py` was refactored so:

- figures are created lazily
- `export_hybrid_flowshop_plot()` and `display_hybrid_flowshop_plot()` always close figures afterward
- extra figure reopening after export is removed

This reduces figure accumulation during repeated debug exports.

### Config and entrypoint changes

The staged diff also includes small updates in:

- `main.py`
- `main_metadata_debug.yaml`
- `configs_pw_cp/subroutine_flow_20260319-02.yaml`

These appear to support running and debugging the updated PW-CP flow.

### Verification/demo update

`verify/pw_cp_positive_boundary_deviation_demo.py` was updated to:

- call `_compute_right_slack_hint_values(...)` with the current `schedule` and `slack_occupying_ops` arguments
- print the currently computed hint values instead of stale hard-coded interpretations

## Follow-up Fixes Added During Review

### 1. Reverse-gap bug fix

During review, a real bug was found in reverse dispatch:

- `_get_latest_feasible_slot_on_machine(...)` failed to recognize feasible interior gaps
- example: with machine intervals `[2,4)` and `[8,10)`, upper bound `8`, duration `3`, the feasible slot `[5,8)` was incorrectly rejected

Fix:

- reverse-gap search now checks the gap `[current_op.end, next_boundary)` instead of mixing the current operation's start and end in the same iteration

Validation:

- a new regression test was added for the latest feasible interior gap case

### 2. Retiming gantt export gated by `debug_export`

During review, `_draw_candidate_retiming_gantts(...)` was found to run unconditionally in `_solve_subproblem(...)`.

Fix:

- retiming gantts are now rendered only when `debug_export=True`

Test updates:

- `_solve_subproblem(...)` tests that previously expected PNG creation with `debug_export=False` were updated to expect no retiming artifacts
- direct tests of `_draw_candidate_retiming_gantts(...)` remain to cover the rendering helper itself

### 3. CP-SAT search log export gated by `debug_export`

The PW-CP solver path was also always enabling `log_search_progress=True`, which caused files like:

- `4-pw_cp_cp_sat_search_batch=1.log`

to be written even when debug export was off.

Fix:

- both final and non-final PW-CP solve paths now pass `log_search_progress=debug_export`

Validation:

- tests now confirm:
  - non-final `_solve_subproblem(...)` disables search log capture when `debug_export=False`
  - final `_solve_subproblem(...)` enables it when `debug_export=True`

## Test Coverage Added or Updated

### `tests/controller/test_pw_cp.py`

Coverage now includes:

- successor reassignment:
  - boundary suffix movement
  - multiple slack intervals
  - no-op when already aligned
  - ordering by solved slack start
  - skip behavior when slack end equals makespan
- predecessor reassignment:
  - backward redispatch with deadlines
  - restriction to targeted machines
  - no-op when no movable predecessor exists
  - reverse stage processing order
  - infeasibility surfacing
- non-final `_solve_subproblem(...)` flow:
  - reassignment before semi-active retiming
  - `obj_value == 0` skips reassignment but still retimes
  - `obj_value < 0` raises
  - optional retiming Gantt rendering only under debug export
  - optional CP-SAT search-log capture only under debug export
- right-slack hint behavior:
  - common slack length calculation
  - `None` boundary skipping
  - clipped boundaries

### `tests/test_schedule_lite.py`

Coverage now includes:

- reverse dispatch:
  - latest feasible slot before LCT
  - latest feasible interior gap before LCT
  - tighter job deadlines
  - next-stage start bound
  - machine choice by latest feasible placement
  - use of `[0, first_start)` gap
  - infeasibility below time zero
- stage suffix and operation-set helpers:
  - suffix collection
  - order preservation
  - retiming via `do_make_semi_active=True`
  - duplicate selection errors
  - same-machine misuse errors

## Net Result

The staged change set plus follow-up fixes establish the following overall behavior:

- PW-CP non-final batches can reconstruct solved right slack more faithfully without changing the CP model
- realized machine ownership can be corrected after `create_schedule(...)`
- predecessor operations can be re-dispatched backward from solved slack anchors
- reverse dispatch now correctly handles interior gaps
- debug-only artifacts are now consistently gated:
  - initial and retiming Gantts
  - CP-SAT search logs
  - solution/debug exports

## Main Files Involved

- `hybridflowshop/controller/pw_cp.py`
- `hybridflowshop/cpsat_model_2/cumulative.py`
- `hybridflowshop/schedule_lite.py`
- `hybridflowshop/painter/gantt.py`
- `tests/controller/test_pw_cp.py`
- `tests/test_schedule_lite.py`
- `verify/pw_cp_positive_boundary_deviation_demo.py`
- `configs_pw_cp/subroutine_flow_20260319-02.yaml`
- `main_metadata_debug.yaml`
- `main.py`
