# Nested progression alignment

## Background

During inspection of one `incremental_pw_cp` run, the controller log showed
five PW-CP makespan updates:

- `982`
- `980`
- `978`
- `977`
- `976`

However, the generated `results/subroutine_progression.json` only kept two
points for `incremental_pw_cp`:

- `980`
- `976`

The same structural risk also applied to `repeat_while_improvement`, because
its parent progression depended on reconstruction of nested child reports.

## Root cause

The mismatch came from two separate issues.

### 1. Nested call start times were not tracked for JSON reconstruction

`get_progression_data()` tries to rebuild parent progression from nested child
reports stored in `solution_manager.history`. That logic needs the child
`call_context` start time to convert child-local timestamps into the parent
time axis.

Previously, only top-level recorded flow calls were stored in
`_subroutine_call_meta_list`. Nested contexts such as:

- `incremental_pw_cp -> unfixed_batch_count_* -> pw_cp`
- `repeat_while_improvement -> reps_* -> child subroutine`

did not always have start/end metadata available through the progression
recorder. As a result, nested report reconstruction silently failed and JSON
fell back to the top-level live recorder, which only had final per-pass
updates such as `980` and `976`.

### 2. PW-CP report progression mixed in the wrong source

`pw_cp` reported progression through `result.sub_obj_store.obj_value_series`.
That series was populated from per-batch incumbent values together with raw
subproblem objective records. For parent wrapper reconstruction, the correct
semantic source is the accepted makespan improvement sequence, not the raw
internal solver objective stream.

## Implemented changes

### Controller progression recorder

In `hybridflowshop/controller/controller_core.py`:

- added method-context metadata tracking for every nested `call_context`
- recorded start/end timestamps for `_call_method()` and
  `temporarily_extended_context()`
- added lookup helpers so nested child reports can be mapped back onto the
  parent global/local time axis
- updated nested reconstruction to use `report.progress_obj_value_records`
  together with `progress_time_basis`
- rebuilt `combined_progress_list` from finalized per-call progression instead
  of returning the raw live-recorder list

This makes parent wrappers reconstruct child progression consistently for both:

- `incremental_pw_cp`
- `repeat_while_improvement`

### PW-CP progression semantics

In `hybridflowshop/controller/pw_cp.py`:

- stopped seeding the main PW-CP progression store with the initial schedule
- stopped copying raw subproblem objective records into the main PW-CP
  progression store
- now record only accepted incumbent makespan improvements into the PW-CP
  progression series

This keeps the child `pw_cp` report aligned with the parent wrapper semantics:
the progression now means accepted makespan improvements over time.

## Expected behavior after the change

For wrapper-like parent calls:

- `incremental_pw_cp` parent progression includes all child-reported
  intermediate improvements
- `repeat_while_improvement` parent progression includes all child-reported
  intermediate improvements
- `combined_progress_list` is aligned with the same reconstructed progression
  used by `subroutine_calls[*].local_progress_list`

For PW-CP child reports:

- progression points represent accepted incumbent makespan improvements only
- non-improving or rejected batches do not add new main progression points

## Tests added or updated

### `tests/controller/test_subroutine_progression_recorder.py`

Added regression coverage for:

- nested `incremental_pw_cp` reconstruction preserving
  `982 -> 980 -> 978 -> 977 -> 976`
- nested `repeat_while_improvement` reconstruction preserving all child
  intermediate points
- `combined_progress_list` matching reconstructed per-call progression

### `tests/controller/test_pw_cp.py`

Added regression coverage verifying that the PW-CP progression store keeps only
accepted incumbent improvements.

## Verification

Executed:

```bash
uv run ruff check hybridflowshop/controller/controller_core.py hybridflowshop/controller/pw_cp.py tests/controller/test_subroutine_progression_recorder.py tests/controller/test_pw_cp.py
uv run pytest tests/controller/test_subroutine_progression_recorder.py tests/controller/test_pw_cp.py tests/controller/test_incremental_pw_cp.py tests/test_method_progression_report.py -q
```

Result:

- Ruff passed
- `47` tests passed

## Notes

- Existing run artifacts are not rewritten automatically.
- To observe the corrected JSON on real data, the target scenario must be run
  again and new `subroutine_progression.json` files must be generated.
