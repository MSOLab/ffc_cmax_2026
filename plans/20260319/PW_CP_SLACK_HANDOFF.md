# PW-CP Slack Redefinition Handoff

This note summarizes the currently staged work related to:

- `left_profile_fixed_batch_count`
- `right_profile_fixed_batch_count`
- `profile_fix_by_machine`
- `machine_precedence_stride`

The goal is to help the next agent continue the PW-CP slack redesign without having to reconstruct the intent from the staged diff.

## 1. High-level summary

The staged changes are moving PW-CP away from the old `right_guard` framing toward a new `right_slack` framing.

At the same time, PW-CP partitioning is being expanded from a 3-way split:

- `left_time_fixed`
- `optimization`
- `right_time_fixed`

to a 5-way split:

- `left_time_fixed_ops`
- `left_profile_fixed_ops`
- `unfixed`
- `right_profile_fixed_ops`
- `right_time_fixed_ops`

The four optional kwargs are part of that new design:

- `left_profile_fixed_batch_count` and `right_profile_fixed_batch_count` control how many neighboring batches are treated as profile-fixed instead of time-fixed.
- `profile_fix_by_machine` and `machine_precedence_stride` control how precedence constraints are generated for profile-fixed operations.

## 2. What is already wired

### Controller forwarding

In [`hybridflowshop/controller/hfs_cp_lns.py`](/home/hjt/code/hybridflowshop/hybridflowshop/controller/hfs_cp_lns.py#L1948), the `pw_cp(...)` entrypoint now accepts and forwards all four kwargs into `PwCpConstructor.run(...)`.

This means the top-level API plumbing for these kwargs is already in place for PW-CP.

### PW-CP run signature

In [`hybridflowshop/controller/pw_cp.py`](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py#L332), `PwCpConstructor.run(...)` now accepts the same four kwargs and passes them into:

- `_build_operation_partition(...)`
- `_solve_subproblem(...)`

### Partitioning usage

In [`hybridflowshop/controller/pw_cp.py`](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py#L1042), the two batch-count kwargs are used only inside `_build_operation_partition(...)`.

Current staged semantics:

- batches before the left profile zone -> `left_time_fixed_ops`
- the previous `left_profile_fixed_batch_count` batches -> `left_profile_fixed_ops`
- current batch -> `unfixed`
- the next `right_profile_fixed_batch_count` batches -> `right_profile_fixed_ops`
- later batches -> `right_time_fixed_ops`

### Precedence usage

In [`hybridflowshop/controller/pw_cp.py`](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py#L632), if `spec.partition.profile_fixed_operations` is non-empty, the code builds a reduced schedule containing only those operations and calls:

- `BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(...)`

The two precedence kwargs are passed into that builder call.

### CP builder support

In [`hybridflowshop/cpsat_model_2/cumulative.py`](/home/hjt/code/hybridflowshop/hybridflowshop/cpsat_model_2/cumulative.py#L469), the builder already supports:

- `profile_fix_by_machine=False`
  - stage-level precedence selection based on start/end ordering
- `profile_fix_by_machine=True`
  - machine-sequence precedence fixing
- `machine_precedence_stride`
  - stride for machine-order arcs

So the optional kwargs are not just plumbed through; they already affect model construction.

## 3. Current staged design in `pw_cp.py`

### OperationPartition redesign

`OperationPartition` was changed to include:

- `left_time_fixed_ops`
- `left_profile_fixed_ops`
- `unfixed`
- `right_profile_fixed_ops`
- `right_time_fixed_ops`

It also now exposes:

- `time_fixed_operations`
- `profile_fixed_operations`
- `slack_occupying_operations`

This is the core structural change behind the new slack definition.

### Slack objective rename

The old `right_guard` naming is being replaced with `right_slack` in:

- log names
- helper names
- objective variable names
- tests

Examples:

- `_build_right_guard_profile(...)` -> `_build_right_slack_profile(...)`
- `_compute_right_guard_hint_values(...)` -> `_compute_right_slack_hint_values(...)`
- `_apply_right_guard_hints(...)` -> `_apply_right_slack_hints(...)`
- objective log name `"right_guard_slack"` -> `"right_slack"`

### New objective inputs

In [`hybridflowshop/cpsat_model_2/cumulative.py`](/home/hjt/code/hybridflowshop/hybridflowshop/cpsat_model_2/cumulative.py#L672), `add_right_slack_objective(...)` now takes:

- `left_profile_fixed_ops`
- `unfixed_ops`
- `right_profile_fixed_ops`
- `right_boundary_profile`
- `right_fixed_intervals`

The model treats all profile-occupying ops as consuming capacity before the right boundary.

## 4. Important behavioral details

### `left_profile_fixed_batch_count`

Current role:

- expands the movable neighborhood to include some earlier batches as profile-fixed instead of fully time-fixed
- those ops keep precedence structure only, not fixed start times

Current implementation point:

- only used by `_build_operation_partition(...)`

### `right_profile_fixed_batch_count`

Current role:

- similarly expands the neighborhood on the right side
- those ops become profile-fixed, while later batches remain time-fixed

Current implementation point:

- only used by `_build_operation_partition(...)`

### `profile_fix_by_machine`

Current role:

- decides how profile-fixed precedence is derived from the incumbent schedule

Current semantics in builder:

- `False`: stage-level precedence selection
- `True`: machine-order precedence selection

### `machine_precedence_stride`

Current role:

- only matters when `profile_fix_by_machine=True`
- controls spacing between predecessor and successor positions in each machine sequence

Examples from builder docstring:

- `1`: adjacent arcs
- `2`: every-other arcs

Validation already exists:

- `machine_precedence_stride < 1` raises `ValueError`

## 5. Places that look unfinished or risky

### The slack concept is still partly transitional

The staged code clearly moves toward a better decomposition, but it is still transitional rather than fully settled.

Reasons:

- the user already wants to redefine slack from scratch
- several names changed, but some semantics still inherit the old boundary/guard structure
- the current objective still depends on a right-justified boundary profile anchored by `right_time_fixed_ops`

### `promote_job_contained_ops()` exists but is not active

In [`hybridflowshop/controller/pw_cp.py`](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py#L393), this line is commented out:

```python
# partition = partition.promote_job_contained_ops()
```

This is an important unfinished decision.

Why it matters:

- without promotion, one job can be split across `unfixed` and `profile_fixed`
- that may be okay if intended
- but the existence of the helper suggests this was already seen as a possible consistency issue

### Stage-wide boundary cap may be stronger than intended

In [`hybridflowshop/cpsat_model_2/cumulative.py`](/home/hjt/code/hybridflowshop/hybridflowshop/cpsat_model_2/cumulative.py#L708), `add_right_slack_objective(...)` computes `stage_max_boundary` and then constrains all profile-occupying ops in the stage with:

```python
mdl.add(variables.op_end[job_id, stage_id] <= stage_max_boundary)
```

This is not per-machine. It is a stage-wide cap.

That may be correct for the current cumulative construction, but it is also a likely place to revisit if slack is being redefined from first principles.

### `_build_operation_partition(...)` still returns `right_justified_sched`

The function still returns `(partition, right_justified_sched)`, but the caller now ignores the second value:

```python
partition, _ = self._build_operation_partition(...)
```

This suggests leftover structure from earlier debug/export flow.

### Debug/export flow was rewritten around the new slack helper

The Gantt debug path now uses `_compute_initial_right_slack_summaries(...)` so visualization matches hint computation. That is good, but it also means future slack changes should update both:

- objective/hint logic
- debug visualization logic

## 6. Tests already added or updated

Most supporting tests are in [`tests/controller/test_pw_cp.py`](/home/hjt/code/hybridflowshop/tests/controller/test_pw_cp.py).

The staged diff includes tests for:

- right slack profile generation
- hint generation under slack naming
- slack helper behavior with profile-fixed ops
- current-machine assignment fallback in slack helper
- debug Gantt export using slack helper values
- objective log rename to `right_slack`
- `promote_job_contained_ops()`

This gives a reasonable safety net for refactoring, but those tests mostly validate the currently staged interpretation, not a clean-sheet slack redesign.

## 7. Suggested direction for the next agent

If the plan is truly "redefine slack from scratch", I would treat the current staged work as scaffolding, not as the final target.

Recommended sequence:

1. Write down the exact new slack definition before changing code.
2. Decide whether slack is:
   - machine-local
   - stage-global
   - based on right-time-fixed anchors only
   - or based on a different boundary notion entirely
3. Decide whether profile-fixed ops should only preserve relative order, or also preserve machine assignment.
4. Decide whether a single job is allowed to span `unfixed` and `profile_fixed` partitions.
5. Only after that, revisit `_build_operation_partition(...)`, `add_right_slack_objective(...)`, and `_compute_initial_right_slack_summaries(...)` together.

## 8. Concrete code areas the next agent will likely edit

- [`hybridflowshop/controller/pw_cp.py`](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)
  - `OperationPartition`
  - `_build_operation_partition(...)`
  - `_build_right_slack_profile(...)`
  - `_compute_initial_right_slack_summaries(...)`
  - `_solve_subproblem(...)`

- [`hybridflowshop/cpsat_model_2/cumulative.py`](/home/hjt/code/hybridflowshop/hybridflowshop/cpsat_model_2/cumulative.py)
  - `add_right_slack_objective(...)`
  - `add_stage_ops_precedence_constraints_after_dispatch_from_schedule(...)`

- [`hybridflowshop/controller/hfs_cp_lns.py`](/home/hjt/code/hybridflowshop/hybridflowshop/controller/hfs_cp_lns.py#L1948)
  - public PW-CP entrypoint kwargs are already present; probably only docs or validation may need updates

- [`tests/controller/test_pw_cp.py`](/home/hjt/code/hybridflowshop/tests/controller/test_pw_cp.py)
  - adjust or replace tests once the new slack semantics are finalized

## 9. Practical takeaway

The optional kwargs are already connected end-to-end for PW-CP:

- controller API -> constructor -> partitioning / precedence builder

So the next agent does not need to do plumbing first.

The real remaining work is semantic:

- redefine slack cleanly
- decide how profile-fixed neighborhoods should behave
- then align partitioning, objective, hints, debug export, and tests to that single definition
