# PW-CP Problem Spec Building Fix

## Background

Current `hybridflowshop/controller/pw_cp.py` builds a subproblem spec from only:

- `optimization_ops`

and then internally derives the fixed-set split using a single time cutoff:

- `cutoff = min(start time of optimization_ops)`
- `time_fixed_set = all_ops - optimization_set`
- `right_boundary_time_fixed = {fixed op | start(op) >= cutoff}`

This is incorrect for the hybrid flow shop / flowshop structure.

## Why The Current Cutoff Logic Is Wrong

In a flowshop-style problem, operations across stages are linked by inter-stage precedence.
Because of that, a fixed operation can:

- start after the minimum start time of `optimization_ops`
- but still belong structurally to the "left" side of the current subproblem

So a single global time cutoff cannot correctly partition fixed operations into:

- left boundary fixed operations
- optimization operations
- right boundary fixed operations

This means the current `_build_subproblem_spec(...)` is making an invalid inference from time order alone.

## Required Design Change

Do not let `_build_subproblem_spec(...)` infer left/right fixed sets from `optimization_ops`.

Instead, change the API so that the caller explicitly passes three disjoint operation sets:

- `left_boundary_time_fixed_ops`
- `optimization_ops`
- `right_boundary_time_fixed_ops`

Optionally keep a fourth bucket for future extension:

- `boundary_profile_fixed_ops`

but v1 can still treat all non-optimization ops as time-fixed in the CP model.

The key point is:

- left / inside / right partitioning must be decided before calling `_build_subproblem_spec(...)`
- `_build_subproblem_spec(...)` should only assemble the spec and compute boundary profiles from the already-decided sets

## Expected Responsibility Split

### Caller side

The loop or a dedicated helper should:

- build the current batch structure
- decide which operations are in the optimization set
- decide which fixed operations belong to the left boundary side
- decide which fixed operations belong to the right boundary side
- pass these sets into `_build_subproblem_spec(...)`

### `_build_subproblem_spec(...)`

This method should:

- validate that the three sets are disjoint
- validate that their union is the intended full scheduled operation set, or at least that they are consistent with the current subproblem scope
- compute `left_boundary_profile` from `left_boundary_time_fixed_ops`
- compute `right_boundary_profile` from `right_boundary_time_fixed_ops`
- store all three sets in `PwCpSubproblemSpec`

It should **not**:

- compute a cutoff from optimization ops
- infer left/right membership from start times

## Boundary Computation After The Change

### Left boundary

Compute left boundary profile only from `left_boundary_time_fixed_ops`.

Interpretation:

- these are the already-fixed operations that occupy machine capacity before the optimization block
- their completion profile defines the first available time on each machine / stage for the current subproblem

### Right boundary

Compute right boundary profile only from `right_boundary_time_fixed_ops`.

Use the existing `schedule_lite.py::make_right_justified(...)` exactly as before:

- start from incumbent copy
- right-justify only `right_boundary_time_fixed_ops`
- collect stage-wise start times of the right-shifted operations
- sort ascending
- take the earliest `|M_i|` values as `D_i1 <= ... <= D_i|M_i|`

## CP Model Treatment In v1

The CP model policy can stay mostly unchanged:

- `optimization_ops`: free
- all other operations: fixed by `start == incumbent_start`

But for debug, boundary construction, and future extensibility, the spec must retain:

- `left_boundary_time_fixed_ops`
- `optimization_ops`
- `right_boundary_time_fixed_ops`

This keeps the implementation compatible with future variants where:

- left boundary fixed ops
- profile-fixed boundary ops
- outside fixed ops

may be treated differently.

## Concrete Refactor Targets

Update `PwCpSubproblemSpec` fields to reflect the explicit 3-way split.

Suggested field set:

- `batch_idx`
- `subproblem_idx`
- `left_boundary_time_fixed_ops`
- `optimization_ops`
- `right_boundary_time_fixed_ops`
- `boundary_profile_fixed_ops`
- `time_fixed_ops`
- `left_boundary_profile`
- `right_boundary_profile`
- `right_boundary_stage_start_times`

Update `_build_subproblem_spec(...)` signature from:

```python
def _build_subproblem_spec(
    self,
    incumbent: HybridFlowshopLiteSchedule,
    stage_2_job_2_p_dict: dict[str, dict[str, int]],
    batch_idx: int,
    optimization_ops: tuple[OperationRef, ...],
) -> PwCpSubproblemSpec:
```

to something like:

```python
def _build_subproblem_spec(
    self,
    incumbent: HybridFlowshopLiteSchedule,
    stage_2_job_2_p_dict: dict[str, dict[str, int]],
    batch_idx: int,
    left_boundary_time_fixed_ops: tuple[OperationRef, ...],
    optimization_ops: tuple[OperationRef, ...],
    right_boundary_time_fixed_ops: tuple[OperationRef, ...],
) -> PwCpSubproblemSpec:
```

## Recommended Implementation Steps

1. Add a helper that constructs the 3-way partition explicitly for each batch.
2. Remove cutoff-based left/right inference from `_build_subproblem_spec(...)`.
3. Make left boundary profile depend only on `left_boundary_time_fixed_ops`.
4. Make right boundary profile depend only on `right_boundary_time_fixed_ops`.
5. Keep non-optimization ops fixed in the CP model, but preserve left/right labels in the spec and debug export.
6. Update tests so they verify:
   - `_build_subproblem_spec(...)` does not infer left/right by cutoff
   - right boundary uses only the explicitly passed right-side fixed set
   - left boundary uses only the explicitly passed left-side fixed set

## Notes For The Next Implementer

- Do not try to repair the current cutoff rule; remove that inference path entirely.
- The bug is conceptual, not just an off-by-one or tie-breaking problem.
- The reason is precedence-coupled multi-stage structure: temporal ordering alone is insufficient to identify left/right boundary membership.
- Keep the refactor minimal if possible: change spec-building responsibility first, then only adjust downstream helpers as needed.
