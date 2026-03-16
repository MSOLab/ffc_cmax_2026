# Fix Right Boundary With Guard Intervals

## Goal

Replace the current non-final PW-CP boundary objective with a lighter formulation based on guard intervals.

This design is intended for the current CP model structure:

- stage-level cumulative resource constraints
- identical parallel machines
- no machine-specific optional interval assignment for optimization ops

The previous frontier-based direction caused too many variables.


## Chosen Direction

For each stage, create `m_i` generic guard intervals, where `m_i` is the machine count of that stage.

Each guard interval represents a protected left-side slack region immediately before a right-side boundary time.

The non-final objective becomes:

- maximize the minimum guard extension across all stage/rank guards

This avoids:

- machine assignment variables
- large Bool encodings for usage frontiers


## Core Idea

For each stage `i`:

1. Compute right-justified start times of `right_time_fixed_ops`.
2. Sort those start times and take the earliest `m_i` values:
   - `b_{i,1} <= b_{i,2} <= ... <= b_{i,m_i}`
3. For each rank `r`, create a guard interval:
   - `guard_end_{i,r} = b_{i,r}` fixed
   - `guard_size_{i,r} = extra_{i,r}` or `min_size_{i,r} + extra_{i,r}`
   - `guard_start_{i,r} = guard_end_{i,r} - guard_size_{i,r}`
4. Build an additional cumulative constraint on each stage containing:
   - optimization operation intervals
   - fixed-time right-side intervals
   - guard intervals
5. Maximize the minimum `extra_{i,r}` across all guards.


## Important Correction

Do not set guard end to `makespan`.

That would make guards overlap the actual right-fixed operations.

Correct design:

- `guard_end_{i,r} = b_{i,r}`

and include right-fixed operations as fixed intervals in the same auxiliary cumulative.


## Why This Should Work

Because the additional cumulative sees:

- optimization ops consuming capacity on the left
- right-fixed ops consuming capacity on the right at their fixed times
- guards trying to reserve extra slack immediately before the right boundary

If the objective increases the guards, optimization ops must move left to make room.

This gives a direct "maximize right-side slack" objective without requiring machine-specific assignment.


## Open Design Choice: Minimum Size

Two options:

### Option A: Zero-baseline guards

- `guard_size_{i,r} = extra_{i,r}`
- `extra_{i,r} >= 0`

Interpretation:

- zero is acceptable
- objective tries to create positive slack where possible

### Option B: Baseline + extra

- `guard_size_{i,r} = min_size_{i,r} + extra_{i,r}`

This only makes sense if a meaningful baseline slack is known and desired.

Current recommendation:

- start with Option A
- keep `extra_{i,r} >= 0`
- maximize the minimum `extra`

Reason:

- simpler
- easier to validate
- aligns with the user's statement that zero extra is acceptable


## Data To Compute Outside The Model

### Right boundary times

For each stage `i`:

- right-justify `right_time_fixed_ops`
- collect their start times
- sort ascending
- take earliest `m_i`

If fewer than `m_i` right-side starts exist, decide a padding rule.

Recommended padding rule:

- pad with `makespan`

Reason:

- a missing right op means that generic slot has no earlier right-side blocker
- `makespan` is the least restrictive boundary


## Proposed Mathematical Formulation

For stage `i` with machine count `m_i`, for each rank `r = 1..m_i`:

- parameter `b_{i,r}`: right boundary time
- variable `extra_{i,r} >= 0`
- variable `guard_start_{i,r} >= 0`
- interval `guard_{i,r} = [guard_start_{i,r}, b_{i,r})`

Constraints:

- `guard_start_{i,r} + extra_{i,r} = b_{i,r}`
- `0 <= extra_{i,r} <= b_{i,r}`

Auxiliary cumulative on stage `i`:

- intervals:
  - optimization op intervals on stage `i`
  - fixed right-side intervals on stage `i`
  - guard intervals on stage `i`
- demands:
  - all 1
- capacity:
  - `m_i`

Objective:

- introduce `stage_guard_min_i`
- `stage_guard_min_i <= extra_{i,r}` for all `r`
- introduce global `guard_min`
- `guard_min <= stage_guard_min_i` for all stages
- maximize `guard_min`

Alternative objective:

- maximize sum of `stage_guard_min_i`

Current recommendation:

- maximize global `guard_min`

Reason:

- encourages balanced improvement across stages/ranks
- simplest interpretation


## What Must Be In The Auxiliary Cumulative

### Must include

- optimization intervals on the stage
- right-fixed operations as fixed-time intervals
- guard intervals

### Must not rely on

- machine-specific assignment of optimization ops
- optional machine choice literals


## Why Right-Fixed Ops Should Be Added

The user requirement is:

- stage completion/late-finish behavior should be judged while right-fixed operations are present at fixed times

So the auxiliary cumulative should explicitly include those fixed right-side intervals.

This makes the additional non-final objective constraint feel like a partial schedule:

- left = optimization ops
- right = fixed right-side ops
- middle slack = guards


## Required Code Changes

### 1. Keep or simplify right boundary computation

Relevant file:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)

Relevant function:

- `_compute_right_boundary_profile()`

Needed behavior:

- return earliest `m_i` right-justified starts per stage
- no frontier/Bool encoding
- just sorted boundary times

Potentially rename for clarity:

- `_compute_right_boundary_starts()`

Current `StageBoundaryProfile = dict[str, list[int]]` can likely stay unchanged.


### 2. Add helper to build fixed right-side intervals

Relevant file:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)

Needed new helper:

- given incumbent and `right_time_fixed_ops`
- produce per-stage fixed intervals:
  - start
  - end
  - duration

These should come from the right-justified schedule used to compute boundaries, not from the original incumbent times.

Suggested helper:

- `_compute_right_fixed_intervals_for_auxiliary_cumulative()`


### 3. Replace `add_boundary_deviation_objective()`

Relevant file:

- [cumulative.py](/home/hjt/code/hybridflowshop/hybridflowshop/cpsat_model_2/cumulative.py)

Current function to replace or deprecate:

- `BaseModelBuilder.add_boundary_deviation_objective()`

Needed replacement:

- something like `add_right_guard_objective()`

Responsibilities:

- create guard vars/intervals
- build auxiliary cumulative with:
  - optimization intervals
  - fixed right intervals
  - guards
- create `stage_guard_min_*`
- create global `guard_min`
- maximize `guard_min`


### 4. Update non-final solve path

Relevant file:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)

Current non-final path:

- calls `BaseModelBuilder.add_boundary_deviation_objective(...)`

Needed change:

- call the new guard-objective builder instead

Also update:

- objective naming in logs
- subproblem metadata objective name

Suggested new name:

- `right_guard_slack`
or
- `guard_min_slack`


### 5. Replace incumbent hint computation

Relevant file:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)

Current functions tied to old formulation:

- `_compute_boundary_deviation_hint_values()`
- `_apply_boundary_deviation_hints()`

Needed replacement:

- compute incumbent guard slack values
- hint:
  - each `extra_{i,r}`
  - each stage min var
  - global min var

Suggested helpers:

- `_compute_right_guard_hint_values()`
- `_apply_right_guard_hints()`

Important:

- right-fixed intervals are fixed, so they do not need hints
- hints are only for objective-related auxiliary vars


### 6. Debug/export metadata update

Relevant file:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)

Current debug export stores:

- `right_boundary_profile`
- `right_boundary_stage_start_times`

Recommended additions:

- right-justified fixed right intervals used in the auxiliary cumulative
- guard boundaries per stage

Suggested fields:

- `right_guard_boundaries`
- `right_fixed_intervals`


## Questions To Resolve In Next Chat

1. Padding rule when right-side start count is less than machine count:
   - use `makespan`?
   - use repeated last boundary?

Current recommendation:

- use `makespan`

2. Objective:
   - maximize global min slack?
   - maximize sum of stage mins?
   - lexicographic combination?

Current recommendation:

- maximize global min slack

3. Should guards be allowed to have zero size?

Current recommendation:

- yes

4. Do we need one auxiliary cumulative per stage only, or any cross-stage linking?

Current recommendation:

- per-stage only
- no extra cross-stage links beyond existing model constraints


## Validation Plan

### Small tests

Use:

- [verify/pw_cp_positive_boundary_deviation_demo.py](/home/hjt/code/hybridflowshop/verify/pw_cp_positive_boundary_deviation_demo.py)

Add a new verify script for the guard formulation:

- construct small stage examples
- compute boundary starts
- compute incumbent guard slack by simulation
- confirm interpretation is intuitive

### Unit tests

Update/add in:

- [test_pw_cp.py](/home/hjt/code/hybridflowshop/tests/controller/test_pw_cp.py)

Suggested tests:

- right boundary starts are computed correctly after right-justification
- fixed right intervals for auxiliary cumulative match the shifted schedule
- incumbent guard slack hints match hand-calculated examples
- guard hint set contains no duplicate variables
- objective name/logging updated from old boundary deviation path

### Real-instance validation

Re-run the problematic scenario and check:

- initial complete-hint objective values
- variable count vs frontier-based attempt
- solve speed
- whether non-final objective behaves more intuitively


## Non-goals

Do not do these in the next implementation step:

- machine-specific optional interval formulation
- full redesign of base cumulative model
- changes to final-batch makespan objective


## Suggested Implementation Order

1. Keep right-boundary start computation simple and stable.
2. Add helper to compute right-justified fixed right intervals.
3. Implement guard-objective builder in `cumulative.py`.
4. Replace non-final objective call path in `pw_cp.py`.
5. Replace hint computation for non-final batches.
6. Update tests.
7. Validate on the real scenario.

