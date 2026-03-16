# Fix Right Boundary Definition For PW-CP

## Summary

Current non-final PW-CP objective can start with a positive incumbent hint value even when the incumbent schedule is feasible.

This is not a CP-SAT hint bug. It comes from a mismatch between:

- how `current batch` completion is summarized, and
- how `right boundary` is summarized.

The current formulation is stage-level and machine-agnostic:

- `C_{i,k}` = k-th largest completion among current-batch operations on stage `i`
- `D_{i,k}` = k-th aligned start from the right batch after right-justifying only the right-side operations

This can produce `C_{i,k} > D_{i,k}` even for a feasible incumbent.


## Confirmed Counterexample

See:

- [verify/pw_cp_positive_boundary_deviation_demo.py](/home/hjt/code/hybridflowshop/verify/pw_cp_positive_boundary_deviation_demo.py)

This script reproduces a feasible 1-stage 2-machine schedule where:

- current batch = `(B, C)`
- right batch = `(A, D)`
- `right_boundary_profile = {'s1': [0, 8]}`
- `global_max_deviation = 2`

The script demonstrates that the current stage-level boundary summary is enough to create a positive deviation for a feasible incumbent.


## Root Cause

### Current batching

In [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py), `_build_stage_batches()` groups stage operations by:

- midpoint by default, not strict time-frontier order
- then slices them into fixed-size batches

Relevant code:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py#L336)

This means `current batch` and `right batch` are not guaranteed to form a clean left/right partition in time.

### Current right boundary profile

In `_compute_right_boundary_profile()`:

- copy incumbent
- right-justify only `right_time_fixed_ops`
- collect all start times of those right-fixed ops stage-wise
- keep only the earliest `mc_cnt` starts

Relevant code:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py#L443)

This throws away machine identity and compresses the right side into a stage-level frontier.

### Current objective

In `BaseModelBuilder.add_boundary_deviation_objective()`:

- collect all completion vars for current batch on a stage
- create `C_{i,k}` as k-th largest completion
- compare them to reversed right-boundary starts

Relevant code:

- [cumulative.py](/home/hjt/code/hybridflowshop/hybridflowshop/cpsat_model_2/cumulative.py#L674)

So the current comparison is:

- "latest current completions"
- versus
- "earliest right-side starts"

This is not a valid certificate of pairwise non-overlap.


## Constraint / Design Context

The CP model is intentionally machine-agnostic.

- It uses cumulative capacity constraints.
- It does not use optional intervals per machine because that formulation was too slow.

Therefore the replacement boundary notion should ideally:

- remain machine-agnostic in the CP model
- avoid optional interval machine assignment variables
- still make incumbent objective values interpretable


## Recommended Direction

Replace the current stage-level right-boundary objective with a cumulative-frontier objective.

### High-level idea

Instead of using:

- `C_{i,k}` = k-th largest current completion
- `D_{i,k}` = stage-wise earliest right starts

use cumulative usage frontiers:

- `T_{i,r}` = last time current-batch usage on stage `i` was at least `r`
- `S_{i,r}` = first time right-batch usage on stage `i` is at least `r` after right-justification

for `r = 1..|M_i|`.

Then define:

- `deviation_{i,r} = T_{i,r} - S_{i,r}`
- objective = `max_{i,r} deviation_{i,r}`

This keeps the formulation machine-agnostic while comparing like with like:

- current-side cumulative frontier
- right-side cumulative frontier


## Why This Direction Fits The Existing Model

### Pros

- compatible with cumulative-capacity modeling
- does not require optional interval machine assignment formulation
- avoids stage-level misalignment from "k-th latest completion vs k-th earliest start"
- should make incumbent objective more meaningful

### Important note

This still needs validation.

The expectation is:

- the incumbent should no longer get misleading positive values caused by the current stage-level summary

But this must be checked with:

- the small counterexample in `verify/`
- several real PW-CP logs


## Alternative Direction Considered

Use machine-wise right boundary:

- right boundary = first right-side start per machine
- current side = last current completion per machine

This would be more intuitive, but it conflicts with the current machine-agnostic CP formulation because:

- the CP model does not know exact machine assignment for optimized operations
- cumulative only tracks aggregate capacity, not machine identity

So machine-wise comparison is not a natural fit unless the model is changed more deeply.


## Required Changes

### 1. Redefine boundary profile data structure

Current alias:

- `StageBoundaryProfile = dict[str, list[int]]`

Potential replacement:

- keep `StageBoundaryProfile` name but reinterpret it as cumulative frontier ranks
- or rename to something like `StageUsageFrontier`

Suggested shape:

- `dict[stage_id, list[int]]`
- index `r-1` stores the frontier for usage level `r`

This preserves a similar outer type while changing the meaning.


### 2. Replace `_compute_right_boundary_profile()`

Current implementation:

- right-justify right-fixed ops
- collect stage-wise start times
- take earliest `mc_cnt`

Needed replacement:

- right-justify right-fixed ops
- for each stage, compute right-side cumulative frontier:
  - `S_{i,1}`, `S_{i,2}`, ..., `S_{i,m_i}`
- store these rank-frontier values instead of raw earliest starts

Relevant file:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)

Relevant function:

- `_compute_right_boundary_profile()`


### 3. Replace `add_boundary_deviation_objective()`

Current implementation uses:

- order statistics on completion vars (`C_{i,k}`)

Needed replacement:

- introduce frontier variables for current-batch cumulative usage:
  - `T_{i,1}`, `T_{i,2}`, ..., `T_{i,m_i}`
- compare with right-side frontier values:
  - `deviation_{i,r} = T_{i,r} - S_{i,r}`

Relevant file:

- [cumulative.py](/home/hjt/code/hybridflowshop/hybridflowshop/cpsat_model_2/cumulative.py)

Relevant function:

- `BaseModelBuilder.add_boundary_deviation_objective()`


### 4. Replace incumbent hint computation

Current hint computation mirrors the current objective:

- computes `C_*`
- computes `boundary_deviation_*`
- computes `max_deviation_*`
- computes `global_max_deviation`

Needed replacement:

- compute incumbent `T_{i,r}` values
- compute `deviation_{i,r}`
- compute stage/global max deviation from the new frontier definition

Relevant file:

- [pw_cp.py](/home/hjt/code/hybridflowshop/hybridflowshop/controller/pw_cp.py)

Relevant functions:

- `_compute_boundary_deviation_hint_values()`
- `_apply_boundary_deviation_hints()`


### 5. Update debug/export fields

Current debug export stores:

- `right_boundary_profile`
- `right_boundary_stage_start_times`

Needed changes:

- update these fields to reflect the new meaning
- possibly add a new field name to avoid confusion

Relevant function:

- `_save_solution_dict()`


### 6. Update terminology

Current terms are easy to misread as:

- hard no-overlap boundary
- infeasibility detector

Suggested terminology update:

- `boundary_deviation` -> something like `right_frontier_deviation`
- `right_boundary_profile` -> something like `right_usage_frontier`

This is optional but strongly recommended.


## Suggested Modeling Approach For `T_{i,r}`

This still needs implementation design work.

Candidate idea:

- Use end times of current-batch operations on stage `i` as candidate event times.
- For each rank `r`, define `T_{i,r}` as the largest candidate time such that current-batch cumulative usage is at least `r` immediately before or at that event.

Possible implementation approaches:

1. Event-based Bool encoding
- candidate times = end times of current-batch ops
- Bool says whether stage usage at that candidate time is at least `r`
- `T_{i,r}` is max selected candidate

2. Simpler incumbent-only prototype first
- first compute frontiers outside the model for the incumbent only
- verify they behave as intended on small and real examples
- then implement the CP encoding

Recommended order:

- start with offline verification in `verify/`
- only then implement in the CP model


## Validation Checklist

### Small examples

Use or extend:

- [verify/pw_cp_positive_boundary_deviation_demo.py](/home/hjt/code/hybridflowshop/verify/pw_cp_positive_boundary_deviation_demo.py)

Need to confirm:

- current formulation gives positive incumbent deviation on the demo
- new frontier formulation does not produce the same misleading positive value

### Unit tests

Update/add tests in:

- [test_pw_cp.py](/home/hjt/code/hybridflowshop/tests/controller/test_pw_cp.py)

Suggested new tests:

- incumbent frontier deviation on the demo instance is `<= 0` or exactly the intended baseline value
- new right-boundary/frontier computation matches hand-calculated examples
- hint variable values match the new objective auxiliaries

### Real-instance validation

Re-run PW-CP on:

- the scenario that produced positive incumbent objective values

Check:

- initial non-final complete hint objective values
- quality of subsequent improvements
- solver speed/regression


## Implementation Order Recommendation

1. Add a new `verify/` script that computes cumulative frontiers for small examples.
2. Decide exact mathematical definition of `T_{i,r}` and `S_{i,r}`.
3. Update `_compute_right_boundary_profile()` to compute right-side frontiers.
4. Update `add_boundary_deviation_objective()` to use frontier vars instead of k-th completion vars.
5. Update incumbent hint computation.
6. Update tests and debug export.
7. Run real-instance validation.


## Non-goals For The Next Chat

Avoid doing these immediately unless needed:

- switching to optional-interval machine assignment formulation
- large refactors unrelated to non-final boundary objective
- changing final-batch objective behavior


## Current State Of Repo

At the time of writing:

- a counterexample demo exists in `verify/`
- tests currently pass after adding the demo-related regression test
- the non-final objective itself has not yet been redesigned

