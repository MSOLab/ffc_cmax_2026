# PW-CP Objective Fix For Mid Batches

## Background

Current objective code in `hybridflowshop/controller/pw_cp.py` is:

- `_add_boundary_violation_objective(...)`
- computes per-stage `max(C_ik - D_ik)`
- minimizes the global maximum violation
- then applies a second lexicographical phase to minimize makespan

This no longer matches the intended behavior.

## Required Objective Change

Replace the current **boundary violation** definition with **boundary improvement**.

For each aligned pair `(C_ik, D_ik)`, define:

```text
boundary_improvement_ik = D_ik - C_ik
```

Interpretation:

- if `boundary_improvement_ik < 0`:
  - this is effectively a boundary violation
  - the optimized block intrudes past the boundary
- if `boundary_improvement_ik > 0`:
  - this is how much smaller / earlier the optimized completion profile became
  - relative to the incumbent boundary

So the sign convention must flip compared with the current implementation:

- current code uses `C_ik - D_ik`
- new code must use `D_ik - C_ik`

## Optimization Goal

The new objective for non-final batches should be:

- maximize the minimum boundary improvement

That is:

```text
maximize min_k(D_ik - C_ik)
```

and across stages:

- take the minimum over all stage/rank improvements participating in the batch
- maximize that global minimum improvement

This gives the intended behavior:

- large positive value means all compared boundaries improved well
- negative value means there is still a violation somewhere
- maximizing the minimum pushes the worst boundary pair upward first

## Important Behavioral Change

Do **not** use lexicographical optimization anymore.

The objective should depend on whether the current batch is the final batch.

### If current batch is not the last batch

Objective:

- maximize minimum(boundary improvement)

Equivalent formulation:

- create per-pair `boundary_improvement`
- create `global_min_boundary_improvement`
- enforce `global_min_boundary_improvement <= boundary_improvement` for every pair
- maximize `global_min_boundary_improvement`

### If current batch is the last batch

Objective:

- minimize makespan

No intermediate boundary-improvement objective is needed for the last batch.

## Required API / Control-Flow Change

The solve path needs to know whether the current batch is the final batch.

Add an explicit boolean such as:

- `is_last_batch: bool`

and pass it from the batch loop into the subproblem solver / objective builder.

Likely affected methods:

- main loop around `batch_idx`
- `_solve_subproblem(...)`
- current `_add_boundary_violation_objective(...)`

Rename the objective helper to reflect the new semantics, for example:

- `_add_boundary_improvement_objective(...)`

## Modeling Guidance

For each stage:

1. Build `C_ik` exactly as today from k-th largest completion values.
2. Keep the boundary alignment logic explicit and documented.
3. Replace:

```text
violation_ik >= C_ik - D_ik
violation_ik >= 0
minimize max(violation_ik)
```

with a direct signed-improvement model:

```text
improvement_ik == D_ik - C_ik
```

Then define:

```text
global_min_improvement <= improvement_ik   for all i, k
maximize global_min_improvement
```

Since improvements may be negative, `global_min_improvement` must allow negative values.
Its lower bound should therefore be something like `-horizon`, not `0`.

Similarly each `improvement_ik` variable should support negative values.

## Why This Is Different From The Old Model

Old model:

- focuses on reducing the worst positive intrusion amount
- uses nonnegative violation vars only
- then performs a second makespan phase

New model:

- keeps the sign of the comparison
- treats negative values as violations and positive values as gains
- directly maximizes the worst signed improvement for mid batches
- switches to pure makespan minimization for the final batch

## Recommended Refactor Steps

1. Rename `_add_boundary_violation_objective(...)` to `_add_boundary_improvement_objective(...)`.
2. Replace nonnegative violation vars with signed improvement vars.
3. Add `global_min_boundary_improvement` with domain `[-horizon, horizon]`.
4. For non-final batches:
   - maximize `global_min_boundary_improvement`
5. For the final batch:
   - skip boundary-improvement objective
   - minimize `variables.makespan`
6. Remove the current lexicographical second solve.
7. Update debug / comments so “violation” terminology is replaced with “improvement”.

## Tests To Update

Adjust tests so they verify:

- signed improvement becomes negative when `C_ik > D_ik`
- signed improvement becomes positive when `C_ik < D_ik`
- mid-batch objective is maximize-min-improvement
- final-batch objective is minimize-makespan
- no lexicographical second optimization phase remains

## Notes For The Next Implementer

- Keep the existing `C_ik` construction unless there is a separate reason to change it.
- The main change is objective semantics and batch-position-dependent branching.
- Be careful with CP variable domains: improvement vars now need negative ranges.
- The final batch is a distinct mode, not just another lexicographical phase.
