# Plan: Add method to remove jobs by ID set

## Context

The user wants to add a new method to `HybridFlowshopLiteSchedule` class that removes all operations for a given set of job IDs. Currently, the class has:

- `remove_operations(removed_ops: set[tuple[JobIdType, StageIdType, McIdType]])` - requires knowing all (job, stage, machine) tuples
- `deepcopy(job_subsequence: set[JobIdType] | None)` - can create a copy with only specific jobs

The user wants a more convenient method that takes just a set of job IDs and removes all their operations across all stages and machines.

## Requirements

- Add a new method that accepts `job_ids: set[JobIdType]`
- Remove all operations for those jobs across all stages and machines
- Update both `__stage_2_mc_2_job_tuple_seq` and `__stage_2_job_2_end_time` data structures
- Follow existing code style and patterns in the class

## Test Plan (First)

Create a new test file `tests/test_schedule_lite_setters_remove.py` with:

1. Copy existing `remove_operations` tests from `test_schedule_lite.py`:
   - `test_remove_operations_removes_specified_ops_and_updates_cache`
   - `test_remove_operations_multiple_ops_same_machine`

2. Add new tests for `remove_jobs` method:
   - Test removing a single job across all stages
   - Test removing multiple jobs
   - Test removing all jobs
   - Test removing empty job set (no change)
   - Test that makespan is correctly updated after removal
   - Test that job end time caches are properly cleared

## Implementation Approach

Add a method `remove_jobs(job_ids: set[JobIdType]) -> None` to `HybridFlowshopLiteSchedule` that:

1. Iterates through all stages and machines
2. For each machine, filters out job tuples matching the given job IDs
3. Removes corresponding entries from the job end time cache

This can reuse the logic from `remove_operations` but simplifies the API for the common case of removing by job ID.

## Critical Files

- **New**: `tests/test_schedule_lite_setters_remove.py` - New test file
- **Modify**: `/home/hjt/code/hybridflowshop/hybridflowshop/schedule_lite.py` - Add the new method near `remove_operations` (around line 1700)
- **Reference**: `/home/hjt/code/hybridflowshop/tests/test_schedule_lite.py` - Existing tests to copy (lines 187-263)

## Verification

Run tests with:

```bash
uv run pytest tests/test_schedule_lite_setters_remove.py -v
```

After implementation, verify:

- All new tests pass
- Existing tests in `test_schedule_lite.py` still pass
- The new method correctly removes all operations for specified jobs
