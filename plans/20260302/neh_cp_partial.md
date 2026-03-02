# NEH-CP Partial Reconstruction Plan

## Context

The `NehCpConstructor.run` method in `/home/hjt/code/hybridflowshop/hybridflowshop/controller/neh_cp.py` currently reconstructs a full schedule from scratch using a reference schedule only as a starting point for job sequence determination. The user wants to add an option to:

1. Sort jobs by their first stage start time in the reference schedule
2. Remove a portion of tail jobs (e.g., 25%) from the reference schedule
3. Reconstruct the full schedule using the current NEH-CP batch-add approach for the removed jobs

This enables "partial reconstruction" where the head portion of a good solution is preserved while the tail is re-optimized.

## Requirements

- Add a new parameter `preserved_head_job_portion: float = 0.0` (numeric kwargs default to 0 per convention)
- Behavior based on parameter value:
  - `0.0`: Current behavior (reconstruct all jobs from scratch)
  - `< 0.0`: Log warning, treat as 0.0 (full reconstruction mode)
  - `>= 1.0`: Log warning, return reference schedule unchanged
  - `(0.0, 1.0)`: Keep head portion, remove tail portion, then reconstruct

## Implementation Approach

### 1. New Parameter

Add `preserved_head_job_portion: float = 0.0` to `NehCpConstructor.run()` method signature after `ref_schedule` parameter.

### 2. Validation and Warning Logic

After timer and batch size processing, add validation:

```python
# Handle out-of-range values with warnings
if preserved_head_job_portion < 0.0:
    logging.warning(
        f"preserved_head_job_portion ({preserved_head_job_portion}) is negative; "
        "treating as 0.0 (full reconstruction mode)."
    )
    preserved_head_job_portion = 0.0
elif preserved_head_job_portion >= 1.0:
    if preserved_head_job_portion > 1.0:
        logging.warning(
            f"preserved_head_job_portion ({preserved_head_job_portion}) exceeds 1.0; "
            "returning reference schedule unchanged."
        )
    # Early return for >= 1.0
    logging.info("preserved_head_job_portion >= 1.0, returning reference schedule unchanged.")
    return NehCpResult(
        schedule=ref_schedule,
        sub_obj_store=sub_obj_store,
        last_obj_value=ref_schedule.makespan,
    )
```

### 3. Helper Function: `get_first_stage_start_sequence`

Add to `schedule_lite.py` (similar to `get_midpoint_sequence`):

```python
def get_first_stage_start_sequence(schedule: HybridFlowshopLiteSchedule) -> list[str]:
    """Get job sequence based on first stage start time.

    Returns:
        List of job IDs ordered by first stage start time, ties broken by original job order.
    """
    start_map = schedule.get_jik_2_start_time_map()
    jobs = schedule.jobs
    idx_map = {j: idx for idx, j in enumerate(jobs)}
    first_stage = schedule.stages[0]

    seq_info: list[tuple[int, int, str]] = []
    for j in jobs:
        s_first = next(
            t for (job, stage, _), t in start_map.items()
            if job == j and stage == first_stage
        )
        seq_info.append((s_first, idx_map[j], j))

    seq_info.sort(key=lambda x: (x[0], x[1]))
    return [info[2] for info in seq_info]
```

### 4. Job Sequence and Split Logic

After `sub_obj_store` setup and before state initialization:

```python
# Determine job sequence based on first stage start time for partial reconstruction
job_sequence: list[str]
if preserved_head_job_portion > 0.0:
    job_sequence = get_first_stage_start_sequence(ref_schedule)
elif job_seq_by_bottleneck_stage:
    job_sequence = get_bottleneck_stage_job_sequence(ref_schedule)
else:
    job_sequence = get_midpoint_sequence(ref_schedule)

job_cnt = len(job_sequence)

# Split into head (preserved) and tail (to reconstruct) jobs
if preserved_head_job_portion > 0.0:
    preserved_cnt = int(job_cnt * preserved_head_job_portion)
    head_jobs = set(job_sequence[:preserved_cnt])
    tail_jobs = job_sequence[preserved_cnt:]
    logging.info(
        f"Partial reconstruction: preserving {preserved_cnt} head jobs ({preserved_head_job_portion*100:.1f}%), "
        f"reconstructing {len(tail_jobs)} tail jobs."
    )
else:
    head_jobs = set()
    tail_jobs = job_sequence
```

### 5. Initialize Partial Solution from Head Jobs

After `self._st` initialization:

```python
st = self._require_state()

# Initialize partial solution from head jobs if in partial reconstruction mode
if head_jobs:
    st.partial_sol = ref_schedule.deepcopy(job_subsequence=head_jobs)
    st.partial_sol.make_semi_active(stage_2_job_2_p_dict)
    logging.info(
        f"Initialized partial solution from {len(head_jobs)} head jobs, "
        f"makespan = {st.partial_sol.makespan}"
    )
```

### 6. Modify Loop to Process Only Tail Jobs

Change `sequence_of_job_sublist` to use `tail_jobs`:

```python
sequence_of_job_sublist = [
    tail_jobs[i : i + _added_batch_size]
    for i in range(0, len(tail_jobs), _added_batch_size)
]
```

### 7. Full Solution Update Logic Adjustment

In the loop's full solution update section, use `tail_jobs` for remaining jobs:

```python
if not st.all_jobs_are_included(job_cnt):
    all_dispatched_sol_dj = st.partial_sol.deepcopy()
    remaining_jobs = [j for j in tail_jobs if j not in st.current_job_id_list]
    for j in remaining_jobs:
        all_dispatched_sol_dj.dispatch_job_by_stages(j, job_2_stage_2_p_dict[j])

    all_dispatched_sol_ds = st.partial_sol.deepcopy()
    for i in instance.stage_id_list:
        all_dispatched_sol_ds.dispatch_stage_by_jobs(i, remaining_jobs, stage_2_job_2_p_dict[i])

    st.full_sol = (
        all_dispatched_sol_dj
        if all_dispatched_sol_dj.makespan <= all_dispatched_sol_ds.makespan
        else all_dispatched_sol_ds
    )
else:
    st.full_sol = st.partial_sol
```

## Critical Files to Modify

1. `/home/hjt/code/hybridflowshop/hybridflowshop/controller/neh_cp.py` - Main implementation
2. `/home/hjt/code/hybridflowshop/hybridflowshop/schedule_lite.py` - Add `get_first_stage_start_sequence` helper

## Key Functions/Patterns to Reuse

- `get_midpoint_sequence()` / `get_bottleneck_stage_job_sequence()` in `schedule_lite.py` - Pattern for job sequence extraction
- `HybridFlowshopLiteSchedule.deepcopy(job_subsequence=...)` - For creating partial schedule with head jobs
- `HybridFlowshopLiteSchedule.make_semi_active()` - For retiming the partial schedule after removing tail jobs
- Existing batch processing loop structure in `run()` method

## Verification Steps

1. Run existing tests to ensure backward compatibility (default `preserved_head_job_portion=0.0`)
2. Test with `preserved_head_job_portion=1.0` - should return reference schedule unchanged with warning
3. Test with `preserved_head_job_portion=0.5` - should preserve head 50% jobs, reconstruct tail 50%
4. Test with `preserved_head_job_portion=1.5` - should log warning and return reference schedule unchanged
5. Test with `preserved_head_job_portion=-0.1` - should log warning and proceed with full reconstruction
6. Verify makespan and objective store are correctly tracked in partial reconstruction mode
7. Verify the head jobs maintain their relative ordering from the reference schedule
