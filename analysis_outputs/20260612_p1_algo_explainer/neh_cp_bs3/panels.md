# NEH-CP panels (incremental NEH + CP re-optimisation)

Demo instance: 10 jobs x 4 stages, 2 machines/stage (`resources/demo_p1_10x4/1.txt`).

- Seed (initialisation) makespan = 426.
- added_batch_size = 3 -> 4 batches; insertion order = midpoint sequence.
- Shared axis: force_start=0, force_end=434.

Per-batch dispatch -> CP makespan:
- batch 1 (3 jobs): 187 -> 174
- batch 2 (6 jobs): 335 -> 308
- batch 3 (9 jobs): 415 -> 414
- batch 4 (10 jobs): 434 -> 434

## Panels
- `step_01_insertion_sequence.md`: Insertion priority sequence (midpoint) split into 4 batches of 3.
- `step_02_batch1_partial.svg`: Batch 1 inserted + CP: construction seed with 3 jobs (highlighted). Partial makespan=174.
- `step_03_batch2_dispatch.svg`: Batch 2 dispatched (before CP): new jobs (highlighted) appended via MixedDispatcher. Partial makespan=335.
- `step_04_batch2_cp.svg`: Batch 2 after CP: the sub-model re-optimises the UNFIXED ops (older ops keep profile-fixed precedence), compressing the partial schedule. Makespan 335 -> 308.
- `step_05_batch3_partial.svg`: Batch 3 after CP: partial now holds 9/10 jobs. New jobs highlighted. Partial makespan=414.
- `step_06_batch4_partial.svg`: Batch 4 after CP: partial now holds 10/10 jobs (all jobs inserted). New jobs highlighted. Partial makespan=434.
- `step_07_final_full.svg`: Final NEH-CP result: best feasible full schedule across batches (makespan=426; seed was 426).
