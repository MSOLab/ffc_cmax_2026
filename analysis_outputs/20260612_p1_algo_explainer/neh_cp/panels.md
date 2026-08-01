# NEH-CP panels (incremental NEH + CP re-optimisation)

Demo instance: 10 jobs x 4 stages, 2 machines/stage (`resources/demo_p1_10x4/1.txt`).

- Seed (initialisation) makespan = 426.
- added_batch_size = 2 -> 5 batches; insertion order = midpoint sequence.
- Shared axis: force_start=0, force_end=427.

Per-batch dispatch -> CP makespan:
- batch 1 (2 jobs): 96 -> 96
- batch 2 (4 jobs): 244 -> 234
- batch 3 (6 jobs): 327 -> 321
- batch 4 (8 jobs): 377 -> 377
- batch 5 (10 jobs): 427 -> 427

## Panels
- `step_01_insertion_sequence.md`: Insertion priority sequence (midpoint) split into 5 batches of 2.
- `step_02_batch1_partial.svg`: Batch 1 inserted + CP: construction seed with 2 jobs (highlighted). Partial makespan=96.
- `step_03_batch2_dispatch.svg`: Batch 2 dispatched (before CP): new jobs (highlighted) appended via MixedDispatcher. Partial makespan=244.
- `step_04_batch2_cp.svg`: Batch 2 after CP: the sub-model re-optimises the UNFIXED ops (older ops keep profile-fixed precedence), compressing the partial schedule. Makespan 244 -> 234.
- `step_05_batch3_partial.svg`: Batch 3 after CP: partial now holds 6/10 jobs. New jobs highlighted. Partial makespan=321.
- `step_06_batch4_partial.svg`: Batch 4 after CP: partial now holds 8/10 jobs. New jobs highlighted. Partial makespan=377.
- `step_07_batch5_partial.svg`: Batch 5 after CP: partial now holds 10/10 jobs (all jobs inserted). New jobs highlighted. Partial makespan=427.
- `step_08_final_full.svg`: Final NEH-CP result: best feasible full schedule across batches (makespan=426; seed was 426).
