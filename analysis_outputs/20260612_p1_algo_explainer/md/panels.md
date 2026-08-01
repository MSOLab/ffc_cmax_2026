# MD panels

- `step_01_priority_sequences.md`: Priority sequences P/G/CDS -> pick pi
- `step_02_pure_A_job_then_stage.svg`: Pure mode A: job-then-stage (n_p = n = 8). Each job is pushed through all stages first -> staircase. makespan=385.
- `step_03_pure_B_stage_then_job.svg`: Pure mode B: stage-then-job (n_p = 0). Stages are filled 1->c by priority order. makespan=378.
- `step_04_mixed_np_half.svg`: Mixed: n_p = ceil(n/2) = 4. Head (highlighted) = first 4 jobs job-then-stage; tail = stage-then-job. makespan=367.
- `step_05_final_selection.svg`: Final MD selection: min makespan over the n_p ladder for pi (chosen n_p = 4). makespan=367.
