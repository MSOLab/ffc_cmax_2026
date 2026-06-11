# ISW-CP panels (incremental sliding-window CP)

Demo instance: 10 jobs x 4 stages, 2 machines/stage (`resources/demo_p1_10x4/1.txt`).

- Incumbent (initialisation) makespan = 426.
- Window config: batch_size=1, step_size=2, U0=2, U_max=8, LPF=1, RPF=1.

## Region colors (5-region partition)
- LTF (left_time_fixed): `#9aa7b5`
- LPF (left_profile_fixed): `#7fb3d5`
- UNFIXED (unfixed): `#f39c12`
- RPF (right_profile_fixed): `#a3d977`
- RTF (right_time_fixed): `#bcaaa4`

## Panels
- `step_01_partition.svg`: 5-region partition at window position w=3 (U=2, LPF=1, RPF=1). Ops on each stage are time-ordered into batches; colors = LTF=#9aa7b5 | LPF=#7fb3d5 | UNFIXED=#f39c12 | RPF=#a3d977 | RTF=#bcaaa4. Red dashed lines bracket the UNFIXED window.
- `step_02a_right_justify_before.svg`: Right-justify (before): incumbent at w=3, makespan=426.
- `step_02b_right_justify_after.svg`: Right-justify (after): backward ALAP pass shifts the non-LTF ops (LPF/UNFIXED/RPF/RTF) as far right as possible, opening slack to the left of the UNFIXED window. Makespan is preserved (426); LTF anchors are untouched.
- `step_03a_window_solve_before.svg`: Window subproblem solve (before): right-justified incumbent at w=3, makespan=426. UNFIXED window is the CP search space.
- `step_03b_window_solve_after.svg`: Window subproblem solve (after): CP re-optimises the UNFIXED ops (LPF/RPF keep within-stage order & may shift; LTF/RTF stay fixed). Candidate makespan=426 (before=426).
- `step_04_slide_1_w3.svg`: Slide 1/3: window at w=3 (advances by Delta=step_size=2 each pass). The UNFIXED band and red boundaries move right while the partition structure is preserved.
- `step_04_slide_2_w5.svg`: Slide 2/3: window at w=5 (advances by Delta=step_size=2 each pass). The UNFIXED band and red boundaries move right while the partition structure is preserved.
- `step_04_slide_3_w7.svg`: Slide 3/3: window at w=7 (advances by Delta=step_size=2 each pass). The UNFIXED band and red boundaries move right while the partition structure is preserved.
- `step_05_enlarge_narrow_U2.svg`: Incremental enlargement (narrow): UNFIXED width U=2 (U0=2 -> U_max=8). When a pass stops improving, U grows so the next window re-optimises a larger block. The orange UNFIXED band widens between the panels.
- `step_05_enlarge_wide_U8.svg`: Incremental enlargement (wide): UNFIXED width U=8 (U0=2 -> U_max=8). When a pass stops improving, U grows so the next window re-optimises a larger block. The orange UNFIXED band widens between the panels.
