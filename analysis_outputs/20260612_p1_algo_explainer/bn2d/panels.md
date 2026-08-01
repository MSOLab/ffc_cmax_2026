# BN2D panels

- `step_01_bottleneck_r_tr.md`: Bottleneck stage i*=i2 highlighted + per-job r_j / tr_j (global argmax is terminal i3; i2 used to show both directions)
- `step_02_cap_selection.svg`: Cap selection at bottleneck i2: L (small r_j) = ['j7', 'j1'], R (small tr_j contribution) = ['j5', 'j9']; dispatch order L || mid || R = ['j7', 'j1'] || ['j0', 'j2', 'j3', 'j8', 'j4', 'j6'] || ['j5', 'j9']. Highlighted bars = L+R caps. (caps per side = 1 x 2 machines.)
- `step_03_bottleneck_pms.svg`: Bottleneck dispatch: stage i2 scheduled as a single-stage parallel-machine problem in order L || mid || R (release times r_j). L+R caps highlighted. Rendered restricted to stage i2.
- `step_04_right_propagation.svg`: Right propagation (->): later stages ['i3'] dispatched forward from i* end times (MD: best of stage-then-job / job-then-stage). Bottleneck stage i2 band highlighted.
- `step_05_left_propagation_final.svg`: Left propagation (<-): former stages ['i0', 'i1'] scheduled on a reversed-time sub-instance (release = bcmax - start at i*), then mapped back. FINAL full schedule, makespan = 465. Bottleneck band highlighted.
