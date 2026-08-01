# BN2D step_01 -- bottleneck stage + r_j / tr_j

Global (machine-normalized load) bottleneck = `i3` (the terminal stage on this demo -> no later stages).
Storyboard bottleneck i* = `i2` (strongest non-terminal candidate; same code path, shows both -> and <- propagation).

Per-job workloads around i* = `i2` (former stages ['i0', 'i1'], later stages ['i3']):

  - j0: r= 68  tr= 98  [mid]
  - j1: r= 33  tr= 39  [L]
  - j2: r= 77  tr= 95  [mid]
  - j3: r= 92  tr= 92  [mid]
  - j4: r=114  tr= 56  [mid]
  - j5: r=133  tr= 51  [R]
  - j6: r=155  tr= 76  [mid]
  - j7: r= 13  tr= 54  [L]
  - j8: r=100  tr= 64  [mid]
  - j9: r=138  tr= 50  [R]

r_j  = sum of processing times in former stages (forward workload)
tr_j = sum of processing times in later stages (rear workload)
