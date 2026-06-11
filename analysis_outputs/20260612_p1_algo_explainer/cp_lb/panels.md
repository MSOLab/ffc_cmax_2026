# CP-LB panels (retained-stage relaxation lower bound)

Demo instance: 10 jobs x 4 stages, 2 machines/stage (`resources/demo_p1_10x4/1.txt`).

- Analytic SHD input LB = 384; incumbent UB = 426.
- Quantile mode `first_middle_last`: R^quant = ['i0', 'i2', 'i3'], LB = 419 (OPTIMAL).
- Adaptive mode `first_bottleneck_last`: R^adap = ['i0', 'i2', 'i3'], internal bottleneck = i2, LB = 419 (OPTIMAL).
- Restored full-schedule makespan = 425; optimality gap to LB = 6.

## Known limitations (faithful approximations)
- R^quant == R^adap == ['i0', 'i2', 'i3'] for this 4-stage demo: the quantile-middle stage and the strongest internal workload bottleneck both resolve to i2 (the true workload bottleneck i3 is the always-retained last stage). Both selector outputs are real and rendered separately; they simply coincide on this instance.
- The relaxation Gantt (step_03) ignores per-stage machine capacity; lane (y) placement is synthetic greedy interval packing so overlaps are visible. Bar times are the exact CP solution and the makespan equals the certified LB.
- Dropped-stage greying is approximated by highlighting the retained ops; captions name the dropped stages and their compressed-lag role.

## Panels
- `step_01_retained_subset_quantile.svg`: Quantile mode (first_middle_last): retained R^quant = ['i0', 'i2', 'i3'] (first + quantile-middle + last). Dropped stages ['i1'] collapse to compressed lags between retained stages. Highlighted = retained-stage ops on the incumbent (makespan=426).
- `step_02_retained_subset_adaptive.svg`: Workload-adaptive mode (first_bottleneck_last): retained R^adap = ['i0', 'i2', 'i3']; internal bottleneck = i2 (strongest sum(p)/m among non-terminal stages). Dropped stages ['i1'] -> compressed lags. NOTE: for this 4-stage demo R^adap == R^quant because the quantile-middle stage and the internal bottleneck both resolve to i2.
- `step_03_relaxation_lower_bound.svg`: Retained-stage CP relaxation (stages ['i0', 'i2', 'i3'] only; dropped stages capacity-ignored and collapsed to lags). The relaxation makespan = 419 is the CERTIFIED lower bound LB = 419 (CP-SAT status OPTIMAL). Lanes are synthetic interval-packing (capacity ignored: overlaps -> extra lanes beyond 2 machines).
- `step_04_hint_based_dispatch_full.svg`: Hint-based dispatch: retained-CP anchors restore a full feasible 4-stage schedule, makespan = 425. Lower bound LB = 419 vs restored makespan 425 -> optimality gap = 6. (Analytic SHD LB input was 384; retained-CP LB refines it to 419.)
