# CP-LB panels (retained-stage relaxation lower bound)

Demo instance: 10 jobs x 15 stages (`resources/ff2020small/3.txt`).

- Analytic SHD input LB = 959; incumbent UB = 992.
- Quantile mode `first_n_quantiles_last`: R^quant = ['i00', 'i04', 'i07', 'i10', 'i14'], LB = 959 (OPTIMAL).
- Adaptive mode `first_topk_bottlenecks_last`: R^adap = ['i00', 'i02', 'i03', 'i08', 'i14'], top bottleneck = i02, LB = 959 (OPTIMAL).
- Restored full-schedule makespan = 992; optimality gap to LB = 33.

## Known limitations (faithful approximations)
- R^quant = ['i00', 'i04', 'i07', 'i10', 'i14'] and R^adap = ['i00', 'i02', 'i03', 'i08', 'i14'] differ on this instance: the quantile rule spreads cuts evenly over the stage axis while the adaptive rule pulls retention toward the incumbent's workload bottlenecks.
- The relaxation Gantt (step_03) ignores per-stage machine capacity; lane (y) placement is synthetic greedy interval packing so overlaps are visible. Bar times are the exact CP solution and the makespan equals the certified LB.
- Dropped-stage greying is approximated by highlighting the retained ops; captions name the dropped stages and their compressed-lag role.

## Panels
- `step_01_retained_subset_quantile.svg`: Quantile mode (first_n_quantiles_last): retained R^quant = ['i00', 'i04', 'i07', 'i10', 'i14'] (first + quantile-cut stages + last). Dropped stages ['i01', 'i02', 'i03', 'i05', 'i06', 'i08', 'i09', 'i11', 'i12', 'i13'] collapse to compressed lags between retained stages. Highlighted = retained-stage ops on the incumbent (makespan=992).
- `step_02_retained_subset_adaptive.svg`: Workload-adaptive mode (first_topk_bottlenecks_last): retained R^adap = ['i00', 'i02', 'i03', 'i08', 'i14']; top bottleneck = i02 (strongest sum(p)/m among non-terminal stages). Dropped stages ['i01', 'i04', 'i05', 'i06', 'i07', 'i09', 'i10', 'i11', 'i12', 'i13'] -> compressed lags. Here R^adap differs from R^quant (['i00', 'i04', 'i07', 'i10', 'i14']).
- `step_03_relaxation_lower_bound.svg`: Retained-stage CP relaxation (stages ['i00', 'i02', 'i03', 'i08', 'i14'] only; dropped stages capacity-ignored and collapsed to lags). The relaxation makespan = 959 is the CERTIFIED lower bound LB = 959 (CP-SAT status OPTIMAL). Lanes are synthetic interval-packing (capacity ignored: overlaps -> extra lanes beyond 2 machines).
- `step_04_hint_based_dispatch_full.svg`: Hint-based dispatch: retained-CP anchors restore a full feasible 4-stage schedule, makespan = 992. Lower bound LB = 959 vs restored makespan 992 -> optimality gap = 33. (Analytic SHD LB input was 959; retained-CP LB refines it to 959.)
- `step_05_compressed_lag_schematic.svg`: Compressed-lag schematic (job j2): top strip = the full job with every stage a real operation; bottom strip = the retained relaxation where each maximal run of dropped stages collapses to a SINGLE hatched compressed lag ℓ (= summed dropped-stage processing time) between consecutive retained stages ['i00', 'i02', 'i03', 'i08', 'i14']. 3 lag(s) for 10 dropped stage(s). Shared x-axis: each lag sits under the dropped run it replaces.
