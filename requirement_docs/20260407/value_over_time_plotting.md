# Value over time plotting updates on 2026-04-07

This note summarizes the plotting behavior introduced by the most recent five
commits:

- `70ca7114` `feat(report): show best-so-far RPD progression`
- `c1b0d6e5` `refactor(chart): fix best-so-far progression logic`
- `425fde39` `fix(report): correct mean RPDf progression by (n,c)`
- `1d2b3728` `fix multi_scenario_subroutine_flow_comparison.html`
- `1b9f57fd` `add hover tooltops in mean regression plot`

The main effect is that three HTML graphs now use explicit value-over-time
progression semantics instead of endpoint-only diagonal line segments.

## Shared progression model

The plotting code now treats each instance as an explicit best-so-far
progression over normalized time:

$$
P_i = \bigl((t_{i,1}, r_{i,1}), (t_{i,2}, r_{i,2}), \ldots \bigr)
$$

where:

- $t_{i,j}$ is normalized time (`norm_time`)
- $r_{i,j}$ is best-so-far `RPDf` at that time
- times are sorted in increasing order
- duplicate times are deduplicated so one time keeps one effective value

The source of these points is the per-call `local_progress_list` inside
`subroutine_progression.json`. For one instance, the line is therefore based on
the full recorded progression, not only on subroutine endpoints.

Step-line rendering uses the standard rule:

- move horizontally to the next time point
- if the new best value improves, drop vertically at that same time

So all progression lines below are stair-step curves, not diagonal
interpolations.

## 1. `summary_method_rpdf_and_norm_time_scatter.html`

This scenario-level HTML keeps two modes:

- `instance progression`
- `mean progression by (job_cnt, stage_cnt)`

### 1.1 `instance progression`

The raw line for each instance uses the full per-instance progression from
`local_progress_list`, converted to best-so-far `RPDf`.

Subroutine endpoint markers are still shown, but marker placement is now
decoupled from line construction:

- marker `x` is the actual endpoint `norm_time`
- marker `y` is the line value at that time, i.e. the latest best-so-far value
  at or before that `x`

This guarantees that endpoint markers always lie on the plotted stair-step
curve, even if the endpoint itself is worse than an earlier best point.

### 1.2 `mean progression by (job_cnt, stage_cnt)`

This mode no longer averages endpoint rows by subroutine. It now builds an
over-time mean curve from the instance progressions in the selected
`(job_cnt, stage_cnt)` group.

For one group:

- first time point:
  the earliest time at which every instance has at least one available value,
  i.e. the maximum of the per-instance first times
- last time point:
  the maximum of the per-instance last times
- intermediate time points:
  the sorted union of all instance progression times between those bounds

At a time $t$, the plotted mean value is:

$$
\bar r(t)=\frac{1}{|G|}\sum_{I_i \in G} r_i^\ast(t)
$$

where $r_i^\ast(t)$ is the latest best-so-far `RPDf` recorded at or before
$t$. If an instance finished earlier than others, its last value is carried
forward for later averaging.

### 1.3 Mean subroutine guide markers

For `mean progression by (job_cnt, stage_cnt)`, subroutine endpoints are shown
as average-time guides instead of points on the mean line.

For each subroutine name:

- compute the average endpoint `norm_time` across instances in the group
- draw a vertical dotted line at that average time
- draw a marker at the x-axis contact point of that dotted line

The guide line and its x-axis marker share the same color as the corresponding
mean curve.

Hover behavior:

- mean line hover is enabled and shows series and `(job_cnt, stage_cnt)` info
- x-axis guide marker hover is enabled and shows the subroutine name and average
  endpoint time
- raw line hover remains disabled; raw endpoint marker hover remains enabled

## 2. `multi_scenario_subroutine_flow_comparison.html`

This top-level comparison HTML no longer plots one endpoint scatter line per
scenario. It now plots one over-time mean progression curve per scenario.

For each scenario:

- build per-instance best-so-far progressions from `subroutine_progression.json`
- aggregate them over time with the same rule used in
  `mean progression by (job_cnt, stage_cnt)`
- render one stair-step mean `RPDf` curve for that scenario

So this chart is effectively a "mean of means over time" comparison at the
scenario level:

- within a scenario: average over instances
- across scenarios: compare the resulting scenario-level mean curves

Subroutine average endpoint times are also shown for each scenario as:

- vertical dotted guide lines
- x-axis contact markers

Hover behavior:

- scenario mean line hover is enabled
- x-axis guide marker hover is enabled and shows scenario name, subroutine
  name, and average endpoint time

## 3. Source selection and fallback

All three graphs prefer progression JSON as the primary source whenever a valid
value-over-time reconstruction is possible.

### Scenario-level chart

`summary_method_rpdf_and_norm_time_scatter.html` uses:

- endpoint rows reconstructed from `subroutine_progression.json` for endpoint
  markers
- raw progression rows from `subroutine_progression.json` for the raw and mean
  stair-step lines

If valid JSON endpoint metrics are not available, the chart falls back to the
existing endpoint CSV path:

- `summary_method_rpdf_and_norm_time_long.csv`

### Top-level multi-scenario chart

`multi_scenario_subroutine_flow_comparison.html` uses scenario-level JSON
progression whenever possible. If JSON progression cannot be used for a
scenario, it falls back to endpoint CSV input for that scenario and emits a
warning log identifying the scenario.

The selected-scenario comparison script follows the same exporter shape, but in
its current form it loads endpoint CSV directly and logs a warning that the
comparison is using CSV fallback.

## 4. Current practical interpretation

After these five commits, the three graph families should be interpreted as
follows:

- scenario raw chart:
  full per-instance best-so-far progression, with endpoint markers snapped onto
  the line
- scenario grouped-mean chart:
  over-time mean of instance progressions for each `(job_cnt, stage_cnt)` group
- top-level multi-scenario chart:
  over-time mean of instance progressions for each scenario, compared across
  scenarios

In all cases, the visual line represents a best-so-far value-over-time
trajectory rendered as a stair-step curve.
