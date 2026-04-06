# Value over time plotting

## Notation

| Notation | Meaning |
| ------ | -------- |
| $I_i$ | $i$-th instance |
| $A_a$ | $a$-th algorithm |
| $S_s$ | $s$-th subalgorithm |
| $k$ | call index of a top-level recorded call within one run |
| $C_k$ | the $k$-th recorded top-level call in one run |
| $t$ | time (or `time / time limit` for the instance) |
| $v$ | value (or statistics such as RPDf) |
| $T_i$ | time limit for instance $I_i$ |
| $name(C_k)$ | subalgorithm type of call $C_k$, so $name(C_k)=S_s$ for some $s$ |
| $t^{\mathrm{start}}_{i,a,k}$ | global elapsed time at which $C_k$ starts while solving $I_i$ with $A_a$ |
| $t^{\mathrm{end}}_{i,a,k}$ | global elapsed time at which $C_k$ ends |
| $\Delta t_{i,a,k}$ | elapsed time of $C_k$, i.e. $t^{\mathrm{end}}_{i,a,k} - t^{\mathrm{start}}_{i,a,k}$ |
| $g_{i,a,k,\ell}$ | global elapsed time of the $\ell$-th recorded objective point inside $C_k$ |
| $\tau_{i,a,k,\ell}$ | local elapsed time of the same point, i.e. $g_{i,a,k,\ell} - t^{\mathrm{start}}_{i,a,k}$ |
| $v_{i,a,k,\ell}$ | objective value recorded at the $\ell$-th point inside $C_k$ |

## Per-instance progression JSON

For each instance $I_i$, the current implementation writes one
`results/subroutine_progression.json` file. This file stores a compact
progression trace of objective values over top-level subalgorithm calls in one
run of $A_a$ on $I_i$.

The file records call boundaries and per-call incumbent points. It does not
store aggregated statistics such as mean curves or RPDf summaries; those are
derived later during post-processing.

### Top-level fields

| JSON field | Meaning |
| ---------- | ------- |
| `artifact_version` | schema version of the JSON artifact |
| `instance_id` | identifier of $I_i$ |
| `timelimit_sec` | $T_i$ |
| `subroutine_calls` | ordered list of recorded top-level calls $C_k$ |
| `combined_progress_list` | flattened list of all recorded compact trace points $(g_{i,a,k,\ell}, \tau_{i,a,k,\ell}, v_{i,a,k,\ell})$ across all calls |
| `subroutine_end_marker_list` | ordered list of end markers for recorded calls, mainly storing $t^{\mathrm{end}}_{i,a,k}$ |

### `subroutine_calls`

Each element of `subroutine_calls` corresponds to one recorded top-level call
$C_k$.

| JSON field | Meaning |
| ---------- | ------- |
| `call_index` | $k$ |
| `subroutine_name` | $name(C_k)=S_s$ |
| `prefixed_subroutine_name` | serialized unique label for $C_k$, currently `"{k}-{subroutine_name}"` |
| `global_start_sec` | $t^{\mathrm{start}}_{i,a,k}$ |
| `global_end_sec` | $t^{\mathrm{end}}_{i,a,k}$ |
| `elapsed_sec` | $\Delta t_{i,a,k}$ |
| `local_progress_list` | ordered list of the compact incumbent trace inside $C_k$ |

### `local_progress_list`

Each element of `local_progress_list` is one recorded objective-value point
inside a fixed call $C_k$. The recorder keeps the first point observed for the
call and later points only when they are a strict improvement over the last
recorded point for that same call.

| JSON field | Meaning |
| ---------- | ------- |
| `local_sec` | $\tau_{i,a,k,\ell}$ |
| `global_sec` | $g_{i,a,k,\ell}$ |
| `obj_value` | $v_{i,a,k,\ell}$ |
| `call_index` | repeated copy of $k$ for easier flat processing |
| `prefixed_subroutine_name` | repeated copy of the unique call label for easier flat processing |

By construction,

$$
\tau_{i,a,k,\ell} = g_{i,a,k,\ell} - t^{\mathrm{start}}_{i,a,k}.
$$

### `combined_progress_list`

`combined_progress_list` is the flattened union of all compact point records
from all `local_progress_list` entries. Each element keeps the same point-level
fields as above, but the list is organized as one global trace over the whole
run on $I_i$.

This duplication is intentional: downstream analysis can either iterate through
per-call nested traces (`subroutine_calls`) or consume one already-flattened
point table (`combined_progress_list`).

### `subroutine_end_marker_list`

Each element of `subroutine_end_marker_list` stores the end boundary of one
recorded call $C_k$.

| JSON field | Meaning |
| ---------- | ------- |
| `global_end_sec` | $t^{\mathrm{end}}_{i,a,k}$ |
| `call_index` | $k$ |
| `prefixed_subroutine_name` | unique serialized label of $C_k$ |
| `subroutine_name` | $name(C_k)=S_s$ |

## Scope of what is recorded

- The JSON is indexed by instance: one artifact per $I_i$.
- The recorder tracks top-level subalgorithm calls executed by the controller
  flow.
- Objective updates produced while a top-level call $C_k$ is active are
  attached to that active call.
- Within a call, the stored trace is compacted to the first point and later
  strict improvements only.
- The JSON stores raw times and raw objective values for the compact trace;
  normalized time, local ratio, RPDf, mean curves, and improvement curves are
  post-processed from this data.

## JSON endpoint metrics for scatter HTML

`summary_method_rpdf_and_norm_time_scatter.html` keeps the existing endpoint
semantics. It does not plot the full compact progression stored in
`subroutine_progression.json`. Instead, it reconstructs one endpoint metric row
per flow-position subalgorithm and then renders the same two views as before:

- instance progression
- mean progression grouped by `(job_cnt, stage_cnt)`

### Endpoint reconstruction rule

Let the scenario-level top-level flow be

$$
(S_{s_1}, S_{s_2}, \ldots, S_{s_m}).
$$

For one instance $I_i$, the JSON stores the executed recorded calls

$$
(C_1, C_2, \ldots, C_q)
$$

in `subroutine_calls`, sorted by `call_index`.

The scatter endpoint builder scans the configured flow left-to-right and aligns
the next unmatched recorded call by exact `subroutine_name`. For each matched
call $C_k$, it creates one endpoint row for the corresponding flow position.

### Endpoint fields used for scatter metrics

For a matched call $C_k$:

- `end_time` is `global_end_sec = t^{\mathrm{end}}_{i,a,k}`
- `obj_value` is the last recorded `obj_value` in `local_progress_list`

If `local_progress_list` is empty, the builder uses the previous effective
objective value carried from the most recent earlier matched call.

### Carry-forward and trailing fill

The scatter reconstruction follows the same practical rule as the existing
CSV/log-based endpoint summary:

- if a matched call has no local point, its endpoint objective is the previous
  effective objective value
- if `record_all_subroutines=True`, every trailing unexecuted flow item after
  the last executed call is filled with the last executed `end_time` and the
  last effective objective value

This means that the scatter HTML can still show late flow positions that did
not execute but should inherit the last known endpoint under the legacy
summary-generation rule.

### Omitted subroutines

`omitted_subroutines` are removed only after endpoint reconstruction. They still
participate in call alignment and in effective-objective carry-forward before
their final rows are dropped from the scatter metrics.

### Source selection policy

The scenario-level scatter HTML uses the following source selection rule:

- first try to reconstruct endpoint metrics from per-instance
  `subroutine_progression.json`
- if JSON endpoint metrics are unavailable or cannot be aligned safely, fall
  back to the existing `summary_method_rpdf_and_norm_time_long.csv` path

The same JSON endpoint reconstruction helper is also used for the top-level
multi-scenario comparison HTML so both scatter views share the same endpoint
semantics.
