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
`results/subroutine_progression.json` file. This file stores the raw
progression trace of objective values over top-level subalgorithm calls in one
run of $A_a$ on $I_i$.

The file records call boundaries and objective-value points. It does not store
aggregated statistics such as mean curves or RPDf summaries; those are derived
later during post-processing.

### Top-level fields

| JSON field | Meaning |
| ---------- | ------- |
| `artifact_version` | schema version of the JSON artifact |
| `instance_id` | identifier of $I_i$ |
| `timelimit_sec` | $T_i$ |
| `subroutine_calls` | ordered list of recorded top-level calls $C_k$ |
| `combined_progress_list` | flattened list of all recorded points $(g_{i,a,k,\ell}, \tau_{i,a,k,\ell}, v_{i,a,k,\ell})$ across all calls |
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
| `local_progress_list` | ordered list of recorded objective points inside $C_k$ |

### `local_progress_list`

Each element of `local_progress_list` is one recorded objective-value point
inside a fixed call $C_k$.

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

`combined_progress_list` is the flattened union of all point records from all
`local_progress_list` entries. Each element keeps the same point-level fields as
above, but the list is organized as one global trace over the whole run on
$I_i$.

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
- The JSON stores raw times and raw objective values; normalized time, local
  ratio, RPDf, mean curves, and improvement curves are post-processed from this
  raw data.
