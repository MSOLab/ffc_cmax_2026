# Post-MIP Dispatch Rules

This note documents the dispatch candidates that are evaluated after
`apply_mip_lb()` solves the bucket-indexed MIP and extracts an incumbent
dispatch-window solution.

The main entry point is:

- `lb_bucket/mip/post_dispatch.py`

The main helper implementations live in:

- `hybridflowshop/dispatcher/utils.py`
- `hybridflowshop/controller/hfs_cp_lns.py`

## 1. Big Picture

After `apply_mip_lb()` finishes, we do **not** directly take the MIP incumbent as a
final machine-level schedule. Instead, we do this:

1. solve the bucket-indexed MIP,
2. convert the incumbent into per-operation dispatch-window data,
3. derive several ES/LS-based order signals from that data,
4. build multiple dispatch candidates,
5. compare them by makespan,
6. optionally repair the current best one,
7. keep the final best schedule.

So the MIP incumbent is used mainly as a **signal generator**:

- per-stage order hints,
- per-job release hints,
- per-job latest-start hints,
- global urgency/ranking hints.

## 2. What Exactly Comes Out of the MIP

The post-MIP dispatcher works from `dispatch_windows`, which are derived from the
saved incumbent variables `a`, `b`, and `x`.

For each operation `(stage s, job j)`, the window record contains values such as:

- `early_start`
  - the incumbent-based ES time for that operation
- `late_start`
  - the incumbent-based LS time for that operation
- `slack = late_start - early_start`
  - the width of the allowed start window implied by the incumbent
- `processing_time`
  - processing time of the operation
- `a_bucket`, `b_bucket`
  - the first and last bucket touched by the operation in the bucket model
- `x_bucket_*`, `x_value_*`
  - how much of that operation occupies each touched bucket

Important:

- these ES/LS values are **derived from the incumbent**, not immutable truths,
- they are useful signals, but they are not guaranteed to be the best final
  start times for the real dispatch schedule,
- that is exactly why some good dispatches start operations **before ES** or
  **after LS**.

## 3. Glossary: Terms Used in This File

### 3.1 ES, LS, Slack

For one operation `(s, j)`:

- `ES(s,j) = early_start`
- `LS(s,j) = late_start`
- `Slack(s,j) = LS(s,j) - ES(s,j)`

Interpretation:

- small `ES` means “the incumbent thinks this operation wants to happen early”
- small `LS` means “the incumbent thinks delaying this operation is dangerous”
- small `Slack` means “the incumbent gives this operation little timing freedom”

### 3.2 Stage Job Sequence

A **stage job sequence** is a separate ordered job list for each stage:

- stage 1: `[j?, j?, ...]`
- stage 2: `[j?, j?, ...]`
- ...

This is produced by:

- `get_stage_job_sequences_from_dispatch_windows()`

and is used by the stage-sequence family of dispatchers.

### 3.3 Global Job Sequence

A **global job sequence** is one single job order for the whole problem:

- `[j?, j?, j?, ...]`

This is produced either from:

- one anchor stage, or
- an aggregate score over all stages.

This is used by the direct mixed family.

### 3.4 Global Tie-Break Rank

A **job tie-break rank** is a dictionary:

- `job_id -> integer rank`

where smaller rank means “prefer this job earlier when the heuristic needs a
tie-break”.

This is used by the best-of-mixed family.

Important:

- in the best-of-mixed family, the rank is usually a **secondary signal**,
- it does not replace CDS/Gupta/Palmer itself,
- it only changes how ties or near-ties are resolved inside those heuristics.

### 3.5 Release Map

The release map is:

- `stage_2_job_release[stage_id][job_id] = int(ES(stage, job))`

This is produced by:

- `get_stage_job_release_times_from_dispatch_windows()`

and is passed to stage-sequence builders as a lower bound on start time.

### 3.6 Latest-Start Map

The latest-start map is:

- `stage_2_job_latest_start[stage_id][job_id] = int(LS(stage, job))`

This is produced by:

- `get_stage_job_latest_start_times_from_dispatch_windows()`

and is currently used by:

- `es_ls_stage_strict_lexicographic_release`

### 3.7 Anchor Stage

An **anchor stage** means:

- “pick one stage, look only at that stage's ES/LS window information, and use
  that stage to define a global job order.”

Two important anchor-stage styles are used here:

- tail-stage anchor
  - use the last stage only
- bottleneck-stage anchor
  - use one bottleneck-like stage only

### 3.8 Bottleneck Anchor Stage

The bottleneck anchor stage is chosen by:

- `get_bottleneck_anchor_stage_from_solution_payload()`

Primary signal:

- maximum stage-bucket congestion from incumbent `x` values
- formula:
  - `max_t [ sum_j x[s,j,t] / (m_s * delta) ]`

where:

- `m_s` = number of machines at stage `s`
- `delta` = bucket width

Fallback signal when payload information is weak/missing:

- average load proxy:
  - `sum_j p[s,j] / m_s`

Interpretation:

- we choose a stage that appears highly congested in the incumbent,
- then use that stage's ES/LS structure as a strong ordering signal.

## 4. Exact Sort Keys Used to Build MIP-Derived Orders

This section answers questions like:

- “What does aggregate ES-slack information mean exactly?”
- “What does bottleneck slack rank mean exactly?”

### 4.1 Per-Stage Sort Rules

These are implemented in `_get_dispatch_window_sort_key()`.

For one operation:

- `ES = early_start`
- `LS = late_start`
- `Slack = LS - ES`
- `Midpoint = ES + LS`
- `p = processing_time`

The built-in rules are:

| Rule name | Exact sort key |
|---|---|
| `es_ls_p_desc` | `(ES, LS, -p, job_idx)` |
| `ls_es_p_desc` | `(LS, ES, -p, job_idx)` |
| `slack_ls_es_p_desc` | `(Slack, LS, ES, -p, job_idx)` |
| `es_slack_ls_p_desc` | `(ES, Slack, LS, -p, job_idx)` |
| `midpoint_slack_ls_p_desc` | `(Midpoint, Slack, LS, -p, job_idx)` |

Meaning:

- sort ascending on the tuple,
- so smaller earlier fields are more urgent,
- `-p` means larger `p` is preferred when earlier fields tie.

### 4.2 Aggregate Sort Rules Across Stages

These are implemented in `_get_dispatch_window_aggregate_key()`.

For one job `j`, define:

- `ES_k(j)` = ES of job `j` on stage `k`
- `LS_k(j)` = LS of job `j` on stage `k`
- `Slack_k(j) = LS_k(j) - ES_k(j)`
- `TotalP(j) = sum_k p_k(j)`

Then the aggregate rules are:

| Aggregation rule | Exact sort key |
|---|---|
| `sum_es_slack_p_desc` | `(sum_k ES_k, sum_k Slack_k, sum_k LS_k, -TotalP, job_idx)` |
| `sum_ls_slack_p_desc` | `(sum_k LS_k, sum_k Slack_k, sum_k ES_k, -TotalP, job_idx)` |
| `tail_ls_sum_slack_p_desc` | `(LS_last, sum_k Slack_k, ES_last, -TotalP, job_idx)` |

### 4.3 What “Aggregate ES-Slack Information” Means

This phrase does **not** mean one special variable stored by the MIP.

It means:

- for each job, collect its ES/LS windows over **all** stages,
- compute:
  - `sum_k ES_k`
  - `sum_k Slack_k`
  - `sum_k LS_k`
  - `TotalP`
- sort jobs by:
  - `(sum ES, sum Slack, sum LS, -TotalP, job_idx)`

So in plain language:

- jobs whose windows are globally earlier across the whole routing come first,
- if two jobs are similarly early, jobs with tighter total slack come first,
- then jobs with earlier total LS come first,
- then longer total processing jobs come first.

That is what the documentation means by:

- `aggregate ES-slack information`

### 4.4 What “Aggregate LS-Slack Information” Means

Similarly, this means:

- collect each job's ES/LS windows over all stages,
- sort by:
  - `(sum LS, sum Slack, sum ES, -TotalP, job_idx)`

So in plain language:

- jobs whose latest-start limits are globally earlier come first,
- then tighter total slack,
- then earlier total ES,
- then longer total processing.

### 4.5 Small Worked Example

Suppose one job has:

- stage 1: `ES=10`, `LS=20`, `Slack=10`
- stage 2: `ES=30`, `LS=45`, `Slack=15`
- stage 3: `ES=55`, `LS=70`, `Slack=15`

Then:

- `sum ES = 10 + 30 + 55 = 95`
- `sum LS = 20 + 45 + 70 = 135`
- `sum Slack = 10 + 15 + 15 = 40`

So its aggregate keys are:

- aggregate ES-slack key:
  - `(95, 40, 135, -TotalP, job_idx)`
- aggregate LS-slack key:
  - `(135, 40, 95, -TotalP, job_idx)`

When jobs are compared, the smaller tuple wins.

## 5. How Each MIP-Derived Sequence/Rank Is Built

This section maps variant names to the exact underlying signal.

### 5.1 Tail LS Sequence

Source:

- `get_job_sequence_from_dispatch_windows_anchor_stage(..., anchor_stage_id=last_stage, sort_rule="ls_es_p_desc")`

Exact key for job `j` on the last stage:

- `(LS_last(j), ES_last(j), -p_last(j), job_idx)`

Interpretation:

- prioritize jobs whose **last-stage** latest start is earliest.

### 5.2 Bottleneck Slack Sequence

Source:

- `get_job_sequence_from_dispatch_windows_anchor_stage(..., anchor_stage_id=bottleneck_anchor_stage, sort_rule="slack_ls_es_p_desc")`

Exact key for job `j` on the chosen bottleneck stage:

- `(Slack_bneck(j), LS_bneck(j), ES_bneck(j), -p_bneck(j), job_idx)`

Interpretation:

- prioritize jobs that are tight on the bottleneck stage.

### 5.3 Aggregate ES Slack Sequence

Source:

- `get_job_sequence_from_dispatch_windows_aggregate(..., aggregation_rule="sum_es_slack_p_desc")`

Exact key:

- `(sum ES, sum Slack, sum LS, -TotalP, job_idx)`

Interpretation:

- prioritize jobs that look globally early and globally tight.

### 5.4 Aggregate LS Slack Sequence

Source:

- `get_job_sequence_from_dispatch_windows_aggregate(..., aggregation_rule="sum_ls_slack_p_desc")`

Exact key:

- `(sum LS, sum Slack, sum ES, -TotalP, job_idx)`

Interpretation:

- prioritize jobs that look globally late-sensitive and globally tight.

### 5.5 Weighted Blended Ranks

These are created in `_build_weighted_job_tiebreak_rank()`.

Mechanics:

1. build multiple sequence-specific ranks:
   - example: `bottleneck rank`, `aggregate ES rank`
2. for each job, read its position in each sequence
3. compute a weighted sum of those positions
4. sort by:
   - `(weighted_sum, min(component_ranks), component_ranks..., original_job_order)`
5. convert that sorted job list into `job_id -> rank`

So the new blended variants mean:

| Variant | Exact blended score |
|---|---|
| `best_of_mixed_dispatches_bottleneck_aggregate_es_rank` | `2 * rank_in_bottleneck_slack_sequence + 1 * rank_in_aggregate_es_sequence` |
| `best_of_mixed_dispatches_bottleneck_aggregate_ls_rank` | `2 * rank_in_bottleneck_slack_sequence + 1 * rank_in_aggregate_ls_sequence` |

Interpretation:

- “trust bottleneck urgency more strongly, but soften it with a global aggregate signal.”

## 6. Candidate Families Evaluated After Bucket MIP

## 6.1 Stage-Sequence Dispatch Family

These candidates use the per-stage job sequence extracted from the MIP windows.

That stage sequence is currently built by:

- `get_stage_job_sequences_from_dispatch_windows(...)`

with the default rule:

- `es_ls_p_desc`
- exact key:
  - `(ES, LS, -p, job_idx)`

So the raw stage order itself means:

- earlier ES first,
- then earlier LS,
- then longer processing first.

Variants:

| Variant | Overlay alias | Exact behavior |
|---|---|---|
| `es_ls_stage_priority_release` | `esls_priority` | Build a release-aware priority queue per stage. The stage sequence is only a priority/tie-break signal. |
| `es_ls_stage_strict_call_release` | `esls_strict_call` | Call jobs in exactly the stage-sequence order, with `ES` as release lower bound. |
| `es_ls_stage_strict_lexicographic_release` | `esls_strict_lex` | Recompute each unscheduled job's current feasible start, then sort by `current feasible time -> ES -> LS -> original stage order`. |
| `es_ls_stage_strict_start_release` | `esls_strict_start` | Try to preserve the given stage order as realized start order, again with `ES` release. |

Important:

- all of these are **stage-local** interpretations of the MIP signal,
- they trust the per-stage ES/LS order more literally than the mixed family.

## 6.2 The ES/LS Dynamic Lexicographic Rule

The rule you specifically asked about is:

- `es_ls_stage_strict_lexicographic_release`

Its semantics are:

1. for a fixed stage, look at every unscheduled job in that stage,
2. compute that job's **current** earliest feasible insertion time on the partial schedule,
3. build the key:
   - `(current feasible time, ES, LS, original_stage_order_rank)`
4. pick the smallest job,
5. place it at its current earliest feasible slot,
6. repeat.

Important:

- this rule does **not** force the operation to start exactly at `ES`,
- `ES` is used as a release lower bound,
- the current feasible time is recomputed after **every** insertion,
- so this is dynamic, not one-time static sorting.

Implementation:

- `dispatch_stage_job_sequences_strict_lexicographic()`
- `build_schedule_from_stage_job_sequences_strict_lexicographic()`

## 6.3 Direct Mixed-Dispatch Family from Aggregate MIP Sequences

These do **not** dispatch stage by stage from a per-stage order.

Instead they:

1. build one global job sequence from aggregate MIP scores,
2. pass that sequence into the mixed dispatcher.

Variants:

| Variant | Overlay alias | Exact source |
|---|---|---|
| `mixed_aggregate_es_slack` | `mixed_agg_es` | Use the global sequence sorted by `(sum ES, sum Slack, sum LS, -TotalP, job_idx)`. |
| `mixed_aggregate_ls_slack` | `mixed_agg_ls` | Use the global sequence sorted by `(sum LS, sum Slack, sum ES, -TotalP, job_idx)`. |

Interpretation:

- these use MIP information as one global sequence,
- then let mixed dispatch do the actual machine-level construction.

## 6.4 Best-of-Mixed Family with MIP-Derived Tie-Break Rank

These do not directly feed a fixed global sequence into dispatch.

Instead they:

1. build a rank map `job -> integer`,
2. run the controller's mixed-dispatch bundle using that rank as a tie-break.

`best_of_mixed_dispatches` itself means:

- run CDS-based mixed dispatch,
- run Gupta-based mixed dispatch,
- run Palmer-based mixed dispatch,
- also try the reversed-stage instance and convert back,
- keep the best resulting schedule.

Variants:

| Variant | Overlay alias | Exact signal used for the tie-break rank |
|---|---|---|
| `best_of_mixed_dispatches_tail_ls_rank` | `mixed_tail_ls_rank` | Rank from tail-stage sequence using `(LS_last, ES_last, -p_last, job_idx)`. |
| `best_of_mixed_dispatches_bottleneck_slack_rank` | `mixed_bneck_rank` | Rank from bottleneck-stage sequence using `(Slack_bneck, LS_bneck, ES_bneck, -p_bneck, job_idx)`. |
| `best_of_mixed_dispatches_aggregate_es_slack_rank` | `mixed_agg_es_rank` | Rank from aggregate key `(sum ES, sum Slack, sum LS, -TotalP, job_idx)`. |
| `best_of_mixed_dispatches_aggregate_ls_slack_rank` | `mixed_agg_ls_rank` | Rank from aggregate key `(sum LS, sum Slack, sum ES, -TotalP, job_idx)`. |
| `best_of_mixed_dispatches_bottleneck_aggregate_es_rank` | `mixed_bneck_agg_es` | Rank from weighted blend `2*bottleneck_rank + 1*aggregate_es_rank`. |
| `best_of_mixed_dispatches_bottleneck_aggregate_ls_rank` | `mixed_bneck_agg_ls` | Rank from weighted blend `2*bottleneck_rank + 1*aggregate_ls_rank`. |
| `best_of_mixed_dispatches` | `best_mixed` | Replay the selected mixed-dispatch bundle with the default ES/LS-derived rank already passed in by post-MIP logic. |

Important:

- the “rank” is **not** the same as “force this exact job order”,
- it is a tie-break signal inside mixed-dispatch heuristics,
- this is one reason mixed-family variants can still outperform strict stage-wise variants.

## 6.5 Replayed Initializer Candidates

Post-MIP dispatch also replays the controller's
`initialize_by_best_of_selected_dispatches()` candidate list, except that:

- `bn2d_all_stages` is intentionally filtered out in post-MIP dispatch.

So by default:

- initializer default method list = `["bn2d_all_stages", "best_of_mixed_dispatches"]`
- post-MIP filtered method list = `["best_of_mixed_dispatches"]`

If the initializer config had included more methods such as:

- `stage_agg_2`
- `stage_agg_2_1`
- `stage_agg_2_2`

then those can also be replayed post-MIP.

## 6.6 Local Repair Candidate

After the best post-MIP candidate is selected once, the controller may generate:

- `selected_post_mip_local_repair`

This is **not** an independent base dispatcher. It is a repair pass applied to
the currently best post-MIP schedule.

### 6.6.1 When Is Local Repair Applied?

Local repair is applied only **after** one current best post-MIP candidate has
already been chosen.

So the flow is:

1. generate all normal dispatch candidates,
2. choose the current best one by makespan,
3. call `_repair_post_mip_dispatch_candidate(...)` on that one schedule only,
4. add the repaired result under:
   - `selected_post_mip_local_repair`
5. compare the base schedule and repaired schedule again.

Important:

- we do **not** repair every candidate,
- we repair only the current winner of the first comparison round.

### 6.6.2 What Inputs Does Local Repair Receive?

The repair step receives:

- the current selected schedule,
- `target_stage_ids`
  - in post-MIP use, this is usually:
    - bottleneck anchor stage
    - last stage
- `insertion_passes`
- `max_shift`
- `swap_passes`
- `stage_2_job_2_release`
  - the ES-derived release map

In the current post-MIP call path, the controller uses:

- `target_stage_ids = [bottleneck_anchor_stage, last_stage]`
- `insertion_passes = max(1, es_ls_local_repair_max_passes)`
- `max_shift = 4`
- `swap_passes = max(1, es_ls_local_repair_max_passes)`

So the repair is intentionally focused on:

- the stage that looks bottleneck-like in the MIP incumbent,
- and the tail stage where final makespan pressure is most visible.

### 6.6.3 What Candidate Schedules Are Actually Generated?

The controller builds a small candidate pool:

1. the original selected schedule itself
2. the result of **critical-stage sequence insertion repair**
3. the result of **critical adjacent swap repair**
4. the result of **adjacent swap applied after insertion repair**

Then it simply returns:

- the schedule with minimum makespan in that pool

So local repair is not “one operator”.
It is really:

- base
- insertion
- swap
- insertion+swap

and keep the best.

### 6.6.4 Insertion Repair: What It Actually Does

Insertion repair is implemented by:

- `improve_schedule_by_critical_stage_sequence_insertions()`

Its logic is:

1. copy the current schedule,
2. normalize it to a semi-active schedule,
3. find critical blocks:
   - `find_critical_blocks(..., include_singletons=False)`
4. extract the current realized stage-wise job sequence from the schedule,
5. for each operation that lies on a critical block:
   - only consider it if its stage is in `target_stage_ids`
6. try moving that job within the same stage sequence by:
   - `shift = -max_shift, ..., -1, +1, ..., +max_shift`
7. rebuild the **entire schedule** from the modified stage sequence using:
   - `build_schedule_from_stage_job_sequences_priority_score(...)`
8. make the rebuilt schedule semi-active,
9. keep the best improving neighbor for that pass,
10. repeat for up to `max_passes`.

Important details:

- this is a **same-stage reinsertion** move,
- it does not arbitrarily rewrite all stages,
- it only moves jobs that are currently on critical blocks,
- it only evaluates moves that stay inside the valid sequence bounds,
- the rebuilt neighbor is generated with the **priority-release stage builder**,
  not with the original base dispatch rule.

That last point is important:

- even if the base candidate came from a mixed heuristic,
- insertion repair evaluates the modified stage sequences by rebuilding with the
  release-aware stage priority dispatcher.

So insertion repair is a **lightweight stage-sequence neighborhood**, not a
re-run of the original full mixed heuristic.

### 6.6.5 Swap Repair: What It Actually Does

Swap repair is implemented by:

- `improve_schedule_by_critical_adjacent_swaps()`

Its logic is:

1. copy the current schedule,
2. make it semi-active,
3. reject immediately if the copied schedule already violates ES release times,
4. find critical blocks:
   - `find_critical_blocks(..., include_singletons=False)`
5. enumerate adjacent job pairs inside those blocks only,
6. for each adjacent pair `(job_a, job_b)` in one stage:
   - swap those two operations inside that stage
7. semi-activate the swapped schedule,
8. reject the candidate if any operation starts before its ES-derived release,
9. keep the best improving swap,
10. repeat for up to `max_passes`.

Important details:

- only **adjacent** pairs are considered,
- only pairs that appear on critical blocks are considered,
- the operator is local and cheap,
- release feasibility is explicitly checked after the swap.

### 6.6.6 Why There Is Also “Insertion + Swap”

The controller also tries:

- swap repair on top of the insertion-repaired schedule

This is meant to capture two scales of local adjustment:

- insertion:
  - move one critical job several positions
- adjacent swap:
  - fine-tune the local order after the bigger move

So `inserted_swapped` is often the “coarse move first, then polish it” variant.

### 6.6.7 What Local Repair Does **Not** Do

Local repair does **not**:

- call CP again,
- re-solve the MIP,
- rebuild every candidate from scratch,
- search a large neighborhood,
- guarantee improvement.

It is intentionally a small, cheap, purely heuristic post-processing step.

### 6.6.8 Practical Interpretation

In plain language, local repair means:

- “Take the current best post-MIP schedule.”
- “Look only at the stages most likely to matter for makespan.”
- “Look only at operations currently on critical blocks.”
- “Try a few local order edits.”
- “Re-time the schedule.”
- “Keep the best of the tiny edited variants.”

So the meaning is:

- choose the current best base candidate first,
- repair that schedule locally in a very targeted way,
- let base and repaired schedule compete again.

## 7. How the Final Post-MIP Winner Is Chosen

The current logic is:

1. generate all post-MIP dispatch candidates,
2. compare by makespan,
3. take the current best one,
4. optionally apply `selected_post_mip_local_repair` to that best one,
5. compare again,
6. keep the final best schedule.

So:

- `selected_dispatch_variant`
  - final winner after the optional repair step
- `pre_local_repair_selected_dispatch_variant`
  - winner before repair

## 8. Meaning of Current Artifact Folder Names

The overlay folders under:

- `mip_lb/dispatch/dispatch_window_overlays/`

now use names like:

- `01_SELECTED+BASE+BEST__mixed_tail_ls_rank__obj3621_gap0`
- `02_REPAIR+BEST__repair_of_mixed_tail_ls_rank__obj3621_gap0`

Interpretation:

- `01`, `02`, ...
  - rank by makespan among post-MIP candidates
- `SELECTED`
  - final selected winner
- `BASE`
  - winner before local repair
- `REPAIR`
  - repaired schedule, not a base dispatcher
- `BEST`
  - tied for the best makespan
- `repair_of_<name>`
  - this repair candidate was produced from that base dispatcher
- `obj####`
  - candidate makespan
- `gap###`
  - gap to the best post-MIP makespan

Each overlay directory also contains:

- `variant_info.yaml`

which stores:

- original long variant name,
- short alias,
- selected/base/repair flags,
- repair base name,
- makespan,
- gap to best,
- elapsed generation time.

## 9. Recommended Mental Model

If you want to analyze why a dispatch is good or bad, the most useful grouping is:

- Stage-sequence family
  - “I trust the per-stage ES/LS order and dispatch directly from it.”
- Direct mixed family
  - “I compress the MIP windows into one global sequence and let mixed dispatch build the schedule.”
- Best-of-mixed family
  - “I keep CDS/Gupta/Palmer/reversed mixed dispatch, but bias it using MIP-derived urgency ranks.”
- Repair family
  - “I take the current best post-MIP schedule and try a small local improvement around critical stages.”

In recent runs, the best schedules have often come from the mixed family,
especially variants like:

- `best_of_mixed_dispatches_tail_ls_rank`
- `best_of_mixed_dispatches_bottleneck_aggregate_es_rank`
- `best_of_mixed_dispatches_bottleneck_aggregate_ls_rank`

while the stricter ES/LS stage-sequence variants tend to preserve the MIP order
more literally but often lose on makespan.

That usually means:

- the MIP incumbent is most useful as a **soft urgency signal**,
- not as a literal stage-by-stage order that must be obeyed.
