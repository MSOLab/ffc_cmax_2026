# Post-MIP Dispatch Rules

이 문서는 `apply_mip_lb()`가 bucket-indexed MIP를 푼 뒤, incumbent
dispatch-window solution을 추출한 다음 평가하는 dispatch 후보들을 정리한
문서입니다.

주요 진입점은 다음 파일입니다.

- `lb_bucket/mip/post_dispatch.py`

주요 helper 구현은 다음 파일들에 있습니다.

- `hybridflowshop/dispatcher/utils.py`
- `hybridflowshop/controller/hfs_cp_lns.py`

## 1. 큰 그림

`apply_mip_lb()`가 끝났다고 해서 MIP incumbent를 바로 최종 machine-level
schedule로 쓰지는 않습니다. 대신 아래 흐름으로 처리합니다.

1. bucket-indexed MIP를 푼다.
2. incumbent를 per-operation dispatch-window 데이터로 변환한다.
3. 그 데이터로부터 여러 ES/LS 기반 order signal을 만든다.
4. 여러 dispatch candidate를 만든다.
5. makespan으로 비교한다.
6. 현재 best 하나에 대해서만 optional repair를 적용한다.
7. 최종 best schedule을 남긴다.

즉 MIP incumbent는 주로 **signal generator** 역할을 합니다.

- stage별 order hint
- job별 release hint
- job별 latest-start hint
- global urgency / ranking hint

## 2. MIP에서 정확히 무엇이 나오는가

post-MIP dispatcher는 `dispatch_windows`를 입력으로 사용합니다. 이 값은 저장된
incumbent 변수 `a`, `b`, `x`로부터 만들어집니다.

각 operation `(stage s, job j)`에 대해 window record에는 다음과 같은 값들이
들어 있습니다.

- `early_start`
  - 해당 operation의 incumbent 기반 ES time
- `late_start`
  - 해당 operation의 incumbent 기반 LS time
- `slack = late_start - early_start`
  - incumbent가 암시하는 start window의 폭
- `processing_time`
  - operation processing time
- `a_bucket`, `b_bucket`
  - bucket model에서 해당 operation이 닿는 첫/마지막 bucket
- `x_bucket_*`, `x_value_*`
  - 각 bucket을 얼마나 점유하는지

중요한 점:

- 이 ES/LS 값들은 **incumbent로부터 유도된 값**이지, 절대적인 진실이 아닙니다.
- 유용한 signal이긴 하지만, real dispatch schedule의 최적 start time을 보장하지는
  않습니다.
- 그래서 좋은 dispatch들이 실제로는 operation을 **ES보다 앞** 또는 **LS보다 뒤**
  에 놓는 경우가 생깁니다.

## 3. 이 문서에서 쓰는 용어 정리

### 3.1 ES, LS, Slack

하나의 operation `(s, j)`에 대해:

- `ES(s,j) = early_start`
- `LS(s,j) = late_start`
- `Slack(s,j) = LS(s,j) - ES(s,j)`

해석:

- `ES`가 작다
  - incumbent 기준으로 이 operation은 일찍 가는 편이 좋아 보인다
- `LS`가 작다
  - incumbent 기준으로 이 operation을 늦추는 것이 위험해 보인다
- `Slack`이 작다
  - incumbent 기준으로 timing freedom이 적다

### 3.2 Stage Job Sequence

**stage job sequence**는 stage마다 따로 있는 ordered job list입니다.

- stage 1: `[j?, j?, ...]`
- stage 2: `[j?, j?, ...]`
- ...

이 값은 다음 함수로 만들어집니다.

- `get_stage_job_sequences_from_dispatch_windows()`

그리고 stage-sequence family의 dispatcher들이 이 값을 사용합니다.

### 3.3 Global Job Sequence

**global job sequence**는 문제 전체에 대해 하나만 있는 job order입니다.

- `[j?, j?, j?, ...]`

이 값은 다음 두 방식 중 하나로 만들어집니다.

- 하나의 anchor stage만 사용
- 모든 stage에 대해 aggregate score를 계산해서 사용

direct mixed family가 이 값을 사용합니다.

### 3.4 Global Tie-Break Rank

**job tie-break rank**는 다음과 같은 dictionary입니다.

- `job_id -> integer rank`

작은 rank일수록 “heuristic이 tie-break가 필요할 때 이 job을 더 앞세워라”는
뜻입니다.

best-of-mixed family가 이 값을 사용합니다.

중요한 점:

- best-of-mixed family에서 rank는 보통 **secondary signal**입니다.
- CDS / Gupta / Palmer 자체를 대체하지는 않습니다.
- 그 heuristic 내부에서 tie 또는 near-tie를 푸는 데에 주로 사용됩니다.

### 3.5 Release Map

release map은 다음과 같습니다.

- `stage_2_job_release[stage_id][job_id] = int(ES(stage, job))`

이 값은 다음 함수로 만들어집니다.

- `get_stage_job_release_times_from_dispatch_windows()`

그리고 stage-sequence builder들에 start time lower bound로 전달됩니다.

### 3.6 Latest-Start Map

latest-start map은 다음과 같습니다.

- `stage_2_job_latest_start[stage_id][job_id] = int(LS(stage, job))`

이 값은 다음 함수로 만들어집니다.

- `get_stage_job_latest_start_times_from_dispatch_windows()`

현재 이 값을 사용하는 대표 규칙은:

- `es_ls_stage_strict_lexicographic_release`

입니다.

### 3.7 Anchor Stage

**anchor stage**란:

- “특정 stage 하나를 고르고, 그 stage의 ES/LS window 정보만 이용해서 global job
  order를 만든다”

라는 뜻입니다.

여기서는 두 가지 anchor-stage 스타일이 중요합니다.

- tail-stage anchor
  - 마지막 stage만 사용
- bottleneck-stage anchor
  - bottleneck 성격이 강한 stage 하나만 사용

### 3.8 Bottleneck Anchor Stage

bottleneck anchor stage는 다음 함수가 고릅니다.

- `get_bottleneck_anchor_stage_from_solution_payload()`

주요 signal:

- incumbent `x` 값으로부터 stage-bucket congestion의 최대치를 본다
- 공식:
  - `max_t [ sum_j x[s,j,t] / (m_s * delta) ]`

여기서:

- `m_s`
  - stage `s`의 machine 수
- `delta`
  - bucket width

payload 정보가 약하거나 없을 때의 fallback signal:

- 평균 load proxy:
  - `sum_j p[s,j] / m_s`

해석:

- incumbent에서 congestion이 강하게 보이는 stage를 하나 고른다
- 그 stage의 ES/LS 구조를 강한 ordering signal로 사용한다

## 4. MIP-derived order를 만들 때 실제로 쓰는 sort key

이 섹션은 이런 질문에 답합니다.

- “aggregate ES-slack information이 정확히 뭐지?”
- “bottleneck slack rank는 정확히 어떻게 계산하지?”

### 4.1 Per-Stage Sort Rules

다음 함수에 구현되어 있습니다.

- `_get_dispatch_window_sort_key()`

하나의 operation에 대해:

- `ES = early_start`
- `LS = late_start`
- `Slack = LS - ES`
- `Midpoint = ES + LS`
- `p = processing_time`

built-in rule은 다음과 같습니다.

| Rule name | Exact sort key |
|---|---|
| `es_ls_p_desc` | `(ES, LS, -p, job_idx)` |
| `ls_es_p_desc` | `(LS, ES, -p, job_idx)` |
| `slack_ls_es_p_desc` | `(Slack, LS, ES, -p, job_idx)` |
| `es_slack_ls_p_desc` | `(ES, Slack, LS, -p, job_idx)` |
| `midpoint_slack_ls_p_desc` | `(Midpoint, Slack, LS, -p, job_idx)` |

의미:

- tuple을 오름차순으로 정렬합니다.
- 따라서 앞쪽 field가 더 작은 job이 더 urgent합니다.
- `-p`는 앞쪽 field가 같을 때 더 긴 operation을 우선시한다는 뜻입니다.

### 4.2 Stage 전체를 합친 Aggregate Sort Rules

다음 함수에 구현되어 있습니다.

- `_get_dispatch_window_aggregate_key()`

job `j`에 대해 다음을 정의합시다.

- `ES_k(j)`
  - stage `k`에서 job `j`의 ES
- `LS_k(j)`
  - stage `k`에서 job `j`의 LS
- `Slack_k(j) = LS_k(j) - ES_k(j)`
- `TotalP(j) = sum_k p_k(j)`

aggregate rule은 다음과 같습니다.

| Aggregation rule | Exact sort key |
|---|---|
| `sum_es_slack_p_desc` | `(sum_k ES_k, sum_k Slack_k, sum_k LS_k, -TotalP, job_idx)` |
| `sum_ls_slack_p_desc` | `(sum_k LS_k, sum_k Slack_k, sum_k ES_k, -TotalP, job_idx)` |
| `tail_ls_sum_slack_p_desc` | `(LS_last, sum_k Slack_k, ES_last, -TotalP, job_idx)` |

### 4.3 “Aggregate ES-Slack Information”이 정확히 의미하는 것

이 표현은 MIP가 따로 저장해주는 어떤 special variable을 뜻하는 것이 아닙니다.

의미는 다음과 같습니다.

- 각 job에 대해 모든 stage의 ES/LS window를 모은다.
- 그 다음 다음 값을 계산한다.
  - `sum_k ES_k`
  - `sum_k Slack_k`
  - `sum_k LS_k`
  - `TotalP`
- 그리고 다음 key로 정렬한다.
  - `(sum ES, sum Slack, sum LS, -TotalP, job_idx)`

즉 쉬운 말로 하면:

- 전체 routing을 봤을 때 전반적으로 더 이르게 잡혀 있는 job을 앞세우고,
- 비슷하게 이른 job들 중에서는 total slack이 더 타이트한 job을 앞세우고,
- 그 다음 total LS가 더 이른 job을 앞세우고,
- 마지막으로 total processing time이 더 큰 job을 앞세웁니다.

이것이 문서에서 말하는:

- `aggregate ES-slack information`

의 정확한 의미입니다.

### 4.4 “Aggregate LS-Slack Information”이 정확히 의미하는 것

비슷하게 이 표현은 다음을 의미합니다.

- 각 job의 ES/LS window를 모든 stage에 대해 모은다.
- 다음 key로 정렬한다.
  - `(sum LS, sum Slack, sum ES, -TotalP, job_idx)`

즉 쉬운 말로 하면:

- 전체적으로 latest-start limit이 더 이른 job을 앞세우고,
- 그 다음 total slack이 더 타이트한 job을 앞세우고,
- 그 다음 total ES가 더 이른 job을 앞세우고,
- 마지막으로 total processing time이 더 큰 job을 앞세웁니다.

### 4.5 간단한 예시

어떤 job이 다음 window를 가진다고 합시다.

- stage 1: `ES=10`, `LS=20`, `Slack=10`
- stage 2: `ES=30`, `LS=45`, `Slack=15`
- stage 3: `ES=55`, `LS=70`, `Slack=15`

그러면:

- `sum ES = 10 + 30 + 55 = 95`
- `sum LS = 20 + 45 + 70 = 135`
- `sum Slack = 10 + 15 + 15 = 40`

따라서 aggregate key는:

- aggregate ES-slack key:
  - `(95, 40, 135, -TotalP, job_idx)`
- aggregate LS-slack key:
  - `(135, 40, 95, -TotalP, job_idx)`

job끼리 비교할 때는 tuple이 더 작은 쪽이 앞섭니다.

## 5. 각 MIP-derived sequence / rank를 정확히 어떻게 만드는가

이 섹션은 variant 이름과 실제 underlying signal을 연결해줍니다.

### 5.1 Tail LS Sequence

원천:

- `get_job_sequence_from_dispatch_windows_anchor_stage(..., anchor_stage_id=last_stage, sort_rule="ls_es_p_desc")`

마지막 stage에서 job `j`에 대한 exact key:

- `(LS_last(j), ES_last(j), -p_last(j), job_idx)`

해석:

- **마지막 stage**에서 latest start가 더 빠른 job을 우선합니다.

### 5.2 Bottleneck Slack Sequence

원천:

- `get_job_sequence_from_dispatch_windows_anchor_stage(..., anchor_stage_id=bottleneck_anchor_stage, sort_rule="slack_ls_es_p_desc")`

선택된 bottleneck stage에서 job `j`에 대한 exact key:

- `(Slack_bneck(j), LS_bneck(j), ES_bneck(j), -p_bneck(j), job_idx)`

해석:

- bottleneck stage에서 더 타이트한 job을 우선합니다.

### 5.3 Aggregate ES Slack Sequence

원천:

- `get_job_sequence_from_dispatch_windows_aggregate(..., aggregation_rule="sum_es_slack_p_desc")`

exact key:

- `(sum ES, sum Slack, sum LS, -TotalP, job_idx)`

해석:

- 전체적으로 더 이르고 더 타이트해 보이는 job을 우선합니다.

### 5.4 Aggregate LS Slack Sequence

원천:

- `get_job_sequence_from_dispatch_windows_aggregate(..., aggregation_rule="sum_ls_slack_p_desc")`

exact key:

- `(sum LS, sum Slack, sum ES, -TotalP, job_idx)`

해석:

- 전체적으로 더 late-sensitive하고 더 타이트해 보이는 job을 우선합니다.

### 5.5 Weighted Blended Ranks

다음 함수에서 만들어집니다.

- `_build_weighted_job_tiebreak_rank()`

동작:

1. 여러 sequence별 rank를 만든다.
   - 예: `bottleneck rank`, `aggregate ES rank`
2. 각 job이 각 sequence에서 몇 번째인지 읽는다.
3. 그 position의 weighted sum을 계산한다.
4. 다음 key로 정렬한다.
   - `(weighted_sum, min(component_ranks), component_ranks..., original_job_order)`
5. 이 sorted job list를 `job_id -> rank`로 변환한다.

따라서 새 blended variant의 의미는:

| Variant | Exact blended score |
|---|---|
| `best_of_mixed_dispatches_bottleneck_aggregate_es_rank` | `2 * rank_in_bottleneck_slack_sequence + 1 * rank_in_aggregate_es_sequence` |
| `best_of_mixed_dispatches_bottleneck_aggregate_ls_rank` | `2 * rank_in_bottleneck_slack_sequence + 1 * rank_in_aggregate_ls_sequence` |

해석:

- “bottleneck urgency를 더 강하게 믿되, global aggregate signal로 약간 보정한다”

## 6. Bucket MIP 이후 평가하는 Candidate Family

## 6.1 Stage-Sequence Dispatch Family

이 family는 MIP window로부터 추출한 per-stage job sequence를 사용합니다.

이 stage sequence는 현재 다음 함수로 만들어집니다.

- `get_stage_job_sequences_from_dispatch_windows(...)`

기본 rule:

- `es_ls_p_desc`
- exact key:
  - `(ES, LS, -p, job_idx)`

즉 raw stage order 자체의 뜻은:

- ES가 더 이른 job 먼저,
- 그 다음 LS가 더 이른 job,
- 그 다음 processing time이 더 긴 job입니다.

variant는 다음과 같습니다.

| Variant | Overlay alias | Exact behavior |
|---|---|---|
| `es_ls_stage_priority_release` | `esls_priority` | stage sequence를 priority / tie-break signal로만 쓰는 release-aware priority queue dispatch |
| `es_ls_stage_strict_call_release` | `esls_strict_call` | 주어진 stage sequence 순서대로 정확히 call하고, `ES`를 release lower bound로 사용 |
| `es_ls_stage_strict_lexicographic_release` | `esls_strict_lex` | 매 step마다 current feasible start를 다시 계산하고 `current feasible time -> ES -> LS -> original stage order`로 정렬 |
| `es_ls_stage_strict_start_release` | `esls_strict_start` | 주어진 stage order를 realized start order로 최대한 보존하려고 하는 방식 |

중요한 점:

- 이 family는 기본적으로 **stage-local**하게 MIP signal을 해석합니다.
- mixed family보다 per-stage ES/LS order를 더 문자 그대로 믿는 편입니다.

## 6.2 ES/LS Dynamic Lexicographic Rule

당신이 특히 물었던 규칙은:

- `es_ls_stage_strict_lexicographic_release`

입니다.

의미는 다음과 같습니다.

1. 하나의 stage에 대해 아직 스케줄되지 않은 모든 job을 본다.
2. 각 job의 **현재** earliest feasible insertion time을 partial schedule에서 계산한다.
3. 다음 key를 만든다.
   - `(current feasible time, ES, LS, original_stage_order_rank)`
4. 가장 작은 key를 가진 job을 고른다.
5. 그 job을 현재 earliest feasible slot에 배치한다.
6. 반복한다.

중요한 점:

- 이 규칙은 operation을 `ES`에 정확히 꽂는 규칙이 아닙니다.
- `ES`는 release lower bound로만 쓰입니다.
- current feasible time은 **매 insertion 후 다시 계산**됩니다.
- 따라서 static sort가 아니라 dynamic rule입니다.

구현:

- `dispatch_stage_job_sequences_strict_lexicographic()`
- `build_schedule_from_stage_job_sequences_strict_lexicographic()`

## 6.3 Aggregate MIP Sequence에서 바로 가는 Direct Mixed-Dispatch Family

이 family는 per-stage order로 stage를 직접 dispatch하지 않습니다.

대신:

1. aggregate MIP score로 global job sequence 하나를 만든다.
2. 그 sequence를 mixed dispatcher에 넣는다.

variant:

| Variant | Overlay alias | Exact source |
|---|---|---|
| `mixed_aggregate_es_slack` | `mixed_agg_es` | `(sum ES, sum Slack, sum LS, -TotalP, job_idx)`로 만든 global sequence를 사용 |
| `mixed_aggregate_ls_slack` | `mixed_agg_ls` | `(sum LS, sum Slack, sum ES, -TotalP, job_idx)`로 만든 global sequence를 사용 |

해석:

- MIP 정보를 하나의 global sequence로 압축하고,
- 실제 machine-level construction은 mixed dispatch에게 맡깁니다.

## 6.4 MIP-Derived Tie-Break Rank를 쓰는 Best-of-Mixed Family

이 family는 fixed global sequence를 직접 넣는 것이 아니라:

1. `job -> integer rank`를 만든다.
2. 그 rank를 tie-break signal로 주면서 controller의 mixed-dispatch bundle을 돌린다.

`best_of_mixed_dispatches` 자체는 다음을 의미합니다.

- CDS-based mixed dispatch 실행
- Gupta-based mixed dispatch 실행
- Palmer-based mixed dispatch 실행
- reversed-stage instance도 돌린 뒤 다시 원래 방향으로 되돌림
- 그중 best schedule 선택

variant:

| Variant | Overlay alias | Exact signal used for the tie-break rank |
|---|---|---|
| `best_of_mixed_dispatches_tail_ls_rank` | `mixed_tail_ls_rank` | `(LS_last, ES_last, -p_last, job_idx)`로 만든 tail-stage sequence에서 rank 생성 |
| `best_of_mixed_dispatches_bottleneck_slack_rank` | `mixed_bneck_rank` | `(Slack_bneck, LS_bneck, ES_bneck, -p_bneck, job_idx)`로 만든 bottleneck-stage sequence에서 rank 생성 |
| `best_of_mixed_dispatches_aggregate_es_slack_rank` | `mixed_agg_es_rank` | `(sum ES, sum Slack, sum LS, -TotalP, job_idx)` aggregate key에서 rank 생성 |
| `best_of_mixed_dispatches_aggregate_ls_slack_rank` | `mixed_agg_ls_rank` | `(sum LS, sum Slack, sum ES, -TotalP, job_idx)` aggregate key에서 rank 생성 |
| `best_of_mixed_dispatches_bottleneck_aggregate_es_rank` | `mixed_bneck_agg_es` | `2*bottleneck_rank + 1*aggregate_es_rank` weighted blend로 rank 생성 |
| `best_of_mixed_dispatches_bottleneck_aggregate_ls_rank` | `mixed_bneck_agg_ls` | `2*bottleneck_rank + 1*aggregate_ls_rank` weighted blend로 rank 생성 |
| `best_of_mixed_dispatches` | `best_mixed` | post-MIP logic가 넘겨주는 default ES/LS-derived rank를 포함하여 selected mixed-dispatch bundle을 다시 평가 |

중요한 점:

- 여기서 말하는 “rank”는 **이 exact order를 강제하라**는 뜻이 아닙니다.
- mixed-dispatch heuristic 내부에서 tie-break signal로 쓰인다는 뜻입니다.
- 그래서 mixed-family variant가 strict stage-wise variant보다 더 잘 나오는 경우가 많습니다.

## 6.5 Replayed Initializer Candidates

post-MIP dispatch는 controller의
`initialize_by_best_of_selected_dispatches()` candidate list도 다시 평가합니다.
다만:

- `bn2d_all_stages`는 post-MIP dispatch에서 의도적으로 제외합니다.

따라서 기본값은:

- initializer default method list = `["bn2d_all_stages", "best_of_mixed_dispatches"]`
- post-MIP filtered method list = `["best_of_mixed_dispatches"]`

만약 initializer config에 다음 같은 method가 더 들어 있었으면:

- `stage_agg_2`
- `stage_agg_2_1`
- `stage_agg_2_2`

이들도 post-MIP에서 replay될 수 있습니다.

## 6.6 Local Repair Candidate

최초 best post-MIP candidate가 한 번 선택된 뒤에는, controller가 다음 후보를
추가로 만들 수 있습니다.

- `selected_post_mip_local_repair`

이것은 독립적인 base dispatcher가 아닙니다. 현재 best post-MIP schedule에
local repair를 적용한 결과입니다.

### 6.6.1 Local Repair는 언제 적용되는가?

local repair는 현재 best post-MIP candidate가 이미 하나 정해진 **후에만**
적용됩니다.

흐름:

1. 일반 dispatch candidate를 모두 만든다.
2. makespan으로 현재 best 하나를 고른다.
3. 그 schedule 하나에만 `_repair_post_mip_dispatch_candidate(...)`를 호출한다.
4. 결과를 다음 이름으로 추가한다.
   - `selected_post_mip_local_repair`
5. base schedule과 repaired schedule을 다시 비교한다.

중요한 점:

- 모든 candidate를 repair하지는 않습니다.
- 1차 winner 하나만 repair합니다.

### 6.6.2 Local Repair는 어떤 입력을 받는가?

repair step은 다음 입력을 받습니다.

- 현재 selected schedule
- `target_stage_ids`
  - post-MIP에서는 보통:
    - bottleneck anchor stage
    - last stage
- `insertion_passes`
- `max_shift`
- `swap_passes`
- `stage_2_job_2_release`
  - ES-derived release map

현재 post-MIP call path에서 controller는 보통 다음 값을 씁니다.

- `target_stage_ids = [bottleneck_anchor_stage, last_stage]`
- `insertion_passes = max(1, es_ls_local_repair_max_passes)`
- `max_shift = 4`
- `swap_passes = max(1, es_ls_local_repair_max_passes)`

즉 repair는 의도적으로:

- MIP incumbent에서 bottleneck처럼 보이는 stage,
- 그리고 makespan에 직접 영향이 큰 tail stage

에 집중합니다.

### 6.6.3 실제로 어떤 Candidate Schedule을 만드는가?

controller는 작은 candidate pool 하나를 만듭니다.

1. 원래 selected schedule 자체
2. **critical-stage sequence insertion repair** 결과
3. **critical adjacent swap repair** 결과
4. **insertion repair 후 다시 swap repair**를 적용한 결과

그리고 그 pool 안에서:

- makespan이 가장 작은 schedule

을 반환합니다.

즉 local repair는 “한 가지 operator”가 아니라:

- base
- insertion
- swap
- insertion+swap

을 모두 만들어 보고 best를 고르는 구조입니다.

### 6.6.4 Insertion Repair는 실제로 무엇을 하는가?

다음 함수에 구현되어 있습니다.

- `improve_schedule_by_critical_stage_sequence_insertions()`

로직:

1. 현재 schedule을 복사한다.
2. semi-active schedule로 normalize한다.
3. critical block을 찾는다.
   - `find_critical_blocks(..., include_singletons=False)`
4. 현재 realized stage-wise job sequence를 추출한다.
5. critical block 위의 각 operation에 대해:
   - 그 stage가 `target_stage_ids` 안에 있을 때만 본다.
6. 같은 stage sequence 안에서 그 job을 다음 shift만큼 이동해 본다.
   - `shift = -max_shift, ..., -1, +1, ..., +max_shift`
7. 수정된 stage sequence로부터 **전체 schedule을 다시 rebuild**한다.
   - `build_schedule_from_stage_job_sequences_priority_score(...)`
8. rebuilt schedule을 semi-active로 만든다.
9. 그 pass에서 best improving neighbor를 남긴다.
10. 최대 `max_passes`번 반복한다.

중요한 점:

- 같은 stage 안에서의 **reinsertion move**입니다.
- 모든 stage를 마음대로 갈아엎는 것이 아닙니다.
- 현재 critical block 위에 있는 job만 이동 대상으로 봅니다.
- sequence 범위를 벗어나는 move는 평가하지 않습니다.
- rebuilt neighbor는 원래 base dispatch rule이 아니라
  **priority-release stage builder**로 평가합니다.

마지막 점이 중요합니다.

- base candidate가 mixed heuristic에서 왔더라도,
- insertion repair는 수정된 sequence를 priority-release builder로 rebuild해서
  neighbor를 평가합니다.

즉 insertion repair는

- “원래 mixed heuristic을 그대로 다시 돌리는 repair”

라기보다,

- “critical-stage sequence neighborhood를 priority-release builder로 평가하는
  local search”

에 더 가깝습니다.

### 6.6.5 Swap Repair는 실제로 무엇을 하는가?

다음 함수에 구현되어 있습니다.

- `improve_schedule_by_critical_adjacent_swaps()`

로직:

1. 현재 schedule을 복사한다.
2. semi-active로 만든다.
3. 복사된 schedule이 이미 ES release를 위반하면 바로 중단한다.
4. critical block을 찾는다.
   - `find_critical_blocks(..., include_singletons=False)`
5. critical block 안에서만 adjacent job pair를 열거한다.
6. 각 adjacent pair `(job_a, job_b)`에 대해:
   - 해당 stage 안에서 두 operation의 순서를 swap한다.
7. swapped schedule을 semi-active로 만든다.
8. 어떤 operation이라도 ES-derived release보다 빨리 시작하면 버린다.
9. best improving swap을 남긴다.
10. 최대 `max_passes`번 반복한다.

중요한 점:

- **adjacent** pair만 봅니다.
- critical block에 있는 pair만 봅니다.
- operator가 매우 local하고 cheap합니다.
- swap 후에는 release feasibility를 명시적으로 체크합니다.

### 6.6.6 왜 “Insertion + Swap”도 따로 보는가?

controller는 다음도 따로 시도합니다.

- insertion-repaired schedule 위에 다시 swap repair를 적용

이유는 두 가지 다른 scale의 local adjustment를 같이 잡기 위해서입니다.

- insertion
  - critical job 하나를 여러 칸 크게 이동
- adjacent swap
  - 그 큰 이동 뒤의 local order를 미세 조정

즉 `inserted_swapped`는 보통:

- “먼저 크게 움직이고, 그다음 국소적으로 다듬는” 후보

라고 보면 됩니다.

### 6.6.7 Local Repair가 하지 않는 것

local repair는 다음을 하지 않습니다.

- CP를 다시 호출하지 않음
- MIP를 다시 풀지 않음
- 모든 candidate를 처음부터 다시 만들지 않음
- 큰 neighborhood search를 하지 않음
- improvement를 보장하지 않음

즉 의도적으로 작고 싼 purely heuristic post-processing step입니다.

### 6.6.8 실무적인 해석

쉽게 말하면 local repair는 다음 뜻입니다.

- “현재 best post-MIP schedule 하나를 잡는다.”
- “makespan에 중요할 가능성이 높은 stage만 본다.”
- “그 안에서도 현재 critical block에 걸린 operation만 본다.”
- “아주 작은 local order edit 몇 개를 시도한다.”
- “schedule을 다시 retime한다.”
- “그중 best를 남긴다.”

즉 의미는:

- 먼저 current best base candidate를 하나 고르고,
- 그 schedule에만 아주 targeted한 local repair를 적용하고,
- base와 repaired schedule을 다시 경쟁시키는 것입니다.

## 7. 최종 Post-MIP Winner는 어떻게 정해지는가

현재 로직은 다음과 같습니다.

1. 모든 post-MIP dispatch candidate를 생성한다.
2. makespan으로 비교한다.
3. 현재 best 하나를 고른다.
4. optional하게 `selected_post_mip_local_repair`를 그 best schedule에 적용한다.
5. 다시 비교한다.
6. 최종 best schedule을 남긴다.

따라서:

- `selected_dispatch_variant`
  - optional repair 이후의 최종 winner
- `pre_local_repair_selected_variant`
  - repair 이전의 winner

## 8. 현재 Artifact Folder 이름의 의미

다음 폴더 아래의 overlay folder는:

- `mip_lb/dispatch/dispatch_window_overlays/`

현재 이런 이름을 사용합니다.

- `01_SELECTED+BASE+BEST__mixed_tail_ls_rank__obj3621_gap0`
- `02_REPAIR+BEST__repair_of_mixed_tail_ls_rank__obj3621_gap0`

해석:

- `01`, `02`, ...
  - post-MIP candidate를 makespan 기준으로 정렬한 rank
- `SELECTED`
  - 최종 selected winner
- `BASE`
  - local repair 이전 winner
- `REPAIR`
  - base dispatcher가 아니라 repaired schedule
- `BEST`
  - best makespan과 tie
- `repair_of_<name>`
  - 이 repair candidate가 어떤 base dispatcher에서 나왔는지
- `obj####`
  - candidate makespan
- `gap###`
  - best post-MIP makespan과의 gap

각 overlay directory에는 다음 파일도 들어 있습니다.

- `variant_info.yaml`

이 안에는:

- original long variant name
- short alias
- selected / base / repair flag
- repair base name
- makespan
- gap to best
- elapsed generation time

등이 저장됩니다.

## 9. 추천하는 Mental Model

어떤 dispatch가 왜 좋고 나쁜지를 분석하려면 다음 grouping이 가장 유용합니다.

- Stage-sequence family
  - “per-stage ES/LS order를 꽤 직접적으로 믿고 dispatch한다.”
- Direct mixed family
  - “MIP window를 하나의 global sequence로 압축하고, 실제 schedule construction은 mixed dispatch에 맡긴다.”
- Best-of-mixed family
  - “CDS / Gupta / Palmer / reversed mixed dispatch는 그대로 두고, MIP-derived urgency rank로 bias만 준다.”
- Repair family
  - “현재 best post-MIP schedule 하나를 잡고 critical 영역 근처만 소규모로 고쳐본다.”

최근 run에서는 best schedule이 mixed family에서 나오는 경우가 많았고,
특히 다음 같은 variant가 강한 편이었습니다.

- `best_of_mixed_dispatches_tail_ls_rank`
- `best_of_mixed_dispatches_bottleneck_aggregate_es_rank`
- `best_of_mixed_dispatches_bottleneck_aggregate_ls_rank`

반면 더 strict한 ES/LS stage-sequence variant는 MIP order를 더 문자 그대로
보존하는 대신, makespan에서는 지는 경우가 많았습니다.

보통 이것은 다음 뜻입니다.

- MIP incumbent는 **soft urgency signal**로 쓸 때 가장 유용하고,
- stage-by-stage order를 그대로 강제하는 방식은 대체로 덜 좋다

라고 해석할 수 있습니다.
