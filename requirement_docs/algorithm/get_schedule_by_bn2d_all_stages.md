# BN2D All-Stages (get_schedule_by_bn2d_all_stages)

호출: `def get_schedule_by_bn2d_all_stages` (hybridflowshop/dispatcher/bn2d.py:645)

의존:
- `_get_schedule_from_bottleneck_stage` — 전체 BN2D 양방향 전파 (동 파일:325)
- `_get_bottleneck_stage_schedule_heuristic` — 병목 단일 스테이지 스케줄 (동 파일:53)
- `_get_bottleneck_stage` — 병목 스테이지 식별 (동 파일:305)
- `solve_selection_problem` — 좌/우 캡 선정 CP (hybridflowshop/select_and_assign.py:6)
- `BN2DOption` — 파라미터 dataclass (hybridflowshop/dispatcher/bn2d_option.py:5)

## 개요

BN2D (Bottleneck-based Two-Way Dispatching)는 Hybrid Flowshop을 병목
스테이지를 기준으로 분해(decompose)하여 전후방으로 양방향 전파하는
구축적 휴리스틱(constructive heuristic)이다. `get_schedule_by_bn2d_all_stages`
는 **모든 스테이지를 차례로 병목으로 가정**하여 BN2D를 적용한 뒤, 가장
작은 makespan을 달성한 스케줄을 반환한다.

적용 조건:

- Hybrid Flowshop (연속 스테이지, 각 스테이지에 병렬 기계)
- 목표: makespan 최소화
- 가공 시간이 정수

---

## 문제 설명

### 파라미터

알고리즘 동작을 제어하는 입력 설정값은 `BN2DOption` dataclass로 전달된다.
전체 목록·기본값은 「파라미터 요약」을 참조.

**병목 집중도 파라미터** — 좌/우 캡(cap) 작업 수를 결정한다. `_cap_multiplier`가
우선 적용되고, `None`이면 `_cap_portion`이 사용된다. 둘 다 `None`이면 해당
방향의 캡을 사용하지 않는다.

| 파라미터 쌍 | 역할 |
|-------------|------|
| `left_cap_multiplier`, `left_cap_portion` | 좌측 캡 = 일찍 시작해야 할 작업군의 크기 |
| `right_cap_multiplier`, `right_cap_portion` | 우측 캡 = 늦게 끝내야 할 작업군의 크기 |

**MID 작업 순서 제어** — 병목 스테이지에서 좌/우 캡 사이에 위치하는
mid-job들의 순서를 결정한다. 우선순위는 `randomize_mid_all` >
`reverse_mid_even` > `reverse_mid_all` 순서로 적용된다.

**디스패치 방식** — 전/후방 스테이지의 디스패치 방식을 제어한다.

| 파라미터 | 역할 |
|----------|------|
| `mixed_schedule_for_former_stages` | 전방 스테이지에 Mixed dispatch 사용 |
| `mixed_schedule_for_later_stages` | 후방 스테이지에 Mixed dispatch 사용 |
| `machine_then_job` | True면 기계 우선, False면 작업 우선 디스패치 |

**정규화**:

- `normalize_by_stage_cnt`: `r_j`(전방 합)와 `tr_j`(후방 합)를 각각
  스테이지 수로 나누고 올림. 전/후방 스테이지 수가 0이면 해당 방향은
  정규화하지 않음.

### 변수

알고리즘이 풀이 과정에서 추적하는 상태 변수다. CP 모델이 아니므로 명시적
결정변수는 없으며, 휴리스틱이 절차적으로 관리하는 값들이다.

| 변수 | 타입 | 의미 |
|------|------|------|
| `bottleneck_stage_id` | `StageIdType` | 현재 시도 중인 병목 스테이지 |
| `r_dict` | `dict[JobIdType, int]` | 작업 j의 전방 스테이지 가공시간 합 (= release time proxy) |
| `p_dict` | `dict[JobIdType, int]` | 병목 스테이지에서의 가공시간 |
| `tr_dict` | `dict[JobIdType, int]` | 작업 j의 후방 스테이지 가공시간 합 (= tail proxy) |
| `left_cap_job_id_list`, `right_cap_job_id_list` | `list[str]` | 좌/우 캡에 선정된 작업 ID 목록 |
| `mid_job_id_list` | `list[str]` | 캡에 속하지 않은 나머지 작업 |
| `sorted_j_list` | `list[JobIdType]` | L + mid + R 순서로 결합된 최종 작업 순서 |
| `job_2_bottleneck_end_time` | `dict[JobIdType, int]` | 병목 스테이지에서의 작업별 종료 시각 |
| `job_2_bottleneck_start_time` | `dict[JobIdType, int]` | 병목 스테이지에서의 작업별 시작 시각 |
| `bcmax` | `int` | 병목 스테이지 스케줄의 makespan |
| `discrepancy` | `int` | 전방 스케줄 makespan과 bcmax의 차이 |

### 목적

makespan = `max(end_time_{j,k})` (모든 작업 j, 스테이지 k) 을 최소화한다.

BN2D는 makespan을 직접 최적화하는 대신 **병목 스테이지의 부하를
가장 효율적으로 배치**하고, 그 결과를 전후방으로 전파하는 방식으로
간접적으로 makespan을 줄인다.

- 병목 스테이지에서의 추정 makespan: `max(end_time_j + tr_j)`
- `tr_j`는 후방 스테이지 처리시간의 대리값이므로, 병목 종료 후 남은
  작업량을 반영한 makespan 추정치다.

### 제약

알고리즘은 아래 불변식을 보장하며 스케줄을 구축한다.

- **작업 단위 처리**: 각 작업 j는 각 스테이지 k를 정확히 한 번
  통과하며, 가공시간 `p[j,k]` 동안 하나의 기계를 점유한다.
- **스테이지 선행 제약**: 동일 작업에 대해 스테이지 k의 종료 시각 ≤
  스테이지 k+1의 시작 시각.
- **기계 용량**: 각 기계는 한 번에 하나의 작업만 처리한다.
- **Release time**: 병목 스테이지에서 작업 j는 `r_j`(전방 가공시간 합)
  이전에 시작할 수 없다. 전방 스테이지 디스패치 시에도 release time이
  적용된다.
- **작업 순서 불변식 (병목 스테이지)**: L(좌측 캡) → mid → R(우측 캡)
  순서로 디스패치된다. 이 순서는 `dispatch_stage_by_jobs`에 전달되어
  readiness(기계 가용 시간, release time)를 고려한 실제 기계 배정의
  tie-break으로 사용된다.
- **전방 스케줄 시간 역전**: 전방 스테이지는 bottleneck start time을
  `bcmax - start_time`으로 변환한 역시간(reversed time) 도메인에서
  디스패치되며, 최종 결과는 다시 원래 시간축으로 변환된다.

---

## 핵심 아이디어

BN2D의 핵심은 **병목 스테이지를 기준으로 한 분해**와 **양방향 전파**다.

1. **병목 집중**: 전체 시스템의 처리 속도를 결정하는 병목 스테이지에
   집중한다. 병목 스테이지의 작업 순서가 전체 makespan에 가장 큰 영향을
   미친다.
2. **3분할 (L-mid-R)**: 병목 스테이지의 작업을 세 그룹으로 나눈다.
   - **L (Left cap)**: 전방 가공시간(`r_j`)이 작은 작업 → 일찍 시작할 수
     있으므로 병목에서 먼저 처리
   - **R (Right cap)**: 후방 가공시간(`tr_j`)이 작은 작업 → 병목 이후 처리할
     작업량이 적어 병목에서 늦게 끝나도 무방하므로 병목에서 나중에 처리
   - **Mid**: `r_j - tr_j` 순서로 정렬 (순수한 전/후방 압력을 반영)
3. **양방향 전파**: 병목 스케줄이 완성되면,
   - **후방 (Later stages)**: 병목 종료 시각 순으로 작업을 정렬하여
     순방향 디스패치
   - **전방 (Former stages)**: 시간을 역전(reverse time)하여 역방향
     디스패치한 후 다시 원래 시간으로 변환

---

## Bottleneck 스테이지 식별

`_get_bottleneck_stage()` (bn2d.py:305)

```text
stage_id_2_bottleneck_index[s] = sum(p[j,s] for j in jobs) / machine_cnt[s]
bottleneck = stage_id_2_bottleneck_index 의 최대값을 가진 스테이지
```

모든 스테이지에 대해 **총 가공시간 ÷ 기계 수** (=스테이지당 평균 부하)를
계산하고, 가장 큰 값을 가진 스테이지를 병목으로 선정한다.
(`get_schedule_by_bn2d_single_stage`에서만 사용됨. `_all_stages` 버전은
모든 스테이지를 순회하므로 이 메서드를 호출하지 않는다.)

---

## BN2D 병목 스케줄 휴리스틱

`_get_bottleneck_stage_schedule_heuristic()` (bn2d.py:53–211)

병목 스테이지 하나에 대한 작업 순서를 결정하고, 해당 스테이지만의 스케줄을
생성한다.

### 1. 전/후방 가공시간 합 (`r_j`, `tr_j`)

```text
r_j  = sum(p[j,s] for s in before_bottleneck_stages)    // release time proxy
p_j  = p[j, bottleneck_stage]                             // bottleneck processing time
tr_j = sum(p[j,s] for s in after_bottleneck_stages)      // tail proxy

if normalize_by_stage_cnt and stage_cnt > 0:
    r_j  = ceil(r_j  / before_stage_cnt)
    tr_j = ceil(tr_j / after_stage_cnt)
```

`r_j`는 작업 j가 병목 스테이지에 도착하기 전까지 전방에서 처리되어야 할
총 시간이므로, 병목에서의 release time 역할을 한다. `tr_j`는 병목 이후
후방 스테이지에서 처리되어야 할 총 시간이다.

### 2. 좌/우 캡 작업 수 산정

```text
if left_cap_multiplier is not None:
    left_cap_op_cnt = left_cap_multiplier × machine_cnt
elif left_cap_portion is not None:
    left_cap_op_cnt = int(left_cap_portion × job_cnt)
else:
    left_cap_op_cnt = 0

// 동일한 방식으로 right_cap_op_cnt 산정
```

`_multiplier`는 기계 수 기반, `_portion`은 전체 작업 수 기반이다.

### 3. 좌/우 캡 선정

`left_cap_op_cnt > 0` 또는 `right_cap_op_cnt > 0`인 경우에만 실행된다.

**1차 시도 — CP-SAT 기반 선정 (`solve_selection_problem`):**

```
변수: x_j ∈ {0,1} (j가 L_set), y_j ∈ {0,1} (j가 R_set)

제약:
  x_j + y_j ≤ 1      (중복 배정 금지)
  Σ x_j = K_L        (정확히 K_L개 선정)
  Σ y_j = K_R        (정확히 K_R개 선정)

1차 목적: minimize Σ (r_j·x_j + tr_j·y_j)
  → L은 r_j가 작은 작업, R은 tr_j가 작은 작업을 선호

2차 목적 (tie-break, optimal cost 고정 후 재최적화):
  minimize Σ idx_j·(x_j + y_j)   (작업 인덱스 합 최소화)
```

결과가 OPTIMAL 또는 FEASIBLE이면:
- `L_set`: `solve_selection_problem`의 결과를 `r_j` 오름차순 정렬
- `R_set`: 결과를 `tr_j` 내림차순 정렬

**Fallback — Greedy 선정 (CP-SAT이 infeasible을 반환한 경우):**

- `L_set`: 전체 작업을 `r_j` 오름차순 정렬 후 상위 `K_L`개 선택
- `R_set`: L_set에 포함되지 않은 작업 중 `tr_j` 오름차순 정렬 후
  상위 `K_R`개 선택

### 4. Mid 작업 순서 결정

좌/우 캡에 속하지 않은 나머지 작업들:

```text
mid_set = all_jobs \ (L_set ∪ R_set)

if randomize_mid_all:
    shuffle(mid_set)
else:
    sort mid_set by (r_j - tr_j, tiebreak_rank) ascending
    if reverse_mid_even:
        reverse_even_positions(mid_set)    // 1-based even index 뒤집기
    elif reverse_mid_all:
        reverse(mid_set)
```

정렬 기준 `r_j - tr_j`는 작업이 전방 압력(일찍 시작해야 함)과 후방 압력(늦게
끝나야 함) 중 어느 쪽이 더 큰지를 나타낸다. 값이 작을수록(음수) 전방 압력이
크므로 먼저 처리한다.

### 5. Bottleneck 스테이지 디스패치

```text
sorted_j_list = L_set + mid_set + R_set
dispatch_stage_by_jobs(bottleneck_stage_id, sorted_j_list, p_dict, job_2_release=r_dict)
```

`dispatch_stage_by_jobs`는 주어진 작업 순서를 tie-break 우선순위로 사용하여,
각 기계의 가용 시각과 release time(`r_dict`)을 고려해 실제 기계를 배정한다.

### 6. Makespan 추정

```text
end_time_dict = schedule.get_jik_2_end_time_map()
makespan = max(end_time + tr_dict[job_id] for all operations)
```

`tr_j`를 후방 처리시간의 대리값으로 사용하여, 병목 스케줄만으로 전체
makespan을 추정한다.

---

## 양방향 전파 (Two-Way Propagation)

`_get_schedule_from_bottleneck_stage()` (bn2d.py:325–445)

병목 스케줄을 기준으로 전방(이전 스테이지)과 후방(이후 스테이지)을 각각
디스패치하여 전체 스케줄을 완성한다.

### 후방 스테이지 (Later Stages)

```text
later_stage_list = stages[bottleneck_index + 1:]
```

후방 스테이지가 존재하면:

1. 작업을 병목 스테이지 **종료 시각** 오름차순 정렬
2. 디스패치 방식 (option에 따라):

   **A. Mixed dispatch** (`mixed_schedule_for_later_stages=True`):
   `mixed_dispatcher.get_best_mixed_schedule_by_sequence()` 호출.
   `job_2_release_t = job_2_bottleneck_end_time`으로 설정하여 각 작업이
   병목을 통과한 후에만 시작하도록 보장.

   **B. DS/DJ 병행** (기본값):
   - DS (dispatch_stage_by_jobs): 스테이지 단위로, 작업 순서를
     tie-break priority로 사용하여 디스패치
   - DJ (dispatch_job_by_stages): 작업 단위로, 각 작업을 모든 후방
     스테이지에 순차 디스패치
   - 두 결과 중 makespan이 더 작은 것을 채택

후방 스테이지가 없으면 병목 스케줄을 그대로 사용한다.

### 전방 스테이지 (Former Stages)

```text
before_stage_list = stages[:bottleneck_index]
```

전방 스테이지가 존재하면:

**1. 역시간 인스턴스 생성** (`_create_reversed_instance_for_former_stages`,
bn2d.py:213):

```text
stage_list = reverse(before_stage_list)     // 스테이지 순서 역전
job_2_release_t = {j: bcmax - bottleneck_start_time_j}
reversed_instance = create_instance_of_stage_subset(instance, stage_list)
```

전방 스테이지의 디스패치를 위해 시간을 역전시킨다. 병목 시작 시각이 늦은
작업일수록 역시간 도메인에서 release time이 작아져 더 일찍 디스패치된다.

**2. 전방 스테이지 디스패치** (`_dispatch_former_stages`, bn2d.py:245):

작업을 역시간 release time 오름차순 정렬 후 디스패치:

- **Mixed dispatch 사용** (`mixed_schedule_for_former_stages=True`):
  `MixedDispatcher.get_best_mixed_schedule_by_sequence()` 호출
- **DS/DJ 병행** (기본값):
  - DS: `dispatch_stages_by_job_sequence` — 스테이지 단위
  - DJ: `dispatch_job_sequence_by_stages` — 작업 단위
  - 더 좋은 쪽 채택

**3. 스케줄 병합:**

```text
discrepancy = former_schedule_makespan - bcmax
schedule.right_shift(discrepancy)

// 역시간 결과를 원래 시간으로 변환하여 추가
for each operation in former_schedule:
    start_time = former_schedule_makespan - end_time
    schedule.add_ops_times_2_mc(stage_id, mc_id, job_id, start_time, start_time + duration)
```

역시간 도메인에서 디스패치된 전방 스테이지의 각 operation은
`former_schedule_makespan - end_time`으로 원래 시간축으로 변환된다.
전방 스케줄이 `bcmax`보다 길면(`discrepancy > 0`), 기존 스케줄을
우측으로 밀어(shift) 시간 충돌을 방지한다.

---

## 전체 실행 흐름

`get_schedule_by_bn2d_all_stages` (bn2d.py:645–672)

```text
입력: BN2DOption, gantt_draw_func (optional)

1. best_obj = None, best_sch = None

2. for each bottleneck_stage_id in self.stage_id_list:
   │
   ├── 2a. _get_schedule_from_bottleneck_stage(bottleneck_stage_id, option)
   │   │
   │   ├── _get_bottleneck_stage_schedule_heuristic
   │   │   ├── r_j, tr_j 계산 (전/후방 가공시간 합)
   │   │   ├── 캡 작업 수 산정
   │   │   ├── solve_selection_problem (또는 greedy fallback)
   │   │   │   └── L_set, R_set 선정
   │   │   ├── mid_set 순서 결정
   │   │   ├── L + mid + R → bottleneck dispatch
   │   │   └── makespan 추정 (max(end_time + tr_j))
   │   │
   │   ├── 후방 스테이지 디스패치
   │   │   └── 병목 종료 시각 순 → mixed 또는 DS/DJ
   │   │
   │   └── 전방 스테이지 디스패치
   │       ├── 역시간 인스턴스 생성
   │       ├── release time 순 → mixed 또는 DS/DJ
   │       └── 시간축 변환 후 기존 스케줄에 병합
   │
   └── 2b. if makespan < best_obj:
               best_obj, best_sch 갱신

3. return best_sch (모든 후보 중 최소 makespan)
         또는 None (stage_id_list가 비어 루프가 한 번도 실행되지 않은 경우)
```

`gantt_draw_func`가 주어지면 각 병목 스테이지 시도마다 Gantt 차트를
출력한다. `_get_schedule_from_bottleneck_stage` 내부에서 병목 스테이지
스케줄 단계와 후방 스테이지까지 디스패치된 스케줄 단계(전방 스테이지
병합 전)에서 각각 호출된다.

---

## 파라미터 요약

### BN2DOption

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `left_cap_multiplier` | `None` | 좌측 캡 작업 수 = `multiplier × machine_cnt`. `portion`보다 우선 |
| `right_cap_multiplier` | `None` | 우측 캡 작업 수 = `multiplier × machine_cnt` |
| `left_cap_portion` | `None` | 좌측 캡 작업 수 = `int(portion × job_cnt)` |
| `right_cap_portion` | `None` | 우측 캡 작업 수 = `int(portion × job_cnt)` |
| `normalize_by_stage_cnt` | `False` | `r_j`, `tr_j`를 스테이지 수로 정규화 |
| `randomize_mid_all` | `False` | mid 작업 순서 랜덤 셔플 (최우선) |
| `reverse_mid_even` | `False` | mid에서 1-based 짝수 위치 작업 순서 역전 |
| `reverse_mid_all` | `False` | mid 전체 순서 역전 |
| `mixed_schedule_for_former_stages` | `False` | 전방 스테이지에 Mixed dispatch 사용 |
| `mixed_schedule_for_later_stages` | `False` | 후방 스테이지에 Mixed dispatch 사용 |
| `machine_then_job` | `False` | True면 기계 우선, False면 작업 우선 디스패치 |

### get_schedule_by_bn2d_all_stages 메서드 시그니처

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `option` | `BN2DOption` | BN2D 동작 제어 파라미터 |
| `gantt_draw_func` | `Callable \| None` | Gantt 차트 출력 함수. `(schedule, force_start, force_end)` 시그니처이며 `force_end`는 생략 가능(기본값 보유) |
| **반환값** | `HybridFlowshopLiteSchedule \| None` | 최소 makespan 스케줄, 또는 `stage_id_list`가 빈 경우 `None` |

---

## 주의사항 및 응용 고려사항

- **모든 스테이지를 병목으로 시도**: `get_schedule_by_bn2d_all_stages`는
  `_get_bottleneck_stage`로 식별한 병목만 사용하는 `_single_stage` 버전과
  달리, 모든 스테이지를 순회한다. 계산량은 스테이지 수에 비례하지만,
  병목 식별 오류에 강건하다(robust).
- **Makespan 추정의 근사성**: 병목 스케줄 단계의 makespan은
  `end_time + tr_j`로 추정되지만, 실제 후방 디스패치 결과와 차이가 발생할
  수 있다. 특히 병렬 기계가 많은 스테이지에서 `tr_j` 합이 실제 처리
  시간보다 길어질 수 있다.
- **전방 스케줄의 시간 역전**: 역시간 도메인에서의 디스패치는 원래
  문제와 동일한 구조(연속 스테이지, 병렬 기계)이므로 동일한 디스패치
  알고리즘을 재사용할 수 있다. 스테이지 순서가 역전된다는 점에 주의.
- **L-mid-R 분할의 민감도**: 캡 작업 수(`cap_multiplier`, `cap_portion`)와
  mid 순서 규칙(`reverse_mid_even`, `randomize_mid_all`)에 따라 결과가
  크게 달라질 수 있다. 가장 효과적인 조합은 문제 인스턴스 특성에 의존하므로,
  여러 설정을 시도하는 것이 바람직하다.
- **fallback greedy 선정**: `solve_selection_problem`이 infeasible을
  반환하면 greedy fallback으로 전환된다. CP-SAT이 infeasible을 반환하는
  경우는 매우 드물지만(K_L + K_R ≤ job_count가 보장됨), 안전장치로 존재한다.
- **Mixed dispatch vs DS/DJ**: Mixed dispatch는 head-tail 개념으로 일부
  작업을 우선 디스패치하는 방식으로, DS/DJ 병행보다 일반적으로 더 좋은
  결과를 내지만 계산량이 더 크다.
