# `dispatch_stage_by_machines`

**하나의 stage에서 여러 job을 동시에 고려하여, “machine 중심”으로 순차적으로 배정해 나가는 디스패칭 알고리즘**입니다.

구현은 `HybridFlowshopLiteSchedule` 내부에 있습니다.

아래에서 구조와 흐름을 단계별로 정리하겠습니다.

---

## 1. 전체 개념

우선순위 구조는 다음과 같습니다.

- Stage 고정
- Machine 선택
- 그 Machine에 배정할 Job 선택

즉,
“어떤 job을 먼저?”가 아니라
“어떤 machine이 가장 빨리 일할 수 있는가?”를 먼저 결정합니다.

---

## 2. 사전 준비 단계 (Precomputation)

### 2.1 인덱스 및 기본 정보

- job_id_2_pos
  → 입력 job sequence의 순서 (최종 tie-breaker)

- stage_idx
  → 현재 stage의 인덱스

- remaining_stages
  → 이후 stage들

- is_last_stage
  → 마지막 stage인지 여부

- lpt_sign
  → 마지막 stage에서 LPT/SPT 선택용 부호

- mc_list
  → 해당 stage의 machine 목록

---

### 2.2 Remaining Processing Time 계산

각 job에 대해:

```
job_2_remaining_pt[j] = 이후 stage들의 processing time 합
```

이 값은 job 선택 시 tie-breaker로 사용됩니다.

- 값이 클수록 downstream 영향이 큼
- 따라서 더 먼저 처리하는 전략 (−remaining_pt 사용)

---

### 2.3 Release Time 정리

각 job에 대해:

```
release_t = max(prev_stage_end, external_release)
```

precedence constraint + release constraint를 미리 정리해 둡니다.

---

## 3. Phase 1 — 전체 EAT 캐시 생성

각 job × 각 machine에 대해:

```
eat, idle = get_eat_for_machine(...)
```

저장 구조:

```
job_2_mc_cache[job][mc] = (eat, idle)
job_2_best[job] = (best_mc, best_eat, best_idle)
```

여기서 best_mc는 (eat, idle) 사전식 비교로 선택됩니다.

---

### Machine 관점 요약

각 machine에 대해:

```
mc_2_best_eat[mc] =
    해당 machine을 best로 가지는 job들 중 최소 EAT
```

이 값이 “이 machine이 다음에 일을 시작할 수 있는 가장 빠른 시점”이 됩니다.

---

## 4. Phase 2 — 반복 디스패칭

while unscheduled_jobs:

### Step 1 — Target Machine 선택

각 machine에 대해 다음 key 계산:

```
(min_eat, idle_at_that_eat, mc_index)
```

가장 작은 machine 선택.

우선순위:

- 가장 작은 EAT
- idle 최소
- machine index

이것이 “다음으로 움직일 machine”입니다.

---

### Step 2 — Candidate Job 필터링

조건:

- 그 machine이 best_mc
- best_eat == target_eat

인 job들만 후보.

---

### Step 3 — Job 선택 기준

`job_sort_key`는 다음 tuple을 반환:

```
(
    effective_start,
    -remaining_pt,
    stage_tb,
    input_order
)
```

각 항목 의미:

- effective_start
  = max(prev_stage_end, release, target_eat)
  → 실제 시작 가능 시점

- -remaining_pt
  → downstream 길면 우선

- stage_tb

  - 마지막 stage이면 LPT 또는 SPT
  - 아니면 현재 stage processing time

- input_order
  → 최종 tie-breaker

---

### Step 4 — 실제 배정

```
end_time = target_eat + duration
add_ops_times_2_mc(...)
```

---

## 5. Incremental Update (핵심 최적화)

job 하나 배정 후:

- 선택된 machine column만 다시 계산
- 다른 machine column은 그대로 둠

즉,

```
job_2_mc_cache[job][target_mc]만 갱신
```

그리고 각 job에 대해:

- best_mc 재계산
- mc_2_best_eat 재구성

전체 O(n²m) 재계산을 피하는 구조입니다.

---

## 6. 알고리즘의 성격

이 방법은 다음 특성을 가집니다.

- Global machine earliest start 기반
- Machine idle 최소화 반영
- Downstream awareness (remaining PT)
- 마지막 stage에서 LPT/SPT 선택 가능
- precedence + release 자동 처리
- Incremental cache로 계산량 감소

---

## 7. 철학적으로 보면

이 알고리즘은:

- job 중심 heuristic (예: SPT, LPT)
- machine 중심 greedy heuristic

를 결합한 hybrid 방식입니다.

특히 중요한 점은:

> machine이 “언제 일할 수 있는가”를 먼저 결정하고
> 그 시점에 가장 적절한 job을 배정한다는 것

그래서 전형적인 job-dispatching rule과는 다른 방향성을 가집니다.
