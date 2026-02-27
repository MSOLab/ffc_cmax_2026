# `dispatch_stage_by_machines_2`

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

## 2. 사전 준비

### 2.1 Input

- `i`: 대상 stage
- `external_job_seq`: Given job sequence
- `external_release`: job ID -> external release time of the job
- `stage_2_job_2_p`: stage ID -> job ID -> processing time

### 2.2 인덱스 및 기본 정보

#### Job 정보

- `r_j[j] := max(prev_stage_end[j], external_release[j])`: Release time
  - Precedence constraint + release constraint를 미리 정리
- `J`: job index(`j`) set (`J=[0,...,n-1]`)
  - `external_job_seq`의 job을 다음의 우선순위로 정렬
    - `r_j[j]` 오름차순
    - `external_job_seq` 순서
- `p_j := stage_2_job_2_p[i][j]`: Processing time for j in J
- `tr_j := \sum_{ip = i+1}^{|I|} stage_2_job_2_p[ip][j]`: Remaining processing time for j in J

### 2.3 States

#### Machine state

- `t_k`: Time cursor of machine k on stage `i`
  - Initialize: `t_k := r_j[0] \forall k\in M_i`
- `t'`: Time cursor for dispatching
  - Initialize: `r_j[0]`

#### Job state

- `J'`: Dispatch 대상 job set
  - Initialize: empty set
- `J''`: 전체 job set - schedule 완료 job set - dispatch 대상 job set
  - Initialize: `J`
- `u`: 다음에 처음으로 `J''`에서 `J'`으로 이동할 job
  - Initialize: 0

---

## 3. Procedure

While $J' \cup J'' \neq \emptyset$ :

### 3.1 Update $J'$

- If $t' \geq \min_{j\in J''} r_j$:
  - $v := \max(j\in J'' | r_j \leq t')$
  - $J' \leftarrow J' \cup \{ u,...,v \}$
  - $J'' \leftarrow J'' \setminus \{ u,...,v \}$
  - $u \leftarrow v + 1$
- Else:
  - If $J' = \emptyset$:
    - $t' := \min_{j\in J''} \{ r_j \}$
    - $t_k \leftarrow max(t_k, t') \forall k$
    - Continue(Skip 3.2)

### 3.2 Dispatch

- $k' = \argmin_k \{ t_k \}$
  - Tie-breaking: (1) idle 최소 (2) machine index 최소
- $j' = \argmin_{j\in J'} \{ p_j + tr_j \}$
  - Tie-breaking: (1) $p_j$ 최소 (2) $J$ 상 앞순서
- Target machine에 target job dispatch
- State update
  - $J' \leftarrow J' \setminus \{ j' \}$
  - $t_{k'} \leftarrow$ dispatch한 operation의 end time
  - $t' := \min \{t_k \}$
