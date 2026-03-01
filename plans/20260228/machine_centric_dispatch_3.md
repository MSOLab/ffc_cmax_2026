# `dispatch_stage_by_machines_3`

**하나의 stage에서 여러 job을 동시에 고려하여, “machine 중심”으로 순차적으로 배정해 나가는 디스패칭 알고리즘**입니다.

구현은 `HybridFlowshopLiteSchedule` 내부에 있습니다.
`dispatch_stage_by_machines_2`은 빈 stage에 dispatching한다는 가정이 있었으며,
`dispatch_stage_by_machines_3`는 이에 더해, 이미 schedule된 operation이 있는 경우를 고려합니다.
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

- `i`: 대상 stage (index: `i_idx`)
- `external_job_seq`: Given job sequence
- `external_release`: job ID -> external release time of the job
- `stage_2_job_2_p`: stage ID -> job ID -> processing time
- `spt_on_last_stage`: Boolean. 마지막 stage에서 SPT 적용 여부 (False이면 LPT)

### 2.2 인덱스 및 기본 정보

#### Job 정보

- `r_j[j] := max(prev_stage_end[j], external_release[j])`: Release time
  - Precedence constraint + release constraint를 미리 정리
- `J`: job index(`j`) set (`J=[0,...,n-1]`)
  - `external_job_seq`의 job을 다음의 우선순위로 정렬
    - `r_j[j]` 오름차순
    - `external_job_seq` 순서
- `p_j := stage_2_job_2_p[i][j]`: Processing time for `j` in `J`
  - `p_max := max(p_j)`: Maximum processing time on stage `i`
- `tr_j := \sum_{ip = i+1}^{|I|} stage_2_job_2_p[ip][j]`: Remaining processing time for `j` in `J`

#### Machine 정보

- `k`: 대상 stage의 machine ID

#### 스케줄링 상수

- `c := |I|`: 전체 stage 수
- `p_multiplier := -(c - i_{idx} - 2) \times c \,/\, 80`
  - 현재 stage 위치에 따라 `p_j`에 부여되는 가중치
  - stage가 앞쪽일수록 음수 크기가 커져 `p_j`의 영향이 커짐
  - 마지막 stage 직전(`i_{idx} = c-2`)에서 0, 그 이후는 양수

### 2.3 States

#### Machine state

- `t'`: Time cursor for dispatching
  - Initialize: $r_0$
- `t_k`: Time cursor of machine k on stage `i`
  - Initialize: $t_k := r_0 \forall k\in M_i$
- `[(l_k0, P_k0), (l_k1, P_k1), ...]`: machine gap queue
  - $l_{kg}$: g번째 idle gap의 길이
  - $P_{kg}$: g번째 idle gap 직후 연속 processing time 합
  - 마지막 element는 (∞, 0)
    - `n*p_max`를 ∞ 대신 사용
  - Initialize: from given stage schedule

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

- If $J'' \neq \emptyset$:
  - If $t' \geq r_u$:
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
- $J'_{k'} := \{ j\in J' | p_j \leq l_{k'g} \}$
- If $J'_{k'}$ is empty: State update & continue
  - $t_{k'} \leftarrow t_{k'} + l_{k'0} + P_{k'0}$
  - $t' := \min \{t_k \}$
  - $l_{k'0} \leftarrow l_{k'1}, P_{k'0} \leftarrow P_{k'1}$
  - Continue(Skip remaining)
- $j' = \argmin_{j\in J'_{k'}} \text{sort\_key}(j)$
  - 아래 3.3 참조
- Target machine에 target job dispatch
  - $\text{start} := t_{k'}$
  - $\text{end} := t_{k'} + p_{j'}$
- State update
  - $J' \leftarrow J' \setminus \{ j' \}$
  - $t_{k'} \leftarrow t_{k'} + p_{j'}$
  - $t' := \min \{t_k \}$
  - $l_{k'0} \leftarrow l_{k'0} - p_{j'}$

### 3.3 Job Selection Key

Job의 선택 우선순위는 다음 sort key를 **오름차순**으로 정렬하여 결정합니다.

$$
\text{sort\_key}(j) = \Bigl(-(tr_j + \alpha \cdot p_j),\;\; \beta \cdot p_j,\;\; j \Bigr)
$$

각 항의 의미:

**1st key** $-(tr_j + \alpha \cdot p_j)$:

Remaining processing time `tr_j`에 현재 stage의 `p_j`를 `α` 가중치로 더한 값의 **음수**. 즉 이 값이 클수록 우선 배정.

$$
\alpha = p\_multiplier = -\frac{(c - i_{idx} - 2) \times c}{80}
$$

- `α < 0` (stage가 앞쪽): `p_j`가 클수록 1st key 감소 → 짧은 job 우선 (SPT 경향)
- `α = 0` (마지막 stage 직전): `tr_j`만 반영
- `α > 0` (마지막 stage): `p_j`가 클수록 1st key 증가 → 긴 job 우선 (LPT 경향)

**2nd key** $\beta \cdot p_j$ (Tie-breaking):

- 마지막 stage가 아닌 경우: $\beta = 1$ → `p_j` 오름차순 (SPT)
- 마지막 stage인 경우: `spt_on_last_stage=True`이면 $\beta = 1$ (SPT), `False`이면 $\beta = -1$ (LPT)

**3rd key** $j$ (Tie-breaking):

`J` 상 앞순서 (입력 sequence 기준 앞에 있는 job 우선)
