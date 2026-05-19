# PW-CP (Prefix-Window CP) 알고리즘

호출: `def pw_cp` (hfs_cp_lns.py)

## 개요

PW-CP는 **슬라이딩 윈도우 기반의 CP 국소 탐색(Local Search)** 알고리즘이다.
기존의 실행 가능한 스케줄(초기해)에서 출발하여, 전체 작업을 시간 순서로
배치(batch)로 묶고, 윈도우를 한 칸씩 이동하면서 윈도우 내 작업만 CP로
재최적화한다. 각 서브문제의 크기가 작아 CP가 빠르게 동작하고, 결과를
누적하면서 전체 makespan을 단계적으로 줄인다.

적용 조건:

- 연속 스테이지(stage)를 순서대로 통과하는 잡숍 계열 문제 (Hybrid Flowshop, Flowshop 등)
- 각 스테이지에 복수의 병렬 기계가 존재 가능
- 목표: makespan 최소화

---

## 핵심 아이디어

전체 작업을 한 번에 CP로 최적화하면 모델이 너무 커진다. 대신:

1. 현재 스케줄 기준으로 작업을 **시간 순서로 정렬**하고 **배치(batch)** 단위로 묶는다.
2. 슬라이딩 윈도우로 전체를 스캔하면서, 윈도우 안의 작업만 CP로 재최적화한다.
3. 윈도우 밖의 작업은 일부는 완전히 고정하고, 일부는 순서만 보존하는 방식으로 제약에 반영한다.
4. CP의 목적함수는 기본적으로 **윈도우 오른쪽 공간(common spacing)을
   최대화**하는 것으로, 이를 통해 작업들을 왼쪽으로 압축시켜 makespan을
   간접적으로 줄인다. 단, 오른쪽 경계(`right_time_fixed`)가 없는
   마지막 윈도우에서는 makespan을 직접 최소화한다.

---

## 작업 분할 (Operation Partition)

각 반복에서, 모든 작업을 5개 영역으로 분류한다.

```text
전체 타임라인:
|← left_time_fixed →|← left_pf →|← UNFIXED →|← right_pf →|← right_time_fixed →|
```

| 영역 | 의미 | CP 모델에서의 역할 |
|------|------|-------------------|
| `left_time_fixed` | 윈도우 왼쪽 끝에서 멀리 떨어진 작업 | 시작 시간 완전 고정 (CP 변수 없음) |
| `left_profile_fixed` | 윈도우 왼쪽 버퍼 | CP 변수 있음, 상대 순서(선행 제약)만 보존 |
| `unfixed` | **현재 윈도우** | CP가 자유롭게 최적화 |
| `right_profile_fixed` | 윈도우 오른쪽 버퍼 | CP 변수 있음, 상대 순서(선행 제약)만 보존 |
| `right_time_fixed` | 윈도우 오른쪽 끝에서 멀리 떨어진 작업 | 시작 시간 완전 고정 (CP 변수 없음) |

`non_time_fixed` = `left_profile_fixed` + `unfixed` + `right_profile_fixed`
→ 이 영역만 CP 변수를 생성한다.

### 윈도우 이동 파라미터

| 파라미터 | 의미 |
|----------|------|
| `batch_size` | 배치당 작업 수 |
| `step_size` | 한 번에 이동하는 배치 수 |
| `unfixed_batch_count` | 윈도우(unfixed) 폭 (배치 수) |
| `left_profile_fixed_batch_count` | 왼쪽 버퍼 폭 |
| `right_profile_fixed_batch_count` | 오른쪽 버퍼 폭 |

---

## 배치 생성 (Batch Building)

각 스테이지에서 작업을 시간 기준으로 정렬한 뒤 `batch_size`씩 묶는다.

정렬 기준:

- `sort_by_start_time=False` (기본): 작업의 중점 시각 `(start + end) / 2` 기준 정렬
- `sort_by_start_time=True`: 시작 시각 기준 정렬

모든 스테이지의 배치 수는 동일해야 한다. 다르면 알고리즘이 중단된다.

---

## CP 서브문제 구성

### 우측-정렬(Right Justification)

윈도우에 `right_time_fixed` 작업이 존재하면, CP 모델 구성 전에 **우측-정렬**을 수행한다.

- `non_left_time_fixed` 작업들을 capacity와 precedence를 지키면서 최대한 늦게 시작하도록 이동
- 목적: 오른쪽 경계(right_boundary)를 명확히 확정하기 위함

```text
우측-정렬 후:
machine 1: [...ltf...]                        [ntf가 오른쪽으로 밀림][rtf]
           0        left_bnd                                   right_bnd  horizon
```

### 윈도우 맵(Window Map)

각 기계별로 CP 모델에서 사용하는 시간 경계를 계산한다.

```text
left_boundary[machine]  = 해당 기계의 left_time_fixed 작업 최대 종료 시각
right_boundary[machine] = 해당 기계의 right_time_fixed 작업 최소 시작 시각
```

### CP 변수

- `op_start[j, i]`, `op_end[j, i]`, `op_intvl[j, i]`: 각 non_time_fixed 작업의 시작, 종료, 인터벌 변수
- 범위: `[head(j,i), horizon - tail(j,i) - p]` (`tighten_ranges=True`인 경우, head/tail은 각 잡의 전후 처리시간 합)
- `common_spacing`: 모든 기계에 공통으로 적용되는 여유 공간 변수 (목적함수)

### 더미 바 변수 (Dummy Bar Variables)

시간-고정 영역의 경계를 용량 제약에 자연스럽게 반영하기 위한 가상의 인터벌 변수다.

**Left dummy bar** (기계별):

```text
[0, left_boundary] 구간을 점유하는 고정 인터벌
```

→ left_time_fixed 영역이 실제로 기계를 점유하고 있음을 누적 제약에 표현

**Right dummy bar** (기계별, common_spacing 최대화 모델에서만 생성):

```text
시작: right_boundary - common_spacing  (변수)
길이: horizon - right_boundary + common_spacing  (변수)
끝:   horizon  (고정)
```

→ `common_spacing`이 클수록 오른쪽 더미 바가 왼쪽으로 확장되어, non_time_fixed 작업들이 그 앞에 배치되도록 강제

makespan 최소화 모델에서는 right dummy bar와 `common_spacing`을 생성하지 않고, left dummy bar만 사용한다.

### 제약 조건

1. **잡 선행 제약 (inter-stage precedence)**
   연속된 두 스테이지 `i → i+1`에 대해:
   - 두 스테이지 모두 non_time_fixed: `end[j,i] ≤ start[j,i+1]`
   - 스테이지 i가 time_fixed, i+1이 non_time_fixed: `start[j,i+1] ≥ end_time_in_rj_sched[j,i]` (필요한 경우에만)
   - 스테이지 i가 non_time_fixed, i+1이 time_fixed: `end[j,i] ≤ start_time_in_rj_sched[j,i+1]` (필요한 경우에만)

2. **용량 제약 (cumulative)**
   각 스테이지에서:

   ```text
   add_cumulative(
       [op_intvl 모음] + [left_dummy bars] + [right_dummy bars],
       [1, 1, ...],
       capacity = 기계 수
   )
   ```

3. **Profile-fixed 선행 제약**
   profile_fixed 작업들의 상대 순서를 CP 변수에 반영:
   - `profile_fix_by_machine=True`: 기계 시퀀스 내 인접 (또는 stride 간격) 선행 제약
   - `profile_fix_by_machine=False`: 스테이지 수준 시간 기반 선행 제약 (후보를 기계 수만큼 제한적으로 선택)

### 목적함수

목적함수는 윈도우에 `right_time_fixed` 작업이 존재하는지에 따라 갈린다.

**common_spacing 최대화 (`right_time_fixed`가 있는 경우)**:

```text
maximize common_spacing
```

→ 윈도우 오른쪽 여유를 극대화 = 비고정 작업들이 왼쪽으로 압축됨 = makespan 간접 감소

**makespan 최소화 (`right_time_fixed`가 없는 경우)**:

```text
minimize makespan = max(op_end[j, last_stage] for j in non_time_fixed)
```

→ 윈도우가 타임라인 끝까지 닿으면 오른쪽 경계를 만들 `right_time_fixed`가
없어, common_spacing 최대화는 horizon에 대고 압축하는 약한 신호만 준다.
이 경우 마지막 스테이지 비고정 작업들의 종료 시각 최대값(= makespan)을
직접 최소화한다. (이 모델은 right dummy bar와 `common_spacing` 변수
자체를 만들지 않으며, 우측-정렬도 수행하지 않는다.)

---

## 스케줄 재구성 (Schedule Reconstruction)

CP가 해를 구하면 다음 순서로 스케줄을 재구성한다.

1. **Left-time-fixed 작업**: 우측-정렬 스케줄에서 그대로 시각을 가져와 지정된 기계에 배정
2. **Non-time-fixed 작업**: CP가 구한 시작 시각 순으로 정렬 후, `add_operation_2_stage`로 기계에 자동 배정 (단, release time = CP 시작 시각)
3. **Right-time-fixed 작업**: 우측-정렬 스케줄에서 시각을 가져오고,
   현재 마지막 종료 시각이 가장 이른(latest-end가 최소인) 기계를 선택하여
   배정 (earliest-free-machine 방식)

---

## 후보 해 수락 기준

```text
if candidate.makespan < incumbent.makespan:
    수락 (feasibility 검사 후 incumbent 갱신)
else:
    기각, 기존 incumbent에 semi-active 변환 적용
```

CP가 현재 makespan을 개선하지 못하면, 다음 윈도우 반복을 위해 기존 incumbent를 semi-active 형태로 정규화한다.

---

## 전체 실행 흐름

```text
입력: 초기 스케줄 (ref_schedule), 인스턴스 파라미터

1. 초기 배치 구성 (build_stage_2_batch_list)
2. 배치 수 검증 (모든 스테이지 동일)

3. for unfixed_batch_start_idx in range(0, max_batch_cnt - unfixed_batch_count + 1, step_size):
   a. 현재 incumbent로 배치 재구성
   b. 5-영역 파티션 계산 (_build_operation_partition)
   c. [optional] enable_promotion_profile_fixed: unfixed 잡의 profile-fixed 작업도 unfixed로 승격
   d. right_time_fixed 있으면 우측-정렬 수행
   e. 윈도우 맵 계산 (_build_window_map)
   f. CP 서브문제 구성 및 풀기
      - right_time_fixed 있음 → common_spacing 최대화
      - right_time_fixed 없음 → makespan 최소화
   g. 스케줄 재구성 + semi-active 변환
   h. 개선 시 incumbent 갱신

4. 결과 반환 (PwCpResult)
```

---

## 파라미터 요약

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `batch_size` | 1 | 배치당 작업 수 |
| `step_size` | 1 | 윈도우 이동 간격 (배치 단위) |
| `unfixed_batch_count` | 1 | 윈도우 폭 |
| `left_profile_fixed_batch_count` | 0 | 왼쪽 버퍼 폭 |
| `right_profile_fixed_batch_count` | 0 | 오른쪽 버퍼 폭 |
| `enable_promotion_profile_fixed` | False | unfixed 잡의 profile-fixed 작업을 unfixed로 승격 |
| `profile_fix_by_machine` | False | True면 기계 시퀀스 기반 선행 제약, False면 스테이지 시간 기반 |
| `machine_precedence_stride` | 1 | `profile_fix_by_machine=True`일 때 stride |
| `max_time_per_batch` | None | 배치별 최대 CP 풀이 시간 (초) |
| `non_time_fixed_op_time_limit_multiplier` | None | non_time_fixed 작업 수 × 배율로 시간 제한 산정 |
| `tighten_ranges` | False | head/tail 기반으로 변수 도메인 압축 |
| `use_lns_only` | False | CP-SAT에서 LNS만 사용 |

---

## 다른 문제에 응용할 때 고려사항

### 필수 요소

- **초기 실행 가능해**: PW-CP는 개선 알고리즘이므로 반드시 초기해가 필요
- **배치 순서화 기준**: 현재는 중점 시각(midpoint)으로 정렬. 문제에 따라 마감시간(due date), 우선순위 등으로 변경 가능
- **CP 모델의 용량/선행 제약**: 대상 문제의 구조에 맞게 재정의 필요

### 확장 포인트

- **목적함수**: common_spacing 최대화 외에, 지연합(sum of tardiness) 등 다른 국소 목적함수로 교체 가능
- **파티션 정책**: 시간 기반 배치 분할 대신, 잡 ID, 우선순위, 클러스터링 등으로 교체 가능
- **수락 기준**: 현재는 순수 그리디(개선 시에만 수락). Simulated Annealing 등의 수락 전략으로 교체 가능
- **윈도우 크기 적응**: 반복 중 풀이 시간이나 개선량에 따라 `unfixed_batch_count`를 동적으로 조절 가능

### 주의사항

- 모든 스테이지에서 배치 수가 동일해야 한다. 스테이지마다 작업 수가 다르면 별도의 배치 정규화가 필요
- right-justification과 common_spacing 최대화는 `right_time_fixed`가
  있을 때만 수행한다. `right_time_fixed`가 없는 윈도우(unfixed 영역이
  타임라인 끝까지 닿은 경우)에서는 makespan을 직접 최소화한다
- profile-fixed 선행 제약의 강도(stride)가 너무 강하면 CP 탐색 공간이 과도하게 제한되어 개선 기회를 놓칠 수 있다
