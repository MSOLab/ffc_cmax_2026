# SW-CP Constructor (SwCpConstructor.run)

호출: `def SwCpConstructor.run` (sw_cp.py)

## 개요

SwCpConstructor.run은 슬라이딩 윈도우 기반 CP 국소 탐색(Local Search) 알고리즘인
SW-CP의 **핵심 구현체**다. 초기 실행 가능 스케줄을 입력받아, 시간 순서로 정렬된
배치(batch) 위로 윈도우를 슬라이딩시키며 각 윈도우 내 작업을 CP 서브문제로
풀어 makespan을 단계적으로 개선한다.

전체 알고리즘 개념(`sw_cp.md`)과 동일하나, 이 문서는 SwCpConstructor.run의
**구체적인 실행 흐름 — 반복 루프 구성, CP 모델 분기, 상태 관리, 배치별 시간
제한 결정** — 을 구현 수준에서 서술한다.

---

## 문제 설명

### 파라미터 (Parameters)

다섯 부류로 나뉜다:

- **윈도우 구조 계열**: `batch_size`, `step_size`, `unfixed_batch_count`,
  `left_profile_fixed_batch_count`, `right_profile_fixed_batch_count`,
  `enable_promotion_profile_fixed` — 슬라이딩 윈도우의 형태와 이동 방식을 결정한다.
- **제약 강도 계열**: `profile_fix_by_machine`, `machine_precedence_stride`,
  `stage_precedence_min_processing_time_diff`,
  `stage_precedence_min_processing_time_diff_ratio` — profile-fixed 작업의
  선행 제약 강도를 조절한다.
- **시간 예산 계열**: `non_time_fixed_op_time_limit_multiplier`,
  `max_time_per_batch` — 배치당 CP 풀이 시간을 결정한다 (둘 중 하나만 사용).
- **CP-SAT 제어 계열**: `solver_thread_cnt`, `use_lns_only`,
  `tighten_ranges` — CP-SAT 솔버의 동작을 제어한다.
- **디버깅/검증 계열**: `debug_export`, `error_if_infeasible` — 디버깅 출력과
  최종 해 검사를 제어한다.

전체 목록과 기본값은 「파라미터 요약」 표를 참조.

### 변수 (Variables)

알고리즘은 두 계층의 변수를 관리한다:

**SwCpRunState (반복 상태)**:
- `timer` (`ElapsedTimer`): 총 경과 시간 추적
- `incumbent` (`HybridFlowshopLiteSchedule`): 현재 최선 스케줄
- `subproblem_idx` (`int`): 전체 서브문제 번호 (1부터 시작, 루프 전체에서 단조 증가)
- `subproblem_logs` (`list[SwCpSubproblemLog]`): 각 서브문제의 결과 로그
- `sub_obj_store` (`ObjValueBoundStore[int]`): 개선 시 obj value 이력 저장
- `max_time_per_batch` (`float | None`): `max_time_per_batch` 파라미터의 패스스루 값 (multiplier 사용 시 `None`). 결과 메타데이터용이며, 실제 배치별 제한은 `_resolve_batch_time_limit`이 산정

**CP 모델 변수 (서브문제별)**:
- `op_start[j, i]`, `op_end[j, i]`, `op_intvl[j, i]`: 각 `non_time_fixed` 작업의
  시작·종료·인터벌 결정변수. 도메인은 `[head(j,i), horizon - tail(j,i) - p]`
  (`tighten_ranges=True`인 경우 head/tail 적용).
- `left_bar_interval[i][mc]` / `left_bar_end[i][mc]`: left-time-fixed 영역의
  기계 점유를 표현하는 고정 인터벌 (시작=0, 길이=left_boundary).
- `right_bar_interval[i][mc]`: right-time-fixed 영역 직전의 여유 공간 변수
  (시작=right_boundary - common_spacing, 끝=horizon). **makespan 최소화 모델에서는
  생성하지 않음**.
- `common_spacing` (`IntVar | None`): 모든 기계에 공통으로 적용되는 여유 변수.
  **makespan 최소화 모델에서는 `None`**.
- `makespan` (`IntVar | None`): makespan 최소화 모델에서만 생성.

### 목적 (Objective)

윈도우에 `right_time_fixed` 작업(오른쪽 경계) 존재 여부에 따라 목적함수가 분기한다:

- **right_time_fixed 있음 (common_spacing 최대화)**: `maximize common_spacing`.
  오른쪽 경계가 존재하므로, 그 앞에 여유 공간을 최대화하면 비고정 작업들이
  자연히 왼쪽으로 밀려 makespan이 간접 감소한다.
- **right_time_fixed 없음 (makespan 최소화)**: `minimize makespan`.
  윈도우가 타임라인 끝까지 닿아 common_spacing 최대화가 의미 없으므로,
  마지막 스테이지의 비고정 작업 종료 시각 중 최대값을 직접 최소화한다.

### 제약 (Constraints)

세 가지 제약군이 CP 서브문제의 해 공간을 정의한다:

1. **잡 선행 제약 (inter-stage job precedence)**:
   연속한 두 스테이지 `i → i+1`에 대해:
   - 두 스테이지 모두 non_time_fixed: `end[j,i] ≤ start[j,i+1]`
   - i가 time_fixed, i+1이 non_time_fixed: `start[j,i+1] ≥` i의 종료 시각
     (right-justified 스케줄 기준, 단 `i_end > next_i_est`인 경우만)
   - i가 non_time_fixed, i+1이 time_fixed: `end[j,i] ≤` i+1의 시작 시각
     (right-justified 스케줄 기준, 단 `next_i_start < i_lct`인 경우만)

2. **용량 제약 (stage capacity via cumulative)**:
   ```
   add_cumulative(intervals=[op_intvl] + [left_dummy_bars] + [right_dummy_bars],
                  demands=[1, 1, ..., 1],
                  capacity=len(M_of[i]))
   ```
   각 스테이지를 병렬 기계 수만큼의 용량을 가진 단일 자원으로 모델링한다.
   더미 바(dummy bar)가 시간-고정 영역의 기계 점유를 자연스럽게 표현한다.

3. **Profile-fixed 선행 제약 (선택사항)**:
   `profile_fixed_op_set`이 존재할 때만 활성화. profile-fixed 작업들이
   right-justified 스케줄에서 가진 상대 순서(기계 시퀀스 또는 스테이지
   시간 순서)를 CP 결정변수에 선행 제약으로 반영한다.

---

## 핵심 아이디어

1. **분할-정복을 CP에 적용**: 전체 문제를 한 번에 CP로 푸는 대신, 시간 축의
   작은 윈도우만 재최적화한다. 각 서브문제는 작업 수가 적어 CP가 제한된
   시간 내에 해결 가능하다.
2. **슬라이딩 윈도우로 전체 커버**: 윈도우가 시작부터 끝까지 이동하며 모든
   작업 구간을 차례로 최적화한다. 구간을 조금씩 겹쳐(`step_size <
   unfixed_batch_count`) 연속성을 확보한다.
3. **5-영역 파티션으로 경계 처리**: 한 윈도우만 풀면 경계 조건이 깨지므로,
   시간-고정(left/right_time_fixed)과 순서-보존(left/right_profile_fixed)
   영역으로 완충하여 윈도우가 전체 맥락에서 동작하게 한다.
4. **두 가지 목적함수로 상황 대응**: 오른쪽에 고정 경계가 있으면
   common_spacing 최대화로 간접 최적화하고, 경계가 없으면 makespan을 직접
   최소화한다.

---

## 슬라이딩 윈도우 반복 루프

### 배치 목록 구성

`build_stage_2_batch_list`가 현재 `incumbent` 스케줄의 각 스테이지에서 작업을
시간 순서로 정렬하여 `batch_size`만큼 묶는다.

정렬 기준 (모두 오름차순):
- 기본: `(midpoint, start_time, machine_id, job_id)` — midpoint = `(start+end)/2`
- `sort_by_start_time=True`: `(start_time, end_time, machine_id, job_id)`

모든 스테이지의 배치 수는 `validate_and_get_batch_count`에서 동일해야 한다는
불변식을 검증한다. 다르면 `ValueError`로 중단된다.

### 윈도우 파라미터 결졍

```
max_batch_cnt = 스테이지별 배치 수 (모두 동일)
max_window_start = max_batch_cnt - unfixed_batch_count
반복: unfixed_batch_start_idx in range(0, max_window_start + 1, step_size)
```

`max_window_start`는 unfixed 영역이 전체 배치를 초과하지 않도록 보장한다.
`step_size`가 `unfixed_batch_count`보다 작으면 윈도우가 겹친다.

### 5-영역 파티셔닝

`_build_operation_partition`이 각 스테이지의 `batch_list_on_stage`를 배치
인덱스 기준으로 5영역으로 분할한다:

```text
                   left_pf_start         unfixed_start     unfixed_end    right_pf_end
                        │                      │                │              │
idx:  0 ... left_pf_start-1 | left_pf_start... | unfixed_start... | right_pf_end... | max
      ← left_time_fixed →     ← left_pf →        ← unfixed →      ← right_pf →    ← right_time_fixed →
```

- `left_pf_start = unfixed_batch_start_idx - left_profile_fixed_batch_count`
- `right_pf_end = unfixed_batch_start_idx + unfixed_batch_count + right_profile_fixed_batch_count`

각 영역의 작업은 `sorted`로 정규화된다.

### Promotion (선택)

`enable_promotion_profile_fixed=True`면, unfixed 영역에 속한 잡의
profile-fixed 작업들을 unfixed로 승격시킨다. 같은 잡의 모든 작업이 unfixed
영역에서 처리되도록 하여, 잡 단위의 일관성을 높인다.

### 배치 명세 구성

`_build_batch_spec`이 5-영역 파티션을 바탕으로 CP 서브문제의 초기 스케줄과
기계별 시간 윈도우를 준비한다:

- **`right_time_fixed`가 존재하면**: `incumbent`를 deepcopy한 후,
  `non_left_time_fixed` 작업들을 우측-정렬(right-justify)하여 오른쪽 경계를
  명확히 한다. 그 다음 `_build_window_map`으로 기계별
  `(left_boundary, right_boundary)` 계산.
- **`right_time_fixed`가 없으면**: 우측-정렬 없이 곧바로 윈도우 맵 계산
  (right_boundary는 `horizon`으로 설정).

### 배치별 시간 제한 결정

`_resolve_batch_time_limit`은 두 가지 방식 중 하나로 시간 제한을 산정한다:

- `non_time_fixed_op_time_limit_multiplier`가 설정됨:
  `non_time_fixed_op_count × multiplier`
- 그 외: `max_time_per_batch` 값을 그대로 사용

### CP 모델 분기

`spec.is_right_time_fixed_empty`에 따라 solve 함수가 갈린다:

- True (right_time_fixed 없음) → `_solve_makespan_batch`
- False (right_time_fixed 있음) → `_solve_batch_sw_cp_model`

---

## CP 모델 준비의 두 분기

### Common Spacing 최대화 모델 (`_solve_batch_sw_cp_model`)

`right_time_fixed`가 존재하는 (비-마지막) 윈도우에 사용된다.

`_prepare_sw_cp_model`이 다음을 구성한다:

1. **Non-time-fixed 변수**: `SwCpModelBuilder.make_non_time_fixed_ops_vars`
2. **더미 바 변수**: `SwCpModelBuilder.make_dummy_bar_vars`
   - Left dummy bar (고정): 각 기계의 `[0, left_boundary]` 구간 점유
   - Right dummy bar (가변): `[right_boundary - common_spacing, horizon]`
     구간 점유. `common_spacing`이 클수록 왼쪽으로 확장된다.
3. **잡 선행 제약**: `SwCpModelBuilder.add_non_fixed_job_precedence_constraints`
4. **용량 제약**: `SwCpModelBuilder.add_capacity_with_dummy_bar_constraints`
5. **Hint 적용**: right-justified 스케줄의 start/end 시각으로 hint 설정
6. **Profile-fixed 선행 제약**: profile_fixed 작업이 있으면 추가
7. **목적함수**: `maximize common_spacing`

### Makespan 최소화 모델 (`_solve_makespan_batch`)

`right_time_fixed`가 없는 (마지막) 윈도우에 사용된다.

`_prepare_makespan_batch_model`이 위와 유사하지만 다음이 다르다:

- **Right dummy bar 생성하지 않음** (`include_right_bars=False`)
- **마지막 스테이지 검증**: 마지막 스테이지에 `non_time_fixed` 작업이
  하나 이상 있어야 함. 없으면 `ValueError`
- **목적함수**: `minimize makespan = max(op_end[j, last_stage])`
  → `SwCpModelBuilder.add_makespan_objective`

### 공통 해결 과정

두 분기 모두 동일한 패턴으로 풀이한다:

1. `ctx.get_remaining_time_limit(max_time_per_batch)`로 남은 시간 예산 확인
2. `ctx.solve_cp_model_2(...)`로 CP-SAT 호출
3. infeasible이면 `None` 반환하고 로그 기록
4. feasible이면 `create_sw_cp_schedule(...)`로 CP 해에서
   `HybridFlowshopLiteSchedule` 재구성
5. `candidate_schedule.make_semi_active(stage_2_job_2_p_dict)`로 semi-active 정규화

### 스케줄 재구성 (create_sw_cp_schedule)

CP 해로부터의 스케줄 재구성은 세 단계로 진행된다 (자세한 내용은
`sw_cp.md`의 「스케줄 재구성」 참조):

1. **Left-time-fixed 작업**: right-justified 스케줄에서 그대로 시각과 기계를 가져와 배정
2. **Non-time-fixed 작업**: CP가 구한 시작 시각 순으로 정렬 후 `add_operation_2_stage`로
   기계에 자동 배정 (release time = CP 시작 시각)
3. **Right-time-fixed 작업**: right-justified 스케줄에서 시각을 가져오고,
   가장 이른 기계(latest-end 최소)를 선택해 배정 (earliest-free-machine 방식)

---

## 후보해 수락 및 상태 관리

### 수락 기준

```text
candidate가 None이 아니고 makespan이 incumbent보다 작음:
  → feasibility 검사 통과 시 수락 (incumbent 갱신, True 반환)
  → feasibility 검사 실패 시 로그만 남기고 기각
그 외:
  → 기각, incumbent에 semi-active 변환 적용 (False 반환)
```

`feasibility` 검사 예외는 로깅만 수행할 뿐 알고리즘을 중단시키지 않는다.

### 상태 관리

- `SwCpRunState` 인스턴스가 `self._st`에 저장되며, `run()` 입장 시 생성되고
  `finally` 블록에서 `None`으로 초기화된다.
- `_require_state()`로 현재 상태에 접근하며, `run()`이 활성화되지 않았으면
  `RuntimeError`를 발생시킨다.
- 각 서브문제 해결 후 `_append_subproblem_log`로 로그를 기록한다.
- `sub_obj_store`에는 수락된 배치의 obj value만 기록된다 (개선이 있었을 때만).

### Debug Export (선택)

`debug_export=True`면:
- 초기 스케줄 Gantt 차트 PNG 저장 (`_batch_000_initial_gantt.png`)
- 각 반복 후 solution YAML 저장 (`_batch_NNN_solution.yaml`)

---

## 전체 실행 흐름

```text
입력: ref_schedule (초기해), instance (문제 인스턴스),
      stage_2_job_2_p_dict (가공 시간 맵), 각종 파라미터

 1. 입력 검증 (step_size >= 1, unfixed_batch_count >= 1, profile_fixed_batch_count >= 0 등)
 2. SwCpRunState 초기화 (timer, incumbent=ref_schedule, ...)
 3. [debug] 초기 Gantt 차트 저장 (debug_export=True)
 4. 초기 배치 목록 구성 (build_stage_2_batch_list)
 5. 배치 수 검증 (모든 스테이지 동일) → max_batch_cnt
 6. max_window_start = max_batch_cnt - unfixed_batch_count
 7. for unfixed_batch_start_idx in range(0, max_window_start + 1, step_size):
      a. 현재 incumbent로 배치 목록 재구성
      b. 배치 수 불변식 검증 (초기와 동일한지)
      c. for each stage_id:
           _build_operation_partition(...) → 5-영역 파티션
      d. [선택] enable_promotion_profile_fixed:
           unfixed 잡의 profile-fixed 작업을 unfixed로 승격
      e. _build_batch_spec(...) → SwCpSubproblemSpec
           - right_time_fixed 존재 → 우측-정렬 → 윈도우 맵
           - right_time_fixed 없음 → 윈도우 맵만
      f. _resolve_batch_time_limit(...) → 배치당 시간 제한
      g. if spec.is_right_time_fixed_empty:
           _solve_makespan_batch(...)  → candidate | None
         else:
           _solve_batch_sw_cp_model(...) → candidate | None
      h. _accept_candidate_or_repair_incumbent(candidate, incumbent, ...)
           → (new_incumbent, accepted)
      i. [debug] solution YAML 저장
      j. [accepted] sub_obj_store에 obj value 기록
 8. [error_if_infeasible] 최종 incumbent feasibility 검증
 9. SwCpResult 반환 (schedule, sub_obj_store, subproblem_logs, ...)

최종: self._st = None (finally 블록)
```

---

## 파라미터 요약

### 윈도우 구조

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `batch_size` | 1 | 배치당 최대 작업 수 |
| `step_size` | 1 | 윈도우 이동 간격 (배치 단위) |
| `unfixed_batch_count` | 1 | unfixed 영역의 배치 수 (윈도우 폭) |
| `left_profile_fixed_batch_count` | 0 | 왼쪽 profile-fixed 버퍼 배치 수 |
| `right_profile_fixed_batch_count` | 0 | 오른쪽 profile-fixed 버퍼 배치 수 |
| `enable_promotion_profile_fixed` | False | unfixed 잡의 profile-fixed 작업을 unfixed로 승격 |

### 제약 강도

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `profile_fix_by_machine` | False | True면 기계 시퀀스 기반 선행 제약, False면 스테이지 시간 기반 |
| `machine_precedence_stride` | 1 | 기계 선행 제약의 stride 간격 |
| `stage_precedence_min_processing_time_diff` | None | 스테이지 선행 제약의 최소 가공 시간 차이 (≥0) |
| `stage_precedence_min_processing_time_diff_ratio` | None | 스테이지 선행 제약의 최소 가공 시간 차이 비율 (≥0) |

### 시간 예산

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `non_time_fixed_op_time_limit_multiplier` | None | non_time_fixed 작업 수 × 이 배율로 배치당 시간 제한 산정. `max_time_per_batch`보다 우선 |
| `max_time_per_batch` | None | 배치당 최대 CP 풀이 시간 (초). multiplier 미설정 시 사용 |

### CP-SAT 제어

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `solver_thread_cnt` | None | CP-SAT 스레드 수 (None → 1) |
| `use_lns_only` | False | True면 CP-SAT에서 LNS 탐색만 사용 |
| `tighten_ranges` | False | True면 head/tail 기반으로 CP 변수 도메인 압축 |

### 디버깅/검증

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `debug_export` | False | True면 각 반복의 Gantt 차트·solution YAML 저장 |
| `error_if_infeasible` | False | True면 최종 해에 대해 feasibility 검증 수행, 실패 시 예외 발생 |

---

## 주의사항 및 응용 고려사항

### 전제 조건
- 반드시 실행 가능한 초기 스케줄(`ref_schedule`)이 필요하다. SW-CP는 개선
  알고리즘이므로 초기해 없이 동작할 수 없다.
- 모든 스테이지의 배치 수가 동일해야 한다. 스테이지별 작업 수 차이가 크면
  빈 배치가 발생하거나 배치 수 불일치로 오류가 발생할 수 있다.

### 두 가지 시간 제한 방식의 관계
- `non_time_fixed_op_time_limit_multiplier`가 설정되면 `max_time_per_batch`는
  `SwCpRunState.max_time_per_batch`로만 전달될 뿐 실제 시간 제한 계산에는
  사용되지 않는다. 즉, **두 방식은 상호 배타적**이다.
- 시간 제한은 `ctx.get_remaining_time_limit(...)`을 통해 전체 남은 시간 예산
  내로 클리핑된다.

### 확장 포인트
- **목적함수 교체**: common_spacing 최대화 대신 다른 국소 목적(예: 지연 최소화)으로
  교체하려면 `_prepare_sw_cp_model`과 `_prepare_makespan_batch_model`의
  objective 부분을 수정한다.
- **파티션 정책 변경**: `_build_operation_partition`의 인덱스 기반 분할 대신,
  다른 기준(우선순위, 클러스터 등)으로 교체 가능하다.
- **수락 기준**: 현재는 순수 개선 시에만 수락하는 그리디 방식.
  Simulated Annealing이나 Tabu Search 방식으로 교체 가능하다.

### 함정
- `unfixed_batch_count`가 너무 크면 CP 서브문제가 커져 시간 제한 내에 풀리지
  않을 수 있다. 너무 작으면 국소적 개선에 그쳐 전역 탐색 효과가 떨어진다.
- `step_size >= unfixed_batch_count`이면 윈도우가 겹치지 않아 연속성이
  사라진다. 일반적으로 `step_size < unfixed_batch_count`로 설정한다.
- `promote_job_contained_ops`는 같은 잡의 profile-fixed 작업을 unfixed로
  승격하지만, 그로 인해 서브문제 크기가 증가할 수 있다.
