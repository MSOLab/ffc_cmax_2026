# NEH-CP (NEH with CP Subproblem) 알고리즘

호출: `def run` (hybridflowshop/controller/neh_cp.py, `NehCpConstructor` 클래스)

## 개요

NEH-CP는 NEH(Nawaz-Enscore-Ham) 구성적 휴리스틱의 원리를 Hybrid Flowshop에
맞게 확장한 알고리즘이다. 기준 스케줄(reference schedule)에서 출발하여, 작업을
일정한 배치(batch)로 나누어 순차적으로 추가한다. 각 배치 단계에서:
(1) `MixedDispatcher`로 새 작업을 디스패치하고, (2) 지금까지 추가된 전체
작업을 대상으로 작은 CP 모델을 풀어 최적화한다. NEH가 한 번에 한 작업을
최적 위치에 삽입하는 것과 달리, NEH-CP는 배치 단위로 작업을 추가하고 CP
서브문제로 한꺼번에 재최적화한다.

적용 조건:
- 복수 스테이지, 복수 병렬 기계가 있는 Hybrid Flowshop ("identical parallel machine" 포함)
- 초기 실행 가능해가 주어질 것 (개선/재구성 알고리즘)
- 기본 목표: makespan 최소화

---

## 문제 설명

### 파라미터 (Parameters)

알고리즘을 제어하는 입력 설정값은 다음 부류로 나뉜다.

**작업 순서 및 배치 제어**: 어떤 순서로 작업을 추가할지, 배치를 어떻게 나눌지
결정한다. 전체 목록은 아래 「파라미터 요약」 표를 참조.

**시간 예산 계열**: CP 서브문제당 시간 제한(`max_time_per_add`, 또는
`cp_tl_nc_multiplier` / `cp_tl_c_multiplier`로 간접 산정)과, 2차 목적함수
사용 시 별도 시간 제한(`cp_tl_nc_multiplier_2nd_obj` /
`cp_tl_c_multiplier_2nd_obj`로 산정되는 내부 변수 `max_time_per_add_2nd_obj`)이
있다.

**시간 가드(Time Guard) 계열**: 남은 시간을 추정하여 알고리즘의 조기 종료를
결정하는 파라미터들이다. 남은 배치 시간 추정치가 실제 남은 시간을 초과하면
진행을 중단하고 지금까지의 최선 해를 반환한다.

**CP 모델 구성 계열**: 프로파일 고정(profile fixing)의 강도와 방식, 변수
도메인 압축(tighten_ranges), 잡 완료 링크 등 CP 모델의 세부 구성을 제어한다.

**목적함수 계열**: 기본 makespan 최소화 외에 2차 목적(`minimize_sum_ci_lex`
= lexicographic sum C_i 최소화, `minimize_sum_ci_lin` = linear combined)을
지원한다.

**솔버 및 검증 계열**: `solver_thread_cnt`, `use_lns_only`,
`error_if_infeasible` 등 CP-SAT 동작과 최종 검증을 제어한다.

### 변수 (Variables)

알고리즘의 핵심 실행 상태는 `NehCpRunState` dataclass로 추적한다.

| 필드 | 타입 | 의미 |
|------|------|------|
| `timer` | `ElapsedTimer` | 전체 경과 시간 측정기 |
| `last_job_id_list` | `list[str]` | 이전 배치에서 처리한 작업 목록 |
| `partial_sol` | `HybridFlowshopLiteSchedule \| None` | 현재까지 추가된 작업만의 부분 스케줄 (CP 서브문제의 기준) |
| `current_job_id_list` | `list[str]` | 지금까지 추가된 전체 작업 목록 |
| `full_sol` | `HybridFlowshopLiteSchedule` | 현재 full schedule (partial + 미추가 작업을 dispatch로 채움) |
| `job_2_inserted_batch_idx` | `dict[str, int]` | 각 작업이 몇 번째 배치에서 삽입되었는지 기록 |

이와 별도로 `run()` 메서드는 다음 지역변수로 최선 해와 시간 추적 정보를 관리한다
(`NehCpRunState` 필드가 아님).

| 변수 | 타입 | 의미 |
|------|------|------|
| `best_full_sol` | `HybridFlowshopLiteSchedule` | 지금까지 찾은 최선의 full schedule |
| `best_full_obj` | `int` | `best_full_sol`의 makespan |
| `batch_elapsed_sec_list` | `list[float]` | 각 배치의 소요 시간 기록 (시간 가드 추정에 사용) |

CP 서브문제의 결정변수는 `create_instance_of_job_subset()`으로 생성된
부분 인스턴스(sub-instance)에 대해 `BaseModelBuilder.build()`가 생성한다:

- `op_start[j,i]`, `op_end[j,i]`, `op_intvl[j,i]`: 시작·종료·인터벌 변수
- `makespan`: 최대 종료 시각 변수

### 목적 (Objective)

**1차 목적 (기본): makespan 최소화**

```text
minimize max(op_end[j, last_stage] for j in current_job_id_list)
```

CP 서브문제의 목적은 현재 배치까지 포함된 부분 스케줄의 makespan을 최소화하는
것이다. 전체가 아닌 부분 작업 집합에 대한 최적화이므로, 실제 full schedule의
makespan 개선으로 이어질 수 있도록 `_add_hints_and_additional_constraints`에서
미추가 작업의 프로파일(순서)을 보존한다.

**2차 목적 (조건부): Sum C_i 최소화**

`minimize_sum_ci_lex=True`인 경우, makespan 최소화 CP가 종료된 후 **동일한
makespan을 유지하면서** sum of completion times(`sum(C_i)`)을 최소화하는
2차 CP 모델을 순차적으로 푼다. 이는 lexicographic 최적화에 해당한다.

`minimize_sum_ci_lin=True`인 경우, makespan과 sum C_i를 하나의 선형 결합
목적함수로 동시에 최적화한다(`BaseModelBuilder.build()`의
`minimize_makespan_plus_sum_other_stages` 파라미터).

### 제약 (Constraints)

CP 서브문제의 제약은 `BaseModelBuilder.build()`가 생성하는 구조적 제약과
`_add_hints_and_additional_constraints`가 추가하는 프로파일 고정 제약으로
나뉜다.

**1. 구조적 제약 (BaseModelBuilder)**

- **스테이지 간 선행 제약 (inter-stage precedence)**: `end[j,i] ≤ start[j,i+1]`
- **용량 제약 (cumulative)**: 각 스테이지의 병렬 기계 수 제약
- **목적함수 정의**: makespan 변수와 sum C_i 변수 연결

**2. 프로파일 고정 제약 (_add_hints_and_additional_constraints)**

이전 배치 단계의 partial solution에서 추출한 작업 순서를 현재 CP 모델에
반영한다. `batch_idx`가 `profile_fix_min_batch_idx` 미만이면 스킵된다.

제약 모드는 `profile_fix_by_machine`으로 결정된다:

- `profile_fix_by_machine=True`: **기계별 선행 제약**. 각 기계 내에서 이전
  partial solution의 작업 순서(시작 시각 기준)를 stride 간격으로 선행
  제약으로 추가. `machine_precedence_stride`로 간격 조절 가능.
  `profile_fix_by_machine_from_batch_idx`가 설정된 경우, 해당 배치부터
  machine 모드로 전환된다.

- `profile_fix_by_machine=False`: **스테이지별 선행 제약**.
  `add_stage_ops_precedence_constraints_after_dispatch_from_schedule()`가
  스테이지 수준의 sparse precedence 제약을 생성한다.

**3. 선택적 제약**

- `tighten_ranges=True`: head/tail 기반 변수 도메인 압축
- `link_job_completion=True`: 작업 완료 시각과 관련된 추가 연결 제약
- `stage_precedence_min_processing_time_diff` / `_ratio`: 선행 제약에
  최소 처리 시간 차이 조건 추가

**4. Hint (soft guidance)**

`BaseModelBuilder.apply_start_hints_from_start_time_map()` 및
`apply_end_hints_from_end_time_map()`으로 partial solution의 시작/종료
시각을 hint로 제공. 이는 hard 제약이 아니라 CP-SAT의 Hint 메커니즘으로
탐색 방향을 안내한다.

---

## 핵심 아이디어

1. **점진적 구성 + CP 최적화의 결합**: NEH는 한 번에 한 작업을 삽입하지만,
   NEH-CP는 배치 단위로 작업을 추가한 후 CP로 한꺼번에 재최적화한다.
   CP 서브문제가 작아 빠르게 풀리면서도, 배치 내 작업들의 상호작용을
   고려할 수 있다.

2. **부분 재구성 (Partial Reconstruction)**: `preserved_head_job_portion`을
   통해 초기 스케줄의 앞부분을 보존하고 뒷부분만 재구성할 수 있다. 이는
   이미 좋은 앞부분을 유지하면서 뒷부분만 개선하는 전략이다.

3. **이중 경로 평가**: 각 배치 단계에서 (a) partial solution (CP 최적화
   결과)의 makespan을 bound(하한 추정치)로 기록하고, (b) 미추가 작업을
   `MixedDispatcher`로 dispatch한 full schedule의 makespan을 value(실제
   목적값)로 기록한다. 이를 통해 부분 최적화가 전체에 미치는 영향을 추적한다.

4. **적응형 시간 가드**: 배치별 실제 소요 시간을 추적하여, 남은 시간으로
   나머지 배치를 완료할 수 없을 것으로 예상되면 조기 종료한다. safety
   factor를 통해 보수적으로 추정한다.

5. **프로파일 고정(Profile Fixing)**: 이전 단계에서 결정된 작업 순서를
   현재 CP 모델에 hard constraint로 점진적으로 고정한다. 배치가 진행될수록
   고정 영역이 확장되어 CP 탐색 공간을 줄인다.

---

## 작업 순서 결정 (Job Sequence Determination)

알고리즘은 작업이 추가되는 **순서**를 먼저 결정한다. 우선순위는 다음과 같다.

```text
1. job_sequence_override != None  → override 사용 (빠진 작업은 fallback)
2. job_seq_by_1st_stage == True   → 첫 번째 스테이지의 시작 시각 순
3. job_seq_by_bottleneck_stage    → 병목 스테이지의 작업 순서
4. default                        → 중점 시각(midpoint) 순
```

`job_sequence_override`가 주어진 경우, fallback 시퀀스(midpoint 순)와
교집합을 취해 유효한 작업만 포함하고, override에 없는 나머지 작업은
fallback 순서로 추가한다. 중복 작업은 제거된다.

이 작업 순서는 `preserved_head_job_portion`에 따라 head(보존)와 tail(재구성)
로 나뉜다:

```text
preserved_cnt = int(job_cnt * preserved_head_job_portion)
head_jobs = sequence[:preserved_cnt]   → 보존
tail_jobs = sequence[preserved_cnt:]   → 재구성 대상
```

---

## 배치 분할 (Batch Partitioning)

재구성 대상(tail) 작업을 배치로 나누는 방법이다.

```text
입력: tail_jobs, added_batch_size, [added_batch_count],
      [min_added_batch_count], [max_added_batch_count]

1. batch_count_from_size = ceil(len(tail_jobs) / added_batch_size)
2. resolved_batch_count =
     - added_batch_count가 있으면 그 값
     - 없으면 batch_count_from_size
   (min/max_added_batch_count 범위로 조정, tail_jobs 수를 초과하지 않음)
3. if added_batch_count == None and resolved == batch_count_from_size:
     tail_jobs를 added_batch_size씩 순차 분할
   else:
     _split_evenly_by_count()로 균등 분할
```

---

## 시간 가드 (Time Guards)

알고리즘은 시간 예산을 초과하지 않도록 두 단계에서 조기 종료를 결정한다.

### 배치 시작 전 조기 종료 검사 (_should_skip_full_neh_before_first_batch)

첫 번째 배치를 시작하기 전에, 전체 NEH-CP 실행 시간 추정치가 남은 시간을
초과하면 실행을 건너뛰고 reference schedule을 그대로 반환한다.

```text
estimated_batch_sec = max_time_per_add
  (+ max_time_per_add_2nd_obj if minimize_sum_ci_lex)
estimated_full_neh_sec = remaining_batch_count × estimated_batch_sec
                        × full_neh_estimate_safety_factor
available_for_neh = remaining_before_final - successor_reserve_sec

if estimated_full_neh_sec > available_for_neh:
    skip (return reference schedule)
```

### 배치 사이 조기 종료 검사 (_should_stop_before_next_batch)

각 배치가 끝난 후, 다음 배치를 시작할지 결정한다. 세 가지 조건 중 하나라도
만족하면 중단한다.

```text
1. stop_before_final_reserve AND final_time_reserve_is_reached()
2. successor_reserve_sec > 0 AND remaining <= successor_reserve_sec
3. 최근 3개 배치의 평균 소요 시간 × safety_factor >= available_for_neh
   (단, completed_batch_count >= time_guard_min_completed_batches)
```

`successor_reserve_sec`는 `min_remaining_sec_after_neh` (초 단위) +
`min_remaining_nc_after_neh × jobs × stages` (nc 단위)로 계산된다.
이는 NEH-CP 이후 실행될 서브루틴을 위해 확보할 시간이다.

---

## CP 서브문제 구성 및 풀이

### 1. 부분 인스턴스 생성

`create_instance_of_job_subset(instance, current_job_id_list)`로 현재까지
추가된 작업만 포함하는 축소된 인스턴스를 생성한다.

### 2. CP 모델 빌드

`BaseModelBuilder.build()`를 호출하여 CP-SAT 모델을 구성한다.

```text
horizon = partial_sol.makespan   ← 현재 부분 스케줄의 makespan을 horizon으로 사용
mdl, params, variables = build(sub_instance, horizon,
    minimize_sum_ci=(minimize_sum_ci_lex),
    minimize_makespan_plus_sum_other_stages=(minimize_sum_ci_lin),
    tighten_ranges=tighten_ranges,
    link_job_completion=link_job_completion)
```

### 3. Hint 및 프로파일 고정 제약 추가

`_add_hints_and_additional_constraints()`가 다음을 수행한다.

1. **Start/End Hint**: partial solution의 시작/종료 시각을 CP-SAT Hint로 설정
2. **Profile Fixing**: 이전 partial solution의 작업 순서를 선행 제약으로 추가
   - `batch_idx < profile_fix_min_batch_idx`면 스킵
   - `profile_fix_min_job_age_batches`가 설정된 경우,
     특정 배치 이상 전에 추가된 작업만 고정 대상
   - 모드: machine-level 또는 stage-level

### 4. CP 풀이

```text
_timelimit = ctx.get_remaining_time_limit(max_time_per_add)
report = ctx.solve_cp_model_2(mdl, _timelimit, solver_thread_cnt,
                              use_lns_only=use_lns_only)

if report.is_feasible:
    new_sol = ctx.create_schedule(params, variables)
    if make_semi_active_every_cp:
        new_sol.make_semi_active(stage_2_job_2_p_dict)
    if new_sol.makespan >= partial_sol.makespan:
        new_sol = partial_sol   # 개선 없으면 유지
```

### 5. Full Schedule 평가

CP는 partial solution(현재 배치까지의 작업만)에 대해서만 최적화하므로,
미추가 작업(tail)이 있으면 full schedule을 위해 dispatch로 채워야 한다.

```text
if !all_jobs_are_included:
    remaining_jobs = tail_jobs - current_job_id_list
    temp = MixedDispatcher.get_best_mixed_schedule_by_sequence(
        remaining_jobs, schedule=partial_sol.deepcopy(), ...)
    if temp is None: fallback to dispatch_job_by_stages
    full_sol = temp
else:
    full_sol = partial_sol

if full_sol.makespan < best_full_obj:
    best_full_obj, best_full_sol = 갱신
```

---

## 2차 목적함수 (Sum C_i 최소화)

`minimize_sum_ci_lex=True`인 경우, makespan 최적화가 끝난 후 2차 CP 모델을
추가로 푼다.

```text
1차 CP 완료 → new_sol 획득 (makespan 최소화)

2차 CP 모델:
  - minimize_sum_ci_lex=True로 build (sum C_i가 objective가 됨)
  - reference = new_sol (1차 최적해)
  - horizon은 new_sol의 makespan 유지 (makespan이 증가하지 않도록)
  - 별도 시간 제한(max_time_per_add_2nd_obj)

2차 CP 완료 → new_sol_2 획득 (sum C_i 최소화, makespan 동일)
```

2차 목적 전용 multiplier(`cp_tl_nc_multiplier_2nd_obj` /
`cp_tl_c_multiplier_2nd_obj`)가 주어지지 않으면 `max_time_per_add`와
동일한 값을 사용한다.

---

## 전체 실행 흐름

```text
입력: ref_schedule (초기 실행 가능해), instance,
      job_2_stage_2_p_dict / stage_2_job_2_p_dict,
      [파라미터들]

 1. 파라미터 정규화 및 검증
    a. added_batch_size 기본값 처리 (None → 1)
    b. max_time_per_add 산정 (cp_tl_nc_multiplier 또는 cp_tl_c_multiplier)
    c. max_time_per_add_2nd_obj 산정 (2차 목적 사용 시)
    d. preserved_head_job_portion 범위 검증 (< 0 → 0, >= 1 → ref_schedule 반환)
    e. successor_reserve_sec 계산

 2. 작업 순서 결정
    a. job_sequence_override → 1st stage → bottleneck → midpoint 순 우선순위
    b. head (보존) / tail (재구성) 분할

 3. 배치 분할
    a. _split_tail_jobs_into_batches()로 tail을 배치 리스트로 변환
    b. effective_profile_fix_min_batch_idx 산정

 4. [시간 가드] 전체 NEH 실행 시간 추정 → 초과 시 ref_schedule 반환

 5. (head_jobs 존재 시) partial solution 초기화
    a. ref_schedule에서 head_jobs만 deepcopy → partial_sol
    b. make_semi_active()로 정규화

 6. for batch_idx, job_sublist in enumerate(batches, start=1):
      a. [시간 가드] _should_stop_before_next_batch() → True면 break
      b. current_job_id_list에 job_sublist 추가
      c. MixedDispatcher.get_best_mixed_schedule_by_sequence()로
         job_sublist 디스패치 → partial_sol_best
      d. _solve_cp_model() → new_sol
         - _create_sub_cp_model() = build + hints + profile fixing
         - ctx.solve_cp_model_2()로 CP 풀이
         - [선택] 2차 목적(sum C_i) CP 추가 풀이
      e. partial_sol 갱신 = new_sol
      f. 남은 작업 dispatch → full_sol 생성
      g. best_full_sol 갱신 (개선 시)
      h. sub_obj_store에 value(bound) 기록
      i. batch_elapsed_sec_list에 소요 시간 기록

 7. error_if_infeasible → best_full_sol 검증

 8. 반환: NehCpResult(best_full_sol, sub_obj_store, best_full_obj)
```

---

## 파라미터 요약

### 작업 순서 및 배치 제어

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `job_sequence_override` | `Sequence[str] \| None` | `None` | 명시적 작업 순서. None이면 자동 결정 |
| `job_seq_by_1st_stage` | `bool` | `False` | 첫 스테이지 시작 시각 순으로 정렬 |
| `job_seq_by_bottleneck_stage` | `bool` | `False` | 병목 스테이지 작업 순서로 정렬 |
| `preserved_head_job_portion` | `float` | `0.0` | 보존할 head 작업 비율 (0.0~1.0). 0.0이면 전체 재구성 |
| `added_batch_size` | `int \| None` | `None` (→1) | 배치당 작업 수. None이거나 ≤0이면 1 |
| `added_batch_count` | `int \| None` | `None` | 강제 배치 수. None이면 batch_size로 자동 결정 |
| `min_added_batch_count` | `int \| None` | `None` | 최소 배치 수 |
| `max_added_batch_count` | `int \| None` | `None` | 최대 배치 수 |

### 시간 예산

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `max_time_per_add` | `float \| None` | `None` | 배치당 CP 시간 제한(초). None이면 multiplier로 산정 |
| `cp_tl_nc_multiplier` | `float \| None` | `None` | 시간 = multiplier × jobs × stages |
| `cp_tl_c_multiplier` | `float \| None` | `None` | 시간 = multiplier × stages |
| `cp_tl_nc_multiplier_2nd_obj` | `float \| None` | `None` | 2차 목적 시간 = multiplier × jobs × stages |
| `cp_tl_c_multiplier_2nd_obj` | `float \| None` | `None` | 2차 목적 시간 = multiplier × stages |

2차 목적 CP의 시간 제한은 내부 변수 `max_time_per_add_2nd_obj`로 산정되며
`run()`의 직접 파라미터가 아니다(위 두 `_2nd_obj` multiplier 또는
`max_time_per_add` 폴백으로 결정).

`max_time_per_add` 우선순위: 명시값 > cp_tl_nc_multiplier > cp_tl_c_multiplier
(둘 다 None이면 None 유지). 2차 목적 시간 제한은 명시값 계층 없이
cp_tl_nc_multiplier_2nd_obj > cp_tl_c_multiplier_2nd_obj > max_time_per_add(폴백)
순으로 산정된다.

### 시간 가드

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `stop_before_final_reserve` | `bool` | `True` | Final reserve에 도달하면 중단 |
| `min_remaining_sec_after_neh` | `float \| None` | `None` | NEH-CP 후속 서브루틴을 위해 확보할 최소 시간(초) |
| `min_remaining_nc_after_neh` | `float \| None` | `None` | NEH-CP 후속을 위해 확보할 nc 시간 = nc × jobs × stages |
| `time_guard_estimate_safety_factor` | `float` | `1.15` | 배치 시간 추정의 안전 계수 (추정치 × 이 값) |
| `time_guard_min_completed_batches` | `int` | `1` | 시간 추정에 필요한 최소 완료 배치 수 |
| `skip_if_estimated_neh_exceeds_remaining` | `bool` | `True` | 전체 추정 시간이 남은 시간 초과 시 처음부터 skip |
| `full_neh_estimate_safety_factor` | `float` | `1.0` | 전체 NEH 시간 추정의 안전 계수 |

### CP 모델 구성

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `profile_fix_by_machine` | `bool` | `False` | True=기계별, False=스테이지별 프로파일 고정 |
| `machine_precedence_stride` | `int` | `1` | 기계별 선행 제약의 stride 간격 |
| `profile_fix_min_batch_idx` | `int` | `2` | 프로파일 고정을 시작할 배치 인덱스 (1부터 시작) |
| `profile_fix_min_batch_portion` | `float \| None` | `None` | 고정을 시작할 배치 비율 (0.0~1.0). min_batch_idx와 max 조합 |
| `profile_fix_max_batch_idx` | `int \| None` | `None` | 프로파일 고정 최대 배치 인덱스 |
| `profile_fix_by_machine_from_batch_idx` | `int \| None` | `None` | Machine 모드로 전환할 배치. None이면 처음부터 machine 모드 |
| `profile_fix_min_job_age_batches` | `int \| None` | `None` | 프로파일 고정 대상 작업의 최소 배치 age |
| `stage_precedence_min_processing_time_diff` | `int \| None` | `None` | 스테이지 선행 제약의 최소 처리 시간 차이 |
| `stage_precedence_min_processing_time_diff_ratio` | `float \| None` | `None` | 스테이지 선행 제약의 최소 처리 시간 차이 비율 |
| `tighten_ranges` | `bool` | `False` | Head/tail 기반 변수 도메인 압축 |
| `link_job_completion` | `bool` | `False` | 작업 완료 연결 제약 추가 |
| `make_semi_active_every_cp` | `bool` | `False` | CP 풀이 후 semi-active 변환 적용 |

### 목적함수

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `minimize_sum_ci_lex` | `bool` | `False` | True면 makespan 최소화 후 sum C_i lexicographic 최소화 |
| `minimize_sum_ci_lin` | `bool` | `False` | True면 makespan + sum C_i 선형 결합 최소화 |

### 솔버 및 검증

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `solver_thread_cnt` | `int \| None` | `None` (→1) | CP-SAT 스레드 수 |
| `use_lns_only` | `bool` | `False` | CP-SAT에서 LNS만 사용 |
| `error_if_infeasible` | `bool` | `False` | True면 최종 해의 실행 가능성 검증 후 오류 발생 |

### 로깅

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `log_cp_subproblem_bounds` | `bool` | `True` | CP 서브문제의 bound 정보 로깅 |
| `log_cp_subproblem_progress` | `bool` | `False` | CP 서브문제의 상세 진행 로깅 |

---

## 주의사항 및 응용 고려사항

### 필수 전제

- **초기 실행 가능해 필요**: NEH-CP는 reference schedule에서 출발하는
  개선/재구성 알고리즘이다. 초기해가 없으면 동작하지 않는다.
- **MixedDispatcher 의존성**: 각 배치의 초기 스케줄 생성에
  `MixedDispatcher`를 사용한다. dispatch가 실패하면 fallback으로
  `dispatch_job_by_stages`가 사용된다.
- **CP-SAT 필요**: CP 서브문제 풀이에 OR-Tools CP-SAT가 필요하다.

### 다른 문제에 응용할 때 고려사항

- **NEH 순서 결정 방식**: 현재는 midpoint / 1st stage / bottleneck 세 가지
  순서 결정 방식을 지원한다. 문제에 따라 due date, release time, 우선순위
  등 다른 기준으로 교체 가능하다.
- **배치 전략**: 배치 분할 방식(added_batch_size, 균등 분할 등)은
  `_split_tail_jobs_into_batches()`에서 결정된다. 배치 크기를 문제 크기나
  시간 예산에 맞게 동적 조절하는 확장이 가능하다.
- **목적함수 확장**: 1차 makespan + 2차 sum C_i 구조는 다른 다중 목적함수
  조합으로 교체 가능하다. 예: tardiness, weighted completion time 등.
- **CP 모델 교체**: BaseModelBuilder.build() 대신 다른 CP 모델 빌더를
  사용할 수 있다. 예를 들어 SwCpModelBuilder 등.

### 주의사항

- **프로파일 고정 강도**: `profile_fix_min_batch_idx`가 너무 작으면 초기
  배치부터 순서가 고정되어 CP 탐색 공간이 지나치게 제한된다. 기본값 2는
  첫 배치에서는 고정 없이 탐색하고, 두 번째 배치부터 점진적으로 고정한다.
- **시간 가드 과적합**: safety factor와 reserve 시간 계산은 현재 문제의
  시간 척도에 의존한다. `cp_tl_nc_multiplier`로 시간을 산정하는 경우
  jobs × stages에 비례하므로, 문제 크기 변화에 어느 정도 강건하다.
- **부분 스케줄 vs 전체 스케줄 괴리**: CP는 partial solution(현재 배치까지의
  작업만)을 최적화하지만, full schedule 평가에는 미추가 작업의 dispatch
  결과가 포함된다. dispatch 품질이 낮으면 CP의 부분 최적화가 전체 개선으로
  이어지지 않을 수 있다.
- **2차 목적의 시간 비용**: `minimize_sum_ci_lex=True`로 설정하면 각 배치에서
  CP를 두 번 푸므로 시간이 두 배 가까이 소요될 수 있다. 이에 대비해
  2차 목적 전용 multiplier(`cp_tl_nc_multiplier_2nd_obj` /
  `cp_tl_c_multiplier_2nd_obj`)를 적절히 조정해야 한다.
