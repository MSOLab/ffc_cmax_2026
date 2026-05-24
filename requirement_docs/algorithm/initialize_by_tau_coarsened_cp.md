# Tau-Coarsened CP 초기해 생성 (initialize_by_tau_coarsened_cp)

호출: `def initialize_by_tau_coarsened_cp` (hfs_cp_lns.py)

## 개요

Tau-Coarsened CP는 **문제 축소 → 대리 최적화 → 복원 → 정련**의 4단계
파이프라인을 통해 Hybrid Flowshop의 초기 실행 가능해(initial feasible
solution)를 생성하는 알고리즘이다.

핵심 전략: 원래 문제를 τ배 축소(가공 시간을 τ로 나누고 올림)하여 작고 빠른 surrogate 문제로 변환한 뒤, 다양한 방법으로 후보해를 대량 생산하고, 이를 원래 문제로 복원해 최선의 초기해를 선택한다.

---

## 문제 설명

### 파라미터 (Parameters)

다섯 부류로 나뉜다:

- **Tau / 시간 예산 계열**: `tau_values`, `surrogate_tl_nc_multiplier`,
  `polish_tl_nc_multiplier` — 축소 배율과 각 CP 단계의 시간 제한을 결정한다.
- **Dispatch 계열**: `surrogate_dispatch_before_cp`, `surrogate_dispatch_cap_portions`,
  `surrogate_dispatch_method_list`, `surrogate_dispatch_include_machine_then_job_variants`,
  `surrogate_dispatch_candidate_top_k` — 축소 문제의 dispatch 휴리스틱 경로를 제어한다.
- **NEH 계열**: `surrogate_neh_enabled`, `surrogate_neh_position`,
  `surrogate_neh_added_batch_sizes`, `surrogate_neh_sequential`,
  `surrogate_neh_cp_tl_nc_multiplier` 등 — NEH-CP 개선 단계를 제어한다.
- **PW-CP 계열**: `surrogate_pw_cp_enabled`, `surrogate_pw_cp_batch_size_ratio`,
  `surrogate_pw_cp_unfixed_batch_count_min/max`, `surrogate_pw_cp_step_size` 등 —
  Prefix-Window CP 체인을 제어한다.
- **Restore / Polish 계열**: `restore_modes`, `polish_profile_modes`,
  `polish_use_lns_only` — 복원 방식과 선택적 재최적화를 제어한다.

주요 파라미터는 「파라미터 요약」 표를 참조. 정확한 기본값은 함수 시그니처를 참조한다.

### 변수 (Variables)

이 알고리즘은 CP 결정변수를 직접 정의하지 않는다. 대신 절차 내에서 다음 상태를 추적한다:

- **`scaled_instance`** (`HybridFlowshopParameters`): τ로 축소된 인스턴스.
  `_make_tau_coarsened_instance(tau)`가 생성하며, `p[j,i] = max(1, ceil(p[j,i]/τ))`.
- **`tau_schedule_candidates`** (`list[_TauScheduleCandidate]`): 한 τ값에서
  축소 문제로부터 생성된 후보해 목록. dispatch, NEH, PW-CP, CP-SAT 각 단계의
  결과가 시그니처 중복 제거를 거쳐 누적된다.
- **`best_schedule` / `best_obj` / `best_label`**: 지금까지 평가된 모든
  복원·정련 후보 중 원래 문제 makespan이 가장 작은 incumbent와 그 값·출처 라벨.
  절차 종료 시 `best_schedule`이 `solution_manager`에 등록된다.
- **`candidate_rows`** (`list[dict]`): 모든 후보의 `(tau, surrogate_source,
  restore_mode, phase, obj, ...)` 기록. `save_candidate_artifacts=True`면
  CSV로 저장된다.

복원·정련 후보해는 별도 리스트에 모이지 않고, 생성 즉시 `maybe_record_candidate`가
`candidate_rows`에 기록하고 incumbent 갱신 여부를 판정한다.

### 목적 (Objective)

모든 후보해(restored + polished) 중 **makespan이 가장 작은 해**를 초기해로 선택한다.
`make_semi_active=True`(기본값)이면 각 후보해에 semi-active 정규화가 적용되어
불필요한 idle이 제거된다.

### 제약 (Constraints)

명시적 CP 제약이 아닌, 절차가 보장하는 불변식(invariant)이다:

1. **스케줄 실행 가능성**: 복원된 모든 후보해는 원래 가공 시간에서 유효한
   Hybrid Flowshop 스케줄이다. `_restore_original_schedule_from_tau_schedule`이
   복원 직후 `validate_schedule()`로 이를 검증한다(`make_semi_active`는 idle
   제거용 정규화일 뿐 실행 가능성 자체를 보장하지는 않는다).
2. **작업 순서 보존**: tau 해의 순서 정보가 복원 시 보존된다.
   `stage_sequence` 모드는 스테이지 내 시간 순서를,
   `machine_sequence` 모드는 기계 내 작업 순서를 그대로 유지한다.
3. **τ 축소의 구조 보존**: 축소 문제의 해 공간은 원래 문제의 축소 버전이다.
   작업 수·기계 수·스테이지 수는 동일하며, 가공 시간만 축소된다.

---

## 핵심 아이디어

1. **문제 축소 (Tau Coarsening)**: 모든 가공 시간 `p[j,i]`를
   `max(1, ceil(p[j,i] / τ))`로 변환. 작업 수와 기계 수는 유지되지만
   가공 시간이 짧아져 CP 모델이 훨씬 빠르게 풀린다.
2. **다양한 경로로 후보해 생성**: 축소된 문제에서 dispatch 휴리스틱, NEH-CP, PW-CP, CP-SAT 등 여러 경로를 통해 다양한 후보해를 확보한다. 단계 실행 순서는 `surrogate_neh_position`에 따라 달라진다(파이프라인 상세 참조).
3. **복원 (Restore)**: 축소된 후보해의 순서/기계 할당 정보를 원래 가공 시간에 적용하여 원래 문제의 실행 가능해로 변환한다.
4. **최선 선택**: 모든 복원된 후보해 중 가장 작은 makespan을 가진 해를 초기해로 채택한다.

---

## 파이프라인 단계별 상세

> **NEH 실행 위치에 따른 순서 차이**: `surrogate_neh_position`의 기본값은
> `after_cp`이다. `before_cp`이면 `dispatch → NEH → PW-CP → CP` 순,
> `after_cp`이면 `dispatch → CP → NEH` 순으로 실행되며 이때 **PW-CP chain은
> 실행되지 않는다**. 아래 1~5절은 `before_cp` 기준 순서로 기술하되, 각 절에서
> `after_cp`와의 차이를 명시한다.

### 1. Tau Coarsening (문제 축소)

```python
scaled_instance = _make_tau_coarsened_instance(tau)
# p[j,i] = max(1, ceil(original_p[j,i] / τ))
```

τ값이 클수록 문제가 더 작아져 CP가 빠르게 풀리지만, 원래 문제와의 괴리도 커진다. 일반적으로 1~10 사이 값을 사용한다.

### 2. Dispatch 초기해 생성

`_get_tau_surrogate_dispatch_schedules()`를 호출하여 축소된 인스턴스에 대해 다양한 dispatch 휴리스틱을 적용한다.

| 변수 | 설정값 |
|------|--------|
| cap_portion | 설정된 값 목록 (예: 0.2, 0.25, 0.3) |
| method_list | bn2d_all_stages, best_of_mixed_dispatches 등 |
| machine_then_job variants | True면 mtj와 jtm 모두 시도 |
| candidate_top_k | 상위 K개 선택 (중복 시그니처 제거) |

`include_surrogate_dispatch_candidate=True`인 경우 모든 dispatch 결과가 후보로 등록된다.

### 3. NEH-CP 개선

NEH-CP는 `surrogate_neh_enabled=True`일 때 동작하며, NEH의 각 삽입(insertion)
단계에서 작은 CP 서브문제를 풀어 최적의 작업 순서를 결정한다. `surrogate_neh_position`
값에 따라 실행 위치와 동작이 다르다.

#### before_cp — dispatch 결과를 NEH로 개선 후 CP에 전달

dispatch 최적해를 시작점으로 NEH-CP를 실행하고, 그 결과를 surrogate CP의
reference(시작점)로 넘긴다. **이 경로에서만 PW-CP chain(4절)이 동작한다.**
`surrogate_dispatch_before_cp=True`이고 `surrogate_neh_sources`에 `"dispatch"`가
포함되어야 실행된다.

| 파라미터 | 의미 |
|----------|------|
| `surrogate_neh_added_batch_sizes` | NEH가 한 번에 삽입할 작업 수 목록. 목록 전체를 순회 |
| `surrogate_neh_sequential` | True면 각 batch size의 결과가 다음 NEH의 입력으로 연결 |
| `surrogate_neh_cp_tl_nc_multiplier` | NEH 내 CP 서브문제당 시간 제한 배율 |

```text
예: sequential=True, added_batch_sizes=[15]
  dispatch → NEH(batch=15) → dispatch_neh_b15
```

`surrogate_neh_also_solve_dispatch_cp=True`이면, NEH-CP 개선과 별개로 dispatch
최적해를 reference로 한 surrogate CP를 한 번 더 풀어 그 결과(`dispatch_hint_cp`
후보)를 추가로 확보한다.

#### after_cp — CP 후보해를 NEH로 개선 (기본값)

surrogate CP 풀이가 끝난 뒤, `tau_schedule_candidates` 중 source가
`surrogate_neh_sources`에 속하는 후보 각각에 NEH-CP를 적용해 `{source}_neh`
후보를 추가한다. 이 경로는 **스칼라 `surrogate_neh_added_batch_size` 하나만**
사용하며, 목록 `surrogate_neh_added_batch_sizes`와 `surrogate_neh_sequential`은
무시된다. PW-CP chain도 실행되지 않는다.

### 4. PW-CP Chain (Prefix-Window CP)

NEH 결과를 시작해로 하여 PW-CP(Prefix-Window CP)를 unfixed_batch_count를 증가시키며 체인으로 실행한다. 각 단계의 결과가 다음 단계의 시작해가 된다.

PW-CP chain은 NEH가 `before_cp` 위치로 실행되어 결과를 산출한 경우에만
동작한다. 즉 `surrogate_pw_cp_enabled=True`라도 NEH가 비활성(또는
`after_cp`)이면 PW-CP chain은 실행되지 않는다.

| 파라미터 | 의미 |
|----------|------|
| batch_size_ratio | 전체 작업 수 대비 배치 크기 비율 |
| unfixed_batch_count_min/max | 윈도우 크기 범위 |
| step_size | 윈도우 이동 간격 |
| lr_profile_fixed_batch_count | 좌/우 고정 버퍼 배치 수 |

```text
예: batch_size=5%, unfixed=2→6, step=1
  NEH 결과 → PW(unfixed=2) → PW(unfixed=3) → ... → PW(unfixed=6)
  각 단계의 결과가 tau_schedule_candidates에 추가됨
```

### 5. Surrogate CP Solve

축소된 인스턴스에 대해 CP-SAT 모델을 풀어 최적해를 구한다.
`surrogate_dispatch_before_cp=True`인 경우 reference schedule을 시작점으로
제공한다. `before_cp` NEH 경로에서는 PW-CP 체인 최선해(PW-CP 미사용 시 NEH
최선 결과)를, NEH가 `after_cp`이거나 비활성이면 best dispatch를 reference로
사용한다. `surrogate_dispatch_before_cp=False`이면 reference 없이 푼다.

```text
CP 모델:
  - 목적함수: minimize makespan
  - 참조 스케줄에서 start/end hint 적용 (add_reference_precedence=False → 순서 hard 제약 없음)
  - snapshot_solution_limit만큼 중간 해 저장
```

시간 제한 = `surrogate_tl_nc_multiplier × job_count × stage_count` (또는 명시적 computational_time)

### 6. Restore (원래 문제로 복원)

축소 문제에서 찾은 각 후보해를 원래 가공 시간으로 복원한다. 두 가지 방식이 있다:

#### stage_sequence (스테이지 순서 기반)

```text
각 스테이지별로:
  tau 스케줄의 (시작 시간, 종료 시간, job index) 순으로 정렬
  → 해당 순서대로 원래 가공 시간으로 add_operation_2_stage (greedy 기계 배정)
```

#### machine_sequence (기계 순서 기반)

```text
각 스테이지의 각 기계별로:
  tau 스케줄의 기계 내 작업 순서를 그대로 유지
  → 같은 기계에 원래 가공 시간으로 add_operation_2_mc
```

`make_semi_active=True`(기본값)이면 두 방식 모두 복원 후 정규화하여 불필요한
idle time을 제거하며, 이후 `validate_schedule()`로 실행 가능성을 검증한다.

```text
각 tau 후보 → stage_sequence 복원 → restored_schedule_A
            → machine_sequence 복원 → restored_schedule_B
```

### 7. Polish (원래 문제 CP 재탐색, 선택사항)

Polish는 `polish_tl_nc_multiplier`와 `polish_computational_time` 중 **하나라도
지정된 경우** 실행되며, 복원된 해를 reference로 하여 **원래 문제**에서 CP로
재탐색한다. 둘 다 `null`이면 polish 단계 전체가 생략된다.

polish_profile_mode에 따라 복원된 해의 제약 강도가 달라진다:

| polish_profile_mode | add_precedence | profile_fix_by_machine |
|---------------------|----------------|------------------------|
| stage_sequence | True | False (스테이지 수준 순서만 고정) |
| machine_sequence | True | True (기계 수준 순서까지 고정) |
| restore | True | restore_mode와 동일 |
| none/hint_only/free | False | False (hint만 제공) |

polish가 성공하면 polished 해도 후보에 추가된다.

### 8. 최종 해 선택

```text
모든 restored + polished 후보 중:
  makespan이 가장 작은 해 → best_schedule
  → solution_manager.register()로 등록
```

선택된 해의 라벨 예:

```text
tau=5 source=dispatch_neh_b15_pw5 mode=stage_sequence phase=restored
```

---

## 후보해 파이프라인 요약

단일 τ값 기준 후보해 수 (R = `restore_modes` 수, P = `polish_profile_modes` 수):

| 단계 | 생성 수 | 비고 |
|------|---------|------|
| Dispatch | ~N개 | `include_surrogate_dispatch_candidate=True`일 때만 후보로 등록. 중복 제거 후 최대 `candidate_top_k`개 |
| NEH (before_cp) | added_batch_sizes 당 1개 | sequential chain |
| NEH (after_cp, 기본값) | 대상 후보 당 1개 | source가 `surrogate_neh_sources`에 속하는 후보마다 |
| PW-CP Chain | ~M개 (최대 5) | `before_cp` NEH 성공 시에만 동작, unfixed_batch_count 2~6 |
| CP Solve 최종해 | 1개 | |
| CP Solve Snapshots | ~K개 | `surrogate_cp_snapshot_solution_limit` 만큼 |
| **Coarsened 후보 합계** | **~N+M+K+(NEH 수)+1개** | 축소 문제 공간의 누적 후보 |
| Restore | coarsened × R | `restore_modes` 각각 |
| Polish (선택) | restore 당 P개 | `polish_profile_modes` 수만큼 |
| **전체 평가 후보** | **coarsened × R × (1 + P)** | restored + polished |

---

## 전체 실행 흐름

```text
입력: 원래 문제 instance, τ 값 목록, 파라미터들

for each τ in tau_values:
  │
  ├── 1. τ-축소 인스턴스 생성
  │
  ├── 2. [선택] Dispatch 초기해 생성 (축소 문제)
  │     · surrogate_dispatch_before_cp 또는
  │       include_surrogate_dispatch_candidate 가 True일 때만 수행
  │     · include_surrogate_dispatch_candidate=True 이면 결과를 후보로 등록
  │
  ├── 3a. [neh_position=before_cp] NEH-CP 개선 (best dispatch 대상)
  │     ├── added_batch_sizes 목록 순차 적용 → CP reference 갱신
  │     └── 4. [선택] PW-CP Chain (best NEH 결과 대상) → CP reference 갱신
  │
  ├── 5. Surrogate CP Solve (축소 문제)
  │     ├── before_cp: NEH/PW-CP 최선해를 reference로 사용
  │     ├── 최종 CP 해 → 1개 후보
  │     └── Snapshot들 (중간 해) → 최대 K개 후보
  │
  ├── 3b. [neh_position=after_cp, 기본값] CP 후보해를 NEH-CP로 개선
  │     └── source가 surrogate_neh_sources에 속하는 후보 → {source}_neh 추가
  │
  ├── 6. 모든 후보해 Restore (원래 문제로 복원)
  │     └── 각 후보 × restore_modes
  │
  └── 7. [선택] Polish (원래 문제 CP 재최적화)
        └── 복원된 해를 reference로 CP 재탐색

8. 모든 restored/polished 해 중 최선 makespan 선택 → 초기해 등록
```

### 구체 설정 예시 (Subroutine Flow YAML)

`uv run main.py`로 실행되는 production flow의 확정 설정이다. 출처:
`configs_mip_lb/20260522/subroutine_flow_20260522-final-candidateD-retained-bottleneck-3lane-2nc.yaml`
(`main_metadata_mip_lb.yaml`의 `subroutine_flow_rel_path`가 가리키는 파일).

```yaml
- method: initialize_by_tau_coarsened_cp
  tau_values:
  - 5
  surrogate_tl_nc_multiplier: 0.12
  polish_tl_nc_multiplier: null           # polish 생략
  restore_modes:
  - stage_sequence
  - machine_sequence
  polish_profile_modes:
  - stage_sequence
  surrogate_dispatch_before_cp: true
  include_surrogate_dispatch_candidate: true
  surrogate_dispatch_cap_portions:
  - 0.20
  - 0.25
  - 0.30
  surrogate_dispatch_method_list:
  - bn2d_all_stages
  - best_of_mixed_dispatches
  surrogate_dispatch_include_machine_then_job_variants: true
  surrogate_dispatch_candidate_top_k: 20
  surrogate_neh_enabled: true
  surrogate_neh_position: before_cp
  surrogate_neh_tau_values:
  - 5
  surrogate_neh_sources:
  - dispatch
  surrogate_neh_added_batch_sizes:
  - 15
  surrogate_neh_sequential: true
  surrogate_neh_also_solve_dispatch_cp: false
  surrogate_neh_cp_tl_nc_multiplier: 0.0015
  surrogate_neh_use_lns_only: true
  surrogate_neh_minimize_sum_ci_lex: false
  surrogate_neh_cp_tl_nc_multiplier_2nd_obj: null
  surrogate_pw_cp_enabled: true
  surrogate_pw_cp_tau_values:
  - 5
  surrogate_pw_cp_batch_size_ratio: 0.05
  surrogate_pw_cp_unfixed_batch_count_min: 2
  surrogate_pw_cp_unfixed_batch_count_max: 6
  surrogate_pw_cp_step_size: 1
  surrogate_pw_cp_lr_profile_fixed_batch_count: 1
  surrogate_pw_cp_enable_promotion_profile_fixed: true
  surrogate_pw_cp_non_time_fixed_op_time_limit_multiplier: 0.0015
  surrogate_pw_cp_use_lns_only: false
  surrogate_pw_cp_stop_on_no_improvement: false
  solver_thread_cnt: 24
  surrogate_use_lns_only: false
  surrogate_cp_snapshot_solution_limit: 20
  polish_use_lns_only: false
  cp_model_probing_level: 1
  cp_sat_params:
    exploit_best_solution: true
    diversify_lns_params: true
    solution_pool_size: 8
  make_semi_active: true
  save_candidate_artifacts: false
  error_if_infeasible: false
  draw_gantt: false
```

`stopping_criteria_2nc.yaml`의 `timelimit_n_by_c_multiplier: 2` 가 곱해져 전체
인스턴스 시간 한도는 `2 × job_count × stage_count` 초가 되고, 그 안에서 위
`surrogate_tl_nc_multiplier: 0.12` 로 surrogate CP에 `0.12 × J × S` 초가
할당된다.

위 설정에서 실제 실행 순서:

```text
τ=5 인스턴스 생성
    │
    ├── dispatch × 12 (2 methods × 3 cap_portions × 2 mtj/jtm variants)
    │     └── tau_schedule_candidates 추가 (중복 제거, top_k=20)
    │
    ├── NEH(batch=15) ← best dispatch (use_lns_only=true)
    │     ├── tau_schedule_candidates 추가
    │     └── PW-CP chain(unfixed=2~6) ← NEH 결과 (use_lns_only=false)
    │           ├── 각 단계 tau_schedule_candidates 추가
    │           └── 최종 best → CP reference
    │
    ├── CP Solve (0.12 × J × S sec, 24 threads) ← PW-CP 최종 결과
    │     ├── 최종해 tau_schedule_candidates 추가
    │     └── snapshot 최대 20개 추가
    │
    ├── for each candidate (최대 ~39개):
    │     ├── stage_sequence restore → 평가
    │     └── machine_sequence restore → 평가
    │
    └── (polish_tl_nc_multiplier=null → polish 전체 생략)

최종: 모든 restored 해 중 makespan 최소 선택 → solution_manager 등록
```

> 후속 단계: 이 초기해를 받은 뒤 production flow는
> `critical_schedule_repair_ls` → `apply_retained_stage_cp_lb` →
> `dispatch_from_retained_cp` → workload-adaptive retained CP → 최종
> `solve_base_cp_model_from_final_time_reserve` 로 이어지지만, 그 단계들은
> 별도 문서에서 다룬다.

---

## 파라미터 요약

### Tau Coarsening

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `tau_values` | `Sequence[int]` | 축소 배율 목록. 각 τ값에 대해 독립적으로 파이프라인 실행 |

### Time Budget

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `surrogate_computational_time` | `float \| None` | 축소 CP 명시 시간 |
| `surrogate_tl_nc_multiplier` | `float \| None` | 축소 CP 시간 = multiplier × jobs × stages |
| `polish_computational_time` | `float \| None` | Polish CP 명시 시간 |
| `polish_tl_nc_multiplier` | `float \| None` | Polish CP 시간 = multiplier × jobs × stages |

`*_tl_nc_multiplier`가 지정되면 그 값이 우선하며 `*_computational_time`은 무시된다.
**`polish_tl_nc_multiplier`와 `polish_computational_time`이 둘 다 `null`이면 polish 단계가 생략된다.**

### Dispatch

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `surrogate_dispatch_before_cp` | `bool` | dispatch 해를 CP의 reference로 사용 |
| `include_surrogate_dispatch_candidate` | `bool` | dispatch 해를 restoration 후보에 포함 |
| `surrogate_dispatch_cap_portions` | `Sequence[float]` | dispatch cap portion 목록 |
| `surrogate_dispatch_method_list` | `Sequence[str]` | 사용할 dispatch 방법 |
| `surrogate_dispatch_include_machine_then_job_variants` | `bool` | mtj/jtm variants 시도 |
| `surrogate_dispatch_candidate_top_k` | `int` | 상위 K개 dispatch 후보 선택 |

### NEH

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `surrogate_neh_enabled` | `bool` | NEH-CP 개선 사용 |
| `surrogate_neh_position` | `str` | CP 전(`before_cp`) 또는 후(`after_cp`). 기본값 `after_cp` |
| `surrogate_neh_tau_values` | `Sequence[int] \| None` | 적용할 τ 필터 |
| `surrogate_neh_sources` | `Sequence[str]` | NEH를 적용할 후보 소스. `before_cp`에서는 `"dispatch"` 포함 필수 |
| `surrogate_neh_added_batch_size` | `int` | NEH batch size (스칼라). **`after_cp` 경로가 사용** |
| `surrogate_neh_added_batch_sizes` | `Sequence[int] \| None` | NEH batch size 목록. **`before_cp` 경로만 사용** |
| `surrogate_neh_sequential` | `bool` | batch size 간 chain 연결 (`before_cp` 경로만) |
| `surrogate_neh_also_solve_dispatch_cp` | `bool` | `before_cp`에서 dispatch 최적해로 surrogate CP를 추가로 풀어 후보 확보 |
| `surrogate_neh_cp_tl_nc_multiplier` | `float \| None` | NEH 내 CP 시간 배율 |
| `surrogate_neh_use_lns_only` | `bool` | NEH CP에서 LNS만 사용 |

### PW-CP

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `surrogate_pw_cp_enabled` | `bool` | PW-CP chain 사용 |
| `surrogate_pw_cp_tau_values` | `Sequence[int] \| None` | 적용할 τ 필터 |
| `surrogate_pw_cp_batch_size_ratio` | `float` | 배치 크기 = ratio × job_count |
| `surrogate_pw_cp_unfixed_batch_count_min/max` | `int` | 윈도우 크기 범위 |
| `surrogate_pw_cp_step_size` | `int` | 윈도우 이동 간격 |
| `surrogate_pw_cp_lr_profile_fixed_batch_count` | `int` | 좌/우 profile-fixed 배치 수 |
| `surrogate_pw_cp_enable_promotion_profile_fixed` | `bool` | unfixed 잡의 profile-fixed 작업을 unfixed로 승격 |
| `surrogate_pw_cp_non_time_fixed_op_time_limit_multiplier` | `float \| None` | PW-CP 배치당 시간 배율 |
| `surrogate_pw_cp_use_lns_only` | `bool` | PW-CP에서 LNS만 사용 |
| `surrogate_pw_cp_stop_on_no_improvement` | `bool` | 개선 없으면 chain 중단 |

### Restore & Polish

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `restore_modes` | `Sequence[str]` | 복원 모드 목록 (`stage_sequence`, `machine_sequence`) |
| `polish_profile_modes` | `Sequence[str]` | Polish 모드 (`restore`, `none`, `machine_sequence`, `stage_sequence`) |
| `polish_use_lns_only` | `bool` | Polish에서 LNS만 사용 |

### CP-SAT 공통

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `solver_thread_cnt` | `int` | CP-SAT 스레드 수 |
| `cp_model_probing_level` | `int` | Probing 레벨 |
| `cp_sat_params` | `Mapping[str, Any]` | 추가 CP-SAT 파라미터 (예: `diversify_lns_params`, `solution_pool_size`) |
| `surrogate_cp_snapshot_solution_limit` | `int` | CP 중간해 스냅샷 수 제한 |
| `make_semi_active` | `bool` | 모든 해에 semi-active 정규화 적용 |
| `save_candidate_artifacts` | `bool` | 후보해 CSV 저장 |
| `draw_gantt` | `bool` | 최종 해 Gantt 차트 출력 |

---

## 주의사항 및 응용 고려사항

- **다양성이 핵심**: 한 가지 경로로 찾은 해보다 여러 경로(dispatch → NEH → PW-CP → CP)를 통해 다양한 구조의 후보해를 확보하는 것이 더 좋은 초기해로 이어진다.
- **Coarsening 오차**: τ가 클수록 축소 문제는 빠르게 풀리지만, restore 시 원래 문제와의 괴리가 커져 restoration 품질이 떨어질 수 있다.
- **Restore 모드 선택**: `stage_sequence`는 유연한 기계 배정을 허용하는 반면,
  `machine_sequence`는 tau 해의 기계별 작업 순서를 엄격히 보존한다.
  두 모드를 모두 시도하여 각각의 장점을 활용한다.
- **시간 배분**: Surrogate CP에 가장 많은 시간을 할당하고, NEH와 PW-CP는 상대적으로 짧은 시간 제한으로 빠르게 개선만 수행한다.
