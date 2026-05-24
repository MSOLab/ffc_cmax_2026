# Mixed Dispatch 최적 np 탐색 (get_best_mixed_schedule_by_sequence)

호출: `def get_best_mixed_schedule_by_sequence` (hybridflowshop/dispatcher/mixed.py, MixedDispatcher 클래스)

## 개요

Mixed Dispatch는 head(tail 개념을 이용한 혼합 디스패치 전략)에서 **최적의 우선순위 작업 수(np, number of priority jobs)**를 결정하는 메서드이다. 주어진 작업 순서(`job_sequence`)에 대해 다양한 np 후보를 시도하고, 각 np에 대해 mixed dispatch를 실행한 뒤 가장 작은 makespan을 가진 스케줄을 반환한다.

Mixed dispatch 자체는 `from_job_sequence_get_schedule_mixed()`가 수행한다. 이 메서드는 그 **외부 루프**로서 "np를 얼마로 설정할 것인가"를 exhaustive하게 탐색한다.

---

## 문제 설명

### 파라미터

알고리즘의 동작을 제어하는 입력 설정값은 다음 부류로 나뉜다.

**디스패치 대상 명세:**

| 부류 | 항목 | 설명 |
|------|------|------|
| 작업 순서 | `job_sequence` | 전체 작업의 우선순위 기준 순서. tie-breaker로 사용 |
| 확장 대상 | `schedule` | 기존 스케줄 위에 증분 디스패치. None이면 빈 스케줄 생성 |
| 시작 위치 | `from_stage` | 디스패치를 시작할 stage. None이면 첫 번째 stage |
| Release time | `job_2_release_t` | 각 job의 release time. 첫 stage의 priority queue 하한으로 사용 |

**디스패치 모드:**

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `machine_then_job` | `False` | True면 stage를 machine-centric으로 dispatch (첫 stage는 예외적으로 job-first) |
| `head_for_all_stages` | `False` | True면 head를 모든 stage에 동일 np로 적용. False면 첫 stage(or from_stage)에만 적용 |
| `use_palmer_index` | `False` | machine_centric_dispatch_4에서 Palmer index 사용 여부 |

**시각화/디버깅:**

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `draw_gantt_per_step` | `False` | 디스패치 단계별 Gantt 차트 출력 |
| `get_file_path_for_subroutine` | `None` | Gantt 차트 저장 경로 생성 콜백 |

**인스턴스 데이터 (MixedDispatcher 멤버):**

| 항목 | 출처 | 설명 |
|------|------|------|
| `job_count` | `instance.job_count` | 전체 job 수. np 후보 생성의 상한 |
| `stage_id_list` | `instance.stage_id_list` | 모든 stage ID 목록 |
| `stage_2_job_2_p` | `instance.stage_2_job_2_p_map` | stage → job → 가공 시간 매핑 |

모든 파라미터의 기본값과 타입에 대한 완전한 목록은 뒤쪽 「파라미터 요약」 표를 참조.

### 변수

알고리즘이 풀이 과정에서 추적하는 상태 변수는 다음과 같다.

| 변수 | 타입 | 설명 |
|------|------|------|
| `np` | `int` | 현재 시도 중인 priority job 수 |
| `np_list` | `list[int]` | 감소 수열로 정렬된 np 후보 목록 (`_get_np_candidates()` 생성) |
| `np_2_stage_2_head` | `dict[int, dict[str, int]]` | np → (stage_id → head 수) 매핑. 각 np별로 head 구성을 캐싱 |
| `best_obj` | `int \| None` | 현재까지 발견된 최소 makespan |
| `best_sch` | `HybridFlowshopLiteSchedule \| None` | best_obj에 대응하는 스케줄 객체 |
| `_schedule` | `HybridFlowshopLiteSchedule` | 현재 np로 mixed dispatch를 실행 중인 스케줄 (deepcopy) |

### 목적

makespan을 최소화한다:

```
best_obj = min_{np in np_list} makespan(from_job_sequence_get_schedule_mixed(np))
```

np를 변화시키면서 같은 `job_sequence`로 mixed dispatch를 반복하고, 산출된 모든 스케줄 중 가장 작은 makespan을 가진 것을 최종 해로 선택한다.

### 제약

Mixed dispatch가 내부적으로 보장하는 제약은 다음 세 가지로 요약된다.

1. **선행 제약**: 각 job은 stage 순서대로 처리되어야 한다. Head job은 `dispatch_job_by_stages`로 모든 남은 stage를 순차 통과하며, tail job은 `dispatch_stage_by_jobs`로 stage별 이전 stage 종료 시각을 하한으로 배정된다.
2. **용량 제약**: 각 stage의 병렬 기계 수를 초과할 수 없다. `add_operation_2_stage`가 가능한 가장 이른 기계에 배정한다.
3. **Head 예산 제약**: `stage_2_head` 값은 stage 간 cumulative하게 적용된다. 앞 stage에서 k개의 job을 head로 소비하면 다음 stage의 유효 head 수는 `min(head_value, 남은_job_수)`로 감소한다.

---

## 핵심 아이디어

단일 `job_sequence`에 대해 디스패치를 한 번만 수행하는 대신, **얼마나 많은 작업을 "head 방식"으로 처리할지(np)**를 변화시키면서 여러 대안을 평가하고 최선을 선택한다.

- np = `job_count`: 모든 작업을 head 방식으로 dispatch → 순서 기반 스케줄 (sequence-driven)
- np = 0: 모든 작업을 tail 방식으로 dispatch → priority 기반 스케줄 (readiness-driven)
- 그 사이 값: head 방식이 우선 적용되는 작업 수를 조절 → 두 극단 사이의 중간 형태

이렇게 함으로써 "순서 엄수"와 "준비 완료 우선" 사이의 스펙트럼을 커버한다.

---

## 우선순위 작업 수(np) 후보 생성

`_get_np_candidates()` (같은 파일, `MixedDispatcher._get_np_candidates()`)는 np 후보 목록을 생성한다.

```text
np = job_count
while np > 1:
    np = ceil(np / 2)
    np_list = [job_count, ceil(job_count/2), ..., 1, 0]
```

예: `job_count=10` → `[10, 5, 3, 2, 1, 0]`

감소 수열로 정렬되며, 항상 0을 마지막에 포함한다.

---

## Head 맵 구성

각 np에 대해 `np_2_stage_2_head[np]`를 다음 규칙으로 구성한다.

분기:

```
if head_for_all_stages == True:
    모든 stage_id에 np를 할당
    → {stage_1: np, stage_2: np, ..., stage_m: np}
else:
    if from_stage != None:
        from_stage만 np 할당 (from_stage 유효성 검증)
        → {from_stage: np}
    else:
        첫 번째 stage만 np 할당
        → {stage_id_list[0]: np}
```

이 매핑은 np 루프 진입 전에 한 번에 모두 생성된다 (line 97-108).

---

## Mixed Dispatch 반복

```text
for np in np_list:
    ─── 시작 스케줄 준비 ───
    if schedule != None:
        _schedule = schedule.deepcopy()
    else:
        _schedule = _create_empty_schedule()
    
    ─── Mixed dispatch 실행 ───
    from_job_sequence_get_schedule_mixed(
        _schedule,
        job_sequence,
        stage_2_job_2_p,
        np_2_stage_2_head[np],   ← 이 np에 대응하는 head 맵
        from_stage=from_stage,
        job_2_release=job_2_release_t,
        machine_then_job=machine_then_job,
        use_palmer_index=use_palmer_index,
        draw_gantt_per_step=draw_gantt_per_step,
        get_file_path_for_subroutine=get_file_path_for_subroutine,
    )
    
    ─── 최선 갱신 ───
    if best_obj is None or _schedule.makespan < best_obj:
        best_obj = _schedule.makespan
        best_sch = _schedule
```

`from_job_sequence_get_schedule_mixed()` 호출에 대한 상세는 `requirement_docs/algorithm/from_job_sequence_get_schedule_mixed.md` (미작성 시 `hybridflowshop/dispatcher/utils.py:913` 참조) 또는 같은 파일의 독스트링을 참조. 간략히: 각 stage에서 priority queue를 구성하고, 상위 np개 job은 `dispatch_job_by_stages`로 모든 잔여 stage를 순차 처리(head), 나머지는 `dispatch_stage_by_jobs`로 stage별 priority 기반 배정(tail)한다.

---

## 최선 해 선택

모든 np를 순회한 후 `best_sch`를 반환한다. 만약 어떤 np에서도 유효한 스케줄이 생성되지 않으면(`_schedule is None`), `best_obj`가 갱신되지 않으며 최종 반환값도 `None`이 된다. (`_schedule is None`이 되는 경우는 없으나, `from_job_sequence_get_schedule_mixed`의 계약상 안전하게 처리)

---

## 전체 실행 흐름

```text
입력: job_sequence, schedule(선택), from_stage(선택), job_2_release_t(선택),
      machine_then_job, head_for_all_stages, use_palmer_index,
      draw_gantt_per_step, get_file_path_for_subroutine

1. np 후보 목록 생성 (_get_np_candidates):
   [job_count, ceil(job_count/2), ..., 1, 0]

2. np_2_stage_2_head 구성 (np별 head 맵 생성):
   if head_for_all_stages:
       모든 stage에 np 할당
   else:
       from_stage 또는 첫 번째 stage에만 np 할당

3. for 각 np in np_list (내림차순):
   a. 시작 스케줄 준비 (deepcopy 또는 빈 스케줄)
   b. from_job_sequence_get_schedule_mixed 호출
      - stage_2_head = np_2_stage_2_head[np]
   c. _schedule.makespan < best_obj 이면 best 갱신

4. best_sch 반환
```

---

## 파라미터 요약

### 메서드 시그니처 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `job_sequence` | `Sequence[JobIdType]` | (필수) | 디스패치 우선순위 기준이 되는 작업 순서 |
| `schedule` | `HybridFlowshopLiteSchedule \| None` | `None` | 증분 디스패치 대상 스케줄. None이면 빈 스케줄 생성 |
| `from_stage` | `StageIdType \| None` | `None` | 디스패치 시작 stage. None이면 첫 번째 stage |
| `job_2_release_t` | `dict[JobIdType, int] \| None` | `None` | job별 release time |
| `machine_then_job` | `bool` | `False` | True면 machine-centric dispatch 우선 사용 |
| `head_for_all_stages` | `bool` | `False` | True면 모든 stage에 head 적용 |
| `use_palmer_index` | `bool` | `False` | machine_centric_dispatch_4에서 Palmer index 사용 |
| `draw_gantt_per_step` | `bool` | `False` | 단계별 Gantt 차트 PNG 출력 |
| `get_file_path_for_subroutine` | `Callable \| None` | `None` | Gantt 저장 경로 생성 콜백. None이면 draw_gantt_per_step 무시 |

### 내부 파라미터 (MixedDispatcher 인스턴스 멤버)

| 멤버 | 타입 | 설명 |
|------|------|------|
| `job_count` | `int` | 전체 job 수. np 후보 생성의 상한 |
| `stage_id_list` | `list[StageIdType]` | 모든 stage ID 목록. head 맵 구성에 사용 |
| `stage_2_job_2_p` | `dict[StageIdType, dict[JobIdType, int]]` | stage → job → processing time. mixed dispatch의 헬퍼에 전달 |

---

## 의존성 및 호출 관계

| 호출 대상 | 파일 (라인) | 역할 |
|-----------|-------------|------|
| `_get_np_candidates()` | `mixed.py:28` | 감소 수열 np 후보 생성 |
| `from_job_sequence_get_schedule_mixed()` | `utils.py:913` | 주어진 stage_2_head로 mixed dispatch 실행 |
| `_create_empty_schedule()` | `base.py:84` (상속) | 빈 스케줄 생성 |
| `schedule.deepcopy()` | `schedule_lite.py` | 스케줄 복제 (in-place 변경 방지) |

이 메서드를 호출하는 상위 메서드:

| 호출자 | 파일 (라인) | 전달하는 job_sequence |
|--------|-------------|----------------------|
| `get_schedule_by_cds` | `mixed.py:181` | `get_cds_sequence(k)` |
| `get_schedule_by_gupta` | `mixed.py:242` | `get_gupta_sequence()` |
| `get_schedule_by_palmer` | `mixed.py:291` | `get_palmer_sequence()` |

위 표는 같은 `MixedDispatcher` 클래스 내 시퀀스 헬퍼만 나열한 것이다. 이 메서드는 그 외에도 다음 위치에서 호출된다:

| 호출자 | 파일 (라인) |
|--------|-------------|
| NEH-CP constructor | `controller/neh_cp.py:531`, `controller/neh_cp.py:587` |
| CP-LNS controller | `controller/hfs_cp_lns.py:15998`, `:16005`, `:17067`, `:17087` |
| BN2D dispatcher | `dispatcher/bn2d.py:264`, `:371`, `:525` |
| 벤치마크 스크립트 | `scripts/benchmark_saved_mip_dispatch_rules.py:132` |

---

## 주의사항 및 응용 고려사항

### 확장 포인트

- **np 후보 생성 전략**: 현재는 `ceil(÷2)` 감소 수열을 사용한다. job 수나 문제 규모에 따라 다른 생성 전략(등간격, log-scale 등)으로 교체 가능하다.
- **head 분포 전략**: 현재는 모든 stage에 동일 np를 적용하거나 첫 stage에만 적용한다. Stage별 특성(bottleneck 정도)에 따라 차등 np를 적용하는 전략으로 확장 가능하다.

### 제한사항

- 같은 `job_sequence`에 대해 다양한 np만 시도한다. `job_sequence` 자체는 외부에서 결정되어 전달되므로, sequence의 품질이 결과의 상한을 결정한다.
- Mixed dispatch의 품질은 `from_job_sequence_get_schedule_mixed`의 priority queue 정책(이전 stage 종료 시각 + tie-breaker)에 의존한다. 다른 우선순위 정책으로 교체하면 결과가 달라진다.
- `head_for_all_stages=True` 시 모든 stage에 동일 np를 적용한다. 앞 stage에서 head job이 많으면 뒤 stage의 유효 head 수가 자동 감소하지만, stage별로 독립적인 head 값을 설정하고 싶다면 별도 구현이 필요하다.
