# subroutine_progression.json Specification

## Overview

`subroutine_progression.json`은 Hybrid Flow Shop 최적화 실행 중 controller가 호출한 각 subroutine의 시간 경과에 따른 목적 함수(objective value) 개선 이력을 기록하는 per-instance JSON 아티팩트이다. 이 파일은 downstream 분석(endpoint metrics, RPDf 계산, mean progression curve, improvement curve, scatter plot 등)의 단일 진실 공급원(single source of truth)으로 사용된다.

### 생성 시점 및 위치

- **생성 주체**: `HfsSingleInstanceRunner.save_subroutine_progression_json()` (`hfs_single_instance_runner.py:315-319`)
- **데이터 소스**: `HybridFlowShopCpLnsControllerCore.get_progression_data()` (`controller_core.py:657-716`)
- **저장 위치**: 각 instance 디렉토리 내 `results/subroutine_progression.json` 또는 `subroutine_progression.json` (legacy 호환)
- **포맷**: JSON, indent=2, `default=str` 직렬화

### 데이터 흐름 요약

```
subroutine 실행
  → _start_subroutine_call()          # call metadata 기록
  → _record_objective_point()         # 개선 발생 시 point 기록
  → solution_manager.register()       # report에 progress_obj_value_records 포함
  → _end_subroutine_call()            # end marker 및 elapsed_sec 계산
  → get_progression_data()            # 전체 구조 조립
  → save_subroutine_progression_json() # 파일 직렬화
```

---

## Top-Level JSON Schema

```json
{
  "artifact_version": 1,
  "instance_id": "string",
  "timelimit_sec": 600.0,
  "subroutine_calls": [ ... ],
  "combined_progress_list": [ ... ],
  "subroutine_end_marker_list": [ ... ]
}
```

### Top-Level Fields

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `artifact_version` | `int` | Y | 스키마 버전. 현재 `1`. 향후 필드 추가/변경 시 증가 |
| `instance_id` | `string` | Y | 문제 인스턴스 식별자 (`self.instance.name`) |
| `timelimit_sec` | `float \| null` | Y | 인스턴스에 할당된 최대 실행 시간(초). `StoppingCriteria.timelimit`에서 가져옴 |
| `subroutine_calls` | `array[object]` | Y | top-level recorded call $C_k$의 정렬된 리스트. `call_index` 오름차순 |
| `combined_progress_list` | `array[object]` | Y | 모든 `local_progress_list`를 평탄화(flatten)하고 `global_sec` → `call_index` → `local_sec` 순으로 정렬한 통합 진행 포인트 리스트 |
| `subroutine_end_marker_list` | `array[object]` | Y | 각 recorded call의 종료 시점을 기록한 마커 리스트 |

---

## `subroutine_calls` 배열

각 요소는 controller flow에서 실행된 하나의 top-level recorded call $C_k$에 대응한다.

### 필드 정의

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `call_index` | `int` | Y | 호출 순서 인덱스 $k$. 1부터 시작, 실행마다 1씩 증가 |
| `subroutine_name` | `string` | Y | 호출된 메서드 이름 (예: `"init"`, `"sw_cp"`, `"repeat_while_improvement"`) |
| `prefixed_subroutine_name` | `string` | Y | 고유 호출 사이트 식별자. 포맷: `"{call_index}-{subroutine_name}"` (예: `"4-sw_cp"`) |
| `global_start_sec` | `float` | Y | call 시작 시의 전역 경과 시간 $t^{\mathrm{start}}_{i,a,k}$ |
| `global_end_sec` | `float \| null` | Y | call 종료 시의 전역 경과 시간 $t^{\mathrm{end}}_{i,a,k}$. 아직 종료되지 않았거나 기록 누락 시 `null` |
| `elapsed_sec` | `float \| null` | Y | call 소요 시간 $\Delta t = t^{\mathrm{end}} - t^{\mathrm{start}}$. 계산 불가 시 `null` |
| `local_progress_list` | `array[object]` | Y | 해당 call 내에서 기록된 목적 함수 개선 포인트 리스트 (아래 참조) |

### `local_progress_list` 필드 정의

각 요소는 call $C_k$ 내에서 기록된 하나의 목적 함수 개선 포인트 $(g_{i,a,k,\ell}, \tau_{i,a,k,\ell}, v_{i,a,k,\ell})$에 대응한다.

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `global_sec` | `float` | Y | 전역 경과 시간 $g_{i,a,k,\ell}$. controller 시작부터의 누적 시간 |
| `local_sec` | `float` | Y | 로컬 경과 시간 $\tau_{i,a,k,\ell} = g_{i,a,k,\ell} - t^{\mathrm{start}}_{i,a,k}$ |
| `obj_value` | `float` | Y | 해당 시점에 기록된 목적 함수 값 $v_{i,a,k,\ell}$ (makespan, 최소화 문제) |
| `call_index` | `int` | Y | 소속 call의 `call_index` (flat 처리 시 조인용 중복 필드) |
| `prefixed_subroutine_name` | `string` | Y | 소속 call의 `prefixed_subroutine_name` (flat 처리 시 조인용 중복 필드) |

### `local_progress_list` 기록 규칙

1. **첫 포인트**: call 내 첫 관측 포인트는 항상 기록됨 (개선 여부 무관)
2. **엄격한 개선(strict improvement)만 기록**: 최소화 문제 기준 `new_obj_value < last_recorded_obj_value`일 때만 새 포인트 추가
3. **동일 값 스킵**: 마지막 기록된 값과 동일하면 기록하지 않음
4. **악화 스킵**: 마지막 기록된 값보다 나쁘면 기록하지 않음
5. **활성 call 조건**: `_active_call_index`가 `None`이 아닐 때만 기록됨
6. **최대화 문제 지원**: `is_maximize=True`인 경우 `new_obj_value > last_recorded_obj_value`를 개선으로 판단

### 데이터 소스 우선순위 (local_progress_list 생성 시)

`get_progression_data()`는 다음 우선순위로 데이터를 수집한다:

1. **Direct report match**: `solution_manager.history`에서 `call_context`가 `prefixed_subroutine_name`과 일치하는 report를 찾음. `progress_obj_value_records`가 존재하면 이를 `_build_progress_point_list()`로 변환
2. **Nested report collection**: direct report가 없거나 container-type subroutine(예: `incremental_sw_cp`, `repeat_while_improvement`)인 경우, `_collect_nested_reports()`로 하위 call들의 report를 수집하여 부모 시간축에 재매핑
3. **Fallback**: 위 두 방법 모두 실패 시 `_subroutine_call_progress_map`의 기존 기록 사용

---

## `combined_progress_list` 배열

모든 `subroutine_calls[*].local_progress_list`의 포인트를 하나의 평탄화된 리스트로 합친 것이다.

### 필드 정의

`local_progress_list`의 각 요소와 동일한 필드를 가진다:

| 필드 | 타입 | 설명 |
|------|------|------|
| `global_sec` | `float` | 전역 경과 시간 |
| `local_sec` | `float` | 소속 call 기준 로컬 경과 시간 |
| `obj_value` | `float` | 목적 함수 값 |
| `call_index` | `int` | 소속 call 인덱스 |
| `prefixed_subroutine_name` | `string` | 소속 call 고유 라벨 |

### 정렬 기준

```python
sorted(key=lambda point: (
    point.get("global_sec", math.inf),
    point.get("call_index", math.inf),
    point.get("local_sec", math.inf),
))
```

즉, `global_sec` 1차, `call_index` 2차, `local_sec` 3차 오름차순 정렬.

### 설계 의도

downstream 분석이 두 가지 접근 방식 중 선택할 수 있도록 의도적 중복(intentional duplication)이다:
- per-call 중첩_trace 순회: `subroutine_calls`
- 단일 평탄화 포인트 테이블 소비: `combined_progress_list`

---

## `subroutine_end_marker_list` 배열

각 recorded call $C_k$의 종료 경계를 별도로 기록한 리스트.

### 필드 정의

| 필드 | 타입 | 설명 |
|------|------|------|
| `global_end_sec` | `float` | call 종료 시점 $t^{\mathrm{end}}_{i,a,k}$ |
| `call_index` | `int` | 해당 call의 인덱스 $k$ |
| `prefixed_subroutine_name` | `string` | 해당 call의 고유 라벨 |
| `subroutine_name` | `string` | 해당 call의 메서드 이름 |

---

## Call Context 및 중첩(Nested) 구조

### Call Context 포맷

`call_context`는 dot-notation으로 중첩 호출 계층을 표현한다:

```
ROOT                                    # 최상위
4-sw_cp                                 # top-level call
1-repeat_while_improvement              # container call
1-repeat_while_improvement.1-reps_001   # repeat 내부 iteration
1-repeat_while_improvement.1-reps_001.1-sw_cp  # repeat 내부의 중첩 call
1-incremental_sw_cp.1-unfixed_batch_count_002  # incremental 내부 배치
```

### Call Context Depth

```python
def _get_call_context_depth(call_context: str) -> int:
    if not call_context or call_context == "ROOT":
        return 0
    return len(call_context.split("."))
```

- `ROOT` → depth 0
- `4-sw_cp` → depth 1
- `1-repeat_while_improvement.1-reps_001` → depth 2

### 중첩 Report 수집 (`_collect_nested_reports`)

container-type subroutine(예: `incremental_sw_cp`, `repeat_while_improvement`)은 자체 progress가 없고 하위 call들로부터 진행 상황을 집계한다:

1. parent의 `prefixed_subroutine_name`에 `"."`를 붙인 prefix로 시작하는 하위 report 검색
2. 하위 report의 depth가 parent depth보다 큰 것만 포함
3. 하위 report의 `progress_time_basis`에 따라 시간축 변환:
   - `"local"`: `global_time = nested_global_start + local_time`
   - `"global"`: `global_time = local_time` (이미 전역 시간)
4. parent 기준 상대 시간: `parent_relative_time = global_time - parent_start`
5. 결과를 `local_sec` 오름차순 정렬

---

## 데이터 기록 메커니즘

### 핵심 내부 상태 변수

| 변수 | 타입 | 설명 |
|------|------|------|
| `_subroutine_call_progress_map` | `dict[str, list[dict]]` | `prefixed_subroutine_name` → progress point 리스트 매핑 |
| `_subroutine_call_meta_list` | `list[dict]` | 각 call의 메타데이터(call_index, subroutine_name, prefixed_subroutine_name, global_start_sec, global_end_sec, elapsed_sec) |
| `_combined_progress_list` | `list[dict]` | 실시간 통합 진행 포인트 리스트 |
| `_subroutine_end_marker_list` | `list[dict]` | call 종료 마커 리스트 |
| `_method_context_meta_map` | `dict[str, dict[str, Any]]` | call_context → 메타데이터(global_start_sec, global_end_sec, elapsed_sec) 매핑. 중첩 call의 시간 조회에 사용 |
| `_active_call_index` | `int \| None` | 현재 활성 call의 인덱스 |
| `_active_subroutine_name` | `str \| None` | 현재 활성 subroutine 이름 |
| `_active_call_global_start` | `float \| None` | 현재 활성 call의 시작 시간 |
| `_call_counter` | `int` | 다음 call_index로 사용할 카운터 |

### `_start_subroutine_call(subroutine_name)`

1. `_call_counter` 증가 → `call_index`
2. `prefixed_name = f"{call_index}-{subroutine_name}"`
3. `global_start = timer.elapsed_sec`
4. 활성 상태 설정: `_active_call_index`, `_active_subroutine_name`, `_active_call_global_start`
5. `_subroutine_call_progress_map[prefixed_name] = []` 초기화
6. `_subroutine_call_meta_list`에 메타데이터 append

### `_end_subroutine_call(subroutine_name)`

1. `global_end = timer.elapsed_sec`
2. `_subroutine_end_marker_list`에 종료 마커 append
3. `_subroutine_call_meta_list`의 해당 call에 `global_end_sec`, `elapsed_sec` 채움
4. 활성 상태 초기화 (`None`으로 설정)

### `_record_objective_point(global_sec, obj_value, is_maximize=None)`

1. `_active_call_index`가 `None`이면 즉시 반환 (noop)
2. `local_sec = global_sec - global_start`
3. 기존 포인트가 있으면 개선 여부 확인:
   - `is_maximize`가 `None`이면 `cp_model.is_maximize()`에서 추론
   - 최소화: `obj_value < last_obj_value`
   - 최대화: `obj_value > last_obj_value`
   - 개선이 아니면 반환 (noop)
4. point dict 생성 → `_subroutine_call_progress_map[prefixed_name]` 및 `_combined_progress_list`에 append

### `_call_method(method_name, **kwargs)`

1. `_method_context_mgr.push(method_name)` → call context 스택에 푸시
2. `_record_method_context_start(call_context)` → 시작 시간 기록
3. 메서드 실행
4. `_record_method_context_end(call_context)` → 종료 시간 및 elapsed_sec 계산
5. `_method_context_mgr.pop()` → 스택에서 팝
6. 예외 발생 시에도 context 정리 후 재던지기

### `temporarily_extended_context(appended_name)`

context manager로, 현재 call context에 이름을 추가하여 중첩 컨텍스트를 임시로 생성한다. `repeat_while_improvement`의 `reps_*` iteration 등에 사용.

### `add_obj_value_log(elapsed, value, is_maximize)` / `extend_obj_value_log(value_log, is_maximize)`

부모 클래스 메서드를 호출한 후 `_record_objective_point()`를 통해 progression 데이터에도 기록한다.

---

## HfsSubroutineReport와의 관계

### HfsSubroutineReport 구조

```python
@dataclass(frozen=True)
class HfsSubroutineReport(SubroutineReport):
    is_init: bool
    subroutine_name: str = ""
    call_context: str = ""                           # "4-sw_cp" 형식
    progress_obj_value_records: tuple[tuple[float, float], ...] = ()  # (local_time, obj_value)
    progress_time_basis: str = "local"               # "local" 또는 "global"
```

### Report → Progression 매핑

| Report 필드 | Progression JSON 필드 | 변환 |
|-------------|----------------------|------|
| `call_context` | `prefixed_subroutine_name` | 직접 매칭 |
| `progress_obj_value_records` | `local_progress_list` | `_build_progress_point_list()`로 변환 |
| `progress_time_basis` | 시간 변환 로직 | `"local"`이면 `global_sec = global_start + local_time`, `"global"`이면 `global_sec = local_time` |

### Solution Manager History

`solution_manager.history`는 실행 중 `register()` 호출마다 `SolutionRecord(report, solution)`를 누적 저장한다. overwrite되지 않으며, 모든 report가 보존된다. `get_progression_data()`는 이 history를 조회하여 `call_context` 매칭으로 progress 데이터를 가져온다.

---

## Downstream 활용

### 1. Endpoint Metrics (`build_instance_endpoint_rows_from_progression`)

scatter plot용 endpoint reconstruction:
- flow를 왼쪽에서 오른쪽으로 스캔하며 `subroutine_name` 정확 매칭으로 call 정렬
- 매칭된 call의 `end_time = global_end_sec`
- `obj_value = local_progress_list[-1].obj_value` (없으면 이전 effective value carry-forward)
- `record_all_subroutines=True` 시 미실행 trailing 항목도 마지막 known 값으로 fill
- `omitted_subroutines`는 reconstruction 이후에 제거

### 2. Progression Points (`build_progression_points`)

각 progress point를 DataFrame row로 변환:
- `norm_time = global_sec / timelimit` (정규화 시간)
- `rpd_f = compute_rpdf(obj_value, ref_obj_value)` (RPDf 계산)
- `local_ratio = local_sec / elapsed_sec` (call 내 상대 진행률)

### 3. Mean Progression Curve (`compute_mean_progression_curve`)

subroutine별 `local_ratio` grid(0~1, 20개 구간)에 대해 `rpd_f` 평균 계산. window ±0.05 버킷 사용.

### 4. Improvement Curve (`compute_improvement_curve`)

subroutine별 `local_ratio` grid에 대해 초기값 대비 개선률 계산:
```
improvement = (initial_obj - final_obj) / initial_obj
```

### 5. Scenario Aggregation (`aggregate_scenario_progression`)

여러 instance의 progression JSON을 로드하여 통합:
- `progression_df`: 전체 progress points DataFrame
- `mean_points`: subroutine별 `(mean_norm_time, mean_rpd_f)`
- `mean_curve`: mean progression curve
- `improvement_curve`: improvement curve

---

## JSON 파일 탐색 규칙

downstream 분석은 다음 경로에서 `subroutine_progression.json`을 탐색한다:

1. `{instance_dir}/results/subroutine_progression.json` (우선)
2. `{instance_dir}/subroutine_progression.json` (fallback, legacy 호환)

---

## 완전한 JSON 예시

```json
{
  "artifact_version": 1,
  "instance_id": "ta01",
  "timelimit_sec": 600.0,
  "subroutine_calls": [
    {
      "call_index": 1,
      "subroutine_name": "set_random_seed",
      "prefixed_subroutine_name": "1-set_random_seed",
      "global_start_sec": 0.0,
      "global_end_sec": 0.001,
      "elapsed_sec": 0.001,
      "local_progress_list": []
    },
    {
      "call_index": 2,
      "subroutine_name": "init",
      "prefixed_subroutine_name": "2-init",
      "global_start_sec": 0.001,
      "global_end_sec": 5.234,
      "elapsed_sec": 5.233,
      "local_progress_list": [
        {
          "global_sec": 3.12,
          "obj_value": 1100.0,
          "call_index": 2,
          "prefixed_subroutine_name": "2-init",
          "local_sec": 3.119
        },
        {
          "global_sec": 5.234,
          "obj_value": 1050.0,
          "call_index": 2,
          "prefixed_subroutine_name": "2-init",
          "local_sec": 5.233
        }
      ]
    },
    {
      "call_index": 3,
      "subroutine_name": "sw_cp",
      "prefixed_subroutine_name": "3-sw_cp",
      "global_start_sec": 5.234,
      "global_end_sec": 25.678,
      "elapsed_sec": 20.444,
      "local_progress_list": [
        {
          "global_sec": 8.5,
          "obj_value": 1020.0,
          "call_index": 3,
          "prefixed_subroutine_name": "3-sw_cp",
          "local_sec": 3.266
        },
        {
          "global_sec": 15.0,
          "obj_value": 995.0,
          "call_index": 3,
          "prefixed_subroutine_name": "3-sw_cp",
          "local_sec": 9.766
        },
        {
          "global_sec": 25.678,
          "obj_value": 980.0,
          "call_index": 3,
          "prefixed_subroutine_name": "3-sw_cp",
          "local_sec": 20.444
        }
      ]
    }
  ],
  "combined_progress_list": [
    {
      "global_sec": 3.12,
      "obj_value": 1100.0,
      "call_index": 2,
      "prefixed_subroutine_name": "2-init",
      "local_sec": 3.119
    },
    {
      "global_sec": 5.234,
      "obj_value": 1050.0,
      "call_index": 2,
      "prefixed_subroutine_name": "2-init",
      "local_sec": 5.233
    },
    {
      "global_sec": 8.5,
      "obj_value": 1020.0,
      "call_index": 3,
      "prefixed_subroutine_name": "3-sw_cp",
      "local_sec": 3.266
    },
    {
      "global_sec": 15.0,
      "obj_value": 995.0,
      "call_index": 3,
      "prefixed_subroutine_name": "3-sw_cp",
      "local_sec": 9.766
    },
    {
      "global_sec": 25.678,
      "obj_value": 980.0,
      "call_index": 3,
      "prefixed_subroutine_name": "3-sw_cp",
      "local_sec": 20.444
    }
  ],
  "subroutine_end_marker_list": [
    {
      "global_end_sec": 0.001,
      "call_index": 1,
      "prefixed_subroutine_name": "1-set_random_seed",
      "subroutine_name": "set_random_seed"
    },
    {
      "global_end_sec": 5.234,
      "call_index": 2,
      "prefixed_subroutine_name": "2-init",
      "subroutine_name": "init"
    },
    {
      "global_end_sec": 25.678,
      "call_index": 3,
      "prefixed_subroutine_name": "3-sw_cp",
      "subroutine_name": "sw_cp"
    }
  ]
}
```

---

## 시간 관계 정리

### 전역 시간 vs 로컬 시간

- **`global_sec`**: controller 인스턴스 생성 시점부터의 누적 경과 시간. 인스턴스 전체에서 절대적 기준
- **`local_sec`**: 해당 call이 시작된 시점(`global_start_sec`)부터의 상대 경과 시간
- **관계식**: `global_sec = global_start_sec + local_sec`
- **`elapsed_sec`**: call 전체 소요 시간 = `global_end_sec - global_start_sec`
- **`local_ratio`**: call 내 상대 진행률 = `local_sec / elapsed_sec` (0~1 범위)
- **`norm_time`**: 인스턴스 전체 정규화 시간 = `global_sec / timelimit_sec` (0~1 범위, timelimit 초과 가능)

### 중첩 call의 시간 변환

하위 call의 report를 부모 시간축에 재매핑할 때:

```
# 하위 report가 progress_time_basis="local"인 경우
global_time = nested_call_global_start + nested_local_time

# 하위 report가 progress_time_basis="global"인 경우
global_time = nested_local_time  # 이미 전역 시간

# 부모 기준 상대 시간
parent_relative_time = global_time - parent_global_start
```

---

## 스키마 버전 관리

현재 `artifact_version: 1`. 향후 변경 시 고려사항:

- 필드 추가: 하위 호환 가능 (downstream이 없는 필드 무시)
- 필드 제거: 하위 호환 깨짐 (기존 consumer가 의존 중일 수 있음)
- 필드 타입 변경: 하위 호환 깨짐
- 필드 의미 변경: 하위 호환 깨짐, 버전 증가 필수

버전 증가가 필요한 변경 시 `artifact_version`를 2로 증가시키고, downstream consumer가 버전을 확인하여 적절히 처리해야 한다.

---

## 관련 파일 일람

| 파일 | 역할 |
|------|------|
| `hybridflowshop/controller/controller_core.py` | progression 데이터 생성의 핵심. `get_progression_data()`, `_start_subroutine_call()`, `_end_subroutine_call()`, `_record_objective_point()`, `_collect_nested_reports()`, `_build_progress_point_list()`, `_build_combined_progress_list()` 등 |
| `hybridflowshop/report/hfs_subroutine_report.py` | `HfsSubroutineReport` 정의. `progress_obj_value_records`, `progress_time_basis`, `call_context` 필드 포함 |
| `hybridflowshop/report/method_progression_report.py` | progression JSON 소비 측. `load_instance_progression_json()`, `build_instance_endpoint_rows_from_progression()`, `build_progression_points()`, `compute_mean_progression_curve()`, `compute_improvement_curve()`, `aggregate_scenario_progression()` |
| `hfs_single_instance_runner.py` | `save_subroutine_progression_json()`에서 JSON 파일 직렬화 |
| `hfs_multi_instance_runner.py` | `aggregate_scenario_progression()` 호출하여 시나리오 레벨 집계 |
| `tests/controller/test_subroutine_progression_recorder.py` | progression recorder 단위 테스트 |
| `tests/test_method_progression_report.py` | progression report consumer 단위 테스트 |
