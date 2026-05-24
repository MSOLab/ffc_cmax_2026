# 계획: Phase 1B LogProcessor 분리

> 날짜: 2026-04-04 03:56 +09:00
> 전제: Phase 1A `ResumeValidator` 분리 완료
> 목표: `scripts/process_logs.py`의 핵심 로직을 `hybridflowshop/report/log_processor.py`로 이동하고, 기존 CLI/import/CSV 산출물 호환성을 유지한다.

## 확인된 현재 상태

- `hfs_multi_instance_runner.py`는 `create_method_end_time_and_obj_value_summary()`를 직접 호출해 로그 후처리의 시작점으로 사용한다.
- `scripts/process_logs.py`는 파싱 + per-instance CSV 작성 + scenario long/wide summary 작성 + CLI entrypoint를 한 파일에서 모두 담당한다.
- `tests/test_process_logs.py`는 현재 `scripts.process_logs` import 경로와 file-writing side effect를 직접 검증한다.
- downstream은 `summary_method_end_time_and_obj_value_long.csv`를 기준으로 `_create_rpd_summary()`와 이후 HTML/SVG 리포트 로직을 이어서 사용한다.

## 이번 단계의 범위

### 포함

- 신규 모듈 `hybridflowshop/report/log_processor.py` 추가
- `scripts/process_logs.py`를 thin wrapper로 축소
- `hfs_multi_instance_runner.py`의 import 경로 정리
- 신규 class 직접 테스트 추가 + 기존 wrapper 경로 회귀 테스트 유지

### 제외

- `_create_rpd_summary()` 분리 (`MetricsCalculator`)는 이번 단계에서 하지 않음
- `_create_timepoint_summaries()` 계열 분리 (`TimepointSampler`)는 이번 단계에서 하지 않음
- multi-scenario Excel/report 구조 변경은 이번 단계에서 하지 않음
- CSV 파일명, 컬럼명, 컬럼 순서, side effect 파일 생성 규칙은 바꾸지 않음

## 설계 원칙

- 클래스 이름은 기존 상위 계획과 맞춰 `LogProcessor`를 사용한다.
- `LogProcessor`는 재사용 가능한 본체이고, `scripts/process_logs.py`는 호환성 shim + CLI wrapper로 남긴다.
- `record_all_subroutines=False`, `omitted_subroutines=None` 같은 generic 기본값은 processor에 두고, 현재 HFS 전용 정책(`record_all_subroutines=True`, 특정 omitted set)은 `HfsMultiInstanceRunner.post_run_process()`에서 계속 명시적으로 넘긴다.
- 이번 단계는 구조 분리가 목적이므로, 내부 알고리즘 변경보다 기존 동작을 최대한 기계적으로 보존하는 쪽을 우선한다.

## 구현 계획

### 1. `hybridflowshop/report/log_processor.py` 신설

- `LogProcessor` 클래스를 추가하고, 현재 `create_method_end_time_and_obj_value_summary()`가 받는 인자를 생성자/메서드로 흡수한다.
- 아래 로직을 새 모듈로 이동한다.
  - `parse_controller_log()`
  - `parse_obj_log()`
  - `_is_missing_obj_value()`
  - `get_obj_value_for_method()`
  - `get_methods_from_flow()`
  - `_build_instance_method_rows()`
  - `process_instance()`
  - `create_method_end_time_and_obj_value_summary()`
- public 진입점은 최소 2개로 둔다.
  - class API: `LogProcessor(...).create_method_end_time_and_obj_value_summary()`
  - compatibility function: module-level `create_method_end_time_and_obj_value_summary(...)`
- module-level 상수는 그대로 유지한다.
  - `DEFAULT_CONTROLLER_LOG_NAME`
  - `DEFAULT_RESULTS_DIR`
  - `OBJ_LOG_FN_FORMAT`

### 2. 동작 보존 포인트를 먼저 고정

- 아래 세부 동작은 그대로 유지해야 한다.
  - `subroutine_flow.yaml` 기준으로 method prefix를 `1-`, `2-` 형태로 생성
  - controller log에서 `call_context` prefix 매칭으로 `method_end_times` 계산
  - obj log 경로를 `results/{instance}_obj_log.yaml` 우선, 없으면 instance root fallback
  - per-instance 산출물 작성
    - `method_time_log.json`
    - `method_end_time_and_obj_value.csv`
  - scenario 산출물 작성
    - `summary_method_end_time_and_obj_value_long.csv`
    - `summary_method_end_time_and_obj_value_wide.csv`
  - `record_all_subroutines=True`일 때 마지막 executed subroutine의 time/obj를 trailing missing rows에 채우는 규칙
  - `omitted_subroutines`는 row 계산 후 출력 단계에서만 제외하는 현재 의미론 유지
  - wide CSV column order는 omitted 적용 이후의 flow 순서를 유지

### 3. `scripts/process_logs.py`를 thin wrapper로 축소

- 새 모듈에서 다음을 import/re-export 하도록 바꾼다.
  - `LogProcessor`
  - `create_method_end_time_and_obj_value_summary()`
  - `process_scenario()` 또는 이에 해당하는 wrapper 함수
- CLI `main()`은 유지하되, 실제 처리는 `LogProcessor`에 위임한다.
- 기존 import 경로 `from scripts.process_logs import create_method_end_time_and_obj_value_summary`가 계속 동작하게 유지한다.

### 4. `hfs_multi_instance_runner.py` 연동 정리

- 직접 script 모듈에 의존하지 않도록 import를 새 모듈로 옮긴다.
- `post_run_process()`의 현재 호출 시그니처와 옵션은 유지한다.
  - `record_all_subroutines=True`
  - `omitted_subroutines={"set_random_seed", "set_cp_model_as_base_cp_model"}`
- 이번 단계에서는 runner 내부에 `LogProcessor` 인스턴스를 캐싱하지 않고, 기존 함수 호출 형태에 가깝게 유지해 diff를 작게 만든다.

### 5. 테스트 전략

- 신규 `tests/test_log_processor.py`를 추가해 class API를 직접 검증한다.
  - 기본 케이스: trailing missing subroutine blank 유지
  - `record_all_subroutines=True`일 때 trailing fill
  - `omitted_subroutines` 적용 시 long/wide/per-instance 출력 일관성
  - baseline metadata merge로 `job_cnt`, `stage_cnt`, `ref_obj_value` 컬럼 채워지는지 확인
- 기존 `tests/test_process_logs.py`는 유지하되, 최소 1개 이상은 `scripts.process_logs` wrapper/import 호환성 검증으로 남긴다.
- runner 회귀는 기존 테스트로 유지한다.
  - `tests/test_process_logs.py`
  - `tests/test_method_summary_chart.py`
  - `tests/test_multi_scenario_timepoint_outputs.py`

## 구현 순서

1. 새 `log_processor.py`에 기존 로직을 거의 그대로 옮겨 class + wrapper를 만든다.
2. 기존 `scripts/process_logs.py`를 import forwarding 중심의 thin wrapper로 줄인다.
3. `hfs_multi_instance_runner.py` import만 새 모듈 기준으로 정리한다.
4. class 직접 테스트를 추가하고, 기존 wrapper 테스트를 필요한 만큼만 조정한다.
5. lint/test로 산출물 경로, CSV schema, wrapper 호환성을 확인한다.

## 주의할 리스크

- `scripts.process_logs` import 경로를 끊으면 기존 테스트와 외부 사용 방식이 바로 깨진다.
- `method_time_log.json` 작성이 빠지면 숨은 소비자가 있을 수 있으므로 유지가 안전하다.
- `omitted_subroutines` 적용 시점을 앞당기면 `third`가 `second`의 마지막 objective를 상속받는 현재 동작이 깨질 수 있다.
- CSV 파일명이나 long/wide 컬럼 순서가 바뀌면 `_create_rpd_summary()`와 차트 테스트가 연쇄적으로 깨진다.
- baseline column 기본값(`Instance`, `n`, `s`, `UB`)은 wrapper/class/runner 어디서 호출하든 동일해야 한다.

## 검증 계획

- `uv run pytest tests/test_log_processor.py tests/test_process_logs.py tests/test_method_summary_chart.py tests/test_multi_scenario_timepoint_outputs.py -q`
- `uv run ruff check hybridflowshop/report/log_processor.py scripts/process_logs.py hfs_multi_instance_runner.py tests/test_log_processor.py tests/test_process_logs.py`

## 완료 기준

- `scripts/process_logs.py`는 thin wrapper가 되고, 핵심 로직은 `hybridflowshop/report/log_processor.py`에 있다.
- `HfsMultiInstanceRunner.post_run_process()`의 외부 동작과 산출물은 기존과 동일하다.
- 기존 wrapper import와 CLI entrypoint가 계속 동작한다.
- 관련 회귀 테스트와 lint가 통과한다.
