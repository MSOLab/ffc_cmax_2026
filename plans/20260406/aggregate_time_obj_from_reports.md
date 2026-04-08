# Plan: Use solution_manager.register() report's progress_obj_value_records for local_progress_list

## Context

현재 `subroutine_progression.json` 의 `local_progress_list` 데이터는 `add_obj_value_log()` / `extend_obj_value_log()` 를 통해 controller-level 에서 기록되고 있습니다. 

하지만 `solution_manager.register()` 에 전달되는 `final_report` (HfsSubroutineReport) 에는 이미 `progress_obj_value_records` 필드가 포함되어 있으며, 이는 subroutine 내부에서 발생한 모든 objective value 변화를 포함합니다.

사용자는 이 중복된 기록 방식을 통합하여, report 에 이미 있는 `progress_obj_value_records` 를 `local_progress_list` 생성에 재사용하고 싶습니다.

## Current State Analysis

### 데이터 흐름 (현재)

1. **Subroutine 내부** (예: `hfs_cp_lns.py:2712-2718`):
   ```python
   final_report = self._make_subroutine_report(
       elapsed_time=sub_timer.elapsed_sec,
       obj_value=obj_value,
       obj_bound=None,
       is_init=False,
       subroutine_name="pw_cp",
       progress_obj_value_records=result.sub_obj_store.obj_value_series.items(),
   )
   ```

2. **Solution Manager 등록** (line 2720-2722):
   ```python
   was_updated: bool = self.solution_manager.register(final_report, result.schedule)
   ```
   - `final_report` 가 `solution_manager.history` 에 저장됨
   - Report 에 `progress_obj_value_records` 필드 포함
   - `HfsSubroutineReport` 구조:
     - `call_context`: "4-pw_cp" 형식의 call identifier
     - `progress_time_basis`: "local" (subroutine 내부 시간 기준)

3. **Controller-level 로깅** (line 2724-2729):
   ```python
   log_time = self.timer.elapsed_sec
   self.add_obj_value_log(log_time, obj_value, is_maximize=False)
   ```
   - 별도의 전역 시간 기준 로깅 발생

4. **Progression JSON 생성** (`controller_core.py:355-398`):
   ```python
   def _record_objective_point(...):
       # _subroutine_call_progress_map 에 local_progress_list 저장
   ```

### Solution Manager History 구조

- `solution_manager.history`: **전체 실행 동안 모든 subroutine report 를 누적 저장**
- 각 `register()` 호출마다 `SolutionRecord` 가 history list 에 append 됨
- **Overwrite 되지 않음** - 모든 report 가 보존됨
- `SolutionRecord` 구조:
  ```python
  @dataclass
  class SolutionRecord:
      report: SubroutineReportT  # HfsSubroutineReport
      solution: SolutionT | None  # HybridFlowshopLiteSchedule | None
  ```

### Parallel Metadata System

`controller_core.py` 에는 이미 subroutine call 을 추적하는 메타데이터 시스템이 존재:
- `_subroutine_call_meta_list`: `call_index`, `subroutine_name`, `prefixed_subroutine_name` ("4-pw_cp"), 타이밍 정보 포함
- `_subroutine_call_progress_map`: `prefixed_subroutine_name` → progress data 매핑
- 각 report 의 `call_context` 필드가 이 `prefixed_subroutine_name` 과 일치

### 문제점

- **이중 로깅**: subroutine report 와 controller obj_store 가 같은 데이터를 별도 경로로 기록
- **시간 기준**: 
  - `progress_obj_value_records`: subroutine 내부 시간 (`progress_time_basis="local"`)
  - `local_progress_list`: `global_sec` 와 `local_sec` 모두 포함

## Proposed Approach

**`get_progression_data()` 에서 solution_manager.history 의 report 를 조회하여 `progress_obj_value_records` 사용**

### Implementation Steps

1. **Helper method 추가**: `controller_core.py` 에 report 조회 메서드
   ```python
   def _get_report_for_call(self, prefixed_name: str) -> HfsSubroutineReport | None:
       # solution_manager.history 에서 call_context 가 prefixed_name 과 일치하는 report 검색
       for record in self.solution_manager.history:
           if hasattr(record.report, 'call_context') and record.report.call_context == prefixed_name:
               return record.report
       return None
   ```

2. **`get_progression_data()` 수정** (`controller_core.py:400-426`):
   ```python
   def get_progression_data(self) -> dict:
       subroutine_calls = []
       for meta in self._subroutine_call_meta_list:
           prefixed_name = meta["prefixed_subroutine_name"]
           
           # Try to get progress from report first
           report = self._get_report_for_call(prefixed_name)
           if report and hasattr(report, 'progress_obj_value_records') and report.progress_obj_value_records:
               # Convert report's progress_obj_value_records to local_progress_list format
               local_list = [
                   {
                       "global_sec": meta["global_start_sec"] + local_time,
                       "obj_value": obj_value,
                       "call_index": meta["call_index"],
                       "prefixed_subroutine_name": prefixed_name,
                       "local_sec": local_time,
                   }
                   for local_time, obj_value in report.progress_obj_value_records
               ]
           else:
               # Fallback to existing _subroutine_call_progress_map
               local_list = self._subroutine_call_progress_map.get(prefixed_name, [])
           
           subroutine_calls.append({...})  # Rest same
   ```

3. **시간 정렬**:
   - `report.progress_obj_value_records`: `(local_time, obj_value)` 튜플 리스트
   - `global_sec = global_start_sec + local_time` 로 변환
   - `local_sec = local_time` (이미 subroutine 내부 시간)

### Critical Files

- `/home/hjt/code/hybridflowshop/hybridflowshop/controller/controller_core.py`
  - `_get_report_for_call()` 메서드 추가
  - `get_progression_data()` 메서드 수정 (line 400-426)
  
- `/home/hjt/code/hybridflowshop/hybridflowshop/controller/hfs_cp_lns.py`
  - `_make_subroutine_report()` (line 41-61) - 이미 `progress_obj_value_records` 전달 중

### Existing Patterns to Reuse

- `solution_manager.history` 접근: `hfs_single_instance_runner.py:287` 에서 이미 사용
  ```python
  reports=[r.report for r in self.ctrlr.solution_manager.history]
  ```
- `call_context` 매칭: report 의 `call_context` 필드가 `_subroutine_call_meta_list` 의 `prefixed_subroutine_name` 과 일치

## Verification

1. **Existing tests 확인**:
   ```bash
   uv run pytest tests/controller/test_subroutine_progression_recorder.py -v
   ```

2. **JSON output 검증**:
   ```bash
   uv run python main.py  # 실행 후
   cat Outputs_scenarios/*/subroutine_progression.json
   ```
   - `local_progress_list` 데이터가 report 의 `progress_obj_value_records` 와 일치하는지 확인
   - `global_sec`, `local_sec` 계산이 올바른지 확인

3. **이중 로깅 제거 확인**:
   - `_record_objective_point()` 가 여전히 호출되는지 확인 (fallback 용도로는 유지)
   - 또는 `add_obj_value_log()` 호출을 줄일지 결정

## User-Confirmed Decisions

- **Fallback 유지**: `get_progression_data()` 에서 report 를 primary source 로 사용하되, fallback 으로 기존 `_subroutine_call_progress_map` 로직 유지
- **`add_obj_value_log()` 유지**: 현재 호출 경로를 변경하지 않음
- **`progress_obj_value_records` vs `obj_value_series`**: list 와 tuple 의 차이일 뿐 내용은 동일하므로 문제없음
