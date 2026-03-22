# Schedule Creation Method Implementation Plan

## Context

PW-CP 모델의 `create_schedule` 함수 (`hybridflowshop/cpsat_model_2/pw_cp.py` 줄 387-405) 가 현재 incomplete 상태로, return statement 만 있습니다. 이 함수는 CP 모델 해로부터 실제 schedule 을 생성해야 합니다.

사용자가 설명한 3 단계 schedule creation 프로세스:

1. **Left-time-fixed operations**: 각 stage별로 machine별로 left_bar_interval 에 해당하는 operation 을 right_justified_schedule 에서 찾아 그대로 dispatch
2. **Remaining operations**: 시작시각 오름차순 → 종료시각 내림차순 정렬 후 stage-wise dispatch
3. **Right-time-fixed operations**: 나머지 operation 종료시간 최대값이 가장 작은 machine 에 right_bar_interval 에 해당하는 operation dispatch

## Implementation Approach

### Phase 1: Left-time-fixed Operations Dispatch

**Algorithm:**

1. 각 stage `i` 를 순회
2. 해당 stage 의 `partition.left_time_fixed` 에서 machine ID 로 그룹화
3. 각 machine `mc_id` 에 대해:
   - `left_bar_interval[i][mc_id]` 에서 interval 변수 추출
   - `right_justified_schedule` 에서 해당 machine 에 배치된 left-time-fixed operations 찾기
   - `dummy_bar_vars.left_bar_interval` 의 machine ID 가 해당 machine 을 지시
   - partition 의 left_time_fixed 를 순회하며 tuple 의 두번째 요소가 machine ID 와 일치하는 job ID 찾기
   - 찾은 operations 을 `add_ops_times_2_mc` 로 dispatch (start, end times 는 right_justified_schedule 에서 가져옴)
4. Assertion: 해당 machine 의 첫 operation 시작시간 == left_bar_interval 시작 (0), 마지막 operation 종료시간 == left_bar_interval 끝

**Key Files:**

- `hybridflowshop/cpsat_model_2/pw_cp.py`: `create_schedule` 함수 구현
- `hybridflowshop/schedule_lite.py`: `add_ops_times_2_mc` method 사용

### Phase 2: Remaining Operations Dispatch

**Algorithm:**

1. 각 stage `i` 에서 `partition.unfixed` operations 추출
2. 각 operation 에 대해 CP model solution 에서 start, end times 추출 (`variables.op_start`, `variables.op_end`)
3. Operations 를 (start_time asc, end_time desc) 로 정렬
4. 정렬된 순서대로 `add_operation_2_stage` 로 dispatch (자동 machine selection)

### Phase 3: Right-time-fixed Operations Dispatch

**Algorithm:**

1. `dummy_bar_vars.right_bar_init_start`의 machine들을 initial start time 오름차순 정렬 (`sorted_mcs_by_start`)
   - right-time-fixed operation 이 없는 machine은 `right_boundary = horizon` 으로 가장 후순위
2. `partition.right_time_fixed`를 machine ID로 그룹화하여 `mc_2_rtf_ops` map 생성
   - 각 machine에 대해 right-time-fixed operation 목록을 start time 순으로 정렬
3. `sorted_mcs_by_start` 순서대로 `source_mc` 를 순회:
   - `source_mc`에 정의된 모든 right-time-fixed operation(`rtf_ops`)을 가져옴
     - Right-time-fixed operation이 없는 경우 해당 source_mc 건너뜀
   - 아직 right-time-fixed operation을 dispatch하지 않은 machines 중 `get_machine_latest_end_time`이 최소인 `target_mc` 선택
   - `rtf_ops`의 **모든** operation 을 하나의 `target_mc`에 배치 (add_ops_times_2_mc)

## Key Functions and Methods to Reuse

- `hybridflowshop/schedule_lite.py:436` - `add_ops_times_2_mc`: 명시적 start/end 시간으로 operation 추가
- `hybridflowshop/schedule_lite.py:564` - `add_operation_2_stage`: 자동 machine selection 으로 operation 추가
- `hybridflowshop/schedule_lite.py:145` - `get_machine_latest_end_time`: machine 의 마지막 operation 종료시간 조회
- `hybridflowshop/cpsat_model_2/pw_cp.py:148` - `DummyBarVars.left_bar_interval`: stage→machine→IntervalVar map
- `hybridflowshop/cpsat_model_2/pw_cp.py:136` - `OperationPartition`: left_time_fixed, non-time-fixed, right_time_fixed operations partition

## Critical Files to Modify

- `hybridflowshop/cpsat_model_2/pw_cp.py`: `create_schedule` 함수 (줄 387-405) 완전한 구현

## Verification Steps

1. **Unit test 작성**: `tests/controller/test_pw_cp.py` 에 schedule creation 테스트 추가
   - Left-time-fixed operations 가 correctly dispatch 되는지 확인
   - Remaining operations 이 correctly sorted 되고 dispatch 되는지 확인
   - Right-time-fixed operations 이 correctly dispatched 되는지 확인

2. **Assertion checks**:
   - Left-time-fixed: 각 machine 의 min start == left_bar_interval start, max end == left_bar_interval end
   - Right-time-fixed: 각 machine 의 right-time-fixed operation 이 correctly positioned

3. **Integration test**: 전체 PW-CP workflow 테스트하여 schedule creation 이후 CP 모델이 correctly solved 되는지 확인
