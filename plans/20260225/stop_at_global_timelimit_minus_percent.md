# Plan: stop_at_global_timelimit_minus_percent 추가

## Context

기존 `stop_at_global_timelimit_minus`는 절대값(초 단위)으로 타임리밋을 설정합니다.
문제 상황에 따라 적절한 상수 값을 설정하기 어려울 수 있으므로,
**전역 타임리밋의 특정 비율(%) 이하로 남았을 때** 서브루틴을 중단하는 기능이 필요합니다.

## Implementation Plan

### 1. `hybridflowshop/controller/reactive/local_stopping_criteria.py` 수정

**신규 파라미터 추가:**

- `stop_at_global_timelimit_minus_percent: float | None` (비율, 0.0 ~ 1.0)

`is_loop_stopping_condition()` 메서드에 로직 추가:

- 두 조건을 **병렬적으로** 확인
- 둘 다 설정되었으면 **엄격한(더 낮은) 조건**으로 중단

```python
# 로컬 서브루틴 중단 여부 확인 (병렬적으로 체크)
is_stopping = False
stop_reason = ""

if self.max_loop_count is not None and loop_count == self.max_loop_count:
    is_stopping = True
    stop_reason = "max_loop_count"
elif (
    self.stop_at_global_timelimit_minus is not None
    and global_remaining_sec <= self.stop_at_global_timelimit_minus
):
    is_stopping = True
    stop_reason = "stop_at_global_timelimit_minus"
elif (
    self.stop_at_global_timelimit_minus_percent is not None
    and global_remaining_sec <= self.stop_at_global_timelimit_minus_percent * self.ctrlr.stopping_criteria.timelimit
):
    is_stopping = True
    stop_reason = "stop_at_global_timelimit_minus_percent"
# ... lb_gap checks ...
```

### 2. `hybridflowshop/controller/reactive/reactive_looper.py` 수정

`_call_subroutine()`에서 subroutine 타임리밋 계산 시 비율 적용:

```python
timelimit_by_global = self.ctrlr.timer.get_remaining_sec(
    self.ctrlr.stopping_criteria.timelimit
)

# 절대값 처리 (기존)
if self.stopping_criteria.stop_at_global_timelimit_minus is not None:
    timelimit_by_global -= self.stopping_criteria.stop_at_global_timelimit_minus

# 비율 처리 (신규) - 기존 타임리밋과 비교하여 더 작은 값으로 제한
if self.stopping_criteria.stop_at_global_timelimit_minus_percent is not None:
    threshold = self.ctrlr.stopping_criteria.timelimit * self.stopping_criteria.stop_at_global_timelimit_minus_percent
    if timelimit_by_global > threshold:
        timelimit_by_global = threshold
```

### 3. Key Design Decisions

| 항목 | 결정 |
|------|------|
| **로직** | 병렬적으로 체크, 둘 다 설정 시 엄격한(더 낮은) 조건으로 중단 |
| **우선순위** | 병렬 체크 (if-elif 대신 각각 독립적으로) |
| **비율 범위** | 0.0 ~ 1.0 (예: 0.05 = 5%) |
| **기본값** | `None` (비활성화, 하위호환성 확보) |
| **동시 사용** | 두 파라미터를 동시에 사용 가능, 둘 다 설정되면 더 엄격한 조건이 적용됨 |

### 4. 동작 예시

| 시나리오 | 전역 time limit | `stop_at_global_timelimit_minus` | `stop_at_global_timelimit_minus_percent` | 결과 |
|----------|----------------|-----------------------------------|-------------------------------------------|------|
| A | 100초 | `None` | 0.05 (5%) | 남은 시간 5초 이하이면 중단 |
| B | 100초 | 10초 | `None` | 남은 시간 10초 이하이면 중단 |
| C | 100초 | 10초 | 0.05 (5%) | 남은 시간 5초 이하이면 중단 (엄격한 조건) |
| D | 100초 | 3초 | 0.1 (10%) | 남은 시간 3초 이하이면 중단 (엄격한 조건) |

### 5. 설정 파일 예시

```yaml
stopping_criteria:
  max_loop_count: 1000
  stop_at_global_timelimit_minus_percent: 0.05  # 전역 타임리밋의 5% 이하이면 서브루틴 중단
```

### 6. Test Strategy

- `tests/test_local_stopping_criteria.py`에 테스트 추가
- 100초 time limit, 5% 설정 시 5초 이하에서 중단되는지 검증
- 두 파라미터를 동시에 설정했을 때 엄격한 조건이 적용되는지 검증

### 7. Files to Modify

- `hybridflowshop/controller/reactive/local_stopping_criteria.py`
- `hybridflowshop/controller/reactive/reactive_looper.py`
- `tests/test_local_stopping_criteria.py` (test 추가)
