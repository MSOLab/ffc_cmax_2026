# Wave Batch Dispatching 구현 메모

## 개요

`schedule_lite.py`에 wave batch dispatching 구현. 입력으로 `job_ids`, `stage_ids`, `batch_size`, `stage_2_job_2_duration`, (optional) `job_2_release_time`을 받음.

## 핵심 아이디어

- 각 stage마다 "해당 stage에는 이미 디스패치되었고, 다음 stage로는 아직 안 넘어간 job"을 완료시간 기준으로 뽑기 위해 **min-heap** 사용
- `add_operation_2_stage(stage, job, dur, release_t=...)`가 precedence(이전 stage end) + idle-gap을 고려해 start/end를 확정해주므로, **end time을 heap key**로 쓰면 "가장 빨리 끝나는 순서"를 쉽게 유지 가능
- heap에는 `(end_time, tie_breaker, job_id, stage_idx)` 저장하여 deterministic 결과 유지

## 구현 세부사항

### WaveBatchState Dataclass

```python
@dataclass
class WaveBatchState:
    ready_heaps: list[list[ReadyHeapEntry]]
    job_2_duration: dict[int, Mapping[JobIdType, int]]
    job_2_pos: dict[JobIdType, int]
    iteration_no: int
    total_jobs: int
    stage_ids: Sequence[StageIdType]
    _prev_scheduled_last_stage: int = 0
```

### Algorithm Flow (Iteration 기반)

**초기화:**
- `ready_heap = [empty_heap for _ in stage_ids]`
- `iter_no = 0`

**Main Loop:**
1. `iter_no += 1`
2. **Feed Stage 1:** `batch = job_ids[(iter_no-1)*B : iter_no*B]` 투입
   - `release_time`은 stage 1 투입에만 적용
   - `add_operation_2_stage(stage1, job, dur, release_t=rel)`
3. **Promote:** `depth = min(iter_no - 1, num_stages - 1)`만큼 연쇄 promote
   - stage 0→1, 1→2, ... 까지 promote
   - `add_operation_2_stage(next_stage, job, dur)`  (**release_t 전달 안 함**)
4. **Termination Check:**
   - `no_more_injection and all(heap empty for all but last stage)`

### Helper Methods

| Method | Purpose |
|--------|---------|
| `_get_scheduled_count(stage_idx)` | Count jobs scheduled at a stage |
| `_push_ready(stage_idx, job_id, tie_breaker)` | Push job to ready heap |
| `_promote(stage_idx, batch_size)` | Promote jobs from stage i to i+1 |
| `_feed_stage1(batch_jobs, job_2_release)` | Inject batch to stage 1 |

### 유효성 검사

- `batch_size > 0` 여야 함
- `stage_ids`는 `self.stages`의 contiguous subsequence여야 함
- 모든 `(stage_id, job_id)` 쌍에 대해 duration이 있어야 함

### 추가 구현 (서브루틴 계획과 다름)

1. **`stage_2_prev_stage` 필드 추가**: `_get_prev_stage_end_time` 최적화 위해
2. **`_feed_stage1` 시그니처 변경**: `stage_id` 파라미터 추가 (인라인로직 제거)
3. **종료 조건**: plan에 있는 `job_ptr` 대신 `end_idx >= num_jobs` 사용
4. **heap empty check**: `all_empty = all(len(h) == 0 for h in ready_heaps[:-1])`

## 엣지 케이스 처리

- job이 B 미만 남았으면 남은 만큼만 투입
- 어떤 stage heap이 비어 promote를 B개 못 채우면 가능한 만큼만 promote
- duration 맵에 (stage, job)이 누락된 경우 예외 처리
- 동일 end time tie는 입력 순서로 고정

## File Changes

| File | Changes |
|------|---------|
| `hybridflowshop/schedule_lite.py` | `WaveBatchState` dataclass, `ReadyHeapEntry` type alias, `stage_2_prev_stage` 필드, `_get_scheduled_count`, `_push_ready`, `_promote`, `_feed_stage1`, `dispatch_wave_batches` 추가 |
| `tests/test_schedule_lite.py` | 6개 테스트 추가: basic, release times, batch_size=1, large batch, 3-stage, error cases |
| `schedule_lite_todo.md` | This file - update to match implementation |
