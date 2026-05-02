from hybridflowshop.controller.neh_cp import NehCpConstructor


class _FakeNehContext:
    def __init__(
        self,
        *,
        remaining_before_final: float,
        final_reserve_reached: bool = False,
    ) -> None:
        self.remaining_before_final = remaining_before_final
        self.final_reserve_reached = final_reserve_reached

    def get_remaining_sec_before_final_reserve(self) -> float:
        return self.remaining_before_final

    def final_time_reserve_is_reached(self) -> bool:
        return self.final_reserve_reached


def test_neh_time_guard_skips_full_block_when_estimate_exceeds_available_time():
    constructor = NehCpConstructor(
        _FakeNehContext(remaining_before_final=30.0)  # type: ignore[arg-type]
    )

    should_skip = constructor._should_skip_full_neh_before_first_batch(
        remaining_batch_count=8,
        max_time_per_add=5.0,
        minimize_sum_ci_lex=True,
        max_time_per_add_2nd_obj=0.5,
        successor_reserve_sec=10.0,
        skip_if_estimated_neh_exceeds_remaining=True,
        full_neh_estimate_safety_factor=1.0,
    )

    assert should_skip


def test_neh_time_guard_allows_full_block_when_estimate_fits_available_time():
    constructor = NehCpConstructor(
        _FakeNehContext(remaining_before_final=60.0)  # type: ignore[arg-type]
    )

    should_skip = constructor._should_skip_full_neh_before_first_batch(
        remaining_batch_count=8,
        max_time_per_add=5.0,
        minimize_sum_ci_lex=True,
        max_time_per_add_2nd_obj=0.5,
        successor_reserve_sec=10.0,
        skip_if_estimated_neh_exceeds_remaining=True,
        full_neh_estimate_safety_factor=1.0,
    )

    assert not should_skip


def test_neh_time_guard_stops_next_batch_when_final_reserve_is_reached():
    constructor = NehCpConstructor(
        _FakeNehContext(
            remaining_before_final=0.0,
            final_reserve_reached=True,
        )  # type: ignore[arg-type]
    )

    should_stop = constructor._should_stop_before_next_batch(
        completed_batch_elapsed_sec_list=[5.0],
        stop_before_final_reserve=True,
        successor_reserve_sec=0.0,
        time_guard_estimate_safety_factor=1.0,
        time_guard_min_completed_batches=1,
        next_batch_idx=2,
        total_batch_count=8,
    )

    assert should_stop


def test_neh_time_guard_stops_next_batch_when_recent_runtime_would_consume_successor_reserve():
    constructor = NehCpConstructor(
        _FakeNehContext(remaining_before_final=16.0)  # type: ignore[arg-type]
    )

    should_stop = constructor._should_stop_before_next_batch(
        completed_batch_elapsed_sec_list=[7.0],
        stop_before_final_reserve=True,
        successor_reserve_sec=10.0,
        time_guard_estimate_safety_factor=1.0,
        time_guard_min_completed_batches=1,
        next_batch_idx=2,
        total_batch_count=8,
    )

    assert should_stop
