from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController


class FakeIncrementalSchedule:
    def __init__(self, makespan: int = 10):
        self.stages = ["S1", "S2"]
        self.makespan = makespan
        self._stage_2_ops = {
            "S1": [
                ("M1", 0, 1, "J1"),
                ("M1", 1, 2, "J2"),
                ("M1", 2, 3, "J3"),
                ("M1", 3, 4, "J4"),
            ],
            "S2": [
                ("M1", 1, 2, "J1"),
                ("M1", 2, 3, "J2"),
                ("M1", 3, 4, "J3"),
                ("M1", 4, 5, "J4"),
            ],
        }

    def iter_operations_on_stage(self, stage_id):
        return iter(self._stage_2_ops[stage_id])


def _make_controller(schedule: FakeIncrementalSchedule):
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=4, stage_count=2)
    ctrl.solution_manager = SimpleNamespace(get_incumbent=lambda: schedule)
    ctrl.is_stopping_condition = lambda: False
    seen_contexts = []

    @contextmanager
    def fake_context(name):
        seen_contexts.append(name)
        yield

    ctrl.temporarily_extended_context = fake_context
    return ctrl, seen_contexts


def test_incremental_pw_cp_always_runs_each_count_once_and_clamps_max():
    schedule = FakeIncrementalSchedule()
    ctrl, seen_contexts = _make_controller(schedule)
    seen_kwargs = []

    def fake_pw_cp(**kwargs):
        seen_kwargs.append(kwargs)

    ctrl.pw_cp = fake_pw_cp
    ctrl.repeat_while_improvement = lambda *args, **kwargs: pytest.fail(
        "repeat_while_improvement should not be used for the always policy"
    )

    ctrl.incremental_pw_cp(
        solver_thread_cnt=8,
        batch_size=2,
        batch_size_ratio=0.25,
        unfixed_batch_count_min=1,
        unfixed_batch_count_max=5,
        increment_unfixed_batch_count_flag="always",
        lr_profile_fixed_batch_count=1,
        enable_promotion_profile_fixed=True,
        use_lns_only=False,
    )

    assert [kwargs["unfixed_batch_count"] for kwargs in seen_kwargs] == [1, 2]
    assert seen_kwargs[0]["solver_thread_cnt"] == 8
    assert seen_kwargs[0]["batch_size"] == 2
    assert seen_kwargs[0]["batch_size_ratio"] == 0.25
    assert seen_kwargs[0]["lr_profile_fixed_batch_count"] == 1
    assert seen_kwargs[0]["enable_promotion_profile_fixed"] is True
    assert seen_contexts == ["batch_001", "batch_002"]


def test_incremental_pw_cp_if_no_improvement_retries_same_count_until_stop():
    schedule = FakeIncrementalSchedule(makespan=100)
    ctrl, seen_contexts = _make_controller(schedule)
    seen_counts = []
    count_2_call_count = {1: 0, 2: 0}
    target_makespans = {
        1: [90, 90],
        2: [80, 80],
    }

    def fake_run_flow(routine_data):
        payload = routine_data.to_obj()
        unfixed_batch_count = payload["unfixed_batch_count"]
        seen_counts.append(unfixed_batch_count)
        idx = count_2_call_count[unfixed_batch_count]
        schedule.makespan = target_makespans[unfixed_batch_count][idx]
        count_2_call_count[unfixed_batch_count] += 1

    ctrl._run_flow = fake_run_flow
    ctrl.pw_cp = lambda **kwargs: pytest.fail(
        "pw_cp should be driven through repeat_while_improvement for this policy"
    )

    ctrl.incremental_pw_cp(
        solver_thread_cnt=4,
        batch_size=2,
        unfixed_batch_count_min=1,
        unfixed_batch_count_max=2,
        increment_unfixed_batch_count_flag="if_no_improvement",
        profile_fix_by_machine=True,
    )

    assert seen_counts == [1, 1, 2, 2]
    assert seen_contexts == [
        "batch_001",
        "reps_001",
        "reps_002",
        "batch_002",
        "reps_001",
        "reps_002",
    ]


def test_incremental_pw_cp_rejects_invalid_flag():
    schedule = FakeIncrementalSchedule()
    ctrl, _seen_contexts = _make_controller(schedule)

    with pytest.raises(
        ValueError, match="increment_unfixed_batch_count_flag must be one of"
    ):
        ctrl.incremental_pw_cp(
            solver_thread_cnt=8,
            batch_size=2,
            unfixed_batch_count_min=1,
            unfixed_batch_count_max=2,
            increment_unfixed_batch_count_flag="sometimes",
        )


def test_incremental_pw_cp_rejects_invalid_range():
    schedule = FakeIncrementalSchedule()
    ctrl, _seen_contexts = _make_controller(schedule)

    with pytest.raises(
        ValueError, match="unfixed_batch_count_max must be >= unfixed_batch_count_min"
    ):
        ctrl.incremental_pw_cp(
            solver_thread_cnt=8,
            batch_size=2,
            unfixed_batch_count_min=2,
            unfixed_batch_count_max=1,
        )


def test_incremental_pw_cp_rejects_min_above_actual_batch_count():
    schedule = FakeIncrementalSchedule()
    ctrl, _seen_contexts = _make_controller(schedule)

    with pytest.raises(
        ValueError,
        match="unfixed_batch_count_min exceeds the available PW-CP batch count",
    ):
        ctrl.incremental_pw_cp(
            solver_thread_cnt=8,
            batch_size=2,
            unfixed_batch_count_min=3,
            unfixed_batch_count_max=4,
        )
