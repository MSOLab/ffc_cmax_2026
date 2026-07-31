from contextlib import nullcontext
from types import SimpleNamespace

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class _FakeSolutionManager:
    def __init__(self, schedule: HybridFlowshopLiteSchedule):
        self._schedule = schedule
        self.best_obj_value = float(schedule.makespan)

    def get_incumbent(self):
        return self._schedule


def _build_schedule() -> HybridFlowshopLiteSchedule:
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2", "J3"],
        stages=["S0", "S1", "S2"],
        machines_per_stage={"S0": ["M1"], "S1": ["M1"], "S2": ["M1"]},
    )
    sched.add_ops_times_2_mc("S0", "M1", "J1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("S0", "M1", "J2", start_time=4, end_time=6)
    sched.add_ops_times_2_mc("S0", "M1", "J3", start_time=8, end_time=10)
    sched.add_ops_times_2_mc("S1", "M1", "J1", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("S1", "M1", "J2", start_time=6, end_time=8)
    sched.add_ops_times_2_mc("S1", "M1", "J3", start_time=10, end_time=12)
    sched.add_ops_times_2_mc("S2", "M1", "J1", start_time=4, end_time=6)
    sched.add_ops_times_2_mc("S2", "M1", "J2", start_time=8, end_time=10)
    sched.add_ops_times_2_mc("S2", "M1", "J3", start_time=12, end_time=14)
    return sched


def _make_controller(schedule: HybridFlowshopLiteSchedule):
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(
        job_count=len(schedule.jobs),
        stage_count=len(schedule.stages),
        stage_id_list=list(schedule.stages),
    )
    ctrl.solution_manager = _FakeSolutionManager(schedule)
    ctrl.stage_2_job_2_p_dict = {
        stage_id: {job_id: 2 for job_id in schedule.jobs}
        for stage_id in schedule.stages
    }
    return ctrl


def test_select_ops_in_stage_time_window_intersects_stage_band_and_time_window():
    schedule = _build_schedule()
    ctrl = _make_controller(schedule)

    selected = ctrl._select_ops_in_stage_time_window(
        schedule,
        selected_stage_ids=["S1"],
        window_start=5,
        window_end=9,
        overlap_mode="intersect",
    )

    assert selected == {("J2", "S1", "M1")}


def test_resolve_stage_time_window_centers_combines_critical_and_bottleneck(
    monkeypatch,
):
    schedule = _build_schedule()
    ctrl = _make_controller(schedule)
    monkeypatch.setattr(
        ctrl,
        "_resolve_critical_stage_id_for_targeted_ns",
        lambda **_kwargs: "S2",
    )
    monkeypatch.setattr(
        ctrl,
        "_resolve_bottleneck_stage_id_for_targeted_ns",
        lambda: "S1",
    )

    centers = ctrl._resolve_stage_time_window_center_stage_ids(
        stage_center_policy="critical_and_retained_bottleneck",
        max_stage_centers=2,
        include_singleton_critical_blocks=True,
    )

    assert centers == ["S2", "S1"]


def test_stage_time_window_sweep_ns_repeats_small_stage_time_solves(monkeypatch):
    schedule = _build_schedule()
    ctrl = _make_controller(schedule)
    captured_selected_ops = []
    contexts = []

    monkeypatch.setattr(
        ctrl, "temporarily_extended_context", lambda name: nullcontext()
    )
    monkeypatch.setattr(
        ctrl,
        "_resolve_stage_time_window_center_stage_ids",
        lambda **_kwargs: ["S1"],
    )

    def fake_fix_time_window(_schedule, selected_ops, **_kwargs):
        captured_selected_ops.append(set(selected_ops))

    def fake_solve(profile_fixing_method, *_args, **_kwargs):
        profile_fixing_method()

    def fake_context(name):
        contexts.append(name)
        return nullcontext()

    monkeypatch.setattr(ctrl, "temporarily_extended_context", fake_context)
    monkeypatch.setattr(
        ctrl,
        "_fix_time_window_profile_except_selected",
        fake_fix_time_window,
    )
    monkeypatch.setattr(ctrl, "_fix_profile_solve_reset", fake_solve)

    ctrl.stage_time_window_sweep_ns(
        solver_thread_cnt=16,
        computational_time=0.1,
        stage_center_policy="critical_and_retained_bottleneck",
        stage_radius=0,
        window_size=4,
        window_size_ratio=None,
        step_size=4,
        step_size_ratio=None,
        max_window_count_per_stage_center=3,
        tail_window_bias=False,
    )

    assert len(captured_selected_ops) == 3
    assert len(contexts) == 3
    assert all(selected for selected in captured_selected_ops)
    assert all(op[1] == "S1" for selected in captured_selected_ops for op in selected)


def test_stage_band_local_time_windows_use_selected_stage_time_range():
    schedule = _build_schedule()
    ctrl = _make_controller(schedule)

    windows = ctrl._build_stage_band_local_time_windows(
        schedule,
        selected_stage_ids=["S1"],
        window_size=4,
        window_size_ratio=None,
        step_size=4,
        step_size_ratio=None,
        max_window_count=3,
        tail_window_bias=False,
        min_tail_start_ratio=None,
    )

    assert windows == [(2, 6), (6, 10), (8, 12)]


def test_resolve_stage_time_window_centers_can_include_tail_stage(monkeypatch):
    schedule = _build_schedule()
    ctrl = _make_controller(schedule)
    monkeypatch.setattr(
        ctrl,
        "_resolve_critical_stage_id_for_targeted_ns",
        lambda **_kwargs: "S1",
    )

    centers = ctrl._resolve_stage_time_window_center_stage_ids(
        stage_center_policy="critical_tail",
        max_stage_centers=None,
        include_singleton_critical_blocks=True,
    )

    assert centers == ["S1", "S2"]
