from pathlib import Path
from types import SimpleNamespace

import pytest

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class FakeCriticalSchedule:
    def __init__(
        self,
        critical_blocks,
        jobs=None,
        stages=None,
        makespan: int = 10,
        start_map=None,
        end_map=None,
    ):
        self._critical_blocks = critical_blocks
        self.jobs = jobs or sorted({op[0] for block in critical_blocks for op in block})
        self.stages = stages or sorted(
            {op[1] for block in critical_blocks for op in block}
        )
        self.makespan = makespan
        self._start_map = start_map or {}
        self._end_map = end_map or {}

    def make_semi_active(self, *_args, **_kwargs):
        return None

    def find_critical_blocks(self, *_args, **_kwargs):
        return self._critical_blocks

    def get_jik_2_start_time_map(self):
        return self._start_map

    def get_jik_2_end_time_map(self):
        return self._end_map

    def iter_operations_on_stage(self, stage_id):
        for op in self._start_map:
            if op[1] == stage_id:
                yield op


def _make_controller(duration=None) -> HybridFlowShopCpLnsController:
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.stage_2_job_2_p_dict = duration or {"S1": {"J1": 1}}
    return ctrl


def _build_actual_schedule():
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2", "J3"],
        stages=["S1", "S2"],
        machines_per_stage={"S1": ["M1"], "S2": ["M1"]},
    )
    sched.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=1)
    sched.add_ops_times_2_mc("S1", "M1", "J2", start_time=1, end_time=2)
    sched.add_ops_times_2_mc("S1", "M1", "J3", start_time=2, end_time=3)
    sched.add_ops_times_2_mc("S2", "M1", "J1", start_time=1, end_time=2)
    sched.add_ops_times_2_mc("S2", "M1", "J2", start_time=2, end_time=3)
    sched.add_ops_times_2_mc("S2", "M1", "J3", start_time=3, end_time=4)
    duration = {
        "S1": {"J1": 1, "J2": 1, "J3": 1},
        "S2": {"J1": 1, "J2": 1, "J3": 1},
    }
    return sched, duration


def test_select_critical_jobs_uses_weighted_block_occurrence_counts(monkeypatch):
    ctrl = _make_controller()
    schedule = FakeCriticalSchedule(
        critical_blocks=[
            [("J1", "S1", "M1"), ("J2", "S1", "M1")],
            [("J1", "S2", "M1")],
        ],
        jobs=["J1", "J2"],
        stages=["S1", "S2"],
    )
    captured = {}

    def fake_choices(population, weights, k):
        captured["population"] = list(population)
        captured["weights"] = list(weights)
        assert k == 1
        return [population[0]]

    monkeypatch.setattr("hybridflowshop.controller.hfs_cp_lns.random.choices", fake_choices)

    selected = ctrl._select_critical_jobs(schedule, 1, "weighted_random")

    assert selected == ["J1"]
    assert captured["population"] == ["J1", "J2"]
    assert captured["weights"] == [2.0, 1.0]


def test_select_critical_jobs_critical_adjacency_expands_from_weighted_seed(
    monkeypatch,
):
    ctrl = _make_controller()
    schedule = FakeCriticalSchedule(
        critical_blocks=[
            [("J1", "S1", "M1"), ("J2", "S1", "M1"), ("J3", "S1", "M1")],
            [("J1", "S2", "M1"), ("J2", "S2", "M1"), ("J3", "S2", "M1")],
        ],
        jobs=["J1", "J2", "J3"],
        stages=["S1", "S2"],
    )

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.choices",
        lambda population, weights, k: ["J2"],
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.shuffle",
        lambda seq: None,
    )

    selected = ctrl._select_critical_jobs(schedule, 3, "critical_adjacency")

    assert selected == ["J2", "J1", "J3"]


def test_select_critical_jobs_logs_seed_for_critical_adjacency(
    monkeypatch, caplog
):
    ctrl = _make_controller()
    schedule = FakeCriticalSchedule(
        critical_blocks=[
            [("J1", "S1", "M1"), ("J2", "S1", "M1"), ("J3", "S1", "M1")],
        ],
        jobs=["J1", "J2", "J3"],
        stages=["S1"],
    )

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.choices",
        lambda population, weights, k: ["J2"],
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.shuffle",
        lambda seq: None,
    )

    with caplog.at_level("INFO"):
        selected = ctrl._select_critical_jobs(schedule, 2, "critical_adjacency")

    assert selected[0] == "J2"
    assert "seed_job=J2" in caplog.text


def test_apply_critical_job_operator_frees_all_ops_of_selected_jobs(monkeypatch):
    schedule, duration = _build_actual_schedule()
    ctrl = _make_controller(duration)
    ctrl.solution_manager = SimpleNamespace(
        has_incumbent=lambda: True,
        get_incumbent=lambda: schedule,
    )
    monkeypatch.setattr(
        ctrl,
        "_select_critical_jobs",
        lambda incumbent, job_count, job_selection_policy: ["J1", "J3"],
    )

    captured = {}

    def fake_fix(selected_ops, **_kwargs):
        captured["selected_ops"] = selected_ops

    monkeypatch.setattr(ctrl, "_fix_operations_profile_except_selected", fake_fix)

    ctrl.apply_critical_job_operator(2, job_selection_policy="critical_adjacency")

    assert captured["selected_ops"] == {
        ("J1", "S1", "M1"),
        ("J1", "S2", "M1"),
        ("J3", "S1", "M1"),
        ("J3", "S2", "M1"),
    }


def test_select_critical_jobs_caps_at_total_candidate_jobs(monkeypatch):
    ctrl = _make_controller()
    schedule = FakeCriticalSchedule(
        critical_blocks=[
            [("J1", "S1", "M1"), ("J2", "S1", "M1"), ("J3", "S1", "M1")],
        ],
        jobs=["J1", "J2", "J3"],
        stages=["S1"],
    )

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.choices",
        lambda population, weights, k: [population[0]],
    )

    selected = ctrl._select_critical_jobs(schedule, 10, "weighted_random")

    assert set(selected) == {"J1", "J2", "J3"}
    assert len(selected) == 3


def test_select_critical_jobs_logs_warning_and_falls_back_when_no_critical_blocks(
    monkeypatch, caplog
):
    ctrl = _make_controller()
    schedule = FakeCriticalSchedule(
        critical_blocks=[],
        jobs=["J1", "J2"],
        stages=["S1", "S2"],
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.sample",
        lambda population, k: list(population)[:k],
    )

    with caplog.at_level("WARNING"):
        selected = ctrl._select_critical_jobs(schedule, 1, "critical_adjacency")

    assert selected == ["J1"]
    assert "No critical blocks found after make_semi_active" in caplog.text


def test_critical_job_ns_draws_highlighted_gantts_before_and_after(monkeypatch):
    before_schedule, duration = _build_actual_schedule()
    after_schedule, _ = _build_actual_schedule()
    ctrl = _make_controller(duration)
    ctrl.solution_manager = SimpleNamespace(get_incumbent=lambda: before_schedule)
    ctrl.get_file_path_for_subroutine = lambda suffix: Path(f"/tmp/{suffix}")

    monkeypatch.setattr(
        ctrl,
        "apply_critical_job_operator",
        lambda *args, **kwargs: ["J1", "J3"],
    )

    captured_calls = []

    def fake_draw_gantt(schedule, output_path=None, highlight_op_set=None, **kwargs):
        captured_calls.append(
            {
                "schedule": schedule,
                "output_path": output_path,
                "highlight_op_set": highlight_op_set,
                "force_end": kwargs.get("force_end"),
            }
        )

    monkeypatch.setattr(ctrl, "draw_gantt", fake_draw_gantt)

    def fake_fix_profile_solve_reset(
        profile_fixing_method,
        computational_time,
        solver_thread_cnt,
        **kwargs,
    ):
        assert computational_time == 5
        assert solver_thread_cnt == 8
        profile_fixing_method()
        kwargs["pre_solve_visualizer"]()
        kwargs["post_solve_visualizer"](after_schedule)

    monkeypatch.setattr(ctrl, "_fix_profile_solve_reset", fake_fix_profile_solve_reset)

    ctrl.critical_job_ns(
        job_count=2,
        computational_time=5,
        solver_thread_cnt=8,
        draw_gantt=True,
    )

    assert len(captured_calls) == 2
    assert str(captured_calls[0]["output_path"]).endswith(
        "_gantt_critical_job_before.png"
    )
    assert str(captured_calls[1]["output_path"]).endswith(
        "_gantt_critical_job_after.png"
    )
    assert captured_calls[0]["highlight_op_set"] == {
        ("J1", "S1"),
        ("J1", "S2"),
        ("J3", "S1"),
        ("J3", "S2"),
    }
    assert captured_calls[1]["highlight_op_set"] == {
        ("J1", "S1"),
        ("J1", "S2"),
        ("J3", "S1"),
        ("J3", "S2"),
    }
