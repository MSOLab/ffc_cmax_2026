from pathlib import Path
from types import SimpleNamespace

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


def _build_stage_band_schedule():
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2"],
        stages=["S0", "S1", "S2", "S3", "S4"],
        machines_per_stage={
            "S0": ["M1"],
            "S1": ["M1"],
            "S2": ["M1"],
            "S3": ["M1"],
            "S4": ["M1"],
        },
    )
    for stage_idx, stage_id in enumerate(sched.stages):
        sched.add_ops_times_2_mc(stage_id, "M1", "J1", start_time=stage_idx * 2, end_time=stage_idx * 2 + 1)
        sched.add_ops_times_2_mc(stage_id, "M1", "J2", start_time=stage_idx * 2 + 1, end_time=stage_idx * 2 + 2)
    return sched


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


def test_apply_retained_cp_bottleneck_band_stage_operator_frees_band_ops(monkeypatch):
    schedule = _build_stage_band_schedule()
    ctrl = _make_controller(
        {
            stage_id: {"J1": 1, "J2": 1}
            for stage_id in schedule.stages
        }
    )
    ctrl.instance = SimpleNamespace(stage_id_list=list(schedule.stages))
    ctrl.solution_manager = SimpleNamespace(
        has_incumbent=lambda: True,
        get_incumbent=lambda: schedule,
    )
    ctrl.last_retained_cp_lb_result = SimpleNamespace(
        selected_bottleneck_stage_ids=["S2"],
        bottleneck_stage_id="S2",
    )

    captured = {}

    def fake_fix(selected_ops, **_kwargs):
        captured["selected_ops"] = selected_ops

    monkeypatch.setattr(ctrl, "_fix_operations_profile_except_selected", fake_fix)

    selected_stages = ctrl.apply_retained_cp_bottleneck_band_stage_operator(radius=1)

    assert selected_stages == ["S1", "S2", "S3"]
    assert captured["selected_ops"] == {
        ("J1", "S1", "M1"),
        ("J2", "S1", "M1"),
        ("J1", "S2", "M1"),
        ("J2", "S2", "M1"),
        ("J1", "S3", "M1"),
        ("J2", "S3", "M1"),
    }


def test_retained_cp_bottleneck_band_stage_ns_uses_tl_nc_multiplier(monkeypatch):
    ctrl = _make_controller()
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=4)

    captured = {}

    def fake_fix_profile_solve_reset(
        profile_fixing_method,
        computational_time,
        solver_thread_cnt,
        **_kwargs,
    ):
        captured["computational_time"] = computational_time
        captured["solver_thread_cnt"] = solver_thread_cnt

    monkeypatch.setattr(ctrl, "_fix_profile_solve_reset", fake_fix_profile_solve_reset)

    ctrl.retained_cp_bottleneck_band_stage_ns(
        solver_thread_cnt=8,
        computational_time=999,
        tl_nc_multiplier=0.5,
    )

    assert captured["computational_time"] == 20.0
    assert captured["solver_thread_cnt"] == 8


def test_stage_block_ns_uses_tl_nc_multiplier(monkeypatch):
    ctrl = _make_controller()
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=4)

    captured = {}

    def fake_fix_profile_solve_reset(
        profile_fixing_method,
        computational_time,
        solver_thread_cnt,
        **_kwargs,
    ):
        captured["computational_time"] = computational_time
        captured["solver_thread_cnt"] = solver_thread_cnt

    monkeypatch.setattr(ctrl, "_fix_profile_solve_reset", fake_fix_profile_solve_reset)

    ctrl.stage_block_ns(
        rho=0.25,
        solver_thread_cnt=8,
        computational_time=999,
        tl_nc_multiplier=0.5,
    )

    assert captured["computational_time"] == 20.0
    assert captured["solver_thread_cnt"] == 8


def test_random_stage_band_operator_selects_random_center_band(monkeypatch):
    schedule = _build_stage_band_schedule()
    ctrl = _make_controller(
        {
            stage_id: {"J1": 1, "J2": 1}
            for stage_id in schedule.stages
        }
    )
    ctrl.instance = SimpleNamespace(stage_id_list=list(schedule.stages))
    ctrl.solution_manager = SimpleNamespace(
        has_incumbent=lambda: True,
        get_incumbent=lambda: schedule,
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.choice",
        lambda seq: "S2",
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.randint",
        lambda _low, _high: 1,
    )

    captured = {}

    def fake_fix(selected_ops, **_kwargs):
        captured["selected_ops"] = selected_ops

    monkeypatch.setattr(ctrl, "_fix_operations_profile_except_selected", fake_fix)

    selected_stages = ctrl.apply_random_stage_band_stage_operator(
        radius=2,
        min_radius=0,
    )

    assert selected_stages == ["S1", "S2", "S3"]
    assert captured["selected_ops"] == {
        ("J1", "S1", "M1"),
        ("J2", "S1", "M1"),
        ("J1", "S2", "M1"),
        ("J2", "S2", "M1"),
        ("J1", "S3", "M1"),
        ("J2", "S3", "M1"),
    }


def test_random_stage_band_stage_ns_uses_tl_nc_multiplier(monkeypatch):
    ctrl = _make_controller()
    ctrl.instance = SimpleNamespace(job_count=12, stage_count=5)

    captured = {}

    def fake_fix_profile_solve_reset(
        profile_fixing_method,
        computational_time,
        solver_thread_cnt,
        **_kwargs,
    ):
        captured["computational_time"] = computational_time
        captured["solver_thread_cnt"] = solver_thread_cnt

    monkeypatch.setattr(ctrl, "_fix_profile_solve_reset", fake_fix_profile_solve_reset)

    ctrl.random_stage_band_stage_ns(
        solver_thread_cnt=6,
        computational_time=999,
        tl_nc_multiplier=0.25,
        radius=2,
    )

    assert captured["computational_time"] == 15.0
    assert captured["solver_thread_cnt"] == 6


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


def test_select_critical_tail_jobs_uses_only_tail_stage_blocks(monkeypatch):
    ctrl = _make_controller()
    ctrl.instance = SimpleNamespace(stage_id_list=["S1", "S2", "S3"])
    schedule = FakeCriticalSchedule(
        critical_blocks=[
            [("J1", "S1", "M1"), ("J2", "S1", "M1")],
            [("J3", "S3", "M1"), ("J4", "S3", "M1")],
        ],
        jobs=["J1", "J2", "J3", "J4"],
        stages=["S1", "S2", "S3"],
        start_map={
            ("J1", "S1", "M1"): 0,
            ("J2", "S1", "M1"): 1,
            ("J3", "S3", "M1"): 2,
            ("J4", "S3", "M1"): 3,
        },
        end_map={
            ("J1", "S1", "M1"): 1,
            ("J2", "S1", "M1"): 2,
            ("J3", "S3", "M1"): 3,
            ("J4", "S3", "M1"): 4,
        },
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.random.choices",
        lambda population, weights, k: [population[0]],
    )

    selected, tail_stages = ctrl._select_critical_tail_jobs(
        schedule,
        1,
        tail_stage_count=1,
        job_selection_policy="weighted_random",
    )

    assert tail_stages == ["S3"]
    assert selected == ["J3"]


def test_apply_critical_tail_job_operator_frees_all_ops_of_selected_jobs(monkeypatch):
    schedule = _build_stage_band_schedule()
    ctrl = _make_controller(
        {
            stage_id: {"J1": 1, "J2": 1}
            for stage_id in schedule.stages
        }
    )
    ctrl.instance = SimpleNamespace(stage_id_list=list(schedule.stages))
    ctrl.solution_manager = SimpleNamespace(
        has_incumbent=lambda: True,
        get_incumbent=lambda: schedule,
    )

    monkeypatch.setattr(
        ctrl,
        "_select_critical_tail_jobs",
        lambda *args, **kwargs: (["J2"], ["S3", "S4"]),
    )

    captured = {}

    def fake_fix(selected_ops, **_kwargs):
        captured["selected_ops"] = selected_ops

    monkeypatch.setattr(ctrl, "_fix_operations_profile_except_selected", fake_fix)

    selected_jobs = ctrl.apply_critical_tail_job_operator(job_count=1, tail_stage_count=2)

    assert selected_jobs == ["J2"]
    assert captured["selected_ops"] == {
        ("J2", "S0", "M1"),
        ("J2", "S1", "M1"),
        ("J2", "S2", "M1"),
        ("J2", "S3", "M1"),
        ("J2", "S4", "M1"),
    }


def test_critical_tail_job_ns_uses_tl_nc_multiplier(monkeypatch):
    ctrl = _make_controller()
    ctrl.instance = SimpleNamespace(job_count=12, stage_count=5)

    captured = {}

    def fake_fix_profile_solve_reset(
        profile_fixing_method,
        computational_time,
        solver_thread_cnt,
        **_kwargs,
    ):
        captured["computational_time"] = computational_time
        captured["solver_thread_cnt"] = solver_thread_cnt

    monkeypatch.setattr(ctrl, "_fix_profile_solve_reset", fake_fix_profile_solve_reset)

    ctrl.critical_tail_job_ns(
        job_count=3,
        solver_thread_cnt=6,
        computational_time=999,
        tl_nc_multiplier=0.25,
    )

    assert captured["computational_time"] == 15.0
    assert captured["solver_thread_cnt"] == 6


def test_critical_cross_machine_insertion_ls_uses_tl_nc_multiplier(monkeypatch):
    schedule = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2"],
        stages=["S1"],
        machines_per_stage={"S1": ["M1", "M2"]},
    )
    schedule.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=5)
    schedule.add_ops_times_2_mc("S1", "M1", "J2", start_time=5, end_time=10)

    ctrl = _make_controller({"S1": {"J1": 5, "J2": 5}})
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=4)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: schedule,
        register=lambda report, solution: False,
    )
    ctrl.timer = SimpleNamespace(elapsed_sec=0.0)
    ctrl.obj_store = SimpleNamespace(
        get_last_obj_value=lambda: None,
        get_last_obj_bound=lambda: None,
        add_last_timestamp_note=lambda *args, **kwargs: None,
    )
    ctrl._make_subroutine_report = lambda **kwargs: SimpleNamespace(**kwargs)
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "test-call"
    ctrl.get_remaining_time_limit = lambda subroutine_time_limit: subroutine_time_limit

    captured = {}

    def fake_improve_schedule_by_critical_cross_machine_insertions(
        *_args,
        computational_time,
        **_kwargs,
    ):
        captured["computational_time"] = computational_time
        return schedule

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.improve_schedule_by_critical_cross_machine_insertions",
        fake_improve_schedule_by_critical_cross_machine_insertions,
    )

    ctrl.critical_cross_machine_insertion_ls(
        max_passes=2,
        computational_time=999.0,
        tl_nc_multiplier=0.5,
        target_stage_mode="all",
    )

    assert captured["computational_time"] == 20.0


def test_solve_base_cp_model_uses_tl_nc_multiplier(monkeypatch):
    ctrl = _make_controller({"S1": {"J1": 1}})
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=4)
    ctrl.base_cp_model_is_set = True
    ctrl.cp_model = SimpleNamespace(delete_added_constraints=lambda: None)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: None,
        register=lambda report, solution: False,
    )
    ctrl.timer = SimpleNamespace(elapsed_sec=0.0)
    ctrl.obj_store = SimpleNamespace(
        get_last_obj_value=lambda: None,
        get_last_obj_bound=lambda: None,
        add_last_timestamp_note=lambda *args, **kwargs: None,
    )
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "test-call"

    captured = {}

    def fake_solve_current_cp_remaining_time_limit(
        computational_time,
        solver_thread_cnt,
        **_kwargs,
    ):
        captured["computational_time"] = computational_time
        captured["solver_thread_cnt"] = solver_thread_cnt
        return (
            SimpleNamespace(status="ok", obj_value=20.0, obj_bound=20.0),
            "solution",
        )

    monkeypatch.setattr(
        ctrl,
        "solve_current_cp_remaining_time_limit",
        fake_solve_current_cp_remaining_time_limit,
    )

    ctrl.solve_base_cp_model(
        computational_time=999.0,
        tl_nc_multiplier=0.5,
        solver_thread_cnt=6,
    )

    assert captured["computational_time"] == 20.0
    assert captured["solver_thread_cnt"] == 6


def test_fix_profile_solve_reset_initializes_base_cp_model_when_missing(monkeypatch):
    ctrl = _make_controller({"S1": {"J1": 1}})
    ctrl.instance = SimpleNamespace(stage_id_list=["S1"])
    ctrl.base_cp_model_is_set = False
    ctrl.solution_manager = SimpleNamespace(
        register=lambda report, solution: False,
        get_incumbent=lambda: None,
    )
    ctrl.obj_store = SimpleNamespace(
        get_last_obj_value=lambda: None,
        get_last_obj_bound=lambda: None,
        add_last_timestamp_note=lambda *args, **kwargs: None,
    )
    ctrl.timer = SimpleNamespace(elapsed_sec=0.0)
    ctrl._get_call_context_of_current_method = lambda: "test-call"

    events = []

    def fake_set_cp_model_as_base_cp_model():
        events.append("set_cp_model")
        ctrl.base_cp_model_is_set = True
        ctrl.cp_model = SimpleNamespace(
            delete_added_constraints=lambda: events.append("delete_added_constraints")
        )

    def fake_profile_fixing_method():
        assert hasattr(ctrl, "cp_model")
        events.append("profile_fix")

    monkeypatch.setattr(ctrl, "set_cp_model_as_base_cp_model", fake_set_cp_model_as_base_cp_model)
    monkeypatch.setattr(
        ctrl,
        "solve_with_initial_solution",
        lambda *args, **kwargs: ("report", "solution"),
    )

    ctrl._fix_profile_solve_reset(
        fake_profile_fixing_method,
        computational_time=1.0,
        solver_thread_cnt=1,
    )

    assert events == [
        "set_cp_model",
        "profile_fix",
        "delete_added_constraints",
    ]


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
