from types import SimpleNamespace

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController


class FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan


def _make_controller(stage_count: int) -> HybridFlowShopCpLnsController:
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(
        stage_count=stage_count,
        stage_id_list=["S1", "S2", "S3"],
        stage_2_machines_map={"S1": ["M1"], "S2": ["M1"], "S3": ["M1"]},
    )
    ctrl.stage_2_job_2_p_dict = {
        "S1": {"J1": 3, "J2": 2},
        "S2": {"J1": 2, "J2": 2},
        "S3": {"J1": 4, "J2": 3},
    }
    ctrl.solution_manager = SimpleNamespace(best_obj_bound=100.0)
    return ctrl


def test_selected_dispatch_init_local_repair_runs_for_deep_instances_with_gap() -> None:
    ctrl = _make_controller(stage_count=15)

    assert ctrl._should_run_selected_dispatch_init_local_repair(FakeSchedule(105))


def test_selected_dispatch_init_local_repair_skips_for_shallow_strong_starts() -> None:
    ctrl = _make_controller(stage_count=5)

    assert not ctrl._should_run_selected_dispatch_init_local_repair(FakeSchedule(104))


def test_initialize_by_best_of_selected_dispatches_can_select_repaired_candidate() -> None:
    ctrl = _make_controller(stage_count=15)
    ctrl.timer = SimpleNamespace(elapsed_sec=0.0)
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *_args, **_kwargs: None
    )
    ctrl.add_obj_value_log = lambda *_args, **_kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "initialize-call"
    ctrl._make_subroutine_report = lambda **kwargs: SimpleNamespace(**kwargs)
    ctrl.draw_incumbent_gantt = lambda: None

    registered = {}

    def _register(report, schedule):
        registered["report"] = report
        registered["schedule"] = schedule
        return True

    ctrl.solution_manager = SimpleNamespace(best_obj_bound=100.0, register=_register)
    ctrl._get_selected_dispatch_candidate_schedules = lambda **_kwargs: {
        "best_of_mixed_dispatches": FakeSchedule(110),
        "bn2d_all_stages": FakeSchedule(108),
    }
    ctrl._repair_post_mip_dispatch_candidate = (
        lambda schedule, **_kwargs: FakeSchedule(schedule.makespan - 5)
    )

    ctrl.initialize_by_best_of_selected_dispatches()

    assert registered["schedule"].makespan == 103
    assert registered["report"].obj_value == 103
