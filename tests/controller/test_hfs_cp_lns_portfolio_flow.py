from types import SimpleNamespace

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def get_jik_2_start_time_map(self):
        return {}

    def make_semi_active(self, *_args, **_kwargs) -> None:
        return None


class _RecorderSolutionManager:
    def __init__(self, incumbent=None) -> None:
        self.incumbent = incumbent
        self.registered = []

    def get_incumbent(self):
        return self.incumbent

    def register(self, report, solution):
        self.registered.append((report, solution))
        if solution is not None:
            self.incumbent = solution
        return True


def _wire_common_controller(ctrl: HybridFlowShopCpLnsController) -> None:
    ctrl.timer = SimpleNamespace(elapsed_sec=0.0)
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None,
    )
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "test-call"
    ctrl._make_subroutine_report = lambda **kwargs: SimpleNamespace(**kwargs)


def test_initialize_by_dispatch_portfolio_registers_best_candidate(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.solution_manager = _RecorderSolutionManager()

    def fake_get_candidates(**config):
        score = 100
        if config["head_for_all_stages"]:
            score -= 15
        if not config["machine_then_job"]:
            score -= 5
        if "stage_agg_2" in config["method_list"]:
            score -= 10
        if config["mi_agg_method"] == "min":
            score -= 3
        return {"fake": _FakeSchedule(score)}

    monkeypatch.setattr(
        ctrl,
        "_get_selected_dispatch_candidate_schedules",
        fake_get_candidates,
    )

    ctrl.initialize_by_dispatch_portfolio(portfolio="balanced", include_stage_agg=True)

    report, solution = ctrl.solution_manager.registered[-1]
    assert report.subroutine_name == "initialize_by_dispatch_portfolio"
    assert report.is_init is True
    assert solution.makespan == 67


def test_critical_schedule_repair_ls_registers_best_repair(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=4)
    ctrl.stage_2_job_2_p_dict = {"S1": {"J1": 1}}
    incumbent = HybridFlowshopLiteSchedule(
        jobs=["J1"],
        stages=["S1"],
        machines_per_stage={"S1": ["M1"]},
    )
    incumbent.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=100)
    ctrl.solution_manager = _RecorderSolutionManager(incumbent)
    ctrl.get_remaining_time_limit = lambda subroutine_time_limit: subroutine_time_limit
    ctrl._resolve_target_stage_ids_for_critical_machine_ls = lambda **_kwargs: None
    ctrl.create_empty_schedule_from_ins = lambda: _FakeSchedule(100)

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.improve_schedule_by_critical_stage_sequence_insertions",
        lambda *_args, **_kwargs: _FakeSchedule(95),
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.improve_schedule_by_critical_cross_machine_insertions",
        lambda *_args, **_kwargs: _FakeSchedule(90),
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.improve_schedule_by_critical_adjacent_swaps",
        lambda schedule, *_args, **_kwargs: _FakeSchedule(schedule.makespan - 1),
    )

    ctrl.critical_schedule_repair_ls(
        max_rounds=1,
        tl_nc_multiplier=0.5,
        target_stage_mode="all",
    )

    report, solution = ctrl.solution_manager.registered[-1]
    assert report.subroutine_name == "critical_schedule_repair_ls"
    assert solution.makespan == 89
