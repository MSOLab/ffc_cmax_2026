from types import SimpleNamespace

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class _FakeCpModel:
    def __init__(self) -> None:
        self.delete_added_constraints_count = 0

    def delete_added_constraints(self) -> None:
        self.delete_added_constraints_count += 1


class _FakeObjStore:
    def __init__(self) -> None:
        self.notes = []

    def get_last_obj_value(self):
        return 9.0

    def get_last_obj_bound(self):
        return 8.0

    def add_last_timestamp_note(
        self,
        note,
        *,
        obj_value_is_valid=False,
        obj_bound_is_valid=False,
    ) -> None:
        self.notes.append((note, obj_value_is_valid, obj_bound_is_valid))


class _FakeSolutionManager:
    def __init__(self, incumbent: HybridFlowshopLiteSchedule) -> None:
        self.incumbent = incumbent
        self.registered = []

    def get_incumbent(self):
        return self.incumbent

    def register(self, report, solution):
        self.registered.append((report, solution))
        return False


def _make_schedule() -> HybridFlowshopLiteSchedule:
    schedule = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    schedule.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    schedule.add_ops_times_2_mc("s1", "m1", "j2", 2, 5)
    schedule.add_ops_times_2_mc("s2", "m1", "j1", 2, 6)
    schedule.add_ops_times_2_mc("s2", "m1", "j2", 6, 7)
    return schedule


def test_full_schedule_stage_precedence_cp_passes_pdiff_options(monkeypatch) -> None:
    schedule = _make_schedule()
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.solution_manager = _FakeSolutionManager(schedule)
    ctrl.base_cp_model_is_set = True
    ctrl.cp_model = _FakeCpModel()
    ctrl.params = SimpleNamespace(name="params")
    ctrl.vars = SimpleNamespace(name="vars")
    ctrl.timer = SimpleNamespace(elapsed_sec=12.0)
    ctrl.obj_store = _FakeObjStore()

    captured = {}

    def fake_add_profile_constraints(*args, **kwargs) -> None:
        captured["profile_args"] = args
        captured["profile_kwargs"] = kwargs

    monkeypatch.setattr(
        BaseModelBuilder,
        "add_stage_ops_precedence_constraints_after_dispatch_from_schedule",
        staticmethod(fake_add_profile_constraints),
    )

    report = SimpleNamespace(status="FEASIBLE", obj_value=9.0, obj_bound=8.0)

    def fake_solve_with_initial_solution(*args, **kwargs):
        captured["solve_args"] = args
        captured["solve_kwargs"] = kwargs
        return report, schedule

    monkeypatch.setattr(
        ctrl,
        "solve_with_initial_solution",
        fake_solve_with_initial_solution,
    )
    monkeypatch.setattr(ctrl, "add_obj_value_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(ctrl, "add_obj_bound_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ctrl,
        "_get_call_context_of_current_method",
        lambda: "1-solve_full_schedule_stage_precedence_cp",
    )

    ctrl.solve_full_schedule_stage_precedence_cp(
        computational_time=1.25,
        solver_thread_cnt=16,
        stage_precedence_min_processing_time_diff=100,
        stage_precedence_min_processing_time_diff_ratio=0.2,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
        use_lns_only=True,
    )

    assert ctrl.cp_model.delete_added_constraints_count == 2
    assert captured["profile_args"][:4] == (
        ctrl.cp_model,
        ctrl.params,
        ctrl.vars,
        schedule,
    )
    assert captured["profile_kwargs"] == {
        "profile_fix_by_machine": False,
        "machine_precedence_stride": 1,
        "stage_precedence_min_processing_time_diff": 100,
        "stage_precedence_min_processing_time_diff_ratio": 0.2,
    }
    assert 0.0 < captured["solve_args"][0] <= 1.25
    assert captured["solve_args"][1] == 16
    assert captured["solve_kwargs"]["use_lns_only"] is True
    assert captured["solve_kwargs"]["obj_value_is_valid"] is True
    assert captured["solve_kwargs"]["obj_bound_is_valid"] is False
    assert ctrl.solution_manager.registered == [(report, schedule)]
    assert ctrl.obj_store.notes == [
        ("1-solve_full_schedule_stage_precedence_cp", True, True)
    ]
