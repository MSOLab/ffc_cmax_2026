from types import SimpleNamespace

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class _FakeVar:
    def __init__(self, name: str) -> None:
        self.name = name

    def __le__(self, other):
        return ("le", self.name, other)


class _FakeCpModel:
    def __init__(self) -> None:
        self.constraints = []

    def add(self, constraint) -> None:
        self.constraints.append(constraint)


class _FakeSolutionManager:
    def __init__(self, incumbent: HybridFlowshopLiteSchedule) -> None:
        self.incumbent = incumbent
        self.best_obj_value = incumbent.makespan
        self.best_obj_bound = None

    def get_incumbent(self):
        return self.incumbent


def _make_schedule() -> HybridFlowshopLiteSchedule:
    schedule = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    schedule.add_ops_times_2_mc("s1", "m1", "j1", 0, 3)
    schedule.add_ops_times_2_mc("s1", "m1", "j2", 3, 5)
    schedule.add_ops_times_2_mc("s1", "m1", "j3", 5, 7)
    schedule.add_ops_times_2_mc("s2", "m1", "j1", 3, 8)
    schedule.add_ops_times_2_mc("s2", "m1", "j2", 8, 11)
    schedule.add_ops_times_2_mc("s2", "m1", "j3", 11, 13)
    schedule.add_ops_times_2_mc("s3", "m1", "j1", 8, 10)
    schedule.add_ops_times_2_mc("s3", "m1", "j2", 11, 15)
    schedule.add_ops_times_2_mc("s3", "m1", "j3", 15, 18)
    return schedule


def _make_controller(schedule: HybridFlowshopLiteSchedule):
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.solution_manager = _FakeSolutionManager(schedule)
    ctrl.cp_model = _FakeCpModel()
    ctrl.vars = SimpleNamespace(makespan=_FakeVar("makespan"))
    ctrl.instance = SimpleNamespace(
        job_count=3,
        stage_count=3,
        stage_id_list=["s1", "s2", "s3"],
    )
    ctrl.stage_2_job_2_p_dict = {
        "s1": {"j1": 3, "j2": 2, "j3": 2},
        "s2": {"j1": 5, "j2": 3, "j3": 2},
        "s3": {"j1": 2, "j2": 4, "j3": 3},
    }
    return ctrl


def test_select_critical_cone_includes_tail_job_and_competitors() -> None:
    schedule = _make_schedule()
    ctrl = _make_controller(schedule)

    selected = ctrl._select_critical_cone_operations(
        schedule,
        slack_tolerance=0,
        tail_time_ratio=0.35,
        seed_op_count=1,
        stage_radius=1,
        time_radius=2,
        time_radius_ratio=None,
        machine_neighbor_depth=1,
        max_selected_ops=None,
    )

    assert ("j3", "s3", "m1") in selected
    assert ("j3", "s2", "m1") in selected
    assert ("j2", "s3", "m1") in selected


def test_critical_cone_operator_fixes_outside_and_requires_improvement(monkeypatch) -> None:
    schedule = _make_schedule()
    ctrl = _make_controller(schedule)
    captured = {}

    def fake_fix_except_selected(selected_ops, **kwargs) -> None:
        captured["selected_ops"] = selected_ops
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        ctrl,
        "_fix_operations_profile_except_selected",
        fake_fix_except_selected,
    )

    selected = ctrl.apply_critical_cone_operator(
        min_improvement=1,
        slack_tolerance=0,
        tail_time_ratio=0.35,
        seed_op_count=1,
        stage_radius=1,
        time_radius=2,
        time_radius_ratio=None,
        machine_neighbor_depth=1,
        max_selected_ops=None,
        fix_outside_start_times=True,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
    )

    assert selected == captured["selected_ops"]
    assert captured["kwargs"]["fix_start_times"] is True
    assert ctrl.cp_model.constraints == [("le", "makespan", schedule.makespan - 1)]
