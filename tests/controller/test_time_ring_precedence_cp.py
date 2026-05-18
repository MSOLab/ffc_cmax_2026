from types import SimpleNamespace

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class _FakeSolutionManager:
    def __init__(self, incumbent: HybridFlowshopLiteSchedule) -> None:
        self.incumbent = incumbent

    def get_incumbent(self):
        return self.incumbent


def _make_schedule() -> HybridFlowshopLiteSchedule:
    schedule = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    schedule.add_ops_times_2_mc("s1", "m1", "j1", 0, 80)
    schedule.add_ops_times_2_mc("s1", "m1", "j2", 120, 180)
    schedule.add_ops_times_2_mc("s1", "m1", "j3", 250, 310)
    schedule.add_ops_times_2_mc("s1", "m1", "j4", 520, 600)
    return schedule


def _make_controller(schedule: HybridFlowshopLiteSchedule):
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.solution_manager = _FakeSolutionManager(schedule)
    ctrl.cp_model = SimpleNamespace()
    ctrl.params = SimpleNamespace()
    ctrl.vars = SimpleNamespace()
    return ctrl


def _ops(schedule: HybridFlowshopLiteSchedule) -> set[tuple[str, str, str]]:
    return set(schedule.get_jik_2_start_time_map())


def test_time_ring_operator_uses_machine_outside_and_stage_shoulder(
    monkeypatch,
) -> None:
    schedule = _make_schedule()
    ctrl = _make_controller(schedule)
    captured = []

    def fake_add_profile_constraints(*args, **kwargs) -> None:
        captured.append(
            {
                "schedule_ops": _ops(args[3]),
                "profile_fix_by_machine": kwargs["profile_fix_by_machine"],
                "machine_precedence_stride": kwargs["machine_precedence_stride"],
            }
        )

    monkeypatch.setattr(
        BaseModelBuilder,
        "add_stage_ops_precedence_constraints_after_dispatch_from_schedule",
        staticmethod(fake_add_profile_constraints),
    )

    result = ctrl.apply_time_ring_precedence_operator(
        200,
        400,
        shoulder_size=100,
        outside_machine_precedence_stride=2,
    )

    assert result["core_ops"] == 1
    assert result["shoulder_ops"] == 1
    assert result["outside_ops"] == 2
    assert captured == [
        {
            "schedule_ops": {("j1", "s1", "m1"), ("j4", "s1", "m1")},
            "profile_fix_by_machine": True,
            "machine_precedence_stride": 2,
        },
        {
            "schedule_ops": {("j2", "s1", "m1")},
            "profile_fix_by_machine": False,
            "machine_precedence_stride": 1,
        },
    ]
