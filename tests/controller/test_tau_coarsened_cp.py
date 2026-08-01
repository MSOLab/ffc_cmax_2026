import pandas as pd
from schore.parameters import JobStageProcessingTimeManager
from schore.parameters_examples import HybridFlowshopParameters

from hybridflowshop.controller.hfs_cp_lns import (
    HybridFlowShopCpLnsController,
    _schedule_sequence_signature,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, validate_schedule


def _make_instance() -> HybridFlowshopParameters:
    job_ids = ["J1", "J2"]
    stage_ids = ["S1", "S2"]
    p_manager = JobStageProcessingTimeManager(
        name="tiny_tau_P",
        df=pd.DataFrame([[6, 4], [1, 9]]),
    )
    return HybridFlowshopParameters(
        name="tiny_tau",
        job_id_list=job_ids,
        stage_id_list=stage_ids,
        stage_2_machines_map={"S1": ["M1"], "S2": ["M1", "M2"]},
        p_manager=p_manager,
    )


def _make_controller() -> HybridFlowShopCpLnsController:
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = _make_instance()
    ctrl.job_2_stage_2_p_dict = ctrl.instance.job_2_stage_2_p_map
    ctrl.stage_2_job_2_p_dict = ctrl.instance.stage_2_job_2_p_map
    return ctrl


def test_make_tau_coarsened_instance_rounds_processing_times_up() -> None:
    ctrl = _make_controller()

    scaled = ctrl._make_tau_coarsened_instance(5)

    assert scaled.job_2_stage_2_p_map == {
        "J1": {"S1": 2, "S2": 1},
        "J2": {"S1": 1, "S2": 2},
    }
    assert scaled.stage_2_machines_map == ctrl.instance.stage_2_machines_map


def test_restore_original_schedule_from_machine_sequence_is_feasible() -> None:
    ctrl = _make_controller()
    tau_schedule = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2"],
        stages=["S1", "S2"],
        machines_per_stage={"S1": ["M1"], "S2": ["M1", "M2"]},
    )
    tau_schedule.add_operation_2_mc("S1", "M1", "J2", 1)
    tau_schedule.add_operation_2_mc("S1", "M1", "J1", 2)
    tau_schedule.add_operation_2_mc("S2", "M1", "J1", 1)
    tau_schedule.add_operation_2_mc("S2", "M2", "J2", 2)

    restored = ctrl._restore_original_schedule_from_tau_schedule(
        tau_schedule,
        restore_mode="machine_sequence",
        make_semi_active=True,
    )

    validate_schedule(restored, ctrl.stage_2_job_2_p_dict)
    assert [op[2] for op in restored.get_job_sequence("S1", "M1")] == ["J2", "J1"]
    assert restored.makespan <= 15


def test_restore_original_schedule_from_stage_sequence_is_feasible() -> None:
    ctrl = _make_controller()
    tau_schedule = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2"],
        stages=["S1", "S2"],
        machines_per_stage={"S1": ["M1"], "S2": ["M1", "M2"]},
    )
    tau_schedule.add_operation_2_mc("S1", "M1", "J2", 1)
    tau_schedule.add_operation_2_mc("S1", "M1", "J1", 2)
    tau_schedule.add_operation_2_mc("S2", "M2", "J2", 2)
    tau_schedule.add_operation_2_mc("S2", "M1", "J1", 1)

    restored = ctrl._restore_original_schedule_from_tau_schedule(
        tau_schedule,
        restore_mode="stage_sequence",
        make_semi_active=True,
    )

    validate_schedule(restored, ctrl.stage_2_job_2_p_dict)
    assert restored.makespan <= 15


def test_tau_surrogate_dispatch_schedules_keep_top_unique_candidates(
    monkeypatch,
) -> None:
    ctrl = _make_controller()

    def make_schedule(first_stage_jobs: list[str]) -> HybridFlowshopLiteSchedule:
        schedule = HybridFlowshopLiteSchedule(
            jobs=["J1", "J2"],
            stages=["S1", "S2"],
            machines_per_stage={"S1": ["M1"], "S2": ["M1", "M2"]},
        )
        for job_id in first_stage_jobs:
            schedule.add_operation_2_stage(
                "S1",
                job_id,
                ctrl.job_2_stage_2_p_dict[job_id]["S1"],
            )
        for job_id in reversed(first_stage_jobs):
            schedule.add_operation_2_stage(
                "S2",
                job_id,
                ctrl.job_2_stage_2_p_dict[job_id]["S2"],
            )
        return schedule

    fast = make_schedule(["J2", "J1"])
    slow = make_schedule(["J1", "J2"])
    duplicate_fast = fast.deepcopy()

    def fake_dispatch_candidates(**_kwargs):
        return {
            "slow": slow,
            "fast": fast,
            "duplicate_fast": duplicate_fast,
        }

    monkeypatch.setattr(
        ctrl,
        "_get_selected_dispatch_candidate_schedules",
        fake_dispatch_candidates,
    )

    entries = ctrl._get_tau_surrogate_dispatch_schedules(
        ctrl.instance,
        cap_portions=[0.25],
        method_list=["slow", "fast", "duplicate_fast"],
        include_machine_then_job_variants=False,
        candidate_top_k=5,
    )

    assert len(entries) == 2
    assert [schedule.makespan for _label, schedule in entries] == sorted(
        [fast.makespan, slow.makespan]
    )
    assert (
        len({_schedule_sequence_signature(schedule) for _label, schedule in entries})
        == 2
    )
