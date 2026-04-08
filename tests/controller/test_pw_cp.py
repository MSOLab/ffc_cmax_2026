from pathlib import Path
from types import SimpleNamespace

import pytest
from mbls.cpsat import ObjValueBoundStore
from ortools.sat.python import cp_model
from routix import ElapsedTimer

from hybridflowshop.controller.pw_cp import (
    OperationPartition,
    PwCpConstructor,
    PwCpRunState,
    PwCpSubproblemSpec,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule
from tests.test_dispatch_stage_by_machines import create_hfs_instance


class FakePwCpContext:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.solver = cp_model.CpSolver()
        self.feasibility_checks = []

    def get_remaining_time_limit(self, subroutine_time_limit: float | None) -> float:
        if subroutine_time_limit is None:
            return 1.0
        return subroutine_time_limit

    def solve_cp_model_2(self, *args, **kwargs):
        raise AssertionError("solve_cp_model_2 should be monkeypatched in this test")

    def create_schedule(self, *args, **kwargs):
        raise AssertionError("create_schedule should be monkeypatched in this test")

    def check_feasibility(self, start_time_map):
        self.feasibility_checks.append(start_time_map)
        return 0.0

    def get_file_path_for_subroutine(self, suffix: str) -> Path:
        return self.tmp_path / suffix.lstrip("_")


def _make_schedule() -> HybridFlowshopLiteSchedule:
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    sched.add_ops_times_2_mc("s1", "m1", "j3", 4, 6)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 0, 3)
    sched.add_ops_times_2_mc("s1", "m2", "j4", 5, 7)
    return sched


def _make_multi_stage_schedule() -> HybridFlowshopLiteSchedule:
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 0, 3)
    sched.add_ops_times_2_mc("s1", "m1", "j3", 4, 6)
    sched.add_ops_times_2_mc("s1", "m2", "j4", 5, 7)
    sched.add_ops_times_2_mc("s2", "m3", "j1", 2, 5)
    sched.add_ops_times_2_mc("s2", "m4", "j2", 3, 6)
    sched.add_ops_times_2_mc("s2", "m3", "j3", 6, 9)
    sched.add_ops_times_2_mc("s2", "m4", "j4", 7, 10)
    return sched


def _make_promotion_target_schedule() -> HybridFlowshopLiteSchedule:
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 2, 5)
    sched.add_ops_times_2_mc("s1", "m1", "j3", 5, 7)
    sched.add_ops_times_2_mc("s1", "m2", "j4", 7, 9)
    sched.add_ops_times_2_mc("s2", "m3", "j1", 0, 3)
    sched.add_ops_times_2_mc("s2", "m3", "j3", 3, 6)
    sched.add_ops_times_2_mc("s2", "m4", "j2", 6, 9)
    sched.add_ops_times_2_mc("s2", "m4", "j4", 9, 12)
    return sched


def _make_early_promotion_target_schedule() -> HybridFlowshopLiteSchedule:
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 2, 5)
    sched.add_ops_times_2_mc("s1", "m1", "j3", 5, 7)
    sched.add_ops_times_2_mc("s1", "m2", "j4", 7, 9)
    sched.add_ops_times_2_mc("s2", "m4", "j2", 0, 3)
    sched.add_ops_times_2_mc("s2", "m3", "j1", 3, 6)
    sched.add_ops_times_2_mc("s2", "m3", "j3", 6, 9)
    sched.add_ops_times_2_mc("s2", "m4", "j4", 9, 12)
    return sched


def _make_solver_report(
    *,
    elapsed_time=0.5,
    is_feasible=True,
    status="FEASIBLE",
    obj_value_records=(),
    obj_bound_records=(),
):
    return SimpleNamespace(
        elapsed_time=elapsed_time,
        is_feasible=is_feasible,
        status=SimpleNamespace(
            to_solver_status_enum=lambda: SimpleNamespace(value=status)
        ),
        obj_value_records=list(obj_value_records),
        obj_bound_records=list(obj_bound_records),
    )


def _get_named_hint_map(mdl):
    proto = mdl.Proto()
    hint_map = dict(
        zip(proto.solution_hint.vars, proto.solution_hint.values, strict=True)
    )
    return {proto.variables[var_idx].name: value for var_idx, value in hint_map.items()}


def _make_instance(stage_ids, machines_per_stage, p_by_job):
    return create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        stage_ids,
        machines_per_stage,
        p_by_job,
    )


def test_build_stage_batches_is_deterministic(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)

    batches = ctor.build_stage_2_batch_list(_make_schedule(), batch_size=2)

    assert batches == {
        "s1": [
            (("j1", "m1"), ("j2", "m2")),
            (("j3", "m1"), ("j4", "m2")),
        ]
    }


def test_build_slack_batch_spec_creates_right_justified_window_map(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    stage_2_partition = {
        "s1": OperationPartition(
            left_time_fixed=(),
            left_profile_fixed=(),
            unfixed=(("j1", "m1"), ("j2", "m2")),
            right_profile_fixed=(),
            right_time_fixed=(("j3", "m1"), ("j4", "m2")),
        )
    }
    ctor._st = PwCpRunState(
        timer=ElapsedTimer(),
        incumbent=incumbent,
        sub_obj_store=ObjValueBoundStore[int](),
        subproblem_idx=0,
        subproblem_logs=[],
        max_time_per_batch=1.0,
    )

    spec = ctor._build_batch_spec(
        incumbent=incumbent,
        stage_2_partition=stage_2_partition,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_idx=0,
    )

    assert spec.stage_2_mc_2_window == {"s1": {"m1": (0, 5), "m2": (0, 5)}}


def test_prepare_subproblem_model_uses_incumbent_hints_for_makespan_batch(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    stage_2_partition = {
        "s1": OperationPartition(
            left_time_fixed=(),
            left_profile_fixed=(),
            unfixed=(("j1", "m1"),),
            right_profile_fixed=(),
            right_time_fixed=(),  # Empty = makespan batch
        )
    }
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        stage_2_partition=stage_2_partition,
        stage_2_mc_2_window={},
        init_schedule=incumbent,
    )
    instance = _make_instance(
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    mdl, _params, _variables = ctor._prepare_pw_cp_model(
        spec=spec,
        instance=instance,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
        tighten_ranges=False,
    )

    hints = _get_named_hint_map(mdl)
    assert hints["start_j1_s1"] == 0
    assert hints["end_j1_s1"] == 2


def test_prepare_subproblem_model_uses_right_justified_hints_for_slack_batch(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    right_justified = incumbent.deepcopy()
    right_justified.remove_operations({("j1", "s1", "m1")})
    right_justified.add_ops_times_2_mc("s1", "m1", "j1", 1, 3)
    stage_2_partition = {
        "s1": OperationPartition(
            left_time_fixed=(),
            left_profile_fixed=(),
            unfixed=(("j1", "m1"),),
            right_profile_fixed=(),
            right_time_fixed=(("j3", "m1"), ("j4", "m2")),  # Non-empty = slack batch
        )
    }
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        stage_2_partition=stage_2_partition,
        stage_2_mc_2_window={"s1": {"m1": (0, 4), "m2": (0, 5)}},
        init_schedule=right_justified,
    )
    instance = _make_instance(
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    mdl, _params, _variables = ctor._prepare_pw_cp_model(
        spec=spec,
        instance=instance,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
        tighten_ranges=False,
    )

    hints = _get_named_hint_map(mdl)
    assert hints["start_j1_s1"] == 1
    assert hints["end_j1_s1"] == 3


def test_accept_candidate_or_repair_incumbent_accepts_improving_solution(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    candidate = incumbent.deepcopy()
    candidate.remove_operations({("j4", "s1", "m2")})
    candidate.add_ops_times_2_mc("s1", "m2", "j4", 4, 6)

    updated, accepted = ctor._accept_candidate_or_repair_incumbent(
        candidate=candidate,
        incumbent=incumbent,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
    )

    assert accepted is True
    assert updated is candidate
    assert ctx.feasibility_checks == [candidate.get_jik_2_start_time_map()]


def test_run_keeps_batch_union_across_stages(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_multi_stage_schedule()
    seen_unfixed = []

    def fake_slack(**kwargs):
        # Collect unfixed ops from all stages
        unfixed = []
        for partition in kwargs["spec"].stage_2_partition.values():
            unfixed.extend(partition.unfixed)
        seen_unfixed.append(tuple(sorted(unfixed)))
        return None

    def fake_makespan(**kwargs):
        # Collect unfixed ops from all stages
        unfixed = []
        for partition in kwargs["spec"].stage_2_partition.values():
            unfixed.extend(partition.unfixed)
        seen_unfixed.append(tuple(sorted(unfixed)))
        return None

    monkeypatch.setattr(ctor, "_solve_batch_pw_cp_model", fake_slack)

    instance = _make_instance(
        ["s1", "s2"],
        {"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
        {
            "j1": {"s1": 2, "s2": 3},
            "j2": {"s1": 3, "s2": 3},
            "j3": {"s1": 2, "s2": 3},
            "j4": {"s1": 2, "s2": 3},
        },
    )

    ctor.run(
        incumbent,
        instance,
        {
            "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2},
            "s2": {"j1": 3, "j2": 3, "j3": 3, "j4": 3},
        },
        batch_size=2,
        max_time_per_batch=1.0,
    )

    assert seen_unfixed == [
        (
            ("j1", "m1"),
            ("j1", "m3"),
            ("j2", "m2"),
            ("j2", "m4"),
        ),
        (
            ("j3", "m1"),
            ("j3", "m3"),
            ("j4", "m2"),
            ("j4", "m4"),
        ),
    ]


def test_promote_job_contained_ops_promotes_only_unfixed_jobs():
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(("j1", "m1"), ("j2", "m2")),
        unfixed=(("j3", "m3"),),
        right_profile_fixed=(("j2", "m4"), ("j4", "m5")),
        right_time_fixed=(),
    )

    promoted = partition.promote_job_contained_ops(promoted_job_id_set={"j2"})

    assert promoted.left_profile_fixed == (("j1", "m1"),)
    assert promoted.unfixed == (("j2", "m2"), ("j2", "m4"), ("j3", "m3"))
    assert promoted.right_profile_fixed == (("j4", "m5"),)


def test_run_applies_promoted_partition_to_subproblem_spec(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_early_promotion_target_schedule()
    seen_slack_batch_count = 0

    class _StopAfterPromotionCheck(Exception):
        pass

    def fake_slack(**kwargs):
        nonlocal seen_slack_batch_count
        seen_slack_batch_count += 1
        if seen_slack_batch_count != 2:
            return None

        promoted_spec = kwargs["spec"].stage_2_partition
        assert promoted_spec["s1"].left_profile_fixed == ()
        assert promoted_spec["s2"].left_profile_fixed == ()
        assert promoted_spec["s1"].unfixed == (("j1", "m1"), ("j2", "m2"))
        assert promoted_spec["s2"].unfixed == (("j1", "m3"), ("j2", "m4"))
        raise _StopAfterPromotionCheck

    def fake_makespan(**kwargs):
        return None

    def fake_accept_candidate_or_repair_incumbent(**kwargs):
        return kwargs["incumbent"], False

    monkeypatch.setattr(ctor, "_solve_batch_pw_cp_model", fake_slack)
    monkeypatch.setattr(
        ctor,
        "_accept_candidate_or_repair_incumbent",
        fake_accept_candidate_or_repair_incumbent,
    )

    instance = _make_instance(
        ["s1", "s2"],
        {"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
        {
            "j1": {"s1": 2, "s2": 3},
            "j2": {"s1": 3, "s2": 3},
            "j3": {"s1": 2, "s2": 3},
            "j4": {"s1": 2, "s2": 3},
        },
    )

    with pytest.raises(_StopAfterPromotionCheck):
        ctor.run(
            incumbent,
            instance,
            {
                "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2},
                "s2": {"j1": 3, "j2": 3, "j3": 3, "j4": 3},
            },
            batch_size=1,
            unfixed_batch_count=1,
            left_profile_fixed_batch_count=1,
            enable_promotion_profile_fixed=True,
            max_time_per_batch=1.0,
        )


def test_run_records_only_accepted_incumbent_improvements(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_multi_stage_schedule()

    dummy_batches = {
        "s1": [(("j1", "m1"),), (("j2", "m2"),), (("j3", "m1"),), (("j4", "m2"),)],
        "s2": [(("j1", "m3"),), (("j2", "m4"),), (("j3", "m3"),), (("j4", "m4"),)],
    }
    monkeypatch.setattr(ctor, "build_stage_2_batch_list", lambda *args, **kwargs: dummy_batches)
    monkeypatch.setattr(
        ctor,
        "_build_batch_spec",
        lambda *, incumbent, stage_2_partition, stage_2_job_2_p_dict, batch_idx: SimpleNamespace(
            non_time_fixed_op_count=1,
            batch_idx=batch_idx,
        ),
    )
    monkeypatch.setattr(ctor, "_solve_batch_pw_cp_model", lambda **kwargs: object())
    accepted_outcomes = iter(
        [
            (SimpleNamespace(makespan=9), False),  # First iteration: incumbent semi-active → makespan=9, no improvement
            (SimpleNamespace(makespan=8), True),   # Second iteration: improvement to 8
            (SimpleNamespace(makespan=8), False),  # Third iteration: no improvement
            (SimpleNamespace(makespan=8), False),  # Fourth iteration: no improvement
        ]
    )

    def fake_accept_candidate_or_repair_incumbent(**kwargs):
        return next(accepted_outcomes)

    monkeypatch.setattr(
        ctor,
        "_accept_candidate_or_repair_incumbent",
        fake_accept_candidate_or_repair_incumbent,
    )

    instance = _make_instance(
        ["s1", "s2"],
        {"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
        {
            "j1": {"s1": 2, "s2": 3},
            "j2": {"s1": 3, "s2": 3},
            "j3": {"s1": 2, "s2": 3},
            "j4": {"s1": 2, "s2": 3},
        },
    )

    result = ctor.run(
        incumbent,
        instance,
        {
            "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2},
            "s2": {"j1": 3, "j2": 3, "j3": 3, "j4": 3},
        },
        batch_size=1,
        max_time_per_batch=1.0,
    )

    assert [value for _, value in result.sub_obj_store.obj_value_series.items()] == [
        8,  # Only the improvement from 9 to 8 is recorded
    ]
