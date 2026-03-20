import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mbls.cpsat import ObjValueBoundStore
from ortools.sat.python import cp_model
from routix.io.yaml import load_yaml

from hybridflowshop.controller.pw_cp import (
    OperationPartition,
    PwCpConstructor,
    PwCpResult,
    PwCpSubproblemLog,
    PwCpSubproblemSpec,
)
from hybridflowshop.io_solution import get_highlight_op_set
from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder
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


def _make_solver_report(
    *,
    elapsed_time=0.5,
    is_feasible=True,
    status="FEASIBLE",
    obj_value=1.0,
    obj_value_records=(),
    obj_bound_records=(),
):
    return SimpleNamespace(
        elapsed_time=elapsed_time,
        is_feasible=is_feasible,
        obj_value=obj_value,
        status=SimpleNamespace(
            to_solver_status_enum=lambda: SimpleNamespace(value=status)
        ),
        obj_value_records=list(obj_value_records),
        obj_bound_records=list(obj_bound_records),
    )


def _get_hint_map(mdl):
    proto = mdl.Proto()
    return dict(zip(proto.solution_hint.vars, proto.solution_hint.values, strict=True))


def _get_named_hint_map(mdl):
    proto = mdl.Proto()
    hint_map = _get_hint_map(mdl)
    return {proto.variables[var_idx].name: value for var_idx, value in hint_map.items()}


def _get_hint_var_indices(mdl):
    return list(mdl.Proto().solution_hint.vars)


def test_build_stage_batches_is_deterministic(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)

    batches = ctor._build_stage_batches(_make_schedule(), batch_size=2)

    assert batches == {
        "s1": [
            (("j1", "s1", "m1"), ("j2", "s1", "m2")),
            (("j3", "s1", "m1"), ("j4", "s1", "m2")),
        ]
    }


def test_build_stage_batches_sorts_by_start_time_when_flag_is_true(tmp_path):
    """When sort_by_start_time=True, operations are sorted by start time."""
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    # j1: start=10, j2: start=5, j3: start=20
    sched.add_ops_times_2_mc("s1", "m1", "j1", 10, 15)
    sched.add_ops_times_2_mc("s1", "m1", "j2", 5, 8)
    sched.add_ops_times_2_mc("s1", "m1", "j3", 20, 25)

    batches = ctor._build_stage_batches(sched, batch_size=1, sort_by_start_time=True)

    # Order should be j2 (5), j1 (10), j3 (20)
    assert batches["s1"] == [
        (("j2", "s1", "m1"),),
        (("j1", "s1", "m1"),),
        (("j3", "s1", "m1"),),
    ]


def test_build_stage_batches_sorts_by_midpoint_when_flag_is_false(tmp_path):
    """When sort_by_start_time=False, operations are sorted by midpoint."""
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    # j1: start=0, end=100 -> midpoint=50
    # j2: start=40, end=50 -> midpoint=45
    # order is [j2, j1] if midpoint, the reverse if start_time
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 100)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 40, 50)

    batches = ctor._build_stage_batches(sched, batch_size=1, sort_by_start_time=False)

    # Order should be j2 (midpoint=45), j1 (midpoint=50)
    assert batches["s1"] == [
        (("j2", "s1", "m2"),),
        (("j1", "s1", "m1"),),
    ]


def test_build_stage_batches_uses_start_time_as_tiebreaker_for_midpoint(tmp_path):
    """When midpoints are equal, start_time breaks the tie."""
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    # Both have midpoint=50: j1 (0-100), j2 (40-60)
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 100)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 40, 60)

    batches = ctor._build_stage_batches(sched, batch_size=1, sort_by_start_time=False)

    # j1 starts earlier (0 < 40), so j1 comes first
    assert batches["s1"] == [
        (("j1", "s1", "m1"),),
        (("j2", "s1", "m2"),),
    ]


def test_build_right_boundary_profile_uses_machine_order_start_times(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = _make_schedule()
    operation_list = [("j3", "s1", "m1"), ("j4", "s1", "m2")]
    boundary_profile, right_justified = ctor._build_right_boundary_profile(
        sched,
        operation_list,
        [],
        operation_list,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
    )

    assert boundary_profile["s1"] == [5, 5]
    assert right_justified.get_jik_2_start_time_map()["j3", "s1", "m1"] == 5
    assert right_justified.get_jik_2_start_time_map()["j4", "s1", "m2"] == 5


def test_build_right_boundary_profile_keeps_machine_boundaries_not_stage_earliest_starts(
    tmp_path,
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    operation_list = [
        ("j1", "s1", "m1"),
        ("j2", "s1", "m1"),
        ("j3", "s1", "m2"),
    ]
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", 2, 4)
    sched.add_ops_times_2_mc("s1", "m2", "j3", 9, 10)

    boundary_profile, _ = ctor._build_right_boundary_profile(
        sched,
        operation_list,
        [],
        operation_list,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 2, "j3": 1}},
    )

    assert boundary_profile["s1"] == [6, 9]


def test_build_right_boundary_profile_marks_missing_machine_with_none(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    operation_list = [("j1", "s1", "m1")]
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 3, 5)

    boundary_profile, _ = ctor._build_right_boundary_profile(
        sched,
        operation_list,
        [],
        operation_list,
        stage_2_job_2_p_dict={"s1": {"j1": 2}},
    )

    assert boundary_profile["s1"] == [3, None]


def test_add_kth_largest_constraint_handles_ties():
    model = cp_model.CpModel()
    xs = [model.new_int_var(v, v, f"x{i}") for i, v in enumerate([7, 7, 3])]
    z = model.new_int_var(0, 10, "z")
    BaseModelBuilder.add_kth_largest_constraint(model, xs, z, 2, "tie_case")

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(z) == 7


def test_run_keeps_incumbent_when_candidate_worsens(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()

    monkeypatch.setattr(
        ctor,
        "_solve_subproblem",
        lambda **kwargs: SimpleNamespace(
            makespan=incumbent.makespan + 1,
            get_jik_2_start_time_map=lambda: {},
        ),
    )

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    result = ctor.run(
        incumbent,
        instance,
        {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_size=2,
        max_time_per_batch=1.0,
    )

    assert result.schedule is incumbent
    assert ctx.feasibility_checks == []


def test_run_uses_batch_union_across_stages_and_exact_iteration_count(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_multi_stage_schedule()
    seen_specs = []

    def fake_solve_subproblem(**kwargs):
        seen_specs.append(kwargs["spec"])
        return None

    monkeypatch.setattr(ctor, "_solve_subproblem", fake_solve_subproblem)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
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

    assert len(seen_specs) == 2
    # Operations are sorted in OperationPartition
    assert seen_specs[0].partition.unfixed == (
        ("j1", "s1", "m1"),
        ("j1", "s2", "m3"),
        ("j2", "s1", "m2"),
        ("j2", "s2", "m4"),
    )
    assert seen_specs[1].partition.unfixed == (
        ("j3", "s1", "m1"),
        ("j3", "s2", "m3"),
        ("j4", "s1", "m2"),
        ("j4", "s2", "m4"),
    )


def test_run_raises_when_stage_batch_counts_differ(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", 3, 5)
    sched.add_ops_times_2_mc("s1", "m1", "j3", 6, 8)
    sched.add_ops_times_2_mc("s2", "m2", "j1", 0, 1)
    sched.add_ops_times_2_mc("s2", "m2", "j2", 2, 3)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3"],
        ["s1", "s2"],
        {"s1": ["m1"], "s2": ["m2"]},
        {
            "j1": {"s1": 2, "s2": 1},
            "j2": {"s1": 2, "s2": 1},
            "j3": {"s1": 2, "s2": 1},
        },
    )

    try:
        ctor.run(
            sched,
            instance,
            {
                "s1": {"j1": 2, "j2": 2, "j3": 2},
                "s2": {"j1": 1, "j2": 1, "j3": 1},
            },
            batch_size=2,
            max_time_per_batch=1.0,
        )
    except ValueError as exc:
        assert "identical batch counts" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched stage batch counts")


def test_run_raises_when_batch_count_changes_mid_run(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()

    original_build_stage_batches = ctor._build_stage_batches
    call_cnt = 0

    def fake_build_stage_batches(schedule, batch_size, sort_by_start_time=False):
        nonlocal call_cnt
        call_cnt += 1
        if call_cnt == 1:
            return original_build_stage_batches(
                schedule,
                batch_size,
                sort_by_start_time=sort_by_start_time,
            )
        return {
            "s1": [
                (("j1", "s1", "m1"),),
                (("j2", "s1", "m2"),),
                (("j3", "s1", "m1"),),
            ]
        }

    monkeypatch.setattr(ctor, "_build_stage_batches", fake_build_stage_batches)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    try:
        ctor.run(
            incumbent,
            instance,
            {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
            batch_size=2,
            max_time_per_batch=1.0,
        )
    except AssertionError as exc:
        assert "batch count changed during run" in str(exc)
    else:
        raise AssertionError("Expected AssertionError for changed batch count")


def test_add_right_slack_objective_maximizes_global_stage_min(tmp_path):
    model = cp_model.CpModel()
    op_start = {
        ("j1", "s1"): model.new_int_var(4, 4, "s_j1_s1"),
        ("j2", "s1"): model.new_int_var(2, 2, "s_j2_s1"),
        ("j1", "s2"): model.new_int_var(0, 0, "s_j1_s2"),
        ("j2", "s2"): model.new_int_var(0, 0, "s_j2_s2"),
    }
    op_end = {
        ("j1", "s1"): model.new_int_var(7, 7, "e_j1_s1"),
        ("j2", "s1"): model.new_int_var(5, 5, "e_j2_s1"),
        ("j1", "s2"): model.new_int_var(1, 1, "e_j1_s2"),
        ("j2", "s2"): model.new_int_var(2, 2, "e_j2_s2"),
    }
    op_intvl = {
        ("j1", "s1"): model.new_interval_var(
            op_start["j1", "s1"], 3, op_end["j1", "s1"], "i_j1_s1"
        ),
        ("j2", "s1"): model.new_interval_var(
            op_start["j2", "s1"], 3, op_end["j2", "s1"], "i_j2_s1"
        ),
        ("j1", "s2"): model.new_interval_var(
            op_start["j1", "s2"], 1, op_end["j1", "s2"], "i_j1_s2"
        ),
        ("j2", "s2"): model.new_interval_var(
            op_start["j2", "s2"], 2, op_end["j2", "s2"], "i_j2_s2"
        ),
    }
    params = SimpleNamespace(M_of={"s1": ["m1", "m2"], "s2": ["m3", "m4"]})
    variables = SimpleNamespace(op_start=op_start, op_end=op_end, op_intvl=op_intvl)

    slack_vars = BaseModelBuilder.add_right_slack_variables(
        model,
        params,
        slack_occupying_ops=(
            ("j1", "s1", "m1"),
            ("j2", "s1", "m2"),
            ("j1", "s2", "m3"),
            ("j2", "s2", "m4"),
        ),
        right_boundary_profile={"s1": [7, 7], "s2": [2, 2]},
        horizon=20,
    )
    BaseModelBuilder.add_right_slack_constraints(
        model,
        params,
        variables,
        slack_occupying_ops=(
            ("j1", "s1", "m1"),
            ("j2", "s1", "m2"),
            ("j1", "s2", "m3"),
            ("j2", "s2", "m4"),
        ),
        right_time_fixed_ops=(),
        right_boundary_profile={"s1": [7, 7], "s2": [2, 2]},
        slack_vars=slack_vars,
    )
    objective = BaseModelBuilder.add_right_slack_objective(
        model,
        slack_occupying_ops=(
            ("j1", "s1", "m1"),
            ("j2", "s1", "m2"),
            ("j1", "s2", "m3"),
            ("j2", "s2", "m4"),
        ),
        slack_vars=slack_vars,
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    # s1 slack lengths are [0, 2] -> stage min = 0
    # s2 slack lengths are [0, 0] -> stage min = 0
    # global slack min = min(0, 0) = 0
    assert solver.Value(objective) == 0


def test_add_right_slack_constraints_caps_unfixed_end_by_stage_max_slack_end():
    model = cp_model.CpModel()
    op_start = {
        ("j1", "s1"): model.new_int_var(0, 20, "s_j1_s1"),
        ("j2", "s1"): model.new_int_var(0, 20, "s_j2_s1"),
    }
    op_end = {
        ("j1", "s1"): model.new_int_var(3, 23, "e_j1_s1"),
        ("j2", "s1"): model.new_int_var(2, 22, "e_j2_s1"),
    }
    op_intvl = {
        ("j1", "s1"): model.new_interval_var(
            op_start["j1", "s1"], 3, op_end["j1", "s1"], "i_j1_s1"
        ),
        ("j2", "s1"): model.new_interval_var(
            op_start["j2", "s1"], 2, op_end["j2", "s1"], "i_j2_s1"
        ),
    }
    params = SimpleNamespace(M_of={"s1": ["m1", "m2"]})
    variables = SimpleNamespace(op_start=op_start, op_end=op_end, op_intvl=op_intvl)
    slack_vars = BaseModelBuilder.add_right_slack_variables(
        model,
        params,
        slack_occupying_ops=(("j1", "s1", "m1"),),
        right_boundary_profile={"s1": [4, 6]},
        horizon=20,
    )

    BaseModelBuilder.add_right_slack_constraints(
        model,
        params,
        variables,
        slack_occupying_ops=(("j1", "s1", "m1"),),
        right_time_fixed_ops=(("j2", "s1", "m2"),),
        right_boundary_profile={"s1": [4, 6]},
        slack_vars=slack_vars,
    )
    model.maximize(op_end["j1", "s1"])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(op_end["j1", "s1"]) == 6


def test_add_right_slack_constraints_skips_none_boundary_machine():
    model = cp_model.CpModel()
    op_start = {
        ("j1", "s1"): model.new_int_var(0, 20, "s_j1_s1"),
        ("j2", "s1"): model.new_int_var(0, 20, "s_j2_s1"),
    }
    op_end = {
        ("j1", "s1"): model.new_int_var(3, 23, "e_j1_s1"),
        ("j2", "s1"): model.new_int_var(2, 22, "e_j2_s1"),
    }
    op_intvl = {
        ("j1", "s1"): model.new_interval_var(
            op_start["j1", "s1"], 3, op_end["j1", "s1"], "i_j1_s1"
        ),
        ("j2", "s1"): model.new_interval_var(
            op_start["j2", "s1"], 2, op_end["j2", "s1"], "i_j2_s1"
        ),
    }
    params = SimpleNamespace(M_of={"s1": ["m1", "m2"]})
    variables = SimpleNamespace(op_start=op_start, op_end=op_end, op_intvl=op_intvl)
    slack_vars = BaseModelBuilder.add_right_slack_variables(
        model,
        params,
        slack_occupying_ops=(("j1", "s1", "m1"),),
        right_boundary_profile={"s1": [4, None]},
        horizon=20,
    )

    BaseModelBuilder.add_right_slack_constraints(
        model,
        params,
        variables,
        slack_occupying_ops=(("j1", "s1", "m1"),),
        right_time_fixed_ops=(("j2", "s1", "m2"),),
        right_boundary_profile={"s1": [4, None]},
        slack_vars=slack_vars,
    )
    model.maximize(op_end["j1", "s1"])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(op_end["j1", "s1"]) == 4


def test_operation_partition_properties_include_profile_fixed_ops():
    partition = OperationPartition(
        left_time_fixed=(("j1", "s1", "m1"),),
        left_profile_fixed=(("j2", "s1", "m2"),),
        unfixed=(("j3", "s1", "m1"),),
        right_profile_fixed=(("j4", "s1", "m2"),),
        right_time_fixed=(("j5", "s1", "m1"),),
    )

    assert partition.all_operations == (
        ("j1", "s1", "m1"),
        ("j2", "s1", "m2"),
        ("j3", "s1", "m1"),
        ("j4", "s1", "m2"),
        ("j5", "s1", "m1"),
    )
    assert partition.time_fixed_operations == (
        ("j1", "s1", "m1"),
        ("j5", "s1", "m1"),
    )
    assert partition.profile_fixed_operations == (
        ("j2", "s1", "m2"),
        ("j4", "s1", "m2"),
    )
    assert partition.slack_occupying_operations == (
        ("j1", "s1", "m1"),
        ("j2", "s1", "m2"),
        ("j3", "s1", "m1"),
        ("j4", "s1", "m2"),
    )


def test_operation_partition_promote_job_contained_ops_moves_profile_fixed_into_unfixed():
    partition = OperationPartition(
        left_time_fixed=(("j1", "s1", "m1"),),
        left_profile_fixed=(("j2", "s1", "m2"), ("j3", "s1", "m1")),
        unfixed=(("j2", "s2", "m3"),),
        right_profile_fixed=(("j2", "s3", "m4"), ("j4", "s1", "m2")),
        right_time_fixed=(("j5", "s1", "m1"),),
    )

    promoted = partition.promote_job_contained_ops()

    assert promoted.left_profile_fixed == (("j3", "s1", "m1"),)
    assert promoted.right_profile_fixed == (("j4", "s1", "m2"),)
    assert promoted.unfixed == (
        ("j2", "s1", "m2"),
        ("j2", "s2", "m3"),
        ("j2", "s3", "m4"),
    )
    assert promoted.left_time_fixed == partition.left_time_fixed
    assert promoted.right_time_fixed == partition.right_time_fixed


def test_save_solution_dict_writes_boundary_metadata(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = _make_schedule()
    partition = OperationPartition(
        left_time_fixed=(("j3", "s1", "m1"),),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"),),
        right_profile_fixed=(),
        right_time_fixed=(("j2", "s1", "m2"),),
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 7]},
        is_last_batch=False,
    )

    ctor._save_solution_dict(spec, sched, accepted=True)

    output_path = ctx.get_file_path_for_subroutine("_batch_001_solution.yaml")
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    saved = load_yaml(output_path)
    assert "highlight_ops:" in text
    assert "partition:" in text
    assert "right_boundary_profile" in text
    assert "accepted: true" in text.lower()
    assert saved["highlight_ops"] == [["j1", "s1"]]
    assert saved["partition"]["left_profile_fixed"] == []
    assert saved["partition"]["right_profile_fixed"] == []
    assert get_highlight_op_set(output_path) == {("j1", "s1")}


def test_save_solution_dict_normalizes_numpy_scalars(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = _make_schedule()
    partition = OperationPartition(
        left_time_fixed=(("j3", "s1", "m1"),),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"),),
        right_profile_fixed=(),
        right_time_fixed=(("j2", "s1", "m2"),),
    )
    spec = PwCpSubproblemSpec(
        batch_idx=np.int64(0),
        subproblem_idx=np.int64(1),
        partition=partition,
        right_boundary_profile={"s1": [np.int64(5), np.int64(7)]},
        is_last_batch=False,
    )

    ctor._save_solution_dict(spec, sched, accepted=True)

    output_path = ctx.get_file_path_for_subroutine("_batch_001_solution.yaml")
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "!!python" not in text
    assert "np.int64" not in text


def test_run_debug_export_saves_accepted_incumbent_with_highlight_ops(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    accepted_schedule = incumbent.deepcopy()
    accepted_schedule.make_semi_active({"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}})

    monkeypatch.setattr(
        ctor,
        "_solve_subproblem",
        lambda **kwargs: accepted_schedule,
    )
    monkeypatch.setattr(ctor.ctx, "check_feasibility", lambda start_time_map: 0.0)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    ctor.run(
        incumbent,
        instance,
        {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_size=2,
        max_time_per_batch=1.0,
        debug_export=True,
    )

    output_path = ctx.get_file_path_for_subroutine("_batch_001_solution.yaml")
    saved = load_yaml(output_path)
    assert saved["start_time_map"] == accepted_schedule.get_jik_2_start_time_map()
    assert saved["end_time_map"] == accepted_schedule.get_jik_2_end_time_map()
    assert saved["highlight_ops"] == [["j1", "s1"], ["j2", "s1"]]
    assert get_highlight_op_set(output_path) == {
        ("j1", "s1"),
        ("j2", "s1"),
    }
    assert (tmp_path / "batch_000_initial_gantt.png").exists()


def test_run_debug_export_saves_rejected_incumbent_with_highlight_ops(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()

    monkeypatch.setattr(
        ctor,
        "_solve_subproblem",
        lambda **kwargs: SimpleNamespace(
            makespan=incumbent.makespan + 1,
            get_jik_2_start_time_map=lambda: {},
        ),
    )

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    ctor.run(
        incumbent,
        instance,
        {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_size=2,
        max_time_per_batch=1.0,
        debug_export=True,
    )

    output_path = ctx.get_file_path_for_subroutine("_batch_001_solution.yaml")
    saved = load_yaml(output_path)
    assert saved["start_time_map"] == incumbent.get_jik_2_start_time_map()
    assert saved["end_time_map"] == incumbent.get_jik_2_end_time_map()
    assert saved["highlight_ops"] == [["j1", "s1"], ["j2", "s1"]]
    assert get_highlight_op_set(output_path) == {
        ("j1", "s1"),
        ("j2", "s1"),
    }


def test_run_collects_non_final_subproblem_obj_records(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    candidate_schedule = incumbent.deepcopy()
    candidate_schedule.make_semi_active({"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}})

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(
            elapsed_time=0.3,
            status="FEASIBLE",
            obj_value_records=((0.1, 4.0), (0.3, 2.0)),
            obj_bound_records=((0.05, 6.0), (0.3, 2.0)),
        ),
    )
    monkeypatch.setattr(
        ctx, "create_schedule", lambda *args, **kwargs: candidate_schedule
    )

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    result = ctor.run(
        incumbent,
        instance,
        {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_size=2,
        max_time_per_batch=1.0,
    )

    assert len(result.subproblem_logs) == 2
    first_log = result.subproblem_logs[0]
    assert first_log.objective_name == "right_slack"
    assert first_log.obj_value_records == ((0.1, 4.0), (0.3, 2.0))
    assert first_log.obj_bound_records == ((0.05, 6.0), (0.3, 2.0))


def test_run_collects_final_batch_as_makespan_objective(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    candidate_schedule = incumbent.deepcopy()

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(
            elapsed_time=0.4,
            status="OPTIMAL",
            obj_value_records=((0.2, 7.0),),
            obj_bound_records=((0.2, 7.0),),
        ),
    )
    monkeypatch.setattr(
        ctx, "create_schedule", lambda *args, **kwargs: candidate_schedule
    )

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    result = ctor.run(
        incumbent,
        instance,
        {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_size=4,
        max_time_per_batch=1.0,
    )

    assert len(result.subproblem_logs) == 1
    assert result.subproblem_logs[0].objective_name == "makespan"


def test_run_adds_makespan_hint_for_final_batch(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    incumbent_makespan = incumbent.makespan
    seen_hints = []

    def fake_solve_cp_model_2(mdl, *args, **kwargs):
        seen_hints.append(_get_named_hint_map(mdl))
        return _make_solver_report(status="OPTIMAL")

    monkeypatch.setattr(ctx, "solve_cp_model_2", fake_solve_cp_model_2)
    monkeypatch.setattr(
        ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy()
    )

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    ctor.run(
        incumbent,
        instance,
        {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_size=4,
        max_time_per_batch=1.0,
    )

    assert len(seen_hints) == 1
    assert seen_hints[0]["makespan"] == incumbent_makespan


def test_run_adds_right_slack_objective_hint_for_non_final_batch(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    seen_hints = []

    def fake_solve_cp_model_2(mdl, *args, **kwargs):
        seen_hints.append(_get_named_hint_map(mdl))
        return _make_solver_report(status="FEASIBLE")

    monkeypatch.setattr(ctx, "solve_cp_model_2", fake_solve_cp_model_2)
    monkeypatch.setattr(
        ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy()
    )

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    ctor.run(
        incumbent.deepcopy(),
        instance,
        {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        batch_size=2,
        max_time_per_batch=1.0,
    )

    assert len(seen_hints) == 2
    assert "slack_length" in seen_hints[0]
    partition, right_justified = ctor._build_operation_partition(
        ctor._build_stage_batches(incumbent, batch_size=2),
        ["s1"],
        current_batch_idx=0,
        incumbent=incumbent,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
    )
    _, params, _ = ctor.builder.build(instance, incumbent.makespan)
    expected_hints = ctor._compute_right_slack_hint_values(
        schedule=right_justified,
        slack_occupying_ops=partition.slack_occupying_operations,
        right_boundary_profile=partition.right_boundary_profile,
        params=params,
    )
    assert seen_hints[0]["slack_length"] == expected_hints["slack_length"]
    assert seen_hints[0]["slack_start_s1_1"] == expected_hints["slack_start_s1_1"]
    assert seen_hints[0]["slack_start_s1_2"] == expected_hints["slack_start_s1_2"]


def test_build_operation_partition_supports_profile_fixed_zones(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    batches = ctor._build_stage_batches(incumbent, batch_size=1)

    partition, _ = ctor._build_operation_partition(
        batches,
        ["s1"],
        current_batch_idx=1,
        incumbent=incumbent,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        left_profile_fixed_batch_count=1,
        right_profile_fixed_batch_count=1,
    )

    assert partition.left_time_fixed == ()
    assert partition.left_profile_fixed == (("j1", "s1", "m1"),)
    assert partition.unfixed == (("j2", "s1", "m2"),)
    assert partition.right_profile_fixed == (("j3", "s1", "m1"),)
    assert partition.right_time_fixed == (("j4", "s1", "m2"),)


def test_build_operation_partition_keeps_left_profile_fixed_on_last_batch(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    batches = ctor._build_stage_batches(incumbent, batch_size=1)

    partition, _ = ctor._build_operation_partition(
        batches,
        ["s1"],
        current_batch_idx=3,
        incumbent=incumbent,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        left_profile_fixed_batch_count=1,
        right_profile_fixed_batch_count=1,
    )

    assert partition.left_time_fixed == (("j1", "s1", "m1"), ("j2", "s1", "m2"))
    assert partition.left_profile_fixed == (("j3", "s1", "m1"),)
    assert partition.unfixed == (("j4", "s1", "m2"),)
    assert partition.right_profile_fixed == ()
    assert partition.right_time_fixed == ()


def test_run_promotes_profile_fixed_ops_when_enabled(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_multi_stage_schedule()
    seen_partitions = []

    def fake_solve_subproblem(**kwargs):
        seen_partitions.append(kwargs["spec"].partition)
        return incumbent.deepcopy()

    monkeypatch.setattr(ctor, "_solve_subproblem", fake_solve_subproblem)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
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
        batch_size=1,
        left_profile_fixed_batch_count=1,
        enable_promotion_profile_fixed=True,
        max_time_per_batch=1.0,
    )

    first_partition = seen_partitions[0]
    assert ("j1", "s1", "m1") in first_partition.unfixed
    assert ("j1", "s2", "m3") in first_partition.unfixed
    assert first_partition.left_profile_fixed == ()


def test_solve_subproblem_adds_profile_fixed_precedence_constraints(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_multi_stage_schedule()
    captured = {}

    def fake_add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
        mdl,
        params,
        variables,
        current_schedule,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
    ):
        captured["ops"] = sorted(current_schedule.get_jik_2_start_time_map())
        captured["profile_fix_by_machine"] = profile_fix_by_machine
        captured["machine_precedence_stride"] = machine_precedence_stride

    monkeypatch.setattr(
        BaseModelBuilder,
        "add_stage_ops_precedence_constraints_after_dispatch_from_schedule",
        staticmethod(
            fake_add_stage_ops_precedence_constraints_after_dispatch_from_schedule
        ),
    )
    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(status="FEASIBLE"),
    )
    monkeypatch.setattr(
        ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy()
    )

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1", "s2"],
        {"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
        {
            "j1": {"s1": 2, "s2": 3},
            "j2": {"s1": 3, "s2": 3},
            "j3": {"s1": 2, "s2": 3},
            "j4": {"s1": 2, "s2": 3},
        },
    )

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(("j1", "s1", "m1"),),
        unfixed=(("j2", "s1", "m2"),),
        right_profile_fixed=(("j3", "s2", "m3"),),
        right_time_fixed=(("j4", "s2", "m4"),),
        right_boundary_profile={"s1": [5, 7], "s2": [8, 10]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 7], "s2": [8, 10]},
        is_last_batch=False,
    )

    ctor._solve_subproblem(
        spec=spec,
        incumbent=incumbent,
        right_justified_schedule=incumbent.deepcopy(),
        instance=instance,
        stage_2_job_2_p_dict={
            "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2},
            "s2": {"j1": 3, "j2": 3, "j3": 3, "j4": 3},
        },
        max_time_per_batch=1.0,
        solver_thread_cnt=1,
        profile_fix_by_machine=True,
        machine_precedence_stride=2,
        use_lns_only=False,
        tighten_ranges=False,
        link_job_completion=False,
        debug_export=False,
    )

    assert captured["ops"] == [("j1", "s1", "m1"), ("j3", "s2", "m3")]
    assert captured["profile_fix_by_machine"] is True
    assert captured["machine_precedence_stride"] == 2
    ctor._st = None


def test_apply_successor_machine_reassignment_moves_boundary_suffix(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s0", "s1"],
        machines_per_stage={"s0": ["u1", "u2", "u3", "u4"], "s1": ["m1", "m2"]},
    )
    candidate.add_ops_times_2_mc("s0", "u1", "j1", 0, 2)
    candidate.add_ops_times_2_mc("s0", "u2", "j2", 0, 2)
    candidate.add_ops_times_2_mc("s0", "u3", "j3", 0, 5)
    candidate.add_ops_times_2_mc("s0", "u4", "j4", 0, 5)
    candidate.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m1", "j4", 5, 7)
    candidate.add_ops_times_2_mc("s1", "m2", "j2", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m2", "j3", 5, 7)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )

    ctor._apply_successor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s1": [("m1", 2, 5)]},
        right_boundary_profile={"s1": [5, 5]},
        partition=partition,
        stage_2_job_2_p_dict={
            "s0": {"j1": 2, "j2": 2, "j3": 5, "j4": 5},
            "s1": {"j1": 2, "j2": 2, "j3": 2, "j4": 2},
        },
    )
    candidate.make_semi_active(
        {
            "s0": {"j1": 2, "j2": 2, "j3": 5, "j4": 5},
            "s1": {"j1": 2, "j2": 2, "j3": 2, "j4": 2},
        }
    )

    start_map = candidate.get_jik_2_start_time_map()
    assert start_map[("j1", "s1", "m1")] == 2
    assert start_map[("j3", "s1", "m1")] == 5
    assert start_map[("j2", "s1", "m2")] == 2
    assert start_map[("j4", "s1", "m2")] == 5


def test_apply_successor_machine_reassignment_handles_multiple_intervals(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["a", "b", "c", "d", "e", "f"],
        stages=["s0", "s1"],
        machines_per_stage={
            "s0": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "s1": ["m1", "m2", "m3"],
        },
    )
    candidate.add_ops_times_2_mc("s0", "u1", "a", 0, 2)
    candidate.add_ops_times_2_mc("s0", "u2", "b", 0, 2)
    candidate.add_ops_times_2_mc("s0", "u3", "c", 0, 2)
    candidate.add_ops_times_2_mc("s0", "u4", "d", 0, 6)
    candidate.add_ops_times_2_mc("s0", "u5", "e", 0, 8)
    candidate.add_ops_times_2_mc("s0", "u6", "f", 0, 10)
    candidate.add_ops_times_2_mc("s1", "m1", "a", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m1", "f", 10, 12)
    candidate.add_ops_times_2_mc("s1", "m2", "b", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m2", "d", 6, 8)
    candidate.add_ops_times_2_mc("s1", "m3", "c", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m3", "e", 8, 10)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("a", "s1", "m1"), ("b", "s1", "m2"), ("c", "s1", "m3")),
        right_profile_fixed=(),
        right_time_fixed=(
            ("d", "s1", "m1"),
            ("e", "s1", "m2"),
            ("f", "s1", "m3"),
        ),
        right_boundary_profile={"s1": [6, 8, 10]},
    )

    ctor._apply_successor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s1": [("m1", 2, 6), ("m2", 2, 8)]},
        right_boundary_profile={"s1": [6, 8, 10]},
        partition=partition,
        stage_2_job_2_p_dict={
            "s0": {"a": 2, "b": 2, "c": 2, "d": 6, "e": 8, "f": 10},
            "s1": {job_id: 2 for job_id in ["a", "b", "c", "d", "e", "f"]},
        },
    )
    candidate.make_semi_active(
        {
            "s0": {"a": 2, "b": 2, "c": 2, "d": 6, "e": 8, "f": 10},
            "s1": {job_id: 2 for job_id in ["a", "b", "c", "d", "e", "f"]},
        }
    )

    start_map = candidate.get_jik_2_start_time_map()
    assert start_map[("d", "s1", "m1")] == 6
    assert start_map[("e", "s1", "m2")] == 8


def test_apply_successor_machine_reassignment_is_noop_when_already_aligned(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = _make_schedule()
    before = candidate.get_jik_2_start_time_map().copy()
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [4, 5]},
    )

    ctor._apply_successor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s1": [("m1", 2, 4), ("m2", 3, 5)]},
        right_boundary_profile={"s1": [4, 5]},
        partition=partition,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
    )

    assert candidate.get_jik_2_start_time_map() == before


def test_apply_successor_machine_reassignment_sorts_by_slack_start(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["a", "b", "c", "d", "e", "f", "g"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2", "m3"]},
    )
    candidate.add_ops_times_2_mc("s1", "m1", "a", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m1", "f", 10, 12)
    candidate.add_ops_times_2_mc("s1", "m2", "b", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m2", "d", 6, 8)
    candidate.add_ops_times_2_mc("s1", "m2", "g", 9, 11)
    candidate.add_ops_times_2_mc("s1", "m3", "c", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m3", "e", 8, 10)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(
            ("a", "s1", "m1"),
            ("b", "s1", "m2"),
            ("c", "s1", "m3"),
            ("g", "s1", "m2"),
        ),
        right_profile_fixed=(),
        right_time_fixed=(
            ("d", "s1", "m1"),
            ("e", "s1", "m2"),
            ("f", "s1", "m3"),
        ),
        right_boundary_profile={"s1": [6, 8, 10]},
    )
    seen_target_machine_ids = []

    def fake_swap_stage_machine_operation_sets(**kwargs):
        seen_target_machine_ids.append(kwargs["to_machine_id"])

    monkeypatch.setattr(
        candidate,
        "swap_stage_machine_operation_sets",
        fake_swap_stage_machine_operation_sets,
    )

    ctor._apply_successor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s1": [("m2", 5, 8), ("m1", 2, 6)]},
        right_boundary_profile={"s1": [6, 8, 10]},
        partition=partition,
        stage_2_job_2_p_dict={
            "s1": {job_id: 2 for job_id in ["a", "b", "c", "d", "e", "f", "g"]}
        },
    )

    assert seen_target_machine_ids == ["m1", "m2"]


def test_apply_successor_machine_reassignment_skips_warning_when_slack_end_is_makespan(
    caplog, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    candidate.add_ops_times_2_mc("s1", "m1", "j1", 0, 10)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(),
        right_profile_fixed=(),
        right_time_fixed=(),
        right_boundary_profile={"s1": [10]},
    )

    with caplog.at_level(logging.WARNING):
        ctor._apply_successor_machine_reassignment(
            candidate_schedule=candidate,
            solved_slack_intervals={"s1": [("m1", 5, 10)]},
            right_boundary_profile={"s1": [10]},
            partition=partition,
            stage_2_job_2_p_dict={"s1": {"j1": 10}},
        )

    assert "no right-boundary job starts at slack_end" not in caplog.text


def test_apply_predecessor_machine_reassignment_redispatches_by_deadline(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4", "j5", "j6"],
        stages=["s0", "s1"],
        machines_per_stage={"s0": ["u1", "u2", "u3", "u4", "u5", "u6"], "s1": ["m1", "m2", "m3"]},
    )
    for machine_id, job_id, end_time in [
        ("u1", "j1", 4),
        ("u2", "j2", 2),
        ("u3", "j3", 1),
        ("u4", "j4", 4),
        ("u5", "j5", 9),
        ("u6", "j6", 8),
    ]:
        candidate.add_ops_times_2_mc("s0", machine_id, job_id, 0, end_time)
    candidate.add_ops_times_2_mc("s1", "m1", "j1", 4, 6)
    candidate.add_ops_times_2_mc("s1", "m1", "j4", 6, 8)
    candidate.add_ops_times_2_mc("s1", "m2", "j2", 2, 4)
    candidate.add_ops_times_2_mc("s1", "m2", "j5", 9, 11)
    candidate.add_ops_times_2_mc("s1", "m3", "j3", 1, 3)
    candidate.add_ops_times_2_mc("s1", "m3", "j6", 8, 10)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2"), ("j3", "s1", "m3")),
        right_profile_fixed=(),
        right_time_fixed=(
            ("j4", "s1", "m1"),
            ("j5", "s1", "m2"),
            ("j6", "s1", "m3"),
        ),
        right_boundary_profile={"s1": [4, 9, 8]},
    )

    ctor._apply_predecessor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s1": [("m1", 1, 4), ("m2", 8, 9), ("m3", 3, 8)]},
        partition=partition,
        stage_2_job_2_p_dict={
            "s0": {"j1": 4, "j2": 2, "j3": 1, "j4": 4, "j5": 9, "j6": 8},
            "s1": {"j1": 2, "j2": 2, "j3": 2, "j4": 2, "j5": 2, "j6": 2},
        },
    )
    candidate.make_semi_active(
        {
            "s0": {"j1": 4, "j2": 2, "j3": 1, "j4": 4, "j5": 9, "j6": 8},
            "s1": {"j1": 2, "j2": 2, "j3": 2, "j4": 2, "j5": 2, "j6": 2},
        }
    )

    start_map = candidate.get_jik_2_start_time_map()
    end_map = candidate.get_jik_2_end_time_map()
    assert end_map[("j1", "s1", "m2")] <= 6
    assert end_map[("j2", "s1", "m2")] <= 4
    assert end_map[("j3", "s1", "m3")] <= 3
    assert start_map[("j4", "s1", "m1")] >= 4


def test_apply_predecessor_machine_reassignment_only_reassigns_target_machines(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2", "m3"]},
    )
    candidate.add_ops_times_2_mc("s1", "m1", "j1", 0, 2)
    candidate.add_ops_times_2_mc("s1", "m1", "j4", 6, 8)
    candidate.add_ops_times_2_mc("s1", "m2", "j2", 2, 4)
    candidate.add_ops_times_2_mc("s1", "m3", "j3", 4, 6)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2"), ("j3", "s1", "m3")),
        right_profile_fixed=(),
        right_time_fixed=(("j4", "s1", "m1"),),
        right_boundary_profile={"s1": [6, 0, 0]},
    )
    before = candidate.get_jik_2_start_time_map().copy()

    ctor._apply_predecessor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s1": [("m1", 4, 6), ("m2", 4, 4)]},
        partition=partition,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 2, "j3": 2, "j4": 2}},
    )

    after = candidate.get_jik_2_start_time_map()
    assert after[("j3", "s1", "m3")] == before[("j3", "s1", "m3")]


def test_apply_predecessor_machine_reassignment_skips_when_no_predecessor_exists(
    tmp_path,
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    candidate.add_ops_times_2_mc("s1", "m1", "j3", 5, 7)
    candidate.add_ops_times_2_mc("s1", "m2", "j4", 5, 7)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    before = candidate.get_jik_2_start_time_map().copy()

    ctor._apply_predecessor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s1": [("m1", 1, 5), ("m2", 2, 5)]},
        partition=partition,
        stage_2_job_2_p_dict={"s1": {"j3": 2, "j4": 2}},
    )

    assert candidate.get_jik_2_start_time_map() == before


def test_apply_predecessor_machine_reassignment_processes_stages_in_reverse_order(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s0", "s1"],
        machines_per_stage={"s0": ["m0"], "s1": ["m1"]},
    )
    candidate.add_ops_times_2_mc("s0", "m0", "j1", 0, 3)
    candidate.add_ops_times_2_mc("s1", "m1", "j1", 3, 5)
    candidate.add_ops_times_2_mc("s0", "m0", "j2", 3, 6)
    candidate.add_ops_times_2_mc("s1", "m1", "j2", 6, 8)

    seen_stage_ids = []
    orig = candidate.dispatch_stage_reversed_by_jobs

    def spy(*args, **kwargs):
        seen_stage_ids.append(kwargs["stage_id"])
        return orig(*args, **kwargs)

    monkeypatch.setattr(candidate, "dispatch_stage_reversed_by_jobs", spy)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s0", "m0"), ("j1", "s1", "m1")),
        right_profile_fixed=(),
        right_time_fixed=(("j2", "s0", "m0"), ("j2", "s1", "m1")),
        right_boundary_profile={"s0": [3], "s1": [6]},
    )

    ctor._apply_predecessor_machine_reassignment(
        candidate_schedule=candidate,
        solved_slack_intervals={"s0": [("m0", 3, 6)], "s1": [("m1", 6, 8)]},
        partition=partition,
        stage_2_job_2_p_dict={"s0": {"j1": 3, "j2": 3}, "s1": {"j1": 2, "j2": 2}},
    )

    assert seen_stage_ids == ["s1", "s0"]


def test_apply_predecessor_machine_reassignment_surfaces_reverse_dispatch_infeasibility(
    tmp_path,
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    candidate = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    candidate.add_ops_times_2_mc("s1", "m1", "j1", 0, 1)
    candidate.add_ops_times_2_mc("s1", "m1", "j2", 1, 3)

    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"),),
        right_profile_fixed=(),
        right_time_fixed=(("j2", "s1", "m1"),),
        right_boundary_profile={"s1": [1]},
    )

    with pytest.raises(ValueError, match="Unable to reverse-dispatch job j1 on stage s1"):
        ctor._apply_predecessor_machine_reassignment(
            candidate_schedule=candidate,
            solved_slack_intervals={"s1": [("m1", 1, 3)]},
            partition=partition,
            stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 2}},
        )


def test_solve_subproblem_non_final_reassigns_slack_before_make_semi_active(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )
    call_order = []

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(status="FEASIBLE", obj_value=1.0),
    )
    monkeypatch.setattr(ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy())
    monkeypatch.setattr(
        ctor,
        "_extract_solved_slack_intervals",
        lambda *args, **kwargs: {"s1": [("m1", 2, 5)]},
    )

    def fake_successor_reassign(*args, **kwargs):
        call_order.append("successor_reassign")

    monkeypatch.setattr(
        ctor, "_apply_successor_machine_reassignment", fake_successor_reassign
    )
    monkeypatch.setattr(
        ctor,
        "_apply_predecessor_machine_reassignment",
        lambda *args, **kwargs: call_order.append("predecessor_reassign"),
    )

    candidate_for_order = incumbent.deepcopy()

    def fake_create_schedule(*args, **kwargs):
        def fake_make_semi_active(*_args, **_kwargs):
            call_order.append("semi_active")

        candidate_for_order.make_semi_active = fake_make_semi_active
        return candidate_for_order

    monkeypatch.setattr(ctx, "create_schedule", fake_create_schedule)

    ctor._solve_subproblem(
        spec=spec,
        incumbent=incumbent,
        right_justified_schedule=incumbent.deepcopy(),
        instance=instance,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        max_time_per_batch=1.0,
        solver_thread_cnt=1,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
        use_lns_only=False,
        tighten_ranges=False,
        link_job_completion=False,
        debug_export=False,
    )

    assert call_order == [
        "successor_reassign",
        "predecessor_reassign",
        "semi_active",
    ]
    ctor._st = None


def test_solve_subproblem_non_final_disables_search_log_when_debug_export_is_false(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )
    seen = {}

    def fake_solve_cp_model_2(*args, **kwargs):
        seen["log_search_progress"] = kwargs["log_search_progress"]
        return _make_solver_report(status="FEASIBLE", obj_value=0.0)

    monkeypatch.setattr(ctx, "solve_cp_model_2", fake_solve_cp_model_2)
    monkeypatch.setattr(ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy())

    ctor._solve_subproblem(
        spec=spec,
        incumbent=incumbent,
        right_justified_schedule=incumbent.deepcopy(),
        instance=instance,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        max_time_per_batch=1.0,
        solver_thread_cnt=1,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
        use_lns_only=False,
        tighten_ranges=False,
        link_job_completion=False,
        debug_export=False,
    )

    assert seen["log_search_progress"] is False
    assert not (tmp_path / "cp_sat_search_batch=1.log").exists()
    ctor._st = None


def test_solve_subproblem_final_enables_search_log_when_debug_export_is_true(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=True,
    )
    seen = {}

    def fake_solve_cp_model_2(*args, **kwargs):
        seen["log_search_progress"] = kwargs["log_search_progress"]
        return _make_solver_report(status="FEASIBLE", obj_value=7.0)

    monkeypatch.setattr(ctx, "solve_cp_model_2", fake_solve_cp_model_2)
    monkeypatch.setattr(ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy())

    ctor._solve_subproblem(
        spec=spec,
        incumbent=incumbent,
        right_justified_schedule=incumbent.deepcopy(),
        instance=instance,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        max_time_per_batch=1.0,
        solver_thread_cnt=1,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
        use_lns_only=False,
        tighten_ranges=False,
        link_job_completion=False,
        debug_export=True,
    )

    assert seen["log_search_progress"] is True
    ctor._st = None


def test_solve_subproblem_non_final_obj_value_zero_skips_reassignments_and_still_retimes(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )
    call_order = []

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(status="FEASIBLE", obj_value=0.0),
    )
    monkeypatch.setattr(
        ctor,
        "_extract_solved_slack_intervals",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("slack intervals should not be extracted when obj_value == 0")
        ),
    )
    monkeypatch.setattr(
        ctor,
        "_apply_successor_machine_reassignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("successor reassignment should be skipped when obj_value == 0")
        ),
    )
    monkeypatch.setattr(
        ctor,
        "_apply_predecessor_machine_reassignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "predecessor reassignment should be skipped when obj_value == 0"
            )
        ),
    )

    candidate_for_order = incumbent.deepcopy()

    def fake_create_schedule(*args, **kwargs):
        def fake_make_semi_active(*_args, **_kwargs):
            call_order.append("semi_active")

        candidate_for_order.make_semi_active = fake_make_semi_active
        return candidate_for_order

    monkeypatch.setattr(ctx, "create_schedule", fake_create_schedule)

    ctor._solve_subproblem(
        spec=spec,
        incumbent=incumbent,
        right_justified_schedule=incumbent.deepcopy(),
        instance=instance,
        stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
        max_time_per_batch=1.0,
        solver_thread_cnt=1,
        profile_fix_by_machine=False,
        machine_precedence_stride=1,
        use_lns_only=False,
        tighten_ranges=False,
        link_job_completion=False,
        debug_export=False,
    )

    assert call_order == ["semi_active"]
    for filename in [
        "batch_001_candidate_01_before_retiming.png",
        "batch_001_candidate_02_after_successor_reassign.png",
        "batch_001_candidate_03_after_predecessor_reassign.png",
        "batch_001_candidate_04_after_semi_active.png",
    ]:
        assert not (tmp_path / filename).exists()
    ctor._st = None


def test_solve_subproblem_non_final_raises_on_negative_obj_value(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(status="FEASIBLE", obj_value=-1.0),
    )
    monkeypatch.setattr(ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy())

    with pytest.raises(ValueError, match="right-slack objective must be >= 0"):
        ctor._solve_subproblem(
            spec=spec,
            incumbent=incumbent,
            right_justified_schedule=incumbent.deepcopy(),
            instance=instance,
            stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
            max_time_per_batch=1.0,
            solver_thread_cnt=1,
            profile_fix_by_machine=False,
            machine_precedence_stride=1,
            use_lns_only=False,
            tighten_ranges=False,
            link_job_completion=False,
            debug_export=False,
        )

    assert (tmp_path / "batch_001_candidate_01_before_retiming.png").exists()
    assert not (tmp_path / "batch_001_candidate_04_after_semi_active.png").exists()
    ctor._st = None


def test_machine_order_right_boundary_profile_differs_from_stage_earliest_starts(
    tmp_path,
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["A", "B", "C", "D"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    # Feasible schedule:
    # m1: A [0,10]
    # m2: B [0,2], C [2,4], D [8,10]
    sched.add_ops_times_2_mc("s1", "m1", "A", 0, 10)
    sched.add_ops_times_2_mc("s1", "m2", "B", 0, 2)
    sched.add_ops_times_2_mc("s1", "m2", "C", 2, 4)
    sched.add_ops_times_2_mc("s1", "m2", "D", 8, 10)

    batches = ctor._build_stage_batches(sched, batch_size=2, sort_by_start_time=False)
    assert batches == {
        "s1": [
            (("B", "s1", "m2"), ("C", "s1", "m2")),
            (("A", "s1", "m1"), ("D", "s1", "m2")),
        ]
    }

    partition, _ = ctor._build_operation_partition(
        batches,
        ["s1"],
        current_batch_idx=0,
        incumbent=sched,
        stage_2_job_2_p_dict={"s1": {"A": 10, "B": 2, "C": 2, "D": 2}},
    )
    assert partition.unfixed == (("B", "s1", "m2"), ("C", "s1", "m2"))
    assert partition.right_time_fixed == (("A", "s1", "m1"), ("D", "s1", "m2"))
    # Machine-order boundaries are derived per machine, not by taking the stage's
    # globally earliest starts.
    assert partition.right_boundary_profile == {"s1": [0, 8]}

    instance = create_hfs_instance(
        "tiny",
        ["A", "B", "C", "D"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"A": {"s1": 10}, "B": {"s1": 2}, "C": {"s1": 2}, "D": {"s1": 2}},
    )
    _, params, _ = ctor.builder.build(instance, sched.makespan)
    hint_values = ctor._compute_right_slack_hint_values(
        schedule=sched,
        slack_occupying_ops=partition.slack_occupying_operations,
        right_boundary_profile=partition.right_boundary_profile,
        params=params,
    )

    # Actual slack lengths are [0, 4], so the shared hint length is 0 and
    # each start is recomputed as slack_end - 0.
    assert hint_values["slack_length"] == 0
    assert hint_values["slack_start_s1_1"] == 0
    assert hint_values["slack_start_s1_2"] == 8


def test_compute_right_slack_hint_values_skips_none_boundary_machine(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 3, 8)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 0, 2)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 5}, "j2": {"s1": 2}},
    )
    _, params, _ = ctor.builder.build(instance, sched.makespan)

    hint_values = ctor._compute_right_slack_hint_values(
        schedule=sched,
        slack_occupying_ops=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_boundary_profile={"s1": [6, None]},
        params=params,
    )

    assert hint_values["slack_length"] == 0
    assert hint_values["slack_start_s1_1"] == 6
    assert "slack_start_s1_2" not in hint_values


def test_compute_right_slack_hint_values_clips_latest_end_at_slack_boundary(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 3, 8)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 0, 2)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 5}, "j2": {"s1": 2}},
    )
    _, params, _ = ctor.builder.build(instance, sched.makespan)

    hint_values = ctor._compute_right_slack_hint_values(
        schedule=sched,
        slack_occupying_ops=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_boundary_profile={"s1": [6, 5]},
        params=params,
    )

    # Actual slack lengths are [0, 3], so the shared hint length is 0 and
    # each start is recomputed as slack_end - 0.
    assert hint_values["slack_length"] == 0
    assert hint_values["slack_start_s1_1"] == 6
    assert hint_values["slack_start_s1_2"] == 5


def test_compute_right_slack_hint_values_aggregates_stage_and_global_mins(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", 1, 3)
    sched.add_ops_times_2_mc("s1", "m2", "j2", 4, 5)
    sched.add_ops_times_2_mc("s2", "m3", "j3", 0, 1)
    sched.add_ops_times_2_mc("s2", "m4", "j4", 3, 6)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1", "s2"],
        {"s1": ["m1", "m2"], "s2": ["m3", "m4"]},
        {
            "j1": {"s1": 2, "s2": 1},
            "j2": {"s1": 1, "s2": 1},
            "j3": {"s1": 1, "s2": 1},
            "j4": {"s1": 1, "s2": 3},
        },
    )
    _, params, _ = ctor.builder.build(instance, sched.makespan)

    hint_values = ctor._compute_right_slack_hint_values(
        schedule=sched,
        slack_occupying_ops=(
            ("j1", "s1", "m1"),
            ("j2", "s1", "m2"),
            ("j3", "s2", "m3"),
            ("j4", "s2", "m4"),
        ),
        right_boundary_profile={"s1": [6, 7], "s2": [4, 8]},
        params=params,
    )

    # Actual slack lengths are [3, 2, 3, 2], so the shared hint length is 2 and
    # each start is recomputed as slack_end - 2.
    assert hint_values["slack_length"] == 2
    assert hint_values["slack_start_s1_1"] == 4
    assert hint_values["slack_start_s1_2"] == 5
    assert hint_values["slack_start_s2_1"] == 2
    assert hint_values["slack_start_s2_2"] == 6


def test_apply_right_slack_hints_skips_duplicates(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    mdl, params, variables = ctor.builder.build(instance, incumbent.makespan)
    slack_vars = BaseModelBuilder.add_right_slack_variables(
        mdl,
        params,
        slack_occupying_ops=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
        horizon=incumbent.makespan,
    )
    BaseModelBuilder.add_right_slack_objective(
        mdl,
        slack_occupying_ops=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        slack_vars=slack_vars,
    )
    ctor._apply_right_slack_hints(
        mdl,
        schedule=incumbent,
        slack_occupying_ops=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
        params=params,
    )

    hint_var_indices = _get_hint_var_indices(mdl)
    assert len(hint_var_indices) == len(set(hint_var_indices))


def test_result_save_yaml_preserves_obj_series_and_adds_cp_sat_logs(tmp_path):
    obj_store = ObjValueBoundStore[int]()
    obj_store.obj_value_series.name = "ObjVal after PW-CP batch"
    obj_store.add_obj_value(0.0, 12, None)
    obj_store.add_last_timestamp_note(
        "initial_schedule", obj_value_is_valid=True, obj_bound_is_valid=False
    )
    obj_store.add_obj_value(1.0, 10, None)
    obj_store.add_last_timestamp_note(
        "batch=1", obj_value_is_valid=True, obj_bound_is_valid=False
    )

    result = PwCpResult(
        schedule=_make_schedule(),
        sub_obj_store=obj_store,
        last_obj_value=10,
        subproblem_logs=(
            PwCpSubproblemLog(
                batch_idx=0,
                subproblem_idx=1,
                is_last_batch=False,
                objective_name="right_slack",
                time_limit_sec=2.0,
                elapsed_time_sec=0.8,
                status="FEASIBLE",
                incumbent_makespan_before=10,
                candidate_makespan=9,
                accepted=True,
                obj_value_records=((0.1, 3.0), (0.4, 1.0)),
                obj_bound_records=((0.1, 4.0), (0.4, 1.0)),
            ),
        ),
        total_pw_cp_elapsed_sec=1.5,
        max_time_per_batch=2.0,
    )
    output_path = tmp_path / "pw_cp_obj_log.yaml"
    result.save_yaml(output_path)

    saved = load_yaml(output_path)
    assert saved["obj_value"]["data"] == {"0.0": 12, "1.0": 10}
    assert saved["obj_value"]["notes"] == {
        "0.0": "initial_schedule",
        "1.0": "batch=1",
    }
    assert "obj_bound" not in saved
    assert (
        saved["pw_cp_metadata"]["cp_sat_subproblems"][0]["objective_name"]
        == "right_slack"
    )
    assert saved["pw_cp_metadata"]["cp_sat_subproblems"][0]["accepted"] is True
    assert saved["pw_cp_metadata"]["cp_sat_subproblems"][0]["improved"] is True
    assert saved["pw_cp_metadata"]["summary"]["subproblem_count"] == 1
    assert saved["pw_cp_metadata"]["summary"]["accepted_count"] == 1
    assert saved["pw_cp_metadata"]["summary"]["improving_subproblem_count"] == 1
    assert saved["pw_cp_metadata"]["summary"]["last_obj_improvement_time_sec"] == 0.4
    assert saved["pw_cp_metadata"]["summary"]["final_incumbent_makespan"] == 10
    assert saved["pw_cp_metadata"]["summary"]["max_time_per_batch"] == 2.0
    loaded_store = ObjValueBoundStore.load_yaml(output_path)
    assert loaded_store.obj_value_series.items() == [(0.0, 12), (1.0, 10)]
    assert loaded_store.obj_bound_series.items() == []


def test_result_save_yaml_handles_rejected_and_empty_records(tmp_path):
    obj_store = ObjValueBoundStore[int]()
    obj_store.obj_value_series.name = "ObjVal after PW-CP batch"
    obj_store.add_obj_value(0.0, 12, None)
    obj_store.add_last_timestamp_note(
        "initial_schedule", obj_value_is_valid=True, obj_bound_is_valid=False
    )
    obj_store.add_obj_value(2.0, 11, None)
    obj_store.add_last_timestamp_note(
        "batch=2", obj_value_is_valid=True, obj_bound_is_valid=False
    )

    result = PwCpResult(
        schedule=_make_schedule(),
        sub_obj_store=obj_store,
        last_obj_value=11,
        subproblem_logs=(
            PwCpSubproblemLog(
                batch_idx=1,
                subproblem_idx=2,
                is_last_batch=True,
                objective_name="makespan",
                time_limit_sec=3.0,
                elapsed_time_sec=3.0,
                status="UNKNOWN",
                incumbent_makespan_before=11,
                candidate_makespan=None,
                accepted=False,
                obj_value_records=(),
                obj_bound_records=(),
            ),
        ),
        total_pw_cp_elapsed_sec=3.0,
        max_time_per_batch=3.0,
    )

    output_path = tmp_path / "pw_cp_obj_log_empty.yaml"
    result.save_yaml(output_path)

    saved = load_yaml(output_path)
    assert saved["pw_cp_metadata"]["cp_sat_subproblems"][0]["obj_value_records"] == []
    assert (
        saved["pw_cp_metadata"]["cp_sat_subproblems"][0]["candidate_makespan"] is None
    )
    assert saved["pw_cp_metadata"]["summary"]["accepted_count"] == 0
    assert saved["pw_cp_metadata"]["summary"]["last_obj_improvement_time_sec"] is None
    assert "obj_bound" not in saved
    loaded_store = ObjValueBoundStore.load_yaml(output_path)
    assert loaded_store.obj_value_series.items() == [(0.0, 12), (2.0, 11)]
    assert loaded_store.obj_bound_series.items() == []


def test_controller_pw_cp_invokes_constructor(monkeypatch):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = module.HybridFlowShopCpLnsController.__new__(
        module.HybridFlowShopCpLnsController
    )
    schedule = SimpleNamespace(makespan=11)
    ctrl.instance = object()
    ctrl.stage_2_job_2_p_dict = {"s1": {"j1": 1}}
    ctrl.timer = SimpleNamespace(elapsed_sec=1.25)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: schedule,
        register=lambda report, solution: True,
    )
    ctrl.get_file_path_for_subroutine = lambda suffix: Path("/tmp") / suffix.lstrip("_")
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "pw_cp"
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None
    )
    ctrl.set_cp_model_as_base_cp_model = lambda *args, **kwargs: None
    ctrl.draw_incumbent_gantt = lambda *args, **kwargs: None

    class FakeConstructor:
        def __init__(self, ctx):
            self.ctx = ctx

        def run(self, *args, **kwargs):
            obj_store = ObjValueBoundStore[int]()
            return PwCpResult(
                schedule=schedule,
                sub_obj_store=obj_store,
                last_obj_value=11,
                subproblem_logs=(),
                total_pw_cp_elapsed_sec=0.1,
                max_time_per_batch=None,
            )

    monkeypatch.setattr(module, "PwCpConstructor", FakeConstructor)

    ctrl.pw_cp(solver_thread_cnt=1)


def test_controller_pw_cp_resolves_batch_size_ratio(monkeypatch):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = module.HybridFlowShopCpLnsController.__new__(
        module.HybridFlowShopCpLnsController
    )
    schedule = SimpleNamespace(makespan=11)
    ctrl.instance = SimpleNamespace(job_count=37)
    ctrl.stage_2_job_2_p_dict = {"s1": {"j1": 1}}
    ctrl.timer = SimpleNamespace(elapsed_sec=1.25)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: schedule,
        register=lambda report, solution: True,
    )
    ctrl.get_file_path_for_subroutine = lambda suffix: Path("/tmp") / suffix.lstrip("_")
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "pw_cp"
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None
    )
    ctrl.set_cp_model_as_base_cp_model = lambda *args, **kwargs: None
    ctrl.draw_incumbent_gantt = lambda *args, **kwargs: None

    seen_kwargs = {}

    class FakeConstructor:
        def __init__(self, ctx):
            self.ctx = ctx

        def run(self, *args, **kwargs):
            seen_kwargs.update(kwargs)
            obj_store = ObjValueBoundStore[int]()
            return PwCpResult(
                schedule=schedule,
                sub_obj_store=obj_store,
                last_obj_value=11,
                subproblem_logs=(),
                total_pw_cp_elapsed_sec=0.1,
                max_time_per_batch=None,
            )

    monkeypatch.setattr(module, "PwCpConstructor", FakeConstructor)

    ctrl.pw_cp(solver_thread_cnt=1, batch_size_ratio=0.10)

    assert seen_kwargs["batch_size"] == 4


def test_controller_pw_cp_prefers_explicit_batch_size_over_ratio(monkeypatch):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = module.HybridFlowShopCpLnsController.__new__(
        module.HybridFlowShopCpLnsController
    )
    schedule = SimpleNamespace(makespan=11)
    ctrl.instance = SimpleNamespace(job_count=37)
    ctrl.stage_2_job_2_p_dict = {"s1": {"j1": 1}}
    ctrl.timer = SimpleNamespace(elapsed_sec=1.25)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: schedule,
        register=lambda report, solution: True,
    )
    ctrl.get_file_path_for_subroutine = lambda suffix: Path("/tmp") / suffix.lstrip("_")
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "pw_cp"
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None
    )
    ctrl.set_cp_model_as_base_cp_model = lambda *args, **kwargs: None
    ctrl.draw_incumbent_gantt = lambda *args, **kwargs: None

    seen_kwargs = {}

    class FakeConstructor:
        def __init__(self, ctx):
            self.ctx = ctx

        def run(self, *args, **kwargs):
            seen_kwargs.update(kwargs)
            obj_store = ObjValueBoundStore[int]()
            return PwCpResult(
                schedule=schedule,
                sub_obj_store=obj_store,
                last_obj_value=11,
                subproblem_logs=(),
                total_pw_cp_elapsed_sec=0.1,
                max_time_per_batch=None,
            )

    monkeypatch.setattr(module, "PwCpConstructor", FakeConstructor)

    ctrl.pw_cp(solver_thread_cnt=1, batch_size=7, batch_size_ratio=0.10)

    assert seen_kwargs["batch_size"] == 7


def test_controller_pw_cp_forwards_profile_fixed_kwargs(monkeypatch):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = module.HybridFlowShopCpLnsController.__new__(
        module.HybridFlowShopCpLnsController
    )
    schedule = SimpleNamespace(makespan=11)
    ctrl.instance = SimpleNamespace(job_count=4, stage_count=2)
    ctrl.stage_2_job_2_p_dict = {"s1": {"j1": 1}}
    ctrl.timer = SimpleNamespace(elapsed_sec=1.25)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: schedule,
        register=lambda report, solution: True,
    )
    ctrl.get_file_path_for_subroutine = lambda suffix: Path("/tmp") / suffix.lstrip("_")
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "pw_cp"
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None
    )
    ctrl.set_cp_model_as_base_cp_model = lambda *args, **kwargs: None
    ctrl.draw_incumbent_gantt = lambda *args, **kwargs: None

    seen_kwargs = {}

    class FakeConstructor:
        def __init__(self, ctx):
            self.ctx = ctx

        def run(self, *args, **kwargs):
            seen_kwargs.update(kwargs)
            obj_store = ObjValueBoundStore[int]()
            return PwCpResult(
                schedule=schedule,
                sub_obj_store=obj_store,
                last_obj_value=11,
                subproblem_logs=(),
                total_pw_cp_elapsed_sec=0.1,
                max_time_per_batch=None,
            )

    monkeypatch.setattr(module, "PwCpConstructor", FakeConstructor)

    ctrl.pw_cp(
        solver_thread_cnt=3,
        left_profile_fixed_batch_count=1,
        right_profile_fixed_batch_count=2,
        enable_promotion_profile_fixed=True,
        profile_fix_by_machine=True,
        machine_precedence_stride=2,
    )

    assert seen_kwargs["left_profile_fixed_batch_count"] == 1
    assert seen_kwargs["right_profile_fixed_batch_count"] == 2
    assert seen_kwargs["enable_promotion_profile_fixed"] is True
    assert seen_kwargs["profile_fix_by_machine"] is True
    assert seen_kwargs["machine_precedence_stride"] == 2


def test_controller_pw_cp_forwards_unfixed_time_limit_multiplier(monkeypatch):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = module.HybridFlowShopCpLnsController.__new__(
        module.HybridFlowShopCpLnsController
    )
    schedule = SimpleNamespace(makespan=11)
    ctrl.instance = SimpleNamespace(job_count=4, stage_count=2)
    ctrl.stage_2_job_2_p_dict = {"s1": {"j1": 1}}
    ctrl.timer = SimpleNamespace(elapsed_sec=1.25)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: schedule,
        register=lambda report, solution: True,
    )
    ctrl.get_file_path_for_subroutine = lambda suffix: Path("/tmp") / suffix.lstrip("_")
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "pw_cp"
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None
    )
    ctrl.set_cp_model_as_base_cp_model = lambda *args, **kwargs: None
    ctrl.draw_incumbent_gantt = lambda *args, **kwargs: None

    seen_kwargs = {}

    class FakeConstructor:
        def __init__(self, ctx):
            self.ctx = ctx

        def run(self, *args, **kwargs):
            seen_kwargs.update(kwargs)
            obj_store = ObjValueBoundStore[int]()
            return PwCpResult(
                schedule=schedule,
                sub_obj_store=obj_store,
                last_obj_value=11,
                subproblem_logs=(),
                total_pw_cp_elapsed_sec=0.1,
                max_time_per_batch=None,
            )

    monkeypatch.setattr(module, "PwCpConstructor", FakeConstructor)

    ctrl.pw_cp(
        solver_thread_cnt=3,
        cp_tl_c_multiplier=0.1,
        max_time_per_batch=9.0,
        unfixed_op_time_limit_multiplier=0.005,
    )

    assert seen_kwargs["unfixed_op_time_limit_multiplier"] == 0.005
    assert seen_kwargs["max_time_per_batch"] == 9.0


def test_controller_pw_cp_rejects_non_positive_unfixed_time_limit_multiplier():
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = module.HybridFlowShopCpLnsController.__new__(
        module.HybridFlowShopCpLnsController
    )
    schedule = SimpleNamespace(makespan=11)
    ctrl.instance = SimpleNamespace(job_count=4, stage_count=2)
    ctrl.stage_2_job_2_p_dict = {"s1": {"j1": 1}}
    ctrl.timer = SimpleNamespace(elapsed_sec=1.25)
    ctrl.solution_manager = SimpleNamespace(
        get_incumbent=lambda: schedule,
        register=lambda report, solution: True,
    )
    ctrl.get_file_path_for_subroutine = lambda suffix: Path("/tmp") / suffix.lstrip("_")
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "pw_cp"
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None
    )
    ctrl.set_cp_model_as_base_cp_model = lambda *args, **kwargs: None
    ctrl.draw_incumbent_gantt = lambda *args, **kwargs: None

    with pytest.raises(
        ValueError, match="unfixed_op_time_limit_multiplier must be > 0"
    ):
        ctrl.pw_cp(solver_thread_cnt=1, unfixed_op_time_limit_multiplier=0.0)


def test_run_uses_unfixed_count_scaled_time_limit(monkeypatch, tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_multi_stage_schedule()
    seen_limits = []

    def fake_solve_subproblem(**kwargs):
        seen_limits.append(kwargs["max_time_per_batch"])
        return None

    monkeypatch.setattr(ctor, "_solve_subproblem", fake_solve_subproblem)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
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
        batch_size=1,
        max_time_per_batch=99.0,
        unfixed_op_time_limit_multiplier=0.5,
    )

    assert seen_limits == [1.0, 1.0, 1.0, 1.0]


def test_run_unfixed_count_scaled_time_limit_overrides_scalar_limits(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    incumbent = _make_multi_stage_schedule()
    seen_ctx_limits = []

    def fake_get_remaining_time_limit(subroutine_time_limit):
        seen_ctx_limits.append(subroutine_time_limit)
        return subroutine_time_limit if subroutine_time_limit is not None else 1.0

    monkeypatch.setattr(ctx, "get_remaining_time_limit", fake_get_remaining_time_limit)
    monkeypatch.setattr(ctor, "_solve_subproblem", lambda **kwargs: None)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
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
        max_time_per_batch=99.0,
        unfixed_op_time_limit_multiplier=0.25,
    )

    assert seen_ctx_limits == [1.0, 1.0]


def test_run_rejects_non_positive_unfixed_time_limit_multiplier(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)

    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )

    with pytest.raises(
        ValueError, match="unfixed_op_time_limit_multiplier must be > 0"
    ):
        ctor.run(
            _make_schedule(),
            instance,
            {"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
            batch_size=1,
            unfixed_op_time_limit_multiplier=0.0,
        )


def test_draw_candidate_retiming_gantts_includes_reassignment_step(tmp_path):
    """Test that _draw_candidate_retiming_gantts draws 4 Gantt charts with correct naming."""
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )

    # Create four schedule states
    before_retiming = _make_schedule()

    after_successor_reassign = _make_schedule().deepcopy()
    after_predecessor_reassign = _make_schedule().deepcopy()

    after_semi_active = _make_schedule().deepcopy()
    after_semi_active.make_semi_active({"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}})

    solved_slack_intervals = {"s1": [("m1", 2, 5)]}

    ctor._draw_candidate_retiming_gantts(
        spec=spec,
        before_retiming=before_retiming,
        after_successor_reassign=after_successor_reassign,
        after_predecessor_reassign=after_predecessor_reassign,
        after_semi_active=after_semi_active,
        solved_slack_intervals=solved_slack_intervals,
    )

    # Verify 4 files created with correct naming
    # Note: get_file_path_for_subroutine strips leading "_", so check without it
    expected_files = [
        "batch_001_candidate_01_before_retiming.png",
        "batch_001_candidate_02_after_successor_reassign.png",
        "batch_001_candidate_03_after_predecessor_reassign.png",
        "batch_001_candidate_04_after_semi_active.png",
    ]
    for filename in expected_files:
        file_path = tmp_path / filename
        assert file_path.exists(), f"Expected file {filename} not found"


def test_draw_candidate_retiming_gantts_skips_missing_optional_steps(tmp_path):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )
    after_semi_active = _make_schedule().deepcopy()
    after_semi_active.make_semi_active({"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}})

    ctor._draw_candidate_retiming_gantts(
        spec=spec,
        before_retiming=_make_schedule(),
        after_successor_reassign=None,
        after_predecessor_reassign=None,
        after_semi_active=after_semi_active,
        solved_slack_intervals={},
    )

    assert (tmp_path / "batch_001_candidate_01_before_retiming.png").exists()
    assert (tmp_path / "batch_001_candidate_04_after_semi_active.png").exists()
    assert not (tmp_path / "batch_001_candidate_02_after_successor_reassign.png").exists()
    assert not (
        tmp_path / "batch_001_candidate_03_after_predecessor_reassign.png"
    ).exists()


def test_solve_subproblem_draws_available_gantts_when_successor_reassignment_fails(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(status="FEASIBLE", obj_value=1.0),
    )
    monkeypatch.setattr(ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy())
    monkeypatch.setattr(
        ctor,
        "_apply_successor_machine_reassignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("synthetic successor failure")
        ),
    )

    with pytest.raises(ValueError, match="synthetic successor failure"):
        ctor._solve_subproblem(
            spec=spec,
            incumbent=incumbent,
            right_justified_schedule=incumbent.deepcopy(),
            instance=instance,
            stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
            max_time_per_batch=1.0,
            solver_thread_cnt=1,
            profile_fix_by_machine=False,
            machine_precedence_stride=1,
            use_lns_only=False,
            tighten_ranges=False,
            link_job_completion=False,
            debug_export=False,
        )

    for filename in [
        "batch_001_candidate_01_before_retiming.png",
        "batch_001_candidate_02_after_successor_reassign.png",
        "batch_001_candidate_03_after_predecessor_reassign.png",
        "batch_001_candidate_04_after_semi_active.png",
    ]:
        assert not (tmp_path / filename).exists()


def test_solve_subproblem_draws_available_gantts_when_predecessor_reassignment_fails(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(status="FEASIBLE", obj_value=1.0),
    )
    monkeypatch.setattr(ctx, "create_schedule", lambda *args, **kwargs: incumbent.deepcopy())
    monkeypatch.setattr(
        ctor,
        "_apply_predecessor_machine_reassignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("synthetic predecessor failure")
        ),
    )

    with pytest.raises(ValueError, match="synthetic predecessor failure"):
        ctor._solve_subproblem(
            spec=spec,
            incumbent=incumbent,
            right_justified_schedule=incumbent.deepcopy(),
            instance=instance,
            stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
            max_time_per_batch=1.0,
            solver_thread_cnt=1,
            profile_fix_by_machine=False,
            machine_precedence_stride=1,
            use_lns_only=False,
            tighten_ranges=False,
            link_job_completion=False,
            debug_export=False,
        )

    for filename in [
        "batch_001_candidate_01_before_retiming.png",
        "batch_001_candidate_02_after_successor_reassign.png",
        "batch_001_candidate_03_after_predecessor_reassign.png",
        "batch_001_candidate_04_after_semi_active.png",
    ]:
        assert not (tmp_path / filename).exists()


def test_solve_subproblem_draws_available_gantts_when_make_semi_active_fails(
    monkeypatch, tmp_path
):
    ctx = FakePwCpContext(tmp_path)
    ctor = PwCpConstructor(ctx)
    ctor._st = SimpleNamespace(subproblem_logs=[])
    incumbent = _make_schedule()
    instance = create_hfs_instance(
        "tiny",
        ["j1", "j2", "j3", "j4"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"j1": {"s1": 2}, "j2": {"s1": 3}, "j3": {"s1": 2}, "j4": {"s1": 2}},
    )
    partition = OperationPartition(
        left_time_fixed=(),
        left_profile_fixed=(),
        unfixed=(("j1", "s1", "m1"), ("j2", "s1", "m2")),
        right_profile_fixed=(),
        right_time_fixed=(("j3", "s1", "m1"), ("j4", "s1", "m2")),
        right_boundary_profile={"s1": [5, 5]},
    )
    spec = PwCpSubproblemSpec(
        batch_idx=0,
        subproblem_idx=1,
        partition=partition,
        right_boundary_profile={"s1": [5, 5]},
        is_last_batch=False,
    )

    monkeypatch.setattr(
        ctx,
        "solve_cp_model_2",
        lambda *args, **kwargs: _make_solver_report(status="FEASIBLE", obj_value=1.0),
    )
    candidate_schedule = incumbent.deepcopy()

    def fake_create_schedule(*args, **kwargs):
        def fail_make_semi_active(*_args, **_kwargs):
            raise ValueError("synthetic semi-active failure")

        candidate_schedule.make_semi_active = fail_make_semi_active
        return candidate_schedule

    monkeypatch.setattr(ctx, "create_schedule", fake_create_schedule)

    with pytest.raises(ValueError, match="synthetic semi-active failure"):
        ctor._solve_subproblem(
            spec=spec,
            incumbent=incumbent,
            right_justified_schedule=incumbent.deepcopy(),
            instance=instance,
            stage_2_job_2_p_dict={"s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 2}},
            max_time_per_batch=1.0,
            solver_thread_cnt=1,
            profile_fix_by_machine=False,
            machine_precedence_stride=1,
            use_lns_only=False,
            tighten_ranges=False,
            link_job_completion=False,
            debug_export=False,
        )

    for filename in [
        "batch_001_candidate_01_before_retiming.png",
        "batch_001_candidate_02_after_successor_reassign.png",
        "batch_001_candidate_03_after_predecessor_reassign.png",
        "batch_001_candidate_04_after_semi_active.png",
    ]:
        assert not (tmp_path / filename).exists()
