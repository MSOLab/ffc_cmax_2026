from types import SimpleNamespace

from mbls.cpsat import CpsatSolverReport, CpsatStatus

from hybridflowshop.controller import neh_cp as neh_module
from hybridflowshop.controller.neh_cp import (
    NehCpConstructor,
    NehCpRunState,
    _format_minimization_gap,
    _format_solver_value,
    _log_cp_subproblem_report,
)


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def deepcopy(self, job_subsequence=None):
        return _FakeSchedule(self.makespan)

    def make_semi_active(self, *_args, **_kwargs) -> None:
        return None

    def get_jik_2_start_time_map(self):
        return {}

    def get_jik_2_end_time_map(self):
        return {}


class _FakeContext:
    def get_remaining_time_limit(self, subroutine_time_limit):
        return subroutine_time_limit

    def get_remaining_sec_before_final_reserve(self):
        return 9999.0

    def final_time_reserve_is_reached(self):
        return False

    def create_empty_schedule_from_ins(self):
        return _FakeSchedule(0)

    def check_feasibility(self, _start_time_map):
        return 0.0


def test_neh_cp_returns_best_intermediate_full_schedule(monkeypatch) -> None:
    class FakeMixedDispatcher:
        call_count = 0

        def __init__(self, _instance) -> None:
            return None

        def get_best_mixed_schedule_by_sequence(self, *_args, **_kwargs):
            FakeMixedDispatcher.call_count += 1
            return _FakeSchedule({1: 80, 2: 90, 3: 120}[FakeMixedDispatcher.call_count])

    monkeypatch.setattr(neh_module, "MixedDispatcher", FakeMixedDispatcher)
    monkeypatch.setattr(
        neh_module,
        "get_midpoint_sequence",
        lambda _schedule: ["A", "B", "C", "D"],
    )

    constructor = NehCpConstructor(_FakeContext())
    monkeypatch.setattr(
        constructor,
        "_solve_cp_model",
        lambda partial_sol, *_args, **_kwargs: (SimpleNamespace(), partial_sol),
    )

    result = constructor.run(
        _FakeSchedule(100),
        SimpleNamespace(job_count=4, stage_count=2, stage_id_list=["S1", "S2"]),
        job_2_stage_2_p_dict={},
        stage_2_job_2_p_dict={},
        added_batch_size=2,
        max_time_per_add=0.1,
    )

    assert result.schedule.makespan == 90
    assert result.last_obj_value == 90


def test_neh_cp_can_split_by_requested_batch_count(monkeypatch) -> None:
    primary_job_sublists = []

    class FakeMixedDispatcher:
        call_count = 0

        def __init__(self, _instance) -> None:
            return None

        def get_best_mixed_schedule_by_sequence(self, job_sublist, *_args, **_kwargs):
            FakeMixedDispatcher.call_count += 1
            if FakeMixedDispatcher.call_count % 2 == 1:
                primary_job_sublists.append(list(job_sublist))
            return _FakeSchedule(100 + FakeMixedDispatcher.call_count)

    monkeypatch.setattr(neh_module, "MixedDispatcher", FakeMixedDispatcher)
    monkeypatch.setattr(
        neh_module,
        "get_midpoint_sequence",
        lambda _schedule: ["A", "B", "C", "D", "E", "F", "G"],
    )

    constructor = NehCpConstructor(_FakeContext())
    monkeypatch.setattr(
        constructor,
        "_solve_cp_model",
        lambda partial_sol, *_args, **_kwargs: (SimpleNamespace(), partial_sol),
    )

    constructor.run(
        _FakeSchedule(200),
        SimpleNamespace(job_count=7, stage_count=2, stage_id_list=["S1", "S2"]),
        job_2_stage_2_p_dict={},
        stage_2_job_2_p_dict={},
        added_batch_size=2,
        added_batch_count=3,
        max_time_per_add=0.1,
    )

    assert primary_job_sublists == [["A", "B", "C"], ["D", "E"], ["F", "G"]]


def test_neh_cp_can_clamp_batch_count_from_batch_size(monkeypatch) -> None:
    primary_job_sublists = []

    class FakeMixedDispatcher:
        call_count = 0

        def __init__(self, _instance) -> None:
            return None

        def get_best_mixed_schedule_by_sequence(self, job_sublist, *_args, **_kwargs):
            FakeMixedDispatcher.call_count += 1
            if FakeMixedDispatcher.call_count % 2 == 1:
                primary_job_sublists.append(list(job_sublist))
            return _FakeSchedule(100 + FakeMixedDispatcher.call_count)

    monkeypatch.setattr(neh_module, "MixedDispatcher", FakeMixedDispatcher)
    monkeypatch.setattr(
        neh_module,
        "get_midpoint_sequence",
        lambda _schedule: ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )

    constructor = NehCpConstructor(_FakeContext())
    monkeypatch.setattr(
        constructor,
        "_solve_cp_model",
        lambda partial_sol, *_args, **_kwargs: (SimpleNamespace(), partial_sol),
    )

    constructor.run(
        _FakeSchedule(200),
        SimpleNamespace(job_count=10, stage_count=2, stage_id_list=["S1", "S2"]),
        job_2_stage_2_p_dict={},
        stage_2_job_2_p_dict={},
        added_batch_size=2,
        max_added_batch_count=3,
        max_time_per_add=0.1,
    )

    assert primary_job_sublists == [
        ["A", "B", "C", "D"],
        ["E", "F", "G"],
        ["H", "I", "J"],
    ]


def test_neh_cp_subproblem_report_logs_ub_lb_gap(caplog) -> None:
    report = CpsatSolverReport(
        elapsed_time=1.25,
        obj_value=120.0,
        obj_bound=100.0,
        status=CpsatStatus.FEASIBLE,
        obj_value_records=[(0.5, 130.0), (1.25, 120.0)],
        obj_bound_records=[(0.2, 90.0), (1.25, 100.0)],
    )

    with caplog.at_level("INFO"):
        _log_cp_subproblem_report(
            prefix="NEH-CP batch 2/8",
            report=report,
            objective_name="primary_makespan",
        )

    assert "NEH-CP batch 2/8 primary_makespan CP" in caplog.text
    assert "ub=120" in caplog.text
    assert "lb=100" in caplog.text
    assert "gap=abs=20, rel=16.667%" in caplog.text
    assert "ub_updates=2 lb_updates=2" in caplog.text


def test_neh_cp_subproblem_report_formatters_handle_missing_values() -> None:
    assert _format_solver_value(None) == "NA"
    assert _format_minimization_gap(None, 10.0) == "NA"


def test_profile_fix_min_batch_idx_can_scale_by_batch_portion() -> None:
    assert (
        NehCpConstructor._resolve_profile_fix_min_batch_idx(
            profile_fix_min_batch_idx=3,
            profile_fix_min_batch_portion=0.25,
            profile_fix_max_batch_idx=6,
            total_batch_count=8,
        )
        == 3
    )
    assert (
        NehCpConstructor._resolve_profile_fix_min_batch_idx(
            profile_fix_min_batch_idx=3,
            profile_fix_min_batch_portion=0.25,
            profile_fix_max_batch_idx=6,
            total_batch_count=12,
        )
        == 4
    )
    assert (
        NehCpConstructor._resolve_profile_fix_min_batch_idx(
            profile_fix_min_batch_idx=3,
            profile_fix_min_batch_portion=0.25,
            profile_fix_max_batch_idx=6,
            total_batch_count=24,
        )
        == 6
    )


def test_neh_cp_profile_fixing_can_be_delayed_and_softened(monkeypatch) -> None:
    calls = []
    constructor = NehCpConstructor(_FakeContext())
    constructor._st = NehCpRunState(
        timer=SimpleNamespace(elapsed_sec=0.0),
        last_job_id_list=[],
        partial_sol=_FakeSchedule(10),
        current_job_id_list=[],
        full_sol=_FakeSchedule(10),
        job_2_inserted_batch_idx={"A": 1, "B": 2, "C": 3},
    )

    monkeypatch.setattr(
        neh_module.BaseModelBuilder,
        "apply_start_hints_from_start_time_map",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        neh_module.BaseModelBuilder,
        "apply_end_hints_from_end_time_map",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        neh_module.BaseModelBuilder,
        "add_stage_ops_precedence_constraints_after_dispatch_from_schedule",
        lambda *_args, **kwargs: calls.append(kwargs),
    )

    constructor._add_hints_and_additional_constraints(
        None,
        None,
        None,
        _FakeSchedule(20),
        profile_fix_min_batch_idx=3,
        batch_idx=2,
    )
    assert calls == []

    constructor._add_hints_and_additional_constraints(
        None,
        None,
        None,
        _FakeSchedule(20),
        profile_fix_by_machine=True,
        profile_fix_by_machine_from_batch_idx=5,
        stage_precedence_min_processing_time_diff_ratio=0.15,
        batch_idx=3,
    )
    assert calls[-1]["profile_fix_by_machine"] is False
    assert calls[-1]["stage_precedence_min_processing_time_diff_ratio"] == 0.15

    constructor._add_hints_and_additional_constraints(
        None,
        None,
        None,
        _FakeSchedule(20),
        profile_fix_min_job_age_batches=2,
        batch_idx=3,
    )
    assert calls[-1]["profile_fix_job_ids"] == {"A"}

    constructor._add_hints_and_additional_constraints(
        None,
        None,
        None,
        _FakeSchedule(20),
        profile_fix_by_machine=True,
        profile_fix_by_machine_from_batch_idx=5,
        batch_idx=5,
    )
    assert calls[-1]["profile_fix_by_machine"] is True
