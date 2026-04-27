from types import SimpleNamespace

from hybridflowshop.controller import neh_cp as neh_module
from hybridflowshop.controller.neh_cp import NehCpConstructor


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def deepcopy(self, job_subsequence=None):
        return _FakeSchedule(self.makespan)

    def make_semi_active(self, *_args, **_kwargs) -> None:
        return None

    def get_jik_2_start_time_map(self):
        return {}


class _FakeContext:
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
