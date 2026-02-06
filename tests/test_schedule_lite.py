import pytest

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


def _get_priv(obj, attr: str):
    # Access name-mangled private attributes for test assertions.
    return getattr(obj, f"_HybridFlowshopLiteSchedule{attr}")


def test_machine_latest_end_time_empty_is_zero():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    assert sched.get_machine_latest_end_time("s1", "m1") == 0


def test_selects_machine_with_min_latest_end_time():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )

    # Make m1 busy until 7, m2 remains idle.
    sched.append_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=7)

    mc_id, t = sched.get_machine_and_earliest_available_time_by_start_idle_idx("s1")
    assert mc_id == "m2"
    assert t == 0


def test_get_job_end_time_and_default_behavior():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    sched.append_ops_times_2_mc("s1", "m1", "j1", start_time=2, end_time=5)
    assert sched.get_job_end_time("s1", "j1") == 5

    assert sched.get_job_end_time("s1", "missing", default_if_missing=0) == 0
    with pytest.raises(ValueError):
        sched.get_job_end_time("s1", "missing")


def test_prev_stage_end_time_first_stage_is_zero():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    assert sched.get_prev_stage_end_time("s1", "j1") == 0


def test_append_operation_2_stage_respects_precedence_constraint():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )

    # Stage 1: j1 ends at 5.
    sched.append_operation_2_stage("s1", "j1", duration=5)

    # Stage 2: machine is idle at 0, but precedence forces start at 5.
    sched.append_operation_2_stage("s2", "j1", duration=3)

    start_s2, end_s2 = _get_priv(sched, "__stage_2_mc_2_job_tuple_seq")["s2"]["m2"][0][
        0:2
    ]

    assert start_s2 == 5
    assert end_s2 == 8


def test_append_operation_2_stage_uses_machine_ready_time_if_larger_than_precedence():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )

    # Pre-load stage 2 machine with j1 so it's busy until 10.
    sched.append_ops_times_2_mc("s2", "m2", "j1", start_time=0, end_time=10)

    # Stage 1 completes j2 at 3.
    sched.append_operation_2_stage("s1", "j2", duration=3)

    # Stage 2 for j2 should start at max(machine_ready=10, prev_stage_end=3) => 10.
    sched.append_operation_2_stage("s2", "j2", duration=2)

    start_map = sched.get_start_time_map()
    end_map = sched.get_end_time_map()

    assert start_map[("j1", "s2", "m2")] == 0
    assert end_map[("j1", "s2", "m2")] == 10
    assert start_map[("j2", "s2", "m2")] == 10
    assert end_map[("j2", "s2", "m2")] == 12


def test_duplicate_job_in_same_stage_raises_value_error():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    sched.append_operation_2_stage("s1", "j1", duration=1)
    with pytest.raises(ValueError):
        sched.append_operation_2_stage("s1", "j1", duration=1)


def test_makespan_is_zero_when_no_operations_scheduled():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    assert sched.makespan == 0


def test_makespan_is_max_latest_end_time_of_last_stage():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m0"], "s2": ["m1", "m2"]},
    )

    # Earlier stage should not affect makespan computation directly.
    sched.append_ops_times_2_mc("s1", "m0", "j1", start_time=0, end_time=100)

    # Last stage: m1 ends at 15, m2 ends at 20 => makespan = 20.
    sched.append_ops_times_2_mc("s2", "m1", "j2", start_time=0, end_time=15)
    sched.append_ops_times_2_mc("s2", "m2", "j3", start_time=0, end_time=20)
    assert sched.makespan == 20


def test_get_start_time_map_and_end_time_map_basic():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1", "m2"]},
    )

    sched.append_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=3)
    sched.append_ops_times_2_mc("s1", "m2", "j2", start_time=1, end_time=5)
    sched.append_ops_times_2_mc("s2", "m1", "j1", start_time=3, end_time=7)
    sched.append_ops_times_2_mc("s2", "m2", "j2", start_time=5, end_time=8)

    assert sched.get_start_time_map() == {
        ("j1", "s1", "m1"): 0,
        ("j2", "s1", "m2"): 1,
        ("j1", "s2", "m1"): 3,
        ("j2", "s2", "m2"): 5,
    }
    assert sched.get_end_time_map() == {
        ("j1", "s1", "m1"): 3,
        ("j2", "s1", "m2"): 5,
        ("j1", "s2", "m1"): 7,
        ("j2", "s2", "m2"): 8,
    }


def test_time_maps_reflect_append_operation_2_stage_precedence():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )

    sched.append_operation_2_stage("s1", "j1", duration=5)
    sched.append_operation_2_stage("s2", "j1", duration=2)

    start_map = sched.get_start_time_map()
    end_map = sched.get_end_time_map()

    assert start_map[("j1", "s1", "m1")] == 0
    assert end_map[("j1", "s1", "m1")] == 5
    assert start_map[("j1", "s2", "m2")] == 5
    assert end_map[("j1", "s2", "m2")] == 7


def test_remove_operations_removes_specified_ops_and_updates_cache():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )

    sched.append_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.append_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.append_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.append_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)

    sched.remove_operations({("j1", "s2", "m1")})

    start_map = sched.get_start_time_map()
    end_map = sched.get_end_time_map()

    assert ("j1", "s2", "m1") not in start_map
    assert ("j1", "s2", "m1") not in end_map

    # Other operations must remain intact.
    assert start_map[("j1", "s1", "m1")] == 0
    assert end_map[("j1", "s1", "m1")] == 2
    assert start_map[("j2", "s2", "m1")] == 5
    assert end_map[("j2", "s2", "m1")] == 7

    with pytest.raises(ValueError):
        sched.get_job_end_time("s2", "j1")
    assert sched.get_job_end_time("s2", "j2") == 7

    # makespan should be computed on remaining ops in the last stage.
    assert sched.makespan == 7


def test_remove_operations_multiple_ops_same_machine():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    sched.append_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.append_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=3)
    sched.append_ops_times_2_mc("s1", "m1", "j3", start_time=3, end_time=5)

    sched.remove_operations({("j2", "s1", "m1"), ("j3", "s1", "m1")})

    assert sched.get_start_time_map() == {("j1", "s1", "m1"): 0}
    assert sched.get_end_time_map() == {("j1", "s1", "m1"): 2}
    assert sched.get_machine_latest_end_time("s1", "m1") == 2
    assert sched.makespan == 2

    with pytest.raises(ValueError):
        sched.get_job_end_time("s1", "j2")
    with pytest.raises(ValueError):
        sched.get_job_end_time("s1", "j3")


def test_deepcopy_copies_and_filters_cache():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    sched.append_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.append_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=5)

    copied = sched.deepcopy(job_subsequence={"j1"})

    assert copied.get_job_end_time("s1", "j1") == 2
    with pytest.raises(ValueError):
        copied.get_job_end_time("s1", "j2")

    # Internal representation is (start, end, job)
    job_tuple_seq = _get_priv(copied, "__stage_2_mc_2_job_tuple_seq")["s1"]["m1"]
    assert job_tuple_seq == [(0, 2, "j1")]

    # Public maps should reflect the filtered schedule
    assert copied.get_start_time_map() == {("j1", "s1", "m1"): 0}
    assert copied.get_end_time_map() == {("j1", "s1", "m1"): 2}
