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
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=7)

    mc_id, t = sched.get_machine_and_earliest_available_time_by_start_idle_idx(
        "s1", duration=5
    )
    assert mc_id == "m2"
    assert t == 0


def test_get_job_end_time_and_default_behavior():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=2, end_time=5)
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
    sched.add_operation_2_stage("s1", "j1", duration=5)

    # Stage 2: machine is idle at 0, but precedence forces start at 5.
    sched.add_operation_2_stage("s2", "j1", duration=3)

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
    sched.add_ops_times_2_mc("s2", "m2", "j1", start_time=0, end_time=10)

    # Stage 1 completes j2 at 3.
    sched.add_operation_2_stage("s1", "j2", duration=3)

    # Stage 2 for j2 should start at max(machine_ready=10, prev_stage_end=3) => 10.
    sched.add_operation_2_stage("s2", "j2", duration=2)

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

    sched.add_operation_2_stage("s1", "j1", duration=1)
    with pytest.raises(ValueError):
        sched.add_operation_2_stage("s1", "j1", duration=1)


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
    sched.add_ops_times_2_mc("s1", "m0", "j1", start_time=0, end_time=100)

    # Last stage: m1 ends at 15, m2 ends at 20 => makespan = 20.
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=0, end_time=15)
    sched.add_ops_times_2_mc("s2", "m2", "j3", start_time=0, end_time=20)
    assert sched.makespan == 20


def test_get_start_time_map_and_end_time_map_basic():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1", "m2"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=3)
    sched.add_ops_times_2_mc("s1", "m2", "j2", start_time=1, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=3, end_time=7)
    sched.add_ops_times_2_mc("s2", "m2", "j2", start_time=5, end_time=8)

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

    sched.add_operation_2_stage("s1", "j1", duration=5)
    sched.add_operation_2_stage("s2", "j1", duration=2)

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

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)

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

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=3)
    sched.add_ops_times_2_mc("s1", "m1", "j3", start_time=3, end_time=5)

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

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=5)

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


def test_inserts_into_idle_gap_on_machine():
    """Test that operations are inserted into idle gaps on machines."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["i0"],
        machines_per_stage={"i0": ["i0_0"]},
    )

    # Create an idle gap [5, 10) by scheduling j1 then j2 with a later release.
    sched.add_operation_2_mc("i0", "i0_0", "j1", duration=5, release_t=0)  # [0, 5)
    sched.add_operation_2_mc("i0", "i0_0", "j2", duration=10, release_t=10)  # [10, 20)

    # This should be inserted into the gap, not appended to the tail.
    sched.add_operation_2_mc("i0", "i0_0", "j3", duration=3, release_t=0)

    start_map = sched.get_start_time_map()
    end_map = sched.get_end_time_map()

    assert start_map[("j1", "i0", "i0_0")] == 0
    assert end_map[("j1", "i0", "i0_0")] == 5

    assert start_map[("j2", "i0", "i0_0")] == 10
    assert end_map[("j2", "i0", "i0_0")] == 20

    assert start_map[("j3", "i0", "i0_0")] == 5
    assert end_map[("j3", "i0", "i0_0")] == 8

    # The schedule should remain correctly ordered and makespan unchanged.
    assert sched.makespan == 20


def test_stage_dispatch_prefers_gap_machine():
    """Test that stage dispatch prefers machines with gaps over fully busy machines."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "b", "c", "x"],
        stages=["i0"],
        machines_per_stage={"i0": ["i0_0", "i0_1"]},
    )

    # Machine i0_0: fully busy [0, 100)
    sched.add_operation_2_mc("i0", "i0_0", "x", duration=100, release_t=0)

    # Machine i0_1: busy [0, 20) and [30, 100), leaving a gap [20, 30)
    sched.add_operation_2_mc("i0", "i0_1", "a", duration=20, release_t=0)
    sched.add_operation_2_mc("i0", "i0_1", "b", duration=70, release_t=30)

    # Dispatching to stage should select i0_1 and start at 20 (in the gap)
    sched.add_operation_2_stage("i0", "c", duration=5, release_t=0)

    start_map = sched.get_start_time_map()
    assert start_map[("c", "i0", "i0_1")] == 20


def test_dispatch_stage_by_jobs_uses_precedence_priority():
    """Test that dispatch_stage_by_jobs uses previous stage end times as priority."""
    # Two stages, one machine each at stage i1.
    # We pre-fill i1 with two fixed ops, creating a critical early gap [0, 8).
    # Job A (ready=1, duration=7) can fit exactly into [1, 8).
    # Job B (ready=2, duration=5) also fits but is more flexible.
    # If B is scheduled first, it will take [2, 7) and block A from using the early gap.
    # With the priority queue (A before B), A should claim the early gap.
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "b", "x1", "x2"],
        stages=["i0", "i1"],
        machines_per_stage={"i0": ["m0"], "i1": ["m1"]},
    )

    # Stage i0 sets readiness times: a ends at 1, b ends at 2.
    sched.add_operation_2_mc("i0", "m0", "a", duration=1, release_t=0)  # [0,1)
    sched.add_operation_2_mc("i0", "m0", "b", duration=1, release_t=1)  # [1,2)

    # Pre-fill stage i1 machine timeline to create gaps: [0,8), [10,17)
    sched.add_ops_times_2_mc("i1", "m1", "x1", start_time=8, end_time=10)
    sched.add_ops_times_2_mc("i1", "m1", "x2", start_time=17, end_time=20)

    # Provide input order that is the reverse of the intended priority.
    sched.dispatch_stage_by_jobs(
        "i1",
        job_id_seq=["b", "a"],
        job_2_duration={"a": 7, "b": 5},
    )

    start_map = sched.get_start_time_map()
    # 'a' should be scheduled first at its earliest feasible time (1), not pushed to 10.
    assert start_map[("a", "i1", "m1")] == 1
    assert start_map[("b", "i1", "m1")] == 10


@pytest.mark.parametrize("duration", [0, -1])
def test_rejects_non_positive_duration(duration: int):
    """Test that non-positive durations are rejected."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j"],
        stages=["i0"],
        machines_per_stage={"i0": ["i0_0"]},
    )

    with pytest.raises(ValueError, match="Duration must be greater than 0"):
        sched.add_operation_2_mc("i0", "i0_0", "j", duration=duration, release_t=0)
