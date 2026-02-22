import pytest

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, validate_schedule


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

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()

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

    assert sched.get_jik_2_start_time_map() == {
        ("j1", "s1", "m1"): 0,
        ("j2", "s1", "m2"): 1,
        ("j1", "s2", "m1"): 3,
        ("j2", "s2", "m2"): 5,
    }
    assert sched.get_jik_2_end_time_map() == {
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

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()

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

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()

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

    assert sched.get_jik_2_start_time_map() == {("j1", "s1", "m1"): 0}
    assert sched.get_jik_2_end_time_map() == {("j1", "s1", "m1"): 2}
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
    assert copied.get_jik_2_start_time_map() == {("j1", "s1", "m1"): 0}
    assert copied.get_jik_2_end_time_map() == {("j1", "s1", "m1"): 2}


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

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()

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

    start_map = sched.get_jik_2_start_time_map()
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

    start_map = sched.get_jik_2_start_time_map()
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


# ============================================================================
# Tests for make_semi_active()
# ============================================================================


def test_make_semi_active_with_slack():
    """Test make_semi_active removes slack by left-shifting operations."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_job_2_duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    # Create a non semi-active schedule with slack:
    # j1: s1[5-10] -> s2[10-15]  (could start s1 at 0)
    # j2: s1[10-15] -> s2[15-20] (could start s1 at 5)
    # After make_semi_active:
    # j1: s1[0-5] -> s2[5-10]
    # j2: s1[5-10] -> s2[10-15]
    sched.add_ops_times_2_mc(
        "s1", "m1", "j1", start_time=5, end_time=5 + stage_2_job_2_duration["s1"]["j1"]
    )
    sched.add_ops_times_2_mc(
        "s1",
        "m1",
        "j2",
        start_time=10,
        end_time=10 + stage_2_job_2_duration["s1"]["j2"],
    )
    sched.add_ops_times_2_mc(
        "s2",
        "m1",
        "j1",
        start_time=10,
        end_time=10 + stage_2_job_2_duration["s2"]["j1"],
    )
    sched.add_ops_times_2_mc(
        "s2",
        "m1",
        "j2",
        start_time=15,
        end_time=15 + stage_2_job_2_duration["s2"]["j2"],
    )

    # Before make_semi_active, schedule has slack
    start_map_before = sched.get_jik_2_start_time_map()
    assert start_map_before[("j1", "s1", "m1")] == 5  # has slack
    assert start_map_before[("j2", "s1", "m1")] == 10

    # Make it semi-active
    sched.make_semi_active(stage_2_job_2_duration)

    # After make_semi_active, operations should be left-shifted
    start_map = sched.get_jik_2_start_time_map()
    # j1@s1 should start at 0 (no job precedence, machine available)
    assert start_map[("j1", "s1", "m1")] == 0
    # j2@s1 should start at 5 (after j1@s1 completes)
    assert start_map[("j2", "s1", "m1")] == 5
    # j1@s2 should start at 5 (after j1@s1 completes at 5)
    assert start_map[("j1", "s2", "m1")] == 5
    # j2@s2 should start at 10 (after j2@s1 completes at 10, and j1@s2 completes at 10)
    assert start_map[("j2", "s2", "m1")] == 10

    # Makespan should be reduced from 20 to 15
    assert sched.makespan == 15


def test_make_semi_active_respects_precedence():
    """Test make_semi_active respects job precedence constraints."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    stage_2_job_2_duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 10},
        "s2": {"j1": 10},
        "s3": {"j1": 10},
    }

    # Create schedule with slack but precedence constraints
    # j1: s1[10-20] -> s2[25-35] -> s3[40-50]
    # (all could be left-shifted)
    sched.add_ops_times_2_mc(
        "s1",
        "m1",
        "j1",
        start_time=10,
        end_time=10 + stage_2_job_2_duration["s1"]["j1"],
    )
    sched.add_ops_times_2_mc(
        "s2",
        "m1",
        "j1",
        start_time=25,
        end_time=25 + stage_2_job_2_duration["s2"]["j1"],
    )
    sched.add_ops_times_2_mc(
        "s3",
        "m1",
        "j1",
        start_time=40,
        end_time=40 + stage_2_job_2_duration["s3"]["j1"],
    )

    sched.make_semi_active(stage_2_job_2_duration)

    start_map = sched.get_jik_2_start_time_map()
    # With precedence: j1@s1 starts at 0 (duration 10), s2 starts at 10 (duration 10), s3 starts at 20 (duration 10)
    assert start_map[("j1", "s1", "m1")] == 0
    assert start_map[("j1", "s2", "m1")] == 10
    assert start_map[("j1", "s3", "m1")] == 20

    assert sched.makespan == 30


def test_make_semi_active_multi_machine():
    """Test make_semi_active with multiple machines per stage."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1"]},
    )
    stage_2_job_2_duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    # Create slack with multi-machine stage 1
    # j1 on m1: s1[5-10], j2 on m2: s1[10-15]
    # Both s2 operations can start at 10 (when first stage completes)
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=5, end_time=10)
    sched.add_ops_times_2_mc("s1", "m2", "j2", start_time=10, end_time=15)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=10, end_time=15)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=15, end_time=20)

    sched.make_semi_active(stage_2_job_2_duration)

    start_map = sched.get_jik_2_start_time_map()
    # j1@s1 should be left-shifted to start at 0 on m1
    assert start_map[("j1", "s1", "m1")] == 0
    # j2@s1 should be left-shifted to start at 0 on m2 (m2 is independent)
    assert start_map[("j2", "s1", "m2")] == 0
    # j1@s2 starts at 5 (after j1@s1 completes at 5)
    # j2@s2 starts at 10 (after j1@s2 completes at 10 on same machine)
    # Note: j2@s1 is now at 0-10 (was 10-15), so j2@s2 doesn't need to wait for j2@s1
    # It only needs to wait for j1@s2 to complete on the same machine (m1)
    assert start_map[("j1", "s2", "m1")] == 5
    assert start_map[("j2", "s2", "m1")] == 10

    assert sched.makespan == 15


def test_make_semi_active_already_semi_active():
    """Test make_semi_active on already semi-active schedule (no change)."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_job_2_duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    # Already semi-active: no slack
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=5, end_time=10)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=5, end_time=10)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=10, end_time=15)

    makespan_before = sched.makespan
    start_map_before = sched.get_jik_2_start_time_map()

    sched.make_semi_active(stage_2_job_2_duration)

    # Should be unchanged
    assert sched.makespan == makespan_before
    assert sched.get_jik_2_start_time_map() == start_map_before


def test_make_semi_active_empty_schedule():
    """Test make_semi_active on empty schedule."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    stage_2_job_2_duration = {"s1": {"j1": 5}}

    # Should not raise any error
    sched.make_semi_active(stage_2_job_2_duration)
    assert sched.makespan == 0


# ============================================================================
# Tests for make_semi_active() with dummy initial times
# ============================================================================


def test_make_semi_active_from_dummy_times():
    """Test make_semi_active starting from dummy (0,0) start/end times."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2", "J3"],
        stages=["S1", "S2"],
        machines_per_stage={"S1": ["M1"], "S2": ["M1", "M2"]},
    )

    # S1.M1: J1, J2, J3 (all with dummy start/end = 0)
    sched.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J2", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J3", start_time=0, end_time=0)
    # S2.M1: J1, J3  |  S2.M2: J2
    sched.add_ops_times_2_mc("S2", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J3", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M2", "J2", start_time=0, end_time=0)

    duration: dict[str, dict[str, int]] = {
        "S1": {"J1": 3, "J2": 4, "J3": 2},
        "S2": {"J1": 5, "J2": 3, "J3": 6},
    }

    sched.make_semi_active(duration)

    validate_schedule(sched, duration)

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()

    # S1.M1: J1[0,3), J2[3,7), J3[7,9)
    assert start_map[("J1", "S1", "M1")] == 0
    assert end_map[("J1", "S1", "M1")] == 3
    assert start_map[("J2", "S1", "M1")] == 3
    assert end_map[("J2", "S1", "M1")] == 7
    assert start_map[("J3", "S1", "M1")] == 7
    assert end_map[("J3", "S1", "M1")] == 9

    # S2.M1: J1[3,8), J3[9,15)
    assert start_map[("J1", "S2", "M1")] == 3
    assert end_map[("J1", "S2", "M1")] == 8
    assert start_map[("J3", "S2", "M1")] == 9
    assert end_map[("J3", "S2", "M1")] == 15

    # S2.M2: J2[7,10)
    assert start_map[("J2", "S2", "M2")] == 7
    assert end_map[("J2", "S2", "M2")] == 10

    assert sched.makespan == 15


# ============================================================================
# Tests for swap_two_operations_within_stage()
# ============================================================================


def _build_3job_2stage_schedule() -> tuple[
    HybridFlowshopLiteSchedule, dict[str, dict[str, int]]
]:
    """Build the standard 3-job, 2-stage fixture and make it semi-active."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2", "J3"],
        stages=["S1", "S2"],
        machines_per_stage={"S1": ["M1"], "S2": ["M1", "M2"]},
    )
    sched.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J2", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J3", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J3", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M2", "J2", start_time=0, end_time=0)

    duration: dict[str, dict[str, int]] = {
        "S1": {"J1": 3, "J2": 4, "J3": 2},
        "S2": {"J1": 5, "J2": 3, "J3": 6},
    }
    sched.make_semi_active(duration)
    return sched, duration


def test_swap_cross_machine_with_make_semi_active():
    """Swap J1 (on M1) and J2 (on M2) in S2 -- different machines."""
    sched, duration = _build_3job_2stage_schedule()

    sched.swap_two_operations_within_stage(
        "S2", "J1", "J2", duration, do_make_semi_active=True
    )

    validate_schedule(sched, duration)

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()

    # S1 unchanged: J1[0,3), J2[3,7), J3[7,9)
    assert start_map[("J1", "S1", "M1")] == 0
    assert start_map[("J2", "S1", "M1")] == 3
    assert start_map[("J3", "S1", "M1")] == 7

    # After swap: S2.M1 = [J2, J3], S2.M2 = [J1]
    # S2.M1: J2 start=max(prev=7, mc=0)=7, end=10; J3 start=max(prev=9,mc=10)=10, end=16
    # S2.M2: J1 start=max(prev=3, mc=0)=3, end=8
    assert start_map[("J2", "S2", "M1")] == 7
    assert end_map[("J2", "S2", "M1")] == 10
    assert start_map[("J3", "S2", "M1")] == 10
    assert end_map[("J3", "S2", "M1")] == 16
    assert start_map[("J1", "S2", "M2")] == 3
    assert end_map[("J1", "S2", "M2")] == 8

    assert sched.makespan == 16


def test_swap_same_machine_with_make_semi_active():
    """Swap J1 and J3 on the same machine (M1) in S2."""
    sched, duration = _build_3job_2stage_schedule()

    sched.swap_two_operations_within_stage(
        "S2", "J1", "J3", duration, do_make_semi_active=True
    )

    validate_schedule(sched, duration)

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()

    # S1 unchanged: J1[0,3), J2[3,7), J3[7,9)
    # After swap: S2.M1 order is [J3, J1], S2.M2 still [J2]
    # S2.M1: J3 start=max(prev=9, mc=0)=9, end=15; J1 start=max(prev=3, mc=15)=15, end=20
    # S2.M2: J2 start=max(prev=7, mc=0)=7, end=10
    assert start_map[("J3", "S2", "M1")] == 9
    assert end_map[("J3", "S2", "M1")] == 15
    assert start_map[("J1", "S2", "M1")] == 15
    assert end_map[("J1", "S2", "M1")] == 20
    assert start_map[("J2", "S2", "M2")] == 7
    assert end_map[("J2", "S2", "M2")] == 10

    assert sched.makespan == 20


def test_swap_without_make_semi_active_invalidates_end_time():
    """With do_make_semi_active=False, end-time cache is invalidated."""
    sched, duration = _build_3job_2stage_schedule()

    sched.swap_two_operations_within_stage(
        "S2", "J1", "J2", duration, do_make_semi_active=False
    )

    # The end-time entries for both jobs at S2 should be removed.
    with pytest.raises(ValueError):
        sched.get_job_end_time("S2", "J1")
    with pytest.raises(ValueError):
        sched.get_job_end_time("S2", "J2")

    # Other end-time entries should remain intact.
    assert sched.get_job_end_time("S1", "J1") == 3
    assert sched.get_job_end_time("S1", "J2") == 7
    assert sched.get_job_end_time("S2", "J3") == 15

    # After a manual make_semi_active, everything should be consistent.
    sched.make_semi_active(duration)
    validate_schedule(sched, duration)


def test_swap_invalid_stage_raises():
    sched, duration = _build_3job_2stage_schedule()
    with pytest.raises(ValueError, match="Invalid stage ID"):
        sched.swap_two_operations_within_stage("INVALID", "J1", "J2", duration)


def test_swap_same_job_raises():
    sched, duration = _build_3job_2stage_schedule()
    with pytest.raises(ValueError, match="Cannot swap a job with itself"):
        sched.swap_two_operations_within_stage("S2", "J1", "J1", duration)


def test_swap_missing_job_raises():
    sched, duration = _build_3job_2stage_schedule()
    # J2 is on S2.M2, but "MISSING" is not scheduled at all.
    with pytest.raises(ValueError, match="not found in stage"):
        sched.swap_two_operations_within_stage("S2", "J1", "MISSING", duration)


# ============================================================================
# Tests for make_semi_active(start_from_stage=...)
# ============================================================================


def _build_3job_3stage_schedule() -> tuple[
    HybridFlowshopLiteSchedule, dict[str, dict[str, int]]
]:
    """Build a 3-job, 3-stage fixture and make it semi-active.

    Layout:
        S1.M1: [J1, J2, J3]
        S2.M1: [J1, J3],  S2.M2: [J2]
        S3.M1: [J2, J3],  S3.M2: [J1]
    """
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2", "J3"],
        stages=["S1", "S2", "S3"],
        machines_per_stage={
            "S1": ["M1"],
            "S2": ["M1", "M2"],
            "S3": ["M1", "M2"],
        },
    )
    # S1
    sched.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J2", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J3", start_time=0, end_time=0)
    # S2
    sched.add_ops_times_2_mc("S2", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J3", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M2", "J2", start_time=0, end_time=0)
    # S3
    sched.add_ops_times_2_mc("S3", "M1", "J2", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S3", "M1", "J3", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S3", "M2", "J1", start_time=0, end_time=0)

    duration: dict[str, dict[str, int]] = {
        "S1": {"J1": 3, "J2": 4, "J3": 2},
        "S2": {"J1": 5, "J2": 3, "J3": 6},
        "S3": {"J1": 4, "J2": 2, "J3": 3},
    }
    sched.make_semi_active(duration)
    return sched, duration


def test_start_from_stage_none_equals_full_retiming():
    """start_from_stage=None should produce the same result as full retiming."""
    sched_full, duration = _build_3job_3stage_schedule()

    sched_partial, _ = _build_3job_3stage_schedule()
    sched_partial.make_semi_active(duration, start_from_stage=None)

    assert (
        sched_full.get_jik_2_start_time_map()
        == sched_partial.get_jik_2_start_time_map()
    )
    assert sched_full.get_jik_2_end_time_map() == sched_partial.get_jik_2_end_time_map()


def test_start_from_stage_leaves_earlier_stages_untouched():
    """Retiming from S2 must not modify S1 data."""
    sched, duration = _build_3job_3stage_schedule()

    # Record S1 state before partial retiming.
    s1_start_before = {
        k: v for k, v in sched.get_jik_2_start_time_map().items() if k[1] == "S1"
    }
    s1_end_before = {
        k: v for k, v in sched.get_jik_2_end_time_map().items() if k[1] == "S1"
    }

    sched.make_semi_active(duration, start_from_stage="S2")

    s1_start_after = {
        k: v for k, v in sched.get_jik_2_start_time_map().items() if k[1] == "S1"
    }
    s1_end_after = {
        k: v for k, v in sched.get_jik_2_end_time_map().items() if k[1] == "S1"
    }

    assert s1_start_before == s1_start_after
    assert s1_end_before == s1_end_after


def test_start_from_stage_equals_full_retiming_result():
    """Partial retiming from any stage should give the same result as full."""
    sched_full, duration = _build_3job_3stage_schedule()

    for stage in ["S1", "S2", "S3"]:
        sched_partial, _ = _build_3job_3stage_schedule()
        sched_partial.make_semi_active(duration, start_from_stage=stage)

        assert (
            sched_full.get_jik_2_start_time_map()
            == sched_partial.get_jik_2_start_time_map()
        ), f"start_from_stage={stage} diverges from full retiming (start_time_map)"
        assert (
            sched_full.get_jik_2_end_time_map()
            == sched_partial.get_jik_2_end_time_map()
        ), f"start_from_stage={stage} diverges from full retiming (end_time_map)"


def test_swap_then_start_from_stage_equals_full_retiming():
    """Swap at S2 + partial retiming must match swap + full retiming."""
    # Build two identical schedules.
    sched_partial, duration = _build_3job_3stage_schedule()
    sched_full, _ = _build_3job_3stage_schedule()

    # Swap J1 <-> J2 in S2 on both, but use different retiming strategies.
    sched_partial.swap_two_operations_within_stage(
        "S2", "J1", "J2", duration, do_make_semi_active=True
    )

    # Manual swap + full retiming on the other schedule.
    priv = "_HybridFlowshopLiteSchedule__stage_2_mc_2_job_tuple_seq"
    seq_m1 = getattr(sched_full, priv)["S2"]["M1"]
    seq_m2 = getattr(sched_full, priv)["S2"]["M2"]
    s1, e1, _ = seq_m1[0]  # J1
    s2, e2, _ = seq_m2[0]  # J2
    seq_m1[0] = (s1, e1, "J2")
    seq_m2[0] = (s2, e2, "J1")
    sched_full.make_semi_active(duration)  # full retiming

    validate_schedule(sched_partial, duration)
    validate_schedule(sched_full, duration)

    assert (
        sched_partial.get_jik_2_start_time_map()
        == sched_full.get_jik_2_start_time_map()
    )
    assert sched_partial.get_jik_2_end_time_map() == sched_full.get_jik_2_end_time_map()


def test_start_from_stage_invariants_on_3_stages():
    """Invariants hold after partial retiming on a 3-stage schedule."""
    sched, duration = _build_3job_3stage_schedule()
    sched.make_semi_active(duration, start_from_stage="S2")
    validate_schedule(sched, duration)


def test_start_from_stage_invalid_raises():
    sched, duration = _build_3job_3stage_schedule()
    with pytest.raises(ValueError, match="Invalid stage ID"):
        sched.make_semi_active(duration, start_from_stage="INVALID")


# ============================================================================
# Tests for find_critical_blocks()
# ============================================================================


def test_find_critical_blocks_not_empty_after_make_semi_active_with_last_stage_ops():
    """After make_semi_active, critical blocks should not be empty when last stage has ops."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1", "J2", "J3"],
        stages=["S1", "S2"],
        machines_per_stage={"S1": ["M1"], "S2": ["M1"]},
    )
    duration: dict[str, dict[str, int]] = {
        "S1": {"J1": 3, "J2": 4, "J3": 2},
        "S2": {"J1": 5, "J2": 3, "J3": 6},
    }

    # Use dummy times and then retime to a feasible semi-active schedule.
    sched.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J2", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S1", "M1", "J3", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J2", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J3", start_time=0, end_time=0)

    sched.make_semi_active(duration)

    # Ensure the precondition from the requirement: the last stage has operations.
    last_stage = sched.stages[-1]
    assert any(True for _ in sched.iter_operations_on_stage(last_stage))

    blocks = sched.find_critical_blocks(duration)

    assert blocks != []


def test_find_critical_blocks_not_empty_when_single_critical_op_in_last_stage():
    """Even a single-op last stage should yield a non-empty critical-block list."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["J1"],
        stages=["S1", "S2"],
        machines_per_stage={"S1": ["M1"], "S2": ["M1"]},
    )
    duration: dict[str, dict[str, int]] = {
        "S1": {"J1": 4},
        "S2": {"J1": 7},
    }

    sched.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=0)
    sched.add_ops_times_2_mc("S2", "M1", "J1", start_time=0, end_time=0)

    sched.make_semi_active(duration)

    last_stage = sched.stages[-1]
    assert any(True for _ in sched.iter_operations_on_stage(last_stage))

    blocks = sched.find_critical_blocks(duration, include_singletons=True)

    assert blocks != []


# ============================================================================
# Tests for dispatch_wave_batches()
# ============================================================================


def test_wave_batch_basic():
    """Test basic wave batch dispatching: 3 jobs, 2 stages, batch_size=2."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    stage_2_job_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
    }
    sched.dispatch_wave_batches(
        batch_size=2,
        job_ids=["j1", "j2", "j3"],
        stage_ids=["s1", "s2"],
        stage_2_job_2_duration=stage_2_job_2_duration,
    )

    # Verify all jobs are scheduled in both stages
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s1", "j3") == 6
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s2", "j3") == 11

    # Validate the schedule
    validate_schedule(sched, stage_2_job_2_duration)


def test_wave_batch_with_release_times():
    """Test wave batch dispatching with release times."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    stage_2_job_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
    }
    job_2_release_time = {"j1": 1, "j2": 0, "j3": 5}
    sched.dispatch_wave_batches(
        batch_size=2,
        job_ids=["j1", "j2", "j3"],
        stage_ids=["s1", "s2"],
        stage_2_job_2_duration=stage_2_job_2_duration,
        job_2_release_time=job_2_release_time,
    )

    # j1 has release time 1, so it starts at 1, ends at 3
    assert sched.get_job_end_time("s1", "j1") == 3
    # j2 has release time 0, starts at 3 (after j1), ends at 6
    assert sched.get_job_end_time("s1", "j2") == 6
    # j3 has release time 5, starts at 6, ends at 7
    assert sched.get_job_end_time("s1", "j3") == 7

    # Validate the schedule
    validate_schedule(sched, stage_2_job_2_duration)


def test_wave_batch_batch_size_one():
    """Test wave batch dispatching with batch_size=1 (degenerate case)."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    stage_2_job_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
    }
    sched.dispatch_wave_batches(
        batch_size=1,
        job_ids=["j1", "j2", "j3"],
        stage_ids=["s1", "s2"],
        stage_2_job_2_duration=stage_2_job_2_duration,
    )

    # Jobs should still be scheduled correctly
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s1", "j3") == 6
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s2", "j3") == 11

    validate_schedule(sched, stage_2_job_2_duration)


def test_wave_batch_batch_size_larger_than_jobs():
    """Test wave batch dispatching with batch_size >= num_jobs (all at once)."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    stage_2_job_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
    }
    sched.dispatch_wave_batches(
        batch_size=10,  # All at once
        job_ids=["j1", "j2", "j3"],
        stage_ids=["s1", "s2"],
        stage_2_job_2_duration=stage_2_job_2_duration,
    )

    # All jobs should be scheduled
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s1", "j3") == 6
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s2", "j3") == 11

    validate_schedule(sched, stage_2_job_2_duration)


def test_wave_batch_3_stage():
    """Test wave batch dispatching with 3 stages (diagonal propagation)."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    stage_2_job_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1, "j4": 2},
        "s2": {"j1": 2, "j2": 2, "j3": 2, "j4": 2},
        "s3": {"j1": 2, "j2": 2, "j3": 2, "j4": 2},
    }
    sched.dispatch_wave_batches(
        batch_size=2,
        job_ids=["j1", "j2", "j3", "j4"],
        stage_ids=["s1", "s2", "s3"],
        stage_2_job_2_duration=stage_2_job_2_duration,
    )

    # All jobs should be scheduled in all stages
    for stage in ["s1", "s2", "s3"]:
        for job in ["j1", "j2", "j3", "j4"]:
            assert sched.get_job_end_time(stage, job) > 0

    # Validate the schedule
    validate_schedule(sched, stage_2_job_2_duration)

    # Makespan should be 13
    assert sched.makespan == 13


def test_wave_batch_invalid_batch_size():
    """Test that invalid batch_size raises ValueError."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    stage_2_job_2_duration = {"s1": {"j1": 5}}

    with pytest.raises(ValueError, match="batch_size must be positive"):
        sched.dispatch_wave_batches(
            batch_size=0,
            job_ids=["j1"],
            stage_ids=["s1"],
            stage_2_job_2_duration=stage_2_job_2_duration,
        )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        sched.dispatch_wave_batches(
            batch_size=-1,
            job_ids=["j1"],
            stage_ids=["s1"],
            stage_2_job_2_duration=stage_2_job_2_duration,
        )


def test_wave_batch_missing_duration():
    """Test that missing duration raises ValueError."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    # Missing duration for j2 at s2
    stage_2_job_2_duration = {
        "s1": {"j1": 2, "j2": 3},
        "s2": {"j1": 3},  # j2 missing
    }

    with pytest.raises(
        ValueError, match="Duration for job ID j2 at stage s2 not provided"
    ):
        sched.dispatch_wave_batches(
            batch_size=2,
            job_ids=["j1", "j2"],
            stage_ids=["s1", "s2"],
            stage_2_job_2_duration=stage_2_job_2_duration,
        )


# ============================================================================
# Tests for from_job_sequence_get_schedule_mixed()
# ============================================================================


def test_from_job_sequence_get_schedule_mixed_basic():
    """Test basic k=2 dispatch strategy."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4", "j5"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 5, "j5": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4, "j4": 2, "j5": 3},
        "s3": {"j1": 1, "j2": 2, "j3": 3, "j4": 4, "j5": 2},
    }

    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 2}
    )

    # j1 and j2 should be scheduled at all stages via dispatch_job_by_stages
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert schedule.get_job_end_time("s2", "j1") == 5
    assert schedule.get_job_end_time("s3", "j1") == 6

    assert schedule.get_job_end_time("s1", "j2") == 5
    assert schedule.get_job_end_time("s2", "j2") == 7
    assert schedule.get_job_end_time("s3", "j2") == 9

    # j3, j4, j5 should be scheduled via dispatch_stage_by_jobs at each stage
    # dispatch_stage_by_jobs uses priority (prev stage end time, then sequence order)
    # At s1: j3@S1 ends at 7 (starts at 5, duration 2)
    # At s1: j4@S1 ends at 12 (starts at 7, duration 5)
    # At s1: j5@S1 ends at 13 (starts at 12, duration 1)
    assert schedule.get_job_end_time("s1", "j3") == 7
    assert schedule.get_job_end_time("s1", "j4") == 12
    assert schedule.get_job_end_time("s1", "j5") == 13

    # At s2: j3 starts at max(7, 5) = 7 (needs to wait for j3@S1 end at 7), so j3@S2 = 7+4=11
    # At s2: j4 starts at max(12, 11) = 12 (needs to wait for j4@S1 end at 12), so j4@S2 = 12+2=14
    # At s2: j5 starts at max(13, 14) = 14 (needs to wait for j5@S1 end at 13 and j4@S2 end at 14), so j5@S2 = 14+3=17
    assert schedule.get_job_end_time("s2", "j3") == 11
    assert schedule.get_job_end_time("s2", "j4") == 14
    assert schedule.get_job_end_time("s2", "j5") == 17

    # At s3: j3 starts at max(11, 6) = 11 (needs to wait for j3@S2 end at 11), so j3@S3 = 11+3=14
    # At s3: j4 starts at max(14, 14) = 14 (needs to wait for j4@S2 end at 14), so j4@S3 = 14+4=18
    # At s3: j5 starts at max(17, 18) = 18 (needs to wait for j5@S2 end at 17 and j4@S3 end at 18), so j5@S3 = 18+2=20
    assert schedule.get_job_end_time("s3", "j3") == 14
    assert schedule.get_job_end_time("s3", "j4") == 18
    assert schedule.get_job_end_time("s3", "j5") == 20

    # Makespan should be 20
    assert schedule.makespan == 20


def test_from_job_sequence_get_schedule_mixed_k_per_stage():
    """Test k_per_stage mapping for different k values at different stages."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4", "j5"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 5, "j5": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4, "j4": 2, "j5": 3},
    }

    # Stage 1: dispatch 2 jobs fully, Stage 2: dispatch 3 jobs fully
    k_per_stage = {"s1": 2, "s2": 3}

    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, k_per_stage
    )

    # j1 and j2 should be scheduled at all stages via dispatch_job_by_stages
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert schedule.get_job_end_time("s2", "j1") == 5

    assert schedule.get_job_end_time("s1", "j2") == 5
    assert schedule.get_job_end_time("s2", "j2") == 7

    # j3 should be scheduled at all stages via dispatch_job_by_stages (k_per_stage["s2"]=3)
    assert schedule.get_job_end_time("s1", "j3") == 7  # starts at 5, duration 2
    assert schedule.get_job_end_time("s2", "j3") == 11  # starts at 7, duration 4

    # j4, j5 should be scheduled via dispatch_stage_by_jobs at each stage
    # For s1: remaining are j4, j5 (j1, j2, j3 already scheduled)
    assert schedule.get_job_end_time("s1", "j4") == 12  # starts at 7, duration 5
    assert schedule.get_job_end_time("s1", "j5") == 13  # starts at 12, duration 1

    # For s2: remaining are j4, j5 (j1, j2, j3 already scheduled)
    # Priority order: j4 (end=12), j5 (end=13)
    assert schedule.get_job_end_time("s2", "j4") == 14  # starts at 12, duration 2
    assert schedule.get_job_end_time("s2", "j5") == 17  # starts at 14, duration 3

    assert schedule.makespan == 17


def test_from_job_sequence_get_schedule_mixed_validation():
    """Test validation of duration mappings."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    # Missing duration for j2 at s2
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3},
        "s2": {"j1": 3},  # j2 missing
    }

    with pytest.raises(
        ValueError, match="Duration for job ID j2 at stage s2 not provided"
    ):
        from_job_sequence_get_schedule_mixed(
            sched, ["j1", "j2"], stage_2_duration, {"s1": 2}
        )


def test_from_job_sequence_get_schedule_mixed_stage_2_head_cumulative():
    """Test that stage_2_head is cumulatively adjusted based on remaining jobs.

    With 5 jobs and stage_2_head={"s1": 3, "s2": 3}:
    - s1 dispatches 3 jobs (j1, j2, j3) via dispatch_job_by_stages
    - s2 sees only 2 jobs remaining, so it dispatches min(3, 2) = 2 jobs via dispatch_job_by_stages
    - All remaining jobs are filled via dispatch_stage_by_jobs
    """
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4", "j5"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 5, "j5": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4, "j4": 2, "j5": 3},
    }

    # s1: dispatch 3 jobs, s2: also requests 3 but only 2 remain
    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 3, "s2": 3}
    )

    # j1, j2, j3 should be scheduled at all stages via dispatch_job_by_stages
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert schedule.get_job_end_time("s2", "j1") == 5

    assert schedule.get_job_end_time("s1", "j2") == 5
    assert schedule.get_job_end_time("s2", "j2") == 7

    assert schedule.get_job_end_time("s1", "j3") == 7
    assert schedule.get_job_end_time("s2", "j3") == 11

    # j4, j5 should be scheduled via dispatch_stage_by_jobs at each stage
    # For s1: remaining are j4, j5
    assert schedule.get_job_end_time("s1", "j4") == 12
    assert schedule.get_job_end_time("s1", "j5") == 13

    # For s2: remaining are j4, j5 (j1, j2, j3 already scheduled)
    assert schedule.get_job_end_time("s2", "j4") == 14
    assert schedule.get_job_end_time("s2", "j5") == 17

    assert schedule.makespan == 17


def test_from_job_sequence_get_schedule_mixed_makespan():
    """Test makespan comparison between different k values."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 2, "j4": 5, "j5": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4, "j4": 2, "j5": 3},
        "s3": {"j1": 1, "j2": 2, "j3": 3, "j4": 4, "j5": 2},
    }

    # Test with k_per_stage={"s1": 1}
    sched_k1 = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4", "j5"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    schedule_k1 = from_job_sequence_get_schedule_mixed(
        sched_k1, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 1}
    )

    # Test with k_per_stage={"s1": 3}
    sched_k3 = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4", "j5"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    schedule_k3 = from_job_sequence_get_schedule_mixed(
        sched_k3, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 3}
    )

    # Makespans should be valid (no overlaps, precedence satisfied)
    assert schedule_k1.makespan > 0
    assert schedule_k3.makespan > 0

    # k=3 might give better or worse makespan depending on the instance
    # We just verify both are valid schedules
    validate_schedule(schedule_k1, stage_2_duration)
    validate_schedule(schedule_k3, stage_2_duration)


def test_from_job_sequence_get_schedule_mixed_single_job():
    """Test scheduling a single job."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2},
        "s2": {"j1": 3},
    }

    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1"], stage_2_duration, {"s1": 1}
    )

    # j1 should be scheduled at all stages via dispatch_job_by_stages
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert schedule.get_job_end_time("s2", "j1") == 5
    assert schedule.makespan == 5


def test_from_job_sequence_get_schedule_mixed_all_via_dispatch_job():
    """Test when all jobs are dispatched via dispatch_job_by_stages."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
    }

    # All 3 jobs dispatched via dispatch_job_by_stages at s1
    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3"], stage_2_duration, {"s1": 3}
    )

    # All jobs should be scheduled at all stages via dispatch_job_by_stages
    # s1: j1(0-2), j2(2-5), j3(5-6)
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert (
        schedule.get_job_end_time("s2", "j1") == 5
    )  # s1 end + s2 duration = 2 + 3 = 5

    assert schedule.get_job_end_time("s1", "j2") == 5
    assert (
        schedule.get_job_end_time("s2", "j2") == 7
    )  # s1 end + s2 duration = 5 + 2 = 7

    assert schedule.get_job_end_time("s1", "j3") == 6
    assert (
        schedule.get_job_end_time("s2", "j3") == 11
    )  # s1 end + s2 duration = 6 + 4 = 10... wait, 11?

    # The actual behavior: j1 ends at 5 on s2, j2 starts at 5 ends at 7, j3 starts at 7 ends at 11
    # This is because dispatch_job_by_stages schedules jobs sequentially on the same machine
    assert schedule.makespan == 11


def test_from_job_sequence_get_schedule_mixed_empty_head():
    """Test when stage_2_head has 0 for all stages (all via dispatch_stage_by_jobs)."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
    }

    # No jobs dispatched via dispatch_job_by_stages (all via dispatch_stage_by_jobs)
    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3"], stage_2_duration, {"s1": 0, "s2": 0}
    )

    # At s1: dispatch_stage_by_jobs schedules sequentially on same machine
    # j1 ends at 2, j2 ends at 5, j3 ends at 6
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert schedule.get_job_end_time("s1", "j2") == 5
    assert schedule.get_job_end_time("s1", "j3") == 6

    # At s2: priority order is j1 (end=2), j3 (end=6), j2 (end=5)
    # j1 starts at 2, ends at 5
    # j3 starts at max(6, 5) = 6, ends at 10
    # j2 starts at max(5, 10) = 10, ends at 12
    # But actual behavior shows sequential dispatch: j1(2-5), j2(5-7), j3(7-11)
    assert schedule.get_job_end_time("s2", "j1") == 5
    assert schedule.get_job_end_time("s2", "j2") == 7
    assert schedule.get_job_end_time("s2", "j3") == 11

    assert schedule.makespan == 11


def test_from_job_sequence_get_schedule_mixed_stages_with_multiple_machines():
    """Test mixed dispatch with multiple machines per stage."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1, "j4": 2},
        "s2": {"j1": 3, "j2": 2, "j3": 4, "j4": 1},
    }

    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4"], stage_2_duration, {"s1": 2}
    )

    # j1 and j2 are dispatched via dispatch_job_by_stages at s1
    # j1 on m1: end=2, j2 on m2: end=3
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert schedule.get_job_end_time("s1", "j2") == 3

    # j3, j4 dispatched via dispatch_stage_by_jobs at s1
    # j3 on m1 (next available after j1) ends at 3, j4 on m2 ends at 5
    assert schedule.get_job_end_time("s1", "j3") == 3
    assert schedule.get_job_end_time("s1", "j4") == 5

    # At s2, all jobs use the same machine m1
    # Priority order based on s1 end times (with tiebreaker by sequence order):
    # j1 (2), j2 (3), j3 (3), j4 (5)
    # j1 starts at 2, ends at 5
    # j2 starts at max(3, 5) = 5, ends at 7
    # j3 starts at max(3, 7) = 7, ends at 11
    # j4 starts at max(5, 11) = 11, ends at 12
    assert schedule.get_job_end_time("s2", "j1") == 5
    assert schedule.get_job_end_time("s2", "j2") == 7
    assert schedule.get_job_end_time("s2", "j3") == 11
    assert schedule.get_job_end_time("s2", "j4") == 12

    assert schedule.makespan == 12


def test_from_job_sequence_get_schedule_mixed_validation_empty_schedule():
    """Test that non-empty schedule raises ValueError."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3},
        "s2": {"j1": 3, "j2": 2},
    }

    # Pre-populate schedule with one operation
    sched.add_operation_2_stage("s1", "j1", 2)

    with pytest.raises(ValueError, match="schedule must be empty"):
        from_job_sequence_get_schedule_mixed(
            sched, ["j1", "j2"], stage_2_duration, {"s1": 2}
        )


def test_from_job_sequence_get_schedule_mixed_validation_invalid_head():
    """Test that invalid stage_2_head raises ValueError."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2},
        "s2": {"j1": 3},
    }

    # Invalid stage_id
    with pytest.raises(ValueError, match="Unknown stage_id in stage_2_head"):
        from_job_sequence_get_schedule_mixed(
            sched, ["j1"], stage_2_duration, {"invalid_stage": 1}
        )

    # Negative value
    with pytest.raises(ValueError, match="stage_2_head values must be non-negative"):
        from_job_sequence_get_schedule_mixed(
            sched, ["j1"], stage_2_duration, {"s1": -1}
        )


def test_from_job_sequence_get_schedule_mixed_single_stage():
    """Test mixed dispatch with a single stage."""
    from hybridflowshop.schedule_lite import from_job_sequence_get_schedule_mixed

    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
    }

    schedule = from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3"], stage_2_duration, {"s1": 2}
    )

    # j1 and j2 via dispatch_job_by_stages, j3 via dispatch_stage_by_jobs
    assert schedule.get_job_end_time("s1", "j1") == 2
    assert schedule.get_job_end_time("s1", "j2") == 5
    assert schedule.get_job_end_time("s1", "j3") == 6

    assert schedule.makespan == 6


def test_get_job_priority_queue_for_stage_dispatch():
    """Test the get_job_priority_queue_for_stage_dispatch method."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
        "s3": {"j1": 2, "j2": 1, "j3": 3},
    }

    # Schedule j1 and j3 at s1 first
    sched.dispatch_stage_by_jobs("s1", ["j1", "j3"], stage_2_duration["s1"])

    # At s1: j1 ends at 2, j3 ends at 3
    # At s2, priority based on s1 end times:
    # j2 has default_if_missing=0, j1 ends at 2, j3 ends at 3
    # Priority order: j2 (0), j1 (2), j3 (3)
    priority = sched.get_job_priority_queue_for_stage_dispatch("s2", ["j1", "j2", "j3"])
    assert priority == ["j2", "j1", "j3"], f"Got {priority}"

    # With release times: add j2 at s2
    sched.add_operation_2_stage("s2", "j2", 2, release_t=10)
    # j2 now ends at 12 at s2

    # At s3, priorities based on s2 end times:
    # j1 ends at 5 (2+3), j3 ends at 7 (3+4), j2 ends at 12
    priority = sched.get_job_priority_queue_for_stage_dispatch("s3", ["j1", "j2", "j3"])
    assert priority == ["j1", "j3", "j2"], f"Got {priority}"


# ============================================================================
# Tests for dispatch_stage_by_jobs_filtered() - DEPRECATED
# ============================================================================

# These tests are deprecated and kept for reference only.
# The dispatch_stage_by_jobs_filtered method was removed in favor of the
# new standalone from_job_sequence_get_schedule_mixed function.
