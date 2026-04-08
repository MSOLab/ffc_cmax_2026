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

    mc_id, t = sched.select_machine_by_earliest_start_then_idle("s1", duration=5)
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
    assert copied.jobs == ["j1", "j2"]
    assert copied.jobs is not sched.jobs
    assert copied.stages == ["s1"]
    assert copied.stages is not sched.stages
    assert copied.machines_per_stage == {"s1": ["m1"]}
    assert copied.machines_per_stage is not sched.machines_per_stage
    assert copied.machines_per_stage["s1"] is not sched.machines_per_stage["s1"]


def test_deepcopy_fully_detaches_metadata_and_internal_state():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)

    copied = sched.deepcopy()

    assert copied.jobs == sched.jobs
    assert copied.jobs is not sched.jobs
    assert copied.stages == sched.stages
    assert copied.stages is not sched.stages
    assert copied.machines_per_stage == sched.machines_per_stage
    assert copied.machines_per_stage is not sched.machines_per_stage
    assert copied.machines_per_stage["s1"] is not sched.machines_per_stage["s1"]

    copied_job_2_end = _get_priv(copied, "__stage_2_job_2_end_time")
    sched_job_2_end = _get_priv(sched, "__stage_2_job_2_end_time")
    assert copied_job_2_end is not sched_job_2_end
    assert copied_job_2_end["s1"] is not sched_job_2_end["s1"]

    copied_seq = _get_priv(copied, "__stage_2_mc_2_job_tuple_seq")
    sched_seq = _get_priv(sched, "__stage_2_mc_2_job_tuple_seq")
    assert copied_seq is not sched_seq
    assert copied_seq["s1"] is not sched_seq["s1"]
    assert copied_seq["s1"]["m1"] is not sched_seq["s1"]["m1"]

    copied.jobs.append("j3")
    copied.stages.append("s2")
    copied.machines_per_stage["s1"].append("m2")
    copied.get_job_sequence("s1", "m1").append((2, 4, "j2"))

    assert sched.jobs == ["j1", "j2"]
    assert sched.stages == ["s1"]
    assert sched.machines_per_stage == {"s1": ["m1"]}
    assert sched.get_job_sequence("s1", "m1") == [(0, 2, "j1")]


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


def test_dispatch_stage_by_jobs_strict_sequence_preserves_input_dispatch_order():
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

    # Strict-sequence dispatch must respect the given input order ["b", "a"].
    sched.dispatch_stage_by_jobs_strict_sequence(
        "i1",
        job_id_seq=["b", "a"],
        job_2_duration={"a": 7, "b": 5},
    )

    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("b", "i1", "m1")] == 2
    assert start_map[("a", "i1", "m1")] == 10


def test_dispatch_stage_by_jobs_strict_start_order_preserves_final_start_order():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "b"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )

    sched.dispatch_stage_by_jobs_strict_start_order(
        "s1",
        job_id_seq=["a", "b"],
        job_2_duration={"a": 5, "b": 5},
    )

    start_map = sched.get_jik_2_start_time_map()
    a_start = min(
        start
        for (job_id, _stage_id, _mc_id), start in start_map.items()
        if job_id == "a"
    )
    b_start = min(
        start
        for (job_id, _stage_id, _mc_id), start in start_map.items()
        if job_id == "b"
    )
    assert a_start == 0
    assert b_start >= 1
    assert a_start < b_start


def test_dispatch_stage_reversed_by_jobs_uses_latest_feasible_slot_before_lct():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "fixed"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "fixed", start_time=8, end_time=10)

    sched.dispatch_stage_reversed_by_jobs(
        "s1",
        job_id_seq=["a"],
        job_2_duration={"a": 3},
        mc_2_lct={"m1": 8},
    )

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()
    assert start_map[("a", "s1", "m1")] == 5
    assert end_map[("a", "s1", "m1")] == 8


def test_dispatch_stage_reversed_by_jobs_uses_latest_interior_gap_before_lct():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "left", "right"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "left", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s1", "m1", "right", start_time=8, end_time=10)

    sched.dispatch_stage_reversed_by_jobs(
        "s1",
        job_id_seq=["a"],
        job_2_duration={"a": 3},
        mc_2_lct={"m1": 8},
    )

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()
    assert start_map[("a", "s1", "m1")] == 5
    assert end_map[("a", "s1", "m1")] == 8


def test_dispatch_stage_reversed_by_jobs_respects_job_deadline():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "fixed"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "fixed", start_time=9, end_time=10)

    sched.dispatch_stage_reversed_by_jobs(
        "s1",
        job_id_seq=["a"],
        job_2_duration={"a": 3},
        mc_2_lct={"m1": 9},
        job_2_deadline={"a": 7},
    )

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()
    assert start_map[("a", "s1", "m1")] == 4
    assert end_map[("a", "s1", "m1")] == 7


def test_dispatch_stage_reversed_by_jobs_uses_next_stage_start_as_upper_bound():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "fixed", "tail"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "fixed", start_time=8, end_time=10)
    sched.add_ops_times_2_mc("s2", "m2", "a", start_time=6, end_time=9)
    sched.add_ops_times_2_mc("s2", "m2", "tail", start_time=9, end_time=11)

    sched.dispatch_stage_reversed_by_jobs(
        "s1",
        job_id_seq=["a"],
        job_2_duration={"a": 3},
        mc_2_lct={"m1": 8},
    )

    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()
    assert start_map[("a", "s1", "m1")] == 3
    assert end_map[("a", "s1", "m1")] == 6


def test_dispatch_stage_reversed_by_jobs_picks_machine_with_latest_slot():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "x", "y"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "x", start_time=6, end_time=8)
    sched.add_ops_times_2_mc("s1", "m2", "y", start_time=8, end_time=10)

    sched.dispatch_stage_reversed_by_jobs(
        "s1",
        job_id_seq=["a"],
        job_2_duration={"a": 2},
        mc_2_lct={"m1": 6, "m2": 8},
    )

    end_map = sched.get_jik_2_end_time_map()
    assert end_map[("a", "s1", "m2")] == 8


def test_dispatch_stage_reversed_by_jobs_uses_gap_before_first_operation():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "fixed"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "fixed", start_time=5, end_time=7)

    sched.dispatch_stage_reversed_by_jobs(
        "s1",
        job_id_seq=["a"],
        job_2_duration={"a": 2},
        mc_2_lct={"m1": 5},
    )

    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("a", "s1", "m1")] == 3


def test_dispatch_stage_reversed_by_jobs_raises_when_no_slot_exists_before_zero():
    sched = HybridFlowshopLiteSchedule(
        jobs=["a", "fixed"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "fixed", start_time=1, end_time=3)

    with pytest.raises(
        ValueError, match="Unable to reverse-dispatch job a on stage s1"
    ):
        sched.dispatch_stage_reversed_by_jobs(
            "s1",
            job_id_seq=["a"],
            job_2_duration={"a": 2},
            mc_2_lct={"m1": 2},
        )


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


def _build_subset_retiming_fixture() -> tuple[
    HybridFlowshopLiteSchedule, dict[str, dict[str, int]]
]:
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1"]},
    )
    duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    # Semi-active baseline:
    # s1.m1: j1[0,5]
    # s1.m2: j2[0,5]
    # s2.m1: j1[5,10], j2[10,15]
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m2", "j2", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=5, end_time=10)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=10, end_time=15)

    return sched, duration


def test_make_semi_active_empty_operation_set_matches_full():
    sched_full = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    sched_empty = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    for sched in (sched_full, sched_empty):
        sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=5, end_time=10)
        sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=10, end_time=15)
        sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=10, end_time=15)
        sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=15, end_time=20)

    sched_full.make_semi_active(duration)
    sched_empty.make_semi_active(duration, operation_set=set())

    assert (
        sched_empty.get_jik_2_start_time_map() == sched_full.get_jik_2_start_time_map()
    )
    assert sched_empty.get_jik_2_end_time_map() == sched_full.get_jik_2_end_time_map()


def test_make_semi_active_subset_keeps_unselected_operations_fixed():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=10, end_time=15)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=5, end_time=10)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=15, end_time=20)

    sched.make_semi_active(duration, operation_set={("j2", "s1", "m1")})

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("j2", "s1", "m1")] == 5
    assert start_map[("j2", "s2", "m1")] == 15
    assert start_map[("j1", "s2", "m1")] == 5


def test_make_semi_active_subset_propagates_across_stages():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=10, end_time=15)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=5, end_time=10)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=15, end_time=20)

    sched.make_semi_active(
        duration,
        operation_set={("j2", "s1", "m1"), ("j2", "s2", "m1")},
    )

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("j2", "s1", "m1")] == 5
    assert start_map[("j2", "s2", "m1")] == 10


def test_make_semi_active_subset_respects_fixed_machine_anchor():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    duration = {"s1": {"j1": 5, "j2": 5, "j3": 5}}

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=8, end_time=13)
    sched.add_ops_times_2_mc("s1", "m1", "j3", start_time=15, end_time=20)

    sched.make_semi_active(duration, operation_set={("j2", "s1", "m1")})

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("j2", "s1", "m1")] == 5
    assert start_map[("j3", "s1", "m1")] == 15


def test_make_semi_active_subset_with_start_from_stage_leaves_earlier_stage_untouched():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    duration: dict[str, dict[str, int]] = {
        "s1": {"j1": 5, "j2": 5},
        "s2": {"j1": 5, "j2": 5},
    }

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=2, end_time=7)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=12, end_time=17)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=7, end_time=12)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=20, end_time=25)

    sched.make_semi_active(
        duration,
        start_from_stage="s2",
        operation_set={("j2", "s2", "m1")},
    )

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("j1", "s1", "m1")] == 2
    assert start_map[("j2", "s1", "m1")] == 12
    assert start_map[("j2", "s2", "m1")] == 17


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


def _build_suffix_swap_schedule() -> tuple[
    HybridFlowshopLiteSchedule, dict[str, dict[str, int]]
]:
    sched = HybridFlowshopLiteSchedule(
        jobs=["A", "B", "C", "D", "E"],
        stages=["S1"],
        machines_per_stage={"S1": ["M1", "M2"]},
    )
    sched.add_ops_times_2_mc("S1", "M1", "A", 0, 2)
    sched.add_ops_times_2_mc("S1", "M1", "B", 2, 4)
    sched.add_ops_times_2_mc("S1", "M1", "C", 4, 6)
    sched.add_ops_times_2_mc("S1", "M2", "D", 0, 2)
    sched.add_ops_times_2_mc("S1", "M2", "E", 2, 4)
    duration = {"S1": {"A": 2, "B": 2, "C": 2, "D": 2, "E": 2}}
    return sched, duration


def test_collect_stage_machine_suffix_job_ids_returns_suffix():
    sched, _duration = _build_suffix_swap_schedule()

    assert sched.collect_stage_machine_suffix_job_ids("S1", "M1", "B") == ["B", "C"]


def test_swap_stage_machine_operation_sets_preserves_unselected_order():
    sched, duration = _build_suffix_swap_schedule()

    sched.swap_stage_machine_operation_sets(
        "S1",
        "M1",
        ["B"],
        "M2",
        ["E"],
        duration,
        do_make_semi_active=False,
    )

    assert [job_id for _s, _e, job_id in sched.get_job_sequence("S1", "M1")] == [
        "A",
        "E",
        "C",
    ]
    assert [job_id for _s, _e, job_id in sched.get_job_sequence("S1", "M2")] == [
        "D",
        "B",
    ]


def test_swap_stage_machine_operation_sets_with_make_semi_active_retimes():
    sched, duration = _build_suffix_swap_schedule()

    sched.swap_stage_machine_operation_sets(
        "S1",
        "M1",
        ["A", "B"],
        "M2",
        ["E"],
        duration,
        do_make_semi_active=True,
    )

    validate_schedule(sched, duration)
    assert [job_id for _s, _e, job_id in sched.get_job_sequence("S1", "M1")] == [
        "E",
        "C",
    ]
    assert [job_id for _s, _e, job_id in sched.get_job_sequence("S1", "M2")] == [
        "D",
        "A",
        "B",
    ]


def test_swap_stage_machine_operation_sets_duplicate_job_ids_raise():
    sched, duration = _build_suffix_swap_schedule()

    with pytest.raises(ValueError, match="Duplicate job IDs"):
        sched.swap_stage_machine_operation_sets(
            "S1",
            "M1",
            ["B", "B"],
            "M2",
            ["E"],
            duration,
        )


def test_swap_stage_machine_operation_sets_same_machine_raises():
    sched, duration = _build_suffix_swap_schedule()

    with pytest.raises(ValueError, match="requires two machines"):
        sched.swap_stage_machine_operation_sets(
            "S1",
            "M1",
            ["B"],
            "M1",
            ["C"],
            duration,
        )


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


# ============= Tests for get_machine_earliest_start_time =============


def test_get_machine_earliest_start_time_empty_machine():
    """Empty machine returns release_t or 0."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # No operations, defaults to 0
    assert sched.get_machine_earliest_start_time("s1", "m1", duration=5) == 0

    # With release_t
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=5, release_t=10)
        == 10
    )


def test_get_machine_earliest_start_time_after_last_flag():
    """after_last=True returns latest end time or release_t, whichever is larger."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [0, 10)
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=10)

    # after_last=True: returns max(makespan=10, release_t=0) = 10
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=5, after_last=True)
        == 10
    )

    # after_last=True with release_t > makespan
    assert (
        sched.get_machine_earliest_start_time(
            "s1", "m1", duration=5, release_t=15, after_last=True
        )
        == 15
    )


def test_get_machine_earliest_start_time_insert_in_gap():
    """Operation can be inserted into an existing idle gap."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [0, 5), J2 [10, 15) - gap from 5 to 10
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=10, end_time=15)

    # duration=3 fits in gap [5, 10)
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=3, release_t=5) == 5
    )

    # duration=6 does NOT fit in gap [5, 10), must start after J2
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=6, release_t=5) == 15
    )


def test_get_machine_earliest_start_time_release_t_within_operation():
    """release_t falls within an operation - must wait for it to complete."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [0, 10)
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=10)

    # release_t=5 is within J1, must wait until 10
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=3, release_t=5) == 10
    )


def test_get_machine_earliest_start_time_release_t_before_first_op():
    """release_t is before all operations - can start immediately if gap fits."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [10, 20)
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=10, end_time=20)

    # release_t=0, duration=5 fits before J1 at 10
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=5, release_t=0) == 0
    )

    # release_t=0, duration=11 does NOT fit before J1 at 10
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=11, release_t=0)
        == 20
    )


def test_get_machine_earliest_start_time_multiple_gaps():
    """Selects the first gap that can fit the duration."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [0, 5), J2 [10, 15), J3 [20, 25)
    # Gaps: [5, 10) size=5, [15, 20) size=5
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=10, end_time=15)
    sched.add_ops_times_2_mc("s1", "m1", "j3", start_time=20, end_time=25)

    # release_t=5, duration=3 fits in first gap [5, 10)
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=3, release_t=5) == 5
    )

    # release_t=5, duration=4 fits in first gap [5, 10) since 5+4=9 <= 10
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=4, release_t=5) == 5
    )

    # release_t=5, duration=6 does NOT fit in first gap [5, 10),
    # after J2 ends at 15, 15+6=21 > 20 (J3 start), so must wait for J3 to end at 25
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=6, release_t=5) == 25
    )


def test_get_machine_earliest_start_time_consecutive_operations():
    """No gaps - operation must be scheduled after the last one."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [0, 5), J2 [5, 10) - no gaps
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=5, end_time=10)

    # Must start after J2
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=3, release_t=0) == 10
    )


def test_get_machine_earliest_start_time_invalid_stage():
    """Invalid stage_id raises ValueError."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    with pytest.raises(ValueError, match="Invalid stage ID"):
        sched.get_machine_earliest_start_time("invalid", "m1", duration=5)


def test_get_machine_earliest_start_time_invalid_machine():
    """Invalid machine_id raises ValueError."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    with pytest.raises(ValueError, match="Invalid machine ID"):
        sched.get_machine_earliest_start_time("s1", "invalid", duration=5)


def test_get_machine_earliest_start_time_invalid_duration():
    """Duration <= 0 raises ValueError."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    with pytest.raises(ValueError, match="Duration must be greater than 0"):
        sched.get_machine_earliest_start_time("s1", "m1", duration=0)

    with pytest.raises(ValueError, match="Duration must be greater than 0"):
        sched.get_machine_earliest_start_time("s1", "m1", duration=-5)


def test_get_machine_earliest_start_time_multi_machine_scenario():
    """Test with multiple machines having different schedules."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )

    # m1: J1 [0, 5), J2 [10, 15)
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=10, end_time=15)

    # m2: J3 [2, 8)
    sched.add_ops_times_2_mc("s1", "m2", "j3", start_time=2, end_time=8)

    # m1 at release_t=7: fits in gap [7, 10)
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=3, release_t=7) == 7
    )

    # m2 at release_t=0: gap [0, 2) fits duration=2
    assert (
        sched.get_machine_earliest_start_time("s1", "m2", duration=2, release_t=0) == 0
    )

    # m2 at release_t=5: within J3 [2, 8), must wait until 8
    assert (
        sched.get_machine_earliest_start_time("s1", "m2", duration=3, release_t=5) == 8
    )


def test_get_machine_earliest_start_time_chained_operations_expand():
    """Consecutive operations where release_t causes expansion."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [0, 5), J2 [5, 10), J3 [10, 15)
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=5, end_time=10)
    sched.add_ops_times_2_mc("s1", "m1", "j3", start_time=10, end_time=15)

    # release_t=7: within J2, must expand past J2 and J3
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=3, release_t=7) == 15
    )


def test_get_machine_earliest_start_time_exact_gap_fit():
    """Duration exactly fits a gap."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [0, 5), J2 [10, 15) - gap [5, 10) size=5
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=10, end_time=15)

    # duration=5 exactly fits gap [5, 10)
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=5, release_t=5) == 5
    )


def test_get_machine_earliest_start_time_release_t_equals_operation_start():
    """release_t exactly equals an operation start time."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    # J1 [10, 20)
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=10, end_time=20)

    # release_t=10: exactly at J1 start, must wait until 20
    assert (
        sched.get_machine_earliest_start_time("s1", "m1", duration=3, release_t=10)
        == 20
    )


def test_as_reversed_reverses_stage_order():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=3)
    sched.add_ops_times_2_mc("s1", "m2", "j2", start_time=1, end_time=4)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=3, end_time=5)
    sched.add_ops_times_2_mc("s2", "m2", "j2", start_time=4, end_time=7)

    reversed_sched = sched.as_reversed()

    assert list(reversed_sched.stages) == ["s2", "s1"]


def test_as_reversed_time_transform_formula_exact():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=3)
    sched.add_ops_times_2_mc("s1", "m2", "j2", start_time=1, end_time=4)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=3, end_time=5)
    sched.add_ops_times_2_mc("s2", "m2", "j2", start_time=4, end_time=7)

    reversed_sched = sched.as_reversed()
    rev_start = reversed_sched.get_jik_2_start_time_map()
    rev_end = reversed_sched.get_jik_2_end_time_map()
    anchor = sched.makespan

    # as_reversed keeps original stage and machine IDs, only reverses time
    expected = {}
    for (
        job_id,
        stage_orig,
        mc_id,
    ), start_orig in sched.get_jik_2_start_time_map().items():
        end_orig = sched.get_jik_2_end_time_map()[(job_id, stage_orig, mc_id)]
        # Stage and machine IDs remain the same; only time is transformed
        expected[(job_id, stage_orig, mc_id)] = (
            anchor - end_orig,
            anchor - start_orig,
        )

    for op, (exp_start, exp_end) in expected.items():
        assert rev_start[op] == exp_start
        assert rev_end[op] == exp_end


def test_as_reversed_preserves_machine_and_job():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1", "m2"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=3)
    sched.add_ops_times_2_mc("s2", "m2", "j1", start_time=3, end_time=6)
    sched.add_ops_times_2_mc("s1", "m2", "j2", start_time=1, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=6, end_time=8)

    reversed_sched = sched.as_reversed()

    # as_reversed keeps original stage and machine IDs, only reverses time
    orig_ops = sched.get_operation_set()
    rev_ops = reversed_sched.get_operation_set()
    # Stage and machine IDs remain unchanged
    assert rev_ops == orig_ops


def test_as_reversed_non_inplace():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    original_start = dict(sched.get_jik_2_start_time_map())
    original_end = dict(sched.get_jik_2_end_time_map())

    reversed_sched = sched.as_reversed()

    assert reversed_sched is not sched
    assert sched.get_jik_2_start_time_map() == original_start
    assert sched.get_jik_2_end_time_map() == original_end


def test_as_reversed_empty_schedule():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )

    reversed_sched = sched.as_reversed()

    assert list(reversed_sched.stages) == ["s3", "s2", "s1"]
    assert reversed_sched.get_jik_2_start_time_map() == {}
    assert reversed_sched.get_jik_2_end_time_map() == {}
    assert reversed_sched.makespan == 0


def test_as_reversed_remaps_machine_by_stage_position():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["i0", "i1"],
        machines_per_stage={
            "i0": ["i0_0", "i0_1"],
            "i1": ["i1_0", "i1_1"],
        },
    )
    sched.add_ops_times_2_mc("i0", "i0_1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("i1", "i1_0", "j1", start_time=2, end_time=5)

    reversed_sched = sched.as_reversed()
    ops = reversed_sched.get_operation_set()

    # as_reversed keeps original stage and machine IDs
    assert ("j1", "i0", "i0_1") in ops
    assert ("j1", "i1", "i1_0") in ops


# ============================================================================
# Tests for make_right_justified()
# ============================================================================


def test_make_right_justified_preserves_makespan_and_feasibility():
    sched, duration = _build_subset_retiming_fixture()
    original_makespan = sched.makespan

    sched.make_right_justified(duration)

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert sched.makespan == original_makespan
    assert start_map[("j2", "s1", "m2")] == 5


def test_make_right_justified_already_right_justified_is_unchanged():
    sched, duration = _build_subset_retiming_fixture()
    sched.make_right_justified(duration)
    start_map_before = sched.get_jik_2_start_time_map()
    end_map_before = sched.get_jik_2_end_time_map()

    sched.make_right_justified(duration)

    assert sched.get_jik_2_start_time_map() == start_map_before
    assert sched.get_jik_2_end_time_map() == end_map_before


def test_make_right_justified_multi_machine_expected_positions():
    sched, duration = _build_subset_retiming_fixture()

    sched.make_right_justified(duration)

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()
    assert start_map[("j1", "s1", "m1")] == 0
    assert end_map[("j1", "s1", "m1")] == 5
    assert start_map[("j2", "s1", "m2")] == 5
    assert end_map[("j2", "s1", "m2")] == 10
    assert start_map[("j1", "s2", "m1")] == 5
    assert start_map[("j2", "s2", "m1")] == 10


def test_make_right_justified_last_stage_respects_original_makespan():
    sched, duration = _build_subset_retiming_fixture()
    original_makespan = sched.makespan

    sched.make_right_justified(duration)

    last_stage = sched.stages[-1]
    for _mc, _start, end, _job in sched.iter_operations_on_stage(last_stage):
        assert end <= original_makespan
    assert sched.makespan == original_makespan


def test_make_right_justified_subset_keeps_unselected_operations_fixed():
    sched, duration = _build_subset_retiming_fixture()
    start_map_before = sched.get_jik_2_start_time_map()

    sched.make_right_justified(duration, operation_set={("j2", "s1", "m2")})

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("j2", "s1", "m2")] == 5
    assert start_map[("j1", "s1", "m1")] == start_map_before[("j1", "s1", "m1")]
    assert start_map[("j1", "s2", "m1")] == start_map_before[("j1", "s2", "m1")]
    assert start_map[("j2", "s2", "m1")] == start_map_before[("j2", "s2", "m1")]


def test_make_right_justified_subset_zero_slack_operation_stays_put():
    sched, duration = _build_subset_retiming_fixture()

    sched.make_right_justified(duration, operation_set={("j1", "s1", "m1")})

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("j1", "s1", "m1")] == 0


def test_make_right_justified_subset_respects_fixed_machine_successor():
    sched, duration = _build_subset_retiming_fixture()

    sched.make_right_justified(duration, operation_set={("j1", "s2", "m1")})

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    assert start_map[("j1", "s2", "m1")] == 5
    assert start_map[("j2", "s2", "m1")] == 10


def test_make_right_justified_subset_respects_next_stage_precedence():
    sched, duration = _build_subset_retiming_fixture()

    sched.make_right_justified(duration, operation_set={("j2", "s1", "m2")})

    validate_schedule(sched, duration)
    start_map = sched.get_jik_2_start_time_map()
    end_map = sched.get_jik_2_end_time_map()
    assert end_map[("j2", "s1", "m2")] == start_map[("j2", "s2", "m1")]
