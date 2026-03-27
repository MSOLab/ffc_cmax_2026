"""Tests for remove_operations and remove_jobs methods in HybridFlowshopLiteSchedule."""

import pytest

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule

# =============================================================================
# Tests for remove_operations (copied from test_schedule_lite.py)
# =============================================================================


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


# =============================================================================
# Tests for remove_jobs
# =============================================================================


def test_remove_jobs_single_job_across_all_stages():
    """Test removing a single job removes all its operations across all stages."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)

    sched.remove_jobs({"j1"})

    # j1 should be completely removed
    assert sched.get_jik_2_start_time_map() == {
        ("j2", "s1", "m1"): 2,
        ("j2", "s2", "m1"): 5,
    }
    assert sched.get_jik_2_end_time_map() == {
        ("j2", "s1", "m1"): 4,
        ("j2", "s2", "m1"): 7,
    }

    with pytest.raises(ValueError):
        sched.get_job_end_time("s1", "j1")
    with pytest.raises(ValueError):
        sched.get_job_end_time("s2", "j1")

    # j2 should remain intact
    assert sched.get_job_end_time("s1", "j2") == 4
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.makespan == 7


def test_remove_jobs_multiple_jobs():
    """Test removing multiple jobs at once."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s1", "m1", "j3", start_time=4, end_time=6)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)
    sched.add_ops_times_2_mc("s2", "m1", "j3", start_time=7, end_time=9)

    sched.remove_jobs({"j1", "j3"})

    # Only j2 should remain
    assert sched.get_jik_2_start_time_map() == {
        ("j2", "s1", "m1"): 2,
        ("j2", "s2", "m1"): 5,
    }
    assert sched.get_jik_2_end_time_map() == {
        ("j2", "s1", "m1"): 4,
        ("j2", "s2", "m1"): 7,
    }

    with pytest.raises(ValueError):
        sched.get_job_end_time("s1", "j1")
    with pytest.raises(ValueError):
        sched.get_job_end_time("s2", "j1")
    with pytest.raises(ValueError):
        sched.get_job_end_time("s1", "j3")
    with pytest.raises(ValueError):
        sched.get_job_end_time("s2", "j3")

    assert sched.makespan == 7


def test_remove_jobs_all_jobs():
    """Test removing all jobs results in an empty schedule."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)

    sched.remove_jobs({"j1", "j2"})

    # All operations should be removed
    assert sched.get_jik_2_start_time_map() == {}
    assert sched.get_jik_2_end_time_map() == {}
    assert sched.makespan == 0

    with pytest.raises(ValueError):
        sched.get_job_end_time("s1", "j1")
    with pytest.raises(ValueError):
        sched.get_job_end_time("s2", "j2")


def test_remove_jobs_empty_set():
    """Test removing an empty job set leaves the schedule unchanged."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)

    original_start_map = sched.get_jik_2_start_time_map().copy()
    original_end_map = sched.get_jik_2_end_time_map().copy()

    sched.remove_jobs(set())

    # Schedule should be unchanged
    assert sched.get_jik_2_start_time_map() == original_start_map
    assert sched.get_jik_2_end_time_map() == original_end_map
    assert sched.makespan == 7


def test_remove_jobs_with_multiple_machines_per_stage():
    """Test removing jobs when there are multiple machines per stage."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1"]},
    )

    # j1 and j2 on m1, j3 on m2 for s1
    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s1", "m2", "j3", start_time=0, end_time=3)
    # All jobs on s2
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)
    sched.add_ops_times_2_mc("s2", "m1", "j3", start_time=7, end_time=9)

    sched.remove_jobs({"j1"})

    # j1 should be removed from both stages
    start_map = sched.get_jik_2_start_time_map()
    assert ("j1", "s1", "m1") not in start_map
    assert ("j1", "s2", "m1") not in start_map

    # j2 and j3 should remain
    assert start_map[("j2", "s1", "m1")] == 2
    assert start_map[("j3", "s1", "m2")] == 0
    assert start_map[("j2", "s2", "m1")] == 5
    assert start_map[("j3", "s2", "m1")] == 7

    assert sched.makespan == 9


def test_remove_jobs_updates_makespan():
    """Test that makespan is correctly updated after removing jobs."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s1", "m1", "j3", start_time=5, end_time=10)

    assert sched.makespan == 10

    sched.remove_jobs({"j3"})
    assert sched.makespan == 5

    sched.remove_jobs({"j1"})
    assert sched.makespan == 5  # j2 still ends at 5

    sched.remove_jobs({"j2"})
    assert sched.makespan == 0  # No jobs left


def test_remove_jobs_nonexistent_job():
    """Test removing a job that doesn't exist doesn't cause errors."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )

    sched.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=2)
    sched.add_ops_times_2_mc("s1", "m1", "j2", start_time=2, end_time=4)
    sched.add_ops_times_2_mc("s2", "m1", "j1", start_time=2, end_time=5)
    sched.add_ops_times_2_mc("s2", "m1", "j2", start_time=5, end_time=7)

    original_start_map = sched.get_jik_2_start_time_map().copy()

    # Removing a non-existent job should not change anything
    sched.remove_jobs({"j999"})

    assert sched.get_jik_2_start_time_map() == original_start_map
    assert sched.makespan == 7
