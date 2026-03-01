"""Tests for dispatch_stage_by_machines_3 with pre-scheduled operations.

These tests verify that the method correctly handles stages that already
have scheduled operations, particularly the gap insertion functionality.
"""

from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    validate_schedule,
)


class TestDispatchStageByMachines3WithPreScheduledOps:
    """Tests for dispatch_stage_by_machines_3 with pre-scheduled operations."""

    def test_basic_gap_insertion_single_machine(self) -> None:
        """Test that new jobs can be inserted into gaps created by pre-scheduled ops.

        Setup:
        - Stage i1 has machine m1 with pre-scheduled ops: [0, 5) for x1, [10, 15) for x2
        - This creates a gap [5, 10) of length 5
        - Job a (duration=3) and job b (duration=2) should fit into this gap

        Expected:
        - Both jobs should be scheduled without overlap
        - validate_schedule should pass
        """
        sched = HybridFlowshopLiteSchedule(
            jobs=["a", "b", "x1", "x2"],
            stages=["i0", "i1"],
            machines_per_stage={"i0": ["m0"], "i1": ["m1"]},
        )

        # Pre-schedule operations to create a gap [5, 10)
        sched.add_ops_times_2_mc("i1", "m1", "x1", start_time=0, end_time=5)
        sched.add_ops_times_2_mc("i1", "m1", "x2", start_time=10, end_time=15)

        # Dispatch new jobs to the stage
        sched.dispatch_stage_by_machines_3(
            "i1",
            job_id_seq=["a", "b"],
            stage_2_job_2_p={"i1": {"a": 3, "b": 2}},
            job_2_release={"a": 0, "b": 0},
        )

        # Validate no overlaps
        validate_schedule(sched, {"i1": {"a": 3, "b": 2, "x1": 5, "x2": 5}})

        # Check that jobs were scheduled in the gap
        start_map = sched.get_jik_2_start_time_map()
        # One job should start at 5 (beginning of gap)
        assert start_map[("a", "i1", "m1")] == 5 or start_map[("b", "i1", "m1")] == 5

    def test_multiple_machines_with_gaps(self) -> None:
        """Test gap selection across multiple machines.

        Setup:
        - Stage i1 has machines m1 and m2
        - m1 has pre-scheduled op: [0, 10) for x1 (gap after: infinite)
        - m2 has pre-scheduled ops: [0, 5) for y1, [8, 13) for y2 (gap [5, 8))
        - Job a (duration=2) should prefer the finite gap on m2

        Expected:
        - Job a should be scheduled in the gap [5, 8) on m2
        - No overlaps should occur
        """
        sched = HybridFlowshopLiteSchedule(
            jobs=["a", "x1", "y1", "y2"],
            stages=["i0", "i1"],
            machines_per_stage={"i0": ["m0"], "i1": ["m1", "m2"]},
        )

        # Pre-schedule operations
        sched.add_ops_times_2_mc("i1", "m1", "x1", start_time=0, end_time=10)
        sched.add_ops_times_2_mc("i1", "m2", "y1", start_time=0, end_time=5)
        sched.add_ops_times_2_mc("i1", "m2", "y2", start_time=8, end_time=13)

        # Dispatch new job
        sched.dispatch_stage_by_machines_3(
            "i1",
            job_id_seq=["a"],
            stage_2_job_2_p={"i1": {"a": 2}},
            job_2_release={"a": 0},
        )

        # Validate no overlaps
        validate_schedule(sched, {"i1": {"a": 2, "x1": 10, "y1": 5, "y2": 5}})

        # Check that job a was scheduled in the gap on m2
        start_map = sched.get_jik_2_start_time_map()
        assert start_map[("a", "i1", "m2")] == 5

    def test_gap_too_small_for_any_job(self) -> None:
        """Test behavior when existing gaps are too small for any job.

        Setup:
        - Stage i1 has machine m1 with pre-scheduled ops creating small gaps
        - Gap [5, 7) of length 2
        - Job a (duration=5) is too large for the gap

        Expected:
        - Job should be scheduled after the gap (at time 10 or later)
        - No overlaps should occur
        """
        sched = HybridFlowshopLiteSchedule(
            jobs=["a", "x1", "x2"],
            stages=["i0", "i1"],
            machines_per_stage={"i0": ["m0"], "i1": ["m1"]},
        )

        # Pre-schedule operations creating a small gap [5, 7)
        sched.add_ops_times_2_mc("i1", "m1", "x1", start_time=0, end_time=5)
        sched.add_ops_times_2_mc("i1", "m1", "x2", start_time=7, end_time=10)

        # Dispatch job too large for the gap
        sched.dispatch_stage_by_machines_3(
            "i1",
            job_id_seq=["a"],
            stage_2_job_2_p={"i1": {"a": 5}},
            job_2_release={"a": 0},
        )

        # Validate no overlaps
        validate_schedule(sched, {"i1": {"a": 5, "x1": 5, "x2": 3}})

        # Job should be scheduled after the current gap is exhausted
        start_map = sched.get_jik_2_start_time_map()
        # Should be scheduled at or after time 10 (after x2)
        assert start_map[("a", "i1", "m1")] >= 10

    def test_gap_exactly_fits_job(self) -> None:
        """Test correct handling when job duration equals gap length.

        Setup:
        - Stage i1 has machine m1 with pre-scheduled ops
        - Gap [5, 10) of length 5
        - Job a (duration=5) exactly fits the gap

        Expected:
        - Job should fill the gap completely: [5, 10)
        - No overlaps should occur
        """
        sched = HybridFlowshopLiteSchedule(
            jobs=["a", "x1", "x2"],
            stages=["i0", "i1"],
            machines_per_stage={"i0": ["m0"], "i1": ["m1"]},
        )

        # Pre-schedule operations creating gap [5, 10)
        sched.add_ops_times_2_mc("i1", "m1", "x1", start_time=0, end_time=5)
        sched.add_ops_times_2_mc("i1", "m1", "x2", start_time=10, end_time=15)

        # Dispatch job that exactly fits the gap
        sched.dispatch_stage_by_machines_3(
            "i1",
            job_id_seq=["a"],
            stage_2_job_2_p={"i1": {"a": 5}},
            job_2_release={"a": 0},
        )

        # Validate no overlaps
        validate_schedule(sched, {"i1": {"a": 5, "x1": 5, "x2": 5}})

        # Job should exactly fill the gap
        start_map = sched.get_jik_2_start_time_map()
        assert start_map[("a", "i1", "m1")] == 5

    def test_multiple_jobs_multiple_gaps(self) -> None:
        """Test scheduling multiple jobs into multiple gaps.

        Setup:
        - Stage i1 has machine m1 with pre-scheduled ops creating multiple gaps
        - Gaps: [3, 6), [9, 12)
        - Jobs a (duration=2), b (duration=2), c (duration=2)

        Expected:
        - Jobs should fill gaps without overlap
        - All jobs should be scheduled
        """
        sched = HybridFlowshopLiteSchedule(
            jobs=["a", "b", "c", "x1", "x2", "x3"],
            stages=["i0", "i1"],
            machines_per_stage={"i0": ["m0"], "i1": ["m1"]},
        )

        # Pre-schedule operations creating gaps [3, 6) and [9, 12)
        sched.add_ops_times_2_mc("i1", "m1", "x1", start_time=0, end_time=3)
        sched.add_ops_times_2_mc("i1", "m1", "x2", start_time=6, end_time=9)
        sched.add_ops_times_2_mc("i1", "m1", "x3", start_time=12, end_time=15)

        # Dispatch multiple jobs
        sched.dispatch_stage_by_machines_3(
            "i1",
            job_id_seq=["a", "b", "c"],
            stage_2_job_2_p={"i1": {"a": 2, "b": 2, "c": 2}},
            job_2_release={"a": 0, "b": 0, "c": 0},
        )

        # Validate no overlaps
        validate_schedule(
            sched, {"i1": {"a": 2, "b": 2, "c": 2, "x1": 3, "x2": 3, "x3": 3}}
        )

        # All jobs should be scheduled
        start_map = sched.get_jik_2_start_time_map()
        assert ("a", "i1", "m1") in start_map
        assert ("b", "i1", "m1") in start_map
        assert ("c", "i1", "m1") in start_map

    def test_pre_scheduled_ops_before_release_time(self) -> None:
        """Test when pre-scheduled ops end before job release time.

        Setup:
        - Stage i1 has machine m1 with pre-scheduled op ending at time 5
        - Job a has release time 10 (from previous stage)
        - Gap from 5 to infinity exists, but job can only start at 10

        Expected:
        - Job should start at release time 10
        - No overlaps should occur
        """
        sched = HybridFlowshopLiteSchedule(
            jobs=["a", "x1"],
            stages=["i0", "i1"],
            machines_per_stage={"i0": ["m0"], "i1": ["m1"]},
        )

        # Pre-schedule operation ending at time 5
        sched.add_ops_times_2_mc("i1", "m1", "x1", start_time=0, end_time=5)

        # Dispatch job with release time 10
        sched.dispatch_stage_by_machines_3(
            "i1",
            job_id_seq=["a"],
            stage_2_job_2_p={"i1": {"a": 3}},
            job_2_release={"a": 10},
        )

        # Validate no overlaps
        validate_schedule(sched, {"i1": {"a": 3, "x1": 5}})

        # Job should start at or after release time
        start_map = sched.get_jik_2_start_time_map()
        assert start_map[("a", "i1", "m1")] == 10

    def test_empty_stage_no_pre_scheduled_ops(self) -> None:
        """Test dispatching to an empty stage (no pre-scheduled ops).

        This is a baseline test to ensure the method still works
        for the common case of empty stages.

        Expected:
        - Jobs should be scheduled starting from time 0
        - No overlaps should occur
        """
        sched = HybridFlowshopLiteSchedule(
            jobs=["a", "b"],
            stages=["i0", "i1"],
            machines_per_stage={"i0": ["m0"], "i1": ["m1"]},
        )

        # Dispatch to empty stage
        sched.dispatch_stage_by_machines_3(
            "i1",
            job_id_seq=["a", "b"],
            stage_2_job_2_p={"i1": {"a": 3, "b": 2}},
            job_2_release={"a": 0, "b": 0},
        )

        # Validate no overlaps
        validate_schedule(sched, {"i1": {"a": 3, "b": 2}})

        # Jobs should be scheduled starting from 0
        start_map = sched.get_jik_2_start_time_map()
        # At least one job should start at 0
        assert start_map[("a", "i1", "m1")] == 0 or start_map[("b", "i1", "m1")] == 0
