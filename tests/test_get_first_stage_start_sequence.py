from types import SimpleNamespace

from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    get_first_stage_start_sequence,
)


def test_get_first_stage_start_sequence_orders_by_start_time_and_job_index():
    """Test basic ordering by first stage start time with tie-breaking by job index."""
    instance = SimpleNamespace(
        job_id_list=["b", "a", "c", "d"],
        stage_id_list=["s1", "s2"],
    )

    schedule = HybridFlowshopLiteSchedule(
        jobs=instance.job_id_list,
        stages=instance.stage_id_list,
        machines_per_stage={"s1": ["m1", "m2", "m3", "m4"], "s2": ["m1"]},
    )

    # Add operations at first stage (s1) with different start times
    # Job "b" at index 0 starts at 10
    # Job "a" at index 1 starts at 5
    # Job "c" at index 2 starts at 10
    # Job "d" at index 3 starts at 15
    schedule.add_ops_times_2_mc("s1", "m1", "b", start_time=10, end_time=20)
    schedule.add_ops_times_2_mc("s1", "m2", "a", start_time=5, end_time=15)
    schedule.add_ops_times_2_mc("s1", "m3", "c", start_time=10, end_time=18)
    schedule.add_ops_times_2_mc("s1", "m4", "d", start_time=15, end_time=25)

    seq = get_first_stage_start_sequence(schedule)

    # Expected order: a (start=5), then b and c (start=10, tie broken by index: b=0, c=2), then d (start=15)
    assert seq == ["a", "b", "c", "d"]


def test_get_first_stage_start_sequence_all_same_start_times():
    """Test tie-breaking when all jobs have the same start time."""
    instance = SimpleNamespace(
        job_id_list=["z", "x", "y"],
        stage_id_list=["s1", "s2"],
    )

    schedule = HybridFlowshopLiteSchedule(
        jobs=instance.job_id_list,
        stages=instance.stage_id_list,
        machines_per_stage={"s1": ["m1", "m2", "m3"], "s2": ["m1"]},
    )

    # All jobs start at the same time at first stage
    schedule.add_ops_times_2_mc("s1", "m1", "z", start_time=10, end_time=20)
    schedule.add_ops_times_2_mc("s1", "m2", "x", start_time=10, end_time=20)
    schedule.add_ops_times_2_mc("s1", "m3", "y", start_time=10, end_time=20)

    seq = get_first_stage_start_sequence(schedule)

    # All have same start time, so order by original job index: z=0, x=1, y=2
    assert seq == ["z", "x", "y"]


def test_get_first_stage_start_sequence_single_job():
    """Test with a single job in the schedule."""
    instance = SimpleNamespace(
        job_id_list=["only_job"],
        stage_id_list=["s1", "s2"],
    )

    schedule = HybridFlowshopLiteSchedule(
        jobs=instance.job_id_list,
        stages=instance.stage_id_list,
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )

    schedule.add_ops_times_2_mc("s1", "m1", "only_job", start_time=0, end_time=10)

    seq = get_first_stage_start_sequence(schedule)

    assert seq == ["only_job"]


def test_get_first_stage_start_sequence_descending_start_times():
    """Test when jobs have descending start times (reverse order)."""
    instance = SimpleNamespace(
        job_id_list=["a", "b", "c"],
        stage_id_list=["s1"],
    )

    schedule = HybridFlowshopLiteSchedule(
        jobs=instance.job_id_list,
        stages=instance.stage_id_list,
        machines_per_stage={"s1": ["m1", "m2", "m3"]},
    )

    # Jobs added in order a, b, c but start times are descending
    schedule.add_ops_times_2_mc("s1", "m1", "a", start_time=30, end_time=40)
    schedule.add_ops_times_2_mc("s1", "m2", "b", start_time=20, end_time=30)
    schedule.add_ops_times_2_mc("s1", "m3", "c", start_time=10, end_time=20)

    seq = get_first_stage_start_sequence(schedule)

    # Should be ordered by start time: c (10), b (20), a (30)
    assert seq == ["c", "b", "a"]
