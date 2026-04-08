"""Tests for dispatcher utilities (dispatch functions)."""

import pytest

from hybridflowshop.dispatcher.utils import (
    build_schedule_from_stage_job_sequences_priority_score,
    build_schedule_from_stage_job_sequences_strict_call_order,
    dispatch_stage_job_sequences_strict_call_order,
    dispatch_job_sequence_by_stages,
    dispatch_stages_by_job_sequence,
    from_job_sequence_get_schedule_mixed,
    get_job_sequence_from_dispatch_windows_aggregate,
    get_job_sequence_from_dispatch_windows_anchor_stage,
    get_job_tiebreak_rank_from_job_sequence,
    get_job_tiebreak_rank_from_stage_job_sequences,
    get_stage_job_release_times_from_dispatch_windows,
    get_stage_job_sequences_from_dispatch_windows,
    improve_schedule_by_critical_adjacent_swaps,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, validate_schedule

# ============================================================================
# Tests for from_job_sequence_get_schedule_mixed()
# ============================================================================


def test_get_job_tiebreak_rank_from_stage_job_sequences_uses_first_stage():
    stage_2_job_sequence = {
        "s1": ["j3", "j1", "j2"],
        "s2": ["j2", "j1", "j3"],
    }

    rank_map = get_job_tiebreak_rank_from_stage_job_sequences(
        ["s1", "s2"], stage_2_job_sequence
    )

    assert rank_map == {"j3": 0, "j1": 1, "j2": 2}


def test_get_stage_job_sequences_from_dispatch_windows_uses_longer_processing_time_tiebreak():
    dispatch_window_lookup = {
        (1, 1): {"early_start": 10, "late_start": 20},
        (1, 2): {"early_start": 10, "late_start": 20},
        (1, 3): {"early_start": 10, "late_start": 21},
    }
    stage_2_job_2_p = {
        "s1": {"j1": 3, "j2": 7, "j3": 20},
    }

    seq = get_stage_job_sequences_from_dispatch_windows(
        ["s1"],
        ["j1", "j2", "j3"],
        dispatch_window_lookup,
        stage_2_job_2_p,
    )

    assert seq["s1"] == ["j2", "j1", "j3"]


def test_get_stage_job_sequences_from_dispatch_windows_supports_ls_first_rule():
    dispatch_window_lookup = {
        (1, 1): {"early_start": 10, "late_start": 50},
        (1, 2): {"early_start": 30, "late_start": 40},
    }
    stage_2_job_2_p = {
        "s1": {"j1": 3, "j2": 7},
    }

    seq = get_stage_job_sequences_from_dispatch_windows(
        ["s1"],
        ["j1", "j2"],
        dispatch_window_lookup,
        stage_2_job_2_p,
        sort_rule="ls_es_p_desc",
    )

    assert seq["s1"] == ["j2", "j1"]


def test_get_job_sequence_from_dispatch_windows_anchor_stage_uses_requested_stage():
    dispatch_window_lookup = {
        (1, 1): {"early_start": 0, "late_start": 20},
        (1, 2): {"early_start": 5, "late_start": 10},
        (2, 1): {"early_start": 100, "late_start": 120},
        (2, 2): {"early_start": 20, "late_start": 30},
    }
    stage_2_job_2_p = {
        "s1": {"j1": 3, "j2": 7},
        "s2": {"j1": 5, "j2": 2},
    }

    seq = get_job_sequence_from_dispatch_windows_anchor_stage(
        ["s1", "s2"],
        ["j1", "j2"],
        dispatch_window_lookup,
        stage_2_job_2_p,
        anchor_stage_id="s2",
        sort_rule="ls_es_p_desc",
    )

    assert seq == ["j2", "j1"]


def test_get_job_sequence_from_dispatch_windows_aggregate_uses_sum_slack_rule():
    dispatch_window_lookup = {
        (1, 1): {"early_start": 0, "late_start": 10},
        (1, 2): {"early_start": 0, "late_start": 5},
        (2, 1): {"early_start": 10, "late_start": 30},
        (2, 2): {"early_start": 5, "late_start": 8},
    }
    stage_2_job_2_p = {
        "s1": {"j1": 3, "j2": 7},
        "s2": {"j1": 5, "j2": 2},
    }

    seq = get_job_sequence_from_dispatch_windows_aggregate(
        ["s1", "s2"],
        ["j1", "j2"],
        dispatch_window_lookup,
        stage_2_job_2_p,
        aggregation_rule="sum_ls_slack_p_desc",
    )

    assert seq == ["j2", "j1"]


def test_get_job_tiebreak_rank_from_job_sequence():
    assert get_job_tiebreak_rank_from_job_sequence(["j3", "j1", "j2"]) == {
        "j3": 0,
        "j1": 1,
        "j2": 2,
    }


def test_get_stage_job_release_times_from_dispatch_windows_uses_early_start():
    dispatch_window_lookup = {
        (1, 1): {"early_start": 10, "late_start": 20},
        (1, 2): {"early_start": 5, "late_start": 8},
    }

    releases = get_stage_job_release_times_from_dispatch_windows(
        ["s1"],
        ["j1", "j2"],
        dispatch_window_lookup,
    )

    assert releases == {"s1": {"j1": 10, "j2": 5}}


def test_build_schedule_from_stage_job_sequences_priority_score_basic():
    stage_2_job_sequence = {
        "s1": ["j2", "j1", "j3"],
        "s2": ["j3", "j2", "j1"],
    }
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
    }

    schedule = build_schedule_from_stage_job_sequences_priority_score(
        lambda: HybridFlowshopLiteSchedule(
            jobs=["j1", "j2", "j3"],
            stages=["s1", "s2"],
            machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
        ),
        stage_2_job_sequence,
        stage_2_job_2_p,
    )

    validate_schedule(schedule, stage_2_job_2_p)
    assert [job_id for _, _, job_id in schedule.get_job_sequence("s1", "m1")] == [
        "j2",
        "j1",
        "j3",
    ]


def test_build_schedule_from_stage_job_sequences_strict_call_order_respects_es_release():
    stage_2_job_sequence = {
        "s1": ["j1", "j2"],
    }
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3},
    }
    stage_2_job_2_release = {
        "s1": {"j1": 5, "j2": 0},
    }

    schedule = build_schedule_from_stage_job_sequences_strict_call_order(
        lambda: HybridFlowshopLiteSchedule(
            jobs=["j1", "j2"],
            stages=["s1"],
            machines_per_stage={"s1": ["m1"]},
        ),
        stage_2_job_sequence,
        stage_2_job_2_p,
        stage_2_job_2_release=stage_2_job_2_release,
    )

    assert schedule.get_job_start_time("s1", "j1") >= 5
    assert schedule.get_job_start_time("s1", "j2") >= 0


def test_build_schedule_from_stage_job_sequences_priority_score_respects_es_release():
    stage_2_job_sequence = {
        "s1": ["j1", "j2"],
    }
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3},
    }
    stage_2_job_2_release = {
        "s1": {"j1": 10, "j2": 0},
    }

    schedule = build_schedule_from_stage_job_sequences_priority_score(
        lambda: HybridFlowshopLiteSchedule(
            jobs=["j1", "j2"],
            stages=["s1"],
            machines_per_stage={"s1": ["m1"]},
        ),
        stage_2_job_sequence,
        stage_2_job_2_p,
        stage_2_job_2_release=stage_2_job_2_release,
    )

    assert schedule.get_job_start_time("s1", "j2") == 0
    assert schedule.get_job_start_time("s1", "j1") == 10


def test_improve_schedule_by_critical_adjacent_swaps_improves_known_case():
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 9, "j3": 7, "j4": 9},
        "s2": {"j1": 5, "j2": 9, "j3": 4, "j4": 4},
        "s3": {"j1": 7, "j2": 5, "j3": 8, "j4": 8},
    }
    schedule = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1", "m2"], "s3": ["m1", "m2"]},
    )

    schedule.add_ops_times_2_mc("s1", "m1", "j4", 0, 9)
    schedule.add_ops_times_2_mc("s1", "m1", "j1", 9, 11)
    schedule.add_ops_times_2_mc("s1", "m2", "j2", 0, 9)
    schedule.add_ops_times_2_mc("s1", "m2", "j3", 9, 16)

    schedule.add_ops_times_2_mc("s2", "m1", "j4", 9, 13)
    schedule.add_ops_times_2_mc("s2", "m1", "j3", 16, 20)
    schedule.add_ops_times_2_mc("s2", "m2", "j2", 9, 18)
    schedule.add_ops_times_2_mc("s2", "m2", "j1", 18, 23)

    schedule.add_ops_times_2_mc("s3", "m1", "j4", 13, 21)
    schedule.add_ops_times_2_mc("s3", "m1", "j1", 23, 30)
    schedule.add_ops_times_2_mc("s3", "m2", "j3", 20, 28)
    schedule.add_ops_times_2_mc("s3", "m2", "j2", 28, 33)

    validate_schedule(schedule, stage_2_job_2_p)
    assert schedule.makespan == 33

    improved = improve_schedule_by_critical_adjacent_swaps(
        schedule,
        stage_2_job_2_p,
        max_passes=3,
    )

    validate_schedule(improved, stage_2_job_2_p)
    assert improved.makespan == 31


def test_improve_schedule_by_critical_adjacent_swaps_preserves_release_constraints():
    stage_2_job_2_p = {
        "s1": {"j1": 5, "j2": 4},
    }
    schedule = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    schedule.add_ops_times_2_mc("s1", "m1", "j2", 0, 4)
    schedule.add_ops_times_2_mc("s1", "m1", "j1", 10, 15)

    improved = improve_schedule_by_critical_adjacent_swaps(
        schedule,
        stage_2_job_2_p,
        max_passes=2,
        stage_2_job_2_release={"s1": {"j1": 10, "j2": 0}},
    )

    assert improved.get_job_start_time("s1", "j1") >= 10


def test_from_job_sequence_get_schedule_mixed_basic():
    """Test basic k=2 dispatch strategy."""
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

    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 2}
    )

    # j1 and j2 should be scheduled at all stages via dispatch_job_by_stages
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s3", "j1") == 6

    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s3", "j2") == 9

    # j3, j4, j5 should be scheduled via dispatch_stage_by_jobs at each stage
    # dispatch_stage_by_jobs uses priority (prev stage end time, then sequence order)
    # At s1: j3@S1 ends at 7 (starts at 5, duration 2)
    # At s1: j4@S1 ends at 12 (starts at 7, duration 5)
    # At s1: j5@S1 ends at 13 (starts at 12, duration 1)
    assert sched.get_job_end_time("s1", "j3") == 7
    assert sched.get_job_end_time("s1", "j4") == 12
    assert sched.get_job_end_time("s1", "j5") == 13

    # At s2: j3 starts at max(7, 5) = 7 (needs to wait for j3@S1 end at 7), so j3@S2 = 7+4=11
    # At s2: j4 starts at max(12, 11) = 12 (needs to wait for j4@S1 end at 12), so j4@S2 = 12+2=14
    # At s2: j5 starts at max(13, 14) = 14 (needs to wait for j5@S1 end at 13 and j4@S2 end at 14), so j5@S2 = 14+3=17
    assert sched.get_job_end_time("s2", "j3") == 11
    assert sched.get_job_end_time("s2", "j4") == 14
    assert sched.get_job_end_time("s2", "j5") == 17

    # At s3: j3 starts at max(11, 6) = 11 (needs to wait for j3@S2 end at 11), so j3@S3 = 11+3=14
    # At s3: j4 starts at max(14, 14) = 14 (needs to wait for j4@S2 end at 14), so j4@S3 = 14+4=18
    # At s3: j5 starts at max(17, 18) = 18 (needs to wait for j5@S2 end at 17 and j4@S3 end at 18), so j5@S3 = 18+2=20
    assert sched.get_job_end_time("s3", "j3") == 14
    assert sched.get_job_end_time("s3", "j4") == 18
    assert sched.get_job_end_time("s3", "j5") == 20

    # Makespan should be 20
    assert sched.makespan == 20


# ============================================================================
# Tests for dispatch_job_sequence_by_stages()
# ============================================================================


def test_dispatch_job_sequence_by_stages_basic():
    """Test basic job sequence dispatch across all stages."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    job_2_stage_2_p = {
        "j1": {"s1": 2, "s2": 3, "s3": 1},
        "j2": {"s1": 3, "s2": 2, "s3": 2},
        "j3": {"s1": 1, "s2": 4, "s3": 3},
    }

    dispatch_job_sequence_by_stages(sched, ["j1", "j2", "j3"], job_2_stage_2_p)

    # j1: s1 ends at 2, s2 ends at 5, s3 ends at 6
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s3", "j1") == 6

    # j2: s1 starts at 2 ends at 5, s2 starts at 5 ends at 7, s3 starts at 7 ends at 9
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s3", "j2") == 9

    # j3: s1 starts at 5 ends at 6, s2 starts at 7 ends at 11, s3 starts at 11 ends at 14
    assert sched.get_job_end_time("s1", "j3") == 6
    assert sched.get_job_end_time("s2", "j3") == 11
    assert sched.get_job_end_time("s3", "j3") == 14

    assert sched.makespan == 14


def test_dispatch_job_sequence_by_stages_with_from_stage():
    """Test dispatch starting from a specific stage."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    job_2_stage_2_p = {
        "j1": {"s1": 2, "s2": 3, "s3": 1},
        "j2": {"s1": 3, "s2": 2, "s3": 2},
    }

    # Schedule both jobs from s2 onward (s1 is skipped)
    dispatch_job_sequence_by_stages(
        sched, ["j1", "j2"], job_2_stage_2_p, from_stage="s2"
    )

    # j1 at s2 starts at 0, ends at 3; at s3 ends at 4
    assert sched.get_job_end_time("s2", "j1") == 3
    assert sched.get_job_end_time("s3", "j1") == 4

    # j2 at s2 starts at 3, ends at 5; at s3 ends at 7
    assert sched.get_job_end_time("s2", "j2") == 5
    assert sched.get_job_end_time("s3", "j2") == 7


def test_dispatch_job_sequence_by_stages_with_release_time():
    """Test dispatch with job release times."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    job_2_stage_2_p = {
        "j1": {"s1": 2, "s2": 3},
        "j2": {"s1": 3, "s2": 2},
    }
    job_2_release_t = {"j1": 0, "j2": 5}

    dispatch_job_sequence_by_stages(
        sched, ["j1", "j2"], job_2_stage_2_p, job_2_release_t=job_2_release_t
    )

    # j1: s1 ends at 2, s2 ends at 5
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5

    # j2: release time is 5, s1 ends at max(5, 2) + 3 = 8, s2 ends at 8 + 2 = 10
    assert sched.get_job_end_time("s1", "j2") == 8
    assert sched.get_job_end_time("s2", "j2") == 10


def test_dispatch_job_sequence_by_stages_multiple_machines():
    """Test dispatch with multiple machines per stage."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1"]},
    )
    job_2_stage_2_p = {
        "j1": {"s1": 2, "s2": 3},
        "j2": {"s1": 3, "s2": 2},
        "j3": {"s1": 1, "s2": 1},
    }

    dispatch_job_sequence_by_stages(sched, ["j1", "j2", "j3"], job_2_stage_2_p)

    # At s1: j1 on m1 ends at 2, j2 on m2 ends at 3, j3 on m1 ends at 3
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 3
    assert sched.get_job_end_time("s1", "j3") == 3

    # At s2: all jobs use m1
    # Priority: j1 (end=2), j2 (end=3), j3 (end=3)
    # j1 starts at 2, ends at 5
    # j2 starts at max(3, 5) = 5, ends at 7
    # j3 starts at max(3, 7) = 7, ends at 8
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s2", "j3") == 8


# ============================================================================
# Tests for dispatch_stages_by_job_sequence()
# ============================================================================


def test_dispatch_stages_by_job_sequence_basic():
    """Test basic stage dispatch with job sequence."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 4},
        "s3": {"j1": 1, "j2": 2, "j3": 3},
    }

    dispatch_stages_by_job_sequence(sched, ["j1", "j2", "j3"], stage_2_job_2_p)

    # All stages scheduled sequentially for each job
    # j1: s1=2, s2=5, s3=6
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s3", "j1") == 6

    # j2: s1=5, s2=7, s3=9
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s3", "j2") == 9

    # j3: s1=6, s2=11, s3=14
    # At s2: j3 starts at max(6, 7) = 7 (wait for j2@S2 end at 7), ends at 11
    # At s3: j3 starts at max(11, 6) = 11 (wait for j3@S2 end at 11), ends at 14
    assert sched.get_job_end_time("s1", "j3") == 6
    assert sched.get_job_end_time("s2", "j3") == 11
    assert sched.get_job_end_time("s3", "j3") == 14

    assert sched.makespan == 14


def test_dispatch_stages_by_job_sequence_with_from_stage():
    """Test stage dispatch starting from a specific stage."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3},
        "s2": {"j1": 3, "j2": 2},
        "s3": {"j1": 1, "j2": 2},
    }

    # First dispatch at s1
    sched.dispatch_stage_by_jobs("s1", ["j1", "j2"], stage_2_job_2_p["s1"])

    # Then dispatch from s2 onward
    dispatch_stages_by_job_sequence(
        sched, ["j1", "j2"], stage_2_job_2_p, from_stage="s2"
    )

    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s3", "j1") == 6
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s3", "j2") == 9


def test_dispatch_stages_by_job_sequence_with_release_time():
    """Test stage dispatch with job release times."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3},
        "s2": {"j1": 3, "j2": 2},
    }
    job_2_release_t = {"j1": 0, "j2": 5}

    dispatch_stages_by_job_sequence(
        sched, ["j1", "j2"], stage_2_job_2_p, job_2_release_t=job_2_release_t
    )

    # j1: s1 ends at 2
    assert sched.get_job_end_time("s1", "j1") == 2

    # j2: release time is 5, s1 ends at max(5, 2) + 3 = 8
    assert sched.get_job_end_time("s1", "j2") == 8

    # j1: s2 starts at 2, ends at 5
    assert sched.get_job_end_time("s2", "j1") == 5

    # j2: s2 starts at max(8, 5) = 8, ends at 10
    assert sched.get_job_end_time("s2", "j2") == 10


def test_dispatch_stages_by_job_sequence_multiple_machines():
    """Test stage dispatch with multiple machines per stage."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1"]},
    )
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
        "s2": {"j1": 3, "j2": 2, "j3": 1},
    }

    dispatch_stages_by_job_sequence(sched, ["j1", "j2", "j3"], stage_2_job_2_p)

    # At s1: j1 on m1 ends at 2, j2 on m2 ends at 3, j3 on m1 ends at 3
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 3
    assert sched.get_job_end_time("s1", "j3") == 3

    # At s2: all jobs use m1, priority by s1 end times
    # j1 (end=2) -> ends at 5
    # j2 (end=3) -> starts at max(3, 5)=5, ends at 7
    # j3 (end=3) -> starts at max(3, 7)=7, ends at 8
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s2", "j3") == 8


def test_from_job_sequence_get_schedule_mixed_k_per_stage():
    """Test k_per_stage mapping for different k values at different stages."""
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

    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, k_per_stage
    )

    # j1 and j2 should be scheduled at all stages via dispatch_job_by_stages
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5

    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s2", "j2") == 7

    # j3 should be scheduled at all stages via dispatch_job_by_stages (k_per_stage["s2"]=3)
    assert sched.get_job_end_time("s1", "j3") == 7  # starts at 5, duration 2
    assert sched.get_job_end_time("s2", "j3") == 11  # starts at 7, duration 4

    # j4, j5 should be scheduled via dispatch_stage_by_jobs at each stage
    # For s1: remaining are j4, j5 (j1, j2, j3 already scheduled)
    assert sched.get_job_end_time("s1", "j4") == 12  # starts at 7, duration 5
    assert sched.get_job_end_time("s1", "j5") == 13  # starts at 12, duration 1

    # For s2: remaining are j4, j5 (j1, j2, j3 already scheduled)
    # Priority order: j4 (end=12), j5 (end=13)
    assert sched.get_job_end_time("s2", "j4") == 14  # starts at 12, duration 2
    assert sched.get_job_end_time("s2", "j5") == 17  # starts at 14, duration 3

    assert sched.makespan == 17


def test_from_job_sequence_get_schedule_mixed_validation():
    """Test validation of duration mappings."""
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
    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 3, "s2": 3}
    )

    # j1, j2, j3 should be scheduled at all stages via dispatch_job_by_stages
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5

    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s2", "j2") == 7

    assert sched.get_job_end_time("s1", "j3") == 7
    assert sched.get_job_end_time("s2", "j3") == 11

    # j4, j5 should be scheduled via dispatch_stage_by_jobs at each stage
    # For s1: remaining are j4, j5
    assert sched.get_job_end_time("s1", "j4") == 12
    assert sched.get_job_end_time("s1", "j5") == 13

    # For s2: remaining are j4, j5 (j1, j2, j3 already scheduled)
    assert sched.get_job_end_time("s2", "j4") == 14
    assert sched.get_job_end_time("s2", "j5") == 17

    assert sched.makespan == 17


def test_from_job_sequence_get_schedule_mixed_makespan():
    """Test makespan comparison between different k values."""
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
    from_job_sequence_get_schedule_mixed(
        sched_k1, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 1}
    )

    # Test with k_per_stage={"s1": 3}
    sched_k3 = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4", "j5"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    from_job_sequence_get_schedule_mixed(
        sched_k3, ["j1", "j2", "j3", "j4", "j5"], stage_2_duration, {"s1": 3}
    )

    # Makespans should be valid (no overlaps, precedence satisfied)
    assert sched_k1.makespan > 0
    assert sched_k3.makespan > 0

    # k=3 might give better or worse makespan depending on the instance
    # We just verify both are valid schedules
    validate_schedule(sched_k1, stage_2_duration)
    validate_schedule(sched_k3, stage_2_duration)


def test_from_job_sequence_get_schedule_mixed_single_job():
    """Test scheduling a single job."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2},
        "s2": {"j1": 3},
    }

    from_job_sequence_get_schedule_mixed(sched, ["j1"], stage_2_duration, {"s1": 1})

    # j1 should be scheduled at all stages via dispatch_job_by_stages
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.makespan == 5


def test_from_job_sequence_get_schedule_mixed_all_via_dispatch_job():
    """Test when all jobs are dispatched via dispatch_job_by_stages."""
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
    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3"], stage_2_duration, {"s1": 3}
    )

    # All jobs should be scheduled at all stages via dispatch_job_by_stages
    # s1: j1(0-2), j2(2-5), j3(5-6)
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s2", "j1") == 5  # s1 end + s2 duration = 2 + 3 = 5

    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s2", "j2") == 7  # s1 end + s2 duration = 5 + 2 = 7

    assert sched.get_job_end_time("s1", "j3") == 6
    assert (
        sched.get_job_end_time("s2", "j3") == 11
    )  # s1 end + s2 duration = 6 + 4 = 10... wait, 11?

    # The actual behavior: j1 ends at 5 on s2, j2 starts at 5 ends at 7, j3 starts at 7 ends at 11
    # This is because dispatch_job_by_stages schedules jobs sequentially on the same machine
    assert sched.makespan == 11


def test_from_job_sequence_get_schedule_mixed_empty_head():
    """Test when stage_2_head has 0 for all stages (all via dispatch_stage_by_jobs)."""
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
    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3"], stage_2_duration, {"s1": 0, "s2": 0}
    )

    # At s1: dispatch_stage_by_jobs schedules sequentially on same machine
    # j1 ends at 2, j2 ends at 5, j3 ends at 6
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s1", "j3") == 6

    # At s2: priority order is j1 (end=2), j3 (end=6), j2 (end=5)
    # j1 starts at 2, ends at 5
    # j3 starts at max(6, 5) = 6, ends at 10
    # j2 starts at max(5, 10) = 10, ends at 12
    # But actual behavior shows sequential dispatch: j1(2-5), j2(5-7), j3(7-11)
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s2", "j3") == 11

    assert sched.makespan == 11


def test_from_job_sequence_get_schedule_mixed_stages_with_multiple_machines():
    """Test mixed dispatch with multiple machines per stage."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3", "j4"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1", "m2"], "s2": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1, "j4": 2},
        "s2": {"j1": 3, "j2": 2, "j3": 4, "j4": 1},
    }

    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3", "j4"], stage_2_duration, {"s1": 2}
    )

    # j1 and j2 are dispatched via dispatch_job_by_stages at s1
    # j1 on m1: end=2, j2 on m2: end=3
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 3

    # j3, j4 dispatched via dispatch_stage_by_jobs at s1
    # j3 on m1 (next available after j1) ends at 3, j4 on m2 ends at 5
    assert sched.get_job_end_time("s1", "j3") == 3
    assert sched.get_job_end_time("s1", "j4") == 5

    # At s2, all jobs use the same machine m1
    # Priority order based on s1 end times (with tiebreaker by sequence order):
    # j1 (2), j2 (3), j3 (3), j4 (5)
    # j1 starts at 2, ends at 5
    # j2 starts at max(3, 5) = 5, ends at 7
    # j3 starts at max(3, 7) = 7, ends at 11
    # j4 starts at max(5, 11) = 11, ends at 12
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s2", "j3") == 11
    assert sched.get_job_end_time("s2", "j4") == 12

    assert sched.makespan == 12


def test_from_job_sequence_get_schedule_mixed_from_stage_empty_schedule():
    """Test from_stage works with empty schedule (start from a mid-stage)."""
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2", "s3"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"], "s3": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3},
        "s2": {"j1": 3, "j2": 2},
        "s3": {"j1": 2, "j2": 2},
    }

    # Schedule first stage manually
    sched.dispatch_stage_by_jobs("s1", ["j1", "j2"], stage_2_duration["s1"])

    # Continue scheduling from s2 using mixed dispatch
    # stage_2_head is applied cumulatively from s2 onward
    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2"], stage_2_duration, {"s2": 1}, from_stage="s2"
    )

    # All jobs should be scheduled
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 5

    # j1 dispatched via dispatch_job_by_stages through s2, s3
    # j1's s2 end time = s1_end + s2_duration = 2 + 3 = 5
    # j1's s3 end time = s2_end + s3_duration = 5 + 2 = 7
    assert sched.get_job_end_time("s2", "j1") == 5
    assert sched.get_job_end_time("s3", "j1") == 7

    # j2 dispatched via dispatch_stage_by_jobs at s2, s3 (priority based on precedence)
    # j2's s2 end time = s1_end + s2_duration = 5 + 2 = 7
    # j2's s3 end time = s2_end + s3_duration = 7 + 2 = 9
    assert sched.get_job_end_time("s2", "j2") == 7
    assert sched.get_job_end_time("s3", "j2") == 9

    assert sched.makespan == 9


def test_from_job_sequence_get_schedule_mixed_validation_invalid_head():
    """Test that invalid stage_2_head raises ValueError."""
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
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2", "j3"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    stage_2_duration = {
        "s1": {"j1": 2, "j2": 3, "j3": 1},
    }

    from_job_sequence_get_schedule_mixed(
        sched, ["j1", "j2", "j3"], stage_2_duration, {"s1": 2}
    )

    # j1 and j2 via dispatch_job_by_stages, j3 via dispatch_stage_by_jobs
    assert sched.get_job_end_time("s1", "j1") == 2
    assert sched.get_job_end_time("s1", "j2") == 5
    assert sched.get_job_end_time("s1", "j3") == 6

    assert sched.makespan == 6


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


def test_get_stage_job_sequences_from_dispatch_windows():
    dispatch_window_lookup = {
        (1, 1): {"early_start": 5.0, "late_start": 7.0},
        (1, 2): {"early_start": 1.0, "late_start": 9.0},
        (1, 3): {"early_start": 5.0, "late_start": 6.0},
        (2, 1): {"early_start": 3.0, "late_start": 4.0},
        (2, 2): {"early_start": 3.0, "late_start": 3.0},
        (2, 3): {"early_start": 8.0, "late_start": 8.0},
    }
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 4, "j3": 3},
        "s2": {"j1": 1, "j2": 5, "j3": 2},
    }

    stage_2_job_sequence = get_stage_job_sequences_from_dispatch_windows(
        stage_id_list=["s1", "s2"],
        job_id_list=["j1", "j2", "j3"],
        dispatch_window_lookup=dispatch_window_lookup,
        stage_2_job_2_p=stage_2_job_2_p,
    )

    assert stage_2_job_sequence["s1"] == ["j2", "j3", "j1"]
    assert stage_2_job_sequence["s2"] == ["j2", "j1", "j3"]


def test_dispatch_stage_job_sequences_strict_call_order():
    sched = HybridFlowshopLiteSchedule(
        jobs=["j1", "j2"],
        stages=["s1", "s2"],
        machines_per_stage={"s1": ["m1"], "s2": ["m1"]},
    )
    stage_2_job_sequence = {"s1": ["j2", "j1"], "s2": ["j1", "j2"]}
    stage_2_job_2_p = {
        "s1": {"j1": 2, "j2": 3},
        "s2": {"j1": 4, "j2": 1},
    }

    dispatch_stage_job_sequences_strict_call_order(
        sched,
        stage_2_job_sequence,
        stage_2_job_2_p,
    )

    assert sched.get_job_end_time("s1", "j2") == 3
    assert sched.get_job_end_time("s1", "j1") == 5
    assert sched.get_job_end_time("s2", "j1") == 9
    # strict_call_order preserves the call order, but later jobs may still be
    # inserted into earlier idle gaps if precedence allows.
    assert sched.get_job_end_time("s2", "j2") == 4
