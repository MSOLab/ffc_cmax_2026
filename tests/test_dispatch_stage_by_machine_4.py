from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


def _make_base_schedule() -> HybridFlowshopLiteSchedule:
    jobs = ["X", "Y", "Z", "J1", "J2"]
    stages = ["S1", "S2"]
    machines_per_stage = {"S1": ["M1", "M2"], "S2": ["M1", "M2"]}
    sch = HybridFlowshopLiteSchedule(
        jobs=jobs, stages=stages, machines_per_stage=machines_per_stage
    )

    # Pre-existing operations on S1
    # (S1, M2) has an idle gap [3, 8)
    sch.add_ops_times_2_mc("S1", "M2", "X", 0, 3)
    sch.add_ops_times_2_mc("S1", "M2", "Y", 8, 12)

    # (S1, M1) is busy until 10
    sch.add_ops_times_2_mc("S1", "M1", "Z", 0, 10)

    return sch


def test_machine_centric_dispatch_4_inserts_into_existing_idle_gap_and_then_uses_other_machine():
    sch = _make_base_schedule()

    stage_2_job_2_p = {
        "S1": {"J1": 5, "J2": 2},
        "S2": {
            "J1": 10,
            "J2": 1,
        },  # remaining pt makes J1 higher priority on the first decision
    }

    sch.machine_centric_dispatch_4(
        stage_id="S1",
        job_id_seq=["J1", "J2"],
        stage_2_job_2_p=stage_2_job_2_p,
        job_2_release=None,
        spt_on_last_stage=False,
    )

    # Expect J1 to be inserted into M2's idle gap [3,8) => [3,8)
    assert sch.get_job_sequence("S1", "M2") == [
        (0, 3, "X"),
        (3, 8, "J1"),
        (8, 12, "Y"),
    ]

    # After that, the earliest feasible for J2 is on M1 at 10 (earlier than M2's next idle at 12)
    assert sch.get_job_sequence("S1", "M1") == [
        (0, 10, "Z"),
        (10, 12, "J2"),
    ]


def test_machine_centric_dispatch_4_respects_release_time_via_time_jump():
    sch = _make_base_schedule()

    stage_2_job_2_p = {
        "S1": {"J1": 2, "J2": 2},
        "S2": {"J1": 10, "J2": 1},
    }
    job_2_release = {"J1": 6, "J2": 0}

    sch.machine_centric_dispatch_4(
        stage_id="S1",
        job_id_seq=["J1", "J2"],
        stage_2_job_2_p=stage_2_job_2_p,
        job_2_release=job_2_release,
        spt_on_last_stage=False,
    )

    # J2 can run in the early idle window; J1 cannot be scheduled before release=6.
    # M2 gap is [3,8). Expected: J2 at [3,5), then time-jump to 6 and J1 at [6,8).
    assert sch.get_job_sequence("S1", "M2") == [
        (0, 3, "X"),
        (3, 5, "J2"),
        (6, 8, "J1"),
        (8, 12, "Y"),
    ]

    sch.machine_centric_dispatch_4(
        stage_id="S2",
        job_id_seq=["J1", "J2"],
        stage_2_job_2_p=stage_2_job_2_p,
        job_2_release=job_2_release,
        spt_on_last_stage=False,
    )

    assert sch.get_job_sequence("S2", "M1") == [
        (5, 6, "J2"),
        (8, 18, "J1"),
    ]
