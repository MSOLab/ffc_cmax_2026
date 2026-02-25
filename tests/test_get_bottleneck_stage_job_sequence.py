from types import SimpleNamespace


from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    get_bottleneck_stage_job_sequence,
)


def test_get_bottleneck_stage_job_sequence_orders_by_start_midpoint_and_job_index():
    instance = SimpleNamespace(
        job_id_list=["b", "a", "c", "d"],
        stage_id_list=["s1", "s2"],
    )

    schedule = HybridFlowshopLiteSchedule(
        jobs=instance.job_id_list,
        stages=instance.stage_id_list,
        machines_per_stage={"s1": ["m1"], "s2": ["m1", "m2", "m3", "m4"]},
    )

    # bottleneck stage(s2)에서의 시각 정보
    schedule.add_ops_times_2_mc("s2", "m1", "a", start_time=10, end_time=20)
    schedule.add_ops_times_2_mc("s2", "m2", "b", start_time=10, end_time=20)
    schedule.add_ops_times_2_mc("s2", "m3", "c", start_time=10, end_time=18)
    schedule.add_ops_times_2_mc("s2", "m4", "d", start_time=12, end_time=16)

    # 병목 stage 식별을 안정적으로 고정
    schedule.get_stage_2_mc_2_idle_time_map = lambda: {
        "s1": {"m1": 10},
        "s2": {"m1": 1, "m2": 1},
    }

    seq = get_bottleneck_stage_job_sequence(schedule)

    # 정렬 기준: start -> midpoint -> original job order index
    # c: (10, 14, 2), b: (10, 15, 0), a: (10, 15, 1), d: (12, 14, 3)
    assert seq == ["c", "b", "a", "d"]
