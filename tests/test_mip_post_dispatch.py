from types import SimpleNamespace

from lb_bucket.mip.post_dispatch import (
    PostMipDispatchDependencies,
    _get_stage_adaptive_direct_mixed_repair_base_variant,
    _get_stage_adaptive_rank_repair_base_variant,
    run_post_mip_dispatch,
)


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def get_jik_2_start_time_map(self):
        return {}


def test_stage_adaptive_rank_repair_prefers_aggregate_ls_for_short_flowshops() -> None:
    assert (
        _get_stage_adaptive_rank_repair_base_variant(5)
        == "best_of_mixed_dispatches_aggregate_ls_slack_rank"
    )
    assert (
        _get_stage_adaptive_rank_repair_base_variant(10)
        == "best_of_mixed_dispatches_aggregate_ls_slack_rank"
    )


def test_stage_adaptive_rank_repair_prefers_tail_rank_for_deep_flowshops() -> None:
    assert (
        _get_stage_adaptive_rank_repair_base_variant(15)
        == "best_of_mixed_dispatches_tail_ls_rank"
    )
    assert (
        _get_stage_adaptive_rank_repair_base_variant(20)
        == "best_of_mixed_dispatches_tail_ls_rank"
    )


def test_stage_adaptive_direct_mixed_repair_uses_aggregate_es_seed() -> None:
    assert _get_stage_adaptive_direct_mixed_repair_base_variant() == (
        "mixed_aggregate_es_slack"
    )


def test_run_post_mip_dispatch_calls_direct_mixed_with_keyword_only_args() -> None:
    instance = SimpleNamespace(
        stage_id_list=["s1", "s2"],
        job_id_list=["j1", "j2"],
        stage_2_machines_map={"s1": ["m1"], "s2": ["m2"]},
        stage_count=2,
    )
    dispatch_window_lookup = {
        (1, 1): {"early_start": 0.0, "late_start": 1.0},
        (1, 2): {"early_start": 1.0, "late_start": 2.0},
        (2, 1): {"early_start": 2.0, "late_start": 3.0},
        (2, 2): {"early_start": 3.0, "late_start": 4.0},
    }
    stage_2_job_2_p_dict = {"s1": {"j1": 1, "j2": 1}, "s2": {"j1": 1, "j2": 1}}
    direct_calls: list[tuple[list[str], bool, bool]] = []

    def _get_best_mixed_schedule_from_job_sequence(
        job_sequence, *, machine_then_job=False, head_for_all_stages=False
    ):
        direct_calls.append(
            (list(job_sequence), machine_then_job, head_for_all_stages)
        )
        return _FakeSchedule(10)

    result = run_post_mip_dispatch(
        instance=instance,
        stage_2_job_2_p_dict=stage_2_job_2_p_dict,
        solution_payload=None,
        dispatch_window_lookup=dispatch_window_lookup,
        es_ls_local_repair_max_passes=1,
        dependencies=PostMipDispatchDependencies(
            check_feasibility=lambda _start_map: 0.0,
            get_selected_dispatch_config=lambda: {
                "left_cap_multiplier": None,
                "right_cap_multiplier": None,
                "left_cap_portion": None,
                "right_cap_portion": None,
                "normalize_by_stage_cnt": False,
                "randomize_mid_all": False,
                "reverse_mid_all": False,
                "reverse_mid_even": False,
                "mixed_schedule_for_former_stages": False,
                "mixed_schedule_for_later_stages": False,
                "machine_then_job": True,
                "head_for_all_stages": True,
                "p_agg_method": "sum",
                "mi_agg_method": "max",
                "method_list": [],
            },
            get_best_mixed_schedule_from_job_sequence=_get_best_mixed_schedule_from_job_sequence,
            get_selected_dispatch_candidate_schedules=lambda **_kwargs: {},
            get_schedule_by_best_of_mixed_dispatches=lambda **_kwargs: None,
            repair_post_mip_dispatch_candidate=lambda schedule, **_kwargs: schedule,
        ),
    )

    assert len(direct_calls) == 2
    assert all(call[1:] == (True, True) for call in direct_calls)
    assert result.dispatched_schedules["mixed_aggregate_es_slack"].makespan == 10
