from types import SimpleNamespace

from lb_bucket.mip.post_dispatch import (
    PostMipDispatchDependencies,
    _get_post_mip_selected_dispatch_method_list,
    _get_stage_adaptive_direct_mixed_repair_base_variant,
    _get_stage_adaptive_rank_repair_base_variant,
    _should_run_expensive_post_mip_repairs,
    _should_run_final_selected_post_mip_local_repair,
    run_post_mip_dispatch,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def get_jik_2_start_time_map(self):
        return {}


def _build_incumbent_schedule() -> HybridFlowshopLiteSchedule:
    schedule = HybridFlowshopLiteSchedule(
        ["j1", "j2"],
        ["s1", "s2"],
        {"s1": ["m1"], "s2": ["m2"]},
    )
    schedule.add_ops_times_2_mc("s1", "m1", "j1", start_time=0, end_time=1)
    schedule.add_ops_times_2_mc("s1", "m1", "j2", start_time=1, end_time=2)
    schedule.add_ops_times_2_mc("s2", "m2", "j1", start_time=1, end_time=2)
    schedule.add_ops_times_2_mc("s2", "m2", "j2", start_time=2, end_time=3)
    return schedule


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


def test_post_mip_selected_dispatch_method_list_keeps_bn2d_for_short_flowshops() -> None:
    assert _get_post_mip_selected_dispatch_method_list(
        ["bn2d_all_stages", "best_of_mixed_dispatches"],
        10,
    ) == ["bn2d_all_stages", "best_of_mixed_dispatches"]


def test_post_mip_selected_dispatch_method_list_drops_bn2d_for_deep_flowshops() -> None:
    assert _get_post_mip_selected_dispatch_method_list(
        ["bn2d_all_stages", "best_of_mixed_dispatches"],
        15,
    ) == ["best_of_mixed_dispatches"]


def test_expensive_post_mip_repairs_trigger_when_cheap_pool_does_not_improve() -> None:
    assert _should_run_expensive_post_mip_repairs(
        stage_count=10,
        incumbent_makespan=100.0,
        best_candidate_makespan=100.0,
    )


def test_expensive_post_mip_repairs_skip_after_meaningful_deep_improvement() -> None:
    assert not _should_run_expensive_post_mip_repairs(
        stage_count=20,
        incumbent_makespan=5000.0,
        best_candidate_makespan=4985.0,
    )


def test_final_selected_post_mip_local_repair_runs_for_non_improving_choice() -> None:
    assert _should_run_final_selected_post_mip_local_repair(
        stage_count=5,
        incumbent_makespan=100.0,
        selected_candidate_makespan=100.0,
    )


def test_final_selected_post_mip_local_repair_skips_after_clear_improvement() -> None:
    assert not _should_run_final_selected_post_mip_local_repair(
        stage_count=20,
        incumbent_makespan=5000.0,
        selected_candidate_makespan=4975.0,
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
        incumbent_schedule=None,
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


def test_run_post_mip_dispatch_builds_incumbent_guided_blend_candidate() -> None:
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

    result = run_post_mip_dispatch(
        instance=instance,
        stage_2_job_2_p_dict=stage_2_job_2_p_dict,
        solution_payload=None,
        dispatch_window_lookup=dispatch_window_lookup,
        incumbent_schedule=_build_incumbent_schedule(),
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
                "machine_then_job": False,
                "head_for_all_stages": False,
                "p_agg_method": "sum",
                "mi_agg_method": "max",
                "method_list": [],
            },
            get_best_mixed_schedule_from_job_sequence=lambda *_args, **_kwargs: None,
            get_selected_dispatch_candidate_schedules=lambda **_kwargs: {},
            get_schedule_by_best_of_mixed_dispatches=lambda **_kwargs: None,
            repair_post_mip_dispatch_candidate=lambda schedule, **_kwargs: schedule,
        ),
    )

    incumbent_guided = result.dispatched_schedules["incumbent_guided_priority_blend"]
    assert incumbent_guided is not None
    assert incumbent_guided.makespan >= 0
