from types import SimpleNamespace

from lb_bucket.mip.post_dispatch import (
    PostMipDispatchDependencies,
    _build_weighted_job_tiebreak_rank,
    _get_post_mip_method_list,
    run_post_mip_dispatch,
)


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def get_jik_2_start_time_map(self):
        return {"makespan": self.makespan}


def test_get_post_mip_method_list_filters_out_bn2d_all_stages() -> None:
    assert _get_post_mip_method_list(
        [
            "bn2d_all_stages",
            "best_of_mixed_dispatches",
            "stage_agg_2",
        ]
    ) == [
        "best_of_mixed_dispatches",
        "stage_agg_2",
    ]
    assert _get_post_mip_method_list(None) == []


def test_run_post_mip_dispatch_filters_removed_variants_and_checks_feasibility_once() -> (
    None
):
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
    stage_2_job_2_p_dict = {
        "s1": {"j1": 1, "j2": 1},
        "s2": {"j1": 1, "j2": 1},
    }
    direct_calls: list[tuple[list[str], bool, bool]] = []
    selected_method_lists: list[list[str]] = []
    feasibility_start_maps: list[dict[str, int]] = []

    def _get_best_mixed_schedule_from_job_sequence(
        job_sequence, *, machine_then_job=False, head_for_all_stages=False
    ):
        direct_calls.append(
            (list(job_sequence), machine_then_job, head_for_all_stages)
        )
        if job_sequence[0] == "j1":
            return _FakeSchedule(10)
        return _FakeSchedule(9)

    def _get_selected_dispatch_candidate_schedules(**kwargs):
        method_list = list(kwargs["method_list"])
        selected_method_lists.append(method_list)
        return {method: _FakeSchedule(8) for method in method_list}

    def _get_schedule_by_best_of_mixed_dispatches(**kwargs):
        rank = kwargs["job_tiebreak_rank"]
        if rank["j1"] < rank["j2"]:
            return _FakeSchedule(7)
        return _FakeSchedule(6)

    def _repair_post_mip_dispatch_candidate(schedule, **_kwargs):
        return _FakeSchedule(schedule.makespan + 5)

    result = run_post_mip_dispatch(
        instance=instance,
        stage_2_job_2_p_dict=stage_2_job_2_p_dict,
        solution_payload=None,
        dispatch_window_lookup=dispatch_window_lookup,
        es_ls_local_repair_max_passes=1,
        dependencies=PostMipDispatchDependencies(
            check_feasibility=lambda start_map: feasibility_start_maps.append(start_map),
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
                "method_list": ["bn2d_all_stages", "best_of_mixed_dispatches"],
            },
            get_best_mixed_schedule_from_job_sequence=_get_best_mixed_schedule_from_job_sequence,
            get_selected_dispatch_candidate_schedules=_get_selected_dispatch_candidate_schedules,
            get_schedule_by_best_of_mixed_dispatches=_get_schedule_by_best_of_mixed_dispatches,
            repair_post_mip_dispatch_candidate=_repair_post_mip_dispatch_candidate,
        ),
    )

    assert len(direct_calls) == 2
    assert all(call[1:] == (True, True) for call in direct_calls)
    assert selected_method_lists == [["best_of_mixed_dispatches"]]
    assert "bn2d_all_stages" not in result.dispatched_schedules
    assert result.dispatched_schedules["es_ls_stage_priority_release"] is not None
    assert result.dispatched_schedules["es_ls_stage_strict_call_release"] is not None
    assert (
        result.dispatched_schedules["es_ls_stage_strict_lexicographic_release"]
        is not None
    )
    assert result.dispatched_schedules["es_ls_stage_strict_start_release"] is not None
    assert (
        result.dispatched_schedules[
            "best_of_mixed_dispatches_bottleneck_aggregate_es_rank"
        ]
        is not None
    )
    assert (
        result.dispatched_schedules[
            "best_of_mixed_dispatches_bottleneck_aggregate_ls_rank"
        ]
        is not None
    )
    assert result.stage_2_job_release == {
        "s1": {"j1": 0, "j2": 1},
        "s2": {"j1": 2, "j2": 3},
    }
    assert result.selected_dispatch_variant == "es_ls_stage_priority_release"
    assert result.dispatched_schedule is not None
    assert result.dispatched_schedule.makespan == 4
    assert result.dispatched_schedules["selected_post_mip_local_repair"].makespan == 9
    assert len(feasibility_start_maps) == 1
    assert any(op[0] == "j1" for op in feasibility_start_maps[0])

def test_build_weighted_job_tiebreak_rank_prefers_weighted_consensus_then_components() -> (
    None
):
    rank = _build_weighted_job_tiebreak_rank(
        job_id_list=["j1", "j2", "j3"],
        sequence_name_2_job_sequence={
            "bottleneck": ["j2", "j1", "j3"],
            "aggregate_es": ["j1", "j3", "j2"],
        },
        weights={"bottleneck": 2, "aggregate_es": 1},
    )

    assert rank == {"j2": 0, "j1": 1, "j3": 2}
