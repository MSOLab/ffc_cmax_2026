import inspect
from pathlib import Path
from types import SimpleNamespace

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule
from lb_bucket.mip.post_dispatch import (
    PostMipDispatchDependencies,
    PostMipDispatchRunResult,
    _get_post_mip_method_list,
    run_post_mip_dispatch,
    write_post_mip_dispatch_artifacts,
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


def test_mip_lb_entry_points_disable_local_repair_by_default() -> None:
    apply_signature = inspect.signature(HybridFlowShopCpLnsController.apply_mip_lb)
    dispatch_signature = inspect.signature(
        HybridFlowShopCpLnsController.dispatch_from_saved_mip_lb
    )

    assert apply_signature.parameters["es_ls_local_repair_max_passes"].default == 0
    assert dispatch_signature.parameters["es_ls_local_repair_max_passes"].default == 0


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
        direct_calls.append((list(job_sequence), machine_then_job, head_for_all_stages))
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
            check_feasibility=lambda start_map: feasibility_start_maps.append(
                start_map
            ),
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
    assert (
        "best_of_mixed_dispatches_aggregate_es_slack_rank"
        not in result.dispatched_schedules
    )
    assert (
        "best_of_mixed_dispatches_aggregate_ls_slack_rank"
        not in result.dispatched_schedules
    )
    assert "mixed_aggregate_es_slack_local_repair" not in result.dispatched_schedules
    assert (
        "best_of_mixed_dispatches_stage_adaptive_rank_local_repair"
        not in result.dispatched_schedules
    )
    assert result.selected_dispatch_variant in result.dispatched_schedules
    assert result.dispatched_schedule is not None
    assert result.dispatched_schedule.makespan == 7
    assert result.dispatched_schedules["selected_post_mip_local_repair"].makespan == 12
    assert feasibility_start_maps == [{"makespan": 7}]


def test_write_post_mip_dispatch_artifacts_writes_release_yaml_and_gantt(
    tmp_path: Path,
) -> None:
    schedule = HybridFlowshopLiteSchedule(
        jobs=["1", "2"],
        stages=["1", "2"],
        machines_per_stage={"1": ["m1"], "2": ["m1"]},
    )
    schedule.dispatch_stage_by_jobs_strict_sequence(
        "1",
        ["1", "2"],
        {"1": 1, "2": 1},
    )
    schedule.dispatch_stage_by_jobs_strict_sequence(
        "2",
        ["1", "2"],
        {"1": 1, "2": 1},
    )

    result = PostMipDispatchRunResult(
        dispatched_schedule=schedule,
        selected_dispatch_variant="best_of_mixed_dispatches_tail_ls_rank",
        dispatched_schedules={"best_of_mixed_dispatches_tail_ls_rank": schedule},
        dispatch_candidate_elapsed_sec={
            "best_of_mixed_dispatches_tail_ls_rank": 0.1
        },
        dispatch_phase_elapsed_sec={"total_post_mip_dispatch_sec": 0.2},
        stage_2_job_sequence={"1": ["1", "2"], "2": ["1", "2"]},
        stage_2_job_release={"1": {"1": 0, "2": 1}, "2": {"1": 1, "2": 2}},
        pre_local_repair_selected_dispatch_variant="best_of_mixed_dispatches_tail_ls_rank",
        pre_local_repair_selected_dispatch_makespan=4.0,
    )
    solution_payload = {
        "metadata": {
            "ins_name": "1",
            "job_ids": ["1", "2"],
            "stage_ids": ["1", "2"],
            "delta": 10,
            "stage_count": 2,
            "dispatch_cmax": 4.0,
        },
        "dispatch_windows": [
            {
                "stage": 1,
                "job": 1,
                "processing_time": 1,
                "a_bucket": 1,
                "b_bucket": 1,
                "x_bucket_1": 1,
                "x_value_1": 1.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 1.0,
                "x_at_b_bucket": 1.0,
                "spans_two_buckets": False,
                "es_candidate": 0.0,
                "completion_candidate": 1.0,
                "early_start": 0.0,
                "late_start": 0.0,
                "slack": 0.0,
            },
            {
                "stage": 2,
                "job": 1,
                "processing_time": 1,
                "a_bucket": 1,
                "b_bucket": 1,
                "x_bucket_1": 1,
                "x_value_1": 1.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 1.0,
                "x_at_b_bucket": 1.0,
                "spans_two_buckets": False,
                "es_candidate": 1.0,
                "completion_candidate": 2.0,
                "early_start": 1.0,
                "late_start": 1.0,
                "slack": 0.0,
            },
            {
                "stage": 1,
                "job": 2,
                "processing_time": 1,
                "a_bucket": 1,
                "b_bucket": 1,
                "x_bucket_1": 1,
                "x_value_1": 1.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 1.0,
                "x_at_b_bucket": 1.0,
                "spans_two_buckets": False,
                "es_candidate": 1.0,
                "completion_candidate": 2.0,
                "early_start": 1.0,
                "late_start": 1.0,
                "slack": 0.0,
            },
            {
                "stage": 2,
                "job": 2,
                "processing_time": 1,
                "a_bucket": 1,
                "b_bucket": 1,
                "x_bucket_1": 1,
                "x_value_1": 1.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 1.0,
                "x_at_b_bucket": 1.0,
                "spans_two_buckets": False,
                "es_candidate": 2.0,
                "completion_candidate": 3.0,
                "early_start": 2.0,
                "late_start": 2.0,
                "slack": 0.0,
            },
        ],
    }

    write_post_mip_dispatch_artifacts(
        output_dir=tmp_path,
        dispatch_result=result,
        solution_payload=solution_payload,
        mip_result=None,
        apply_mip_lb_elapsed_sec=1.0,
        post_mip_dispatch_elapsed_sec=0.2,
    )

    assert (tmp_path / "dispatch" / "dispatch_summary.yaml").is_file()
    assert (tmp_path / "dispatch" / "dispatch_variant_objectives.csv").is_file()
    assert (tmp_path / "dispatch" / "dispatch_variant_manifest.yaml").is_file()
    assert (tmp_path / "dispatch" / "es_ls_stage_job_sequence.yaml").is_file()
    assert (tmp_path / "dispatch" / "es_ls_stage_job_release.yaml").is_file()
    assert (tmp_path / "dispatch" / "es_ls_stage_job_latest_start.yaml").is_file()
    assert (
        tmp_path / "dispatch" / "gantt" / "best_of_mixed_dispatches_tail_ls_rank.png"
    ).is_file()
    overlay_root = tmp_path / "dispatch" / "dispatch_window_overlays"
    overlay_pngs = list(overlay_root.glob("*/*.png"))
    assert overlay_pngs
    overlay_info_files = list(overlay_root.glob("*/variant_info.yaml"))
    assert overlay_info_files


def test_write_post_mip_dispatch_artifacts_can_skip_visualizations(
    tmp_path: Path,
) -> None:
    schedule = HybridFlowshopLiteSchedule(
        jobs=["1"],
        stages=["1"],
        machines_per_stage={"1": ["m1"]},
    )
    schedule.dispatch_stage_by_jobs_strict_sequence(
        "1",
        ["1"],
        {"1": 1},
    )

    result = PostMipDispatchRunResult(
        dispatched_schedule=schedule,
        selected_dispatch_variant="best_of_mixed_dispatches",
        dispatched_schedules={"best_of_mixed_dispatches": schedule},
        dispatch_candidate_elapsed_sec={"best_of_mixed_dispatches": 0.1},
        dispatch_phase_elapsed_sec={"total_post_mip_dispatch_sec": 0.2},
        stage_2_job_sequence={"1": ["1"]},
        stage_2_job_release={"1": {"1": 0}},
        pre_local_repair_selected_dispatch_variant="best_of_mixed_dispatches",
        pre_local_repair_selected_dispatch_makespan=1.0,
    )
    solution_payload = {
        "metadata": {
            "ins_name": "1",
            "job_ids": ["1"],
            "stage_ids": ["1"],
            "delta": 10,
            "stage_count": 1,
            "dispatch_cmax": 1.0,
        },
        "dispatch_windows": [
            {
                "stage": 1,
                "job": 1,
                "processing_time": 1,
                "a_bucket": 1,
                "b_bucket": 1,
                "x_bucket_1": 1,
                "x_value_1": 1.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 1.0,
                "x_at_b_bucket": 1.0,
                "spans_two_buckets": False,
                "es_candidate": 0.0,
                "completion_candidate": 1.0,
                "early_start": 0.0,
                "late_start": 0.0,
                "slack": 0.0,
            },
        ],
    }

    write_post_mip_dispatch_artifacts(
        output_dir=tmp_path,
        dispatch_result=result,
        solution_payload=solution_payload,
        mip_result=None,
        apply_mip_lb_elapsed_sec=1.0,
        post_mip_dispatch_elapsed_sec=0.2,
        draw_visualizations=False,
    )

    assert (tmp_path / "dispatch" / "dispatch_summary.yaml").is_file()
    assert (tmp_path / "dispatch" / "dispatch_variant_objectives.csv").is_file()
    assert (tmp_path / "dispatch" / "best_of_mixed_dispatches_solution.yaml").is_file()
    assert (tmp_path / "dispatch" / "es_ls_stage_job_release.yaml").is_file()
    assert not (tmp_path / "dispatch" / "gantt").exists()
    assert not (tmp_path / "dispatch" / "dispatch_window_overlays").exists()
