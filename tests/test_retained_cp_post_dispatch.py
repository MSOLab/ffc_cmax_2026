from pathlib import Path
from types import SimpleNamespace

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule
from lb_bucket.cp.post_dispatch import (
    PostRetainedCpDispatchDependencies,
    PostRetainedCpDispatchRunResult,
    run_post_retained_cp_dispatch,
    write_post_retained_cp_dispatch_artifacts,
)
from lb_bucket.cp.search import RetainedStageCpResult


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def get_jik_2_start_time_map(self):
        return {"makespan": self.makespan}


def _make_retained_cp_result() -> RetainedStageCpResult:
    return RetainedStageCpResult(
        ins_name="demo",
        input_lb=10.0,
        input_ub=50.0,
        retained_stage_mode="first_topk_bottlenecks_last",
        retained_stage_ids=("s1", "s2", "s4", "s5"),
        bottleneck_stage_id="s4",
        selected_bottleneck_stage_ids=("s4",),
        bottleneck_band_radius=None,
        middle_band_radius=None,
        retained_stage_ratios=(),
        quantile_count=None,
        job_count=3,
        stage_count=5,
        machine_count_per_stage="1 1 1 1 1",
        objective_ub=42.0,
        objective_lb=30.0,
        certified_final_lb=30.0,
        status_name="FEASIBLE",
        time_limit_sec_used=1.0,
        solver_runtime_sec=0.8,
        wall_runtime_sec=0.8,
        model_build_wall_sec=0.1,
    )


def test_run_post_retained_cp_dispatch_generates_multiple_candidate_families() -> None:
    instance = SimpleNamespace(
        stage_id_list=["s1", "s2", "s3", "s4", "s5"],
        job_id_list=["j1", "j2", "j3"],
    )
    retained_solution_rows = [
        {
            "stage_id": "s2",
            "job_id": "j1",
            "start": 1,
            "end": 3,
            "processing_time": 2,
            "head": 0,
            "tail": 7,
        },
        {
            "stage_id": "s2",
            "job_id": "j2",
            "start": 3,
            "end": 5,
            "processing_time": 2,
            "head": 0,
            "tail": 6,
        },
        {
            "stage_id": "s2",
            "job_id": "j3",
            "start": 5,
            "end": 7,
            "processing_time": 2,
            "head": 0,
            "tail": 5,
        },
        {
            "stage_id": "s4",
            "job_id": "j2",
            "start": 10,
            "end": 12,
            "processing_time": 2,
            "head": 6,
            "tail": 2,
        },
        {
            "stage_id": "s4",
            "job_id": "j1",
            "start": 12,
            "end": 14,
            "processing_time": 2,
            "head": 7,
            "tail": 2,
        },
        {
            "stage_id": "s4",
            "job_id": "j3",
            "start": 14,
            "end": 16,
            "processing_time": 2,
            "head": 8,
            "tail": 2,
        },
    ]
    feasibility_calls: list[dict[str, int]] = []

    def _get_best_mixed_schedule_from_job_sequence(
        job_sequence, *, machine_then_job=False, head_for_all_stages=False
    ):
        score = 96 if job_sequence[0] == "j1" else 94
        return _FakeSchedule(score - int(machine_then_job) - int(head_for_all_stages))

    def _get_schedule_by_best_of_mixed_dispatches(
        *, machine_then_job=False, head_for_all_stages=False, job_tiebreak_rank=None
    ):
        if job_tiebreak_rank is None:
            return _FakeSchedule(98)
        if job_tiebreak_rank.get("j2", 99) == 0:
            return _FakeSchedule(89)
        return _FakeSchedule(91)

    def _get_two_way_schedule_by_stage_band(
        *,
        anchor_stage_ids,
        stage_2_job_sequence,
        mixed_schedule_for_former_stages,
        mixed_schedule_for_later_stages,
        machine_then_job,
        stage_2_job_2_release,
        anchor_dispatch_mode,
    ):
        del mixed_schedule_for_former_stages
        del mixed_schedule_for_later_stages
        del machine_then_job
        del stage_2_job_sequence
        base = 100 if anchor_stage_ids == ["s4"] else 95
        if len(anchor_stage_ids) > 2:
            base += 8
        if anchor_dispatch_mode == "strict_call":
            base -= 7
        elif anchor_dispatch_mode == "priority":
            base -= 9
        elif anchor_dispatch_mode == "strict_start":
            base -= 4
        if stage_2_job_2_release is None:
            base -= 3
        return _FakeSchedule(base)

    def _get_schedule_by_stage_job_sequences_priority(
        *, stage_2_job_sequence, stage_2_job_2_release=None
    ):
        del stage_2_job_sequence, stage_2_job_2_release
        return _FakeSchedule(90)

    def _repair_post_retained_cp_dispatch_candidate(schedule, **_kwargs):
        return _FakeSchedule(schedule.makespan - 2)

    result = run_post_retained_cp_dispatch(
        instance=instance,
        retained_cp_result=_make_retained_cp_result(),
        retained_solution_rows=retained_solution_rows,
        cp_local_repair_max_passes=1,
        include_release_anchor_candidates=True,
        include_consensus_rank=True,
        include_tail_bottleneck_rank=True,
        include_extended_rank_variants=True,
        include_dynamic_priority=False,
        include_piecewise_stage_priority=True,
        prune_unproductive_dispatch_candidates=False,
        dependencies=PostRetainedCpDispatchDependencies(
            check_feasibility=lambda start_map: feasibility_calls.append(start_map),
            get_selected_dispatch_config=lambda: {
                "mixed_schedule_for_former_stages": True,
                "mixed_schedule_for_later_stages": True,
                "machine_then_job": True,
                "head_for_all_stages": True,
            },
            get_best_mixed_schedule_from_job_sequence=_get_best_mixed_schedule_from_job_sequence,
            get_schedule_by_best_of_mixed_dispatches=_get_schedule_by_best_of_mixed_dispatches,
            get_two_way_schedule_by_stage_band=_get_two_way_schedule_by_stage_band,
            get_schedule_by_stage_job_sequences_priority=_get_schedule_by_stage_job_sequences_priority,
            repair_post_retained_cp_dispatch_candidate=_repair_post_retained_cp_dispatch_candidate,
        ),
    )

    assert result.selected_dispatch_variant == "selected_post_retained_cp_local_repair"
    assert result.dispatched_schedule is not None
    assert result.dispatched_schedule.makespan == 87
    assert "cp_band_preferred_strict_start_release" in result.dispatched_schedules
    assert "cp_band_preferred_strict_start_no_release" not in result.dispatched_schedules
    assert "cp_band_preferred_priority_release" in result.dispatched_schedules
    assert "cp_band_first_strict_start_release" in result.dispatched_schedules
    assert "mixed_cp_aggregate_start_slack" in result.dispatched_schedules
    assert "mixed_cp_consensus" in result.dispatched_schedules
    assert "mixed_cp_tail_bottleneck" not in result.dispatched_schedules
    assert "best_of_mixed_dispatches_cp_consensus_rank" in result.dispatched_schedules
    assert (
        "best_of_mixed_dispatches_cp_tail_bottleneck_rank"
        in result.dispatched_schedules
    )
    assert "cp_dynamic_priority_soft_release" not in result.dispatched_schedules
    assert (
        "best_of_mixed_dispatches_cp_aggregate_start_slack_rank"
        in result.dispatched_schedules
    )
    assert "best_of_mixed_dispatches_cp_slack_urgency_rank" in result.dispatched_schedules
    assert "piecewise_cp_nearest_priority" in result.dispatched_schedules
    assert "piecewise_cp_blend_priority" in result.dispatched_schedules
    assert feasibility_calls == [{"makespan": 87}]


def test_run_post_retained_cp_dispatch_prunes_unproductive_candidates_by_default() -> None:
    instance = SimpleNamespace(
        stage_id_list=["s1", "s2", "s3", "s4", "s5"],
        job_id_list=["j1", "j2", "j3"],
    )
    retained_solution_rows = [
        {
            "stage_id": stage_id,
            "job_id": job_id,
            "start": start,
            "end": start + 2,
            "processing_time": 2,
            "head": 0,
            "tail": 5,
        }
        for stage_id, rows in {
            "s2": [("j1", 1), ("j2", 3), ("j3", 5)],
            "s4": [("j2", 10), ("j1", 12), ("j3", 14)],
        }.items()
        for job_id, start in rows
    ]

    result = run_post_retained_cp_dispatch(
        instance=instance,
        retained_cp_result=_make_retained_cp_result(),
        retained_solution_rows=retained_solution_rows,
        cp_local_repair_max_passes=0,
        include_release_anchor_candidates=True,
        include_consensus_rank=True,
        include_tail_bottleneck_rank=True,
        include_extended_rank_variants=True,
        include_dynamic_priority=False,
        include_piecewise_stage_priority=True,
        dependencies=PostRetainedCpDispatchDependencies(
            check_feasibility=lambda _start_map: None,
            get_selected_dispatch_config=lambda: {
                "mixed_schedule_for_former_stages": True,
                "mixed_schedule_for_later_stages": True,
                "machine_then_job": True,
                "head_for_all_stages": True,
            },
            get_best_mixed_schedule_from_job_sequence=lambda *_args, **_kwargs: _FakeSchedule(
                94
            ),
            get_schedule_by_best_of_mixed_dispatches=lambda **_kwargs: _FakeSchedule(
                91
            ),
            get_two_way_schedule_by_stage_band=lambda **_kwargs: _FakeSchedule(97),
            get_schedule_by_stage_job_sequences_priority=lambda **_kwargs: _FakeSchedule(
                90
            ),
            repair_post_retained_cp_dispatch_candidate=lambda schedule, **_kwargs: schedule,
        ),
    )

    variants = set(result.dispatched_schedules)
    assert "best_of_mixed_dispatches_cp_aggregate_start_slack_rank" in variants
    assert "best_of_mixed_dispatches_cp_weighted_median_rank" not in variants
    assert "best_of_mixed_dispatches_cp_front_tail_blend_rank" not in variants
    assert "best_of_mixed_dispatches_cp_last_anchor_rank" not in variants
    assert "best_of_mixed_dispatches_cp_baseline" not in variants
    assert "piecewise_cp_nearest_priority" not in variants
    assert "piecewise_cp_blend_priority" not in variants


def test_run_post_retained_cp_dispatch_repairs_top_k_candidates() -> None:
    instance = SimpleNamespace(
        stage_id_list=["s1", "s2", "s3", "s4", "s5"],
        job_id_list=["j1", "j2", "j3"],
    )
    retained_solution_rows = [
        {
            "stage_id": stage_id,
            "job_id": job_id,
            "start": start,
            "end": start + 2,
            "processing_time": 2,
            "head": 0,
            "tail": 5,
        }
        for stage_id, rows in {
            "s2": [("j1", 1), ("j2", 3), ("j3", 5)],
            "s4": [("j2", 10), ("j1", 12), ("j3", 14)],
        }.items()
        for job_id, start in rows
    ]

    def _get_best_mixed_schedule_from_job_sequence(
        job_sequence, *, machine_then_job=False, head_for_all_stages=False
    ):
        del job_sequence, machine_then_job, head_for_all_stages
        return _FakeSchedule(94)

    def _get_schedule_by_best_of_mixed_dispatches(
        *, machine_then_job=False, head_for_all_stages=False, job_tiebreak_rank=None
    ):
        del machine_then_job, head_for_all_stages
        if job_tiebreak_rank is None:
            return _FakeSchedule(98)
        if job_tiebreak_rank.get("j2", 99) == 0:
            return _FakeSchedule(89)
        return _FakeSchedule(91)

    def _get_two_way_schedule_by_stage_band(**_kwargs):
        return _FakeSchedule(97)

    def _get_schedule_by_stage_job_sequences_priority(**_kwargs):
        return _FakeSchedule(96)

    def _repair_post_retained_cp_dispatch_candidate(schedule, **_kwargs):
        if schedule.makespan == 91:
            return _FakeSchedule(80)
        return _FakeSchedule(schedule.makespan - 1)

    result = run_post_retained_cp_dispatch(
        instance=instance,
        retained_cp_result=_make_retained_cp_result(),
        retained_solution_rows=retained_solution_rows,
        cp_local_repair_max_passes=1,
        cp_local_repair_top_k=6,
        include_release_anchor_candidates=True,
        include_consensus_rank=True,
        include_tail_bottleneck_rank=True,
        include_dynamic_priority=False,
        dependencies=PostRetainedCpDispatchDependencies(
            check_feasibility=lambda _start_map: None,
            get_selected_dispatch_config=lambda: {
                "mixed_schedule_for_former_stages": True,
                "mixed_schedule_for_later_stages": True,
                "machine_then_job": True,
                "head_for_all_stages": True,
            },
            get_best_mixed_schedule_from_job_sequence=_get_best_mixed_schedule_from_job_sequence,
            get_schedule_by_best_of_mixed_dispatches=_get_schedule_by_best_of_mixed_dispatches,
            get_two_way_schedule_by_stage_band=_get_two_way_schedule_by_stage_band,
            get_schedule_by_stage_job_sequences_priority=_get_schedule_by_stage_job_sequences_priority,
            repair_post_retained_cp_dispatch_candidate=_repair_post_retained_cp_dispatch_candidate,
        ),
    )

    assert result.dispatched_schedule is not None
    assert result.dispatched_schedule.makespan == 80
    assert str(result.selected_dispatch_variant).startswith(
        "post_retained_cp_local_repair__"
    )


def test_write_post_retained_cp_dispatch_artifacts_writes_expected_files(
    tmp_path: Path,
) -> None:
    schedule = HybridFlowshopLiteSchedule(
        jobs=["1", "2"],
        stages=["1", "2"],
        machines_per_stage={"1": ["m1"], "2": ["m1"]},
    )
    schedule.dispatch_stage_by_jobs_strict_sequence("1", ["1", "2"], {"1": 1, "2": 1})
    schedule.dispatch_stage_by_jobs_strict_sequence("2", ["1", "2"], {"1": 1, "2": 1})

    result = PostRetainedCpDispatchRunResult(
        dispatched_schedule=schedule,
        selected_dispatch_variant="cp_band_preferred_strict_start_release",
        dispatched_schedules={"cp_band_preferred_strict_start_release": schedule},
        dispatch_candidate_elapsed_sec={
            "cp_band_preferred_strict_start_release": 0.1
        },
        dispatch_phase_elapsed_sec={"total_post_retained_cp_dispatch_sec": 0.2},
        anchor_blocks={"preferred": ["1"]},
        anchor_stage_sequences={"preferred": {"1": ["1", "2"]}},
        anchor_stage_releases={"preferred": {"1": {"1": 0, "2": 1}}},
        variant_2_anchor_stage_ids={
            "cp_band_preferred_strict_start_release": ["1"],
        },
        pre_local_repair_selected_dispatch_variant="cp_band_preferred_strict_start_release",
        pre_local_repair_selected_dispatch_makespan=4.0,
    )

    write_post_retained_cp_dispatch_artifacts(
        cp_lb_dir=tmp_path,
        dispatch_result=result,
        retained_cp_result=_make_retained_cp_result(),
        apply_elapsed_sec=1.0,
        dispatch_elapsed_sec=0.2,
        draw_visualizations=False,
    )

    assert (tmp_path / "dispatch" / "dispatch_summary.yaml").is_file()
    assert (tmp_path / "dispatch" / "dispatch_variant_objectives.csv").is_file()
    assert (tmp_path / "dispatch" / "dispatch_variant_manifest.yaml").is_file()
    assert (tmp_path / "dispatch" / "dispatch_anchor_blocks.yaml").is_file()
    assert (
        tmp_path / "dispatch" / "anchor_preferred_stage_job_sequence.yaml"
    ).is_file()
    assert (
        tmp_path / "dispatch" / "anchor_preferred_stage_job_release.yaml"
    ).is_file()
    assert (
        tmp_path / "dispatch" / "cp_band_preferred_strict_start_release_solution.yaml"
    ).is_file()
    assert not (tmp_path / "dispatch" / "gantt").exists()
