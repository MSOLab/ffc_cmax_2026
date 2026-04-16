from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from routix import ElapsedTimer
from routix.io.yaml import dump_yaml
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.dispatcher.utils import (
    get_bottleneck_anchor_stage_from_solution_payload,
    get_job_sequence_from_dispatch_windows_aggregate,
    get_job_sequence_from_dispatch_windows_anchor_stage,
    get_job_tiebreak_rank_from_job_sequence,
    get_job_tiebreak_rank_from_stage_job_sequences,
    get_stage_job_latest_start_times_from_dispatch_windows,
    get_stage_job_release_times_from_dispatch_windows,
    get_stage_job_sequences_from_dispatch_windows,
)
from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, OperationType
from lb_bucket.mip.dispatch_windows import build_dispatch_window_lookup
from lb_bucket.mip.shared import BucketSearchResult
from lb_bucket.mip.visualization import write_dispatch_schedule_window_visualizations


@dataclass(frozen=True)
class PostMipDispatchDependencies:
    check_feasibility: Callable[[dict[OperationType, int]], float]
    get_selected_dispatch_config: Callable[[], dict[str, Any]]
    get_best_mixed_schedule_from_job_sequence: Callable[
        [list[str], bool, bool], HybridFlowshopLiteSchedule | None
    ]
    get_selected_dispatch_candidate_schedules: Callable[
        ..., dict[str, HybridFlowshopLiteSchedule | None]
    ]
    get_schedule_by_best_of_mixed_dispatches: Callable[
        ..., HybridFlowshopLiteSchedule | None
    ]
    repair_post_mip_dispatch_candidate: Callable[
        ..., HybridFlowshopLiteSchedule | None
    ]


@dataclass(frozen=True)
class PostMipDispatchRunResult:
    dispatched_schedule: HybridFlowshopLiteSchedule | None
    selected_dispatch_variant: str | None
    dispatched_schedules: dict[str, HybridFlowshopLiteSchedule | None]
    dispatch_candidate_elapsed_sec: dict[str, float]
    dispatch_phase_elapsed_sec: dict[str, float]
    stage_2_job_sequence: Mapping[str, Sequence[str]] | None
    stage_2_job_release: Mapping[str, Mapping[str, int]] | None
    pre_local_repair_selected_dispatch_variant: str | None
    pre_local_repair_selected_dispatch_makespan: float | None


def _get_post_mip_method_list(method_list: Sequence[str] | None) -> list[str]:
    """Filter post-MIP candidates down to the families that still earn their keep."""
    if method_list is None:
        return []
    return [method for method in method_list if method != "bn2d_all_stages"]


def _select_best_dispatch_candidate(
    dispatch_candidates: Mapping[str, HybridFlowshopLiteSchedule | None],
) -> tuple[str | None, HybridFlowshopLiteSchedule | None]:
    best_variant: str | None = None
    best_schedule: HybridFlowshopLiteSchedule | None = None
    for variant, schedule in dispatch_candidates.items():
        if schedule is None:
            continue
        if best_schedule is None or schedule.makespan < best_schedule.makespan:
            best_variant = variant
            best_schedule = schedule
    return best_variant, best_schedule


def run_post_mip_dispatch(
    *,
    instance: HybridFlowshopParameters,
    stage_2_job_2_p_dict: dict[str, dict[str, int]],
    solution_payload: dict[str, Any] | None,
    dispatch_window_lookup: Any,
    es_ls_local_repair_max_passes: int,
    dependencies: PostMipDispatchDependencies,
) -> PostMipDispatchRunResult:
    dispatched_schedule = None
    selected_dispatch_variant: str | None = None

    if dispatch_window_lookup is None:
        return PostMipDispatchRunResult(
            dispatched_schedule=None,
            selected_dispatch_variant=None,
            dispatched_schedules={},
            dispatch_candidate_elapsed_sec={},
            dispatch_phase_elapsed_sec={},
            stage_2_job_sequence=None,
            stage_2_job_release=None,
            pre_local_repair_selected_dispatch_variant=None,
            pre_local_repair_selected_dispatch_makespan=None,
        )

    post_dispatch_total_timer = ElapsedTimer()
    dispatch_phase_elapsed_sec: dict[str, float] = {}
    dispatch_candidate_elapsed_sec: dict[str, float] = {}
    sequence_setup_timer = ElapsedTimer()
    stage_2_job_sequence = get_stage_job_sequences_from_dispatch_windows(
        instance.stage_id_list,
        instance.job_id_list,
        dispatch_window_lookup,
        stage_2_job_2_p_dict,
    )
    stage_2_job_release = get_stage_job_release_times_from_dispatch_windows(
        instance.stage_id_list,
        instance.job_id_list,
        dispatch_window_lookup,
    )
    es_ls_job_tiebreak_rank = get_job_tiebreak_rank_from_stage_job_sequences(
        instance.stage_id_list,
        stage_2_job_sequence,
    )
    bottleneck_anchor_stage_id = get_bottleneck_anchor_stage_from_solution_payload(
        instance.stage_id_list,
        stage_2_job_2_p_dict,
        instance.stage_2_machines_map,
        solution_payload,
    )
    tail_ls_sequence = get_job_sequence_from_dispatch_windows_anchor_stage(
        instance.stage_id_list,
        instance.job_id_list,
        dispatch_window_lookup,
        stage_2_job_2_p_dict,
        anchor_stage_id=instance.stage_id_list[-1],
        sort_rule="ls_es_p_desc",
    )
    bottleneck_slack_sequence = get_job_sequence_from_dispatch_windows_anchor_stage(
        instance.stage_id_list,
        instance.job_id_list,
        dispatch_window_lookup,
        stage_2_job_2_p_dict,
        anchor_stage_id=bottleneck_anchor_stage_id,
        sort_rule="slack_ls_es_p_desc",
    )
    aggregate_es_slack_sequence = get_job_sequence_from_dispatch_windows_aggregate(
        instance.stage_id_list,
        instance.job_id_list,
        dispatch_window_lookup,
        stage_2_job_2_p_dict,
        aggregation_rule="sum_es_slack_p_desc",
    )
    aggregate_ls_slack_sequence = get_job_sequence_from_dispatch_windows_aggregate(
        instance.stage_id_list,
        instance.job_id_list,
        dispatch_window_lookup,
        stage_2_job_2_p_dict,
        aggregation_rule="sum_ls_slack_p_desc",
    )
    dispatch_phase_elapsed_sec["sequence_setup_sec"] = sequence_setup_timer.elapsed_sec

    selected_dispatch_config = dependencies.get_selected_dispatch_config()
    candidate_generation_timer = ElapsedTimer()
    dispatch_candidates: dict[str, HybridFlowshopLiteSchedule | None] = {}
    target_stage_ids = [bottleneck_anchor_stage_id, instance.stage_id_list[-1]]

    direct_mixed_sequences = {
        "mixed_aggregate_es_slack": aggregate_es_slack_sequence,
        "mixed_aggregate_ls_slack": aggregate_ls_slack_sequence,
    }
    for variant, job_sequence in direct_mixed_sequences.items():
        variant_timer = ElapsedTimer()
        try:
            candidate_schedule = dependencies.get_best_mixed_schedule_from_job_sequence(
                job_sequence,
                machine_then_job=selected_dispatch_config["machine_then_job"],
                head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
            )
            dispatch_candidates[variant] = candidate_schedule
            logging.info(
                "[MIP LB] %s has makespan=%s",
                variant,
                candidate_schedule.makespan if candidate_schedule is not None else None,
            )
        except Exception:
            logging.exception(
                "[MIP LB] %s failed while constructing direct mixed dispatch from an ES/LS-derived sequence.",
                variant,
            )
            dispatch_candidates[variant] = None
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec

    heuristic_candidates = dependencies.get_selected_dispatch_candidate_schedules(
        left_cap_multiplier=selected_dispatch_config["left_cap_multiplier"],
        right_cap_multiplier=selected_dispatch_config["right_cap_multiplier"],
        left_cap_portion=selected_dispatch_config["left_cap_portion"],
        right_cap_portion=selected_dispatch_config["right_cap_portion"],
        normalize_by_stage_cnt=selected_dispatch_config["normalize_by_stage_cnt"],
        randomize_mid_all=selected_dispatch_config["randomize_mid_all"],
        reverse_mid_all=selected_dispatch_config["reverse_mid_all"],
        reverse_mid_even=selected_dispatch_config["reverse_mid_even"],
        mixed_schedule_for_former_stages=selected_dispatch_config[
            "mixed_schedule_for_former_stages"
        ],
        mixed_schedule_for_later_stages=selected_dispatch_config[
            "mixed_schedule_for_later_stages"
        ],
        machine_then_job=selected_dispatch_config["machine_then_job"],
        head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
        p_agg_method=selected_dispatch_config["p_agg_method"],
        mi_agg_method=selected_dispatch_config["mi_agg_method"],
        method_list=_get_post_mip_method_list(selected_dispatch_config["method_list"]),
        job_tiebreak_rank=es_ls_job_tiebreak_rank,
        candidate_elapsed_sec_out=dispatch_candidate_elapsed_sec,
    )

    # Keep the post-MIP candidate pool conservative: drop only the slow rank
    # variants that underperformed in the 240-instance delta=400 study, while
    # retaining the stronger/faster families.
    stronger_mixed_rank_candidates = {
        "best_of_mixed_dispatches_tail_ls_rank": get_job_tiebreak_rank_from_job_sequence(
            tail_ls_sequence
        ),
        "best_of_mixed_dispatches_bottleneck_slack_rank": get_job_tiebreak_rank_from_job_sequence(
            bottleneck_slack_sequence
        ),
    }
    for variant, job_tiebreak_rank in stronger_mixed_rank_candidates.items():
        variant_timer = ElapsedTimer()
        try:
            candidate_schedule = dependencies.get_schedule_by_best_of_mixed_dispatches(
                machine_then_job=selected_dispatch_config["machine_then_job"],
                head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
                job_tiebreak_rank=job_tiebreak_rank,
            )
            dispatch_candidates[variant] = candidate_schedule
            logging.info(
                "[MIP LB] %s has makespan=%s",
                variant,
                candidate_schedule.makespan if candidate_schedule is not None else None,
            )
        except Exception:
            logging.exception(
                "[MIP LB] %s failed while constructing mixed dispatch with ES/LS-derived rank.",
                variant,
            )
            dispatch_candidates[variant] = None
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec

    for variant, candidate_schedule in heuristic_candidates.items():
        dispatch_candidates[variant] = candidate_schedule
        logging.info(
            "[MIP LB] %s dispatch with ES/LS tie-break has makespan=%s",
            variant,
            candidate_schedule.makespan if candidate_schedule is not None else None,
        )

    dispatch_phase_elapsed_sec["candidate_generation_sec"] = (
        candidate_generation_timer.elapsed_sec
    )
    logging.info(
        "[MIP LB] Post-MIP dispatch candidate summary: %s",
        {
            variant: (schedule.makespan if schedule is not None else None)
            for variant, schedule in dispatch_candidates.items()
        },
    )
    pre_local_repair_selected_dispatch_variant: str | None = None
    pre_local_repair_selected_dispatch_makespan: float | None = None
    selected_dispatch_variant, dispatched_schedule = _select_best_dispatch_candidate(
        dispatch_candidates
    )
    if dispatched_schedule is not None:
        pre_local_repair_selected_dispatch_variant = selected_dispatch_variant
        pre_local_repair_selected_dispatch_makespan = float(
            dispatched_schedule.makespan
        )
        if dispatched_schedule is not None and es_ls_local_repair_max_passes > 0:
            final_local_repair_timer = ElapsedTimer()
            try:
                repaired_selected_schedule = (
                    dependencies.repair_post_mip_dispatch_candidate(
                        dispatched_schedule,
                        target_stage_ids=target_stage_ids,
                        insertion_passes=max(1, es_ls_local_repair_max_passes),
                        max_shift=4,
                        swap_passes=max(1, es_ls_local_repair_max_passes),
                    )
                )
                dispatch_candidates["selected_post_mip_local_repair"] = (
                    repaired_selected_schedule
                )
                logging.info(
                    "[MIP LB] selected_post_mip_local_repair has makespan=%s (base=%s, base_variant=%s)",
                    repaired_selected_schedule.makespan,
                    dispatched_schedule.makespan,
                    selected_dispatch_variant,
                )
                selected_dispatch_variant, dispatched_schedule = (
                    _select_best_dispatch_candidate(dispatch_candidates)
                )
            except Exception:
                logging.exception("[MIP LB] selected_post_mip_local_repair failed.")
            dispatch_candidate_elapsed_sec["selected_post_mip_local_repair"] = (
                final_local_repair_timer.elapsed_sec
            )
        dependencies.check_feasibility(dispatched_schedule.get_jik_2_start_time_map())
        logging.info(
            "[MIP LB] Selected post-MIP dispatch variant %s with makespan=%s",
            selected_dispatch_variant,
            dispatched_schedule.makespan,
        )
    else:
        logging.info("[MIP LB] No feasible post-MIP dispatch candidate was generated.")

    dispatch_phase_elapsed_sec["total_post_mip_dispatch_sec"] = (
        post_dispatch_total_timer.elapsed_sec
    )
    return PostMipDispatchRunResult(
        dispatched_schedule=dispatched_schedule,
        selected_dispatch_variant=selected_dispatch_variant,
        dispatched_schedules=dispatch_candidates,
        dispatch_candidate_elapsed_sec=dispatch_candidate_elapsed_sec,
        dispatch_phase_elapsed_sec=dispatch_phase_elapsed_sec,
        stage_2_job_sequence=stage_2_job_sequence,
        stage_2_job_release=stage_2_job_release,
        pre_local_repair_selected_dispatch_variant=pre_local_repair_selected_dispatch_variant,
        pre_local_repair_selected_dispatch_makespan=pre_local_repair_selected_dispatch_makespan,
    )


def write_post_mip_dispatch_artifacts(
    *,
    output_dir: Path,
    dispatch_result: PostMipDispatchRunResult,
    solution_payload: Mapping[str, Any] | None = None,
    mip_result: BucketSearchResult | None,
    apply_mip_lb_elapsed_sec: float | None,
    post_mip_dispatch_elapsed_sec: float | None,
    draw_visualizations: bool = True,
) -> None:
    """Write ES/LS-guided dispatch outputs under an instance mip_lb directory."""
    dispatch_dir = output_dir / "dispatch"
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    dispatch_candidates = {
        variant: (float(schedule.makespan) if schedule is not None else None)
        for variant, schedule in dispatch_result.dispatched_schedules.items()
    }
    candidate_elapsed_sec = {
        str(variant): float(elapsed_sec)
        for variant, elapsed_sec in dispatch_result.dispatch_candidate_elapsed_sec.items()
    }
    best_candidate_makespan = min(
        (makespan for makespan in dispatch_candidates.values() if makespan is not None),
        default=None,
    )
    candidate_rankings = []
    for variant, makespan in sorted(
        dispatch_candidates.items(),
        key=lambda item: (
            item[1] is None,
            float("inf") if item[1] is None else item[1],
            item[0],
        ),
    ):
        candidate_rankings.append(
            {
                "variant": variant,
                "makespan": makespan,
                "gap_to_best": (
                    None
                    if best_candidate_makespan is None or makespan is None
                    else float(makespan - best_candidate_makespan)
                ),
                "elapsed_sec": candidate_elapsed_sec.get(variant),
                "is_selected": variant == dispatch_result.selected_dispatch_variant,
            }
        )

    summary_dict: dict[str, Any] = {
        "selected_variant": dispatch_result.selected_dispatch_variant,
        "selected_makespan": (
            float(dispatch_result.dispatched_schedule.makespan)
            if dispatch_result.dispatched_schedule is not None
            else None
        ),
        "selected_pre_final_local_repair_variant": (
            dispatch_result.pre_local_repair_selected_dispatch_variant
        ),
        "selected_pre_final_local_repair_makespan": (
            dispatch_result.pre_local_repair_selected_dispatch_makespan
        ),
        "dispatch_candidates": dispatch_candidates,
        "dispatch_candidate_elapsed_sec": candidate_elapsed_sec,
        "dispatch_candidate_rankings": candidate_rankings,
        "dispatch_phase_elapsed_sec": dispatch_result.dispatch_phase_elapsed_sec,
        "timing": {
            "apply_mip_lb_elapsed_sec": apply_mip_lb_elapsed_sec,
            "post_mip_dispatch_elapsed_sec": post_mip_dispatch_elapsed_sec,
            "mip_time_limit_sec_used": (
                float(mip_result.time_limit_sec_used)
                if mip_result is not None
                else None
            ),
            "mip_solver_runtime_sec": (
                float(mip_result.total_runtime_sec) if mip_result is not None else None
            ),
            "mip_wall_runtime_sec": (
                float(mip_result.wall_runtime_sec)
                if mip_result is not None and mip_result.wall_runtime_sec is not None
                else None
            ),
            "mip_model_build_wall_sec": (
                float(mip_result.model_build_wall_sec)
                if mip_result is not None
                and mip_result.model_build_wall_sec is not None
                else None
            ),
        },
    }
    dump_yaml(summary_dict, dispatch_dir / "dispatch_summary.yaml")
    variant_metadata = _build_dispatch_variant_metadata(
        candidate_rankings=candidate_rankings,
        selected_variant=dispatch_result.selected_dispatch_variant,
        pre_local_repair_selected_variant=(
            dispatch_result.pre_local_repair_selected_dispatch_variant
        ),
    )
    dump_yaml(variant_metadata, dispatch_dir / "dispatch_variant_manifest.yaml")
    with (dispatch_dir / "dispatch_variant_objectives.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank",
                "artifact_slug",
                "variant",
                "short_name",
                "primary_role",
                "status_summary",
                "status_tags_text",
                "makespan",
                "gap_to_best",
                "elapsed_sec",
                "is_selected_final",
                "is_pre_repair_base",
                "is_local_repair_variant",
                "repair_base_variant",
                "repair_base_short_name",
            ),
        )
        writer.writeheader()
        for metadata in variant_metadata.values():
            writer.writerow(
                {
                    "rank": metadata["rank"],
                    "artifact_slug": metadata["artifact_slug"],
                    "variant": metadata["variant"],
                    "short_name": metadata["short_name"],
                    "primary_role": metadata["primary_role"],
                    "status_summary": metadata["status_summary"],
                    "status_tags_text": metadata["status_tags_text"],
                    "makespan": metadata["makespan"],
                    "gap_to_best": metadata["gap_to_best"],
                    "elapsed_sec": metadata["elapsed_sec"],
                    "is_selected_final": metadata["is_selected_final"],
                    "is_pre_repair_base": metadata["is_pre_repair_base"],
                    "is_local_repair_variant": metadata["is_local_repair_variant"],
                    "repair_base_variant": metadata["repair_base_variant"],
                    "repair_base_short_name": metadata["repair_base_short_name"],
                }
            )

    if dispatch_result.stage_2_job_sequence is not None:
        dump_yaml(
            {
                str(stage_id): [str(job_id) for job_id in job_ids]
                for stage_id, job_ids in dispatch_result.stage_2_job_sequence.items()
            },
            dispatch_dir / "es_ls_stage_job_sequence.yaml",
        )
    if dispatch_result.stage_2_job_release is not None:
        dump_yaml(
            {
                str(stage_id): {
                    str(job_id): int(release_t)
                    for job_id, release_t in job_2_release.items()
                }
                for stage_id, job_2_release in dispatch_result.stage_2_job_release.items()
            },
            dispatch_dir / "es_ls_stage_job_release.yaml",
        )
    if solution_payload is not None:
        metadata = dict(solution_payload.get("metadata", {}) or {})
        stage_id_list = [
            str(stage_id) for stage_id in list(metadata.get("stage_ids") or [])
        ]
        job_id_list = [str(job_id) for job_id in list(metadata.get("job_ids") or [])]
        dispatch_windows = list(solution_payload.get("dispatch_windows", []) or [])
        if stage_id_list and job_id_list and dispatch_windows:
            dispatch_window_lookup = build_dispatch_window_lookup(dispatch_windows)
            stage_2_job_latest_start = (
                get_stage_job_latest_start_times_from_dispatch_windows(
                    stage_id_list,
                    job_id_list,
                    dispatch_window_lookup,
                )
            )
            dump_yaml(
                {
                    str(stage_id): {
                        str(job_id): int(latest_start)
                        for job_id, latest_start in job_2_latest_start.items()
                    }
                    for stage_id, job_2_latest_start in stage_2_job_latest_start.items()
                },
                dispatch_dir / "es_ls_stage_job_latest_start.yaml",
            )

    gantt_dir: Path | None = None
    if draw_visualizations:
        gantt_dir = dispatch_dir / "gantt"
        gantt_dir.mkdir(parents=True, exist_ok=True)

    for variant, schedule in dispatch_result.dispatched_schedules.items():
        if schedule is None:
            continue
        dump_yaml(
            {
                "start_time_map": schedule.get_jik_2_start_time_map(),
                "end_time_map": schedule.get_jik_2_end_time_map(),
            },
            dispatch_dir / f"{variant}_solution.yaml",
        )
        if draw_visualizations and gantt_dir is not None:
            _write_schedule_gantt(gantt_dir / f"{variant}.png", schedule)

    if draw_visualizations and solution_payload is not None:
        write_dispatch_schedule_window_visualizations(
            output_dir,
            solution_payload,
            dispatch_result.dispatched_schedules,
            variant_metadata=variant_metadata,
        )
        overlay_root = dispatch_dir / "dispatch_window_overlays"
        for metadata in variant_metadata.values():
            variant_dir = overlay_root / str(metadata["artifact_slug"])
            if variant_dir.is_dir():
                dump_yaml(metadata, variant_dir / "variant_info.yaml")


def _write_schedule_gantt(
    output_path: Path,
    schedule: HybridFlowshopLiteSchedule,
) -> None:
    plotter = GanttPlotter()
    plotter.export_hybrid_flowshop_plot(
        output_path,
        schedule.get_jik_2_start_time_map(),
        schedule.get_jik_2_end_time_map(),
        job_list=list(schedule.jobs),
        stage_list=list(schedule.stages),
        machine_list_per_stage={
            stage_id: list(schedule.machines_per_stage[stage_id])
            for stage_id in schedule.stages
        },
        all_job_list=list(schedule.jobs),
    )


def _build_dispatch_variant_metadata(
    *,
    candidate_rankings: Sequence[Mapping[str, Any]],
    selected_variant: str | None,
    pre_local_repair_selected_variant: str | None,
) -> dict[str, dict[str, Any]]:
    metadata_by_variant: dict[str, dict[str, Any]] = {}
    for rank, ranking in enumerate(candidate_rankings, start=1):
        variant = str(ranking["variant"])
        makespan = ranking["makespan"]
        gap_to_best = ranking.get("gap_to_best")
        is_selected = variant == selected_variant
        is_pre_repair_base = variant == pre_local_repair_selected_variant
        is_local_repair = variant == "selected_post_mip_local_repair"
        ties_best = float(gap_to_best or 0.0) == 0.0
        repair_base_variant = (
            pre_local_repair_selected_variant if is_local_repair else None
        )
        repair_base_short_name = (
            _get_dispatch_variant_short_name(str(repair_base_variant))
            if repair_base_variant is not None
            else None
        )

        tags: list[str] = []
        if is_selected:
            tags.append("FINAL_SELECTED")
        if is_pre_repair_base:
            tags.append("PRE_REPAIR_BASE")
        if is_local_repair:
            tags.append("LOCAL_REPAIR")
        if ties_best:
            tags.append("TIES_BEST")
        if not tags:
            tags.append("CANDIDATE")

        if is_selected:
            primary_role = "selected"
        elif is_local_repair:
            primary_role = "repair"
        elif is_pre_repair_base:
            primary_role = "base"
        else:
            primary_role = "cand"

        short_name = _get_dispatch_variant_short_name(variant)
        status_slug_parts: list[str] = []
        if is_selected:
            status_slug_parts.append("SELECTED")
        if is_pre_repair_base:
            status_slug_parts.append("BASE")
        if is_local_repair:
            status_slug_parts.append("REPAIR")
        if ties_best:
            status_slug_parts.append("BEST")
        if not status_slug_parts:
            status_slug_parts.append("CAND")
        status_slug = "+".join(status_slug_parts)
        name_slug = (
            f"repair_of_{repair_base_short_name}"
            if repair_base_short_name is not None
            else short_name
        )
        obj_token = f"obj{int(float(makespan))}" if makespan is not None else "objNA"
        gap_token = (
            f"gap{int(float(gap_to_best))}" if gap_to_best is not None else "gapNA"
        )
        artifact_slug = (
            f"{rank:02d}_{status_slug}__{name_slug}__{obj_token}_{gap_token}"
        )
        status_summary_parts = list(tags)
        if repair_base_short_name is not None:
            status_summary_parts.append(f"REPAIR_OF={repair_base_short_name}")
        metadata_by_variant[variant] = {
            "rank": rank,
            "variant": variant,
            "artifact_slug": artifact_slug,
            "short_name": short_name,
            "primary_role": primary_role,
            "status_tags": list(tags),
            "status_tags_text": " | ".join(tags),
            "status_summary": " | ".join(status_summary_parts),
            "is_selected_final": is_selected,
            "is_pre_repair_base": is_pre_repair_base,
            "is_local_repair_variant": is_local_repair,
            "repair_base_variant": repair_base_variant,
            "repair_base_short_name": repair_base_short_name,
            "makespan": makespan,
            "gap_to_best": gap_to_best,
            "elapsed_sec": ranking.get("elapsed_sec"),
        }
    return metadata_by_variant


def _get_dispatch_variant_short_name(variant: str) -> str:
    alias_map = {
        "best_of_mixed_dispatches_tail_ls_rank": "mixed_tail_ls_rank",
        "best_of_mixed_dispatches_bottleneck_slack_rank": "mixed_bneck_rank",
        "best_of_mixed_dispatches_aggregate_es_slack_rank": "mixed_agg_es_rank",
        "best_of_mixed_dispatches_aggregate_ls_slack_rank": "mixed_agg_ls_rank",
        "best_of_mixed_dispatches": "best_mixed",
        "mixed_aggregate_es_slack": "mixed_agg_es",
        "mixed_aggregate_ls_slack": "mixed_agg_ls",
        "selected_post_mip_local_repair": "local_repair",
    }
    return alias_map.get(variant, variant)
