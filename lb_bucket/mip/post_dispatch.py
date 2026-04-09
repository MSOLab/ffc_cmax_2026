from __future__ import annotations

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
    build_schedule_from_stage_job_sequences_priority_score,
    get_bottleneck_anchor_stage_from_solution_payload,
    get_job_sequence_from_dispatch_windows_aggregate,
    get_job_sequence_from_dispatch_windows_anchor_stage,
    get_job_tiebreak_rank_from_job_sequence,
    get_job_tiebreak_rank_from_stage_job_sequences,
    get_stage_job_release_times_from_dispatch_windows,
    get_stage_job_sequences_from_dispatch_windows,
    get_stage_job_sequences_from_schedule,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, OperationType
from lb_bucket.mip.shared import BucketSearchResult


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
    pre_local_repair_selected_dispatch_variant: str | None
    pre_local_repair_selected_dispatch_makespan: float | None


def _get_stage_adaptive_rank_repair_base_variant(stage_count: int) -> str:
    """Choose the strongest repair seed family observed in the ff2020 runs.

    Shorter flowshops benefited most from aggregate late-start rank seeds,
    while deeper flowshops were better served by tail-stage rank seeds.
    """
    if stage_count <= 10:
        return "best_of_mixed_dispatches_aggregate_ls_slack_rank"
    return "best_of_mixed_dispatches_tail_ls_rank"


def _get_stage_adaptive_direct_mixed_repair_base_variant() -> str:
    """Prefer the stronger direct mixed aggregate seed before local repair."""
    return "mixed_aggregate_es_slack"


def _get_post_mip_selected_dispatch_method_list(
    method_list: Sequence[str],
    stage_count: int,
) -> list[str]:
    """Trim weak heuristic families from the post-MIP pool when they rarely pay off."""
    selected_methods = list(method_list)
    if stage_count > 10:
        selected_methods = [
            method_name
            for method_name in selected_methods
            if method_name != "bn2d_all_stages"
        ]
    return selected_methods or list(method_list)


def _should_run_expensive_post_mip_repairs(
    *,
    stage_count: int,
    incumbent_makespan: float | None,
    best_candidate_makespan: float | None,
) -> bool:
    """Run heavy repair only when the cheap pool failed or barely improved."""
    if best_candidate_makespan is None:
        return True
    if incumbent_makespan is None:
        return stage_count >= 15
    if best_candidate_makespan >= incumbent_makespan:
        return True

    absolute_improvement = incumbent_makespan - best_candidate_makespan
    tiny_improvement_cutoff = max(6.0, 0.0015 * incumbent_makespan)
    return stage_count >= 15 and absolute_improvement <= tiny_improvement_cutoff


def _should_run_final_selected_post_mip_local_repair(
    *,
    stage_count: int,
    incumbent_makespan: float | None,
    selected_candidate_makespan: float | None,
) -> bool:
    """Reserve the final polish pass for weak or borderline-selected schedules."""
    if selected_candidate_makespan is None:
        return False
    if incumbent_makespan is None:
        return stage_count >= 15
    if selected_candidate_makespan >= incumbent_makespan:
        return True

    absolute_improvement = incumbent_makespan - selected_candidate_makespan
    tiny_improvement_cutoff = max(3.0, 0.0008 * incumbent_makespan)
    return stage_count >= 15 and absolute_improvement <= tiny_improvement_cutoff


def _create_empty_schedule_from_instance(
    instance: HybridFlowshopParameters,
) -> HybridFlowshopLiteSchedule:
    return HybridFlowshopLiteSchedule(
        instance.job_id_list,
        instance.stage_id_list,
        instance.stage_2_machines_map,
    )


def _build_incumbent_guided_priority_blend(
    *,
    instance: HybridFlowshopParameters,
    incumbent_schedule: HybridFlowshopLiteSchedule | None,
    mip_stage_2_job_sequence: Mapping[str, Sequence[str]],
    dispatch_window_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
    stage_2_job_2_p_dict: Mapping[str, Mapping[str, int]],
    anchor_stage_ids: Sequence[str],
) -> HybridFlowshopLiteSchedule | None:
    """Keep most incumbent stage orders and overwrite only the MIP-guided anchor stages."""
    if incumbent_schedule is None:
        return None

    blended_stage_sequences = {
        stage_id: list(job_sequence)
        for stage_id, job_sequence in get_stage_job_sequences_from_schedule(
            incumbent_schedule
        ).items()
    }
    for stage_id in anchor_stage_ids:
        if stage_id in mip_stage_2_job_sequence:
            blended_stage_sequences[stage_id] = list(mip_stage_2_job_sequence[stage_id])

    stage_2_job_2_release = get_stage_job_release_times_from_dispatch_windows(
        instance.stage_id_list,
        instance.job_id_list,
        dispatch_window_lookup,
    )
    schedule = build_schedule_from_stage_job_sequences_priority_score(
        lambda: _create_empty_schedule_from_instance(instance),
        blended_stage_sequences,
        stage_2_job_2_p_dict,
        stage_2_job_2_release=stage_2_job_2_release,
    )
    schedule.make_semi_active(stage_2_job_2_p_dict)
    return schedule


def run_post_mip_dispatch(
    *,
    instance: HybridFlowshopParameters,
    stage_2_job_2_p_dict: dict[str, dict[str, int]],
    solution_payload: dict[str, Any] | None,
    dispatch_window_lookup: Any,
    incumbent_schedule: HybridFlowshopLiteSchedule | None,
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
    anchor_stage_ids: list[str] = []
    for stage_id in [bottleneck_anchor_stage_id, instance.stage_id_list[-1]]:
        if stage_id not in anchor_stage_ids:
            anchor_stage_ids.append(stage_id)
    dispatch_phase_elapsed_sec["sequence_setup_sec"] = sequence_setup_timer.elapsed_sec

    selected_dispatch_config = dependencies.get_selected_dispatch_config()
    candidate_generation_timer = ElapsedTimer()
    dispatch_candidates: dict[str, HybridFlowshopLiteSchedule | None] = {}
    incumbent_makespan = (
        float(incumbent_schedule.makespan)
        if incumbent_schedule is not None
        else None
    )

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
            if candidate_schedule is not None:
                dependencies.check_feasibility(
                    candidate_schedule.get_jik_2_start_time_map()
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

    incumbent_blend_timer = ElapsedTimer()
    try:
        incumbent_guided_schedule = _build_incumbent_guided_priority_blend(
            instance=instance,
            incumbent_schedule=incumbent_schedule,
            mip_stage_2_job_sequence=stage_2_job_sequence,
            dispatch_window_lookup=dispatch_window_lookup,
            stage_2_job_2_p_dict=stage_2_job_2_p_dict,
            anchor_stage_ids=anchor_stage_ids,
        )
        if incumbent_guided_schedule is not None:
            dependencies.check_feasibility(
                incumbent_guided_schedule.get_jik_2_start_time_map()
            )
        dispatch_candidates["incumbent_guided_priority_blend"] = (
            incumbent_guided_schedule
        )
        logging.info(
            "[MIP LB] incumbent_guided_priority_blend has makespan=%s",
            (
                incumbent_guided_schedule.makespan
                if incumbent_guided_schedule is not None
                else None
            ),
        )
    except Exception:
        logging.exception(
            "[MIP LB] incumbent_guided_priority_blend failed while rebuilding from incumbent stage orders."
        )
        dispatch_candidates["incumbent_guided_priority_blend"] = None
    dispatch_candidate_elapsed_sec["incumbent_guided_priority_blend"] = (
        incumbent_blend_timer.elapsed_sec
    )

    post_mip_method_list = _get_post_mip_selected_dispatch_method_list(
        selected_dispatch_config["method_list"],
        instance.stage_count,
    )
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
        method_list=post_mip_method_list,
        job_tiebreak_rank=es_ls_job_tiebreak_rank,
        candidate_elapsed_sec_out=dispatch_candidate_elapsed_sec,
    )

    stronger_mixed_rank_candidates = {
        "best_of_mixed_dispatches_tail_ls_rank": get_job_tiebreak_rank_from_job_sequence(
            tail_ls_sequence
        ),
        "best_of_mixed_dispatches_bottleneck_slack_rank": get_job_tiebreak_rank_from_job_sequence(
            bottleneck_slack_sequence
        ),
        "best_of_mixed_dispatches_aggregate_es_slack_rank": get_job_tiebreak_rank_from_job_sequence(
            aggregate_es_slack_sequence
        ),
        "best_of_mixed_dispatches_aggregate_ls_slack_rank": get_job_tiebreak_rank_from_job_sequence(
            aggregate_ls_slack_sequence
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
            if candidate_schedule is not None:
                dependencies.check_feasibility(
                    candidate_schedule.get_jik_2_start_time_map()
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
        if candidate_schedule is not None:
            try:
                dependencies.check_feasibility(
                    candidate_schedule.get_jik_2_start_time_map()
                )
            except Exception:
                logging.exception(
                    "[MIP LB] %s dispatch failed feasibility check after construction.",
                    variant,
                )
                candidate_schedule = None
        dispatch_candidates[variant] = candidate_schedule
        logging.info(
            "[MIP LB] %s dispatch with ES/LS tie-break has makespan=%s",
            variant,
            candidate_schedule.makespan if candidate_schedule is not None else None,
        )

    current_best_schedule = min(
        (
            schedule
            for schedule in dispatch_candidates.values()
            if schedule is not None
        ),
        key=lambda schedule: schedule.makespan,
        default=None,
    )
    should_run_expensive_repairs = (
        es_ls_local_repair_max_passes > 0
        and _should_run_expensive_post_mip_repairs(
            stage_count=instance.stage_count,
            incumbent_makespan=incumbent_makespan,
            best_candidate_makespan=(
                float(current_best_schedule.makespan)
                if current_best_schedule is not None
                else None
            ),
        )
    )
    if should_run_expensive_repairs:
        for variant, base_name in {
            "mixed_aggregate_es_slack_local_repair": _get_stage_adaptive_direct_mixed_repair_base_variant(),
        }.items():
            variant_timer = ElapsedTimer()
            base_schedule = dispatch_candidates.get(base_name)
            if base_schedule is None:
                dispatch_candidates[variant] = None
                dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
                continue
            try:
                repaired_schedule = dependencies.repair_post_mip_dispatch_candidate(
                    base_schedule,
                    target_stage_ids=anchor_stage_ids,
                    insertion_passes=max(1, es_ls_local_repair_max_passes),
                    max_shift=4,
                    swap_passes=max(1, es_ls_local_repair_max_passes),
                )
                dependencies.check_feasibility(
                    repaired_schedule.get_jik_2_start_time_map()
                )
                dispatch_candidates[variant] = repaired_schedule
                logging.info(
                    "[MIP LB] %s has makespan=%s (base=%s)",
                    variant,
                    repaired_schedule.makespan,
                    base_schedule.makespan,
                )
            except Exception:
                logging.exception(
                    "[MIP LB] %s failed during makespan-oriented local repair.",
                    variant,
                )
                dispatch_candidates[variant] = None
            dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec

        adaptive_rank_repair_variant = (
            "best_of_mixed_dispatches_stage_adaptive_rank_local_repair"
        )
        adaptive_rank_repair_base_variant = _get_stage_adaptive_rank_repair_base_variant(
            instance.stage_count
        )
        base_schedule = dispatch_candidates.get(adaptive_rank_repair_base_variant)
        if base_schedule is not None:
            adaptive_rank_local_repair_timer = ElapsedTimer()
            try:
                repaired_schedule = dependencies.repair_post_mip_dispatch_candidate(
                    base_schedule,
                    target_stage_ids=anchor_stage_ids,
                    insertion_passes=max(1, es_ls_local_repair_max_passes),
                    max_shift=4,
                    swap_passes=max(1, es_ls_local_repair_max_passes),
                )
                dependencies.check_feasibility(
                    repaired_schedule.get_jik_2_start_time_map()
                )
                dispatch_candidates[adaptive_rank_repair_variant] = repaired_schedule
                logging.info(
                    "[MIP LB] %s has makespan=%s (base=%s, base_variant=%s)",
                    adaptive_rank_repair_variant,
                    repaired_schedule.makespan,
                    base_schedule.makespan,
                    adaptive_rank_repair_base_variant,
                )
            except Exception:
                logging.exception(
                    "[MIP LB] %s failed during makespan-oriented local repair.",
                    adaptive_rank_repair_variant,
                )
                dispatch_candidates[adaptive_rank_repair_variant] = None
            dispatch_candidate_elapsed_sec[adaptive_rank_repair_variant] = (
                adaptive_rank_local_repair_timer.elapsed_sec
            )
    else:
        logging.info(
            "[MIP LB] Skipping expensive local-repair variants after cheap pool already achieved a meaningful improvement%s.",
            (
                ""
                if incumbent_makespan is None
                else " over the incumbent"
            ),
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
    feasible_candidates = [
        (variant, schedule)
        for variant, schedule in dispatch_candidates.items()
        if schedule is not None
    ]
    pre_local_repair_selected_dispatch_variant: str | None = None
    pre_local_repair_selected_dispatch_makespan: float | None = None
    if feasible_candidates:
        selected_dispatch_variant, dispatched_schedule = min(
            feasible_candidates,
            key=lambda item: item[1].makespan,
        )
        pre_local_repair_selected_dispatch_variant = selected_dispatch_variant
        pre_local_repair_selected_dispatch_makespan = float(
            dispatched_schedule.makespan
        )
        if (
            dispatched_schedule is not None
            and es_ls_local_repair_max_passes > 0
            and _should_run_final_selected_post_mip_local_repair(
                stage_count=instance.stage_count,
                incumbent_makespan=incumbent_makespan,
                selected_candidate_makespan=float(dispatched_schedule.makespan),
            )
        ):
            final_local_repair_timer = ElapsedTimer()
            try:
                repaired_selected_schedule = (
                    dependencies.repair_post_mip_dispatch_candidate(
                        dispatched_schedule,
                        target_stage_ids=anchor_stage_ids,
                        insertion_passes=max(1, es_ls_local_repair_max_passes),
                        max_shift=4,
                        swap_passes=max(1, es_ls_local_repair_max_passes),
                    )
                )
                dependencies.check_feasibility(
                    repaired_selected_schedule.get_jik_2_start_time_map()
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
                feasible_candidates = [
                    (variant, schedule)
                    for variant, schedule in dispatch_candidates.items()
                    if schedule is not None
                ]
                selected_dispatch_variant, dispatched_schedule = min(
                    feasible_candidates,
                    key=lambda item: item[1].makespan,
                )
            except Exception:
                logging.exception("[MIP LB] selected_post_mip_local_repair failed.")
            dispatch_candidate_elapsed_sec["selected_post_mip_local_repair"] = (
                final_local_repair_timer.elapsed_sec
            )
        elif dispatched_schedule is not None:
            logging.info(
                "[MIP LB] Skipping selected_post_mip_local_repair because the chosen candidate is already strong enough."
            )
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
        pre_local_repair_selected_dispatch_variant=pre_local_repair_selected_dispatch_variant,
        pre_local_repair_selected_dispatch_makespan=pre_local_repair_selected_dispatch_makespan,
    )


def write_post_mip_dispatch_artifacts(
    *,
    output_dir: Path,
    dispatch_result: PostMipDispatchRunResult,
    mip_result: BucketSearchResult | None,
    apply_mip_lb_elapsed_sec: float | None,
    post_mip_dispatch_elapsed_sec: float | None,
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
                float(mip_result.time_limit_sec_used) if mip_result is not None else None
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

    if dispatch_result.stage_2_job_sequence is not None:
        dump_yaml(
            {
                str(stage_id): [str(job_id) for job_id in job_ids]
                for stage_id, job_ids in dispatch_result.stage_2_job_sequence.items()
            },
            dispatch_dir / "es_ls_stage_job_sequence.yaml",
        )

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
