from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from routix import ElapsedTimer
from routix.io.yaml import dump_yaml
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.dispatcher.utils import get_job_tiebreak_rank_from_job_sequence
from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, OperationType
from lb_bucket.cp.search import RetainedStageCpResult


@dataclass(frozen=True)
class PostRetainedCpDispatchDependencies:
    check_feasibility: Callable[[dict[OperationType, int]], float]
    get_selected_dispatch_config: Callable[[], dict[str, Any]]
    get_best_mixed_schedule_from_job_sequence: Callable[
        [Sequence[str], bool, bool], HybridFlowshopLiteSchedule | None
    ]
    get_schedule_by_best_of_mixed_dispatches: Callable[
        ..., HybridFlowshopLiteSchedule | None
    ]
    get_two_way_schedule_by_stage_band: Callable[
        ...,
        HybridFlowshopLiteSchedule | None,
    ]
    get_schedule_by_stage_job_sequences_priority: Callable[
        ...,
        HybridFlowshopLiteSchedule | None,
    ]
    repair_post_retained_cp_dispatch_candidate: Callable[
        ...,
        HybridFlowshopLiteSchedule | None,
    ]


@dataclass(frozen=True)
class PostRetainedCpDispatchRunResult:
    dispatched_schedule: HybridFlowshopLiteSchedule | None
    selected_dispatch_variant: str | None
    dispatched_schedules: dict[str, HybridFlowshopLiteSchedule | None]
    dispatch_candidate_elapsed_sec: dict[str, float]
    dispatch_phase_elapsed_sec: dict[str, float]
    anchor_blocks: Mapping[str, Sequence[str]]
    anchor_stage_sequences: Mapping[str, Mapping[str, Sequence[str]]]
    anchor_stage_releases: Mapping[str, Mapping[str, Mapping[str, int]]]
    variant_2_anchor_stage_ids: Mapping[str, Sequence[str]]
    pre_local_repair_selected_dispatch_variant: str | None
    pre_local_repair_selected_dispatch_makespan: float | None


def run_post_retained_cp_dispatch(
    *,
    instance: HybridFlowshopParameters,
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    cp_local_repair_max_passes: int,
    cp_local_repair_top_k: int = 1,
    include_release_anchor_candidates: bool,
    include_consensus_rank: bool = True,
    include_tail_bottleneck_rank: bool = True,
    include_extended_rank_variants: bool = False,
    include_dynamic_priority: bool = False,
    include_piecewise_stage_priority: bool = False,
    randomized_mixed_rank_trials: int = 0,
    dependencies: PostRetainedCpDispatchDependencies,
) -> PostRetainedCpDispatchRunResult:
    total_timer = ElapsedTimer()
    dispatch_phase_elapsed_sec: dict[str, float] = {}
    dispatch_candidate_elapsed_sec: dict[str, float] = {}
    variant_2_anchor_stage_ids: dict[str, list[str]] = {}

    setup_timer = ElapsedTimer()
    selected_dispatch_config = dependencies.get_selected_dispatch_config()
    anchor_blocks = _resolve_anchor_blocks(
        instance.stage_id_list,
        retained_cp_result,
    )
    anchor_stage_sequences: dict[str, dict[str, list[str]]] = {}
    anchor_stage_releases: dict[str, dict[str, dict[str, int]]] = {}
    for anchor_key, anchor_stage_ids in anchor_blocks.items():
        stage_2_job_sequence, stage_2_job_2_release = (
            _extract_anchor_stage_sequence_from_retained_rows(
                retained_solution_rows,
                anchor_stage_ids,
            )
        )
        anchor_stage_sequences[anchor_key] = stage_2_job_sequence
        anchor_stage_releases[anchor_key] = stage_2_job_2_release

    preferred_anchor_key = "preferred" if "preferred" in anchor_blocks else None
    preferred_anchor_stage_ids = list(anchor_blocks.get(preferred_anchor_key, []))
    preferred_anchor_first_stage_id = (
        preferred_anchor_stage_ids[0] if preferred_anchor_stage_ids else None
    )
    preferred_anchor_last_stage_id = (
        preferred_anchor_stage_ids[-1] if preferred_anchor_stage_ids else None
    )
    bottleneck_stage_id = _resolve_retained_bottleneck_stage_id(
        instance.stage_id_list,
        retained_cp_result,
        retained_solution_rows,
    )
    direct_sequences = _build_direct_mixed_sequences(
        instance=instance,
        retained_cp_result=retained_cp_result,
        retained_solution_rows=retained_solution_rows,
        preferred_anchor_first_stage_id=preferred_anchor_first_stage_id,
        preferred_anchor_last_stage_id=preferred_anchor_last_stage_id,
        bottleneck_stage_id=bottleneck_stage_id,
    )
    consensus_sequence = _get_job_sequence_from_retained_rows_consensus(
        stage_id_list=instance.stage_id_list,
        job_id_list=instance.job_id_list,
        retained_cp_result=retained_cp_result,
        retained_solution_rows=retained_solution_rows,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )
    tail_bottleneck_sequence = _get_job_sequence_from_retained_rows_tail_bottleneck(
        stage_id_list=instance.stage_id_list,
        job_id_list=instance.job_id_list,
        retained_cp_result=retained_cp_result,
        retained_solution_rows=retained_solution_rows,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )
    if include_consensus_rank:
        direct_sequences["mixed_cp_consensus"] = consensus_sequence
    if include_tail_bottleneck_rank:
        direct_sequences["mixed_cp_tail_bottleneck"] = tail_bottleneck_sequence
    propagated_stage_sequences = _build_full_stage_sequences_from_retained_rows(
        stage_id_list=instance.stage_id_list,
        retained_solution_rows=retained_solution_rows,
        fallback_job_sequence=consensus_sequence,
    )
    dispatch_phase_elapsed_sec["sequence_setup_sec"] = setup_timer.elapsed_sec

    dispatch_candidates: dict[str, HybridFlowshopLiteSchedule | None] = {}
    candidate_generation_timer = ElapsedTimer()

    if preferred_anchor_key is not None:
        _evaluate_anchor_band_candidates(
            dispatch_candidates=dispatch_candidates,
            dispatch_candidate_elapsed_sec=dispatch_candidate_elapsed_sec,
            variant_2_anchor_stage_ids=variant_2_anchor_stage_ids,
            dependencies=dependencies,
            option_kwargs=selected_dispatch_config,
            anchor_key=preferred_anchor_key,
            anchor_stage_ids=preferred_anchor_stage_ids,
            stage_2_job_sequence=anchor_stage_sequences[preferred_anchor_key],
            stage_2_job_2_release=anchor_stage_releases[preferred_anchor_key],
            include_release_candidates=include_release_anchor_candidates,
        )

    for anchor_key, anchor_stage_ids in anchor_blocks.items():
        if anchor_key == preferred_anchor_key:
            continue
        variant = f"cp_band_{anchor_key}_strict_start_release"
        variant_timer = ElapsedTimer()
        try:
            schedule = dependencies.get_two_way_schedule_by_stage_band(
                anchor_stage_ids=anchor_stage_ids,
                stage_2_job_sequence=anchor_stage_sequences[anchor_key],
                mixed_schedule_for_former_stages=selected_dispatch_config[
                    "mixed_schedule_for_former_stages"
                ],
                mixed_schedule_for_later_stages=selected_dispatch_config[
                    "mixed_schedule_for_later_stages"
                ],
                machine_then_job=selected_dispatch_config["machine_then_job"],
                stage_2_job_2_release=anchor_stage_releases[anchor_key],
                anchor_dispatch_mode="strict_start",
            )
        except Exception:
            logging.exception(
                "[CP LB] %s failed while constructing anchor-band dispatch.",
                variant,
            )
            schedule = None
        dispatch_candidates[variant] = schedule
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
        variant_2_anchor_stage_ids[variant] = list(anchor_stage_ids)
        logging.info(
            "[CP LB] %s has makespan=%s",
            variant,
            schedule.makespan if schedule is not None else None,
        )

    for variant, job_sequence in direct_sequences.items():
        variant_timer = ElapsedTimer()
        try:
            schedule = dependencies.get_best_mixed_schedule_from_job_sequence(
                job_sequence,
                machine_then_job=selected_dispatch_config["machine_then_job"],
                head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
            )
        except Exception:
            logging.exception(
                "[CP LB] %s failed while constructing direct mixed dispatch from retained-CP sequence.",
                variant,
            )
            schedule = None
        dispatch_candidates[variant] = schedule
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
        logging.info(
            "[CP LB] %s has makespan=%s",
            variant,
            schedule.makespan if schedule is not None else None,
        )

    extra_rank_sequences: dict[str, list[str]] = {}
    if include_consensus_rank:
        extra_rank_sequences["best_of_mixed_dispatches_cp_consensus_rank"] = (
            consensus_sequence
        )
    if include_tail_bottleneck_rank:
        extra_rank_sequences["best_of_mixed_dispatches_cp_tail_bottleneck_rank"] = (
            tail_bottleneck_sequence
        )
    if include_extended_rank_variants:
        existing_rank_sequences = {
            tuple(job_sequence) for job_sequence in extra_rank_sequences.values()
        }
        for variant, job_sequence in _build_extended_rank_sequences(
            stage_id_list=instance.stage_id_list,
            job_id_list=instance.job_id_list,
            retained_cp_result=retained_cp_result,
            retained_solution_rows=retained_solution_rows,
            preferred_anchor_stage_ids=preferred_anchor_stage_ids,
        ).items():
            sequence_tuple = tuple(job_sequence)
            if not sequence_tuple or sequence_tuple in existing_rank_sequences:
                continue
            extra_rank_sequences[variant] = job_sequence
            existing_rank_sequences.add(sequence_tuple)
    for variant, job_sequence in extra_rank_sequences.items():
        variant_timer = ElapsedTimer()
        try:
            schedule = dependencies.get_schedule_by_best_of_mixed_dispatches(
                machine_then_job=selected_dispatch_config["machine_then_job"],
                head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
                job_tiebreak_rank=get_job_tiebreak_rank_from_job_sequence(job_sequence),
            )
        except Exception:
            logging.exception(
                "[CP LB] %s failed while constructing mixed dispatch with specialized retained-CP rank.",
                variant,
            )
            schedule = None
        dispatch_candidates[variant] = schedule
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
        logging.info(
            "[CP LB] %s has makespan=%s",
            variant,
            schedule.makespan if schedule is not None else None,
        )

    if include_dynamic_priority:
        dynamic_priority_timer = ElapsedTimer()
        try:
            schedule = dependencies.get_two_way_schedule_by_stage_band(
                anchor_stage_ids=list(instance.stage_id_list),
                stage_2_job_sequence=propagated_stage_sequences,
                mixed_schedule_for_former_stages=selected_dispatch_config[
                    "mixed_schedule_for_former_stages"
                ],
                mixed_schedule_for_later_stages=selected_dispatch_config[
                    "mixed_schedule_for_later_stages"
                ],
                machine_then_job=selected_dispatch_config["machine_then_job"],
                stage_2_job_2_release=None,
                anchor_dispatch_mode="priority",
            )
        except Exception:
            logging.exception(
                "[CP LB] %s failed while constructing dynamic full-stage priority dispatch.",
                "cp_dynamic_priority_soft_release",
            )
            schedule = None
        dispatch_candidates["cp_dynamic_priority_soft_release"] = schedule
        dispatch_candidate_elapsed_sec["cp_dynamic_priority_soft_release"] = (
            dynamic_priority_timer.elapsed_sec
        )
        variant_2_anchor_stage_ids["cp_dynamic_priority_soft_release"] = list(
            instance.stage_id_list
        )
        logging.info(
            "[CP LB] %s has makespan=%s",
            "cp_dynamic_priority_soft_release",
            schedule.makespan if schedule is not None else None,
        )

    if include_piecewise_stage_priority:
        piecewise_stage_sequence_variants = _build_piecewise_stage_sequence_variants(
            stage_id_list=instance.stage_id_list,
            job_id_list=instance.job_id_list,
            retained_solution_rows=retained_solution_rows,
            fallback_job_sequence=consensus_sequence,
        )
        for variant, stage_2_job_sequence in piecewise_stage_sequence_variants.items():
            variant_timer = ElapsedTimer()
            try:
                schedule = dependencies.get_schedule_by_stage_job_sequences_priority(
                    stage_2_job_sequence=stage_2_job_sequence,
                    stage_2_job_2_release=None,
                )
            except Exception:
                logging.exception(
                    "[CP LB] %s failed while constructing piecewise stage-priority dispatch.",
                    variant,
                )
                schedule = None
            dispatch_candidates[variant] = schedule
            dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
            variant_2_anchor_stage_ids[variant] = list(instance.stage_id_list)
            logging.info(
                "[CP LB] %s has makespan=%s",
                variant,
                schedule.makespan if schedule is not None else None,
            )

    rank_candidates = {
        "best_of_mixed_dispatches_cp_first_anchor_rank": (
            direct_sequences.get("mixed_cp_first_anchor")
        ),
        "best_of_mixed_dispatches_cp_last_anchor_rank": (
            direct_sequences.get("mixed_cp_last_anchor")
        ),
        "best_of_mixed_dispatches_cp_bottleneck_rank": (
            direct_sequences.get("mixed_cp_bottleneck_anchor")
        ),
        "best_of_mixed_dispatches_cp_aggregate_start_slack_rank": (
            direct_sequences.get("mixed_cp_aggregate_start_slack")
        ),
    }
    for variant, job_sequence in rank_candidates.items():
        if not job_sequence:
            continue
        variant_timer = ElapsedTimer()
        try:
            schedule = dependencies.get_schedule_by_best_of_mixed_dispatches(
                machine_then_job=selected_dispatch_config["machine_then_job"],
                head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
                job_tiebreak_rank=get_job_tiebreak_rank_from_job_sequence(job_sequence),
            )
        except Exception:
            logging.exception(
                "[CP LB] %s failed while constructing mixed dispatch with retained-CP rank.",
                variant,
            )
            schedule = None
        dispatch_candidates[variant] = schedule
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
        logging.info(
            "[CP LB] %s has makespan=%s",
            variant,
            schedule.makespan if schedule is not None else None,
        )

    baseline_timer = ElapsedTimer()
    baseline_schedule = dependencies.get_schedule_by_best_of_mixed_dispatches(
        machine_then_job=selected_dispatch_config["machine_then_job"],
        head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
        job_tiebreak_rank=None,
    )
    dispatch_candidates["best_of_mixed_dispatches_cp_baseline"] = baseline_schedule
    dispatch_candidate_elapsed_sec["best_of_mixed_dispatches_cp_baseline"] = (
        baseline_timer.elapsed_sec
    )
    logging.info(
        "[CP LB] %s has makespan=%s",
        "best_of_mixed_dispatches_cp_baseline",
        baseline_schedule.makespan if baseline_schedule is not None else None,
    )

    for trial_idx in range(max(0, int(randomized_mixed_rank_trials))):
        variant = f"best_of_mixed_dispatches_random_rank_{trial_idx + 1}"
        variant_timer = ElapsedTimer()
        shuffled_jobs = [str(job_id) for job_id in instance.job_id_list]
        random.shuffle(shuffled_jobs)
        try:
            schedule = dependencies.get_schedule_by_best_of_mixed_dispatches(
                machine_then_job=selected_dispatch_config["machine_then_job"],
                head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
                job_tiebreak_rank=get_job_tiebreak_rank_from_job_sequence(
                    shuffled_jobs
                ),
            )
        except Exception:
            logging.exception(
                "[CP LB] %s failed while constructing mixed dispatch with random rank.",
                variant,
            )
            schedule = None
        dispatch_candidates[variant] = schedule
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
        logging.info(
            "[CP LB] %s has makespan=%s",
            variant,
            schedule.makespan if schedule is not None else None,
        )

    dispatch_phase_elapsed_sec["candidate_generation_sec"] = (
        candidate_generation_timer.elapsed_sec
    )
    logging.info(
        "[CP LB] Post-retained-CP dispatch candidate summary: %s",
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
        if cp_local_repair_max_passes > 0:
            repair_top_k = max(1, int(cp_local_repair_top_k))
            sorted_repair_seed_variants = [
                (variant, schedule)
                for variant, schedule in sorted(
                    dispatch_candidates.items(),
                    key=lambda item: (
                        item[1] is None,
                        float("inf") if item[1] is None else item[1].makespan,
                        item[0],
                    ),
                )
                if schedule is not None
            ]
            repair_seed_variants = [
                (
                    str(pre_local_repair_selected_dispatch_variant),
                    dispatched_schedule,
                )
            ]
            for seed_variant, seed_schedule in sorted_repair_seed_variants:
                if seed_variant == pre_local_repair_selected_dispatch_variant:
                    continue
                if len(repair_seed_variants) >= repair_top_k:
                    break
                repair_seed_variants.append((seed_variant, seed_schedule))
            for repair_rank, (seed_variant, seed_schedule) in enumerate(
                repair_seed_variants, start=1
            ):
                local_repair_timer = ElapsedTimer()
                repair_variant = (
                    "selected_post_retained_cp_local_repair"
                    if repair_rank == 1
                    and seed_variant == pre_local_repair_selected_dispatch_variant
                    else "post_retained_cp_local_repair__"
                    + _get_dispatch_variant_short_name(seed_variant)
                )
                try:
                    repair_anchor_stage_ids = list(
                        variant_2_anchor_stage_ids.get(
                            str(seed_variant),
                            preferred_anchor_stage_ids,
                        )
                    )
                    repair_release = _get_anchor_stage_release_by_stage_ids(
                        anchor_stage_releases=anchor_stage_releases,
                        anchor_stage_ids=repair_anchor_stage_ids,
                    )
                    repaired_schedule = dependencies.repair_post_retained_cp_dispatch_candidate(
                        seed_schedule,
                        target_stage_ids=repair_anchor_stage_ids,
                        insertion_passes=max(1, cp_local_repair_max_passes),
                        max_shift=4,
                        swap_passes=max(1, cp_local_repair_max_passes),
                        stage_2_job_2_release=repair_release,
                    )
                    dispatch_candidates[repair_variant] = repaired_schedule
                    dispatch_candidate_elapsed_sec[repair_variant] = (
                        local_repair_timer.elapsed_sec
                    )
                    if repair_anchor_stage_ids:
                        variant_2_anchor_stage_ids[repair_variant] = (
                            repair_anchor_stage_ids
                        )
                    logging.info(
                        "[CP LB] %s repaired %s from makespan=%s to %s",
                        repair_variant,
                        seed_variant,
                        seed_schedule.makespan,
                        (
                            repaired_schedule.makespan
                            if repaired_schedule is not None
                            else None
                        ),
                    )
                except Exception:
                    logging.exception(
                        "[CP LB] %s failed while repairing %s.",
                        repair_variant,
                        seed_variant,
                    )
                    dispatch_candidate_elapsed_sec[repair_variant] = (
                        local_repair_timer.elapsed_sec
                    )
            selected_dispatch_variant, dispatched_schedule = (
                _select_best_dispatch_candidate(dispatch_candidates)
            )
        dependencies.check_feasibility(dispatched_schedule.get_jik_2_start_time_map())
        logging.info(
            "[CP LB] Selected post-retained-CP dispatch variant %s with makespan=%s",
            selected_dispatch_variant,
            dispatched_schedule.makespan,
        )
    else:
        logging.info(
            "[CP LB] No feasible post-retained-CP dispatch candidate was generated."
        )

    dispatch_phase_elapsed_sec["total_post_retained_cp_dispatch_sec"] = (
        total_timer.elapsed_sec
    )
    return PostRetainedCpDispatchRunResult(
        dispatched_schedule=dispatched_schedule,
        selected_dispatch_variant=selected_dispatch_variant,
        dispatched_schedules=dispatch_candidates,
        dispatch_candidate_elapsed_sec=dispatch_candidate_elapsed_sec,
        dispatch_phase_elapsed_sec=dispatch_phase_elapsed_sec,
        anchor_blocks=anchor_blocks,
        anchor_stage_sequences=anchor_stage_sequences,
        anchor_stage_releases=anchor_stage_releases,
        variant_2_anchor_stage_ids=variant_2_anchor_stage_ids,
        pre_local_repair_selected_dispatch_variant=pre_local_repair_selected_dispatch_variant,
        pre_local_repair_selected_dispatch_makespan=pre_local_repair_selected_dispatch_makespan,
    )


def write_post_retained_cp_dispatch_artifacts(
    *,
    cp_lb_dir: Path,
    dispatch_result: PostRetainedCpDispatchRunResult,
    retained_cp_result: RetainedStageCpResult | None,
    apply_elapsed_sec: float | None,
    dispatch_elapsed_sec: float | None,
    draw_visualizations: bool = True,
) -> None:
    dispatch_dir = cp_lb_dir / "dispatch"
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
                "anchor_stages": " ".join(
                    str(stage_id)
                    for stage_id in dispatch_result.variant_2_anchor_stage_ids.get(
                        variant, ()
                    )
                ),
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
        "anchor_blocks": {
            key: [str(stage_id) for stage_id in stage_ids]
            for key, stage_ids in dispatch_result.anchor_blocks.items()
        },
        "timing": {
            "retained_cp_apply_elapsed_sec": apply_elapsed_sec,
            "post_retained_cp_dispatch_elapsed_sec": dispatch_elapsed_sec,
            "retained_cp_solver_runtime_sec": (
                getattr(retained_cp_result, "solver_runtime_sec", None)
                if retained_cp_result is not None
                else None
            ),
            "retained_cp_time_limit_sec_used": (
                getattr(retained_cp_result, "time_limit_sec_used", None)
                if retained_cp_result is not None
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
                "anchor_stages",
                "is_selected_final",
                "is_pre_repair_base",
                "is_local_repair_variant",
                "repair_base_variant",
                "repair_base_short_name",
            ),
        )
        writer.writeheader()
        for variant, metadata in variant_metadata.items():
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
                    "anchor_stages": " ".join(
                        str(stage_id)
                        for stage_id in dispatch_result.variant_2_anchor_stage_ids.get(
                            variant, ()
                        )
                    ),
                    "is_selected_final": metadata["is_selected_final"],
                    "is_pre_repair_base": metadata["is_pre_repair_base"],
                    "is_local_repair_variant": metadata["is_local_repair_variant"],
                    "repair_base_variant": metadata["repair_base_variant"],
                    "repair_base_short_name": metadata["repair_base_short_name"],
                }
            )

    dump_yaml(
        {
            key: [str(stage_id) for stage_id in stage_ids]
            for key, stage_ids in dispatch_result.anchor_blocks.items()
        },
        dispatch_dir / "dispatch_anchor_blocks.yaml",
    )
    for anchor_key, stage_2_job_sequence in dispatch_result.anchor_stage_sequences.items():
        dump_yaml(
            {
                str(stage_id): [str(job_id) for job_id in job_ids]
                for stage_id, job_ids in stage_2_job_sequence.items()
            },
            dispatch_dir / f"anchor_{anchor_key}_stage_job_sequence.yaml",
        )
    for anchor_key, stage_2_job_release in dispatch_result.anchor_stage_releases.items():
        dump_yaml(
            {
                str(stage_id): {
                    str(job_id): int(release_t)
                    for job_id, release_t in job_2_release.items()
                }
                for stage_id, job_2_release in stage_2_job_release.items()
            },
            dispatch_dir / f"anchor_{anchor_key}_stage_job_release.yaml",
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


def _resolve_anchor_blocks(
    stage_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
) -> dict[str, list[str]]:
    if not stage_id_list:
        return {}

    first_stage_id = stage_id_list[0]
    last_stage_id = stage_id_list[-1]
    internal_retained_stage_ids = [
        str(stage_id)
        for stage_id in retained_cp_result.retained_stage_ids
        if str(stage_id) not in {first_stage_id, last_stage_id}
    ]
    blocks = _get_contiguous_stage_blocks(stage_id_list, internal_retained_stage_ids)
    if not blocks:
        return {}

    preferred_block = _select_preferred_anchor_block(
        blocks,
        stage_id_list=stage_id_list,
        bottleneck_stage_id=retained_cp_result.bottleneck_stage_id,
    )

    named_blocks: list[tuple[str, list[str]]] = [("preferred", preferred_block)]
    named_blocks.append(("first", blocks[0]))
    named_blocks.append(("last", blocks[-1]))

    anchor_blocks: dict[str, list[str]] = {}
    seen_stage_tuples: set[tuple[str, ...]] = set()
    for anchor_key, stage_ids in named_blocks:
        stage_tuple = tuple(stage_ids)
        if stage_tuple in seen_stage_tuples:
            continue
        seen_stage_tuples.add(stage_tuple)
        anchor_blocks[anchor_key] = list(stage_ids)
    return anchor_blocks


def _get_contiguous_stage_blocks(
    stage_id_list: Sequence[str],
    retained_stage_ids: Sequence[str],
) -> list[list[str]]:
    if not retained_stage_ids:
        return []

    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}
    sorted_retained_stage_ids = sorted(
        [str(stage_id) for stage_id in retained_stage_ids],
        key=lambda stage_id: stage_2_index[stage_id],
    )
    blocks: list[list[str]] = []
    current_block: list[str] = []
    former_idx: int | None = None
    for stage_id in sorted_retained_stage_ids:
        stage_idx = stage_2_index[stage_id]
        if former_idx is None or stage_idx == former_idx + 1:
            current_block.append(stage_id)
        else:
            blocks.append(current_block)
            current_block = [stage_id]
        former_idx = stage_idx
    if current_block:
        blocks.append(current_block)
    return blocks


def _select_preferred_anchor_block(
    blocks: Sequence[Sequence[str]],
    *,
    stage_id_list: Sequence[str],
    bottleneck_stage_id: str | None,
) -> list[str]:
    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}
    if bottleneck_stage_id is not None:
        for block in blocks:
            if str(bottleneck_stage_id) in block:
                return list(block)
    return list(
        max(
            blocks,
            key=lambda block: (len(block), -stage_2_index[str(block[0])]),
        )
    )


def _extract_anchor_stage_sequence_from_retained_rows(
    retained_solution_rows: Sequence[Mapping[str, Any]],
    anchor_stage_ids: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    anchor_stage_id_set = set(anchor_stage_ids)
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {
        str(stage_id): [] for stage_id in anchor_stage_ids
    }
    for row in retained_solution_rows:
        stage_id = str(row["stage_id"])
        if stage_id in anchor_stage_id_set:
            stage_2_rows[stage_id].append(row)

    stage_2_job_sequence: dict[str, list[str]] = {}
    stage_2_job_2_release: dict[str, dict[str, int]] = {}
    for stage_id in anchor_stage_ids:
        rows = list(stage_2_rows.get(str(stage_id), []))
        if not rows:
            raise ValueError(
                f"Missing retained-stage CP rows for anchor stage {stage_id}."
            )
        rows.sort(
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["job_id"]),
            )
        )
        stage_2_job_sequence[str(stage_id)] = [str(row["job_id"]) for row in rows]
        stage_2_job_2_release[str(stage_id)] = {
            str(row["job_id"]): int(row["start"]) for row in rows
        }
    return stage_2_job_sequence, stage_2_job_2_release


def _resolve_retained_bottleneck_stage_id(
    stage_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
) -> str | None:
    retained_stage_set = {str(row["stage_id"]) for row in retained_solution_rows}
    if (
        retained_cp_result.bottleneck_stage_id is not None
        and str(retained_cp_result.bottleneck_stage_id) in retained_stage_set
    ):
        return str(retained_cp_result.bottleneck_stage_id)
    for stage_id in retained_cp_result.selected_bottleneck_stage_ids:
        if str(stage_id) in retained_stage_set:
            return str(stage_id)
    for stage_id in stage_id_list:
        if stage_id in retained_stage_set:
            return str(stage_id)
    return None


def _build_direct_mixed_sequences(
    *,
    instance: HybridFlowshopParameters,
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_first_stage_id: str | None,
    preferred_anchor_last_stage_id: str | None,
    bottleneck_stage_id: str | None,
) -> dict[str, list[str]]:
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in retained_solution_rows:
        stage_id = str(row["stage_id"])
        stage_2_rows.setdefault(stage_id, []).append(row)

    def stage_sequence(stage_id: str | None) -> list[str] | None:
        if stage_id is None:
            return None
        rows = list(stage_2_rows.get(str(stage_id), []))
        if not rows:
            return None
        rows.sort(
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["job_id"]),
            )
        )
        return [str(row["job_id"]) for row in rows]

    aggregate_sequence = _get_job_sequence_from_retained_rows_aggregate(
        job_id_list=instance.job_id_list,
        retained_cp_result=retained_cp_result,
        retained_solution_rows=retained_solution_rows,
    )

    sequence_map: dict[str, list[str]] = {}
    maybe_first = stage_sequence(preferred_anchor_first_stage_id)
    if maybe_first:
        sequence_map["mixed_cp_first_anchor"] = maybe_first
    maybe_last = stage_sequence(preferred_anchor_last_stage_id)
    if maybe_last:
        sequence_map["mixed_cp_last_anchor"] = maybe_last
    maybe_bottleneck = stage_sequence(bottleneck_stage_id)
    if maybe_bottleneck:
        sequence_map["mixed_cp_bottleneck_anchor"] = maybe_bottleneck
    sequence_map["mixed_cp_aggregate_start_slack"] = aggregate_sequence
    return sequence_map


def _build_full_stage_sequences_from_retained_rows(
    *,
    stage_id_list: Sequence[str],
    retained_solution_rows: Sequence[Mapping[str, Any]],
    fallback_job_sequence: Sequence[str],
) -> dict[str, list[str]]:
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in retained_solution_rows:
        stage_id = str(row["stage_id"])
        stage_2_rows.setdefault(stage_id, []).append(row)

    retained_stage_ids = [
        stage_id for stage_id in stage_id_list if stage_id in stage_2_rows
    ]
    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}

    retained_stage_sequences: dict[str, list[str]] = {}
    for stage_id in retained_stage_ids:
        rows = list(stage_2_rows[stage_id])
        rows.sort(
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["job_id"]),
            )
        )
        retained_stage_sequences[stage_id] = [str(row["job_id"]) for row in rows]

    full_stage_sequences: dict[str, list[str]] = {}
    for stage_id in stage_id_list:
        if stage_id in retained_stage_sequences:
            full_stage_sequences[str(stage_id)] = list(retained_stage_sequences[stage_id])
            continue
        if retained_stage_ids:
            nearest_retained_stage_id = min(
                retained_stage_ids,
                key=lambda retained_stage_id: (
                    abs(stage_2_index[stage_id] - stage_2_index[retained_stage_id]),
                    stage_2_index[retained_stage_id],
                ),
            )
            full_stage_sequences[str(stage_id)] = list(
                retained_stage_sequences[nearest_retained_stage_id]
            )
        else:
            full_stage_sequences[str(stage_id)] = list(fallback_job_sequence)
    return full_stage_sequences


def _build_piecewise_stage_sequence_variants(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_solution_rows: Sequence[Mapping[str, Any]],
    fallback_job_sequence: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    retained_stage_ids, retained_stage_sequences = _get_retained_stage_sequences(
        stage_id_list=stage_id_list,
        job_id_list=job_id_list,
        retained_solution_rows=retained_solution_rows,
    )
    if not retained_stage_ids:
        fallback = {
            str(stage_id): _complete_job_sequence(fallback_job_sequence, job_id_list)
            for stage_id in stage_id_list
        }
        return {"piecewise_cp_nearest_priority": fallback}

    return {
        "piecewise_cp_nearest_priority": _build_piecewise_stage_sequences_by_policy(
            stage_id_list=stage_id_list,
            job_id_list=job_id_list,
            retained_stage_ids=retained_stage_ids,
            retained_stage_sequences=retained_stage_sequences,
            policy="nearest",
        ),
        "piecewise_cp_left_priority": _build_piecewise_stage_sequences_by_policy(
            stage_id_list=stage_id_list,
            job_id_list=job_id_list,
            retained_stage_ids=retained_stage_ids,
            retained_stage_sequences=retained_stage_sequences,
            policy="left",
        ),
        "piecewise_cp_right_priority": _build_piecewise_stage_sequences_by_policy(
            stage_id_list=stage_id_list,
            job_id_list=job_id_list,
            retained_stage_ids=retained_stage_ids,
            retained_stage_sequences=retained_stage_sequences,
            policy="right",
        ),
        "piecewise_cp_blend_priority": _build_blended_piecewise_stage_sequences(
            stage_id_list=stage_id_list,
            job_id_list=job_id_list,
            retained_stage_ids=retained_stage_ids,
            retained_stage_sequences=retained_stage_sequences,
        ),
    }


def _get_retained_stage_sequences(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_solution_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in retained_solution_rows:
        stage_2_rows.setdefault(str(row["stage_id"]), []).append(row)

    retained_stage_ids = [
        str(stage_id) for stage_id in stage_id_list if str(stage_id) in stage_2_rows
    ]
    retained_stage_sequences: dict[str, list[str]] = {}
    for stage_id in retained_stage_ids:
        rows = list(stage_2_rows[stage_id])
        rows.sort(
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["job_id"]),
            )
        )
        retained_stage_sequences[stage_id] = _complete_job_sequence(
            [str(row["job_id"]) for row in rows],
            job_id_list,
        )
    return retained_stage_ids, retained_stage_sequences


def _complete_job_sequence(
    job_sequence: Sequence[str],
    job_id_list: Sequence[str],
) -> list[str]:
    seen: set[str] = set()
    completed_sequence: list[str] = []
    for job_id in job_sequence:
        job_id_str = str(job_id)
        if job_id_str in seen:
            continue
        completed_sequence.append(job_id_str)
        seen.add(job_id_str)
    for job_id in job_id_list:
        job_id_str = str(job_id)
        if job_id_str not in seen:
            completed_sequence.append(job_id_str)
            seen.add(job_id_str)
    return completed_sequence


def _build_piecewise_stage_sequences_by_policy(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_stage_ids: Sequence[str],
    retained_stage_sequences: Mapping[str, Sequence[str]],
    policy: str,
) -> dict[str, list[str]]:
    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}
    retained_stage_indices = [
        stage_2_index[str(stage_id)] for stage_id in retained_stage_ids
    ]
    stage_2_sequence: dict[str, list[str]] = {}
    for stage_id in stage_id_list:
        stage_id_str = str(stage_id)
        stage_idx = stage_2_index[stage_id_str]
        selected_stage_id = _select_retained_stage_for_piecewise_policy(
            stage_idx=stage_idx,
            stage_2_index=stage_2_index,
            retained_stage_ids=retained_stage_ids,
            retained_stage_indices=retained_stage_indices,
            policy=policy,
        )
        stage_2_sequence[stage_id_str] = _complete_job_sequence(
            retained_stage_sequences[selected_stage_id],
            job_id_list,
        )
    return stage_2_sequence


def _select_retained_stage_for_piecewise_policy(
    *,
    stage_idx: int,
    stage_2_index: Mapping[str, int],
    retained_stage_ids: Sequence[str],
    retained_stage_indices: Sequence[int],
    policy: str,
) -> str:
    if policy == "left":
        eligible_stage_ids = [
            stage_id
            for stage_id in retained_stage_ids
            if stage_2_index[str(stage_id)] <= stage_idx
        ]
        selected_stage_id = (
            eligible_stage_ids[-1] if eligible_stage_ids else retained_stage_ids[0]
        )
        return str(selected_stage_id)
    if policy == "right":
        eligible_stage_ids = [
            stage_id
            for stage_id in retained_stage_ids
            if stage_2_index[str(stage_id)] >= stage_idx
        ]
        selected_stage_id = (
            eligible_stage_ids[0] if eligible_stage_ids else retained_stage_ids[-1]
        )
        return str(selected_stage_id)
    if policy != "nearest":
        raise ValueError(f"Unknown piecewise stage sequence policy: {policy!r}")
    nearest_index = min(
        retained_stage_indices,
        key=lambda retained_idx: (
            abs(stage_idx - retained_idx),
            retained_idx,
        ),
    )
    return str(retained_stage_ids[retained_stage_indices.index(nearest_index)])


def _build_blended_piecewise_stage_sequences(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_stage_ids: Sequence[str],
    retained_stage_sequences: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}
    retained_stage_indices = [
        stage_2_index[str(retained_stage_id)]
        for retained_stage_id in retained_stage_ids
    ]
    stage_2_sequence: dict[str, list[str]] = {}
    for stage_id in stage_id_list:
        stage_id_str = str(stage_id)
        stage_idx = stage_2_index[stage_id_str]
        left_stage_id = _select_retained_stage_for_piecewise_policy(
            stage_idx=stage_idx,
            stage_2_index=stage_2_index,
            retained_stage_ids=retained_stage_ids,
            retained_stage_indices=retained_stage_indices,
            policy="left",
        )
        right_stage_id = _select_retained_stage_for_piecewise_policy(
            stage_idx=stage_idx,
            stage_2_index=stage_2_index,
            retained_stage_ids=retained_stage_ids,
            retained_stage_indices=retained_stage_indices,
            policy="right",
        )
        left_idx = stage_2_index[left_stage_id]
        right_idx = stage_2_index[right_stage_id]
        left_sequence = _complete_job_sequence(
            retained_stage_sequences[left_stage_id],
            job_id_list,
        )
        if left_stage_id == right_stage_id or left_idx == right_idx:
            stage_2_sequence[stage_id_str] = left_sequence
            continue
        right_sequence = _complete_job_sequence(
            retained_stage_sequences[right_stage_id],
            job_id_list,
        )
        right_weight = (stage_idx - left_idx) / max(1, right_idx - left_idx)
        stage_2_sequence[stage_id_str] = _blend_job_sequences_by_rank(
            job_id_list=job_id_list,
            left_sequence=left_sequence,
            right_sequence=right_sequence,
            right_weight=right_weight,
        )
    return stage_2_sequence


def _blend_job_sequences_by_rank(
    *,
    job_id_list: Sequence[str],
    left_sequence: Sequence[str],
    right_sequence: Sequence[str],
    right_weight: float,
) -> list[str]:
    clamped_right_weight = min(1.0, max(0.0, float(right_weight)))
    left_weight = 1.0 - clamped_right_weight
    left_rank = {str(job_id): idx for idx, job_id in enumerate(left_sequence)}
    right_rank = {str(job_id): idx for idx, job_id in enumerate(right_sequence)}
    job_id_order = {str(job_id): idx for idx, job_id in enumerate(job_id_list)}
    missing_rank = len(job_id_order)

    sortable_rows: list[tuple[tuple[float, float, float, float], str]] = []
    for job_id in job_id_list:
        job_id_str = str(job_id)
        left_pos = float(left_rank.get(job_id_str, missing_rank))
        right_pos = float(right_rank.get(job_id_str, missing_rank))
        sortable_rows.append(
            (
                (
                    left_weight * left_pos + clamped_right_weight * right_pos,
                    right_pos,
                    left_pos,
                    float(job_id_order[job_id_str]),
                ),
                job_id_str,
            )
        )
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def _get_retained_stage_weights(
    *,
    stage_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_stage_ids: Sequence[str],
    preferred_anchor_stage_ids: Sequence[str],
) -> dict[str, float]:
    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}
    denominator = max(len(stage_id_list) - 1, 1)
    preferred_set = {str(stage_id) for stage_id in preferred_anchor_stage_ids}
    bottleneck_set = {
        str(stage_id) for stage_id in retained_cp_result.selected_bottleneck_stage_ids
    }
    if retained_cp_result.bottleneck_stage_id is not None:
        bottleneck_set.add(str(retained_cp_result.bottleneck_stage_id))

    weights: dict[str, float] = {}
    for stage_id in retained_stage_ids:
        idx = stage_2_index[str(stage_id)]
        tail_bonus = idx / denominator
        weight = 1.0 + tail_bonus
        if str(stage_id) in preferred_set:
            weight += 0.75
        if str(stage_id) in bottleneck_set:
            weight += 1.75
        weights[str(stage_id)] = weight
    return weights


def _get_job_sequence_from_retained_rows_consensus(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_stage_ids: Sequence[str],
) -> list[str]:
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in retained_solution_rows:
        stage_2_rows.setdefault(str(row["stage_id"]), []).append(row)
    retained_stage_ids = [
        stage_id for stage_id in stage_id_list if stage_id in stage_2_rows
    ]
    stage_weights = _get_retained_stage_weights(
        stage_id_list=stage_id_list,
        retained_cp_result=retained_cp_result,
        retained_stage_ids=retained_stage_ids,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )

    job_2_score = {str(job_id): 0.0 for job_id in job_id_list}
    job_2_weight = {str(job_id): 0.0 for job_id in job_id_list}
    job_2_latest_start = {str(job_id): 0.0 for job_id in job_id_list}
    job_2_slack = {str(job_id): 0.0 for job_id in job_id_list}
    objective_ub = (
        float(retained_cp_result.objective_ub)
        if retained_cp_result.objective_ub is not None
        else None
    )

    for stage_id in retained_stage_ids:
        rows = list(stage_2_rows[str(stage_id)])
        rows.sort(
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["job_id"]),
            )
        )
        stage_weight = stage_weights[str(stage_id)]
        for rank, row in enumerate(rows):
            job_id = str(row["job_id"])
            p = float(row["processing_time"])
            head_t = float(row["head"])
            tail_t = float(row["tail"])
            latest_start = (
                max(head_t, objective_ub - tail_t - p)
                if objective_ub is not None
                else float(row["start"])
            )
            slack = latest_start - head_t
            job_2_score[job_id] += stage_weight * float(rank)
            job_2_weight[job_id] += stage_weight
            job_2_latest_start[job_id] += stage_weight * latest_start
            job_2_slack[job_id] += stage_weight * slack

    sortable_rows: list[tuple[tuple[float, ...], str]] = []
    for job_idx, job_id in enumerate(job_id_list):
        total_weight = max(job_2_weight[str(job_id)], 1e-9)
        sortable_rows.append(
            (
                (
                    job_2_score[str(job_id)] / total_weight,
                    job_2_latest_start[str(job_id)] / total_weight,
                    job_2_slack[str(job_id)] / total_weight,
                    float(job_idx),
                ),
                str(job_id),
            )
        )
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def _get_job_sequence_from_retained_rows_tail_bottleneck(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_stage_ids: Sequence[str],
) -> list[str]:
    objective_ub = (
        float(retained_cp_result.objective_ub)
        if retained_cp_result.objective_ub is not None
        else None
    )
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in retained_solution_rows:
        stage_2_rows.setdefault(str(row["stage_id"]), []).append(row)
    retained_stage_ids = [
        stage_id for stage_id in stage_id_list if stage_id in stage_2_rows
    ]
    if not retained_stage_ids:
        return list(job_id_list)
    stage_weights = _get_retained_stage_weights(
        stage_id_list=stage_id_list,
        retained_cp_result=retained_cp_result,
        retained_stage_ids=retained_stage_ids,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )
    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}
    last_retained_stage_id = max(
        retained_stage_ids, key=lambda stage_id: stage_2_index[str(stage_id)]
    )
    bottleneck_stage_id = _resolve_retained_bottleneck_stage_id(
        stage_id_list,
        retained_cp_result,
        retained_solution_rows,
    )

    job_2_metrics: dict[str, dict[str, float]] = {
        str(job_id): {
            "weighted_start": 0.0,
            "weighted_latest_start": 0.0,
            "weighted_slack": 0.0,
            "tail_latest_start": 0.0,
            "bneck_latest_start": 0.0,
            "total_p": 0.0,
        }
        for job_id in job_id_list
    }

    for row in retained_solution_rows:
        stage_id = str(row["stage_id"])
        job_id = str(row["job_id"])
        start_t = float(row["start"])
        p = float(row["processing_time"])
        head_t = float(row["head"])
        tail_t = float(row["tail"])
        latest_start = (
            max(head_t, objective_ub - tail_t - p)
            if objective_ub is not None
            else start_t
        )
        slack = latest_start - head_t
        weight = stage_weights.get(stage_id, 1.0)
        metrics = job_2_metrics[job_id]
        metrics["weighted_start"] += weight * start_t
        metrics["weighted_latest_start"] += weight * latest_start
        metrics["weighted_slack"] += weight * slack
        metrics["total_p"] += p
        if stage_id == last_retained_stage_id:
            metrics["tail_latest_start"] = latest_start
        if bottleneck_stage_id is not None and stage_id == bottleneck_stage_id:
            metrics["bneck_latest_start"] = latest_start

    sortable_rows: list[tuple[tuple[float, ...], str]] = []
    for job_idx, job_id in enumerate(job_id_list):
        metrics = job_2_metrics[str(job_id)]
        sortable_rows.append(
            (
                (
                    metrics["tail_latest_start"],
                    metrics["bneck_latest_start"],
                    metrics["weighted_latest_start"],
                    metrics["weighted_slack"],
                    metrics["weighted_start"],
                    -metrics["total_p"],
                    float(job_idx),
                ),
                str(job_id),
            )
        )
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def _build_extended_rank_sequences(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_stage_ids: Sequence[str],
) -> dict[str, list[str]]:
    return {
        "best_of_mixed_dispatches_cp_weighted_median_rank": (
            _get_job_sequence_from_retained_rows_weighted_median(
                stage_id_list=stage_id_list,
                job_id_list=job_id_list,
                retained_cp_result=retained_cp_result,
                retained_solution_rows=retained_solution_rows,
                preferred_anchor_stage_ids=preferred_anchor_stage_ids,
            )
        ),
        "best_of_mixed_dispatches_cp_slack_urgency_rank": (
            _get_job_sequence_from_retained_rows_slack_urgency(
                stage_id_list=stage_id_list,
                job_id_list=job_id_list,
                retained_cp_result=retained_cp_result,
                retained_solution_rows=retained_solution_rows,
                preferred_anchor_stage_ids=preferred_anchor_stage_ids,
            )
        ),
        "best_of_mixed_dispatches_cp_front_tail_blend_rank": (
            _get_job_sequence_from_retained_rows_front_tail_blend(
                stage_id_list=stage_id_list,
                job_id_list=job_id_list,
                retained_cp_result=retained_cp_result,
                retained_solution_rows=retained_solution_rows,
                preferred_anchor_stage_ids=preferred_anchor_stage_ids,
            )
        ),
    }


def _get_job_sequence_from_retained_rows_weighted_median(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_stage_ids: Sequence[str],
) -> list[str]:
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in retained_solution_rows:
        stage_2_rows.setdefault(str(row["stage_id"]), []).append(row)
    retained_stage_ids = [
        stage_id for stage_id in stage_id_list if stage_id in stage_2_rows
    ]
    if not retained_stage_ids:
        return [str(job_id) for job_id in job_id_list]

    stage_weights = _get_retained_stage_weights(
        stage_id_list=stage_id_list,
        retained_cp_result=retained_cp_result,
        retained_stage_ids=retained_stage_ids,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )
    objective_ub = (
        float(retained_cp_result.objective_ub)
        if retained_cp_result.objective_ub is not None
        else None
    )
    job_2_rank_weight_pairs: dict[str, list[tuple[float, float]]] = {
        str(job_id): [] for job_id in job_id_list
    }
    job_2_weighted_rank = {str(job_id): 0.0 for job_id in job_id_list}
    job_2_weighted_latest_start = {str(job_id): 0.0 for job_id in job_id_list}
    job_2_weighted_slack = {str(job_id): 0.0 for job_id in job_id_list}
    job_2_weight = {str(job_id): 0.0 for job_id in job_id_list}

    for stage_id in retained_stage_ids:
        rows = list(stage_2_rows[str(stage_id)])
        rows.sort(
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["job_id"]),
            )
        )
        stage_weight = stage_weights.get(str(stage_id), 1.0)
        for rank, row in enumerate(rows):
            job_id = str(row["job_id"])
            p = float(row["processing_time"])
            head_t = float(row["head"])
            tail_t = float(row["tail"])
            latest_start = (
                max(head_t, objective_ub - tail_t - p)
                if objective_ub is not None
                else float(row["start"])
            )
            slack = latest_start - head_t
            job_2_rank_weight_pairs[job_id].append((float(rank), stage_weight))
            job_2_weighted_rank[job_id] += stage_weight * float(rank)
            job_2_weighted_latest_start[job_id] += stage_weight * latest_start
            job_2_weighted_slack[job_id] += stage_weight * slack
            job_2_weight[job_id] += stage_weight

    sortable_rows: list[tuple[tuple[float, ...], str]] = []
    for job_idx, job_id in enumerate(job_id_list):
        job_id_str = str(job_id)
        total_weight = max(job_2_weight[job_id_str], 1e-9)
        sortable_rows.append(
            (
                (
                    _weighted_median_rank(job_2_rank_weight_pairs[job_id_str]),
                    job_2_weighted_rank[job_id_str] / total_weight,
                    job_2_weighted_latest_start[job_id_str] / total_weight,
                    job_2_weighted_slack[job_id_str] / total_weight,
                    float(job_idx),
                ),
                job_id_str,
            )
        )
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def _weighted_median_rank(rank_weight_pairs: Sequence[tuple[float, float]]) -> float:
    if not rank_weight_pairs:
        return float("inf")
    total_weight = sum(max(0.0, weight) for _rank, weight in rank_weight_pairs)
    if total_weight <= 0.0:
        return min(rank for rank, _weight in rank_weight_pairs)
    cumulative_weight = 0.0
    for rank, weight in sorted(rank_weight_pairs, key=lambda item: item[0]):
        cumulative_weight += max(0.0, weight)
        if cumulative_weight >= total_weight / 2.0:
            return rank
    return max(rank for rank, _weight in rank_weight_pairs)


def _get_job_sequence_from_retained_rows_slack_urgency(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_stage_ids: Sequence[str],
) -> list[str]:
    metrics = _summarize_retained_job_metrics(
        stage_id_list=stage_id_list,
        job_id_list=job_id_list,
        retained_cp_result=retained_cp_result,
        retained_solution_rows=retained_solution_rows,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )
    sortable_rows: list[tuple[tuple[float, ...], str]] = []
    for job_idx, job_id in enumerate(job_id_list):
        job_id_str = str(job_id)
        row = metrics[job_id_str]
        total_weight = max(row["weight"], 1e-9)
        sortable_rows.append(
            (
                (
                    row["weighted_slack"] / total_weight,
                    row["weighted_latest_start"] / total_weight,
                    row["weighted_start"] / total_weight,
                    -row["total_p"],
                    float(job_idx),
                ),
                job_id_str,
            )
        )
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def _get_job_sequence_from_retained_rows_front_tail_blend(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_stage_ids: Sequence[str],
) -> list[str]:
    metrics = _summarize_retained_job_metrics(
        stage_id_list=stage_id_list,
        job_id_list=job_id_list,
        retained_cp_result=retained_cp_result,
        retained_solution_rows=retained_solution_rows,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )
    sortable_rows: list[tuple[tuple[float, ...], str]] = []
    for job_idx, job_id in enumerate(job_id_list):
        job_id_str = str(job_id)
        row = metrics[job_id_str]
        total_weight = max(row["weight"], 1e-9)
        front_start = row["first_start"]
        if front_start == float("inf"):
            front_start = row["weighted_start"] / total_weight
        tail_latest = row["last_latest_start"]
        if tail_latest == float("inf"):
            tail_latest = row["weighted_latest_start"] / total_weight
        blended_time = 0.45 * front_start + 0.55 * tail_latest
        sortable_rows.append(
            (
                (
                    blended_time,
                    tail_latest,
                    front_start,
                    row["weighted_rank"] / total_weight,
                    row["weighted_slack"] / total_weight,
                    -row["total_p"],
                    float(job_idx),
                ),
                job_id_str,
            )
        )
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def _summarize_retained_job_metrics(
    *,
    stage_id_list: Sequence[str],
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
    preferred_anchor_stage_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    objective_ub = (
        float(retained_cp_result.objective_ub)
        if retained_cp_result.objective_ub is not None
        else None
    )
    stage_2_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in retained_solution_rows:
        stage_2_rows.setdefault(str(row["stage_id"]), []).append(row)
    retained_stage_ids = [
        stage_id for stage_id in stage_id_list if stage_id in stage_2_rows
    ]
    stage_weights = _get_retained_stage_weights(
        stage_id_list=stage_id_list,
        retained_cp_result=retained_cp_result,
        retained_stage_ids=retained_stage_ids,
        preferred_anchor_stage_ids=preferred_anchor_stage_ids,
    )
    stage_2_index = {str(stage_id): idx for idx, stage_id in enumerate(stage_id_list)}
    first_retained_stage_id = (
        min(retained_stage_ids, key=lambda stage_id: stage_2_index[str(stage_id)])
        if retained_stage_ids
        else None
    )
    last_retained_stage_id = (
        max(retained_stage_ids, key=lambda stage_id: stage_2_index[str(stage_id)])
        if retained_stage_ids
        else None
    )
    metrics: dict[str, dict[str, float]] = {
        str(job_id): {
            "weighted_rank": 0.0,
            "weighted_start": 0.0,
            "weighted_latest_start": 0.0,
            "weighted_slack": 0.0,
            "weight": 0.0,
            "first_start": float("inf"),
            "last_latest_start": float("inf"),
            "total_p": 0.0,
        }
        for job_id in job_id_list
    }
    for stage_id in retained_stage_ids:
        rows = list(stage_2_rows[str(stage_id)])
        rows.sort(
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["job_id"]),
            )
        )
        stage_weight = stage_weights.get(str(stage_id), 1.0)
        for rank, row in enumerate(rows):
            job_id = str(row["job_id"])
            start_t = float(row["start"])
            p = float(row["processing_time"])
            head_t = float(row["head"])
            tail_t = float(row["tail"])
            latest_start = (
                max(head_t, objective_ub - tail_t - p)
                if objective_ub is not None
                else start_t
            )
            slack = latest_start - head_t
            job_metrics = metrics[job_id]
            job_metrics["weighted_rank"] += stage_weight * float(rank)
            job_metrics["weighted_start"] += stage_weight * start_t
            job_metrics["weighted_latest_start"] += stage_weight * latest_start
            job_metrics["weighted_slack"] += stage_weight * slack
            job_metrics["weight"] += stage_weight
            job_metrics["total_p"] += p
            if stage_id == first_retained_stage_id:
                job_metrics["first_start"] = start_t
            if stage_id == last_retained_stage_id:
                job_metrics["last_latest_start"] = latest_start
    return metrics


def _get_job_sequence_from_retained_rows_aggregate(
    *,
    job_id_list: Sequence[str],
    retained_cp_result: RetainedStageCpResult,
    retained_solution_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    objective_ub = (
        float(retained_cp_result.objective_ub)
        if retained_cp_result.objective_ub is not None
        else None
    )
    job_2_metrics: dict[str, dict[str, float]] = {
        str(job_id): {
            "sum_start": 0.0,
            "sum_head": 0.0,
            "sum_latest_start": 0.0,
            "sum_slack": 0.0,
            "total_p": 0.0,
        }
        for job_id in job_id_list
    }
    for row in retained_solution_rows:
        job_id = str(row["job_id"])
        start_t = float(row["start"])
        processing_time = float(row["processing_time"])
        head_t = float(row["head"])
        tail_t = float(row["tail"])
        latest_start = (
            max(head_t, objective_ub - tail_t - processing_time)
            if objective_ub is not None
            else start_t
        )
        slack = latest_start - head_t
        metrics = job_2_metrics[job_id]
        metrics["sum_start"] += start_t
        metrics["sum_head"] += head_t
        metrics["sum_latest_start"] += latest_start
        metrics["sum_slack"] += slack
        metrics["total_p"] += processing_time

    sortable_rows: list[tuple[tuple[float, ...], str]] = []
    for job_idx, job_id in enumerate(job_id_list):
        metrics = job_2_metrics[str(job_id)]
        sortable_rows.append(
            (
                (
                    metrics["sum_start"],
                    metrics["sum_latest_start"],
                    metrics["sum_slack"],
                    -metrics["total_p"],
                    float(job_idx),
                ),
                str(job_id),
            )
        )
    sortable_rows.sort(key=lambda row: row[0])
    return [job_id for _key, job_id in sortable_rows]


def _evaluate_anchor_band_candidates(
    *,
    dispatch_candidates: dict[str, HybridFlowshopLiteSchedule | None],
    dispatch_candidate_elapsed_sec: dict[str, float],
    variant_2_anchor_stage_ids: dict[str, list[str]],
    dependencies: PostRetainedCpDispatchDependencies,
    option_kwargs: Mapping[str, Any],
    anchor_key: str,
    anchor_stage_ids: Sequence[str],
    stage_2_job_sequence: Mapping[str, Sequence[str]],
    stage_2_job_2_release: Mapping[str, Mapping[str, int]],
    include_release_candidates: bool,
) -> None:
    variant_specs: list[tuple[str, str, Mapping[str, Mapping[str, int]] | None]] = [
        ("strict_start_release", "strict_start", stage_2_job_2_release),
        ("strict_start_no_release", "strict_start", None),
        ("strict_call_release", "strict_call", stage_2_job_2_release),
        ("priority_release", "priority", stage_2_job_2_release),
    ]
    if not include_release_candidates:
        variant_specs = [
            ("strict_start_no_release", "strict_start", None),
        ]

    for suffix, anchor_dispatch_mode, release_map in variant_specs:
        variant = f"cp_band_{anchor_key}_{suffix}"
        variant_timer = ElapsedTimer()
        try:
            schedule = dependencies.get_two_way_schedule_by_stage_band(
                anchor_stage_ids=anchor_stage_ids,
                stage_2_job_sequence=stage_2_job_sequence,
                mixed_schedule_for_former_stages=option_kwargs[
                    "mixed_schedule_for_former_stages"
                ],
                mixed_schedule_for_later_stages=option_kwargs[
                    "mixed_schedule_for_later_stages"
                ],
                machine_then_job=option_kwargs["machine_then_job"],
                stage_2_job_2_release=release_map,
                anchor_dispatch_mode=anchor_dispatch_mode,
            )
        except Exception:
            logging.exception(
                "[CP LB] %s failed while constructing anchor-band dispatch.",
                variant,
            )
            schedule = None
        dispatch_candidates[variant] = schedule
        dispatch_candidate_elapsed_sec[variant] = variant_timer.elapsed_sec
        variant_2_anchor_stage_ids[variant] = list(anchor_stage_ids)
        logging.info(
            "[CP LB] %s has makespan=%s",
            variant,
            schedule.makespan if schedule is not None else None,
        )


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


def _get_anchor_stage_release_by_stage_ids(
    *,
    anchor_stage_releases: Mapping[str, Mapping[str, Mapping[str, int]]],
    anchor_stage_ids: Sequence[str],
) -> Mapping[str, Mapping[str, int]] | None:
    if not anchor_stage_ids:
        return None
    anchor_stage_id_tuple = tuple(str(stage_id) for stage_id in anchor_stage_ids)
    for stage_2_job_2_release in anchor_stage_releases.values():
        release_stage_id_tuple = tuple(
            str(stage_id) for stage_id in stage_2_job_2_release
        )
        if release_stage_id_tuple == anchor_stage_id_tuple:
            return stage_2_job_2_release
    return None


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
        is_local_repair = variant == "selected_post_retained_cp_local_repair" or variant.startswith(
            "post_retained_cp_local_repair__"
        )
        ties_best = float(gap_to_best or 0.0) == 0.0

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
        obj_token = f"obj{int(float(makespan))}" if makespan is not None else "objNA"
        gap_token = (
            f"gap{int(float(gap_to_best))}" if gap_to_best is not None else "gapNA"
        )
        artifact_slug = (
            f"{rank:02d}_{status_slug}__{short_name}__{obj_token}_{gap_token}"
        )
        metadata_by_variant[variant] = {
            "rank": rank,
            "variant": variant,
            "artifact_slug": artifact_slug,
            "short_name": short_name,
            "primary_role": primary_role,
            "status_tags": list(tags),
            "status_tags_text": " | ".join(tags),
            "status_summary": " | ".join(tags),
            "is_selected_final": is_selected,
            "is_pre_repair_base": is_pre_repair_base,
            "is_local_repair_variant": is_local_repair,
            "repair_base_variant": (
                pre_local_repair_selected_variant if is_local_repair else None
            ),
            "repair_base_short_name": (
                _get_dispatch_variant_short_name(pre_local_repair_selected_variant)
                if is_local_repair and pre_local_repair_selected_variant is not None
                else None
            ),
            "makespan": makespan,
            "gap_to_best": gap_to_best,
            "elapsed_sec": ranking.get("elapsed_sec"),
        }
    return metadata_by_variant


def _get_dispatch_variant_short_name(variant: str | None) -> str | None:
    if variant is None:
        return None
    alias_map = {
        "cp_band_preferred_strict_start_release": "band_pref_start_rel",
        "cp_band_preferred_strict_start_no_release": "band_pref_start_free",
        "cp_band_preferred_strict_call_release": "band_pref_call_rel",
        "cp_band_preferred_priority_release": "band_pref_prio_rel",
        "cp_band_first_strict_start_release": "band_first_start_rel",
        "cp_band_last_strict_start_release": "band_last_start_rel",
        "mixed_cp_first_anchor": "mixed_cp_first",
        "mixed_cp_last_anchor": "mixed_cp_last",
        "mixed_cp_bottleneck_anchor": "mixed_cp_bneck",
        "mixed_cp_aggregate_start_slack": "mixed_cp_agg",
        "mixed_cp_consensus": "mixed_cp_cons",
        "mixed_cp_tail_bottleneck": "mixed_cp_tailbn",
        "best_of_mixed_dispatches_cp_consensus_rank": "best_mixed_cp_cons",
        "best_of_mixed_dispatches_cp_tail_bottleneck_rank": "best_mixed_cp_tailbn",
        "cp_dynamic_priority_soft_release": "cp_dyn_prio",
        "best_of_mixed_dispatches_cp_first_anchor_rank": "best_mixed_cp_first",
        "best_of_mixed_dispatches_cp_last_anchor_rank": "best_mixed_cp_last",
        "best_of_mixed_dispatches_cp_bottleneck_rank": "best_mixed_cp_bneck",
        "best_of_mixed_dispatches_cp_aggregate_start_slack_rank": "best_mixed_cp_agg",
        "piecewise_cp_nearest_priority": "piece_near_prio",
        "piecewise_cp_left_priority": "piece_left_prio",
        "piecewise_cp_right_priority": "piece_right_prio",
        "piecewise_cp_blend_priority": "piece_blend_prio",
        "best_of_mixed_dispatches_cp_baseline": "best_mixed_base",
        "selected_post_retained_cp_local_repair": "local_repair",
    }
    return alias_map.get(variant, variant)
