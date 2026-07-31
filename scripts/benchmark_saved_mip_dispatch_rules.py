from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from routix.io.yaml import load_yaml
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.dispatcher import (
    MixedDispatcher,
    build_schedule_from_stage_job_sequences_priority_score,
    build_schedule_from_stage_job_sequences_strict_call_order,
    dispatch_stages_by_job_sequence,
    get_bottleneck_anchor_stage_from_solution_payload,
    get_job_sequence_from_dispatch_windows_aggregate,
    get_job_sequence_from_dispatch_windows_anchor_stage,
    get_job_tiebreak_rank_from_job_sequence,
    get_stage_job_sequences_from_dispatch_windows,
    improve_schedule_by_critical_adjacent_swaps,
    improve_schedule_by_critical_stage_sequence_insertions,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, validate_schedule
from lb_bucket.mip.dispatch_windows import build_dispatch_window_lookup
from lb_bucket.mip.solution_io import read_solution_payload

DEFAULT_INPUT_DIR = REPO_ROOT / "resources" / "ff2020big"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "Outputs_scenarios"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ES/LS-aware dispatch rules on saved MIP payloads."
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=None,
        help="Scenario directory like Outputs_scenarios/<ts>/ff2020/<scenario-name>.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing ff2020 instance text files.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of instance ids to benchmark.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-instance results.",
    )
    return parser.parse_args()


def _resolve_latest_scenario_dir(outputs_root: Path) -> Path:
    latest_timestamp_dir = max(
        (path for path in outputs_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    ff2020_dirs = sorted(
        (path for path in latest_timestamp_dir.glob("ff2020/*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not ff2020_dirs:
        raise FileNotFoundError(
            f"No ff2020 scenario directories were found under {latest_timestamp_dir}."
        )
    return ff2020_dirs[-1]


def _load_instance(input_dir: Path, ins_name: str) -> HybridFlowshopParameters:
    instance_path = input_dir / f"{ins_name}.txt"
    if not instance_path.is_file():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    with instance_path.open("r", encoding="utf-8") as handle:
        return HybridFlowshopParameters.from_ff2020_data(ins_name, handle)


def _get_bottleneck_stage_id(instance: HybridFlowshopParameters) -> str:
    return max(
        instance.stage_id_list,
        key=lambda stage_id: (
            sum(
                instance.stage_2_job_2_p_map[stage_id][job_id]
                for job_id in instance.job_id_list
            )
            / len(instance.stage_2_machines_map[stage_id])
        ),
    )


def _build_empty_schedule(
    instance: HybridFlowshopParameters,
) -> HybridFlowshopLiteSchedule:
    return HybridFlowshopLiteSchedule(
        jobs=instance.job_id_list,
        stages=instance.stage_id_list,
        machines_per_stage=instance.stage_2_machines_map,
    )


def _build_stage_dispatch_schedule(
    instance: HybridFlowshopParameters,
    job_sequence: list[str],
) -> HybridFlowshopLiteSchedule:
    schedule = _build_empty_schedule(instance)
    dispatch_stages_by_job_sequence(
        schedule,
        job_sequence,
        instance.stage_2_job_2_p_map,
    )
    return schedule


def _get_best_mixed_schedule_from_sequence(
    instance: HybridFlowshopParameters,
    job_sequence: list[str],
) -> HybridFlowshopLiteSchedule | None:
    dispatcher = MixedDispatcher(instance)
    return dispatcher.get_best_mixed_schedule_by_sequence(job_sequence)


def _get_best_of_mixed_dispatches_with_rank(
    instance: HybridFlowshopParameters,
    job_tiebreak_rank: dict[str, int],
) -> HybridFlowshopLiteSchedule | None:
    from schore.parameters_examples.parallel_shop.identical_flow.hybrid_flowshop import (
        reverse_stages,
    )

    def get_best_for_instance(
        target_instance: HybridFlowshopParameters,
    ) -> HybridFlowshopLiteSchedule | None:
        dispatcher = MixedDispatcher(
            target_instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        candidates = [
            dispatcher.get_schedule_by_cds(),
            dispatcher.get_schedule_by_gupta(),
            dispatcher.get_schedule_by_palmer(),
        ]
        candidates = [schedule for schedule in candidates if schedule is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda schedule: schedule.makespan)

    schedule = get_best_for_instance(instance)
    reversed_schedule = get_best_for_instance(reverse_stages(instance))
    if reversed_schedule is not None:
        reversed_schedule = reversed_schedule.as_reversed()
        reversed_schedule.make_semi_active(instance.stage_2_job_2_p_map)

    if schedule is None:
        return reversed_schedule
    if reversed_schedule is None or schedule.makespan <= reversed_schedule.makespan:
        return schedule
    return reversed_schedule


def _load_saved_dispatch_summary(instance_dir: Path) -> dict[str, Any]:
    dispatch_summary_path = (
        instance_dir / "mip_lb" / "dispatch" / "dispatch_summary.yaml"
    )
    if not dispatch_summary_path.is_file():
        return {}
    summary = load_yaml(dispatch_summary_path, encoding="utf-8")
    if not isinstance(summary, dict):
        return {}
    return summary


def _safe_makespan(schedule: HybridFlowshopLiteSchedule | None) -> int | None:
    if schedule is None:
        return None
    return int(schedule.makespan)


def _repair_schedule(
    instance: HybridFlowshopParameters,
    schedule: HybridFlowshopLiteSchedule | None,
    *,
    target_stage_ids: list[str] | None = None,
    insertion_passes: int = 3,
    max_shift: int = 4,
    swap_passes: int = 3,
) -> HybridFlowshopLiteSchedule | None:
    if schedule is None:
        return None

    candidate_pool = [schedule]
    try:
        inserted = improve_schedule_by_critical_stage_sequence_insertions(
            lambda: _build_empty_schedule(instance),
            schedule,
            instance.stage_2_job_2_p_map,
            target_stage_ids=target_stage_ids,
            max_passes=max(1, insertion_passes),
            max_shift=max(1, max_shift),
        )
        candidate_pool.append(inserted)
    except Exception:
        inserted = None

    try:
        swapped = improve_schedule_by_critical_adjacent_swaps(
            schedule,
            instance.stage_2_job_2_p_map,
            max_passes=max(1, swap_passes),
        )
        candidate_pool.append(swapped)
    except Exception:
        pass

    if inserted is not None:
        try:
            inserted_swapped = improve_schedule_by_critical_adjacent_swaps(
                inserted,
                instance.stage_2_job_2_p_map,
                max_passes=max(1, swap_passes),
            )
            candidate_pool.append(inserted_swapped)
        except Exception:
            pass

    return min(candidate_pool, key=lambda sch: sch.makespan)


def _validate_schedule(
    schedule: HybridFlowshopLiteSchedule | None,
    stage_2_job_2_p: dict[str, dict[str, int]],
) -> None:
    if schedule is None:
        return
    validate_schedule(schedule, stage_2_job_2_p)


def _evaluate_instance(
    scenario_dir: Path,
    input_dir: Path,
    ins_name: str,
) -> dict[str, Any] | None:
    instance_dir = scenario_dir / ins_name
    payload = read_solution_payload(instance_dir / "mip_lb", ins_name)
    if payload is None:
        return None

    instance = _load_instance(input_dir, ins_name)
    dispatch_window_lookup = build_dispatch_window_lookup(payload["dispatch_windows"])
    stage_ids = instance.stage_id_list
    job_ids = instance.job_id_list
    stage_2_job_2_p = instance.stage_2_job_2_p_map
    bottleneck_stage_id = _get_bottleneck_stage_id(instance)
    bottleneck_anchor_stage_id = get_bottleneck_anchor_stage_from_solution_payload(
        stage_ids,
        stage_2_job_2_p,
        instance.stage_2_machines_map,
        payload,
    )
    saved_dispatch_summary = _load_saved_dispatch_summary(instance_dir)
    saved_candidates = saved_dispatch_summary.get("dispatch_candidates", {}) or {}

    seq_stage_es_ls = get_stage_job_sequences_from_dispatch_windows(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        sort_rule="es_ls_p_desc",
    )
    seq_stage_ls_es = get_stage_job_sequences_from_dispatch_windows(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        sort_rule="ls_es_p_desc",
    )
    seq_stage_slack_ls = get_stage_job_sequences_from_dispatch_windows(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        sort_rule="slack_ls_es_p_desc",
    )
    seq_stage_midpoint = get_stage_job_sequences_from_dispatch_windows(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        sort_rule="midpoint_slack_ls_p_desc",
    )

    tail_stage_sequence = get_job_sequence_from_dispatch_windows_anchor_stage(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        anchor_stage_id=stage_ids[-1],
        sort_rule="ls_es_p_desc",
    )
    bottleneck_sequence = get_job_sequence_from_dispatch_windows_anchor_stage(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        anchor_stage_id=bottleneck_stage_id,
        sort_rule="slack_ls_es_p_desc",
    )
    aggregate_es_slack_sequence = get_job_sequence_from_dispatch_windows_aggregate(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        aggregation_rule="sum_es_slack_p_desc",
    )
    aggregate_ls_slack_sequence = get_job_sequence_from_dispatch_windows_aggregate(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        aggregation_rule="sum_ls_slack_p_desc",
    )
    anchor_midpoint_sequence = get_job_sequence_from_dispatch_windows_anchor_stage(
        stage_ids,
        job_ids,
        dispatch_window_lookup,
        stage_2_job_2_p,
        anchor_stage_id=bottleneck_anchor_stage_id,
        sort_rule="midpoint_slack_ls_p_desc",
    )
    aggregate_es_slack_rank = get_job_tiebreak_rank_from_job_sequence(
        aggregate_es_slack_sequence
    )
    aggregate_ls_slack_rank = get_job_tiebreak_rank_from_job_sequence(
        aggregate_ls_slack_sequence
    )
    candidate_schedules: dict[str, HybridFlowshopLiteSchedule | None] = {
        "priority_stage_es_ls": build_schedule_from_stage_job_sequences_priority_score(
            lambda: _build_empty_schedule(instance),
            seq_stage_es_ls,
            stage_2_job_2_p,
        ),
        "priority_stage_ls_es": build_schedule_from_stage_job_sequences_priority_score(
            lambda: _build_empty_schedule(instance),
            seq_stage_ls_es,
            stage_2_job_2_p,
        ),
        "priority_stage_slack_ls": build_schedule_from_stage_job_sequences_priority_score(
            lambda: _build_empty_schedule(instance),
            seq_stage_slack_ls,
            stage_2_job_2_p,
        ),
        "priority_stage_midpoint": build_schedule_from_stage_job_sequences_priority_score(
            lambda: _build_empty_schedule(instance),
            seq_stage_midpoint,
            stage_2_job_2_p,
        ),
        "strict_stage_es_ls": build_schedule_from_stage_job_sequences_strict_call_order(
            lambda: _build_empty_schedule(instance),
            seq_stage_es_ls,
            stage_2_job_2_p,
        ),
        "global_stage_tail_ls": _build_stage_dispatch_schedule(
            instance,
            tail_stage_sequence,
        ),
        "mixed_tail_ls": _get_best_mixed_schedule_from_sequence(
            instance,
            tail_stage_sequence,
        ),
        "mixed_bottleneck_slack": _get_best_mixed_schedule_from_sequence(
            instance,
            bottleneck_sequence,
        ),
        "mixed_aggregate_es_slack": _get_best_mixed_schedule_from_sequence(
            instance,
            aggregate_es_slack_sequence,
        ),
        "mixed_aggregate_ls_slack": _get_best_mixed_schedule_from_sequence(
            instance,
            aggregate_ls_slack_sequence,
        ),
        "best_mixed_rank_tail_ls": _get_best_of_mixed_dispatches_with_rank(
            instance,
            get_job_tiebreak_rank_from_job_sequence(tail_stage_sequence),
        ),
        "best_mixed_rank_bottleneck_slack": _get_best_of_mixed_dispatches_with_rank(
            instance,
            get_job_tiebreak_rank_from_job_sequence(bottleneck_sequence),
        ),
        "best_mixed_rank_agg_es_slack": _get_best_of_mixed_dispatches_with_rank(
            instance,
            aggregate_es_slack_rank,
        ),
        "best_mixed_rank_agg_ls_slack": _get_best_of_mixed_dispatches_with_rank(
            instance,
            aggregate_ls_slack_rank,
        ),
    }

    slack_local_repair_base = candidate_schedules["priority_stage_slack_ls"]
    if slack_local_repair_base is not None:
        candidate_schedules["priority_stage_slack_ls_local"] = _repair_schedule(
            instance,
            slack_local_repair_base,
            target_stage_ids=None,
            insertion_passes=3,
            max_shift=4,
            swap_passes=3,
        )
    else:
        candidate_schedules["priority_stage_slack_ls_local"] = None

    if candidate_schedules["mixed_aggregate_ls_slack"] is not None:
        candidate_schedules["mixed_aggregate_ls_slack_local_repair"] = _repair_schedule(
            instance,
            candidate_schedules["mixed_aggregate_ls_slack"],
            target_stage_ids=[bottleneck_anchor_stage_id, stage_ids[-1]],
            insertion_passes=3,
            max_shift=4,
            swap_passes=3,
        )
    else:
        candidate_schedules["mixed_aggregate_ls_slack_local_repair"] = None

    if candidate_schedules["best_mixed_rank_agg_ls_slack"] is not None:
        candidate_schedules["best_mixed_rank_agg_ls_slack_local_repair"] = (
            _repair_schedule(
                instance,
                candidate_schedules["best_mixed_rank_agg_ls_slack"],
                target_stage_ids=[bottleneck_anchor_stage_id, stage_ids[-1]],
                insertion_passes=3,
                max_shift=4,
                swap_passes=3,
            )
        )
    else:
        candidate_schedules["best_mixed_rank_agg_ls_slack_local_repair"] = None

    for schedule in candidate_schedules.values():
        _validate_schedule(schedule, stage_2_job_2_p)

    candidate_makespans = {
        name: _safe_makespan(schedule) for name, schedule in candidate_schedules.items()
    }
    feasible_new_candidates = {
        name: makespan
        for name, makespan in candidate_makespans.items()
        if makespan is not None
    }
    if feasible_new_candidates:
        base_best_name = min(feasible_new_candidates, key=feasible_new_candidates.get)
        base_best_schedule = candidate_schedules[base_best_name]
        if base_best_schedule is not None:
            repaired_best_schedule = _repair_schedule(
                instance,
                base_best_schedule,
                target_stage_ids=[bottleneck_anchor_stage_id, stage_ids[-1]],
                insertion_passes=3,
                max_shift=4,
                swap_passes=3,
            )
            candidate_schedules["selected_post_mip_local_repair"] = (
                repaired_best_schedule
            )
            candidate_makespans["selected_post_mip_local_repair"] = _safe_makespan(
                repaired_best_schedule
            )
            if candidate_makespans["selected_post_mip_local_repair"] is not None:
                feasible_new_candidates["selected_post_mip_local_repair"] = (
                    candidate_makespans["selected_post_mip_local_repair"]
                )

    best_new_name = min(feasible_new_candidates, key=feasible_new_candidates.get)
    best_new_makespan = feasible_new_candidates[best_new_name]

    summary_csv = instance_dir / "results" / f"{ins_name}_summary.csv"
    incumbent_ub = None
    if summary_csv.is_file():
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
            if row is not None and row.get("bestObj"):
                incumbent_ub = int(float(row["bestObj"]))

    result_row: dict[str, Any] = {
        "ins_name": ins_name,
        "incumbent_ub": incumbent_ub,
        "dispatch_cmax": payload.get("metadata", {}).get("dispatch_cmax"),
        "saved_selected_variant": saved_dispatch_summary.get("selected_variant"),
        "saved_selected_makespan": saved_dispatch_summary.get("selected_makespan"),
        "saved_best_of_mixed_dispatches": saved_candidates.get(
            "best_of_mixed_dispatches"
        ),
        "saved_es_ls_priority_score": saved_candidates.get("es_ls_priority_score"),
        "saved_strict_call_order": saved_candidates.get("strict_call_order"),
        "best_new_rule": best_new_name,
        "best_new_makespan": best_new_makespan,
        "bottleneck_stage_id": bottleneck_stage_id,
        "bottleneck_anchor_stage_id": bottleneck_anchor_stage_id,
    }
    result_row.update(candidate_makespans)
    return result_row


def main() -> None:
    args = _parse_args()
    scenario_dir = (
        args.scenario_dir.resolve()
        if args.scenario_dir is not None
        else _resolve_latest_scenario_dir(DEFAULT_OUTPUTS_DIR)
    )
    input_dir = args.input_dir.resolve()

    instance_dirs = sorted(
        [
            path
            for path in scenario_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key=lambda path: int(path.name),
    )
    if args.instances:
        selected = {str(ins) for ins in args.instances}
        instance_dirs = [path for path in instance_dirs if path.name in selected]

    rows: list[dict[str, Any]] = []
    for instance_dir in instance_dirs:
        row = _evaluate_instance(scenario_dir, input_dir, instance_dir.name)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No saved MIP payloads were found under {scenario_dir}."
        )

    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else scenario_dir / "dispatch_rule_benchmark.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote dispatch benchmark results to {output_csv}")


if __name__ == "__main__":
    main()
