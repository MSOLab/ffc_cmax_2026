from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.dispatcher import JobDispatcher
from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule, validate_schedule

Operation = tuple[str, str, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a base schedule from a PRA instance and export Gantt charts for "
            "right-justified and semi-active retiming variants."
        )
    )
    parser.add_argument(
        "--instance",
        type=Path,
        default=Path("resources/pra/0.txt"),
        help="Path to a PRA-format instance file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Outputs_debug/retiming_pra_0"),
        help="Directory where the PNG charts will be written.",
    )
    return parser.parse_args()


def _load_instance(instance_path: Path) -> HybridFlowshopParameters:
    with instance_path.open("r", encoding="utf-8") as stream:
        return HybridFlowshopParameters.from_pra_data(instance_path.stem, stream)


def _build_base_schedule(
    instance: HybridFlowshopParameters,
) -> HybridFlowshopLiteSchedule:
    schedule = JobDispatcher(instance).get_schedule_by_dj_gupta()
    if schedule is None:
        raise ValueError("JobDispatcher returned None for the base schedule.")
    return schedule


def _sorted_operations(operations: Iterable[Operation]) -> list[Operation]:
    return sorted(operations, key=lambda op: (op[1], op[0], op[2]))


def _format_transition_logs(
    selected_rj_jobs: list[str],
    selected_last_stage_jobs: list[tuple[str, str, int]],
    total_shift: int,
    partial_rj_operation_set: set[Operation],
    partial_sa_jobs: list[str],
    selected_first_stage_jobs: list[tuple[str, str, int]],
    partial_sa_operation_set: set[Operation],
) -> list[str]:
    return [
        "base: JobDispatcher.get_schedule_by_dj_gupta()로 생성한 초기 schedule",
        (
            "right_justified_full: base에서 전체 operation에 "
            "make_right_justified(duration) 적용"
        ),
        (
            "right_justified_partial: base에서 마지막 stage 완료시각이 가장 늦은 "
            f"job {selected_rj_jobs}의 모든 operation "
            f"({len(partial_rj_operation_set)}개)에만 make_right_justified 적용 "
            f"(last-stage summary={selected_last_stage_jobs}, total_shift={total_shift})"
        ),
        (
            "semi_active_after_rj_partial: right_justified_full에서 첫 stage 시작이 가장 빠른 "
            f"job {partial_sa_jobs}의 모든 operation "
            f"({len(partial_sa_operation_set)}개)에만 make_semi_active 적용 "
            f"(first-stage summary={selected_first_stage_jobs})"
        ),
        (
            "semi_active_after_rj_full: semi_active_after_rj_partial에서 남은 operation까지 포함해 "
            "전체 make_semi_active(duration) 적용"
        ),
    ]


def _choose_partial_rj_operation(
    base: HybridFlowshopLiteSchedule,
    right_justified_full: HybridFlowshopLiteSchedule,
) -> tuple[set[Operation], list[str], list[tuple[str, str, int]], int]:
    last_stage_id = base.stages[-1]
    machine_count = len(base.machines_per_stage[last_stage_id])
    last_stage_ops = sorted(
        (
            (end_time, job_id, mc_id)
            for mc_id, _start_time, end_time, job_id in base.iter_operations_on_stage(
                last_stage_id
            )
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    selected_job_count = min(machine_count, len(last_stage_ops))
    selected_last_stage_ops = last_stage_ops[:selected_job_count]
    selected_jobs = [job_id for _end_time, job_id, _mc_id in selected_last_stage_ops]
    selected_job_set = set(selected_jobs)

    selected_operations = {
        (job_id, stage_id, mc_id)
        for stage_id, mc_id, _start_time, _end_time, job_id in base._iter_operations()
        if job_id in selected_job_set
    }

    total_shift = 0
    base_start_map = base.get_jik_2_start_time_map()
    rj_start_map = right_justified_full.get_jik_2_start_time_map()
    for operation in selected_operations:
        total_shift += rj_start_map[operation] - base_start_map[operation]

    selected_job_summaries = [
        (job_id, mc_id, int(end_time))
        for end_time, job_id, mc_id in selected_last_stage_ops
    ]
    return selected_operations, selected_jobs, selected_job_summaries, total_shift


def _choose_partial_sa_operation_set(
    schedule: HybridFlowshopLiteSchedule,
) -> tuple[set[Operation], list[str], list[tuple[str, str, int]]]:
    first_stage_id = schedule.stages[0]
    first_stage_ops = sorted(
        (
            (start_time, job_id, mc_id)
            for mc_id, start_time, _end_time, job_id in schedule.iter_operations_on_stage(
                first_stage_id
            )
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    machine_count = len(schedule.machines_per_stage[first_stage_id])
    selected_job_count = min(machine_count + 1, len(first_stage_ops))
    selected_first_stage_ops = first_stage_ops[:selected_job_count]
    selected_jobs = [job_id for _start_time, job_id, _mc_id in selected_first_stage_ops]
    selected_job_set = set(selected_jobs)

    selected_operations = {
        (job_id, stage_id, mc_id)
        for stage_id, mc_id, _start_time, _end_time, job_id in schedule._iter_operations()
        if job_id in selected_job_set
    }
    selected_job_summaries = [
        (job_id, mc_id, int(start_time))
        for start_time, job_id, mc_id in selected_first_stage_ops
    ]
    return selected_operations, selected_jobs, selected_job_summaries


def _validate_states(
    states: dict[str, HybridFlowshopLiteSchedule],
    duration: dict[str, dict[str, int]],
) -> None:
    for name, schedule in states.items():
        try:
            validate_schedule(schedule, duration)
        except ValueError as exc:
            raise ValueError(f"{name} schedule is invalid: {exc}") from exc


def _compute_force_end(states: dict[str, HybridFlowshopLiteSchedule]) -> int:
    return max(
        max(schedule.get_jik_2_end_time_map().values()) for schedule in states.values()
    )


def _export_gantt_charts(
    instance: HybridFlowshopParameters,
    states: dict[str, HybridFlowshopLiteSchedule],
    out_dir: Path,
) -> None:
    plotter = GanttPlotter()
    force_end = _compute_force_end(states)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (name, schedule) in enumerate(states.items()):
        plotter.export_hybrid_flowshop_plot(
            out_dir / f"{idx}_{name}.png",
            schedule.get_jik_2_start_time_map(),
            schedule.get_jik_2_end_time_map(),
            job_list=instance.job_id_list,
            stage_list=instance.stage_id_list,
            machine_list_per_stage=instance.stage_2_machines_map,
            force_start=0,
            force_end=force_end,
        )


def run_inspection(instance_path: Path, out_dir: Path) -> dict[str, object]:
    instance = _load_instance(instance_path)
    duration = instance.stage_2_job_2_p_map

    base = _build_base_schedule(instance)

    right_justified_full = base.deepcopy()
    right_justified_full.make_right_justified(duration)

    (
        partial_rj_operation_set,
        selected_jobs,
        selected_last_stage_jobs,
        total_shift,
    ) = _choose_partial_rj_operation(base, right_justified_full)

    right_justified_partial = base.deepcopy()
    right_justified_partial.make_right_justified(
        duration, operation_set=partial_rj_operation_set
    )

    semi_active_after_rj_partial = right_justified_full.deepcopy()
    (
        partial_sa_operation_set,
        partial_sa_jobs,
        selected_first_stage_jobs,
    ) = _choose_partial_sa_operation_set(semi_active_after_rj_partial)
    semi_active_after_rj_partial.make_semi_active(
        duration,
        operation_set=partial_sa_operation_set,
    )

    semi_active_after_rj_full = semi_active_after_rj_partial.deepcopy()
    semi_active_after_rj_full.make_semi_active(duration)

    states = {
        "base": base,
        "right_justified_partial": right_justified_partial,
        "right_justified_full": right_justified_full,
        "semi_active_after_rj_partial": semi_active_after_rj_partial,
        "semi_active_after_rj_full": semi_active_after_rj_full,
    }
    _validate_states(states, duration)
    _export_gantt_charts(instance, states, out_dir)

    makespans = {name: schedule.makespan for name, schedule in states.items()}
    transition_logs = _format_transition_logs(
        selected_rj_jobs=selected_jobs,
        selected_last_stage_jobs=selected_last_stage_jobs,
        total_shift=total_shift,
        partial_rj_operation_set=partial_rj_operation_set,
        partial_sa_jobs=partial_sa_jobs,
        selected_first_stage_jobs=selected_first_stage_jobs,
        partial_sa_operation_set=partial_sa_operation_set,
    )

    return {
        "instance_name": instance.name,
        "transition_logs": transition_logs,
        "selected_jobs": selected_jobs,
        "selected_last_stage_jobs": selected_last_stage_jobs,
        "selected_total_shift": total_shift,
        "partial_rj_operation_set": partial_rj_operation_set,
        "partial_sa_jobs": partial_sa_jobs,
        "selected_first_stage_jobs": selected_first_stage_jobs,
        "partial_sa_operation_set": partial_sa_operation_set,
        "makespans": makespans,
        "out_dir": out_dir,
    }


def main() -> None:
    args = _parse_args()
    result = run_inspection(args.instance, args.out_dir)

    selected_jobs = result["selected_jobs"]
    selected_last_stage_jobs = result["selected_last_stage_jobs"]
    partial_rj_operation_set = _sorted_operations(result["partial_rj_operation_set"])
    partial_sa_jobs = result["partial_sa_jobs"]
    selected_first_stage_jobs = result["selected_first_stage_jobs"]
    partial_sa_operation_set = _sorted_operations(result["partial_sa_operation_set"])
    out_dir = result["out_dir"]
    makespans = result["makespans"]
    transition_logs = result["transition_logs"]

    print(f"instance: {result['instance_name']}")
    print(f"out_dir: {out_dir}")
    print("transition_logs:")
    for idx, line in enumerate(transition_logs, start=1):
        print(f"  {idx}. {line}")
    print(f"selected_partial_jobs: {selected_jobs}")
    print(f"selected_last_stage_jobs: {selected_last_stage_jobs}")
    print(f"selected_total_shift: {result['selected_total_shift']}")
    print(f"partial_right_justified_operation_set: {partial_rj_operation_set}")
    print(f"selected_partial_sa_jobs: {partial_sa_jobs}")
    print(f"selected_first_stage_jobs: {selected_first_stage_jobs}")
    print(f"partial_semi_active_operation_set: {partial_sa_operation_set}")
    for name, makespan in makespans.items():
        print(f"makespan[{name}]={makespan}")


if __name__ == "__main__":
    main()
