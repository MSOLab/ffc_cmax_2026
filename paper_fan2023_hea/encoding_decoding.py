from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule

JobId = str
StageId = str
MachineId = str
DecodingFlag = Literal["forward", "backward"]
OriginTag = Literal["F", "B", "T"]
MachineEncoding = tuple[JobId, ...]
StageEncoding = tuple[MachineEncoding, ...]
CompleteEncoding = tuple[StageEncoding, ...]


class HfsLikeInstance(Protocol):
    name: str
    job_id_list: Sequence[JobId]
    stage_id_list: Sequence[StageId]
    stage_2_machines_map: dict[StageId, Sequence[MachineId]]
    stage_2_job_2_p_map: dict[StageId, dict[JobId, int]]


@dataclass(frozen=True)
class DecodedSolution:
    schedule: HybridFlowshopLiteSchedule
    encoding: CompleteEncoding
    obj_value: int


def origin_tag_from_decoding(decoding: DecodingFlag) -> OriginTag:
    return "F" if decoding == "forward" else "B"


def decode_permutation(
    instance: HfsLikeInstance,
    permutation: Sequence[JobId],
    decoding: DecodingFlag,
) -> DecodedSolution:
    if decoding == "forward":
        return forward_decode(instance, permutation)
    if decoding == "backward":
        return backward_decode(instance, permutation)
    raise ValueError(f"Unsupported decoding flag: {decoding}")


def forward_decode(
    instance: HfsLikeInstance,
    permutation: Sequence[JobId],
) -> DecodedSolution:
    stages = list(instance.stage_id_list)
    return _decode_forward_for_stage_order(
        jobs=list(instance.job_id_list),
        stages=stages,
        machines_per_stage=instance.stage_2_machines_map,
        stage_2_job_2_p=instance.stage_2_job_2_p_map,
        permutation=permutation,
    )


def backward_decode(
    instance: HfsLikeInstance,
    permutation: Sequence[JobId],
) -> DecodedSolution:
    """Decode by the paper's ALAP/latest-available rule via reversed-time ASAP.

    The repository schedule object is forward-time.  Running the Fan forward rule on
    reversed stage order and then reversing time gives the same ALAP semantics while
    keeping all feasibility checks in one implementation.
    """

    reversed_solution = _decode_forward_for_stage_order(
        jobs=list(instance.job_id_list),
        stages=list(reversed(instance.stage_id_list)),
        machines_per_stage=instance.stage_2_machines_map,
        stage_2_job_2_p=instance.stage_2_job_2_p_map,
        permutation=permutation,
    )
    schedule = reversed_solution.schedule.as_reversed()
    encoding = extract_complete_encoding(schedule)
    return DecodedSolution(
        schedule=schedule,
        encoding=encoding,
        obj_value=int(schedule.makespan),
    )


def _decode_forward_for_stage_order(
    jobs: list[JobId],
    stages: list[StageId],
    machines_per_stage: dict[StageId, Sequence[MachineId]],
    stage_2_job_2_p: dict[StageId, dict[JobId, int]],
    permutation: Sequence[JobId],
) -> DecodedSolution:
    _validate_permutation(jobs, permutation)
    schedule = HybridFlowshopLiteSchedule(
        jobs=jobs,
        stages=stages,
        machines_per_stage=machines_per_stage,
    )
    rank = {job: idx for idx, job in enumerate(permutation)}

    previous_order = list(permutation)
    for stage_idx, stage_id in enumerate(stages):
        if stage_idx == 0:
            stage_order = list(permutation)
        else:
            prev_stage = stages[stage_idx - 1]
            stage_order = sorted(
                previous_order,
                key=lambda job: (
                    schedule.get_job_end_time(prev_stage, job),
                    rank[job],
                ),
            )

        for job_id in stage_order:
            duration = int(stage_2_job_2_p[stage_id][job_id])
            release_t = (
                0
                if stage_idx == 0
                else schedule.get_job_end_time(stages[stage_idx - 1], job_id)
            )
            mc_id = _select_first_available_machine(
                schedule=schedule,
                stage_id=stage_id,
                machines=list(machines_per_stage[stage_id]),
                duration=duration,
                release_t=release_t,
            )
            schedule.add_operation_2_mc(
                stage_id=stage_id,
                mc_id=mc_id,
                job_id=job_id,
                duration=duration,
                release_t=release_t,
            )
        previous_order = stage_order

    encoding = extract_complete_encoding(schedule)
    return DecodedSolution(
        schedule=schedule,
        encoding=encoding,
        obj_value=int(schedule.makespan),
    )


def _select_first_available_machine(
    schedule: HybridFlowshopLiteSchedule,
    stage_id: StageId,
    machines: list[MachineId],
    duration: int,
    release_t: int,
) -> MachineId:
    candidates: list[tuple[int, int, MachineId]] = []
    for idx, mc_id in enumerate(machines):
        earliest_start = schedule.get_machine_earliest_start_time(
            stage_id=stage_id,
            mc_id=mc_id,
            duration=duration,
            release_t=release_t,
        )
        candidates.append((earliest_start, idx, mc_id))
    return min(candidates)[2]


def extract_complete_encoding(
    schedule: HybridFlowshopLiteSchedule,
) -> CompleteEncoding:
    stages: list[StageEncoding] = []
    for stage_id in schedule.stages:
        machines: list[MachineEncoding] = []
        for mc_id in schedule.machines_per_stage[stage_id]:
            machines.append(
                tuple(
                    job_id
                    for _start, _end, job_id in schedule.get_job_sequence(
                        stage_id, mc_id
                    )
                )
            )
        stages.append(tuple(machines))
    return tuple(stages)


def decode_encoding(
    instance: HfsLikeInstance,
    encoding: CompleteEncoding,
) -> DecodedSolution:
    """Evaluate a disjunctive-graph encoding by exact semi-active recomputation."""

    validate_complete_encoding(instance, encoding)
    schedule = HybridFlowshopLiteSchedule(
        jobs=list(instance.job_id_list),
        stages=list(instance.stage_id_list),
        machines_per_stage=instance.stage_2_machines_map,
    )

    for stage_idx, stage_id in enumerate(instance.stage_id_list):
        for machine_idx, mc_id in enumerate(instance.stage_2_machines_map[stage_id]):
            for job_id in encoding[stage_idx][machine_idx]:
                release_t = (
                    0
                    if stage_idx == 0
                    else schedule.get_job_end_time(
                        instance.stage_id_list[stage_idx - 1],
                        job_id,
                    )
                )
                schedule.add_operation_2_mc(
                    stage_id=stage_id,
                    mc_id=mc_id,
                    job_id=job_id,
                    duration=int(instance.stage_2_job_2_p_map[stage_id][job_id]),
                    release_t=release_t,
                )
    return DecodedSolution(
        schedule=schedule,
        encoding=extract_complete_encoding(schedule),
        obj_value=int(schedule.makespan),
    )


def validate_complete_encoding(
    instance: HfsLikeInstance,
    encoding: CompleteEncoding,
) -> None:
    expected_jobs = set(instance.job_id_list)
    stages = list(instance.stage_id_list)
    if len(encoding) != len(stages):
        raise ValueError("Encoding stage count does not match the instance.")
    for stage_idx, stage_id in enumerate(stages):
        expected_machine_count = len(instance.stage_2_machines_map[stage_id])
        if len(encoding[stage_idx]) != expected_machine_count:
            raise ValueError(
                f"Encoding machine count mismatch at stage {stage_id}: "
                f"{len(encoding[stage_idx])} != {expected_machine_count}"
            )
        seen = [job_id for machine_seq in encoding[stage_idx] for job_id in machine_seq]
        if set(seen) != expected_jobs or len(seen) != len(expected_jobs):
            raise ValueError(f"Encoding is not a job permutation at stage {stage_id}.")


def _validate_permutation(jobs: Sequence[JobId], permutation: Sequence[JobId]) -> None:
    if set(jobs) != set(permutation) or len(jobs) != len(permutation):
        raise ValueError("Permutation must contain every job exactly once.")
