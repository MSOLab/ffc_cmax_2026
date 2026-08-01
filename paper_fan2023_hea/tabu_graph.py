from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from paper_fan2023_hea.encoding_decoding import (
    CompleteEncoding,
    DecodingFlag,
    HfsLikeInstance,
    JobId,
    decode_encoding,
    validate_complete_encoding,
)
from paper_fan2023_hea.population import Individual

MoveKind = Literal["swap", "insert", "transfer"]


@dataclass(frozen=True)
class OperationRef:
    stage_idx: int
    machine_idx: int
    pos: int
    job_id: JobId


@dataclass(frozen=True)
class Move:
    kind: MoveKind
    stage_idx: int
    src_machine_idx: int
    src_pos: int
    dst_machine_idx: int
    dst_pos: int
    job_id: JobId
    other_job_id: JobId | None = None

    @property
    def tabu_attribute(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.stage_idx,
            self.job_id,
            self.src_machine_idx,
            self.src_pos,
            self.dst_machine_idx,
            self.dst_pos,
            self.other_job_id,
        )


@dataclass(frozen=True)
class CriticalContext:
    encoding: CompleteEncoding
    starts: dict[tuple[int, JobId], int]
    ends: dict[tuple[int, JobId], int]
    loc: dict[tuple[int, JobId], OperationRef]
    critical_blocks: list[list[OperationRef]]
    critical_ops: list[OperationRef]
    makespan: int


@dataclass(frozen=True)
class TabuSearchResult:
    encoding: CompleteEncoding
    obj_value: int
    iterations: int
    improved: bool


def run_tabu_search(
    instance: HfsLikeInstance,
    individual: Individual,
    rng: random.Random,
    global_best_obj: int,
    max_iter: int,
    tabu_len: int,
    deadline: float | None = None,
) -> TabuSearchResult:
    if individual.encoding is None or individual.obj_value is None:
        raise ValueError("Tabu search requires an evaluated individual.")

    current_encoding = individual.encoding
    current_obj = individual.obj_value
    best_encoding = current_encoding
    best_obj = current_obj
    tabu_expiry: dict[tuple[object, ...], int] = {}
    iterations = 0

    for iter_idx in range(max_iter):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        iterations = iter_idx + 1
        moves = generate_tabu_moves(instance, current_encoding, individual.decoding)
        rng.shuffle(moves)
        selected: tuple[int, CompleteEncoding, Move] | None = None

        for move in moves:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            candidate_encoding = apply_move(current_encoding, move)
            if not is_complete_encoding(instance, candidate_encoding):
                continue
            candidate = decode_encoding(instance, candidate_encoding)
            candidate_obj = candidate.obj_value
            is_tabu = tabu_expiry.get(move.tabu_attribute, -1) > iter_idx
            aspiration = candidate_obj < global_best_obj
            if is_tabu and not aspiration:
                continue
            if selected is None or candidate_obj < selected[0]:
                selected = (candidate_obj, candidate.encoding, move)

        if selected is None:
            break

        current_obj, current_encoding, selected_move = selected
        tabu_expiry[selected_move.tabu_attribute] = iter_idx + tabu_len
        if current_obj < best_obj:
            best_obj = current_obj
            best_encoding = current_encoding
            global_best_obj = min(global_best_obj, best_obj)

    return TabuSearchResult(
        encoding=best_encoding,
        obj_value=best_obj,
        iterations=iterations,
        improved=best_obj < individual.obj_value,
    )


def generate_tabu_moves(
    instance: HfsLikeInstance,
    encoding: CompleteEncoding,
    decoding: DecodingFlag,
) -> list[Move]:
    ctx = build_critical_context(instance, encoding)
    moves: list[Move] = []
    if decoding == "forward":
        moves.extend(_forward_n7_moves(ctx))
    else:
        moves.extend(_backward_n7_moves(instance, ctx))
    moves.extend(_k_insertion_moves(instance, ctx, decoding))
    return _dedupe_valid_moves(instance, encoding, moves)


def build_critical_context(
    instance: HfsLikeInstance,
    encoding: CompleteEncoding,
) -> CriticalContext:
    decoded = decode_encoding(instance, encoding)
    schedule = decoded.schedule
    stage_to_idx = {
        stage_id: idx for idx, stage_id in enumerate(instance.stage_id_list)
    }
    starts: dict[tuple[int, JobId], int] = {}
    ends: dict[tuple[int, JobId], int] = {}
    loc: dict[tuple[int, JobId], OperationRef] = {}

    for stage_idx, stage_id in enumerate(instance.stage_id_list):
        for machine_idx, mc_id in enumerate(instance.stage_2_machines_map[stage_id]):
            for pos, (start, end, job_id) in enumerate(
                schedule.get_job_sequence(stage_id, mc_id)
            ):
                starts[(stage_idx, job_id)] = int(start)
                ends[(stage_idx, job_id)] = int(end)
                loc[(stage_idx, job_id)] = OperationRef(
                    stage_idx=stage_idx,
                    machine_idx=machine_idx,
                    pos=pos,
                    job_id=job_id,
                )

    raw_blocks = schedule.find_critical_blocks(
        instance.stage_2_job_2_p_map,
        include_singletons=False,
    )
    critical_blocks: list[list[OperationRef]] = []
    for block in raw_blocks:
        refs = [
            loc[(stage_to_idx[stage_id], job_id)] for job_id, stage_id, _mc in block
        ]
        refs.sort(key=lambda ref: ref.pos)
        critical_blocks.append(refs)

    raw_singletons = schedule.find_critical_blocks(
        instance.stage_2_job_2_p_map,
        include_singletons=True,
    )
    critical_ops = _unique_ops(
        loc[(stage_to_idx[stage_id], job_id)]
        for block in raw_singletons
        for job_id, stage_id, _mc in block
    )
    if not critical_ops:
        last_stage_idx = len(instance.stage_id_list) - 1
        latest_end = max(
            ends[(last_stage_idx, job_id)] for job_id in instance.job_id_list
        )
        critical_ops = [
            loc[(last_stage_idx, job_id)]
            for job_id in instance.job_id_list
            if ends[(last_stage_idx, job_id)] == latest_end
        ]

    return CriticalContext(
        encoding=decoded.encoding,
        starts=starts,
        ends=ends,
        loc=loc,
        critical_blocks=critical_blocks,
        critical_ops=critical_ops,
        makespan=decoded.obj_value,
    )


def apply_move(encoding: CompleteEncoding, move: Move) -> CompleteEncoding:
    mutable = [
        [list(machine_seq) for machine_seq in stage_encoding]
        for stage_encoding in encoding
    ]
    source_seq = mutable[move.stage_idx][move.src_machine_idx]
    if move.kind == "swap":
        source_seq[move.src_pos], source_seq[move.dst_pos] = (
            source_seq[move.dst_pos],
            source_seq[move.src_pos],
        )
    elif move.kind == "insert":
        job_id = source_seq.pop(move.src_pos)
        dst_pos = move.dst_pos
        if move.dst_machine_idx == move.src_machine_idx and move.dst_pos > move.src_pos:
            dst_pos -= 1
        source_seq.insert(dst_pos, job_id)
    elif move.kind == "transfer":
        job_id = source_seq.pop(move.src_pos)
        target_seq = mutable[move.stage_idx][move.dst_machine_idx]
        dst_pos = move.dst_pos
        if move.dst_machine_idx == move.src_machine_idx and move.dst_pos > move.src_pos:
            dst_pos -= 1
        target_seq.insert(dst_pos, job_id)
    else:
        raise ValueError(f"Unsupported move kind: {move.kind}")
    return tuple(
        tuple(tuple(machine_seq) for machine_seq in stage) for stage in mutable
    )


def is_complete_encoding(
    instance: HfsLikeInstance,
    encoding: CompleteEncoding,
) -> bool:
    try:
        validate_complete_encoding(instance, encoding)
    except ValueError:
        return False
    return is_acyclic_encoding(instance, encoding)


def is_acyclic_encoding(
    instance: HfsLikeInstance,
    encoding: CompleteEncoding,
) -> bool:
    """Validate the disjunctive graph has no cycle.

    For a hybrid flow shop the job arcs only advance stage index, while machine
    arcs stay within a stage.  The graph is therefore normally acyclic, but this
    explicit check keeps the TS legality contract testable.
    """

    nodes = [
        (stage_idx, job_id)
        for stage_idx in range(len(instance.stage_id_list))
        for job_id in instance.job_id_list
    ]
    adjacency: dict[tuple[int, JobId], list[tuple[int, JobId]]] = {
        node: [] for node in nodes
    }
    indegree: dict[tuple[int, JobId], int] = {node: 0 for node in nodes}

    for stage_idx in range(len(instance.stage_id_list) - 1):
        for job_id in instance.job_id_list:
            src = (stage_idx, job_id)
            dst = (stage_idx + 1, job_id)
            adjacency[src].append(dst)
            indegree[dst] += 1

    for stage_idx, stage_encoding in enumerate(encoding):
        for machine_seq in stage_encoding:
            for src_job, dst_job in zip(machine_seq, machine_seq[1:]):
                src = (stage_idx, src_job)
                dst = (stage_idx, dst_job)
                adjacency[src].append(dst)
                indegree[dst] += 1

    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for dst in adjacency[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
    return visited == len(nodes)


def _forward_n7_moves(ctx: CriticalContext) -> list[Move]:
    moves: list[Move] = []
    for block in ctx.critical_blocks:
        if len(block) < 2:
            continue
        first, second = block[0], block[1]
        moves.append(
            Move(
                kind="swap",
                stage_idx=first.stage_idx,
                src_machine_idx=first.machine_idx,
                src_pos=first.pos,
                dst_machine_idx=second.machine_idx,
                dst_pos=second.pos,
                job_id=first.job_id,
                other_job_id=second.job_id,
            )
        )
        if len(block) > 2:
            before_last, last = block[-2], block[-1]
            moves.append(
                Move(
                    kind="swap",
                    stage_idx=before_last.stage_idx,
                    src_machine_idx=before_last.machine_idx,
                    src_pos=before_last.pos,
                    dst_machine_idx=last.machine_idx,
                    dst_pos=last.pos,
                    job_id=before_last.job_id,
                    other_job_id=last.job_id,
                )
            )
    return moves


def _backward_n7_moves(
    instance: HfsLikeInstance,
    ctx: CriticalContext,
) -> list[Move]:
    moves: list[Move] = []
    for block in ctx.critical_blocks:
        if len(block) < 2:
            continue
        first = block[0]
        last = block[-1]
        if _theorem_1_allows_first_after_last(instance, ctx, first, last):
            moves.append(
                Move(
                    kind="insert",
                    stage_idx=first.stage_idx,
                    src_machine_idx=first.machine_idx,
                    src_pos=first.pos,
                    dst_machine_idx=last.machine_idx,
                    dst_pos=last.pos + 1,
                    job_id=first.job_id,
                    other_job_id=last.job_id,
                )
            )
        if _theorem_2_allows_last_before_first(instance, ctx, first, last):
            moves.append(
                Move(
                    kind="insert",
                    stage_idx=last.stage_idx,
                    src_machine_idx=last.machine_idx,
                    src_pos=last.pos,
                    dst_machine_idx=first.machine_idx,
                    dst_pos=first.pos,
                    job_id=last.job_id,
                    other_job_id=first.job_id,
                )
            )
    return moves


def _k_insertion_moves(
    instance: HfsLikeInstance,
    ctx: CriticalContext,
    decoding: DecodingFlag,
) -> list[Move]:
    moves: list[Move] = []
    for op in ctx.critical_ops:
        stage_id = instance.stage_id_list[op.stage_idx]
        machine_count = len(instance.stage_2_machines_map[stage_id])
        if machine_count <= 1:
            continue
        for target_machine_idx in range(machine_count):
            if target_machine_idx == op.machine_idx:
                continue
            target_seq = ctx.encoding[op.stage_idx][target_machine_idx]
            target_positions = _k_insertion_positions(
                instance=instance,
                ctx=ctx,
                op=op,
                target_machine_idx=target_machine_idx,
                target_seq=target_seq,
                decoding=decoding,
            )
            for dst_pos in target_positions:
                moves.append(
                    Move(
                        kind="transfer",
                        stage_idx=op.stage_idx,
                        src_machine_idx=op.machine_idx,
                        src_pos=op.pos,
                        dst_machine_idx=target_machine_idx,
                        dst_pos=dst_pos,
                        job_id=op.job_id,
                    )
                )
    return moves


def _k_insertion_positions(
    instance: HfsLikeInstance,
    ctx: CriticalContext,
    op: OperationRef,
    target_machine_idx: int,
    target_seq: Sequence[JobId],
    decoding: DecodingFlag,
) -> list[int]:
    if not target_seq:
        return [0]

    succ = _job_successor(instance, op)
    pred = _job_predecessor(instance, op)
    q_succ_plus_p = _tail_plus_duration(instance, ctx, succ)
    r_pred_plus_p = _head_plus_duration(instance, ctx, pred)

    left_positions: set[int] = set()
    right_positions: set[int] = set()
    for pos, job_id in enumerate(target_seq):
        target = OperationRef(
            stage_idx=op.stage_idx,
            machine_idx=target_machine_idx,
            pos=pos,
            job_id=job_id,
        )
        q_plus_p = _tail_plus_duration(instance, ctx, target)
        r_plus_p = _head_plus_duration(instance, ctx, target)
        if decoding == "forward":
            if q_plus_p > q_succ_plus_p:
                left_positions.add(pos)
            if r_plus_p > r_pred_plus_p:
                right_positions.add(pos)
        else:
            if r_plus_p > r_pred_plus_p:
                left_positions.add(pos)
            if q_plus_p > q_succ_plus_p:
                right_positions.add(pos)

    intersection = left_positions.intersection(right_positions)
    insertion_positions: set[int] = set()
    if intersection:
        for pos in intersection:
            insertion_positions.add(pos)
            insertion_positions.add(pos + 1)
    else:
        for pos in left_positions:
            insertion_positions.add(pos + 1)
        for pos in right_positions:
            insertion_positions.add(pos)

    return sorted(
        pos for pos in insertion_positions if 0 <= pos <= len(target_seq)
    ) or [len(target_seq)]


def _theorem_1_allows_first_after_last(
    instance: HfsLikeInstance,
    ctx: CriticalContext,
    first: OperationRef,
    last: OperationRef,
) -> bool:
    successor = _job_successor(instance, first)
    return _tail_plus_duration(instance, ctx, last) >= _tail_plus_duration(
        instance,
        ctx,
        successor,
    )


def _theorem_2_allows_last_before_first(
    instance: HfsLikeInstance,
    ctx: CriticalContext,
    first: OperationRef,
    last: OperationRef,
) -> bool:
    predecessor = _job_predecessor(instance, last)
    return _head(instance, ctx, first) >= _head(instance, ctx, predecessor)


def _job_successor(
    instance: HfsLikeInstance,
    op: OperationRef | None,
) -> OperationRef | None:
    if op is None or op.stage_idx >= len(instance.stage_id_list) - 1:
        return None
    return OperationRef(
        stage_idx=op.stage_idx + 1,
        machine_idx=-1,
        pos=-1,
        job_id=op.job_id,
    )


def _job_predecessor(
    instance: HfsLikeInstance,
    op: OperationRef | None,
) -> OperationRef | None:
    if op is None or op.stage_idx <= 0:
        return None
    return OperationRef(
        stage_idx=op.stage_idx - 1,
        machine_idx=-1,
        pos=-1,
        job_id=op.job_id,
    )


def _duration(instance: HfsLikeInstance, op: OperationRef | None) -> int:
    if op is None:
        return 0
    stage_id = instance.stage_id_list[op.stage_idx]
    return int(instance.stage_2_job_2_p_map[stage_id][op.job_id])


def _head(
    instance: HfsLikeInstance, ctx: CriticalContext, op: OperationRef | None
) -> int:
    if op is None:
        return 0
    return ctx.starts[(op.stage_idx, op.job_id)]


def _tail(
    instance: HfsLikeInstance, ctx: CriticalContext, op: OperationRef | None
) -> int:
    if op is None:
        return 0
    return ctx.makespan - ctx.ends[(op.stage_idx, op.job_id)]


def _head_plus_duration(
    instance: HfsLikeInstance,
    ctx: CriticalContext,
    op: OperationRef | None,
) -> int:
    return _head(instance, ctx, op) + _duration(instance, op)


def _tail_plus_duration(
    instance: HfsLikeInstance,
    ctx: CriticalContext,
    op: OperationRef | None,
) -> int:
    return _tail(instance, ctx, op) + _duration(instance, op)


def _dedupe_valid_moves(
    instance: HfsLikeInstance,
    encoding: CompleteEncoding,
    moves: Iterable[Move],
) -> list[Move]:
    unique: dict[tuple[object, ...], Move] = {}
    for move in moves:
        key = move.tabu_attribute
        if key in unique:
            continue
        candidate = apply_move(encoding, move)
        if is_complete_encoding(instance, candidate):
            unique[key] = move
    return list(unique.values())


def _unique_ops(ops: Iterable[OperationRef]) -> list[OperationRef]:
    seen: set[tuple[int, JobId]] = set()
    result: list[OperationRef] = []
    for op in ops:
        key = (op.stage_idx, op.job_id)
        if key not in seen:
            result.append(op)
            seen.add(key)
    return result
