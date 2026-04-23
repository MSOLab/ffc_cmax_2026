from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from typing import Sequence

from paper_fan2023_hea.encoding_decoding import (
    CompleteEncoding,
    DecodedSolution,
    DecodingFlag,
    HfsLikeInstance,
    JobId,
    decode_permutation,
)

POP_SIZE = 50
EVO_REP = 15
TOURNAMENT_SIZE = 2
NO_IMPROVE = 5
ELITE_RATIO = 0.04


@dataclass(frozen=True)
class Individual:
    permutation: tuple[JobId, ...]
    decoding: DecodingFlag
    obj_value: int | None = None
    encoding: CompleteEncoding | None = None
    stagnation: int = 0

    def evaluated(self) -> bool:
        return self.obj_value is not None and self.encoding is not None


def clone_individual(individual: Individual) -> Individual:
    return replace(individual)


def evaluate_individual(
    instance: HfsLikeInstance,
    individual: Individual,
) -> Individual:
    decoded = decode_permutation(instance, individual.permutation, individual.decoding)
    return replace(
        individual,
        obj_value=decoded.obj_value,
        encoding=decoded.encoding,
    )


def decode_individual(
    instance: HfsLikeInstance,
    individual: Individual,
) -> DecodedSolution:
    return decode_permutation(instance, individual.permutation, individual.decoding)


def random_individual(
    instance: HfsLikeInstance,
    rng: random.Random,
    decoding: DecodingFlag | None = None,
) -> Individual:
    permutation = list(instance.job_id_list)
    rng.shuffle(permutation)
    flag: DecodingFlag = decoding or rng.choice(["forward", "backward"])
    return evaluate_individual(
        instance,
        Individual(permutation=tuple(permutation), decoding=flag),
    )


def initialize_population(
    instance: HfsLikeInstance,
    rng: random.Random,
    pop_size: int = POP_SIZE,
) -> list[Individual]:
    population: list[Individual] = []
    forward_count = pop_size // 2
    backward_count = pop_size - forward_count
    for _ in range(forward_count):
        population.append(random_individual(instance, rng, "forward"))
    for _ in range(backward_count):
        population.append(random_individual(instance, rng, "backward"))
    return population


def evolve_population(
    instance: HfsLikeInstance,
    population: Sequence[Individual],
    rng: random.Random,
    evo_rep: int = EVO_REP,
    deadline: float | None = None,
) -> list[Individual]:
    evolved: list[Individual] = []
    for individual in population:
        if deadline is not None and time.perf_counter() >= deadline:
            evolved.append(individual)
            continue
        evolved.append(self_evolve(instance, individual, rng, evo_rep, deadline=deadline))
    return evolved


def self_evolve(
    instance: HfsLikeInstance,
    individual: Individual,
    rng: random.Random,
    evo_rep: int = EVO_REP,
    deadline: float | None = None,
) -> Individual:
    current = (
        individual
        if individual.evaluated()
        else evaluate_individual(instance, individual)
    )
    improved = False

    for _ in range(evo_rep):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        candidates = [
            _candidate(instance, current, insertion(current.permutation, rng)),
            _candidate(instance, current, adjacent_swap(current.permutation, rng)),
            _candidate(instance, current, pairwise_exchange(current.permutation, rng)),
        ]
        best_candidate = min(candidates, key=lambda item: item.obj_value or float("inf"))
        if (best_candidate.obj_value or float("inf")) < (
            current.obj_value or float("inf")
        ):
            current = replace(best_candidate, stagnation=0)
            improved = True

    if improved:
        return replace(current, stagnation=0)
    return replace(current, stagnation=current.stagnation + 1)


def insertion(permutation: Sequence[JobId], rng: random.Random) -> tuple[JobId, ...]:
    seq = list(permutation)
    if len(seq) < 2:
        return tuple(seq)
    src_idx, dst_idx = rng.sample(range(len(seq)), 2)
    item = seq.pop(src_idx)
    if dst_idx > src_idx:
        dst_idx -= 1
    seq.insert(dst_idx, item)
    return tuple(seq)


def adjacent_swap(
    permutation: Sequence[JobId],
    rng: random.Random,
) -> tuple[JobId, ...]:
    seq = list(permutation)
    if len(seq) < 2:
        return tuple(seq)
    idx = rng.randrange(len(seq) - 1)
    seq[idx], seq[idx + 1] = seq[idx + 1], seq[idx]
    return tuple(seq)


def pairwise_exchange(
    permutation: Sequence[JobId],
    rng: random.Random,
) -> tuple[JobId, ...]:
    seq = list(permutation)
    if len(seq) < 2:
        return tuple(seq)
    if len(seq) == 2:
        idx_a, idx_b = 0, 1
    else:
        idx_a, idx_b = rng.sample(range(len(seq)), 2)
        attempts = 0
        while abs(idx_a - idx_b) == 1 and attempts < 20:
            idx_a, idx_b = rng.sample(range(len(seq)), 2)
            attempts += 1
    seq[idx_a], seq[idx_b] = seq[idx_b], seq[idx_a]
    return tuple(seq)


def _candidate(
    instance: HfsLikeInstance,
    source: Individual,
    permutation: tuple[JobId, ...],
) -> Individual:
    return evaluate_individual(
        instance,
        Individual(permutation=permutation, decoding=source.decoding),
    )
