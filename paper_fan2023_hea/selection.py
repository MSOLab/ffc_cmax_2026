from __future__ import annotations

import random
from typing import Callable, Sequence

from paper_fan2023_hea.population import (
    ELITE_RATIO,
    NO_IMPROVE,
    TOURNAMENT_SIZE,
    Individual,
    clone_individual,
)

ReplacementFactory = Callable[[], Individual]


def elitist_tournament_selection(
    population: Sequence[Individual],
    rng: random.Random,
    replacement_factory: ReplacementFactory,
    elite_ratio: float = ELITE_RATIO,
    tournament_size: int = TOURNAMENT_SIZE,
    no_improve: int = NO_IMPROVE,
) -> list[Individual]:
    if not population:
        return []
    pop_size = len(population)
    elite_count = max(1, int(pop_size * elite_ratio))
    sorted_population = sorted(
        population,
        key=lambda individual: individual.obj_value or float("inf"),
    )

    next_population: list[Individual] = [
        clone_individual(individual) for individual in sorted_population[:elite_count]
    ]

    while len(next_population) < pop_size:
        contenders = rng.sample(
            list(population),
            k=min(tournament_size, len(population)),
        )
        winner = min(
            contenders, key=lambda individual: individual.obj_value or float("inf")
        )
        next_population.append(clone_individual(winner))

    for idx in range(elite_count, len(next_population)):
        if next_population[idx].stagnation >= no_improve:
            next_population[idx] = replacement_factory()

    return next_population
