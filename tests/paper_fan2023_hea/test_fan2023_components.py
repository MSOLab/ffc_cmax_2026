from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from paper_fan2023_hea.config import PaperRunConfig
from paper_fan2023_hea.encoding_decoding import backward_decode, forward_decode
from paper_fan2023_hea.population import (
    Individual,
    adjacent_swap,
    insertion,
    pairwise_exchange,
)
from paper_fan2023_hea.runner import paper_time_policy, run_from_config
from paper_fan2023_hea.selection import elitist_tournament_selection
from paper_fan2023_hea.tabu_graph import (
    apply_move,
    generate_tabu_moves,
    is_complete_encoding,
)


@dataclass(frozen=True)
class TinyInstance:
    name: str = "tiny"
    job_id_list: tuple[str, ...] = ("j1", "j2", "j3")
    stage_id_list: tuple[str, ...] = ("s1", "s2")
    stage_2_machines_map: dict[str, tuple[str, ...]] | None = None
    stage_2_job_2_p_map: dict[str, dict[str, int]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_2_machines_map",
            {
                "s1": ("m10", "m11"),
                "s2": ("m20", "m21"),
            },
        )
        object.__setattr__(
            self,
            "stage_2_job_2_p_map",
            {
                "s1": {"j1": 5, "j2": 1, "j3": 4},
                "s2": {"j1": 1, "j2": 5, "j3": 1},
            },
        )


def test_forward_backward_decoding_create_feasible_distinct_schedules() -> None:
    instance = TinyInstance()
    permutation = ("j1", "j2", "j3")

    forward = forward_decode(instance, permutation)
    backward = backward_decode(instance, permutation)

    assert forward.obj_value == 7
    assert backward.obj_value == 7
    assert forward.encoding != backward.encoding
    assert is_complete_encoding(instance, forward.encoding)
    assert is_complete_encoding(instance, backward.encoding)


def test_self_evolution_operators_preserve_permutation() -> None:
    rng = random.Random(7)
    permutation = ("j1", "j2", "j3", "j4")

    for operator in (insertion, adjacent_swap, pairwise_exchange):
        mutated = operator(permutation, rng)
        assert sorted(mutated) == sorted(permutation)
        assert len(mutated) == len(permutation)


def test_elitist_tournament_keeps_elite_and_reinitializes_stale_non_elites() -> None:
    rng = random.Random(3)
    population = [
        Individual(("a", "b"), "forward", obj_value=1, stagnation=5),
        Individual(("b", "a"), "forward", obj_value=2, stagnation=5),
        Individual(("a", "b"), "backward", obj_value=3, stagnation=5),
        Individual(("b", "a"), "backward", obj_value=4, stagnation=5),
    ]
    replacement = Individual(("x", "y"), "forward", obj_value=999, stagnation=0)

    selected = elitist_tournament_selection(
        population=population,
        rng=rng,
        replacement_factory=lambda: replacement,
        elite_ratio=0.25,
        tournament_size=len(population),
        no_improve=5,
    )

    assert selected[0].obj_value == 1
    assert [individual.obj_value for individual in selected[1:]] == [999, 999, 999]


def test_backward_moves_keep_disjunctive_graph_cycle_free() -> None:
    instance = TinyInstance()
    solution = backward_decode(instance, ("j1", "j2", "j3"))

    moves = generate_tabu_moves(instance, solution.encoding, "backward")

    assert moves
    assert all(
        is_complete_encoding(instance, apply_move(solution.encoding, move))
        for move in moves
    )


def test_paper_time_policy_uses_n_z_activation_and_2_n_z_total() -> None:
    policy = paper_time_policy(job_count=40, stage_count=5, scale=1.0)

    assert policy.ts_activation_seconds == 200
    assert policy.total_seconds == 400


def test_ff2020_smoke_run_writes_exp_compare_compatible_summary(tmp_path) -> None:
    config = PaperRunConfig(
        input_dir="resources/ff2020big",
        baseline_csv_path="resources/ff2020big_ref/fan2023.csv",
        benchmark_idx_list=[1],
        seed_count=1,
        output_dir=tmp_path / "Outputs_paper_fan2023",
        draw_convergence=False,
        random_seed_base=11,
        scenario="fan2023_hea_test",
        time_limit_scale=0.000001,
    )

    output_dir = run_from_config(config, quiet=True)
    seed_df = pd.read_csv(output_dir / "seed_runs.csv")
    summary_df = pd.read_csv(output_dir / "all_scenarios_summary.csv")

    assert len(seed_df) == 1
    assert len(summary_df) == 1
    assert {"name", "insName", "scenario", "bestObj"}.issubset(summary_df.columns)
    assert summary_df.loc[0, "scenario"] == "fan2023_hea_test"
