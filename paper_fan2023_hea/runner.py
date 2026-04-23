from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from paper_fan2023_hea.config import PaperRunConfig
from paper_fan2023_hea.encoding_decoding import (
    CompleteEncoding,
    HfsLikeInstance,
    origin_tag_from_decoding,
)
from paper_fan2023_hea.instance_io import BaselineRecord, load_baseline_records, load_instances
from paper_fan2023_hea.population import (
    ELITE_RATIO,
    EVO_REP,
    NO_IMPROVE,
    POP_SIZE,
    evolve_population,
    initialize_population,
    random_individual,
)
from paper_fan2023_hea.reporting import (
    SeedRunRecord,
    SummaryRecord,
    rpdf,
    rpdv,
    write_baseline_comparison,
    write_convergence_plot,
    write_seed_runs,
    write_summary,
)
from paper_fan2023_hea.selection import elitist_tournament_selection
from paper_fan2023_hea.tabu_graph import run_tabu_search


@dataclass(frozen=True)
class TimePolicy:
    ts_activation_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SingleSeedResult:
    seed: int
    best_obj: int
    best_origin_tag: str
    best_encoding: CompleteEncoding
    runtime_seconds: float
    history_rows: list[dict[str, object]]


def paper_time_policy(
    job_count: int,
    stage_count: int,
    scale: float = 1.0,
) -> TimePolicy:
    activation = job_count * stage_count * scale
    return TimePolicy(
        ts_activation_seconds=activation,
        total_seconds=2 * activation,
    )


def run_from_config(config: PaperRunConfig, quiet: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    log_path = output_dir / "fan2023_hea.log"
    _setup_file_logging(log_path, quiet=quiet)
    logging.info("Starting Fan 2023 HEA standalone run.")

    baseline_records = load_baseline_records(config.baseline_csv_path)
    instances = load_instances(config.input_dir, config.benchmark_filenames())

    seed_records: list[SeedRunRecord] = []
    summary_records: list[SummaryRecord] = []
    convergence_by_instance: dict[str, list[dict[str, object]]] = {}

    for benchmark_idx, instance in zip(config.benchmark_indices, instances):
        logging.info("Running instance %s.", benchmark_idx)
        baseline = baseline_records.get(benchmark_idx)
        seed_results: list[SingleSeedResult] = []
        for seed_offset in range(config.seed_count):
            seed = config.random_seed_base + seed_offset
            result = run_single_seed(instance, seed=seed, config=config)
            seed_results.append(result)
            seed_records.append(
                _seed_record(
                    instance_name=str(benchmark_idx),
                    scenario=config.scenario,
                    baseline=baseline,
                    result=result,
                )
            )
            convergence_by_instance.setdefault(str(benchmark_idx), []).extend(
                result.history_rows
            )

        best_result = min(seed_results, key=lambda item: item.best_obj)
        summary_records.append(
            _summary_record(
                instance_name=str(benchmark_idx),
                scenario=config.scenario,
                baseline=baseline,
                result=best_result,
                seed_count=config.seed_count,
            )
        )

    seed_df = write_seed_runs(output_dir / "seed_runs.csv", seed_records)
    summary_df = write_summary(
        output_dir / "all_scenarios_summary.csv",
        summary_records,
    )
    write_baseline_comparison(output_dir / "baseline_comparison.csv", summary_df)

    if config.draw_convergence:
        plot_ids = set(str(idx) for idx in (config.plot_instance_ids or config.benchmark_indices))
        for instance_name, rows in convergence_by_instance.items():
            if instance_name in plot_ids:
                write_convergence_plot(
                    output_dir / f"convergence_{instance_name}.png",
                    rows,
                )

    seed_df.to_csv(output_dir / "seed_runs.csv", index=False)
    summary_df.to_csv(output_dir / "all_scenarios_summary.csv", index=False)
    logging.info("Finished Fan 2023 HEA run. Output: %s", output_dir)
    return output_dir


def run_single_seed(
    instance: HfsLikeInstance,
    seed: int,
    config: PaperRunConfig,
) -> SingleSeedResult:
    rng = random.Random(seed)
    job_count = len(instance.job_id_list)
    stage_count = len(instance.stage_id_list)
    time_policy = paper_time_policy(
        job_count=job_count,
        stage_count=stage_count,
        scale=config.time_limit_scale,
    )
    max_iter_ts = 100 * job_count
    len_ts = job_count

    start = time.perf_counter()
    deadline = start + time_policy.total_seconds
    ts_activation_at = start + time_policy.ts_activation_seconds
    population = initialize_population(instance, rng, POP_SIZE)
    best_individual = min(population, key=lambda item: item.obj_value or float("inf"))
    assert best_individual.obj_value is not None
    assert best_individual.encoding is not None
    best_obj = best_individual.obj_value
    best_origin_tag = origin_tag_from_decoding(best_individual.decoding)
    best_encoding = best_individual.encoding
    history_rows = [
        _history_row(
            seed=seed,
            elapsed_seconds=0.0,
            generation=0,
            best_obj=best_obj,
            origin_tag=best_origin_tag,
        )
    ]

    generation = 0
    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        if config.max_iterations is not None and generation >= config.max_iterations:
            break

        generation += 1
        population = evolve_population(
            instance,
            population,
            rng,
            EVO_REP,
            deadline=deadline,
        )
        best_population_individual = min(
            population,
            key=lambda item: item.obj_value or float("inf"),
        )
        if (best_population_individual.obj_value or float("inf")) < best_obj:
            assert best_population_individual.obj_value is not None
            assert best_population_individual.encoding is not None
            best_obj = best_population_individual.obj_value
            best_encoding = best_population_individual.encoding
            best_origin_tag = origin_tag_from_decoding(best_population_individual.decoding)
            history_rows.append(
                _history_row(
                    seed=seed,
                    elapsed_seconds=time.perf_counter() - start,
                    generation=generation,
                    best_obj=best_obj,
                    origin_tag=best_origin_tag,
                )
            )

        if time.perf_counter() >= ts_activation_at:
            elite_count = max(1, int(POP_SIZE * ELITE_RATIO))
            ts_sources = sorted(
                population,
                key=lambda item: item.obj_value or float("inf"),
            )[:elite_count]
            for source in ts_sources:
                if time.perf_counter() >= deadline:
                    break
                assert source.obj_value is not None
                result = run_tabu_search(
                    instance=instance,
                    individual=source,
                    rng=rng,
                    global_best_obj=best_obj,
                    max_iter=max_iter_ts,
                    tabu_len=len_ts,
                    deadline=deadline,
                )
                if result.obj_value < best_obj:
                    best_obj = result.obj_value
                    best_encoding = result.encoding
                    best_origin_tag = "T"
                    history_rows.append(
                        _history_row(
                            seed=seed,
                            elapsed_seconds=time.perf_counter() - start,
                            generation=generation,
                            best_obj=best_obj,
                            origin_tag=best_origin_tag,
                        )
                    )

        population = elitist_tournament_selection(
            population=population,
            rng=rng,
            replacement_factory=lambda: random_individual(instance, rng),
            no_improve=NO_IMPROVE,
        )

    runtime_seconds = time.perf_counter() - start
    history_rows.append(
        _history_row(
            seed=seed,
            elapsed_seconds=runtime_seconds,
            generation=generation,
            best_obj=best_obj,
            origin_tag=best_origin_tag,
        )
    )
    return SingleSeedResult(
        seed=seed,
        best_obj=best_obj,
        best_origin_tag=best_origin_tag,
        best_encoding=best_encoding,
        runtime_seconds=runtime_seconds,
        history_rows=history_rows,
    )


def _seed_record(
    instance_name: str,
    scenario: str,
    baseline: BaselineRecord | None,
    result: SingleSeedResult,
) -> SeedRunRecord:
    ref_ub = baseline.ub if baseline else None
    return SeedRunRecord(
        name=instance_name,
        insName=instance_name,
        scenario=scenario,
        seed=result.seed,
        bestOriginTag=result.best_origin_tag,
        bestSeed=result.seed,
        bestObj=result.best_obj,
        refUb=ref_ub,
        rpdv=rpdv(result.best_obj, ref_ub),
        rpdf=rpdf(result.best_obj, ref_ub),
        runtimeSeconds=result.runtime_seconds,
    )


def _summary_record(
    instance_name: str,
    scenario: str,
    baseline: BaselineRecord | None,
    result: SingleSeedResult,
    seed_count: int,
) -> SummaryRecord:
    ref_ub = baseline.ub if baseline else None
    return SummaryRecord(
        name=instance_name,
        insName=instance_name,
        scenario=scenario,
        bestOriginTag=result.best_origin_tag,
        bestSeed=result.seed,
        bestObj=result.best_obj,
        refUb=ref_ub,
        rpdv=rpdv(result.best_obj, ref_ub),
        rpdf=rpdf(result.best_obj, ref_ub),
        seedCount=seed_count,
    )


def _history_row(
    seed: int,
    elapsed_seconds: float,
    generation: int,
    best_obj: int,
    origin_tag: str,
) -> dict[str, object]:
    return {
        "seed": seed,
        "elapsedSeconds": elapsed_seconds,
        "generation": generation,
        "bestObj": best_obj,
        "bestOriginTag": origin_tag,
    }


def _setup_file_logging(log_path: Path, quiet: bool) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)
