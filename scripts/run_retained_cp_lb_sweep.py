from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from routix import DynamicDataObject, StoppingCriteria
from routix.io.yaml import dump_yaml
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_COMMON_PARAMS_PATH = REPO_ROOT / "resources" / "ff2020_common_params.yaml"
DEFAULT_INPUT_DIR = REPO_ROOT / "resources" / "ff2020big"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "Outputs_debug"
DEFAULT_INSTANCES = [1, 2, 3, 4, 5]
DEFAULT_TIME_LIMIT_MULTIPLIER = 0.01
DEFAULT_THREADS = 16
DEFAULT_SEED = 42
DEFAULT_STOPPING_TIME_LIMIT = 10**9
STANDARD_TOPK_BOTTLENECKS = [1, 2, 3, 4, 5]
STANDARD_QUANTILE_LEVELS = [1, 2, 3, 4, 5]

DEFAULT_INITIALIZATION_STEP = {
    "method": "initialize_by_best_of_selected_dispatches",
    "left_cap_portion": 0.25,
    "right_cap_portion": 0.25,
    "normalize_by_stage_cnt": False,
    "randomize_mid_all": False,
    "reverse_mid_even": False,
    "reverse_mid_all": False,
    "mixed_schedule_for_former_stages": True,
    "mixed_schedule_for_later_stages": True,
    "machine_then_job": True,
    "head_for_all_stages": False,
    "p_agg_method": "sum",
    "mi_agg_method": "max",
    "error_if_infeasible": False,
    "draw_gantt": False,
    "method_list": ["bn2d_all_stages", "best_of_mixed_dispatches"],
}


@dataclass(frozen=True)
class StrategySpec:
    label: str
    description: str
    apply_kwargs: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retained-stage CP-LB sweeps across multiple selector strategies "
            "and save comparison tables."
        )
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        type=int,
        default=DEFAULT_INSTANCES,
        help="Instance IDs to evaluate. Default: 1 2 3 4 5",
    )
    parser.add_argument(
        "--instances-all",
        action="store_true",
        help="Run all numeric instance files found under --input-dir.",
    )
    parser.add_argument(
        "--topk-bottlenecks",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Top-k bottleneck selector counts to include. "
            "Use 1 for first_bottleneck_last, 2 for top-2 bottlenecks, etc."
        ),
    )
    parser.add_argument(
        "--quantile-levels",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Quantile level counts to include, where 1=middle, 2=third-points, "
            "3=quarter-points, etc."
        ),
    )
    parser.add_argument(
        "--quantile-counts",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Direct n values for first_n_quantiles_last. "
            "Use 2 for middle, 3 for tertiles, 4 for quartiles, etc."
        ),
    )
    parser.add_argument(
        "--skip-first-last",
        action="store_true",
        help="Skip the plain first_last baseline strategy.",
    )
    parser.add_argument(
        "--standard-11",
        action="store_true",
        help=(
            "Run the standard 11 cases: first_last, top-1..5 bottlenecks, "
            "and quantile levels 1..5."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"CP-SAT worker count. Default: {DEFAULT_THREADS}",
    )
    parser.add_argument(
        "--tl-nc-multiplier",
        type=float,
        default=DEFAULT_TIME_LIMIT_MULTIPLIER,
        help=(
            "Time limit multiplier for apply_retained_stage_cp_lb. "
            "Actual time limit is multiplier * n * c."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for initialization. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing ff2020 instances. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--common-params-path",
        type=Path,
        default=DEFAULT_COMMON_PARAMS_PATH,
        help=f"Shared common-parameter YAML path. Default: {DEFAULT_COMMON_PARAMS_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root directory. Default: "
            "Outputs_debug/retained_cp_lb_selector_sweep_<timestamp>"
        ),
    )
    return parser.parse_args()


def build_strategy_specs(
    *,
    include_first_last: bool,
    topk_bottlenecks: list[int],
    quantile_levels: list[int],
    quantile_counts: list[int],
) -> list[StrategySpec]:
    specs: list[StrategySpec] = []

    if include_first_last:
        specs.append(
            StrategySpec(
                label="first_last",
                description="Retain only the first and last stages.",
                apply_kwargs={"retained_stage_mode": "first_last"},
            )
        )

    for topk in topk_bottlenecks:
        if topk <= 0:
            continue
        if topk == 1:
            specs.append(
                StrategySpec(
                    label="first_bottleneck_last",
                    description=(
                        "Retain the first stage, the strongest internal bottleneck "
                        "by sum(p_ij)/m_i, and the last stage."
                    ),
                    apply_kwargs={"retained_stage_mode": "first_bottleneck_last"},
                )
            )
        else:
            specs.append(
                StrategySpec(
                    label=f"top{topk}_bottlenecks",
                    description=(
                        f"Retain the first stage, the top-{topk} internal bottleneck "
                        "stages by sum(p_ij)/m_i, and the last stage."
                    ),
                    apply_kwargs={
                        "retained_stage_mode": "first_topk_bottlenecks_last",
                        "extra_bottleneck_count": topk,
                    },
                )
            )

    for quantile_level in quantile_levels:
        if quantile_level <= 0:
            continue
        quantile_count = quantile_level + 1
        if quantile_level == 1:
            specs.append(
                StrategySpec(
                    label="quantile1",
                    description="Retain the first, middle, and last stages.",
                    apply_kwargs={"retained_stage_mode": "first_middle_last"},
                )
            )
        else:
            specs.append(
                StrategySpec(
                    label=f"quantile{quantile_level}",
                    description=(
                        "Retain the first stage and "
                        f"{quantile_level} evenly spaced internal stage points, "
                        "plus the last stage."
                    ),
                    apply_kwargs={
                        "retained_stage_mode": "first_n_quantiles_last",
                        "quantile_count": quantile_count,
                    },
                )
            )

    for quantile_count in quantile_counts:
        if quantile_count < 2:
            continue
        if quantile_count == 2:
            specs.append(
                StrategySpec(
                    label="middle",
                    description="Retain the first, middle, and last stages.",
                    apply_kwargs={"retained_stage_mode": "first_middle_last"},
                )
            )
        else:
            specs.append(
                StrategySpec(
                    label=f"{quantile_count}_quantiles",
                    description=(
                        "Retain the first stage, internal stages at "
                        f"{quantile_count}-quantile cut points, and the last stage."
                    ),
                    apply_kwargs={
                        "retained_stage_mode": "first_n_quantiles_last",
                        "quantile_count": quantile_count,
                    },
                )
            )

    deduped_specs: list[StrategySpec] = []
    seen_labels: set[str] = set()
    for spec in specs:
        if spec.label in seen_labels:
            continue
        seen_labels.add(spec.label)
        deduped_specs.append(spec)
    return deduped_specs


def build_strategy_step(
    *,
    threads: int,
    tl_nc_multiplier: float,
    apply_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": "apply_retained_stage_cp_lb",
        "threads": threads,
        "tl_nc_multiplier": tl_nc_multiplier,
        "save_cp_lb_artifacts": True,
        **apply_kwargs,
    }


def build_full_flow(
    *,
    threads: int,
    tl_nc_multiplier: float,
    seed: int,
    apply_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"method": "set_random_seed", "seed": seed},
        {"method": "apply_shdlb"},
        dict(DEFAULT_INITIALIZATION_STEP),
        build_strategy_step(
            threads=threads,
            tl_nc_multiplier=tl_nc_multiplier,
            apply_kwargs=apply_kwargs,
        ),
    ]


def load_shared_params(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_instance(input_dir: Path, instance_id: int) -> HybridFlowshopParameters:
    instance_path = input_dir / f"{instance_id}.txt"
    with instance_path.open("r", encoding="utf-8") as handle:
        return HybridFlowshopParameters.from_ff2020_data(str(instance_id), handle)


def resolve_instance_ids(args: argparse.Namespace) -> list[int]:
    if args.instances_all:
        instance_ids = sorted(
            int(path.stem)
            for path in args.input_dir.glob("*.txt")
            if path.stem.isdigit()
        )
        if not instance_ids:
            raise ValueError(f"No numeric instance files found under {args.input_dir}.")
        return instance_ids
    return list(args.instances)


def prepare_controller_for_instance(
    *,
    instance: HybridFlowshopParameters,
    shared_param_dict: dict[str, Any],
    seed: int,
    initialization_output_dir: Path,
) -> tuple[Any, float | None, float | None]:
    from hybridflowshop.controller import HybridFlowShopCpLnsController

    ctrl = HybridFlowShopCpLnsController(
        instance,
        shared_param_dict,
        DynamicDataObject.from_obj([]),
        StoppingCriteria.from_dict({"timelimit": DEFAULT_STOPPING_TIME_LIMIT}),
    )
    ctrl.set_working_dir(initialization_output_dir)
    ctrl.set_random_seed(seed=seed)
    ctrl.apply_shdlb()
    ctrl.initialize_by_best_of_selected_dispatches(
        **{
            key: value
            for key, value in DEFAULT_INITIALIZATION_STEP.items()
            if key != "method"
        }
    )
    base_input_lb = ctrl.solution_manager.best_obj_bound
    base_input_ub = ctrl.solution_manager.best_obj_value
    return ctrl, base_input_lb, base_input_ub


def run_strategy_for_prepared_instance(
    *,
    ctrl: Any,
    strategy_step: dict[str, Any],
    output_dir: Path,
    strategy_label: str,
    base_input_lb: float | None,
    base_input_ub: float | None,
) -> dict[str, Any]:
    ctrl.set_working_dir(output_dir)
    result_payload = ctrl.apply_retained_stage_cp_lb(
        **{key: value for key, value in strategy_step.items() if key != "method"}
    )

    result = ctrl.last_retained_cp_lb_result
    if result is None:
        raise RuntimeError(
            "Retained-stage CP-LB did not produce a result for "
            f"instance {ctrl.instance.name}."
        )

    if result_payload is None:
        raise RuntimeError(
            f"Retained-stage CP-LB returned no payload for instance {ctrl.instance.name}."
        )

    return {
        "instance_id": int(ctrl.instance.name),
        "ins_name": ctrl.instance.name,
        "job_count": ctrl.instance.job_count,
        "stage_count": ctrl.instance.stage_count,
        "strategy": strategy_label,
        "retained_stage_mode": result.retained_stage_mode,
        "selected_bottleneck_stage_ids": " ".join(result.selected_bottleneck_stage_ids),
        "bottleneck_stage_id": result.bottleneck_stage_id,
        "retained_stage_ids": " ".join(result.retained_stage_ids),
        "retained_stage_count": len(result.retained_stage_ids),
        "retained_stage_ratios": " ".join(
            f"{ratio:.6g}" for ratio in result.retained_stage_ratios
        ),
        "quantile_count": result.quantile_count,
        "objective_lb": result.objective_lb,
        "objective_ub": result.objective_ub,
        "status": result.status_name,
        "solver_runtime_sec": result.solver_runtime_sec,
        "wall_runtime_sec": result.wall_runtime_sec,
        "time_limit_sec_used": result.time_limit_sec_used,
        "input_lb": base_input_lb,
        "input_ub": base_input_ub,
        "cp_lb_summary_path": str((output_dir / "cp_lb" / "summary.yaml").resolve()),
    }


def write_markdown_table(path: Path, title: str, df: pd.DataFrame) -> None:
    columns = list(df.columns)
    rows = [columns] + df.astype(object).where(pd.notna(df), "").values.tolist()
    col_widths = [
        max(len(str(row[col_idx])) for row in rows) for col_idx in range(len(columns))
    ]

    def format_row(row: list[Any]) -> str:
        cells = [str(value).ljust(col_widths[idx]) for idx, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    lines = [f"# {title}", ""]
    lines.append(format_row(columns))
    lines.append(
        "| " + " | ".join("-" * col_widths[idx] for idx in range(len(columns))) + " |"
    )
    for row in rows[1:]:
        lines.append(format_row(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_comparison_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    df = (
        pd.DataFrame(rows)
        .sort_values(["instance_id", "strategy"])
        .reset_index(drop=True)
    )
    if df.empty:
        return

    df["best_bound_for_instance"] = df.groupby("instance_id")["objective_lb"].transform(
        "max"
    )
    df["bound_gap_to_best"] = df["best_bound_for_instance"] - df["objective_lb"]
    df["bound_gap_pct_to_best"] = df.apply(
        lambda row: (
            row["bound_gap_to_best"] / row["best_bound_for_instance"]
            if pd.notna(row["best_bound_for_instance"])
            and row["best_bound_for_instance"] not in (0, None)
            else None
        ),
        axis=1,
    )
    comparison_csv_path = output_dir / "comparison_long.csv"
    df.to_csv(comparison_csv_path, index=False)

    obj_pivot = df.pivot(index="instance_id", columns="strategy", values="objective_ub")
    bound_pivot = df.pivot(
        index="instance_id", columns="strategy", values="objective_lb"
    )
    runtime_pivot = df.pivot(
        index="instance_id",
        columns="strategy",
        values="solver_runtime_sec",
    )
    stage_pivot = df.pivot(
        index="instance_id",
        columns="strategy",
        values="retained_stage_ids",
    )
    strategy_summary_df = (
        df.groupby("strategy", dropna=False)
        .agg(
            instance_count=("instance_id", "count"),
            mean_objective_lb=("objective_lb", "mean"),
            mean_objective_ub=("objective_ub", "mean"),
            mean_solver_runtime_sec=("solver_runtime_sec", "mean"),
            max_objective_lb=("objective_lb", "max"),
            min_objective_lb=("objective_lb", "min"),
        )
        .reset_index()
        .sort_values(["mean_objective_lb", "mean_objective_ub"], ascending=False)
    )
    obj_pivot.to_csv(output_dir / "comparison_obj_pivot.csv")
    bound_pivot.to_csv(output_dir / "comparison_bound_pivot.csv")
    runtime_pivot.to_csv(output_dir / "comparison_runtime_pivot.csv")
    stage_pivot.to_csv(output_dir / "comparison_stage_pivot.csv")
    strategy_summary_df.to_csv(
        output_dir / "comparison_strategy_summary.csv", index=False
    )

    md_sections = [
        "# Retained-Stage CP-LB Sweep",
        "",
        "## Objective UB",
        "",
        obj_pivot.reset_index().to_string(index=False),
        "",
        "## Objective LB",
        "",
        bound_pivot.reset_index().to_string(index=False),
        "",
        "## Solver Runtime (sec)",
        "",
        runtime_pivot.reset_index().to_string(index=False),
        "",
        "## Retained Stages",
        "",
        stage_pivot.reset_index().to_string(index=False),
        "",
        "## Strategy Summary",
        "",
        strategy_summary_df.to_string(index=False),
        "",
        f"Long-form CSV: `{comparison_csv_path.name}`",
    ]
    (output_dir / "comparison.md").write_text(
        "\n".join(md_sections) + "\n", encoding="utf-8"
    )


def append_comparison_row(output_dir: Path, row: dict[str, Any]) -> None:
    comparison_csv_path = output_dir / "comparison_long.csv"
    comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not comparison_csv_path.exists()
    with comparison_csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def refresh_comparison_outputs_from_long_csv(output_dir: Path) -> None:
    comparison_csv_path = output_dir / "comparison_long.csv"
    if not comparison_csv_path.exists():
        return
    df = pd.read_csv(comparison_csv_path)
    create_comparison_outputs(output_dir, df.to_dict(orient="records"))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    topk_bottlenecks = list(args.topk_bottlenecks or [])
    quantile_levels = list(args.quantile_levels or [])
    quantile_counts = list(args.quantile_counts or [])

    if args.standard_11:
        if not topk_bottlenecks:
            topk_bottlenecks = list(STANDARD_TOPK_BOTTLENECKS)
        if not quantile_levels and not quantile_counts:
            quantile_levels = list(STANDARD_QUANTILE_LEVELS)

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        output_dir = DEFAULT_OUTPUTS_DIR / f"retained_cp_lb_selector_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_param_dict = load_shared_params(args.common_params_path)
    strategy_specs = build_strategy_specs(
        include_first_last=not args.skip_first_last,
        topk_bottlenecks=topk_bottlenecks,
        quantile_levels=quantile_levels,
        quantile_counts=quantile_counts,
    )
    if not strategy_specs:
        raise ValueError("No strategy specifications were resolved.")

    instance_ids = resolve_instance_ids(args)
    logging.info(
        "Resolved %d strategies across %d instances.",
        len(strategy_specs),
        len(instance_ids),
    )

    config_payload = {
        "instances": instance_ids,
        "instances_all": args.instances_all,
        "topk_bottlenecks": topk_bottlenecks,
        "quantile_levels": quantile_levels,
        "quantile_counts": quantile_counts,
        "include_first_last": not args.skip_first_last,
        "standard_11": args.standard_11,
        "threads": args.threads,
        "tl_nc_multiplier": args.tl_nc_multiplier,
        "seed": args.seed,
        "input_dir": str(args.input_dir.resolve()),
        "common_params_path": str(args.common_params_path.resolve()),
        "reuse_initialization_per_instance": True,
        "strategies": [
            {
                "label": spec.label,
                "description": spec.description,
                "apply_kwargs": spec.apply_kwargs,
            }
            for spec in strategy_specs
        ],
    }
    dump_yaml(config_payload, output_dir / "experiment_config.yaml", encoding="utf-8")

    instances = {
        instance_id: load_instance(args.input_dir, instance_id)
        for instance_id in instance_ids
    }

    for spec in strategy_specs:
        dump_yaml(
            build_full_flow(
                threads=args.threads,
                tl_nc_multiplier=args.tl_nc_multiplier,
                seed=args.seed,
                apply_kwargs=spec.apply_kwargs,
            ),
            output_dir / f"{spec.label}_flow.yaml",
            encoding="utf-8",
        )
    for instance_id, instance in instances.items():
        init_dir = output_dir / "_shared_init" / str(instance_id)
        init_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Preparing shared initialization for instance=%s", instance_id)
        ctrl, base_input_lb, base_input_ub = prepare_controller_for_instance(
            instance=instance,
            shared_param_dict=shared_param_dict,
            seed=args.seed,
            initialization_output_dir=init_dir,
        )
        for spec in strategy_specs:
            strategy_step = build_strategy_step(
                threads=args.threads,
                tl_nc_multiplier=args.tl_nc_multiplier,
                apply_kwargs=spec.apply_kwargs,
            )
            instance_output_dir = output_dir / spec.label / str(instance_id)
            instance_output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(
                "Running retained-stage CP-LB sweep: strategy=%s instance=%s",
                spec.label,
                instance_id,
            )
            row = run_strategy_for_prepared_instance(
                ctrl=ctrl,
                strategy_step=strategy_step,
                output_dir=instance_output_dir,
                strategy_label=spec.label,
                base_input_lb=base_input_lb,
                base_input_ub=base_input_ub,
            )
            row["strategy_description"] = spec.description
            append_comparison_row(output_dir, row)
            refresh_comparison_outputs_from_long_csv(output_dir)

    logging.info(
        "Completed retained-stage CP-LB sweep. Outputs saved to %s", output_dir
    )


if __name__ == "__main__":
    main()
