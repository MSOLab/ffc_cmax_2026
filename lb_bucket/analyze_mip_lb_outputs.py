from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from routix.io.yaml import dump_yaml, load_yaml

APPLY_MIP_LB_ELAPSED_RE = re.compile(
    r"\{'method': 'apply_mip_lb'.*'elapsed_sec': ([0-9.eE+-]+)\}"
)
MIP_SOLVE_RUNTIME_RE = re.compile(
    r"(?:Completed|Stopped) instance (\S+): .*total_runtime_sec=([0-9.eE+-]+), "
    r"wall_runtime_sec=([0-9.eE+-]+), model_build_wall_sec=([0-9.eE+-]+)"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze apply_mip_lb runtimes and post-MIP dispatch candidate quality "
            "for a scenario output directory."
        )
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        required=True,
        help="Scenario directory like Outputs_scenarios/<ts>/ff2020/<scenario-name>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/YAML analysis outputs. Defaults to <scenario-dir>/mip_lb_analysis.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _parse_controller_log(
    controller_log_path: Path,
    ins_name: str,
) -> dict[str, float | None]:
    apply_elapsed_sec: float | None = None
    mip_solver_runtime_sec: float | None = None
    mip_wall_runtime_sec: float | None = None
    mip_model_build_wall_sec: float | None = None

    if not controller_log_path.is_file():
        return {
            "apply_mip_lb_elapsed_sec": None,
            "mip_solver_runtime_sec": None,
            "mip_wall_runtime_sec": None,
            "mip_model_build_wall_sec": None,
        }

    for line in controller_log_path.read_text(encoding="utf-8").splitlines():
        if apply_elapsed_sec is None:
            match = APPLY_MIP_LB_ELAPSED_RE.search(line)
            if match is not None:
                apply_elapsed_sec = float(match.group(1))
        if mip_solver_runtime_sec is None:
            match = MIP_SOLVE_RUNTIME_RE.search(line)
            if match is not None and match.group(1) == ins_name:
                mip_solver_runtime_sec = float(match.group(2))
                mip_wall_runtime_sec = float(match.group(3))
                mip_model_build_wall_sec = float(match.group(4))
        if (
            apply_elapsed_sec is not None
            and mip_solver_runtime_sec is not None
            and mip_wall_runtime_sec is not None
            and mip_model_build_wall_sec is not None
        ):
            break

    return {
        "apply_mip_lb_elapsed_sec": apply_elapsed_sec,
        "mip_solver_runtime_sec": mip_solver_runtime_sec,
        "mip_wall_runtime_sec": mip_wall_runtime_sec,
        "mip_model_build_wall_sec": mip_model_build_wall_sec,
    }


def _build_candidate_rankings(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rankings = summary.get("dispatch_candidate_rankings")
    if isinstance(rankings, list):
        return [dict(row) for row in rankings if isinstance(row, dict)]

    dispatch_candidates = summary.get("dispatch_candidates", {}) or {}
    candidate_elapsed_sec = summary.get("dispatch_candidate_elapsed_sec", {}) or {}
    numeric_candidates = {
        str(variant): _safe_float(makespan)
        for variant, makespan in dispatch_candidates.items()
    }
    best_makespan = min(
        (value for value in numeric_candidates.values() if value is not None),
        default=None,
    )
    return [
        {
            "variant": variant,
            "makespan": makespan,
            "gap_to_best": (
                None
                if best_makespan is None or makespan is None
                else float(makespan - best_makespan)
            ),
            "elapsed_sec": _safe_float(candidate_elapsed_sec.get(variant)),
            "is_selected": variant == summary.get("selected_variant"),
        }
        for variant, makespan in sorted(
            numeric_candidates.items(),
            key=lambda item: (
                item[1] is None,
                float("inf") if item[1] is None else item[1],
                item[0],
            ),
        )
    ]


def analyze_scenario_dir(
    scenario_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = scenario_dir / "mip_lb_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    flow = load_yaml(scenario_dir / "subroutine_flow.yaml", encoding="utf-8")
    if not isinstance(flow, list):
        raise ValueError(
            f"Invalid subroutine flow at {scenario_dir / 'subroutine_flow.yaml'}"
        )
    mip_step = next(
        (
            step
            for step in flow
            if isinstance(step, dict) and step.get("method") == "apply_mip_lb"
        ),
        None,
    )
    if mip_step is None:
        raise ValueError("apply_mip_lb step was not found in subroutine_flow.yaml")
    tl_nc_multiplier = _safe_float(mip_step.get("tl_nc_multiplier"))

    instance_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    missing_instances: list[str] = []

    variant_present_count = Counter()
    variant_selected_count = Counter()
    variant_best_count = Counter()
    variant_gap_values: dict[str, list[float]] = defaultdict(list)
    variant_elapsed_values: dict[str, list[float]] = defaultdict(list)

    for instance_dir in sorted(
        path for path in scenario_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ):
        ins_name = instance_dir.name
        metadata_path = (
            instance_dir / "mip_lb" / "solutions" / ins_name / "metadata.json"
        )
        dispatch_summary_path = (
            instance_dir / "mip_lb" / "dispatch" / "dispatch_summary.yaml"
        )
        controller_log_path = instance_dir / "subroutine_controller.log"
        if not metadata_path.is_file() or not dispatch_summary_path.is_file():
            missing_instances.append(ins_name)
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dispatch_summary = load_yaml(dispatch_summary_path, encoding="utf-8") or {}
        if not isinstance(dispatch_summary, dict):
            dispatch_summary = {}

        log_metrics = _parse_controller_log(controller_log_path, ins_name)
        timing = dispatch_summary.get("timing", {}) or {}
        rankings = _build_candidate_rankings(dispatch_summary)
        numeric_makespans = [
            _safe_float(row.get("makespan"))
            for row in rankings
            if _safe_float(row.get("makespan")) is not None
        ]
        best_candidate_makespan = min(numeric_makespans, default=None)
        best_variants = [
            str(row["variant"])
            for row in rankings
            if _safe_float(row.get("makespan")) == best_candidate_makespan
        ]

        job_count = int(metadata["job_count"])
        stage_count = int(metadata["stage_count"])
        nominal_mip_time_limit_sec = (
            float(tl_nc_multiplier) * float(job_count) * float(stage_count)
            if tl_nc_multiplier is not None
            else None
        )
        selected_makespan = _safe_float(dispatch_summary.get("selected_makespan"))
        input_ub = _safe_float(metadata.get("input_ub"))
        apply_elapsed_sec = _safe_float(timing.get("apply_mip_lb_elapsed_sec"))
        if apply_elapsed_sec is None:
            apply_elapsed_sec = log_metrics["apply_mip_lb_elapsed_sec"]
        mip_solver_runtime_sec = _safe_float(timing.get("mip_solver_runtime_sec"))
        if mip_solver_runtime_sec is None:
            mip_solver_runtime_sec = (
                _safe_float(metadata.get("total_runtime_sec"))
                or log_metrics["mip_solver_runtime_sec"]
            )
        mip_wall_runtime_sec = _safe_float(timing.get("mip_wall_runtime_sec"))
        if mip_wall_runtime_sec is None:
            mip_wall_runtime_sec = (
                _safe_float(metadata.get("wall_runtime_sec"))
                or log_metrics["mip_wall_runtime_sec"]
            )
        mip_model_build_wall_sec = _safe_float(timing.get("mip_model_build_wall_sec"))
        if mip_model_build_wall_sec is None:
            mip_model_build_wall_sec = (
                _safe_float(metadata.get("model_build_wall_sec"))
                or log_metrics["mip_model_build_wall_sec"]
            )
        post_mip_dispatch_elapsed_sec = _safe_float(
            timing.get("post_mip_dispatch_elapsed_sec")
        )
        if post_mip_dispatch_elapsed_sec is None:
            if apply_elapsed_sec is not None and mip_wall_runtime_sec is not None:
                post_mip_dispatch_elapsed_sec = apply_elapsed_sec - mip_wall_runtime_sec

        instance_rows.append(
            {
                "ins_name": ins_name,
                "job_count": job_count,
                "stage_count": stage_count,
                "nominal_mip_time_limit_sec": nominal_mip_time_limit_sec,
                "mip_time_limit_sec_used": _safe_float(
                    timing.get("mip_time_limit_sec_used")
                    or metadata.get("time_limit_sec_used")
                ),
                "mip_solver_runtime_sec": mip_solver_runtime_sec,
                "mip_wall_runtime_sec": mip_wall_runtime_sec,
                "mip_model_build_wall_sec": mip_model_build_wall_sec,
                "apply_mip_lb_elapsed_sec": apply_elapsed_sec,
                "post_mip_dispatch_elapsed_sec": post_mip_dispatch_elapsed_sec,
                "selected_variant": dispatch_summary.get("selected_variant"),
                "selected_makespan": selected_makespan,
                "selected_pre_final_local_repair_variant": dispatch_summary.get(
                    "selected_pre_final_local_repair_variant"
                ),
                "selected_pre_final_local_repair_makespan": _safe_float(
                    dispatch_summary.get("selected_pre_final_local_repair_makespan")
                ),
                "best_candidate_makespan": best_candidate_makespan,
                "best_candidate_variants": " ".join(best_variants),
                "input_ub": input_ub,
                "selected_improvement_vs_input_ub": (
                    None
                    if input_ub is None or selected_makespan is None
                    else float(input_ub - selected_makespan)
                ),
                "apply_over_nominal_time_limit": (
                    None
                    if apply_elapsed_sec is None
                    or nominal_mip_time_limit_sec in {None, 0.0}
                    else float(apply_elapsed_sec / nominal_mip_time_limit_sec)
                ),
                "solver_over_nominal_time_limit": (
                    None
                    if mip_solver_runtime_sec is None
                    or nominal_mip_time_limit_sec in {None, 0.0}
                    else float(mip_solver_runtime_sec / nominal_mip_time_limit_sec)
                ),
                "post_dispatch_over_nominal_time_limit": (
                    None
                    if post_mip_dispatch_elapsed_sec is None
                    or nominal_mip_time_limit_sec in {None, 0.0}
                    else float(
                        post_mip_dispatch_elapsed_sec / nominal_mip_time_limit_sec
                    )
                ),
            }
        )

        for row in rankings:
            variant = str(row["variant"])
            makespan = _safe_float(row.get("makespan"))
            gap_to_best = _safe_float(row.get("gap_to_best"))
            elapsed_sec = _safe_float(row.get("elapsed_sec"))
            is_selected = bool(row.get("is_selected"))
            is_best = (
                best_candidate_makespan is not None
                and makespan == best_candidate_makespan
            )
            variant_present_count[variant] += 1
            if is_selected:
                variant_selected_count[variant] += 1
            if is_best:
                variant_best_count[variant] += 1
            if gap_to_best is not None:
                variant_gap_values[variant].append(gap_to_best)
            if elapsed_sec is not None:
                variant_elapsed_values[variant].append(elapsed_sec)
            candidate_rows.append(
                {
                    "ins_name": ins_name,
                    "job_count": job_count,
                    "stage_count": stage_count,
                    "selected_variant": dispatch_summary.get("selected_variant"),
                    "selected_pre_final_local_repair_variant": dispatch_summary.get(
                        "selected_pre_final_local_repair_variant"
                    ),
                    "variant": variant,
                    "makespan": makespan,
                    "gap_to_best": gap_to_best,
                    "elapsed_sec": elapsed_sec,
                    "is_selected": is_selected,
                    "is_best": is_best,
                    "input_ub": input_ub,
                    "improvement_vs_input_ub": (
                        None
                        if input_ub is None or makespan is None
                        else float(input_ub - makespan)
                    ),
                }
            )

    variant_rows: list[dict[str, Any]] = []
    for variant in sorted(variant_present_count):
        gap_values = variant_gap_values[variant]
        elapsed_values = variant_elapsed_values[variant]
        variant_rows.append(
            {
                "variant": variant,
                "present_count": variant_present_count[variant],
                "selected_count": variant_selected_count[variant],
                "best_count": variant_best_count[variant],
                "avg_gap_to_best": _safe_mean(gap_values),
                "median_gap_to_best": _safe_median(gap_values),
                "max_gap_to_best": (max(gap_values) if gap_values else None),
                "avg_elapsed_sec": _safe_mean(elapsed_values),
                "median_elapsed_sec": _safe_median(elapsed_values),
                "exact_best_count": sum(1 for value in gap_values if value == 0.0),
                "within_1_count": sum(1 for value in gap_values if value <= 1.0),
                "within_5_count": sum(1 for value in gap_values if value <= 5.0),
            }
        )
    variant_rows.sort(
        key=lambda row: (
            row["avg_gap_to_best"] is None,
            float("inf") if row["avg_gap_to_best"] is None else row["avg_gap_to_best"],
            -int(row["best_count"]),
            row["variant"],
        )
    )

    instance_csv_path = output_dir / "mip_lb_dispatch_instance_summary.csv"
    candidate_csv_path = output_dir / "mip_lb_dispatch_candidate_long.csv"
    variant_csv_path = output_dir / "mip_lb_dispatch_variant_summary.csv"
    summary_yaml_path = output_dir / "mip_lb_dispatch_analysis.yaml"

    _write_csv(instance_csv_path, instance_rows)
    _write_csv(candidate_csv_path, candidate_rows)
    _write_csv(variant_csv_path, variant_rows)

    apply_ratios = [
        float(row["apply_over_nominal_time_limit"])
        for row in instance_rows
        if row["apply_over_nominal_time_limit"] is not None
    ]
    solver_ratios = [
        float(row["solver_over_nominal_time_limit"])
        for row in instance_rows
        if row["solver_over_nominal_time_limit"] is not None
    ]
    post_dispatch_ratios = [
        float(row["post_dispatch_over_nominal_time_limit"])
        for row in instance_rows
        if row["post_dispatch_over_nominal_time_limit"] is not None
    ]
    summary = {
        "scenario_dir": str(scenario_dir),
        "output_dir": str(output_dir),
        "tl_nc_multiplier": tl_nc_multiplier,
        "analyzed_instance_count": len(instance_rows),
        "missing_instance_count": len(missing_instances),
        "missing_instances": missing_instances,
        "apply_over_nominal_time_limit_mean": _safe_mean(apply_ratios),
        "apply_over_nominal_time_limit_median": _safe_median(apply_ratios),
        "solver_over_nominal_time_limit_mean": _safe_mean(solver_ratios),
        "solver_over_nominal_time_limit_median": _safe_median(solver_ratios),
        "post_dispatch_over_nominal_time_limit_mean": _safe_mean(post_dispatch_ratios),
        "post_dispatch_over_nominal_time_limit_median": _safe_median(
            post_dispatch_ratios
        ),
        "top_selected_variants": [
            {"variant": variant, "selected_count": count}
            for variant, count in variant_selected_count.most_common(10)
        ],
        "top_best_variants": [
            {"variant": variant, "best_count": count}
            for variant, count in variant_best_count.most_common(10)
        ],
        "paths": {
            "instance_summary_csv": str(instance_csv_path),
            "candidate_long_csv": str(candidate_csv_path),
            "variant_summary_csv": str(variant_csv_path),
            "summary_yaml": str(summary_yaml_path),
        },
    }
    dump_yaml(summary, summary_yaml_path)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    summary = analyze_scenario_dir(args.scenario_dir, output_dir=args.output_dir)
    print(
        "Analyzed",
        summary["analyzed_instance_count"],
        "instances;",
        "apply/budget mean=",
        summary["apply_over_nominal_time_limit_mean"],
        "solver/budget mean=",
        summary["solver_over_nominal_time_limit_mean"],
        "post-dispatch/budget mean=",
        summary["post_dispatch_over_nominal_time_limit_mean"],
    )
    print("Saved analysis to", summary["paths"]["summary_yaml"])


if __name__ == "__main__":
    main()
