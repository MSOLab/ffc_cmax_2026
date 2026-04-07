from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lb_bucket.mip.app import main as run_mip_main
from lb_bucket.mip.experiment_presets import get_preset, list_presets
from lb_bucket.mip.shared import (
    DEFAULT_INPUT_DIR,
    DEFAULT_SOLUTION_ROOT,
    DEFAULT_SUMMARY_CSV,
    log_progress,
)


@dataclass(frozen=True)
class RunnerConfig:
    preset: str
    preset_description: str
    experiment_name: str
    created_at: str
    note: str | None
    output_dir: str
    log_dir: str
    runner_args: list[str]
    mip_args: list[str]


def _default_experiments_root() -> Path:
    return REPO_ROOT / "lb_bucket" / "runs" / "mip"


def _timestamp_suffix() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a named lb_bucket MIP experiment using a short preset-based command."
        )
    )
    parser.add_argument(
        "--preset",
        default="binary_auto",
        help="Named experiment preset.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Experiment folder name under --experiments-root. "
            "If omitted, '<preset>_<timestamp>' is used."
        ),
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=_default_experiments_root(),
        help="Root directory where experiment folders are created.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional short description of what this experiment is testing.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Override summary CSV.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Override instance directory.",
    )
    parser.add_argument(
        "--solution-root",
        type=Path,
        default=DEFAULT_SOLUTION_ROOT,
        help="Override UB warm-start solution root.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        nargs="*",
        help="Optional explicit list of instance ids.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=24,
        help="Gurobi threads passed through to the MIP runner.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=None,
        help="Override per-instance Gurobi time limit.",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=None,
        help="Optional fixed delta override.",
    )
    parser.add_argument(
        "--delta-pmax-plus-one",
        action="store_true",
        help="Override preset and force delta = p_max + 1 per instance.",
    )
    parser.add_argument(
        "--same-bucket-threshold",
        type=int,
        default=None,
        help="Optional override for automatic delta selection threshold.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the experiment folder if it already exists.",
    )
    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Forward Gurobi logs to the console.",
    )
    parser.add_argument(
        "--display-interval-sec",
        type=int,
        default=None,
        help="Override Gurobi DisplayInterval.",
    )
    parser.add_argument(
        "--disable-ub-warm-start",
        action="store_true",
        help="Do not load persisted UB schedule hints.",
    )
    parser.add_argument(
        "--precedence-formulation",
        choices=("bucket", "d", "e"),
        default=None,
        help="Override the precedence formulation used by the preset.",
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Override the preset and disable all extra strengthening families.",
    )
    parser.add_argument(
        "--disable-valid-ineq-i",
        action="store_true",
        help="Override the preset and disable valid-inequality family (i).",
    )
    parser.add_argument(
        "--disable-valid-ineq-ii",
        action="store_true",
        help="Override the preset and disable valid-inequality family (ii).",
    )
    parser.add_argument(
        "--disable-valid-ineq-iii",
        action="store_true",
        help="Override the preset and disable valid-inequality family (iii).",
    )
    parser.add_argument(
        "--disable-valid-ineq-iv",
        action="store_true",
        help="Override the preset and disable valid-inequality family (iv).",
    )
    parser.add_argument(
        "--show-presets",
        action="store_true",
        help="List the available experiment presets and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved MIP arguments without launching the run.",
    )
    return parser


def _extend_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def _extend_optional_value(args: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def _resolve_experiment_name(preset_name: str, user_name: str | None) -> str:
    if user_name:
        return user_name
    return f"{preset_name}_{_timestamp_suffix()}"


def _build_mip_argv(args: argparse.Namespace, output_dir: Path, log_dir: Path) -> list[str]:
    preset = get_preset(args.preset)
    mip_args = list(preset.mip_args)

    if args.delta is not None and "--delta-pmax-plus-one" in mip_args:
        mip_args = [arg for arg in mip_args if arg != "--delta-pmax-plus-one"]

    mip_args.extend(["--summary-csv", str(args.summary_csv)])
    mip_args.extend(["--input-dir", str(args.input_dir)])
    mip_args.extend(["--solution-root", str(args.solution_root)])
    mip_args.extend(["--output-dir", str(output_dir)])
    mip_args.extend(["--log-dir", str(log_dir)])
    mip_args.extend(["--threads", str(args.threads)])

    _extend_optional_value(mip_args, "--time-limit-sec", args.time_limit_sec)
    _extend_optional_value(mip_args, "--delta", args.delta)
    _extend_optional_value(mip_args, "--same-bucket-threshold", args.same_bucket_threshold)
    _extend_optional_value(mip_args, "--display-interval-sec", args.display_interval_sec)
    _extend_optional_value(mip_args, "--precedence-formulation", args.precedence_formulation)

    _extend_flag(mip_args, "--delta-pmax-plus-one", args.delta_pmax_plus_one)
    _extend_flag(mip_args, "--resume", args.resume)
    _extend_flag(mip_args, "--log-to-console", args.log_to_console)
    _extend_flag(mip_args, "--disable-ub-warm-start", args.disable_ub_warm_start)
    _extend_flag(mip_args, "--base-model-only", args.base_model_only)
    _extend_flag(mip_args, "--disable-valid-ineq-i", args.disable_valid_ineq_i)
    _extend_flag(mip_args, "--disable-valid-ineq-ii", args.disable_valid_ineq_ii)
    _extend_flag(mip_args, "--disable-valid-ineq-iii", args.disable_valid_ineq_iii)
    _extend_flag(mip_args, "--disable-valid-ineq-iv", args.disable_valid_ineq_iv)

    if args.instances:
        mip_args.append("--instances")
        mip_args.extend(str(instance) for instance in args.instances)

    return mip_args


def _write_runner_config(path: Path, config: RunnerConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _write_runner_readme(path: Path, config: RunnerConfig) -> None:
    note_text = config.note or "No note provided."
    lines = [
        f"# {config.experiment_name}",
        "",
        f"- Created at: `{config.created_at}`",
        f"- Preset: `{config.preset}`",
        f"- Preset description: {config.preset_description}",
        f"- Note: {note_text}",
        f"- Output directory: `{config.output_dir}`",
        f"- Log directory: `{config.log_dir}`",
        "",
        "## Runner Args",
        "",
        "```text",
        " ".join(config.runner_args),
        "```",
        "",
        "## Resolved MIP Args",
        "",
        "```text",
        " ".join(config.mip_args),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_presets() -> None:
    print("Available presets:")
    for preset in list_presets():
        print(f"- {preset.name}: {preset.description}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.show_presets:
        _print_presets()
        return

    if args.delta is not None and args.delta_pmax_plus_one:
        raise ValueError("Use either --delta or --delta-pmax-plus-one, not both.")

    experiment_name = _resolve_experiment_name(args.preset, args.name)
    output_dir = args.experiments_root / experiment_name
    log_dir = output_dir / "gurobi_logs"
    preset = get_preset(args.preset)
    mip_argv = _build_mip_argv(args, output_dir, log_dir)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    runner_args = ["uv", "run", "python", "lb_bucket\\run_mip_experiment.py"]
    runner_args.extend(argv if argv is not None else sys.argv[1:])

    runner_config = RunnerConfig(
        preset=preset.name,
        preset_description=preset.description,
        experiment_name=experiment_name,
        created_at=created_at,
        note=args.note,
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        runner_args=runner_args,
        mip_args=mip_argv,
    )
    _write_runner_config(output_dir / "experiment_config.json", runner_config)
    _write_runner_readme(output_dir / "README.md", runner_config)

    log_progress(
        f"Prepared experiment '{experiment_name}' with preset={args.preset}. "
        f"Output directory: {output_dir}"
    )
    log_progress(f"Resolved MIP arguments: {' '.join(mip_argv)}")

    if args.dry_run:
        log_progress("Dry run requested; skipping solver launch.")
        return

    run_mip_main(mip_argv)


if __name__ == "__main__":
    main()
