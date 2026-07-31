from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _checkpoint_sort_key(summary_path: Path) -> tuple[int, str, int]:
    checkpoint_name = summary_path.parents[2].name
    instance_name = summary_path.parents[1].name
    step_text = checkpoint_name.split("-", 1)[0]
    try:
        step = int(step_text)
    except ValueError:
        step = 10**9
    try:
        instance_id = int(instance_name)
    except ValueError:
        instance_id = 10**9
    return step, checkpoint_name, instance_id


def _read_refs(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    refs: dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            instance_id = (
                row.get("Instance") or row.get("instance_id") or row.get("insName")
            )
            ref_value = row.get("UB") or row.get("ref_obj_value") or row.get("ref")
            if instance_id and ref_value:
                refs[str(int(float(instance_id)))] = float(ref_value)
    return refs


def _rpdf(obj: float, ref: float) -> float:
    return (obj - ref) / ((obj + ref) / 2.0) * 100.0


def _read_final_objectives(scenario_dir: Path) -> dict[str, float]:
    objectives: dict[str, float] = {}
    for summary_path in sorted(scenario_dir.glob("*/results/*_summary.csv")):
        with summary_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[-1]
        instance_text = row.get("insName")
        obj_text = row.get("bestObj")
        if not instance_text or not obj_text:
            continue
        objectives[str(int(float(instance_text)))] = float(obj_text)
    return objectives


def _read_compare_objectives(paths: list[Path]) -> dict[str, tuple[float, str]]:
    best_by_instance: dict[str, tuple[float, str]] = {}
    for path in paths:
        for instance_id, obj in _read_final_objectives(path).items():
            previous = best_by_instance.get(instance_id)
            if previous is None or obj < previous[0]:
                best_by_instance[instance_id] = (obj, path.name)
    return best_by_instance


def _iter_checkpoint_summaries(scenario_dir: Path):
    checkpoint_root = scenario_dir / "checkpoints"
    for summary_path in sorted(
        checkpoint_root.glob("*/*/results/*_summary.csv"),
        key=_checkpoint_sort_key,
    ):
        with summary_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[-1]
        yield summary_path, row


def _iter_final_summaries(scenario_dir: Path):
    for summary_path in sorted(scenario_dir.glob("*/results/*_summary.csv")):
        with summary_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        yield summary_path, rows[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List per-instance checkpoint objectives for a scenario directory."
    )
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument(
        "--compare-scenario-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed scenario directory to compare against. Repeatable.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only include this instance id. Repeatable.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Print only the latest saved checkpoint row per instance.",
    )
    parser.add_argument(
        "--include-final-results",
        action="store_true",
        help="Also include completed final results from the scenario root.",
    )
    args = parser.parse_args()

    refs = _read_refs(args.reference_csv)
    compare_objectives = _read_compare_objectives(args.compare_scenario_dir)
    allowed_instances = {
        str(int(float(instance_id))) for instance_id in args.instance_id
    }
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "checkpoint",
            "instance_id",
            "best_obj",
            "best_bound",
            "ref_obj",
            "rpdf",
            "compare_best_obj",
            "delta_vs_compare_best",
            "compare_source",
            "elapsed_sec",
            "summary_path",
        ],
    )
    writer.writeheader()
    rows_to_write: list[tuple[Path, dict[str, str]]] = list(
        _iter_checkpoint_summaries(args.scenario_dir)
    )
    if args.include_final_results:
        rows_to_write.extend(_iter_final_summaries(args.scenario_dir))
    if allowed_instances:
        rows_to_write = [
            (summary_path, row)
            for summary_path, row in rows_to_write
            if row.get("insName")
            and str(int(float(row["insName"]))) in allowed_instances
        ]
    if args.latest_only:
        latest_by_instance: dict[str, tuple[Path, dict[str, str]]] = {}
        for summary_path, row in rows_to_write:
            instance_id = str(int(float(row["insName"])))
            previous = latest_by_instance.get(instance_id)
            if previous is None or _checkpoint_sort_key(
                summary_path
            ) > _checkpoint_sort_key(previous[0]):
                latest_by_instance[instance_id] = (summary_path, row)
        rows_to_write = sorted(
            latest_by_instance.values(),
            key=lambda item: int(float(item[1]["insName"])),
        )
    for summary_path, row in rows_to_write:
        instance_id = str(int(float(row["insName"])))
        best_obj_text = row.get("bestObj", "")
        best_obj = float(best_obj_text) if best_obj_text else None
        if best_obj is not None and not math.isfinite(best_obj):
            best_obj = None
        ref_obj = refs.get(instance_id)
        compare = compare_objectives.get(instance_id)
        compare_best_obj = compare[0] if compare is not None else None
        writer.writerow(
            {
                "checkpoint": row.get("checkpointCallContext")
                or summary_path.parents[2].name,
                "instance_id": instance_id,
                "best_obj": "" if best_obj is None else f"{best_obj:.0f}",
                "best_bound": row.get("bestBound", ""),
                "ref_obj": "" if ref_obj is None else f"{ref_obj:.0f}",
                "rpdf": ""
                if best_obj is None or ref_obj is None
                else f"{_rpdf(best_obj, ref_obj):.6f}",
                "compare_best_obj": ""
                if compare_best_obj is None
                else f"{compare_best_obj:.0f}",
                "delta_vs_compare_best": ""
                if best_obj is None or compare_best_obj is None
                else f"{best_obj - compare_best_obj:.0f}",
                "compare_source": "" if compare is None else compare[1],
                "elapsed_sec": row.get("totalElapsedTime", ""),
                "summary_path": summary_path,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
