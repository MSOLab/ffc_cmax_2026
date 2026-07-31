from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


@dataclass(frozen=True)
class InstanceInfo:
    instance_id: str
    n: int
    c: int
    ref: float

    @property
    def workload(self) -> int:
        return self.n * self.c


@dataclass
class RestoreRecord:
    label: str
    instance_id: str
    n: int
    c: int
    ref: float
    final_obj: float | None
    init_obj: float | None
    repair_obj: float | None
    post1_obj: float | None
    post2_obj: float | None
    neh1_obj: float | None
    neh_last_obj: float | None
    pw_last_obj: float | None
    mode: str
    retained_stages: str
    retained_count: int
    input_ub: float | None
    cp_obj_ub: float | None
    cp_obj_lb: float | None
    dispatch_obj: float | None
    dispatch_variant: str
    dispatch_updated: bool | None

    @property
    def final_rpdf(self) -> float | None:
        if self.final_obj is None:
            return None
        return rpdf(self.final_obj, self.ref)

    @property
    def relaxed_gain(self) -> float | None:
        if self.input_ub is None or self.cp_obj_ub is None:
            return None
        return self.input_ub - self.cp_obj_ub

    @property
    def restore_gain(self) -> float | None:
        if self.input_ub is None or self.dispatch_obj is None:
            return None
        return self.input_ub - self.dispatch_obj

    @property
    def restore_loss(self) -> float | None:
        if self.dispatch_obj is None or self.cp_obj_ub is None:
            return None
        return self.dispatch_obj - self.cp_obj_ub

    @property
    def post_to_final_gain(self) -> float | None:
        post = self.post2_obj if self.post2_obj is not None else self.post1_obj
        if post is None or self.final_obj is None:
            return None
        return post - self.final_obj


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def rpdf(obj: float, ref: float) -> float:
    return (obj - ref) / ((obj + ref) / 2.0) * 100.0


def read_reference(path: Path) -> dict[str, InstanceInfo]:
    rows: dict[str, InstanceInfo] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            instance_id = str(int(float(row["Instance"])))
            rows[instance_id] = InstanceInfo(
                instance_id=instance_id,
                n=int(row["n"]),
                c=int(row["c"]),
                ref=float(row["UB"]),
            )
    return rows


def read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}


def read_method_phases(path: Path) -> dict[str, float]:
    phases: dict[str, float] = {}
    if not path.exists():
        return phases
    dispatch_count = 0
    neh_values: list[tuple[float, float]] = []
    pw_values: list[tuple[float, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            method = row.get("method_name", "")
            obj = as_float(row.get("objective_value"))
            end_sec = as_float(row.get("method_end_sec")) or 0.0
            if obj is None:
                continue
            if method == "initialize_by_dispatch_portfolio":
                phases["init"] = obj
            elif method == "critical_schedule_repair_ls":
                phases["repair"] = obj
            elif method == "dispatch_from_retained_cp":
                dispatch_count += 1
                phases["post1" if dispatch_count == 1 else "post2"] = obj
            elif method == "neh_cp" or method.endswith("neh_cp"):
                neh_values.append((end_sec, obj))
            elif method == "incremental_sw_cp" or method.endswith("sw_cp"):
                pw_values.append((end_sec, obj))
            elif method.startswith("solve_base_cp_model"):
                phases["final"] = obj
    if neh_values:
        phases["neh1"] = sorted(neh_values)[0][1]
        phases["neh_last"] = sorted(neh_values)[-1][1]
    if pw_values:
        phases["pw_last"] = sorted(pw_values)[-1][1]
    return phases


def load_run(label: str, scenario_dir: Path, ref: dict[str, InstanceInfo]) -> list[RestoreRecord]:
    records: list[RestoreRecord] = []
    for instance_dir in sorted(scenario_dir.iterdir()):
        if not instance_dir.is_dir() or not instance_dir.name.isdigit():
            continue
        instance_id = str(int(instance_dir.name))
        info = ref.get(instance_id)
        if info is None:
            continue
        summary = read_summary(instance_dir / "results" / f"{instance_id}_summary.csv")
        phases = read_method_phases(instance_dir / "method_end_time_and_obj_value.csv")
        cp_summary = read_yaml(instance_dir / "cp_lb" / "summary.yaml")
        dispatch_summary = read_yaml(
            instance_dir / "cp_lb" / "dispatch" / "dispatch_summary.yaml"
        )
        retained_stage_ids = cp_summary.get("retained_stage_ids") or []
        if not isinstance(retained_stage_ids, list):
            retained_stage_ids = []
        dispatch_obj = as_float(dispatch_summary.get("selected_makespan"))
        if dispatch_obj is None:
            dispatch_obj = as_float(summary.get("retainedCpDispatchObj"))
        updated_text = summary.get("retainedCpDispatchUpdatedIncumbent")
        updated = None
        if updated_text:
            updated = updated_text.strip().lower() == "true"
        records.append(
            RestoreRecord(
                label=label,
                instance_id=instance_id,
                n=info.n,
                c=info.c,
                ref=info.ref,
                final_obj=as_float(summary.get("bestObj")) or phases.get("final"),
                init_obj=phases.get("init"),
                repair_obj=phases.get("repair"),
                post1_obj=phases.get("post1"),
                post2_obj=phases.get("post2"),
                neh1_obj=phases.get("neh1"),
                neh_last_obj=phases.get("neh_last"),
                pw_last_obj=phases.get("pw_last"),
                mode=str(cp_summary.get("retained_stage_mode") or ""),
                retained_stages=" ".join(str(stage) for stage in retained_stage_ids),
                retained_count=len(retained_stage_ids),
                input_ub=as_float(cp_summary.get("input_ub")),
                cp_obj_ub=as_float(cp_summary.get("objective_ub")),
                cp_obj_lb=as_float(cp_summary.get("objective_lb")),
                dispatch_obj=dispatch_obj,
                dispatch_variant=str(dispatch_summary.get("selected_variant") or ""),
                dispatch_updated=updated,
            )
        )
    return records


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def summarize(records: list[RestoreRecord], top: int, focus: set[str]) -> None:
    final_records = [record for record in records if record.final_obj is not None]
    print(f"records={len(records)} final_records={len(final_records)}")
    if final_records:
        print(f"mean_final_rpdf={mean(record.final_rpdf for record in final_records if record.final_rpdf is not None):.6f}")

    by_mode: dict[str, list[RestoreRecord]] = {}
    by_type: dict[tuple[int, int], list[RestoreRecord]] = {}
    for record in records:
        by_mode.setdefault(record.mode, []).append(record)
        by_type.setdefault((record.n, record.c), []).append(record)

    print("\nby_mode")
    print("mode,count,mean_final_rpdf,mean_relaxed_gain,mean_restore_gain,mean_restore_loss,mean_post_to_final_gain")
    for mode, group in sorted(by_mode.items(), key=lambda item: item[0]):
        print(
            ",".join(
                [
                    mode or "(missing)",
                    str(len(group)),
                    fmt(mean_or_none(record.final_rpdf for record in group), 6),
                    fmt(mean_or_none(record.relaxed_gain for record in group), 2),
                    fmt(mean_or_none(record.restore_gain for record in group), 2),
                    fmt(mean_or_none(record.restore_loss for record in group), 2),
                    fmt(mean_or_none(record.post_to_final_gain for record in group), 2),
                ]
            )
        )

    print("\nworst_restore_loss")
    print(detail_header())
    rows = sorted(
        (record for record in records if record.restore_loss is not None),
        key=lambda record: record.restore_loss or 0.0,
        reverse=True,
    )
    for record in rows[:top]:
        print(detail_row(record))

    print("\nbest_restore_gain")
    print(detail_header())
    rows = sorted(
        (record for record in records if record.restore_gain is not None),
        key=lambda record: record.restore_gain or 0.0,
        reverse=True,
    )
    for record in rows[:top]:
        print(detail_row(record))

    print("\nby_type")
    print("n,c,count,mean_final_rpdf,mean_restore_gain,mean_restore_loss,mean_post_to_final_gain")
    for (n, c), group in sorted(by_type.items()):
        print(
            f"{n},{c},{len(group)},"
            f"{fmt(mean_or_none(record.final_rpdf for record in group), 6)},"
            f"{fmt(mean_or_none(record.restore_gain for record in group), 2)},"
            f"{fmt(mean_or_none(record.restore_loss for record in group), 2)},"
            f"{fmt(mean_or_none(record.post_to_final_gain for record in group), 2)}"
        )

    if focus:
        print("\nfocus")
        print(detail_header())
        for record in sorted(records, key=lambda item: int(item.instance_id)):
            if record.instance_id in focus:
                print(detail_row(record))


def mean_or_none(values: Any) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return mean(filtered)


def detail_header() -> str:
    return (
        "label,ins,n,c,final,rpdf,init,repair,post1,post2,neh1,neh_last,pw_last,"
        "mode,retained_count,input_ub,cp_ub,cp_lb,dispatch,relaxed_gain,"
        "restore_gain,restore_loss,post_to_final_gain,dispatch_variant,retained_stages"
    )


def detail_row(record: RestoreRecord) -> str:
    values = [
        record.label,
        record.instance_id,
        str(record.n),
        str(record.c),
        fmt(record.final_obj, 0),
        fmt(record.final_rpdf, 6),
        fmt(record.init_obj, 0),
        fmt(record.repair_obj, 0),
        fmt(record.post1_obj, 0),
        fmt(record.post2_obj, 0),
        fmt(record.neh1_obj, 0),
        fmt(record.neh_last_obj, 0),
        fmt(record.pw_last_obj, 0),
        record.mode,
        str(record.retained_count),
        fmt(record.input_ub, 0),
        fmt(record.cp_obj_ub, 0),
        fmt(record.cp_obj_lb, 0),
        fmt(record.dispatch_obj, 0),
        fmt(record.relaxed_gain, 0),
        fmt(record.restore_gain, 0),
        fmt(record.restore_loss, 0),
        fmt(record.post_to_final_gain, 0),
        record.dispatch_variant,
        record.retained_stages,
    ]
    return ",".join('"' + value.replace('"', '""') + '"' if "," in value else value for value in values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze retained CP-LB restoration versus final incumbent."
    )
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "SCENARIO_DIR"),
        required=True,
    )
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--focus", action="append", default=[])
    args = parser.parse_args()

    ref = read_reference(args.reference_csv)
    records: list[RestoreRecord] = []
    for label, scenario_dir in args.run:
        records.extend(load_run(label, Path(scenario_dir), ref))
    focus = {str(int(float(instance_id))) for instance_id in args.focus}
    summarize(records, args.top, focus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
