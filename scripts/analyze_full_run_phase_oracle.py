from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

DEFAULT_RUNS = {
    "0427": Path("Outputs_scenarios/20260427T232937_696372/ff2020/20260427-1"),
    "0429": Path("Outputs_scenarios/20260429T155141_962595/ff2020/20260429-1"),
    "0501": Path("Outputs_scenarios/20260501T034519_627934/ff2020/20260429-2"),
    "0504": Path("Outputs_scenarios/20260504T184555_707193/ff2020/20260504-01"),
}
DEFAULT_REF = Path("resources/ff2020big_ref/fan2023.csv")
PAIR_RUNS = ("0501", "0504")
OLD_RUNS = ("0427", "0429")
PHASE_ORDER = (
    "init",
    "repair",
    "post1",
    "post2",
    "neh1",
    "neh_last",
    "pw1",
    "pw_last",
    "final",
)


@dataclass(frozen=True)
class InstanceInfo:
    ins: int
    n: int
    c: int
    ref_ub: float
    ref_lb: float


@dataclass
class RunRecord:
    run: str
    ins: int
    final_obj: float | None
    best_bound: float | None
    phases: dict[str, float]
    phase_times: dict[str, float]
    summary: dict[str, str]


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def _rpdf(obj: float, ref: float) -> float:
    return (obj - ref) / ((obj + ref) / 2.0) * 100.0


def load_reference(path: Path) -> dict[int, InstanceInfo]:
    rows: dict[int, InstanceInfo] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ins = int(row["Instance"])
            rows[ins] = InstanceInfo(
                ins=ins,
                n=int(row["n"]),
                c=int(row["c"]),
                ref_ub=float(row["UB"]),
                ref_lb=float(row["LB"]),
            )
    return rows


def read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_phase_csv(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    phases: dict[str, float] = {}
    phase_times: dict[str, float] = {}
    neh_values: list[tuple[float, float]] = []
    pw_values: list[tuple[float, float]] = []
    dispatch_count = 0

    if not path.exists():
        return phases, phase_times

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            method = row["method_name"]
            obj = _as_float(row.get("objective_value"))
            end_sec = _as_float(row.get("method_end_sec")) or 0.0
            if obj is None:
                continue
            if method == "initialize_by_dispatch_portfolio":
                phases["init"] = obj
                phase_times["init"] = end_sec
            elif method == "critical_schedule_repair_ls":
                phases["repair"] = obj
                phase_times["repair"] = end_sec
            elif method == "dispatch_from_retained_cp":
                dispatch_count += 1
                key = "post1" if dispatch_count == 1 else "post2"
                phases[key] = obj
                phase_times[key] = end_sec
            elif method == "neh_cp" or method.endswith("neh_cp"):
                neh_values.append((end_sec, obj))
            elif method == "incremental_sw_cp" or method.endswith("sw_cp"):
                pw_values.append((end_sec, obj))
            elif method.startswith("solve_base_cp_model"):
                phases["final"] = obj
                phase_times["final"] = end_sec

    if neh_values:
        phase_times["neh1"], phases["neh1"] = neh_values[0]
        phase_times["neh_last"], phases["neh_last"] = neh_values[-1]
    if pw_values:
        phase_times["pw1"], phases["pw1"] = pw_values[0]
        phase_times["pw_last"], phases["pw_last"] = pw_values[-1]
    return phases, phase_times


def load_run(run_name: str, run_dir: Path) -> dict[int, RunRecord]:
    records: dict[int, RunRecord] = {}
    for method_path in run_dir.glob("*/method_end_time_and_obj_value.csv"):
        try:
            ins = int(method_path.parent.name)
        except ValueError:
            continue
        phases, phase_times = read_phase_csv(method_path)
        summary = read_summary(method_path.parent / "results" / f"{ins}_summary.csv")
        final_obj = _as_float(summary.get("bestObj")) or phases.get("final")
        best_bound = _as_float(summary.get("bestBound"))
        records[ins] = RunRecord(
            run=run_name,
            ins=ins,
            final_obj=final_obj,
            best_bound=best_bound,
            phases=phases,
            phase_times=phase_times,
            summary=summary,
        )
    return records


def choose_best(
    records: dict[str, RunRecord], names: tuple[str, ...]
) -> tuple[str, RunRecord] | None:
    candidates = [
        (name, records[name])
        for name in names
        if name in records and records[name].final_obj is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[1].final_obj, item[0]))


def first_better_phase(source: RunRecord, target: RunRecord) -> str:
    for phase in PHASE_ORDER:
        source_obj = source.phases.get(phase)
        target_obj = target.phases.get(phase)
        if (
            source_obj is not None
            and target_obj is not None
            and source_obj < target_obj
        ):
            return phase
    return "final_only_or_missing"


def phase_vector(record: RunRecord) -> str:
    parts: list[str] = []
    for phase in PHASE_ORDER:
        obj = record.phases.get(phase)
        if obj is not None:
            parts.append(f"{phase}={obj:.0f}")
    return " ".join(parts)


def summarize(
    ref: dict[int, InstanceInfo],
    runs: dict[str, dict[int, RunRecord]],
    top: int,
    focus: str | None,
) -> None:
    by_ins: dict[int, dict[str, RunRecord]] = defaultdict(dict)
    for run_name, records in runs.items():
        for ins, record in records.items():
            by_ins[ins][run_name] = record

    pair_values: list[float] = []
    four_values: list[float] = []
    extra_rows: list[dict[str, object]] = []
    harmful_rows: list[dict[str, object]] = []
    phase_counter: Counter[str] = Counter()
    phase_gain: Counter[str] = Counter()
    type_gain: Counter[tuple[int, int]] = Counter()
    type_count: Counter[tuple[int, int]] = Counter()
    source_gain: Counter[str] = Counter()
    source_count: Counter[str] = Counter()
    final_gain_by_type: dict[tuple[int, int], list[float]] = defaultdict(list)
    old_loss_final_gain_by_type: dict[tuple[int, int], list[float]] = defaultdict(list)

    for ins in sorted(ref):
        records = by_ins.get(ins, {})
        pair = choose_best(records, PAIR_RUNS)
        four = choose_best(records, tuple(DEFAULT_RUNS))
        if pair is None or four is None:
            continue
        info = ref[ins]
        pair_name, pair_record = pair
        four_name, four_record = four
        pair_obj = pair_record.final_obj
        four_obj = four_record.final_obj
        if pair_obj is None or four_obj is None:
            continue
        pair_values.append(_rpdf(pair_obj, info.ref_ub))
        four_values.append(_rpdf(four_obj, info.ref_ub))
        if four_obj < pair_obj:
            gain = pair_obj - four_obj
            phase = first_better_phase(four_record, pair_record)
            extra_rows.append(
                {
                    "ins": ins,
                    "n": info.n,
                    "c": info.c,
                    "gain": gain,
                    "pair": pair_name,
                    "pair_obj": pair_obj,
                    "source": four_name,
                    "source_obj": four_obj,
                    "phase": phase,
                    "pair_rpdf": _rpdf(pair_obj, info.ref_ub),
                    "source_rpdf": _rpdf(four_obj, info.ref_ub),
                }
            )
            phase_counter[phase] += 1
            phase_gain[phase] += int(gain)
            type_gain[(info.n, info.c)] += int(gain)
            type_count[(info.n, info.c)] += 1
            source_gain[four_name] += int(gain)
            source_count[four_name] += 1

        run_0504 = records.get("0504")
        if run_0504 is not None and run_0504.final_obj is not None:
            pre_final_obj = (
                run_0504.phases.get("pw_last")
                or run_0504.phases.get("neh_last")
                or run_0504.phases.get("post2")
                or run_0504.phases.get("post1")
                or run_0504.phases.get("init")
            )
            if pre_final_obj is not None:
                final_gain = pre_final_obj - run_0504.final_obj
                final_gain_by_type[(info.n, info.c)].append(final_gain)

        old = choose_best(records, OLD_RUNS)
        best0504 = records.get("0504")
        if (
            old is not None
            and best0504 is not None
            and old[1].final_obj is not None
            and best0504.final_obj is not None
        ):
            delta = old[1].final_obj - best0504.final_obj
            if delta > 0:
                pre_final_obj = (
                    best0504.phases.get("pw_last")
                    or best0504.phases.get("neh_last")
                    or best0504.phases.get("post2")
                    or best0504.phases.get("post1")
                    or best0504.phases.get("init")
                )
                if pre_final_obj is not None:
                    old_loss_final_gain_by_type[(info.n, info.c)].append(
                        pre_final_obj - best0504.final_obj
                    )
                harmful_rows.append(
                    {
                        "ins": ins,
                        "n": info.n,
                        "c": info.c,
                        "loss": delta,
                        "old": old[0],
                        "old_obj": old[1].final_obj,
                        "0504_obj": best0504.final_obj,
                        "0504_rpdf": _rpdf(best0504.final_obj, info.ref_ub),
                    }
                )

    print("== Overall ==")
    print(f"instances={len(pair_values)}")
    print(f"pair(0501/0504) mean RPDf={mean(pair_values):.6f}%")
    print(f"four-run oracle mean RPDf={mean(four_values):.6f}%")
    print(
        f"extra winners over pair={len(extra_rows)} total_obj_gain={sum(float(r['gain']) for r in extra_rows):.0f}"
    )
    print()

    print("== Extra Winner Sources ==")
    for run_name, count in source_count.most_common():
        print(f"{run_name}: count={count} gain={source_gain[run_name]}")
    print()

    print("== First Better Phase ==")
    for phase, count in phase_counter.most_common():
        print(f"{phase}: count={count} gain={phase_gain[phase]}")
    print()

    print("== Extra Gain By Type ==")
    for (n, c), gain in type_gain.most_common():
        print(f"{n}x{c}: count={type_count[(n, c)]} gain={gain}")
    print()

    print("== 0504 Final CP Gain By Type ==")
    for (n, c), gains in sorted(
        final_gain_by_type.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        positive_count = sum(1 for gain in gains if gain > 0)
        print(
            f"{n}x{c}: count={len(gains)} gain_sum={sum(gains):.0f} "
            f"gain_mean={mean(gains):.2f} positive={positive_count}"
        )
    print()

    print("== 0504 Final CP Gain On Old-Worse Cases ==")
    for (n, c), gains in sorted(
        old_loss_final_gain_by_type.items(), key=lambda item: -sum(item[1])
    ):
        print(
            f"{n}x{c}: cases={len(gains)} gain_sum={sum(gains):.0f} "
            f"gain_mean={mean(gains):.2f}"
        )
    print()

    print(f"== Top {top} Extra Winners ==")
    extra_rows.sort(key=lambda row: (-float(row["gain"]), int(row["ins"])))
    for row in extra_rows[:top]:
        print(
            f"ins={row['ins']:>3} type={row['n']}x{row['c']} gain={row['gain']:.0f} "
            f"{row['source']}:{row['source_obj']:.0f} vs {row['pair']}:{row['pair_obj']:.0f} "
            f"first={row['phase']} rpdf {row['pair_rpdf']:.3f}->{row['source_rpdf']:.3f}"
        )
    print()

    print(f"== Top {top} Cases Where Old Best Hurts vs 0504 ==")
    harmful_rows.sort(key=lambda row: (-float(row["loss"]), int(row["ins"])))
    for row in harmful_rows[:top]:
        print(
            f"ins={row['ins']:>3} type={row['n']}x{row['c']} loss={row['loss']:.0f} "
            f"{row['old']}:{row['old_obj']:.0f} vs 0504:{row['0504_obj']:.0f} "
            f"0504_rpdf={row['0504_rpdf']:.3f}"
        )
    print()

    if focus:
        focus_ids = [int(part) for part in focus.split(",") if part.strip()]
        print("== Focus Phase Tables ==")
        for ins in focus_ids:
            info = ref[ins]
            print(f"ins={ins} type={info.n}x{info.c} ref={info.ref_ub:.0f}")
            for run_name in DEFAULT_RUNS:
                record = by_ins.get(ins, {}).get(run_name)
                if record is None or record.final_obj is None:
                    continue
                print(
                    f"  {run_name} final={record.final_obj:.0f} rpdf={_rpdf(record.final_obj, info.ref_ub):.3f} "
                    f"bound={record.best_bound if record.best_bound is not None else ''} "
                    f"{phase_vector(record)}"
                )
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze phase-level outcomes across full 2nc runs."
    )
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--focus",
        default="120,68,24,88,179,83,131,203,183,107,87,103,223",
        help="Comma-separated instance ids for phase tables.",
    )
    for run_name, default_path in DEFAULT_RUNS.items():
        parser.add_argument(f"--run-{run_name}", type=Path, default=default_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref = load_reference(args.ref)
    runs = {
        "0427": load_run("0427", args.run_0427),
        "0429": load_run("0429", args.run_0429),
        "0501": load_run("0501", args.run_0501),
        "0504": load_run("0504", args.run_0504),
    }
    summarize(ref, runs, top=args.top, focus=args.focus)


if __name__ == "__main__":
    main()
