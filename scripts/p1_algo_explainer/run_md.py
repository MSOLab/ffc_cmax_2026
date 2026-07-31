"""Generate the MD (Mixed Dispatch) panel sequence as SVGs (plan §4.1).

Storyboard (one SVG per Gantt panel under
``analysis_outputs/20260612_p1_algo_explainer/md/``):

  step_01  priority sequences  -> NOT a Gantt; emitted as a markdown note.
  step_02  pure mode A (job-then-stage, n_p = n)        Gantt
  step_03  pure mode B (stage-then-job, n_p = 0)        Gantt
  step_04  mixed (n_p = ceil(n/2))                       Gantt
  step_05  final selection (min makespan over ladder)   Gantt, makespan caption

MD mechanism (from hybridflowshop/dispatcher/mixed.py +
hybridflowshop/dispatcher/utils.py::from_job_sequence_get_schedule_mixed):

  ``stage_2_head`` maps the first stage -> n_p ("head"). The first n_p jobs of
  the priority sequence are pushed through ALL stages via
  ``dispatch_job_by_stages`` (job-then-stage, the "staircase"), and the
  remaining n - n_p jobs are filled per stage by ``dispatch_stage_by_jobs``
  (stage-then-job). So:
    n_p = n  -> pure job-then-stage (mode A)
    n_p = 0  -> pure stage-then-job (mode B)
    n_p = ceil(n/2) -> head/tail mix.

  ``MixedDispatcher.get_best_mixed_schedule_by_sequence`` internally sweeps the
  n_p ladder (n, ceil(n/2), ..., 1, 0) and returns only the best. To render the
  distinct ladder rungs as separate panels we drive the underlying
  ``from_job_sequence_get_schedule_mixed`` directly with an explicit n_p.

CLI options (the MD panel build may use a job subset that differs from the
other algorithms -- per-algorithm job sampling is allowed since color/axis
consistency only needs to hold *within* one algorithm's build):

  --no-gantt            Skip all SVG/file output; print objValues only.
  --jobs 0,2,5,...      Build the panels on this 0-based subset of the demo's
                        10 jobs (default: all 10).
  --search-subset-size K  Analysis mode (implies --no-gantt): enumerate every
                        K-of-10 job subset, evaluate the (sequence x n_p) ladder,
                        and report subsets where some *intermediate* n_p
                        (0 < n_p < n) strictly beats BOTH pure modes -- i.e.
                        where mixed dispatch is genuinely meaningful.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schore.parameters_examples.parallel_shop.identical_flow import (  # noqa: E402
    HybridFlowshopParameters,
)

from hybridflowshop.dispatcher.mixed import MixedDispatcher  # noqa: E402
from hybridflowshop.dispatcher.utils import (  # noqa: E402
    from_job_sequence_get_schedule_mixed,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule  # noqa: E402
from scripts.p1_algo_explainer.recorder import GanttSnapshotRecorder  # noqa: E402
from scripts.p1_algo_explainer.render import render_panel  # noqa: E402

INSTANCE_PATH = REPO_ROOT / "resources" / "demo_p1_10x4" / "1.txt"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "20260612_p1_algo_explainer" / "md"

# MD-specific 8-of-10 job subset (0-based into the demo's 10 jobs). Per-algorithm
# job sampling is allowed (color/axis consistency only needs to hold within one
# algorithm's build). This subset is the one where an *intermediate* head size
# (n_p=4, Palmer) strictly beats BOTH pure modes -- pure A (n_p=n)=385 and pure B
# (n_p=0)=378 -- with makespan 367, so the "mixed > pure" story is genuine and
# visible. Found by `run_md.py --search-subset-size 8` (best of 3 such subsets).
MD_JOB_SUBSET = [0, 1, 3, 4, 5, 7, 8, 9]


def load_instance() -> HybridFlowshopParameters:
    with open(INSTANCE_PATH) as f:
        return HybridFlowshopParameters.from_ff2020_data("demo_p1_10x4", f)


def _demo_stage_rows() -> list[list[int]]:
    """Return the 4 stage rows (each 10 processing times) of the demo instance."""
    lines = INSTANCE_PATH.read_text().splitlines()
    # line0: "10 4"  line1: "2 2 2 2"  lines2..5: stage rows (10 ints each)
    return [list(map(int, lines[2 + i].split())) for i in range(4)]


def build_subset_instance(
    job_indices: list[int], name: str = "md_subset"
) -> HybridFlowshopParameters:
    """Build a sub-instance keeping only ``job_indices`` (0-based) of the demo.

    Stages (4) and machines (2/stage) are unchanged; processing times are the
    demo's p_ij for the selected job columns (untouched).
    """
    rows = _demo_stage_rows()
    txt = [f"{len(job_indices)} 4", "2 2 2 2"]
    for r in rows:
        txt.append(" ".join(str(r[c]) for c in job_indices))
    return HybridFlowshopParameters.from_ff2020_data(
        name, io.StringIO("\n".join(txt) + "\n")
    )


def schedule_with_np(
    md: MixedDispatcher,
    job_sequence,
    n_p: int,
) -> HybridFlowshopLiteSchedule:
    """Build a single mixed schedule for an explicit head size n_p.

    n_p is applied at the first stage; the cumulative-cap logic inside
    ``from_job_sequence_get_schedule_mixed`` then propagates it forward.
    """
    schedule = md._create_empty_schedule()
    first_stage = md.stage_id_list[0]
    from_job_sequence_get_schedule_mixed(
        schedule,
        job_sequence,
        md.stage_2_job_2_p,
        {first_stage: n_p},
    )
    return schedule


def eval_ladder(md: MixedDispatcher, job_sequence) -> dict[int, int]:
    """Return {n_p -> makespan} across the full ladder n..0 for one sequence."""
    n = len(md.job_id_list)
    return {
        n_p: schedule_with_np(md, job_sequence, n_p).makespan
        for n_p in range(n, -1, -1)
    }


def _sequences(md: MixedDispatcher) -> dict[str, list[str]]:
    return {
        "Palmer": md.get_palmer_sequence(),
        "Gupta": md.get_gupta_sequence(),
        "CDS": md.get_cds_sequence(k=1),
    }


def search_subsets(subset_size: int) -> None:
    """Enumerate K-of-10 job subsets; report where mixed beats both pure modes.

    'Mixed meaningful' := some intermediate n_p (0 < n_p < n) yields a strictly
    smaller makespan than BOTH pure mode A (n_p=n) and pure mode B (n_p=0), for
    at least one priority sequence.
    """
    pool = list(range(10))
    hits: list[tuple[int, tuple[int, ...], str, int, int, int, int]] = []
    total = 0
    for combo in combinations(pool, subset_size):
        total += 1
        inst = build_subset_instance(list(combo))
        md = MixedDispatcher(inst)
        n = len(md.job_id_list)
        for sname, seq in _sequences(md).items():
            ladder = eval_ladder(md, seq)
            pure_min = min(ladder[n], ladder[0])
            inter = {k: v for k, v in ladder.items() if 0 < k < n}
            if not inter:
                continue
            best_np = min(inter, key=lambda k: inter[k])
            if inter[best_np] < pure_min:  # strictly beats BOTH pure modes
                hits.append(
                    (
                        pure_min - inter[best_np],
                        combo,
                        sname,
                        best_np,
                        inter[best_np],
                        ladder[n],
                        ladder[0],
                    )
                )
    hits.sort(reverse=True)
    print(
        f"search-subset-size={subset_size}: {total} subsets, "
        f"{len(hits)} (subset, sequence) cases where intermediate n_p beats both pure modes"
    )
    for gap, combo, sname, bnp, m, pa, pb in hits:
        print(
            f"  gap={gap:3d}  jobs={list(combo)}  seq={sname:6s}  "
            f"best n_p={bnp} -> {m}   (pure A n_p={n_of(combo)}={pa}, pure B n_p=0={pb})"
        )
    if not hits:
        print("  -> no subset makes intermediate n_p strictly beat both pure modes.")


def n_of(combo: tuple[int, ...]) -> int:
    return len(combo)


def build_and_report(
    instance: HybridFlowshopParameters,
    *,
    no_gantt: bool,
) -> None:
    """Build the MD panels for ``instance``; render SVGs unless no_gantt."""
    md = MixedDispatcher(instance)
    n = instance.job_count
    all_jobs = instance.job_id_list

    seqs = _sequences(md)
    palmer_seq, gupta_seq, cds_seq = seqs["Palmer"], seqs["Gupta"], seqs["CDS"]
    pi = palmer_seq  # selected sequence for the build

    # objValue-only view: full (sequence x n_p) ladder.
    if no_gantt:
        print(f"instance jobs={all_jobs} (n={n})")
        for sname, seq in seqs.items():
            ladder = eval_ladder(md, seq)
            best_np = min(ladder, key=lambda k: ladder[k])
            row = "  ".join(
                f"n_p={k}:{ladder[k]}" for k in sorted(ladder, reverse=True)
            )
            print(f"  {sname:6s} {row}   BEST n_p={best_np}->{ladder[best_np]}")
        return

    recorder = GanttSnapshotRecorder()

    np_mixed = math.ceil(n / 2)
    sched_A = schedule_with_np(md, pi, n)  # pure job-then-stage
    sched_B = schedule_with_np(md, pi, 0)  # pure stage-then-job
    sched_mixed = schedule_with_np(md, pi, np_mixed)

    # Final selection: min makespan across the FULL n_p ladder for pi.
    np_ladder = md._get_np_candidates()
    best_np = None
    best_sched: HybridFlowshopLiteSchedule | None = None
    for cand in np_ladder:
        cand_sched = schedule_with_np(md, pi, cand)
        if best_sched is None or cand_sched.makespan < best_sched.makespan:
            best_sched = cand_sched
            best_np = cand
    assert best_sched is not None

    force_start = 0
    force_end = max(
        sched_A.makespan,
        sched_B.makespan,
        sched_mixed.makespan,
        best_sched.makespan,
    )

    head_jobs = set(pi[:np_mixed])
    mixed_highlight = {
        (job, stage) for job in head_jobs for stage in instance.stage_id_list
    }

    recorder.record_schedule(
        "pure_A_job_then_stage",
        sched_A,
        axis=(force_start, force_end),
        note=(
            f"Pure mode A: job-then-stage (n_p = n = {n}). Each job is pushed "
            f"through all stages first -> staircase. makespan={sched_A.makespan}."
        ),
    )
    recorder.record_schedule(
        "pure_B_stage_then_job",
        sched_B,
        axis=(force_start, force_end),
        note=(
            f"Pure mode B: stage-then-job (n_p = 0). Stages are filled 1->c by "
            f"priority order. makespan={sched_B.makespan}."
        ),
    )
    recorder.record_schedule(
        "mixed_np_half",
        sched_mixed,
        highlight=mixed_highlight,
        axis=(force_start, force_end),
        note=(
            f"Mixed: n_p = ceil(n/2) = {np_mixed}. Head (highlighted) = first "
            f"{np_mixed} jobs job-then-stage; tail = stage-then-job. "
            f"makespan={sched_mixed.makespan}."
        ),
    )
    recorder.record_schedule(
        "final_selection",
        best_sched,
        axis=(force_start, force_end),
        note=(
            f"Final MD selection: min makespan over the n_p ladder for pi "
            f"(chosen n_p = {best_np}). makespan={best_sched.makespan}."
        ),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captions: list[tuple[str, str]] = []

    seq_note = (
        "MD panels use an MD-specific 8-job subset of the demo (jobs "
        f"{MD_JOB_SUBSET}), chosen so an intermediate head size beats both pure "
        "modes; other algorithms keep the full 10-job demo.\n\n"
        "Priority sequences (job indices):\n"
        f"- Palmer: {palmer_seq}\n"
        f"- Gupta:  {gupta_seq}\n"
        f"- CDS(k=1): {cds_seq}\n"
        f"\nSelected sequence pi = Palmer = {pi}"
    )
    (OUT_DIR / "step_01_priority_sequences.md").write_text(seq_note + "\n")
    captions.append(
        ("step_01_priority_sequences.md", "Priority sequences P/G/CDS -> pick pi")
    )

    for idx, snap in enumerate(recorder.snapshots, start=2):
        out_name = f"step_{idx:02d}_{snap.label}.svg"
        render_panel(
            OUT_DIR / out_name,
            snap.start_map,
            snap.end_map,
            all_job_list=all_jobs,
            highlight_op_set=snap.highlight,
            force_start=force_start,
            force_end=force_end,
            show_labels=False,
            stage_list=instance.stage_id_list,
            machine_list_per_stage=instance.stage_2_machines_map,
        )
        captions.append((out_name, snap.note or ""))

    panels_md = ["# MD panels", ""]
    for name, cap in captions:
        panels_md.append(f"- `{name}`: {cap}")
    (OUT_DIR / "panels.md").write_text("\n".join(panels_md) + "\n")

    print(f"force_start={force_start} force_end={force_end} best_np={best_np}")
    for name, cap in captions:
        print(f"  {name}: {cap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MD explainer panels / analysis")
    parser.add_argument(
        "--no-gantt",
        action="store_true",
        help="Skip SVG/file output; print objValues only.",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=None,
        help="Comma-separated 0-based job indices (subset of the demo's 10).",
    )
    parser.add_argument(
        "--search-subset-size",
        type=int,
        default=None,
        help="Analysis mode: scan K-of-10 subsets for meaningful mixed dispatch.",
    )
    args = parser.parse_args()

    if args.search_subset_size is not None:
        search_subsets(args.search_subset_size)
        return

    if args.jobs is not None:
        job_indices = [int(x) for x in args.jobs.split(",") if x.strip() != ""]
        instance = build_subset_instance(job_indices, name="md_subset")
    else:
        # Default MD build uses the 8-job subset where mixed dispatch is
        # meaningful (see MD_JOB_SUBSET). Pass --jobs to override.
        instance = build_subset_instance(MD_JOB_SUBSET, name="md_demo_8job")

    build_and_report(instance, no_gantt=args.no_gantt)


if __name__ == "__main__":
    main()
