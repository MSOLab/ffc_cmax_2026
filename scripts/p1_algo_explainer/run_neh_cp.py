"""Generate the NEH-CP (incremental NEH + CP re-optimisation) panel sequence
(plan §10).

NEH-CP combines an NEH-style *incremental construction* with a per-insertion
CP re-optimisation:

  1. An insertion priority sequence is read from the reference schedule
     (default: midpoint sequence).
  2. The jobs are split into batches of ``added_batch_size``.
  3. For each batch in order:
       (a) dispatch the new batch's jobs onto the current partial solution via
           MixedDispatcher (head_for_all_stages) -> the partial schedule grows;
       (b) solve a CP sub-model over the current job subset, hinted by the
           dispatch and with profile-fix precedence on older ops, re-optimising
           the UNFIXED ops -> makespan drops (kept only if it improves);
       (c) dispatch the not-yet-inserted tail jobs to obtain a full feasible
           schedule, tracking the best full makespan.
  4. Return the best full schedule.

Storyboard (one SVG per Gantt panel under
``analysis_outputs/20260612_p1_algo_explainer/neh_cp/``):

  step_01  insertion priority sequence + batch partition  -> markdown note
  step_02  batch 1 inserted + CP                            Gantt (partial)
  step_03  batch 2 just dispatched (before CP)              Gantt (partial)
  step_04  batch 2 after CP (makespan drop)                 Gantt (partial)
  step_05  batch 3 partial (after CP)                       Gantt (partial)
  step_06  batch 4 partial (after CP)                       Gantt (partial)
  step_07  batch 5 partial (after CP) = all jobs            Gantt (partial)
  step_08  final returned best full schedule + makespan     Gantt (full)

Non-invasive capture (plan §10 / §2.1, zero edits to ``hybridflowshop/**``):
we build a ``NehCpConstructor`` and *wrap its bound ``_solve_cp_model``* from
this script (attribute reassignment only). The wrapper records, per batch, the
dispatched partial (CP input), the CP-optimised partial (CP output), the current
job subset, and the job -> inserted-batch map, then defers to the real method.
The actual ``run()`` is then invoked normally, so library behaviour / seed /
time budget are unchanged. Explainer-mode guards are relaxed
(``skip_if_estimated_neh_exceeds_remaining=False``, ``stop_before_final_reserve
=False``, generous ``max_time_per_add``) so every batch always runs on the tiny
demo instance.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from routix import DynamicDataObject, StoppingCriteria  # noqa: E402
from schore.parameters_examples.parallel_shop.identical_flow import (  # noqa: E402
    HybridFlowshopParameters,
)

from hybridflowshop.controller.hfs_cp_lns import (  # noqa: E402
    HybridFlowShopCpLnsController,
)
from hybridflowshop.controller.neh_cp import NehCpConstructor  # noqa: E402
from hybridflowshop.schedule_lite import (  # noqa: E402
    HybridFlowshopLiteSchedule,
    get_midpoint_sequence,
)

from scripts.p1_algo_explainer.render import render_panel  # noqa: E402

INSTANCE_PATH = REPO_ROOT / "resources" / "demo_p1_10x4" / "1.txt"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "20260612_p1_algo_explainer" / "neh_cp"

SEED = 42
STOPPING_TIME_LIMIT = 10**9
THREADS = 1  # single thread -> deterministic CP on the tiny demo instance

# Construction granularity for the storyboard: 10 jobs / 2 = 5 batches.
ADDED_BATCH_SIZE = 2
# Generous per-insertion CP budget so every batch solves to optimality.
MAX_TIME_PER_ADD = 3.0

OpKey = tuple[str, str, str]  # (job, stage, machine)


def load_instance() -> HybridFlowshopParameters:
    with open(INSTANCE_PATH) as f:
        return HybridFlowshopParameters.from_ff2020_data("demo_p1_10x4", f)


def build_initialized_controller(
    instance: HybridFlowshopParameters,
) -> HybridFlowShopCpLnsController:
    """Construct + seed-initialise a controller (mirrors run_isw_cp)."""
    controller = HybridFlowShopCpLnsController(
        instance,
        {"horizon": 100000},
        DynamicDataObject.from_obj([]),
        StoppingCriteria.from_dict({"timelimit": STOPPING_TIME_LIMIT}),
    )
    controller.set_working_dir(Path(tempfile.mkdtemp(prefix="neh_cp_explainer_")))
    controller.set_random_seed(seed=SEED)
    controller.apply_shdlb()
    controller.initialize_by_best_of_selected_dispatches(
        method_list=["bn2d_all_stages", "best_of_mixed_dispatches"],
        error_if_infeasible=False,
        draw_gantt=False,
    )
    return controller


class BatchCapture:
    """One captured NEH-CP insertion step."""

    def __init__(
        self,
        batch_idx: int,
        dispatch_partial: HybridFlowshopLiteSchedule,
        cp_partial: HybridFlowshopLiteSchedule,
        subset: list[str],
        inserted_batch_idx: dict[str, int],
    ) -> None:
        self.batch_idx = batch_idx
        self.dispatch_partial = dispatch_partial
        self.cp_partial = cp_partial
        self.subset = subset
        self.inserted_batch_idx = inserted_batch_idx

    def new_jobs(self) -> set[str]:
        """Jobs first inserted in this batch."""
        return {
            job_id
            for job_id, idx in self.inserted_batch_idx.items()
            if int(idx) == self.batch_idx
        }


def highlight_for_jobs(
    jobs: set[str], stage_list
) -> set[tuple[str, str]]:
    """(job, stage) highlight set spanning every stage of the given jobs."""
    return {(job_id, stage_id) for job_id in jobs for stage_id in stage_list}


def run_neh_cp_with_capture(
    controller: HybridFlowShopCpLnsController,
    instance: HybridFlowshopParameters,
    ref_schedule: HybridFlowshopLiteSchedule,
    *,
    added_batch_size: int = ADDED_BATCH_SIZE,
) -> tuple[list[BatchCapture], HybridFlowshopLiteSchedule]:
    """Run NEH-CP, capturing per-batch (dispatch, CP) partials non-invasively."""
    constructor = NehCpConstructor(controller)
    captures: list[BatchCapture] = []
    original_solve = constructor._solve_cp_model

    def wrapped_solve(partial_sol_best, *args, **kwargs):
        report, new_sol = original_solve(partial_sol_best, *args, **kwargs)
        st = constructor._st
        captures.append(
            BatchCapture(
                batch_idx=int(kwargs.get("batch_idx") or len(captures) + 1),
                dispatch_partial=partial_sol_best.deepcopy(),
                cp_partial=new_sol.deepcopy(),
                subset=list(st.current_job_id_list),
                inserted_batch_idx=dict(st.job_2_inserted_batch_idx),
            )
        )
        return report, new_sol

    constructor._solve_cp_model = wrapped_solve  # type: ignore[method-assign]
    result = constructor.run(
        ref_schedule,
        instance,
        controller.job_2_stage_2_p_dict,
        controller.stage_2_job_2_p_dict,
        added_batch_size=added_batch_size,
        max_time_per_add=MAX_TIME_PER_ADD,
        preserved_head_job_portion=0.0,
        solver_thread_cnt=THREADS,
        # Explainer mode: never skip / early-stop on the tiny instance (plan §2.4).
        skip_if_estimated_neh_exceeds_remaining=False,
        stop_before_final_reserve=False,
    )
    return captures, result.schedule


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NEH-CP explainer panels")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ADDED_BATCH_SIZE,
        help=f"added_batch_size for the insertion (default: {ADDED_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help=(
            "Output subfolder under analysis_outputs/20260612_p1_algo_explainer/ "
            "(default: 'neh_cp' for batch size 2, else 'neh_cp_bs<N>')."
        ),
    )
    args = parser.parse_args()
    batch_size = args.batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    out_subdir = args.out_subdir or (
        "neh_cp" if batch_size == ADDED_BATCH_SIZE else f"neh_cp_bs{batch_size}"
    )
    out_dir = OUT_DIR.parent / out_subdir

    logging.basicConfig(level=logging.WARNING)
    instance = load_instance()
    all_jobs = instance.job_id_list
    all_stages = instance.stage_id_list
    machines_map = instance.stage_2_machines_map

    out_dir.mkdir(parents=True, exist_ok=True)

    controller = build_initialized_controller(instance)
    ref_schedule = controller.solution_manager.get_incumbent()
    seed_makespan = int(ref_schedule.makespan)
    insertion_sequence = get_midpoint_sequence(ref_schedule)

    captures, final_full = run_neh_cp_with_capture(
        controller, instance, ref_schedule, added_batch_size=batch_size
    )
    if not captures:
        raise RuntimeError("NEH-CP produced no batch captures.")

    # Shared axis across the whole build: 0 .. max makespan seen (partial or full)
    # so the incremental growth reads on one timeline.
    force_start = 0
    force_end = max(
        max(int(c.cp_partial.makespan) for c in captures),
        max(int(c.dispatch_partial.makespan) for c in captures),
        int(final_full.makespan),
    )

    captions: list[tuple[str, str]] = []

    def render(name, sched, highlight):
        render_panel(
            out_dir / name,
            sched.get_jik_2_start_time_map(),
            sched.get_jik_2_end_time_map(),
            all_job_list=all_jobs,
            highlight_op_set=highlight,
            force_start=force_start,
            force_end=force_end,
            show_labels=False,
            stage_list=all_stages,
            machine_list_per_stage=machines_map,
        )

    # --- step_01: insertion priority sequence + batch partition (note) ---
    batches = [insertion_sequence[i : i + batch_size]
               for i in range(0, len(insertion_sequence), batch_size)]
    seq_note = (
        "NEH-CP incremental construction with per-insertion CP re-optimisation.\n\n"
        f"Reference (seed) schedule makespan = {seed_makespan}.\n\n"
        "Insertion priority sequence (midpoint order, job IDs):\n"
        f"- {insertion_sequence}\n\n"
        f"Batch partition (added_batch_size = {batch_size}, "
        f"{len(batches)} batches):\n"
    )
    for bi, batch in enumerate(batches, start=1):
        seq_note += f"- batch {bi}: {batch}\n"
    (out_dir / "step_01_insertion_sequence.md").write_text(seq_note + "\n")
    captions.append(
        (
            "step_01_insertion_sequence.md",
            "Insertion priority sequence (midpoint) split into "
            f"{len(batches)} batches of {batch_size}.",
        )
    )

    # --- step_02: batch 1 inserted + CP (construction seed) ---
    c1 = captures[0]
    render(
        "step_02_batch1_partial.svg",
        c1.cp_partial,
        highlight_for_jobs(c1.new_jobs(), all_stages),
    )
    captions.append(
        (
            "step_02_batch1_partial.svg",
            f"Batch 1 inserted + CP: construction seed with "
            f"{len(c1.subset)} jobs (highlighted). Partial makespan="
            f"{int(c1.cp_partial.makespan)}.",
        )
    )

    # --- step_03 / step_04: batch 2 dispatch (before CP) -> after CP ---
    c2 = captures[1]
    hl2 = highlight_for_jobs(c2.new_jobs(), all_stages)
    render("step_03_batch2_dispatch.svg", c2.dispatch_partial, hl2)
    render("step_04_batch2_cp.svg", c2.cp_partial, hl2)
    captions.append(
        (
            "step_03_batch2_dispatch.svg",
            "Batch 2 dispatched (before CP): new jobs (highlighted) appended via "
            f"MixedDispatcher. Partial makespan={int(c2.dispatch_partial.makespan)}.",
        )
    )
    captions.append(
        (
            "step_04_batch2_cp.svg",
            "Batch 2 after CP: the sub-model re-optimises the UNFIXED ops "
            "(older ops keep profile-fixed precedence), compressing the partial "
            f"schedule. Makespan {int(c2.dispatch_partial.makespan)} -> "
            f"{int(c2.cp_partial.makespan)}.",
        )
    )

    # --- step_05 .. step_07: remaining batches' CP partials (growth) ---
    for slot, cap in enumerate(captures[2:], start=5):
        out_name = f"step_{slot:02d}_batch{cap.batch_idx}_partial.svg"
        render(out_name, cap.cp_partial, highlight_for_jobs(cap.new_jobs(), all_stages))
        all_in = len(cap.subset) == instance.job_count
        captions.append(
            (
                out_name,
                f"Batch {cap.batch_idx} after CP: partial now holds "
                f"{len(cap.subset)}/{instance.job_count} jobs"
                f"{' (all jobs inserted)' if all_in else ''}. New jobs "
                f"highlighted. Partial makespan={int(cap.cp_partial.makespan)}.",
            )
        )

    # --- step_08: final returned best full schedule ---
    final_idx = 5 + max(0, len(captures) - 2)
    final_name = f"step_{final_idx:02d}_final_full.svg"
    render(final_name, final_full, set())
    captions.append(
        (
            final_name,
            "Final NEH-CP result: best feasible full schedule across batches "
            f"(makespan={int(final_full.makespan)}; seed was {seed_makespan}).",
        )
    )

    # --- panels.md ---
    panels_md = [
        "# NEH-CP panels (incremental NEH + CP re-optimisation)",
        "",
        "Demo instance: 10 jobs x 4 stages, 2 machines/stage "
        "(`resources/demo_p1_10x4/1.txt`).",
        "",
        f"- Seed (initialisation) makespan = {seed_makespan}.",
        f"- added_batch_size = {batch_size} -> {len(batches)} batches; "
        f"insertion order = midpoint sequence.",
        f"- Shared axis: force_start={force_start}, force_end={force_end}.",
        "",
        "Per-batch dispatch -> CP makespan:",
    ]
    for cap in captures:
        panels_md.append(
            f"- batch {cap.batch_idx} ({len(cap.subset)} jobs): "
            f"{int(cap.dispatch_partial.makespan)} -> {int(cap.cp_partial.makespan)}"
        )
    panels_md += ["", "## Panels"]
    for name, cap_text in captions:
        panels_md.append(f"- `{name}`: {cap_text}")
    (out_dir / "panels.md").write_text("\n".join(panels_md) + "\n")

    print(
        f"seed={seed_makespan} force_end={force_end} "
        f"final_full={int(final_full.makespan)} batches={len(captures)}"
    )
    for name, cap_text in captions:
        print(f"  {name}: {cap_text}")


if __name__ == "__main__":
    main()
