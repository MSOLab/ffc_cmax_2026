"""Generate the BN2D panel sequence as SVGs (plan section 4.2).

Storyboard (under analysis_outputs/20260612_p1_algo_explainer/bn2d/):

  step_01  bottleneck stage i* + per-job r_j / tr_j   -> markdown note (no Gantt)
  step_02  cap selection L (small r_j) / R (small tr_j), order L || mid || R
           -> Gantt of the bottleneck-only schedule, L+R caps highlighted
  step_03  bottleneck dispatch: i* as a single-stage PMS (L || mid || R)
           -> Gantt restricted to stage i*, L+R caps highlighted
  step_04  right propagation: later stages forward (->, MD)
           -> Gantt of i* + later stages, bottleneck band highlighted
  step_05  left propagation: former stages reversed-time (<-)
           -> FINAL full Gantt, bottleneck band highlighted, makespan in caption

BN2D mechanism (verified against hybridflowshop/dispatcher/bn2d.py +
hybridflowshop/dispatcher/bn2d_option.py + hybridflowshop/select_and_assign.py):

  * Bottleneck stage i*: ``_get_bottleneck_stage`` picks argmax over stages of
    ``index(i) = sum_j p[i][j] / |machines(i)|`` (machine-normalized total load).
  * Per-job r_j (forward workload before i*) and tr_j (rear workload after i*)
    are plain processing-time sums:
        r_j  = sum_{s < i*} p[s][j]
        tr_j = sum_{s > i*} p[s][j]
    (``normalize_by_stage_cnt`` would divide by the stage count; off here.)
  * Cap lists: with ``left_cap_multiplier`` / ``right_cap_multiplier`` the cap
    sizes are K_L = mult_L * |machines(i*)| and K_R = mult_R * |machines(i*)|.
    ``select_and_assign.solve_selection_problem`` chooses disjoint L, R sets
    minimizing sum_{L} r_j + sum_{R} tr_j (CP-SAT). L is then sorted ascending by
    r_j, R descending by tr_j; mid = remaining jobs sorted by (r_j - tr_j). The
    dispatch order at i* is ``L || mid || R``.
  * Bottleneck schedule: i* is dispatched as a single-stage parallel-machine
    problem in that order, with r_j used as per-job release times.
  * Later stages (after i*): dispatched forward, jobs ordered by their i* end
    time -- best of stage-then-job / job-then-stage (MD-style).
  * Former stages (before i*): scheduled on a REVERSED-time sub-instance with
    release times bcmax - start_j, then mapped back (start = makespan - end). This
    is the "<- reversed" pass; later stages are the "-> forward" pass.

Bottleneck choice for the storyboard
-------------------------------------
On this 10x4 demo the machine-normalized bottleneck index is maximal at the LAST
stage i3 (index 337.5 vs i0=259, i1=202.5, i2=231.5). A terminal bottleneck has
no later stages, so right propagation (->) would be empty and the two-way story
collapses to one direction. To illustrate the full BN2D two-way mechanism we run
the dispatcher with the strongest NON-terminal stage, i2, as the bottleneck --
exactly one of the candidates ``get_schedule_by_bn2d_all_stages`` itself sweeps.
This is a faithful single-bottleneck BN2D run (same code path), just not the
global argmax; it is documented in step_01 and in the captions. (It also happens
to beat the i3 run: makespan 465 vs 521 on this instance.)

Capturing intermediates non-invasively
---------------------------------------
``_get_schedule_from_bottleneck_stage`` exposes a ``gantt_draw_func`` callback
that fires (1) right after the bottleneck-only dispatch and (2) after later
stages are appended; the method then returns the FINAL schedule with former
stages added. We deep-copy the schedule inside the callback to snapshot each
intermediate without touching the library. r_j / tr_j and the L/mid/R caps are
recomputed here from ``stage_2_job_2_p_map`` (simple sums + the same selection
helper) for the highlight panels and the step_01 annotation -- the library does
not return them directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schore.parameters_examples.parallel_shop.identical_flow import (  # noqa: E402
    HybridFlowshopParameters,
)

from hybridflowshop.dispatcher.bn2d import BN2DDispatcher  # noqa: E402
from hybridflowshop.dispatcher.bn2d_option import BN2DOption  # noqa: E402
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule  # noqa: E402
from hybridflowshop.select_and_assign import (  # noqa: E402
    solve_selection_problem,
)

from scripts.p1_algo_explainer.recorder import GanttSnapshotRecorder  # noqa: E402
from scripts.p1_algo_explainer.render import render_panel  # noqa: E402

INSTANCE_PATH = REPO_ROOT / "resources" / "demo_p1_10x4" / "1.txt"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "20260612_p1_algo_explainer" / "bn2d"

# Cap multiplier (jobs per cap = multiplier * machines-at-bottleneck).
CAP_MULTIPLIER = 1


def load_instance() -> HybridFlowshopParameters:
    with open(INSTANCE_PATH) as f:
        return HybridFlowshopParameters.from_ff2020_data("demo_p1_10x4", f)


def compute_r_tr(
    instance: HybridFlowshopParameters, bottleneck: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Recompute per-job forward (r_j) and rear (tr_j) workloads around i*.

    Mirrors ``_get_bottleneck_stage_schedule_heuristic`` with
    ``normalize_by_stage_cnt = False``: plain processing-time sums.
    """
    idx = instance.stage_id_list.index(bottleneck)
    before = instance.stage_id_list[:idx]
    after = instance.stage_id_list[idx + 1 :]
    j2s2p = instance.job_2_stage_2_p_map
    r = {j: sum(int(j2s2p[j][s]) for s in before) for j in instance.job_id_list}
    tr = {j: sum(int(j2s2p[j][s]) for s in after) for j in instance.job_id_list}
    return r, tr


def select_caps(
    instance: HybridFlowshopParameters,
    bottleneck: str,
    r: dict[str, int],
    tr: dict[str, int],
) -> tuple[list[str], list[str], list[str]]:
    """Reproduce the L / mid / R selection and ordering used by the dispatcher."""
    machine_cnt = len(instance.stage_2_machines_map[bottleneck])
    k_l = CAP_MULTIPLIER * machine_cnt
    k_r = CAP_MULTIPLIER * machine_cnt
    job_index = {j: i for i, j in enumerate(instance.job_id_list)}

    result = solve_selection_problem(
        jobs=instance.job_id_list, r=r, t=tr, K_L=k_l, K_R=k_r
    )
    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"cap selection failed: {result['status']}")

    left = sorted(result["L_set"], key=lambda j: (r[j], job_index[j]))
    right = sorted(result["R_set"], key=lambda j: (-tr[j], job_index[j]))
    mid = [j for j in instance.job_id_list if j not in left and j not in right]
    mid.sort(key=lambda j: (r[j] - tr[j], job_index[j]))
    return left, mid, right


def main() -> None:
    instance = load_instance()
    all_jobs = instance.job_id_list
    bn2d = BN2DDispatcher(instance)

    # Global (machine-normalized) bottleneck is the terminal stage i3 -> no later
    # stages. Use the strongest non-terminal stage so both -> and <- show.
    global_bottleneck = bn2d._get_bottleneck_stage()
    storyboard_bottleneck = "i2"
    bottleneck = storyboard_bottleneck

    stage_idx = instance.stage_id_list.index(bottleneck)
    later_stages = instance.stage_id_list[stage_idx + 1 :]
    former_stages = instance.stage_id_list[:stage_idx]

    r, tr = compute_r_tr(instance, bottleneck)
    left, mid, right = select_caps(instance, bottleneck, r, tr)
    cap_jobs = set(left) | set(right)
    cap_highlight = {(j, bottleneck) for j in cap_jobs}
    bottleneck_band = {(j, bottleneck) for j in all_jobs}

    # --- run the real dispatcher, snapshotting intermediates via the callback ---
    snaps: list[HybridFlowshopLiteSchedule] = []

    def draw(schedule: HybridFlowshopLiteSchedule, **_kw) -> None:
        snaps.append(schedule.deepcopy())

    option = BN2DOption(
        left_cap_multiplier=CAP_MULTIPLIER, right_cap_multiplier=CAP_MULTIPLIER
    )
    final_schedule = bn2d._get_schedule_from_bottleneck_stage(
        bottleneck, option, gantt_draw_func=draw
    )

    # Callback firing order (verified): [0] bottleneck-only, [1] + later stages.
    # The returned value is the final schedule (former stages added).
    bottleneck_only = snaps[0]
    after_later = snaps[1] if len(snaps) > 1 else snaps[0]

    # Shared axis across all BN2D Gantt panels for comparability.
    force_start = 0
    force_end = max(
        bottleneck_only.makespan,
        after_later.makespan,
        final_schedule.makespan,
    )

    recorder = GanttSnapshotRecorder()

    # step_02: cap selection on the bottleneck-only schedule (L+R highlighted).
    recorder.record_schedule(
        "cap_selection",
        bottleneck_only,
        highlight=cap_highlight,
        axis=(force_start, force_end),
        note=(
            f"Cap selection at bottleneck {bottleneck}: L (small r_j) = {left}, "
            f"R (small tr_j contribution) = {right}; dispatch order "
            f"L || mid || R = {left} || {mid} || {right}. Highlighted bars = L+R "
            f"caps. (caps per side = {CAP_MULTIPLIER} x "
            f"{len(instance.stage_2_machines_map[bottleneck])} machines.)"
        ),
    )

    # step_03: bottleneck dispatched as a single-stage PMS (stage i* only).
    recorder.record_schedule(
        "bottleneck_pms",
        bottleneck_only,
        highlight=cap_highlight,
        axis=(force_start, force_end),
        note=(
            f"Bottleneck dispatch: stage {bottleneck} scheduled as a single-stage "
            f"parallel-machine problem in order L || mid || R (release times r_j). "
            f"L+R caps highlighted. Rendered restricted to stage {bottleneck}."
        ),
    )

    # step_04: right propagation -> later stages forward (MD).
    recorder.record_schedule(
        "right_propagation",
        after_later,
        highlight=bottleneck_band,
        axis=(force_start, force_end),
        note=(
            f"Right propagation (->): later stages {later_stages} dispatched "
            f"forward from i* end times (MD: best of stage-then-job / "
            f"job-then-stage). Bottleneck stage {bottleneck} band highlighted."
        ),
    )

    # step_05: left propagation <- former stages on reversed-time sub-instance.
    recorder.record_schedule(
        "left_propagation_final",
        final_schedule,
        highlight=bottleneck_band,
        axis=(force_start, force_end),
        note=(
            f"Left propagation (<-): former stages {former_stages} scheduled on a "
            f"reversed-time sub-instance (release = bcmax - start at i*), then "
            f"mapped back. FINAL full schedule, makespan = "
            f"{final_schedule.makespan}. Bottleneck band highlighted."
        ),
    )

    # --- render ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captions: list[tuple[str, str]] = []

    # step_01: bottleneck + r_j / tr_j annotation (note only, no Gantt).
    r_lines = "\n".join(
        f"  - {j}: r={r[j]:>3}  tr={tr[j]:>3}  "
        f"{'[L]' if j in left else '[R]' if j in right else '[mid]'}"
        for j in all_jobs
    )
    seq_note = (
        f"# BN2D step_01 -- bottleneck stage + r_j / tr_j\n\n"
        f"Global (machine-normalized load) bottleneck = `{global_bottleneck}` "
        f"(the terminal stage on this demo -> no later stages).\n"
        f"Storyboard bottleneck i* = `{bottleneck}` (strongest non-terminal "
        f"candidate; same code path, shows both -> and <- propagation).\n\n"
        f"Per-job workloads around i* = `{bottleneck}` "
        f"(former stages {former_stages}, later stages {later_stages}):\n\n"
        f"{r_lines}\n\n"
        f"r_j  = sum of processing times in former stages (forward workload)\n"
        f"tr_j = sum of processing times in later stages (rear workload)\n"
    )
    (OUT_DIR / "step_01_bottleneck_r_tr.md").write_text(seq_note)
    captions.append(
        (
            "step_01_bottleneck_r_tr.md",
            f"Bottleneck stage i*={bottleneck} highlighted + per-job r_j / tr_j "
            f"(global argmax is terminal i3; i2 used to show both directions)",
        )
    )

    restrict_to_bottleneck = {
        "bottleneck_pms": [bottleneck],
    }

    for idx, snap in enumerate(recorder.snapshots, start=2):
        out_name = f"step_{idx:02d}_{snap.label}.svg"
        stage_list = restrict_to_bottleneck.get(snap.label, instance.stage_id_list)
        render_panel(
            OUT_DIR / out_name,
            snap.start_map,
            snap.end_map,
            all_job_list=all_jobs,
            highlight_op_set=snap.highlight,
            force_start=force_start,
            force_end=force_end,
            show_labels=False,
            stage_list=stage_list,
            machine_list_per_stage=instance.stage_2_machines_map,
        )
        captions.append((out_name, snap.note or ""))

    panels_md = ["# BN2D panels", ""]
    for name, cap in captions:
        panels_md.append(f"- `{name}`: {cap}")
    (OUT_DIR / "panels.md").write_text("\n".join(panels_md) + "\n")

    print(
        f"global_bottleneck={global_bottleneck} storyboard_bottleneck={bottleneck} "
        f"force_start={force_start} force_end={force_end} "
        f"final_makespan={final_schedule.makespan}"
    )
    for name, cap in captions:
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    main()
