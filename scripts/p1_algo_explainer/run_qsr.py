"""Generate the QSR (Quantize-Schedule-Reconstruct) panels (plan §4.3).

QSR is implemented in this repo as the TAU-COARSENING chain inside
``HybridFlowShopCpLnsController.initialize_by_tau_coarsened_cp``. The slide
message is the "low-resolution proxy" metaphor: solve a cheap 1/tau copy of the
problem, learn a good operation ORDER on it, then upscale that order back to
full resolution and finish with a local repair.

Storyboard (SVGs under ``analysis_outputs/20260612_p1_algo_explainer/qsr/``):

  step_01  QUANTIZE (left)   a real-resolution schedule, shown ONLY to anchor the
                             100-axis -- NOT a baseline. QSR quantizes the
                             INSTANCE p_ij, not this schedule (pure constructor).
  step_02  QUANTIZE (right)  surrogate dispatch (width = ceil(p / tau), tau=25)
                             -> deliberately NOT a shared axis: every panel
                                auto-fits its own makespan. The compression
                                story is carried by the x-tick NUMBERS, not by
                                bar width (plan §9).
  step_03  SCHEDULE (surrogate, before)   first surrogate-CP incumbent
  step_04  SCHEDULE (surrogate, after)    final surrogate-CP optimum
                             -> surrogate axis; makespan drops cheaply.
  step_05  RECONSTRUCT  machine-sequence restore (original p_ij)
  step_06  RECONSTRUCT  stage-sequence restore   (original p_ij)
                             -> better one (by makespan) is marked the winner.
  step_07  REPAIR (before)  the reconstruction installed as QSR's OWN first
                             incumbent (pure constructor -- not an improvement
                             over the step_01 reference)
  step_08  REPAIR (after)   critical-cone local search result
                             -> makespan drops.

x-tick compression story (plan §9)
================================================================================
Each panel auto-fits its own axis (NO shared force_start/force_end). The
tau-compression is read off the x-axis TICK NUMBERS:

  * real-time panels (01, 05-08): x ticks at multiples of 100 -> 100, 200, ...
  * surrogate panels (02-04)    : x ticks at multiples of   4 -> 4, 8, 12, ...

The tick spacing ratio 100 : 4 = 25 = tau is the punchline: a surrogate 1-tick
(4 units) x tau = 100 real units. Bars are the same width on both axes; only the
numbers are 25x apart, so the "low-resolution proxy" reads visually.

tau = 25 -- DOCUMENTED, INTENTIONAL exaggeration for THIS explainer figure ONLY
================================================================================
The paper/production value is tau=5. For this QSR slide figure we exaggerate to
tau=25 so the low-resolution effect is unmistakable. This is a slide-only
figure: the thesis experiment results and data/ffc_cmax/references.md are
UNAFFECTED -- a documented, intentional exception to the "same parameters"
principle. With p in [1, 99], surrogate width = ceil(p / 25) in {1, 2, 3, 4}
(varied, no collapse). See plan §9. (resources/demo_p1_10x4/PROVENANCE.md is NOT
touched: it documents the instance p_ij only; tau is not an instance property.)

NON-INVASIVE direct-helper orchestration (no production edits).
================================================================================
Instead of patching production snapshot hooks, this script constructs the real
controller and calls its tau-coarsen helper methods directly, recording each
return as a snapshot. The orchestration mirrors the real subroutine
``initialize_by_tau_coarsened_cp`` (hfs_cp_lns.py @ line 2948) step for step:

  1. QUANTIZE   ctrl._make_tau_coarsened_instance(tau=5)            (@ 2624)
                width = max(1, ceil(p / tau)); horizon ~ 1/tau of the original.
  2. SCHEDULE   ctrl._get_tau_surrogate_dispatch_schedule(scaled)   (@ 2727)
                  -> surrogate warm-start dispatch (BN2D + mixed), then
                ctrl._solve_local_base_cp_candidate(                (@ 2847)
                    instance=scaled, reference_schedule=<dispatch>,
                    add_reference_precedence=False,
                    snapshot_solution_limit=N)
                  -> (report, final_tau_schedule, intermediate_snapshots).
                The production loop solves the surrogate at @ 3686 with exactly
                this call; the built-in ``snapshot_solution_limit`` /
                ``_FullScheduleSnapshotRecorder`` (@ 2898) already captures the
                CP's improving incumbents, so before->after needs no new hook.
  3. RECONSTRUCT ctrl._restore_original_schedule_from_tau_schedule( (@ 2796)
                    tau_schedule, restore_mode=<machine_sequence|stage_sequence>,
                    make_semi_active=True)
                The production loop restores both modes at @ 3829-3834 and keeps
                the better by makespan (maybe_record_candidate @ 3139); we render
                both and mark the winner.
  4. REPAIR     ctrl.critical_cone_cp(...)                          (@ 716)
                The critical-cone operator works on the controller incumbent.
                QSR is a pure constructor: production registers the reconstructed
                schedule as the FIRST incumbent (is_init @ 3983) and polishes THAT
                (reference_schedule=restored_schedule @ 3914). The explainer seeds
                a dispatch only for the step_01 picture, so before repair it RESETS
                the manager and installs the reconstruction as the first incumbent
                -- otherwise the better dispatch leaks in and the wrong schedule
                gets repaired.

A generous ``computational_time`` (a few seconds) is passed to the surrogate CP
so the tiny 10x4 demo actually solves to optimality (plan §2.4 -- avoid the
time-budget guards skipping the solve). Zero production files are modified.
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
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule  # noqa: E402
from scripts.p1_algo_explainer.recorder import GanttSnapshotRecorder  # noqa: E402
from scripts.p1_algo_explainer.render import render_panel  # noqa: E402

INSTANCE_PATH = REPO_ROOT / "resources" / "demo_p1_10x4" / "1.txt"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "20260612_p1_algo_explainer" / "qsr"

SEED = 42
THREADS = 8
STOPPING_TIME_LIMIT = 10**9  # generous controller budget -> never trips guards.

# Quantization factor. tau=25 is a DOCUMENTED, INTENTIONAL exaggeration for THIS
# explainer figure ONLY (plan §9): surrogate width = ceil(p / TAU) in {1,2,3,4}.
# Paper/production tau=5; thesis results & data/ffc_cmax/references.md UNAFFECTED.
TAU = 25
SURROGATE_CP_TIME = 6.0  # seconds; tiny surrogate solves to OPTIMAL well within.
SURROGATE_SNAPSHOT_LIMIT = 20  # capture the CP's improving incumbents.
REPAIR_TIME = 6.0  # seconds for the critical-cone local search.
REPAIR_REPEATS = 4

OpKey = tuple[str, str, str]  # (job, stage, machine)
HighlightKey = tuple[str, str]  # (job, stage)


def load_instance() -> HybridFlowshopParameters:
    with open(INSTANCE_PATH) as f:
        return HybridFlowshopParameters.from_ff2020_data("demo_p1_10x4", f)


def build_initialized_controller(
    instance: HybridFlowshopParameters,
) -> HybridFlowShopCpLnsController:
    """Construct + seed-initialise a controller (mirrors run_cp_lb / sweep)."""
    controller = HybridFlowShopCpLnsController(
        instance,
        {"horizon": 100000},
        DynamicDataObject.from_obj([]),
        StoppingCriteria.from_dict({"timelimit": STOPPING_TIME_LIMIT}),
    )
    controller.set_working_dir(Path(tempfile.mkdtemp(prefix="qsr_explainer_")))
    controller.set_random_seed(seed=SEED)
    controller.apply_shdlb()
    controller.initialize_by_best_of_selected_dispatches(
        method_list=["bn2d_all_stages", "best_of_mixed_dispatches"],
        error_if_infeasible=False,
        draw_gantt=False,
    )
    return controller


def makespan_tail_highlight(
    schedule: HybridFlowshopLiteSchedule,
) -> set[HighlightKey]:
    """Highlight the makespan-determining tail operations.

    These are the ops whose completion equals the schedule makespan -- the
    binding tail of the critical path, which the repair step targets.
    """
    end_map = schedule.get_jik_2_end_time_map()
    makespan = int(schedule.makespan)
    return {
        (job, stage)
        for (job, stage, _machine), end in end_map.items()
        if int(end) == makespan
    }


def install_as_first_incumbent(
    controller: HybridFlowShopCpLnsController,
    schedule: HybridFlowshopLiteSchedule,
) -> None:
    """Make ``schedule`` the QSR pipeline's OWN first incumbent, then repair it.

    QSR (``initialize_by_tau_coarsened_cp``) is a *pure constructor*: it does not
    start from a pre-existing incumbent. Production registers the reconstructed
    schedule as the FIRST incumbent (is_init, @ 3983) and polishes THAT
    (reference_schedule=restored_schedule @ 3914).

    The explainer seeds a dispatch schedule only to draw the step_01 axis-
    reference panel. If we left it in the manager, ``register`` would KEEP the
    better dispatch and the critical-cone repair would polish the dispatch
    instead of the reconstruction -- repairing the wrong schedule. So we clear
    the manager and install the reconstruction as the genuine first incumbent,
    exactly the schedule production polishes.
    """
    manager = controller.solution_manager
    manager.incumbent_solution = None
    manager.best_obj_value = None
    manager.best_obj_bound = None
    report = controller._make_subroutine_report(
        elapsed_time=0.0,
        obj_value=float(schedule.makespan),
        obj_bound=None,
        is_init=True,
        subroutine_name="qsr_explainer_install_restore",
    )
    manager.register(report, schedule)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    instance = load_instance()
    all_jobs = instance.job_id_list
    stage_list = instance.stage_id_list
    machine_list_per_stage = instance.stage_2_machines_map

    recorder = GanttSnapshotRecorder()

    # === Shared QSR controller (steps 1-6) =================================
    ctrl = build_initialized_controller(instance)

    # --- step_01 QUANTIZE (left): original full-resolution schedule --------
    original_schedule = ctrl.solution_manager.get_incumbent()
    assert original_schedule is not None
    original_ms = int(original_schedule.makespan)

    # 1. QUANTIZE: build the tau-coarsened surrogate instance.
    scaled_instance = ctrl._make_tau_coarsened_instance(TAU)

    # 2a. SCHEDULE (warm start): surrogate dispatch (BN2D + mixed).
    surrogate_dispatch = ctrl._get_tau_surrogate_dispatch_schedule(scaled_instance)
    assert surrogate_dispatch is not None
    surrogate_dispatch_ms = int(surrogate_dispatch.makespan)

    # 2b. SCHEDULE (CP): solve the surrogate, capturing improving incumbents.
    surrogate_report, tau_schedule, surrogate_snapshots = (
        ctrl._solve_local_base_cp_candidate(
            instance=scaled_instance,
            computational_time=SURROGATE_CP_TIME,
            solver_thread_cnt=THREADS,
            use_lns_only=False,
            cp_model_probing_level=1,
            cp_sat_params=None,
            make_semi_active=True,
            reference_schedule=surrogate_dispatch,
            add_reference_precedence=False,
            stage_2_job_2_p_dict=scaled_instance.stage_2_job_2_p_map,
            snapshot_solution_limit=SURROGATE_SNAPSHOT_LIMIT,
        )
    )
    assert tau_schedule is not None, "surrogate CP found no feasible solution"
    surrogate_cp_ms = int(tau_schedule.makespan)

    # First vs final surrogate-CP incumbent (before -> after).
    ordered_snaps = sorted(
        surrogate_snapshots, key=lambda s: int(s.get("snapshot_index", 0))
    )
    if ordered_snaps:
        surrogate_before = ordered_snaps[0]["schedule"]
    else:
        surrogate_before = surrogate_dispatch
    surrogate_before_ms = int(surrogate_before.makespan)

    # 3. RECONSTRUCT: restore the learned order on the original p_ij, both modes.
    restore_machine = ctrl._restore_original_schedule_from_tau_schedule(
        tau_schedule, restore_mode="machine_sequence", make_semi_active=True
    )
    restore_stage = ctrl._restore_original_schedule_from_tau_schedule(
        tau_schedule, restore_mode="stage_sequence", make_semi_active=True
    )
    restore_machine_ms = int(restore_machine.makespan)
    restore_stage_ms = int(restore_stage.makespan)

    # Better restore wins (ties -> machine_sequence, the production list order).
    if restore_stage_ms < restore_machine_ms:
        best_mode = "stage_sequence"
        best_restore = restore_stage
    else:
        best_mode = "machine_sequence"
        best_restore = restore_machine
    best_restore_ms = int(best_restore.makespan)

    # 4. REPAIR: make the reconstruction QSR's own first incumbent, then
    # critical-cone CP. The seeded dispatch (step_01) is cleared first so it
    # cannot leak in -- QSR is a constructor and repairs its OWN restore.
    install_as_first_incumbent(ctrl, best_restore)
    repair_before = ctrl.solution_manager.get_incumbent()
    assert repair_before is not None
    repair_before_ms = int(repair_before.makespan)
    ctrl.critical_cone_cp(
        solver_thread_cnt=THREADS,
        computational_time=REPAIR_TIME,
        repeat_count=REPAIR_REPEATS,
        tail_time_ratio=0.5,
        seed_op_count=10,
        stage_radius=3,
        time_radius_ratio=0.3,
        machine_neighbor_depth=4,
        max_selected_ops=None,
        use_lns_only=False,
    )
    repaired = ctrl.solution_manager.get_incumbent()
    assert repaired is not None
    repaired_ms = int(repaired.makespan)

    # === Axes ==============================================================
    # Plan §9: ALL panels auto-fit their own makespan -- NO shared
    # force_start/force_end. The compression story is carried by the x-tick
    # NUMBERS (real step=100 vs surrogate step=4 = 25x = tau), not bar width.
    # Per-panel x_tick_step keyed by snapshot label.
    real_axis_note = "real-resolution axis, auto-fit; x ticks step 100"
    surrogate_axis_note = "surrogate axis, auto-fit; x ticks step 4 (1/tau resolution)"
    x_tick_step_by_label: dict[str, int] = {
        "quantize_original": 100,
        "quantize_surrogate": 4,
        "schedule_surrogate_before": 4,
        "schedule_surrogate_after": 4,
        "reconstruct_machine_sequence": 100,
        "reconstruct_stage_sequence": 100,
        "repair_before": 100,
        "repair_after": 100,
    }

    # === Record snapshots ==================================================
    winner_note = (
        "machine-sequence wins"
        if best_mode == "machine_sequence"
        else "stage-sequence wins"
    )
    if restore_machine_ms == restore_stage_ms:
        winner_note = "tie -> machine-sequence adopted (production list order)"

    recorder.record_schedule(
        "quantize_original",
        original_schedule,
        note=(
            "QUANTIZE (left): a full-resolution (real p_ij) schedule, shown ONLY "
            "to anchor the real-time 100-axis. QSR quantizes the INSTANCE "
            "(p_ij -> ceil(p/tau)), not this schedule -- it is a pure constructor "
            f"with no baseline, so this makespan is NOT a reference value "
            f"({real_axis_note}). Caption: '실해상도(real-time) 100축 예시 -- "
            "처리시간만 1/tau로 다운샘플(순서 비교 대상 아님)'."
        ),
    )
    recorder.record_schedule(
        "quantize_surrogate",
        surrogate_dispatch,
        note=(
            f"QUANTIZE (right): surrogate at 1/tau resolution (tau={TAU}, "
            f"width=ceil(p/{TAU})). makespan={surrogate_dispatch_ms} "
            f"({surrogate_axis_note}). Bars same width as the real panel; only "
            f"the x-tick numbers are {TAU}x apart. Caption: '원본을 1/tau "
            f"해상도로 다운샘플; x눈금 4단위 (1/tau 해상도, tau={TAU})'."
        ),
    )
    recorder.record_schedule(
        "schedule_surrogate_before",
        surrogate_before,
        note=(
            "SCHEDULE (surrogate, before): first surrogate-CP incumbent. "
            f"makespan={surrogate_before_ms} ({surrogate_axis_note})."
        ),
    )
    recorder.record_schedule(
        "schedule_surrogate_after",
        tau_schedule,
        note=(
            "SCHEDULE (surrogate, after): final surrogate-CP optimum "
            f"(status={surrogate_report.status.name}). "
            f"makespan={surrogate_cp_ms} ({surrogate_axis_note}). Caption: "
            "'작은 탐색공간 -> CP가 좋은 operation 순서를 싸게 학습'."
        ),
    )
    recorder.record_schedule(
        "reconstruct_machine_sequence",
        restore_machine,
        highlight=makespan_tail_highlight(restore_machine),
        note=(
            "RECONSTRUCT (machine-sequence restore): replay the surrogate "
            "per-machine op order at original p_ij. "
            f"makespan={restore_machine_ms} ({real_axis_note})."
        ),
    )
    recorder.record_schedule(
        "reconstruct_stage_sequence",
        restore_stage,
        highlight=makespan_tail_highlight(restore_stage),
        note=(
            "RECONSTRUCT (stage-sequence restore): replay the surrogate per-stage "
            f"op order at original p_ij. makespan={restore_stage_ms} "
            f"({real_axis_note}). {winner_note}. Caption: '학습한 순서를 "
            "원해상도로 업스케일 (2가지 복원, better 채택)'."
        ),
    )
    recorder.record_schedule(
        "repair_before",
        repair_before,
        highlight=makespan_tail_highlight(repair_before),
        note=(
            f"REPAIR (before): the reconstruction ({best_mode}) is QSR's own "
            "first incumbent (pure constructor -- this IS the schedule QSR built, "
            "not an improvement over anything). "
            f"makespan={repair_before_ms} ({real_axis_note})."
        ),
    )
    recorder.record_schedule(
        "repair_after",
        repaired,
        highlight=makespan_tail_highlight(repaired),
        note=(
            "REPAIR (after): critical-cone local search result. "
            f"makespan={repaired_ms} ({real_axis_note}). Caption: "
            "'임계경로 국소수선으로 마무리'."
        ),
    )

    # === Render ============================================================
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captions: list[tuple[str, str]] = []
    for idx, snap in enumerate(recorder.snapshots, start=1):
        out_name = f"step_{idx:02d}_{snap.label}.svg"
        # Plan §9: every panel auto-fits its own makespan (no shared axis); the
        # compression reads off numeric x ticks (real step 100 vs surrogate 4).
        render_panel(
            OUT_DIR / out_name,
            snap.start_map,
            snap.end_map,
            all_job_list=all_jobs,
            highlight_op_set=snap.highlight,
            show_labels=False,
            show_x_ticks=True,
            x_tick_step=x_tick_step_by_label[snap.label],
            stage_list=stage_list,
            machine_list_per_stage=machine_list_per_stage,
        )
        captions.append((out_name, snap.note or ""))

    # === panels.md =========================================================
    panels_md = [
        "# QSR panels (Quantize-Schedule-Reconstruct, tau-coarsening)",
        "",
        "Low-resolution-proxy story: solve a cheap 1/tau copy, learn the operation",
        "ORDER, upscale it back to full resolution, then repair.",
        "",
        "tau=25 (설명용 과장값; 논문/프로덕션은 tau=5). 슬라이드 전용 figure이며 thesis",
        "실험 결과 및 data/ffc_cmax/references.md 와 무관 (plan §9의 의도적·문서화된 예외).",
        "",
        "압축 가시화: 모든 패널은 자체 makespan auto-fit (공유축 폐기). tau 압축은 x축 눈금",
        "숫자로 읽는다 -- real 패널 step=100 (100,200,...) vs surrogate 패널 step=4",
        "(4,8,12,...). 눈금 간격비 100:4 = 25 = tau 가 압축률이며, 막대 너비는 양쪽이 동일.",
        "",
        f"- tau = {TAU} (surrogate width = ceil(p / {TAU})).",
        f"- step_01 axis-reference    : {original_ms} (real p_ij schedule; NOT a "
        "baseline -- QSR is a pure constructor)",
        f"- surrogate dispatch        : {surrogate_dispatch_ms}",
        f"- surrogate CP before->after: {surrogate_before_ms} -> {surrogate_cp_ms} "
        f"(status={surrogate_report.status.name})",
        f"- restore machine-sequence  : {restore_machine_ms}",
        f"- restore stage-sequence    : {restore_stage_ms}  ({winner_note})",
        f"- QSR build -> repair        : {repair_before_ms} -> {repaired_ms} "
        "(reconstruction polished; QSR's own trajectory)",
        "",
        "## Panels",
        "",
    ]
    for name, cap in captions:
        panels_md.append(f"- `{name}`: {cap}")
    (OUT_DIR / "panels.md").write_text("\n".join(panels_md) + "\n")

    print(
        f"tau={TAU} original={original_ms} surrogate_dispatch={surrogate_dispatch_ms} "
        f"surrogate_cp={surrogate_before_ms}->{surrogate_cp_ms} "
        f"restore(machine={restore_machine_ms}, stage={restore_stage_ms}) "
        f"best={best_mode}({best_restore_ms}) "
        f"repair={repair_before_ms}->{repaired_ms}"
    )
    for name, cap in captions:
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    main()
