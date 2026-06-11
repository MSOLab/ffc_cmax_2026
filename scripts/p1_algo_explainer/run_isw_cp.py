"""Generate the ISW-CP (incremental sliding-window CP) panel sequence (plan §4.5).

ISW-CP == the prefix-window / PW-CP machinery: operations on each stage are
sorted into time-ordered batches, and a sliding window partitions them into
five regions around the UNFIXED center. The window subproblem re-optimises the
UNFIXED ops (CP-SAT) while keeping profile-fixed precedence and time-fixed
anchors, then slides forward; once a pass stops improving, the UNFIXED width is
incrementally enlarged.

Storyboard (one SVG per Gantt panel under
``analysis_outputs/20260612_p1_algo_explainer/isw_cp/``), in the sister repo
ffc_dw_wET_2026's pw_cp/visual.py style: 5-region bar coloring, a black-bordered
UNFIXED core, and -- instead of one global window -- per-machine LTF/active/RTF
background bands with dashed time-fixed boundary segments on each lane:

  step_01  5-region partition at window position w (LTF/LPF/UNFIXED/RPF/RTF)
  step_02  right-justify (backward ALAP pass on non-LTF ops): before -> after
  step_03  window subproblem solve (UNFIXED rearranged): before -> after
  step_04  slide: window advances by Delta (step_size) -> 3 consecutive positions
  step_05  incremental enlargement: U0 = 2 (narrow) vs U_max = 8 (wide)

The REAL ISW/PW-CP API (discovered + verified, file:line)
================================================================================
- hybridflowshop/cpsat_model_2/pw_cp.py
    OperationPartition (frozen dataclass @ line 24): fields left_time_fixed,
    left_profile_fixed, unfixed, right_profile_fixed, right_time_fixed, each a
    tuple of (job_id, mc_id) on a stage. Region membership is by BATCH INDEX
    along the time-ordered batches per stage.
- hybridflowshop/controller/pw_cp.py
    PwCpConstructor(ctx)                                          (@ line 245)
      .build_stage_2_batch_list(schedule, batch_size, sort_by_start_time=False)
                                                                  (@ line 478)
        -> dict[stage_id, list[batch]] of time-ordered (job, mc) batches.
      ._build_operation_partition(batch_list_on_stage, stage_id, *,
        unfixed_batch_start_idx, unfixed_batch_count,
        left_profile_fixed_batch_count=0, right_profile_fixed_batch_count=0)
                                                                  (@ line 539)
        -> OperationPartition for one stage at a given window position.
      ._build_batch_spec(incumbent, stage_2_partition, stage_2_job_2_p_dict,
        batch_idx) -> PwCpSubproblemSpec  (@ line 662)  -- runs the backward
        right-justification (make_right_justified on non-left-time-fixed ops)
        internally and stores the result as spec.init_schedule.
      ._solve_batch_pw_cp_model / ._solve_makespan_batch         (@ 944 / 1043)
        -- solve one window subproblem; dispatch chosen by
        spec.is_right_time_fixed_empty (same rule as .run @ line 420).
      .run(ref_schedule, instance, stage_2_job_2_p_dict, *, batch_size,
        step_size, unfixed_batch_count, left_profile_fixed_batch_count,
        right_profile_fixed_batch_count, ...) -> PwCpResult       (@ line 268)
        -- the full sliding loop.
    PwCpRunState (@ line 180) -- the run-state object the solve helpers need;
      we construct one directly (non-invasive) for the single-window panel.
- HybridFlowshopLiteSchedule.make_right_justified(stage_2_job_2_duration, *,
  operation_set)  (schedule_lite.py @ line 1571) -- the ALAP backward pass.

Non-invasive strategy (plan §2.1 / §4.5): we call build_stage_2_batch_list +
_build_operation_partition directly on the incumbent to COMPUTE 5-region
membership per op (panels 1, 4, 5). For the right-justify panel (2) we copy the
incumbent and call make_right_justified on the non-LTF ops, exactly as
_build_batch_spec does. For the window-solve panel (3) we drive
_build_batch_spec + the existing solve helper with a manually constructed
PwCpRunState. No production module is modified and no snapshot hook is added.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria  # noqa: E402
from schore.parameters_examples.parallel_shop.identical_flow import (  # noqa: E402
    HybridFlowshopParameters,
)

from hybridflowshop.controller.hfs_cp_lns import (  # noqa: E402
    HybridFlowShopCpLnsController,
)
from hybridflowshop.controller.pw_cp import (  # noqa: E402
    PwCpConstructor,
    PwCpRunState,
)
from hybridflowshop.cpsat_model_2.pw_cp import (  # noqa: E402
    OperationPartition,
    create_pw_cp_schedule,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule  # noqa: E402
from mbls.cpsat import ObjValueBoundStore  # noqa: E402

from scripts.p1_algo_explainer.render import render_panel  # noqa: E402

INSTANCE_PATH = REPO_ROOT / "resources" / "demo_p1_10x4" / "1.txt"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "20260612_p1_algo_explainer" / "isw_cp"

SEED = 42
STOPPING_TIME_LIMIT = 10**9
THREADS = 8

# Window configuration for the storyboard (batch_size = 1 -> one op per batch).
BATCH_SIZE = 1
STEP_SIZE = 2  # Delta for the slide panel
UNFIXED_COUNT = 2  # U0 (narrow window)
UNFIXED_COUNT_WIDE = 8  # U_max (wide window)
LEFT_PF = 1  # left profile-fixed buffer (batches)
RIGHT_PF = 1  # right profile-fixed buffer (batches)
WINDOW_START = 3  # window position w for the partition / right-justify / solve panels

OpKey = tuple[str, str, str]  # (job, stage, machine)

# Five region colors, mirroring the sister repo ffc_dw_wET_2026's
# pw_cp/visual.py palette: blue-grays for the time-fixed anchors (LTF/RTF),
# ambers for the profile-fixed shoulders (LPF/RPF), and a green UNFIXED core.
# UNFIXED is additionally made salient at draw time via a thick black border
# (passed as the panel's highlight_op_set), so the active window pops without
# needing a clashing color.
REGION_COLORS: dict[str, str] = {
    "left_time_fixed": "#90A4AE",  # blue-grey (anchored, far left)
    "left_profile_fixed": "#FFCC80",  # light amber (order kept, may shift)
    "unfixed": "#81C784",  # green (the re-optimised core; bordered)
    "right_profile_fixed": "#FF9800",  # strong amber (order kept, may shift)
    "right_time_fixed": "#607D8B",  # dark blue-grey (anchored, far right)
}
REGION_LABEL: dict[str, str] = {
    "left_time_fixed": "LTF",
    "left_profile_fixed": "LPF",
    "unfixed": "UNFIXED",
    "right_profile_fixed": "RPF",
    "right_time_fixed": "RTF",
}

# (LTF-zone, active-zone, RTF-zone) colors for the faint per-lane background
# bands. The middle band spans the LPF/UNFIXED/RPF "active" region, so it reuses
# the UNFIXED green.
BAND_REGION_COLORS: tuple[str, str, str] = (
    REGION_COLORS["left_time_fixed"],
    REGION_COLORS["unfixed"],
    REGION_COLORS["right_time_fixed"],
)

# Legend (color key) shown on every ISW-CP panel.
REGION_LEGEND: list[tuple[str, str]] = [
    (REGION_LABEL[name], color) for name, color in REGION_COLORS.items()
]


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
    controller.set_working_dir(Path(tempfile.mkdtemp(prefix="isw_cp_explainer_")))
    controller.set_random_seed(seed=SEED)
    controller.apply_shdlb()
    controller.initialize_by_best_of_selected_dispatches(
        method_list=["bn2d_all_stages", "best_of_mixed_dispatches"],
        error_if_infeasible=False,
        draw_gantt=False,
    )
    return controller


def build_stage_partitions(
    constructor: PwCpConstructor,
    schedule: HybridFlowshopLiteSchedule,
    *,
    unfixed_batch_start_idx: int,
    unfixed_batch_count: int,
    left_profile_fixed_batch_count: int,
    right_profile_fixed_batch_count: int,
) -> dict[str, OperationPartition]:
    """Compute the per-stage 5-region partition at one window position."""
    stage_2_batch_list = constructor.build_stage_2_batch_list(
        schedule, batch_size=BATCH_SIZE
    )
    stage_2_partition: dict[str, OperationPartition] = {}
    for stage_id in schedule.stages:
        stage_2_partition[stage_id] = constructor._build_operation_partition(
            stage_2_batch_list[stage_id],
            stage_id,
            unfixed_batch_start_idx=unfixed_batch_start_idx,
            unfixed_batch_count=unfixed_batch_count,
            left_profile_fixed_batch_count=left_profile_fixed_batch_count,
            right_profile_fixed_batch_count=right_profile_fixed_batch_count,
        )
    return stage_2_partition


def partition_to_js_color_map(
    stage_2_partition: dict[str, OperationPartition],
) -> dict[tuple[str, str], str]:
    """Map each (job, stage) to its region color.

    Keyed by (job, stage) -- NOT (job, stage, mc) -- because a job has exactly
    one op per stage, and the window-solve candidate may reassign an op to the
    other machine in its stage. A (job, stage, mc) key would then miss the
    reassigned op against the after-schedule, dropping it back to the by-job
    color; the (job, stage) key is reassignment-proof.
    """
    js_color: dict[tuple[str, str], str] = {}
    for stage_id, partition in stage_2_partition.items():
        for region_name, color in REGION_COLORS.items():
            for job_id, _mc_id in getattr(partition, region_name):
                js_color[(job_id, stage_id)] = color
    return js_color


def schedule_region_color_map(
    schedule: HybridFlowshopLiteSchedule,
    stage_2_partition: dict[str, OperationPartition],
) -> dict[OpKey, str]:
    """Color a schedule's ACTUAL ops by region, via the (job, stage) lookup.

    Building the map from the schedule's own (job, stage, mc) keys guarantees
    every drawn bar is region-colored even after the window solve moves an op to
    a different machine."""
    js_color = partition_to_js_color_map(stage_2_partition)
    start_map = schedule.get_jik_2_start_time_map()
    return {
        (job_id, stage_id, mc_id): js_color[(job_id, stage_id)]
        for (job_id, stage_id, mc_id) in start_map
        if (job_id, stage_id) in js_color
    }


def compute_lane_boundaries(
    schedule: HybridFlowshopLiteSchedule,
    stage_2_partition: dict[str, OperationPartition],
    stage_list: list[str],
    machines_per_stage,
    *,
    force_start: int,
    force_end: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Per-machine (stage, mc) -> (left_b, right_b) time-fixed boundaries.

    Mirrors ffc_dw_wET_2026's pw_cp/visual.py: on each machine the left boundary
    is the rightmost end among that machine's LTF ops (default ``force_start``
    when it has none), and the right boundary is the leftmost start among its RTF
    ops (default ``force_end``). The render layer paints the LTF / active / RTF
    background bands and dashed boundary segments from this map.
    """
    start_map = schedule.get_jik_2_start_time_map()
    end_map = schedule.get_jik_2_end_time_map()
    lane_boundaries: dict[tuple[str, str], tuple[float, float]] = {}
    for stage_id in stage_list:
        partition = stage_2_partition.get(stage_id)
        for mc_id in machines_per_stage.get(stage_id, []):
            ltf_ends: list[int] = []
            rtf_starts: list[int] = []
            if partition is not None:
                for job_id, k in partition.left_time_fixed:
                    if k == mc_id:
                        end_t = end_map.get((job_id, stage_id, k))
                        if end_t is not None:
                            ltf_ends.append(end_t)
                for job_id, k in partition.right_time_fixed:
                    if k == mc_id:
                        start_t = start_map.get((job_id, stage_id, k))
                        if start_t is not None:
                            rtf_starts.append(start_t)
            left_b = max(ltf_ends) if ltf_ends else float(force_start)
            right_b = min(rtf_starts) if rtf_starts else float(force_end)
            lane_boundaries[(stage_id, mc_id)] = (float(left_b), float(right_b))
    return lane_boundaries


def unfixed_op_set(
    stage_2_partition: dict[str, OperationPartition],
) -> set[tuple[str, str]]:
    """(job, stage) keys of the UNFIXED ops, used as the panel highlight set so
    the active window's bars get a thick black border."""
    return {
        (job_id, stage_id)
        for stage_id, partition in stage_2_partition.items()
        for job_id, _ in partition.unfixed
    }


def solve_one_window(
    constructor: PwCpConstructor,
    instance: HybridFlowshopParameters,
    stage_2_job_2_p_dict,
    incumbent: HybridFlowshopLiteSchedule,
    stage_2_partition: dict[str, OperationPartition],
    *,
    batch_idx: int,
) -> tuple[HybridFlowshopLiteSchedule, HybridFlowshopLiteSchedule | None]:
    """Run a single window subproblem (non-invasive replay of .run's body).

    Returns (right_justified_before, candidate_after). The "before" schedule is
    spec.init_schedule, i.e. the incumbent after the backward right-justify pass
    that _build_batch_spec performs internally.

    The "after" is the RAW CP subproblem solution, taken straight from
    ``create_pw_cp_schedule`` BEFORE production's ``make_semi_active`` left-shift
    (pw_cp.py:1021). This is deliberate for the figure: in the raw CP solution
    the time-fixed ops (LTF/RTF) sit at EXACTLY their before/right-justified
    positions -- they are constants of the subproblem, not decision variables --
    so the panel honestly shows only the UNFIXED window being rearranged.
    Production's later semi-active left-shift (which makes RTF appear to move
    during the solve) is a separate post-pass and is intentionally omitted here.
    Falls back to the production helper (with semi-active) only when a window has
    no right-time-fixed ops, where the distinction is moot.
    """
    # The solve helpers and _build_batch_spec require an active run state.
    sub_obj_store: ObjValueBoundStore[int] = ObjValueBoundStore[int]()
    constructor._st = PwCpRunState(
        timer=ElapsedTimer(),
        incumbent=incumbent,
        sub_obj_store=sub_obj_store,
        subproblem_idx=0,
        subproblem_logs=[],
        max_time_per_batch=None,
    )
    try:
        spec = constructor._build_batch_spec(
            incumbent=incumbent,
            stage_2_partition=stage_2_partition,
            stage_2_job_2_p_dict=stage_2_job_2_p_dict,
            batch_idx=batch_idx,
        )
        before = spec.init_schedule

        if spec.is_right_time_fixed_empty:
            # No RTF ops: the semi-active distinction does not matter, so reuse
            # the production helper as-is.
            candidate = constructor._solve_makespan_batch(
                spec=spec,
                instance=instance,
                stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                profile_fix_by_machine=False,
                machine_precedence_stride=1,
                stage_precedence_min_processing_time_diff=None,
                stage_precedence_min_processing_time_diff_ratio=None,
                max_time_per_batch=None,
                solver_thread_cnt=THREADS,
                use_lns_only=False,
                tighten_ranges=False,
                debug_export=False,
            )
            return before, candidate

        # RTF present: replay _solve_batch_pw_cp_model's body, but stop at the
        # raw CP schedule (skip make_semi_active) so LTF/RTF stay pinned.
        mdl, params, variables = constructor._prepare_pw_cp_model(
            spec=spec,
            instance=instance,
            profile_fix_by_machine=False,
            machine_precedence_stride=1,
            stage_precedence_min_processing_time_diff=None,
            stage_precedence_min_processing_time_diff_ratio=None,
            tighten_ranges=False,
        )
        timelimit = constructor.ctx.get_remaining_time_limit(None)
        report = constructor.ctx.solve_cp_model_2(
            mdl,
            timelimit,
            THREADS,
            e_timer=ElapsedTimer(),
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            use_lns_only=False,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
            log_search_progress=False,
            last_timestamp_note=f"batch={spec.batch_idx + 1}",
        )
        if not getattr(report, "is_feasible", False):
            return before, None
        candidate = create_pw_cp_schedule(
            constructor.ctx.solver,
            params,
            spec.stage_2_partition,
            spec.init_schedule,
            variables,
        )
        return before, candidate
    finally:
        constructor._st = None


def region_legend_text() -> str:
    return " | ".join(
        f"{REGION_LABEL[name]}={color}" for name, color in REGION_COLORS.items()
    )


def render_isw_panel(
    out_path,
    schedule: HybridFlowshopLiteSchedule,
    partition: dict[str, OperationPartition],
    *,
    all_jobs,
    all_stages,
    machines_map,
    force_start: int,
    force_end: int,
) -> None:
    """Render one ISW-CP partition panel in the sister-repo 5-region style.

    Bars are region-colored (resolved from the schedule's ACTUAL ops, so a
    window-solve machine reassignment never drops an op back to its by-job
    color); the UNFIXED ops get a thick black border (via ``highlight_op_set``);
    each lane carries its own LTF/active/RTF background bands and dashed
    time-fixed boundaries; a region legend is drawn. Boundaries are computed from
    ``schedule`` so before/after panels stay self-consistent.
    """
    render_panel(
        out_path,
        schedule.get_jik_2_start_time_map(),
        schedule.get_jik_2_end_time_map(),
        all_job_list=all_jobs,
        force_start=force_start,
        force_end=force_end,
        show_labels=False,
        stage_list=all_stages,
        machine_list_per_stage=machines_map,
        op_color_map=schedule_region_color_map(schedule, partition),
        highlight_op_set=unfixed_op_set(partition),
        lane_boundaries=compute_lane_boundaries(
            schedule,
            partition,
            list(all_stages),
            machines_map,
            force_start=force_start,
            force_end=force_end,
        ),
        band_region_colors=BAND_REGION_COLORS,
        region_legend=REGION_LEGEND,
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    instance = load_instance()
    all_jobs = instance.job_id_list
    all_stages = instance.stage_id_list
    machines_map = instance.stage_2_machines_map

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    controller = build_initialized_controller(instance)
    stage_2_job_2_p_dict = controller.stage_2_job_2_p_dict
    incumbent = controller.solution_manager.get_incumbent()
    constructor = PwCpConstructor(controller)

    # Shared axis for the non-enlargement panels (steps 1-4). Bounded by the
    # incumbent makespan (right-justify preserves makespan; the window solve can
    # only shrink it). The enlargement panel (step_05) uses its own axis.
    incumbent_makespan = int(incumbent.makespan)
    force_start = 0
    force_end = incumbent_makespan

    captions: list[tuple[str, str]] = []

    # --- step_01: 5-region partition at window position w (narrow U0) ---
    part_w = build_stage_partitions(
        constructor,
        incumbent,
        unfixed_batch_start_idx=WINDOW_START,
        unfixed_batch_count=UNFIXED_COUNT,
        left_profile_fixed_batch_count=LEFT_PF,
        right_profile_fixed_batch_count=RIGHT_PF,
    )
    render_isw_panel(
        OUT_DIR / "step_01_partition.svg",
        incumbent,
        part_w,
        all_jobs=all_jobs,
        all_stages=all_stages,
        machines_map=machines_map,
        force_start=force_start,
        force_end=force_end,
    )
    captions.append(
        (
            "step_01_partition.svg",
            f"5-region partition at window position w={WINDOW_START} "
            f"(U={UNFIXED_COUNT}, LPF={LEFT_PF}, RPF={RIGHT_PF}). Ops on each "
            f"stage are time-ordered into batches; colors = {region_legend_text()}. "
            "Each lane shows its own LTF/active/RTF bands; dashed segments mark "
            "the per-machine time-fixed boundaries; the UNFIXED bars are "
            "black-bordered.",
        )
    )

    # --- step_02: right-justify (backward ALAP on non-LTF ops): before/after ---
    # Before: the incumbent. After: a copy right-justified on the non-LTF ops,
    # exactly as PwCpConstructor._build_batch_spec does internally.
    rj_schedule = incumbent.deepcopy()
    non_ltf_op_set: set[tuple[str, str, str]] = set()
    for stage_id, partition in part_w.items():
        for job_id, mc_id in partition.non_left_time_fixed:
            non_ltf_op_set.add((job_id, stage_id, mc_id))
    rj_schedule.make_right_justified(stage_2_job_2_p_dict, operation_set=non_ltf_op_set)
    render_isw_panel(
        OUT_DIR / "step_02a_right_justify_before.svg",
        incumbent,
        part_w,
        all_jobs=all_jobs,
        all_stages=all_stages,
        machines_map=machines_map,
        force_start=force_start,
        force_end=force_end,
    )
    render_isw_panel(
        OUT_DIR / "step_02b_right_justify_after.svg",
        rj_schedule,
        part_w,
        all_jobs=all_jobs,
        all_stages=all_stages,
        machines_map=machines_map,
        force_start=force_start,
        force_end=force_end,
    )
    captions.append(
        (
            "step_02a_right_justify_before.svg",
            f"Right-justify (before): incumbent at w={WINDOW_START}, "
            f"makespan={incumbent_makespan}.",
        )
    )
    captions.append(
        (
            "step_02b_right_justify_after.svg",
            "Right-justify (after): backward ALAP pass shifts the non-LTF ops "
            "(LPF/UNFIXED/RPF/RTF) as far right as possible, opening slack to "
            "the left of the UNFIXED window. Makespan is preserved "
            f"({int(rj_schedule.makespan)}); LTF anchors are untouched.",
        )
    )

    # --- step_03: window subproblem solve: before -> after ---
    before, candidate = solve_one_window(
        constructor,
        instance,
        stage_2_job_2_p_dict,
        incumbent.deepcopy(),
        part_w,
        batch_idx=WINDOW_START,
    )
    render_isw_panel(
        OUT_DIR / "step_03a_window_solve_before.svg",
        before,
        part_w,
        all_jobs=all_jobs,
        all_stages=all_stages,
        machines_map=machines_map,
        force_start=force_start,
        force_end=force_end,
    )
    after_schedule = candidate if candidate is not None else before
    # The candidate may reassign active-region ops to the other machine in their
    # stage; render_isw_panel colors from after_schedule's ACTUAL ops via the
    # (job, stage) region lookup, so every bar keeps its region color.
    render_isw_panel(
        OUT_DIR / "step_03b_window_solve_after.svg",
        after_schedule,
        part_w,
        all_jobs=all_jobs,
        all_stages=all_stages,
        machines_map=machines_map,
        force_start=force_start,
        force_end=force_end,
    )
    if candidate is not None:
        solve_note = (
            "Window subproblem solve (after, raw CP solution): the UNFIXED ops "
            "are re-optimised (LPF/RPF keep within-stage order & may shift) while "
            "the time-fixed LTF/RTF ops stay at EXACTLY their before positions -- "
            "they are constants of the subproblem, so every grey/blue-grey bar is "
            "unchanged from the before panel. Production then applies a separate "
            "make_semi_active left-shift (omitted here so 'time-fixed' reads "
            f"honestly). Candidate makespan={int(candidate.makespan)} "
            f"(before={int(before.makespan)})."
        )
    else:
        solve_note = (
            "Window subproblem solve (after): CP found no improving candidate "
            f"at w={WINDOW_START}, so the right-justified incumbent "
            f"(makespan={int(before.makespan)}) is retained; the UNFIXED window "
            "is shown re-evaluated in place."
        )
    captions.append(
        (
            "step_03a_window_solve_before.svg",
            f"Window subproblem solve (before): right-justified incumbent at "
            f"w={WINDOW_START}, makespan={int(before.makespan)}. UNFIXED window "
            "is the CP search space.",
        )
    )
    captions.append(("step_03b_window_solve_after.svg", solve_note))

    # --- step_04: slide -> 3 consecutive window positions (partition-colored) ---
    slide_positions = [
        WINDOW_START,
        WINDOW_START + STEP_SIZE,
        WINDOW_START + 2 * STEP_SIZE,
    ]
    max_window_start = len(all_jobs) - UNFIXED_COUNT  # batches per stage == n_jobs
    slide_positions = [p for p in slide_positions if p <= max_window_start]
    for slot, w in enumerate(slide_positions, start=1):
        part = build_stage_partitions(
            constructor,
            incumbent,
            unfixed_batch_start_idx=w,
            unfixed_batch_count=UNFIXED_COUNT,
            left_profile_fixed_batch_count=LEFT_PF,
            right_profile_fixed_batch_count=RIGHT_PF,
        )
        out_name = f"step_04_slide_{slot}_w{w}.svg"
        render_isw_panel(
            OUT_DIR / out_name,
            incumbent,
            part,
            all_jobs=all_jobs,
            all_stages=all_stages,
            machines_map=machines_map,
            force_start=force_start,
            force_end=force_end,
        )
        captions.append(
            (
                out_name,
                f"Slide {slot}/{len(slide_positions)}: window at w={w} "
                f"(advances by Delta=step_size={STEP_SIZE} each pass). The "
                "UNFIXED core (black-bordered) and the per-lane boundaries move "
                "right while the partition structure is preserved.",
            )
        )

    # --- step_05: incremental enlargement: U0=2 (narrow) vs U_max=8 (wide) ---
    for tag, u in (("narrow", UNFIXED_COUNT), ("wide", UNFIXED_COUNT_WIDE)):
        # Center the enlarged window near the timeline middle so the width
        # contrast is visible. Clamp the start so the window fits.
        n_batches = len(all_jobs)
        start_idx = max(0, min(WINDOW_START, n_batches - u))
        part = build_stage_partitions(
            constructor,
            incumbent,
            unfixed_batch_start_idx=start_idx,
            unfixed_batch_count=u,
            left_profile_fixed_batch_count=LEFT_PF if tag == "narrow" else 0,
            right_profile_fixed_batch_count=RIGHT_PF if tag == "narrow" else 0,
        )
        out_name = f"step_05_enlarge_{tag}_U{u}.svg"
        render_isw_panel(
            OUT_DIR / out_name,
            incumbent,
            part,
            all_jobs=all_jobs,
            all_stages=all_stages,
            machines_map=machines_map,
            force_start=force_start,
            force_end=force_end,
        )
        captions.append(
            (
                out_name,
                f"Incremental enlargement ({tag}): UNFIXED width U={u} "
                f"(U0={UNFIXED_COUNT} -> U_max={UNFIXED_COUNT_WIDE}). When a "
                "pass stops improving, U grows so the next window re-optimises "
                "a larger block. The green black-bordered UNFIXED core widens "
                "between the panels.",
            )
        )

    # --- panels.md: panel <-> caption mapping for the slide author ---
    panels_md = [
        "# ISW-CP panels (incremental sliding-window CP)",
        "",
        "Demo instance: 10 jobs x 4 stages, 2 machines/stage "
        "(`resources/demo_p1_10x4/1.txt`).",
        "",
        f"- Incumbent (initialisation) makespan = {incumbent_makespan}.",
        f"- Window config: batch_size={BATCH_SIZE}, step_size={STEP_SIZE}, "
        f"U0={UNFIXED_COUNT}, U_max={UNFIXED_COUNT_WIDE}, "
        f"LPF={LEFT_PF}, RPF={RIGHT_PF}.",
        "",
        "## Region colors (5-region partition)",
    ]
    for name, color in REGION_COLORS.items():
        panels_md.append(f"- {REGION_LABEL[name]} ({name}): `{color}`")
    panels_md += ["", "## Panels"]
    for name, cap in captions:
        panels_md.append(f"- `{name}`: {cap}")
    (OUT_DIR / "panels.md").write_text("\n".join(panels_md) + "\n")

    print(
        f"force_start={force_start} force_end={force_end} "
        f"incumbent_makespan={incumbent_makespan} "
        f"window_solve_after={int(after_schedule.makespan)}"
    )
    for name, cap in captions:
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    main()
