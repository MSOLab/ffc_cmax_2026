"""Generate the CP-LB (retained-stage relaxation lower bound) panels (plan §4.4).

Storyboard (one SVG per Gantt panel under
``analysis_outputs/20260612_p1_algo_explainer/cp_lb/``):

  step_01  retained-subset highlight, quantile mode  R^quant            Gantt
  step_02  retained-subset highlight, adaptive mode  R^adap             Gantt
  step_03  relaxation solution (retained stages only) = certified LB    Gantt
  step_04  hint-based dispatch: full feasible restore, LB vs incumbent  Gantt

CP-LB mechanism -- the REAL API (discovered, not the hallucinated recipe).
================================================================================
The earlier recipe referencing ``from lb_bucket.cp import
build_retained_stage_cp_model`` was only HALF right and a method named
``complete_from_retained_cp`` does NOT exist. The actual, working standalone
path is the controller method ``apply_retained_stage_cp_lb`` (mirrored from
``scripts/run_retained_cp_lb_sweep.py``), which internally calls
``build_retained_stage_cp_model``:

  hybridflowshop/controller/hfs_cp_lns.py
    - apply_retained_stage_cp_lb(...)                       (def @ line 9370)
        retained_stage_mode in {"first_middle_last" (quantile),
        "first_bottleneck_last" (workload-adaptive bottleneck), "first_last",
        "first_n_quantiles_last", "first_topk_bottlenecks_last", ...}.
        Builds + solves the retained-stage CP-SAT relaxation; certified LB is
        ``best_objective_bound``. Returns a payload dict and stores state on the
        controller.
    - ctrl.last_retained_cp_lb_result : RetainedStageCpResult
        (lb_bucket/cp/search.py @ line 22): .objective_lb / .certified_final_lb
        (the certified LB), .objective_ub, .retained_stage_ids,
        .bottleneck_stage_id, .selected_bottleneck_stage_ids, .status_name.
    - ctrl.last_retained_cp_lb_retained_solution_rows : list[dict]
        per retained (stage_id, job_id): start/end/processing_time/head/tail
        (lb_bucket/cp/solution_io.py::extract_retained_stage_solution_rows @ 16).
        The relaxation makespan over these rows == the certified LB.
    - dispatch_from_retained_cp(...)                        (def @ line 10155)
        Restores a FULL feasible 4-stage schedule from the retained-CP anchors
        and registers it. Read the restored schedule from
        ``ctrl.solution_manager.get_incumbent()`` and the makespan from
        ``ctrl.last_retained_cp_post_dispatch_obj``.

  lb_bucket/cp/model.py::build_retained_stage_cp_model       (def @ line 494)
  lb_bucket/cp/search.py::build_retained_stage_cp_result     (def @ line 105)
  hybridflowshop/lower_bounds.py                             analytic LBs (LB0,
        Santos LB2, Chen LB4) -- the SHD LB used as ``input_lb``; the
        retained-stage CP LB above strictly dominates / refines these.
  hybridflowshop/lb_enum.py                                  (LbModelType /
        AggregationType -- NOT the retained-subset selector; the selector is the
        ``retained_stage_mode`` string argument above).

Controller construction (verified against
scripts/run_retained_cp_lb_sweep.py::prepare_controller_for_instance and
hfs_single_instance_runner.py): shared_param_dict needs ``horizon``; a generous
``timelimit`` (1e9) lets the tiny 10x4 solve to OPTIMAL so the LB is certified
(plan §2.4 guard avoidance); the subroutine flow is an empty DynamicDataObject
because we drive the controller methods directly.

Two-mode contrast (plan §4.4 R^quant vs R^adap).
================================================================================
Both modes are exposed and both run. For THIS 4-stage demo the quantile-middle
stage and the strongest INTERNAL workload bottleneck both resolve to i2, so
R^quant == R^adap == {i0, i2, i3}. That coincidence is real (not fabricated):
the true workload bottleneck i3 (sum_p/m = 337.5) is the last stage, always
retained, so the internal bottleneck the adaptive selector reports is i2
(231.5), which is exactly the middle index of {0,1,2,3}. Both panels are still
rendered so the slide can show each selector's output side by side; the captions
state the coincidence explicitly.

Limitations / faithful approximations (documented in panels.md too):
- The retained-stage CP relaxation IGNORES per-stage machine capacity inside
  each retained stage (the dropped stages collapse to compressed lags). The
  relaxation rows therefore carry no machine assignment. For the relaxation
  Gantt (step_03) we synthesize lanes by greedy interval packing per retained
  stage, which makes the capacity-ignored overlaps visible as extra lanes
  (more than the instance's 2 machines/stage). The bar TIMES are the exact CP
  solution; only the lane (y) placement is synthetic.
- Greying of dropped stages is approximated via highlight (the renderer thickens
  highlighted op edges); the retained-subset panels highlight the retained ops
  and the caption names the dropped stage(s) and their compressed-lag role.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

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
from scripts.p1_algo_explainer.recorder import GanttSnapshotRecorder  # noqa: E402
from scripts.p1_algo_explainer.render import render_panel  # noqa: E402

INSTANCE_PATH = REPO_ROOT / "resources" / "demo_p1_10x4" / "1.txt"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "20260612_p1_algo_explainer" / "cp_lb"

SEED = 42
THREADS = 8
TL_NC_MULTIPLIER = 0.5  # 0.5 * n(10) * c(4) = 20 s budget -> tiny instance solves to OPTIMAL
# A generous controller-level time budget so the explainer never trips the
# pre-call time guards (plan §2.4) and the LB is certified.
STOPPING_TIME_LIMIT = 10**9

# Both selectors retain FIVE stages so the slide shows a richer subset.
# R^quant: first + 3 quartile-cut stages + last (thesis q=4 quantile rule).
QUANTILE_MODE = "first_n_quantiles_last"
QUANTILE_KW: dict[str, int] = {"quantile_count": 4}
# R^adap: first + top-3 workload bottlenecks + last.
ADAPTIVE_MODE = "first_topk_bottlenecks_last"
ADAPTIVE_KW: dict[str, int] = {"extra_bottleneck_count": 3}

# One representative job for the compressed-lag schematic (step_05). j2 has four
# clearly distinct stage durations (i0=48, i1=29, i2=62, i3=95) so the dropped
# stage i1 reads as a visibly sized lag without dominating the strip.
SCHEMATIC_JOB = "j2"

OpKey = tuple[str, str, str]  # (job, stage, machine)


def load_instance(
    instance_path: Path = INSTANCE_PATH, name: str = "demo_p1_10x4"
) -> HybridFlowshopParameters:
    with open(instance_path) as f:
        return HybridFlowshopParameters.from_ff2020_data(name, f)


def build_initialized_controller(
    instance: HybridFlowshopParameters,
) -> HybridFlowShopCpLnsController:
    """Construct + seed-initialise a controller (mirrors run_retained_cp_lb_sweep)."""
    controller = HybridFlowShopCpLnsController(
        instance,
        {"horizon": 100000},
        DynamicDataObject.from_obj([]),
        StoppingCriteria.from_dict({"timelimit": STOPPING_TIME_LIMIT}),
    )
    controller.set_working_dir(Path(tempfile.mkdtemp(prefix="cp_lb_explainer_")))
    controller.set_random_seed(seed=SEED)
    controller.apply_shdlb()
    controller.initialize_by_best_of_selected_dispatches(
        method_list=["bn2d_all_stages", "best_of_mixed_dispatches"],
        error_if_infeasible=False,
        draw_gantt=False,
    )
    return controller


def run_retained_cp(
    controller: HybridFlowShopCpLnsController, mode: str, **mode_kw: int
):
    """Run one retained-stage CP-LB pass; return (result, retained_solution_rows).

    ``mode_kw`` forwards the mode-specific selector knobs (e.g.
    ``quantile_count`` for the quantile rule, ``extra_bottleneck_count`` for the
    top-k bottleneck rule) straight to ``apply_retained_stage_cp_lb``.
    """
    controller.apply_retained_stage_cp_lb(
        threads=THREADS,
        tl_nc_multiplier=TL_NC_MULTIPLIER,
        retained_stage_mode=mode,
        save_cp_lb_artifacts=False,
        **mode_kw,
    )
    return (
        controller.last_retained_cp_lb_result,
        controller.last_retained_cp_lb_retained_solution_rows,
    )


def relaxation_maps_with_synthetic_lanes(
    retained_solution_rows: list[dict],
) -> tuple[dict[OpKey, int], dict[OpKey, int], dict[str, list[str]]]:
    """Build start/end maps for the retained relaxation with synthetic lanes.

    The relaxation ignores per-stage machine capacity, so the CP rows carry no
    machine. We greedily pack each stage's ops into lanes (a new lane only when
    an op overlaps every existing lane), surfacing capacity-ignored overlaps as
    extra lanes. Bar times are the exact CP solution; lane placement is synthetic.
    """
    start_map: dict[OpKey, int] = {}
    end_map: dict[OpKey, int] = {}
    machine_list_per_stage: dict[str, list[str]] = {}

    by_stage: dict[str, list[dict]] = {}
    for row in retained_solution_rows:
        by_stage.setdefault(str(row["stage_id"]), []).append(row)

    for stage_id, rows in by_stage.items():
        lane_free_at: list[int] = []  # lane index -> earliest free time
        ordered = sorted(rows, key=lambda r: (int(r["start"]), int(r["end"])))
        for row in ordered:
            start, end = int(row["start"]), int(row["end"])
            lane = next(
                (i for i, free in enumerate(lane_free_at) if free <= start),
                None,
            )
            if lane is None:
                lane = len(lane_free_at)
                lane_free_at.append(end)
            else:
                lane_free_at[lane] = end
            machine = f"{stage_id}_lane{lane}"
            key: OpKey = (str(row["job_id"]), stage_id, machine)
            start_map[key] = start
            end_map[key] = end
        machine_list_per_stage[stage_id] = [
            f"{stage_id}_lane{i}" for i in range(len(lane_free_at))
        ]

    return start_map, end_map, machine_list_per_stage


def stage_index_label(instance: HybridFlowshopParameters, stage_id: str) -> str:
    return f"{stage_id} (#{instance.stage_id_list.index(stage_id)})"


def render_compressed_lag_schematic(
    instance: HybridFlowshopParameters,
    retained_R: list[str],
    dropped: list[str],
    out_path: Path,
    job: str,
) -> str:
    """Draw the standalone compressed-lag schematic (step_05).

    Two stacked strips share ONE x-axis built from job ``job``'s real per-stage
    durations laid back-to-back (a job chain). The top strip shows every stage as
    a real operation; the bottom strip keeps the retained stages as solid boxes
    and replaces each maximal run of consecutive DROPPED stages -- the gap
    between two consecutive retained stages -- with a SINGLE hatched ``lag``
    block whose width is the summed dropped duration. One lag per gap mirrors the
    model exactly (one precedence delay per consecutive retained pair). Because
    the x layout is shared, each lag sits directly under the dropped run it
    collapses, so the eye reads ``sum p^{dropped} -> lag`` from the vertical
    alignment alone. This is the only panel that carries text (stage ids and the
    lag formula); it is a concept diagram, not instance meta.
    """
    stages = list(instance.stage_id_list)
    dropped_set = set(dropped)
    p = instance.job_2_stage_2_p_map[job]

    # Back-to-back chain: stage s occupies [x, x + p_s] on the shared axis.
    span: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for s in stages:
        span[s] = (cursor, cursor + float(p[s]))
        cursor += float(p[s])
    total = cursor

    # Gaps: maximal runs of consecutive dropped stages (each becomes one lag).
    gaps: list[list[str]] = []
    run: list[str] = []
    for s in stages:
        if s in dropped_set:
            run.append(s)
        elif run:
            gaps.append(run)
            run = []
    if run:
        gaps.append(run)

    retained_c = "#4C78A8"  # solid retained operation
    drop_fore_c = "#C9CDD2"  # dropped op (top strip, foreshadowed as removed)
    lag_face_c = "#ECEFF1"  # hatched lag fill (bottom strip)
    lag_edge_c = "#5A6470"
    label_min_frac = 0.030  # only label boxes at least this wide (avoid overlap)
    h = 0.7
    full_y, relax_y = 1.65, 0.30

    fig, ax = plt.subplots(figsize=(9.6, 3.4))

    def _box(x0: float, x1: float, y: float, **kw) -> None:
        ax.add_patch(mpatches.Rectangle((x0, y), x1 - x0, h, **kw))

    def _wide(x0: float, x1: float) -> bool:
        return (x1 - x0) >= label_min_frac * total

    # --- top strip: full job (every stage is a real operation) ---
    for s in stages:
        x0, x1 = span[s]
        is_drop = s in dropped_set
        _box(
            x0,
            x1,
            full_y,
            facecolor=drop_fore_c if is_drop else retained_c,
            edgecolor="black",
            linewidth=1.6,
            linestyle="--" if is_drop else "-",
            alpha=0.85 if is_drop else 1.0,
        )
        # Retained stages always labelled; dropped only when wide enough.
        if (not is_drop) or _wide(x0, x1):
            ax.text(
                (x0 + x1) / 2,
                full_y + h / 2,
                s,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black" if is_drop else "white",
            )

    # --- bottom strip: retained operations + one merged lag per gap ---
    for s in stages:
        if s in dropped_set:
            continue
        x0, x1 = span[s]
        _box(x0, x1, relax_y, facecolor=retained_c, edgecolor="black", linewidth=1.6)
        ax.text(
            (x0 + x1) / 2,
            relax_y + h / 2,
            s,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
    multi = len(gaps) > 1
    for k, gap_run in enumerate(gaps, start=1):
        gx0 = span[gap_run[0]][0]
        gx1 = span[gap_run[-1]][1]
        _box(
            gx0,
            gx1,
            relax_y,
            facecolor=lag_face_c,
            edgecolor=lag_edge_c,
            linewidth=1.4,
            hatch="////",
        )
        lag_label = rf"$\ell_{{{k}}}$" if multi else r"$\ell$"
        ax.text(
            (gx0 + gx1) / 2,
            relax_y + h / 2,
            lag_label,
            ha="center",
            va="center",
            fontsize=13,
            color=lag_edge_c,
            fontweight="bold",
        )
        # One "collapse" arrow per gap: dropped-run centre (top) -> lag (bottom).
        xc = (gx0 + gx1) / 2
        ax.annotate(
            "",
            xy=(xc, relax_y + h + 0.04),
            xytext=(xc, full_y - 0.04),
            arrowprops=dict(arrowstyle="-|>", color="#B00020", linewidth=2.0),
        )
        ax.text(
            xc,
            (full_y + relax_y + h) / 2,
            "collapse",
            ha="center",
            va="center",
            fontsize=8.5,
            style="italic",
            color="#B00020",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
        )

    # --- row labels (left of the chain) ---
    label_x = -total * 0.015
    ax.text(label_x, full_y + h / 2, "Full job", ha="right", va="center", fontsize=11)
    ax.text(
        label_x,
        relax_y + h / 2,
        "Retained\nrelaxation",
        ha="right",
        va="center",
        fontsize=11,
    )

    # --- lag formula under the bottom strip (generic; no per-term expansion) ---
    ax.text(
        total / 2,
        relax_y - 0.34,
        r"$\ell_{\,i^- i^+ j} = \sum_{i^- < i' < i^+} p_{i'j}$  "
        "(one precedence delay per consecutive retained pair "
        r"$i^-\!,\,i^+$)",
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )

    ax.set_xlim(-total * 0.20, total * 1.02)
    ax.set_ylim(-0.45, full_y + h + 0.30)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return (
        f"Compressed-lag schematic (job {job}): top strip = the full job with "
        f"every stage a real operation; bottom strip = the retained relaxation "
        f"where each maximal run of dropped stages collapses to a SINGLE hatched "
        f"compressed lag {chr(0x2113)} (= summed dropped-stage processing time) "
        f"between consecutive retained stages {retained_R}. {len(gaps)} lag(s) "
        f"for {len(dropped)} dropped stage(s). Shared x-axis: each lag sits under "
        f"the dropped run it replaces."
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        type=Path,
        default=INSTANCE_PATH,
        help="ff2020-format instance file (default: the 10x4 demo).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for the SVG panels and panels.md.",
    )
    parser.add_argument(
        "--schematic-job",
        default=SCHEMATIC_JOB,
        help="Representative job id for the step_05 compressed-lag schematic.",
    )
    cli = parser.parse_args()
    out_dir = cli.outdir
    schematic_job = cli.schematic_job

    instance = load_instance(cli.instance, cli.instance.parent.name)
    all_jobs = instance.job_id_list
    all_stages = instance.stage_id_list

    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = GanttSnapshotRecorder()

    # --- quantile pass (R^quant): retained-subset highlight on the incumbent ---
    controller = build_initialized_controller(instance)
    incumbent_before = controller.solution_manager.get_incumbent()
    input_ub = controller.solution_manager.best_obj_value
    input_lb = controller.solution_manager.best_obj_bound

    quant_result, _ = run_retained_cp(controller, QUANTILE_MODE, **QUANTILE_KW)
    quant_R = list(quant_result.retained_stage_ids)
    quant_dropped = [s for s in all_stages if s not in quant_R]

    # --- adaptive pass (R^adap): independent fresh controller ---
    controller_adap = build_initialized_controller(instance)
    adap_result, adap_rows = run_retained_cp(
        controller_adap, ADAPTIVE_MODE, **ADAPTIVE_KW
    )
    adap_R = list(adap_result.retained_stage_ids)
    adap_dropped = [s for s in all_stages if s not in adap_R]
    adap_bottleneck = adap_result.bottleneck_stage_id

    # Relaxation solution (step_03) uses the adaptive pass's rows (identical R
    # here, so the relaxation Gantt is the same for both modes).
    relax_start, relax_end, relax_machines = relaxation_maps_with_synthetic_lanes(
        list(adap_rows)
    )
    relax_makespan = max(relax_end.values())  # == certified LB
    certified_lb = adap_result.objective_lb

    # --- hint-based dispatch (step_04): full feasible 4-stage restore ---
    controller_adap.dispatch_from_retained_cp(
        save_cp_dispatch_artifacts=False,
        draw_gantt=False,
    )
    restored = controller_adap.solution_manager.get_incumbent()
    restored_makespan = restored.makespan

    # Shared axis for the full-schedule panels (incumbent + restored).
    force_start = 0
    force_end = max(int(input_ub), int(restored_makespan))

    # --- step_01: retained-subset highlight, quantile mode ---
    quant_highlight = {
        (job, stage) for stage in quant_R for job in all_jobs
    }
    recorder.record_schedule(
        "retained_subset_quantile",
        incumbent_before,
        highlight=quant_highlight,
        axis=(force_start, force_end),
        note=(
            f"Quantile mode ({QUANTILE_MODE}): retained R^quant = {quant_R} "
            f"(first + quantile-cut stages + last). Dropped stages "
            f"{quant_dropped} collapse to compressed lags between retained "
            f"stages. Highlighted = retained-stage ops on the incumbent "
            f"(makespan={int(input_ub)})."
        ),
    )

    # --- step_02: retained-subset highlight, adaptive mode ---
    rules_coincide = set(quant_R) == set(adap_R)
    coincide_note = (
        " NOTE: R^adap == R^quant on this instance (the quantile cuts and the "
        "workload bottlenecks happen to resolve to the same stages)."
        if rules_coincide
        else f" Here R^adap differs from R^quant ({quant_R})."
    )
    adap_highlight = {
        (job, stage) for stage in adap_R for job in all_jobs
    }
    recorder.record_schedule(
        "retained_subset_adaptive",
        incumbent_before,
        highlight=adap_highlight,
        axis=(force_start, force_end),
        note=(
            f"Workload-adaptive mode ({ADAPTIVE_MODE}): retained R^adap = "
            f"{adap_R}; top bottleneck = {adap_bottleneck} (strongest "
            f"sum(p)/m among non-terminal stages). Dropped stages "
            f"{adap_dropped} -> compressed lags.{coincide_note}"
        ),
    )

    # --- step_03: relaxation solution = certified LB (retained stages only) ---
    recorder.record(
        "relaxation_lower_bound",
        relax_start,
        relax_end,
        axis=(force_start, force_end),
        note=(
            f"Retained-stage CP relaxation (stages {adap_R} only; dropped "
            f"stages capacity-ignored and collapsed to lags). The relaxation "
            f"makespan = {relax_makespan} is the CERTIFIED lower bound "
            f"LB = {certified_lb:g} (CP-SAT status "
            f"{adap_result.status_name}). Lanes are synthetic interval-packing "
            f"(capacity ignored: overlaps -> extra lanes beyond 2 machines)."
        ),
    )

    # --- step_04: hint-based dispatch, full feasible restore, LB vs incumbent ---
    gap = restored_makespan - certified_lb
    recorder.record_schedule(
        "hint_based_dispatch_full",
        restored,
        axis=(force_start, force_end),
        note=(
            f"Hint-based dispatch: retained-CP anchors restore a full feasible "
            f"4-stage schedule, makespan = {int(restored_makespan)}. "
            f"Lower bound LB = {certified_lb:g} vs restored makespan "
            f"{int(restored_makespan)} -> optimality gap = {gap:g}. "
            f"(Analytic SHD LB input was {int(input_lb)}; retained-CP LB "
            f"refines it to {certified_lb:g}.)"
        ),
    )

    # --- render panels ---
    captions: list[tuple[str, str]] = []
    for idx, snap in enumerate(recorder.snapshots, start=1):
        out_name = f"step_{idx:02d}_{snap.label}.svg"
        # step_03 keeps ALL stages on the y-axis so the dropped stages read as
        # empty lanes (the slide shows the full 15-stage axis, not just the
        # retained subset). Retained stages use the synthetic relaxation lanes;
        # dropped stages fall back to their real machine lanes, which carry no
        # bars and therefore render as blank rows.
        if snap.label == "relaxation_lower_bound":
            stage_list = all_stages
            machine_list_per_stage = {
                stage: relax_machines.get(stage)
                or instance.stage_2_machines_map[stage]
                for stage in all_stages
            }
        else:
            stage_list = all_stages
            machine_list_per_stage = instance.stage_2_machines_map
        render_panel(
            out_dir / out_name,
            snap.start_map,
            snap.end_map,
            all_job_list=all_jobs,
            highlight_op_set=snap.highlight,
            force_start=force_start,
            force_end=force_end,
            show_labels=False,
            stage_list=stage_list,
            machine_list_per_stage=machine_list_per_stage,
        )
        captions.append((out_name, snap.note or ""))

    # --- step_05: standalone compressed-lag schematic (one job, before/after) ---
    schematic_name = "step_05_compressed_lag_schematic.svg"
    schematic_caption = render_compressed_lag_schematic(
        instance,
        adap_R,
        adap_dropped,
        out_dir / schematic_name,
        schematic_job,
    )
    captions.append((schematic_name, schematic_caption))

    # --- panels.md: panel <-> caption mapping for the slide author ---
    rules_md = (
        "- R^quant == R^adap == "
        f"{adap_R} on this instance: the quantile-cut stages and the strongest "
        "internal workload bottlenecks happen to resolve to the same stages. "
        "Both selector outputs are real and rendered separately; they simply "
        "coincide here."
        if rules_coincide
        else f"- R^quant = {quant_R} and R^adap = {adap_R} differ on this "
        "instance: the quantile rule spreads cuts evenly over the stage axis "
        "while the adaptive rule pulls retention toward the incumbent's "
        "workload bottlenecks."
    )
    panels_md = [
        "# CP-LB panels (retained-stage relaxation lower bound)",
        "",
        f"Demo instance: {instance.job_count} jobs x {instance.stage_count} "
        f"stages (`{cli.instance}`).",
        "",
        f"- Analytic SHD input LB = {int(input_lb)}; incumbent UB = "
        f"{int(input_ub)}.",
        f"- Quantile mode `{QUANTILE_MODE}`: R^quant = {quant_R}, "
        f"LB = {quant_result.objective_lb:g} ({quant_result.status_name}).",
        f"- Adaptive mode `{ADAPTIVE_MODE}`: R^adap = {adap_R}, "
        f"top bottleneck = {adap_bottleneck}, "
        f"LB = {adap_result.objective_lb:g} ({adap_result.status_name}).",
        f"- Restored full-schedule makespan = {int(restored_makespan)}; "
        f"optimality gap to LB = {gap:g}.",
        "",
        "## Known limitations (faithful approximations)",
        rules_md,
        "- The relaxation Gantt (step_03) ignores per-stage machine capacity; "
        "lane (y) placement is synthetic greedy interval packing so overlaps "
        "are visible. Bar times are the exact CP solution and the makespan "
        "equals the certified LB.",
        "- Dropped-stage greying is approximated by highlighting the retained "
        "ops; captions name the dropped stages and their compressed-lag role.",
        "",
        "## Panels",
    ]
    for name, cap in captions:
        panels_md.append(f"- `{name}`: {cap}")
    (out_dir / "panels.md").write_text("\n".join(panels_md) + "\n")

    print(
        f"force_start={force_start} force_end={force_end} "
        f"input_lb={int(input_lb)} input_ub={int(input_ub)} "
        f"LB={certified_lb:g} restored={int(restored_makespan)} gap={gap:g}"
    )
    for name, cap in captions:
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    main()
