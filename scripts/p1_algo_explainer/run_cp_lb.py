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

QUANTILE_MODE = "first_middle_last"  # R^quant: first + quantile-middle + last
ADAPTIVE_MODE = "first_bottleneck_last"  # R^adap: first + internal bottleneck + last

OpKey = tuple[str, str, str]  # (job, stage, machine)


def load_instance() -> HybridFlowshopParameters:
    with open(INSTANCE_PATH) as f:
        return HybridFlowshopParameters.from_ff2020_data("demo_p1_10x4", f)


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
    controller: HybridFlowShopCpLnsController, mode: str
):
    """Run one retained-stage CP-LB pass; return (result, retained_solution_rows)."""
    controller.apply_retained_stage_cp_lb(
        threads=THREADS,
        tl_nc_multiplier=TL_NC_MULTIPLIER,
        retained_stage_mode=mode,
        save_cp_lb_artifacts=False,
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


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    instance = load_instance()
    all_jobs = instance.job_id_list
    all_stages = instance.stage_id_list

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recorder = GanttSnapshotRecorder()

    # --- quantile pass (R^quant): retained-subset highlight on the incumbent ---
    controller = build_initialized_controller(instance)
    incumbent_before = controller.solution_manager.get_incumbent()
    input_ub = controller.solution_manager.best_obj_value
    input_lb = controller.solution_manager.best_obj_bound

    quant_result, _ = run_retained_cp(controller, QUANTILE_MODE)
    quant_R = list(quant_result.retained_stage_ids)
    quant_dropped = [s for s in all_stages if s not in quant_R]

    # --- adaptive pass (R^adap): independent fresh controller ---
    controller_adap = build_initialized_controller(instance)
    adap_result, adap_rows = run_retained_cp(controller_adap, ADAPTIVE_MODE)
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
            f"(first + quantile-middle + last). Dropped stages "
            f"{quant_dropped} collapse to compressed lags between retained "
            f"stages. Highlighted = retained-stage ops on the incumbent "
            f"(makespan={int(input_ub)})."
        ),
    )

    # --- step_02: retained-subset highlight, adaptive mode ---
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
            f"{adap_R}; internal bottleneck = {adap_bottleneck} (strongest "
            f"sum(p)/m among non-terminal stages). Dropped stages "
            f"{adap_dropped} -> compressed lags. NOTE: for this 4-stage demo "
            f"R^adap == R^quant because the quantile-middle stage and the "
            f"internal bottleneck both resolve to {adap_bottleneck}."
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
        # step_03 uses the synthetic relaxation lanes; others use real machines.
        if snap.label == "relaxation_lower_bound":
            stage_list = adap_R
            machine_list_per_stage = relax_machines
        else:
            stage_list = all_stages
            machine_list_per_stage = instance.stage_2_machines_map
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
            machine_list_per_stage=machine_list_per_stage,
        )
        captions.append((out_name, snap.note or ""))

    # --- panels.md: panel <-> caption mapping for the slide author ---
    panels_md = [
        "# CP-LB panels (retained-stage relaxation lower bound)",
        "",
        "Demo instance: 10 jobs x 4 stages, 2 machines/stage "
        "(`resources/demo_p1_10x4/1.txt`).",
        "",
        f"- Analytic SHD input LB = {int(input_lb)}; incumbent UB = "
        f"{int(input_ub)}.",
        f"- Quantile mode `{QUANTILE_MODE}`: R^quant = {quant_R}, "
        f"LB = {quant_result.objective_lb:g} ({quant_result.status_name}).",
        f"- Adaptive mode `{ADAPTIVE_MODE}`: R^adap = {adap_R}, "
        f"internal bottleneck = {adap_bottleneck}, "
        f"LB = {adap_result.objective_lb:g} ({adap_result.status_name}).",
        f"- Restored full-schedule makespan = {int(restored_makespan)}; "
        f"optimality gap to LB = {gap:g}.",
        "",
        "## Known limitations (faithful approximations)",
        "- R^quant == R^adap == "
        f"{adap_R} for this 4-stage demo: the quantile-middle stage and the "
        f"strongest internal workload bottleneck both resolve to "
        f"{adap_bottleneck} (the true workload bottleneck i3 is the always-"
        "retained last stage). Both selector outputs are real and rendered "
        "separately; they simply coincide on this instance.",
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
    (OUT_DIR / "panels.md").write_text("\n".join(panels_md) + "\n")

    print(
        f"force_start={force_start} force_end={force_end} "
        f"input_lb={int(input_lb)} input_ub={int(input_ub)} "
        f"LB={certified_lb:g} restored={int(restored_makespan)} gap={gap:g}"
    )
    for name, cap in captions:
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    main()
