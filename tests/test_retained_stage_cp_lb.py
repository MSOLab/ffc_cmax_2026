from pathlib import Path

import pandas as pd
from mbls.cpsat import CpsatStatus
from ortools.sat.python import cp_model
from schore.parameters import JobStageProcessingTimeManager
from schore.parameters_examples import HybridFlowshopParameters

from lb_bucket.cp import (
    build_retained_stage_cp_model,
    build_retained_stage_cp_result,
    build_trace_rows,
    extract_retained_stage_solution_rows,
    select_bottleneck_stage_by_average_load,
    select_bottleneck_stage_ids_by_average_load,
    write_retained_stage_cp_artifacts,
)
from hybridflowshop.controller.hfs_cp_lns import _extract_retained_cp_trace_records


def _make_instance() -> HybridFlowshopParameters:
    job_ids = ["j1", "j2", "j3"]
    stage_ids = ["i0", "i1", "i2"]
    stage_2_machines_map = {
        "i0": ["m0", "m1"],
        "i1": ["m0"],
        "i2": ["m0", "m1"],
    }
    df = pd.DataFrame([[2, 8, 2], [2, 7, 2], [2, 9, 2]])
    p_manager = JobStageProcessingTimeManager(name="tiny_cp_lb", df=df)
    return HybridFlowshopParameters(
        name="tiny_cp_lb",
        job_id_list=job_ids,
        stage_id_list=stage_ids,
        stage_2_machines_map=stage_2_machines_map,
        p_manager=p_manager,
    )


def test_select_bottleneck_stage_by_average_load_uses_sum_p_over_m() -> None:
    instance = _make_instance()
    build = build_retained_stage_cp_model(
        instance,
        input_ub=40,
        retained_stage_mode="first_last",
    )

    assert select_bottleneck_stage_by_average_load(build.params) == "i1"


def test_select_bottleneck_stage_ids_by_average_load_returns_ranked_stage_ids() -> None:
    instance = _make_instance()
    build = build_retained_stage_cp_model(
        instance,
        input_ub=40,
        retained_stage_mode="first_last",
    )

    assert select_bottleneck_stage_ids_by_average_load(build.params, count=2) == [
        "i1",
        "i0",
    ]


def test_extract_retained_cp_trace_records_keeps_objective_ub_and_lb() -> None:
    trace_rows = [
        {"runtime_sec": 0.05, "objective_ub": None, "objective_lb": 20.0},
        {"runtime_sec": 0.10, "objective_ub": 30.0, "objective_lb": 20.0},
        {"runtime_sec": "bad", "objective_ub": 29.0, "objective_lb": 21.0},
        {"runtime_sec": 0.20, "objective_ub": 28.0, "objective_lb": None},
    ]

    assert _extract_retained_cp_trace_records(trace_rows, "objective_ub") == [
        (0.10, 30.0),
        (0.20, 28.0),
    ]
    assert _extract_retained_cp_trace_records(trace_rows, "objective_lb") == [
        (0.05, 20.0),
        (0.10, 20.0),
    ]


def test_retained_stage_cp_modes_solve_and_bottleneck_mode_is_not_weaker() -> None:
    instance = _make_instance()
    first_last = build_retained_stage_cp_model(
        instance,
        input_ub=40,
        retained_stage_mode="first_last",
    )
    first_bottleneck_last = build_retained_stage_cp_model(
        instance,
        input_ub=40,
        retained_stage_mode="first_bottleneck_last",
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    first_last_status = solver.Solve(first_last.model)
    assert first_last_status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    first_last_obj = solver.ObjectiveValue()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    first_bottleneck_last_status = solver.Solve(first_bottleneck_last.model)
    assert first_bottleneck_last_status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    first_bottleneck_last_obj = solver.ObjectiveValue()

    assert first_last.retained_stage_ids == ["i0", "i2"]
    assert first_bottleneck_last.retained_stage_ids == ["i0", "i1", "i2"]
    assert first_bottleneck_last.bottleneck_stage_id == "i1"
    assert first_bottleneck_last_obj >= first_last_obj


def test_retained_stage_cp_middle_quantiles_and_topk_modes_resolve_expected_stages() -> (
    None
):
    instance = HybridFlowshopParameters(
        name="selector_demo",
        job_id_list=["j1", "j2"],
        stage_id_list=["i0", "i1", "i2", "i3", "i4", "i5", "i6"],
        stage_2_machines_map={f"i{k}": ["m0"] for k in range(7)},
        p_manager=JobStageProcessingTimeManager(
            name="selector_demo",
            df=pd.DataFrame(
                [
                    [2, 20, 3, 4, 30, 5, 6],
                    [2, 21, 3, 4, 31, 5, 6],
                ]
            ),
        ),
    )

    middle = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_middle_last",
    )
    thirds = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_n_quantiles_last",
        quantile_count=3,
    )
    top2 = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_topk_bottlenecks_last",
        extra_bottleneck_count=2,
    )
    bottleneck_band = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_bottleneck_band_last",
        bottleneck_band_radius=1,
    )
    middle_band = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_middle_band_last",
        middle_band_radius=1,
    )
    explicit_ratios = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_ratio_points_last",
        retained_stage_ratios=[0.25, 0.75],
    )
    bottleneck_midpoints = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_bottleneck_midpoints_last",
    )
    processing_jump_band = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_processing_jump_band_last",
        bottleneck_band_radius=1,
    )
    middle_only_band = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="middle_band",
        middle_band_radius=1,
    )

    assert middle.retained_stage_ids == ["i0", "i3", "i6"]
    assert middle.retained_stage_ratios == [0.5]
    assert thirds.retained_stage_ids == ["i0", "i2", "i4", "i6"]
    assert thirds.quantile_count == 3
    assert top2.retained_stage_ids == ["i0", "i1", "i4", "i6"]
    assert top2.selected_bottleneck_stage_ids == ["i1", "i4"]
    assert bottleneck_band.retained_stage_ids == ["i0", "i3", "i4", "i5", "i6"]
    assert bottleneck_band.bottleneck_stage_id == "i4"
    assert bottleneck_band.bottleneck_band_radius == 1
    assert middle_band.retained_stage_ids == ["i0", "i2", "i3", "i4", "i6"]
    assert middle_band.middle_band_radius == 1
    assert middle_band.retained_stage_ratios == [0.5]
    assert explicit_ratios.retained_stage_ids == ["i0", "i2", "i4", "i6"]
    assert bottleneck_midpoints.retained_stage_ids == ["i0", "i2", "i4", "i5", "i6"]
    assert bottleneck_midpoints.bottleneck_stage_id == "i4"
    assert processing_jump_band.retained_stage_ids == ["i0", "i3", "i4", "i5", "i6"]
    assert processing_jump_band.bottleneck_stage_id == "i4"
    assert processing_jump_band.bottleneck_band_radius == 1
    assert middle_only_band.retained_stage_ids == ["i2", "i3", "i4"]
    assert middle_only_band.middle_band_radius == 1
    assert middle_only_band.retained_stage_ratios == [0.5]


def test_processing_jump_band_prefers_internal_stage_when_edge_jump_is_largest() -> None:
    instance = HybridFlowshopParameters(
        name="edge_jump_demo",
        job_id_list=["j1"],
        stage_id_list=["i0", "i1", "i2", "i3", "i4"],
        stage_2_machines_map={f"i{k}": ["m0"] for k in range(5)},
        p_manager=JobStageProcessingTimeManager(
            name="edge_jump_demo",
            df=pd.DataFrame([[100, 1, 2, 3, 4]]),
        ),
    )

    processing_jump_band = build_retained_stage_cp_model(
        instance,
        input_ub=200,
        retained_stage_mode="first_processing_jump_band_last",
        bottleneck_band_radius=1,
    )

    assert processing_jump_band.retained_stage_ids == ["i0", "i1", "i2", "i4"]
    assert processing_jump_band.bottleneck_stage_id == "i1"


def test_write_retained_stage_cp_artifacts_writes_expected_files(
    tmp_path: Path,
) -> None:
    instance = _make_instance()
    build = build_retained_stage_cp_model(
        instance,
        input_ub=40,
        retained_stage_mode="first_bottleneck_last",
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    status = solver.Solve(build.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    retained_rows = extract_retained_stage_solution_rows(solver, build)
    trace_rows = build_trace_rows(
        [(0.1, 26.0)],
        [(0.05, 20.0), (0.1, 26.0)],
        final_runtime_sec=0.1,
        final_objective_ub=26.0,
        final_objective_lb=26.0,
    )
    result = build_retained_stage_cp_result(
        ins_name=instance.name,
        input_lb=18,
        input_ub=40,
        retained_stage_mode="first_bottleneck_last",
        retained_stage_ids=build.retained_stage_ids,
        bottleneck_stage_id=build.bottleneck_stage_id,
        selected_bottleneck_stage_ids=build.selected_bottleneck_stage_ids,
        bottleneck_band_radius=build.bottleneck_band_radius,
        middle_band_radius=build.middle_band_radius,
        retained_stage_ratios=build.retained_stage_ratios,
        quantile_count=build.quantile_count,
        job_count=instance.job_count,
        stage_count=instance.stage_count,
        machine_count_per_stage=instance.machine_count_per_stage,
        status=CpsatStatus.OPTIMAL,
        objective_ub=26.0,
        objective_lb=26.0,
        time_limit_sec_used=2.0,
        solver_runtime_sec=0.1,
        wall_runtime_sec=0.1,
        model_build_wall_sec=0.01,
    )

    write_retained_stage_cp_artifacts(
        tmp_path,
        result=result,
        build=build,
        trace_rows=trace_rows,
        retained_solution_rows=retained_rows,
    )

    assert (tmp_path / "summary.yaml").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "trace.csv").is_file()
    assert (tmp_path / "retained_solution.csv").is_file()
