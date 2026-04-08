import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from routix.constants import SubroutineReportStatisticsKeys

from hfs_multi_scenario_runner import HfsMultiScenarioRunner


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_method_rpdf_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_subroutine_flow(path: Path, method_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump([{"method": name} for name in method_names], f)


def _write_progression_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_post_run_process_generates_configured_25p_outputs(tmp_path: Path):
    scenario_1 = tmp_path / "ff2020" / "s1"
    scenario_2 = tmp_path / "ff2020" / "s2"

    base_rows_s1 = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1000.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 200.0,
        },
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "2",
            SubroutineReportStatisticsKeys.BEST_OBJ: 900.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 180.0,
        },
    ]
    base_rows_s2 = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 990.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 210.0,
        },
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "2",
            SubroutineReportStatisticsKeys.BEST_OBJ: 880.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 170.0,
        },
    ]

    rows_25p_s1 = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1050.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 100.0,
        },
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "2",
            SubroutineReportStatisticsKeys.BEST_OBJ: 920.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 100.0,
        },
    ]
    rows_25p_s2 = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1030.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 100.0,
        },
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "2",
            SubroutineReportStatisticsKeys.BEST_OBJ: 910.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 100.0,
        },
    ]

    _write_summary(scenario_1 / "multi_instance_summary.csv", base_rows_s1)
    _write_summary(scenario_2 / "multi_instance_summary.csv", base_rows_s2)
    _write_summary(scenario_1 / "multi_instance_summary_25p.csv", rows_25p_s1)
    _write_summary(scenario_2 / "multi_instance_summary_25p.csv", rows_25p_s2)

    runner = HfsMultiScenarioRunner.__new__(HfsMultiScenarioRunner)
    runner.runners = [
        SimpleNamespace(working_dir=scenario_1),
        SimpleNamespace(working_dir=scenario_2),
    ]
    runner.scenario_configs = [
        {"output_subdir": "ff2020/s1", "description": "scenario 1"},
        {"output_subdir": "ff2020/s2", "description": "scenario 2"},
    ]
    runner.output_dir = tmp_path
    runner.baseline_df = pd.DataFrame()
    runner.base_output_metadata = {
        "timepoint_summaries": [
            {"label": "25p", "mode": "timelimit_ratio", "value": 0.25}
        ]
    }

    runner.post_run_process()

    assert (tmp_path / "all_scenarios_summary.csv").exists()
    assert (tmp_path / "multi_scenario_report.xlsx").exists()
    assert (tmp_path / "all_scenarios_summary_25p.csv").exists()
    assert (tmp_path / "multi_scenario_report_25p.xlsx").exists()

    all_df = pd.read_csv(tmp_path / "all_scenarios_summary.csv")
    all_25_df = pd.read_csv(tmp_path / "all_scenarios_summary_25p.csv")

    assert len(all_df) == 4
    assert len(all_25_df) == 4
    assert "scenario" in all_df.columns
    assert "scenario" in all_25_df.columns


def test_post_run_process_creates_top_level_method_comparison_html(
    tmp_path: Path, caplog
):
    scenario_1 = tmp_path / "ff2020" / "s1"
    scenario_2 = tmp_path / "ff2020" / "s2"

    base_rows = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1000.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 200.0,
        }
    ]
    method_rows_s1 = [
        {
            "instance_id": "1",
            "subroutine_name": "initialize",
            "norm_time": 0.10,
            "rpd_f": 0.08,
            "rpd_v": 0.09,
        },
        {
            "instance_id": "1",
            "subroutine_name": "repeat",
            "norm_time": 0.30,
            "rpd_f": 0.04,
            "rpd_v": 0.05,
        },
    ]
    method_rows_s2 = [
        {
            "instance_id": "1",
            "subroutine_name": "initialize",
            "norm_time": 0.12,
            "rpd_f": 0.07,
            "rpd_v": 0.08,
        },
        {
            "instance_id": "1",
            "subroutine_name": "repeat",
            "norm_time": 0.40,
            "rpd_f": 0.02,
            "rpd_v": 0.03,
        },
    ]

    _write_summary(scenario_1 / "multi_instance_summary.csv", base_rows)
    _write_summary(scenario_2 / "multi_instance_summary.csv", base_rows)
    _write_method_rpdf_summary(
        scenario_1 / "summary_method_rpdf_and_norm_time_long.csv", method_rows_s1
    )
    _write_method_rpdf_summary(
        scenario_2 / "summary_method_rpdf_and_norm_time_long.csv", method_rows_s2
    )

    runner = HfsMultiScenarioRunner.__new__(HfsMultiScenarioRunner)
    runner.runners = [
        SimpleNamespace(working_dir=scenario_1),
        SimpleNamespace(working_dir=scenario_2),
    ]
    runner.scenario_configs = [
        {"output_subdir": "ff2020/s1", "description": "scenario 1"},
        {"output_subdir": "ff2020/s2", "description": "scenario 2"},
    ]
    runner.output_dir = tmp_path
    runner.baseline_df = pd.DataFrame()
    runner.base_output_metadata = {}

    runner.post_run_process()

    html_path = tmp_path / "multi_scenario_subroutine_flow_comparison.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "Subroutine Flow Comparison" in content
    assert "s1" in content
    assert "s2" in content
    assert "initialize" in content
    assert "repeat" in content
    assert "step_x" in content
    assert "guide_marker_x" in content
    assert 'legendgroup: trace.scenario' in content
    assert 'groupclick: "togglegroup"' in content
    assert "buildVisibleGuideShapes" in content
    assert 'gd.on("plotly_restyle", syncGuideShapes);' in content
    assert "range: [0, payload.x_max]" in content
    assert "range: [0, payload.y_max]" in content
    assert (
        "Falling back to endpoint CSV for top-level method comparison in scenario 1"
        in caplog.text
    )
    assert (
        "Falling back to endpoint CSV for top-level method comparison in scenario 2"
        in caplog.text
    )


def test_post_run_process_uses_json_endpoint_metrics_for_top_level_comparison(
    tmp_path: Path,
):
    scenario_1 = tmp_path / "ff2020" / "s1"
    scenario_2 = tmp_path / "ff2020" / "s2"

    base_rows = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1000.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 200.0,
        }
    ]
    progression_s1 = {
        "artifact_version": 1,
        "instance_id": "1",
        "timelimit_sec": 50.0,
        "subroutine_calls": [
            {
                "call_index": 1,
                "subroutine_name": "initialize",
                "prefixed_subroutine_name": "1-initialize",
                "global_start_sec": 0.0,
                "global_end_sec": 5.0,
                "elapsed_sec": 5.0,
                "local_progress_list": [
                    {"local_sec": 0.0, "global_sec": 0.0, "obj_value": 120.0},
                    {"local_sec": 5.0, "global_sec": 5.0, "obj_value": 110.0},
                ],
            },
            {
                "call_index": 2,
                "subroutine_name": "repeat",
                "prefixed_subroutine_name": "2-repeat",
                "global_start_sec": 5.0,
                "global_end_sec": 20.0,
                "elapsed_sec": 15.0,
                "local_progress_list": [
                    {"local_sec": 0.0, "global_sec": 5.0, "obj_value": 110.0},
                    {"local_sec": 15.0, "global_sec": 20.0, "obj_value": 100.0},
                ],
            },
        ],
        "combined_progress_list": [],
        "subroutine_end_marker_list": [],
    }
    progression_s2 = {
        "artifact_version": 1,
        "instance_id": "1",
        "timelimit_sec": 50.0,
        "subroutine_calls": [
            {
                "call_index": 1,
                "subroutine_name": "initialize",
                "prefixed_subroutine_name": "1-initialize",
                "global_start_sec": 0.0,
                "global_end_sec": 6.0,
                "elapsed_sec": 6.0,
                "local_progress_list": [
                    {"local_sec": 0.0, "global_sec": 0.0, "obj_value": 118.0},
                    {"local_sec": 6.0, "global_sec": 6.0, "obj_value": 108.0},
                ],
            },
            {
                "call_index": 2,
                "subroutine_name": "repeat",
                "prefixed_subroutine_name": "2-repeat",
                "global_start_sec": 6.0,
                "global_end_sec": 24.0,
                "elapsed_sec": 18.0,
                "local_progress_list": [
                    {"local_sec": 0.0, "global_sec": 6.0, "obj_value": 108.0},
                    {"local_sec": 18.0, "global_sec": 24.0, "obj_value": 102.0},
                ],
            },
        ],
        "combined_progress_list": [],
        "subroutine_end_marker_list": [],
    }

    _write_summary(scenario_1 / "multi_instance_summary.csv", base_rows)
    _write_summary(scenario_2 / "multi_instance_summary.csv", base_rows)
    _write_subroutine_flow(
        scenario_1 / "subroutine_flow.yaml", ["initialize", "repeat", "finalize"]
    )
    _write_subroutine_flow(
        scenario_2 / "subroutine_flow.yaml", ["initialize", "repeat", "finalize"]
    )
    _write_progression_json(
        scenario_1 / "1" / "results" / "subroutine_progression.json", progression_s1
    )
    _write_progression_json(
        scenario_2 / "1" / "results" / "subroutine_progression.json", progression_s2
    )

    runner = HfsMultiScenarioRunner.__new__(HfsMultiScenarioRunner)
    runner.runners = [
        SimpleNamespace(working_dir=scenario_1),
        SimpleNamespace(working_dir=scenario_2),
    ]
    runner.scenario_configs = [
        {"output_subdir": "ff2020/s1", "description": "scenario 1"},
        {"output_subdir": "ff2020/s2", "description": "scenario 2"},
    ]
    runner.output_dir = tmp_path
    runner.baseline_df = pd.DataFrame([{"Instance": "1", "UB": 100.0}])
    runner.baseline_instance_col = "Instance"
    runner.baseline_obj_val_col = "UB"
    runner.base_output_metadata = {}

    runner.post_run_process()

    html_path = tmp_path / "multi_scenario_subroutine_flow_comparison.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "Subroutine Flow Comparison" in content
    assert "s1" in content
    assert "s2" in content
    assert "initialize" in content
    assert "repeat" in content
    assert "finalize" in content
    assert "step_x" in content
    assert "guide_marker_x" in content
    assert 'legendgroup: trace.scenario' in content
    assert 'groupclick: "togglegroup"' in content
    assert "buildVisibleGuideShapes" in content


def test_post_run_process_skips_top_level_method_comparison_when_method_summaries_missing(
    tmp_path: Path,
):
    scenario_1 = tmp_path / "ff2020" / "s1"
    scenario_2 = tmp_path / "ff2020" / "s2"

    base_rows = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1000.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 200.0,
        }
    ]

    _write_summary(scenario_1 / "multi_instance_summary.csv", base_rows)
    _write_summary(scenario_2 / "multi_instance_summary.csv", base_rows)

    runner = HfsMultiScenarioRunner.__new__(HfsMultiScenarioRunner)
    runner.runners = [
        SimpleNamespace(working_dir=scenario_1),
        SimpleNamespace(working_dir=scenario_2),
    ]
    runner.scenario_configs = [
        {"output_subdir": "ff2020/s1", "description": "scenario 1"},
        {"output_subdir": "ff2020/s2", "description": "scenario 2"},
    ]
    runner.output_dir = tmp_path
    runner.baseline_df = pd.DataFrame()
    runner.base_output_metadata = {}

    runner.post_run_process()

    assert not (tmp_path / "multi_scenario_subroutine_flow_comparison.html").exists()


def test_post_run_process_skips_25p_report_when_25p_summary_missing(tmp_path: Path):
    scenario_1 = tmp_path / "ff2020" / "s1"
    scenario_2 = tmp_path / "ff2020" / "s2"

    base_rows = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1000.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 200.0,
        }
    ]

    _write_summary(scenario_1 / "multi_instance_summary.csv", base_rows)
    _write_summary(scenario_2 / "multi_instance_summary.csv", base_rows)
    # Intentionally no multi_instance_summary_25p.csv

    runner = HfsMultiScenarioRunner.__new__(HfsMultiScenarioRunner)
    runner.runners = [
        SimpleNamespace(working_dir=scenario_1),
        SimpleNamespace(working_dir=scenario_2),
    ]
    runner.scenario_configs = [
        {"output_subdir": "ff2020/s1", "description": "scenario 1"},
        {"output_subdir": "ff2020/s2", "description": "scenario 2"},
    ]
    runner.output_dir = tmp_path
    runner.baseline_df = pd.DataFrame()
    runner.base_output_metadata = {
        "timepoint_summaries": [
            {"label": "25p", "mode": "timelimit_ratio", "value": 0.25}
        ]
    }

    runner.post_run_process()

    assert (tmp_path / "all_scenarios_summary.csv").exists()
    assert (tmp_path / "multi_scenario_report.xlsx").exists()
    assert not (tmp_path / "all_scenarios_summary_25p.csv").exists()
    assert not (tmp_path / "multi_scenario_report_25p.xlsx").exists()


def test_post_run_process_skips_timepoint_reports_when_not_configured(tmp_path: Path):
    scenario_1 = tmp_path / "ff2020" / "s1"
    scenario_2 = tmp_path / "ff2020" / "s2"

    base_rows = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1000.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 200.0,
        }
    ]

    _write_summary(scenario_1 / "multi_instance_summary.csv", base_rows)
    _write_summary(scenario_2 / "multi_instance_summary.csv", base_rows)
    # Even if timepoint files exist, they should be ignored when not configured.
    _write_summary(scenario_1 / "multi_instance_summary_25p.csv", base_rows)
    _write_summary(scenario_2 / "multi_instance_summary_25p.csv", base_rows)

    runner = HfsMultiScenarioRunner.__new__(HfsMultiScenarioRunner)
    runner.runners = [
        SimpleNamespace(working_dir=scenario_1),
        SimpleNamespace(working_dir=scenario_2),
    ]
    runner.scenario_configs = [
        {"output_subdir": "ff2020/s1", "description": "scenario 1"},
        {"output_subdir": "ff2020/s2", "description": "scenario 2"},
    ]
    runner.output_dir = tmp_path
    runner.baseline_df = pd.DataFrame()
    runner.base_output_metadata = {}

    runner.post_run_process()

    assert (tmp_path / "all_scenarios_summary.csv").exists()
    assert (tmp_path / "multi_scenario_report.xlsx").exists()
    assert not (tmp_path / "all_scenarios_summary_25p.csv").exists()
    assert not (tmp_path / "multi_scenario_report_25p.xlsx").exists()


def test_post_run_process_generates_reports_for_configured_timepoint_labels(
    tmp_path: Path,
):
    scenario_1 = tmp_path / "ff2020" / "s1"
    scenario_2 = tmp_path / "ff2020" / "s2"

    base_rows = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1000.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 200.0,
        }
    ]
    rows_10p = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1200.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 40.0,
        }
    ]
    rows_100s = [
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: "1",
            SubroutineReportStatisticsKeys.BEST_OBJ: 1100.0,
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: 100.0,
        }
    ]

    for scenario in [scenario_1, scenario_2]:
        _write_summary(scenario / "multi_instance_summary.csv", base_rows)
        _write_summary(scenario / "multi_instance_summary_10p.csv", rows_10p)
        _write_summary(scenario / "multi_instance_summary_100s.csv", rows_100s)

    runner = HfsMultiScenarioRunner.__new__(HfsMultiScenarioRunner)
    runner.runners = [
        SimpleNamespace(
            working_dir=scenario_1,
            output_metadata={
                "timepoint_summaries": [
                    {"label": "10p", "mode": "timelimit_ratio", "value": 0.10},
                    {"label": "100s", "mode": "absolute_sec", "value": 100.0},
                ]
            },
        ),
        SimpleNamespace(working_dir=scenario_2, output_metadata={}),
    ]
    runner.scenario_configs = [
        {"output_subdir": "ff2020/s1", "description": "scenario 1"},
        {"output_subdir": "ff2020/s2", "description": "scenario 2"},
    ]
    runner.output_dir = tmp_path
    runner.baseline_df = pd.DataFrame()

    runner.post_run_process()

    assert (tmp_path / "all_scenarios_summary_10p.csv").exists()
    assert (tmp_path / "multi_scenario_report_10p.xlsx").exists()
    assert (tmp_path / "all_scenarios_summary_100s.csv").exists()
    assert (tmp_path / "multi_scenario_report_100s.xlsx").exists()

    assert not (tmp_path / "all_scenarios_summary_25p.csv").exists()
    assert not (tmp_path / "multi_scenario_report_25p.xlsx").exists()
