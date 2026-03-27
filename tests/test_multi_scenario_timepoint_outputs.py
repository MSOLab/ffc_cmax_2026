from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from routix.constants import SubroutineReportStatisticsKeys

from hfs_multi_scenario_runner import HfsMultiScenarioRunner


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


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
