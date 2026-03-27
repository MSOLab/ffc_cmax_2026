from pathlib import Path

import pandas as pd
from routix.constants import SubroutineReportStatisticsKeys

from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hybridflowshop.constants import INPUT_TIMELIMIT_COLUMN


def _make_runner(output_metadata: dict | None = None) -> HfsMultiInstanceRunner:
    runner = HfsMultiInstanceRunner.__new__(HfsMultiInstanceRunner)
    runner.working_dir = Path(".")
    runner.runners = []
    runner.output_metadata = output_metadata or {}
    return runner


def test_sample_last_known_value():
    points = [(1.0, 100.0), (3.0, 90.0), (5.0, 80.0)]

    assert HfsMultiInstanceRunner._sample_last_known_value(points, 0.5) is None
    assert HfsMultiInstanceRunner._sample_last_known_value(points, 3.0) == 90.0
    assert HfsMultiInstanceRunner._sample_last_known_value(points, 4.0) == 90.0


def test_resolve_timepoint_summaries_uses_metadata_when_present():
    runner = _make_runner(
        {
            "timepoint_summaries": [
                {"label": "15p", "mode": "timelimit_ratio", "value": 0.15},
                {
                    "label": "80s",
                    "mode": "absolute_sec",
                    "value": 80,
                    "exclude_if_timelimit_lt": 80,
                },
            ]
        }
    )

    cfgs = runner._resolve_timepoint_summaries()

    assert [c["label"] for c in cfgs] == ["15p", "80s"]
    assert cfgs[0]["mode"] == "timelimit_ratio"
    assert cfgs[0]["value"] == 0.15
    assert cfgs[1]["mode"] == "absolute_sec"
    assert cfgs[1]["value"] == 80.0
    assert cfgs[1]["exclude_if_timelimit_lt"] == 80.0


def test_resolve_timepoint_summaries_returns_empty_when_not_configured():
    runner = _make_runner()

    cfgs = runner._resolve_timepoint_summaries()

    assert cfgs == []


def test_build_timepoint_summary_updates_obj_bound_elapsed_and_columns():
    base_df = pd.DataFrame(
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: ["1", "2"],
            INPUT_TIMELIMIT_COLUMN: [400.0, 200.0],
            SubroutineReportStatisticsKeys.BEST_OBJ: [999.0, 888.0],
            SubroutineReportStatisticsKeys.BEST_BOUND: [950.0, 850.0],
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: [120.0, 30.0],
            "extraCol": ["a", "b"],
        }
    )

    obj_value_map = {
        "1": [(40.0, 1000.0), (95.0, 930.0), (120.0, 900.0)],
        "2": [(10.0, 870.0), (45.0, 860.0)],
    }
    obj_bound_map = {
        "1": [(50.0, 700.0), (99.0, 720.0)],
        "2": [(20.0, 640.0), (50.0, 650.0)],
    }

    out_df = HfsMultiInstanceRunner._build_timepoint_summary(
        base_df=base_df,
        label="25p",
        mode="timelimit_ratio",
        value=0.25,
        exclude_if_timelimit_lt=None,
        obj_value_points_map=obj_value_map,
        obj_bound_points_map=obj_bound_map,
    )

    row1 = out_df[out_df[SubroutineReportStatisticsKeys.INSTANCE_NAME] == "1"].iloc[0]
    row2 = out_df[out_df[SubroutineReportStatisticsKeys.INSTANCE_NAME] == "2"].iloc[0]

    assert row1["targetSec"] == 100.0
    assert row2["targetSec"] == 50.0
    assert row1["timepointLabel"] == "25p"
    assert row2["timepointLabel"] == "25p"

    assert row1[SubroutineReportStatisticsKeys.BEST_OBJ] == 930.0
    assert row1[SubroutineReportStatisticsKeys.BEST_BOUND] == 720.0
    assert row2[SubroutineReportStatisticsKeys.BEST_OBJ] == 860.0
    assert row2[SubroutineReportStatisticsKeys.BEST_BOUND] == 650.0

    assert row1[SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME] == 100.0
    assert row2[SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME] == 30.0

    assert "extraCol" in out_df.columns
    assert "targetSec" in out_df.columns
    assert "timepointLabel" in out_df.columns


def test_build_timepoint_summary_excludes_rows_for_threshold_and_absolute_mode():
    base_df = pd.DataFrame(
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: ["1", "2", "3"],
            INPUT_TIMELIMIT_COLUMN: [400.0, 99.0, 100.0],
            SubroutineReportStatisticsKeys.BEST_OBJ: [999.0, 888.0, 777.0],
            SubroutineReportStatisticsKeys.BEST_BOUND: [950.0, 850.0, 750.0],
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: [120.0, 20.0, 150.0],
        }
    )

    out_df = HfsMultiInstanceRunner._build_timepoint_summary(
        base_df=base_df,
        label="100s",
        mode="absolute_sec",
        value=100.0,
        exclude_if_timelimit_lt=100.0,
        obj_value_points_map={"1": [(50.0, 990.0)], "3": [(90.0, 770.0)]},
        obj_bound_points_map={"1": [(80.0, 940.0)], "3": [(95.0, 740.0)]},
    )

    ids = set(out_df[SubroutineReportStatisticsKeys.INSTANCE_NAME].astype(str))
    assert ids == {"1", "3"}

    row1 = out_df[out_df[SubroutineReportStatisticsKeys.INSTANCE_NAME] == "1"].iloc[0]
    row3 = out_df[out_df[SubroutineReportStatisticsKeys.INSTANCE_NAME] == "3"].iloc[0]

    assert row1[SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME] == 100.0
    assert row3[SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME] == 100.0


def test_build_timepoint_summary_matches_numeric_instance_names_after_filtering():
    base_df = pd.DataFrame(
        {
            SubroutineReportStatisticsKeys.INSTANCE_NAME: [1.0, 2.0],
            INPUT_TIMELIMIT_COLUMN: [400.0, 50.0],
            SubroutineReportStatisticsKeys.BEST_OBJ: [999.0, 888.0],
            SubroutineReportStatisticsKeys.BEST_BOUND: [950.0, 850.0],
            SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: [120.0, 30.0],
        }
    )

    out_df = HfsMultiInstanceRunner._build_timepoint_summary(
        base_df=base_df,
        label="100s",
        mode="absolute_sec",
        value=100.0,
        exclude_if_timelimit_lt=100.0,
        obj_value_points_map={"1": [(50.0, 900.0)]},
        obj_bound_points_map={"1": [(50.0, 700.0)]},
    )

    assert len(out_df) == 1
    row = out_df.iloc[0]
    assert row[SubroutineReportStatisticsKeys.BEST_OBJ] == 900.0
    assert row[SubroutineReportStatisticsKeys.BEST_BOUND] == 700.0
