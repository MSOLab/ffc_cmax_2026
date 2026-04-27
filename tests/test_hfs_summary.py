from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.hfs_input_summary import HfsInputSummary
from hybridflowshop.hfs_summary import HfsSummary


class _FakeStats:
    def to_string_dict(self) -> dict[str, str]:
        return {
            "foundFeasibleSol": "True",
            "totalElapsedTime": "12.5",
            "initObj": "100",
            "initBound": "90",
            "bestObj": "80",
            "bestBound": "95",
            "improvementRatio": "0.2",
            "methodCallCounts": '"{}"',
            "reportCount": "3",
        }


def test_hfs_summary_save_includes_extra_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.csv"
    summary = HfsSummary(
        inputs=HfsInputSummary(
            name="1",
            job_count=20,
            stage_count=5,
            machines_per_stage=2,
            timelimit=100.0,
        ),
        outputs=_FakeStats(),
        extra_outputs={
            "mipLbBound": 1234,
            "mipLbStatus": "TIME_LIMIT",
            "mipLbApplyElapsedSec": 12.3,
            "retainedCpBound": 1200,
            "retainedCpCallCount": 2,
            "retainedCpBounds": "1180;1200",
            "retainedCpModes": "first_last;first_topk_bottlenecks_last",
            "retainedCpBestBound": 1200,
            "retainedCpBestStatus": "FEASIBLE",
            "retainedCpBestMode": "first_topk_bottlenecks_last",
            "retainedCpBestStages": "i0 i2 i4",
            "retainedCpBestBottleneckStage": "i2",
            "retainedCpBestSolverRuntimeSec": 4.2,
            "retainedCpBestApplyElapsedSec": 4.5,
            "retainedCpBestBoundObj": 1210,
            "retainedCpStatus": "FEASIBLE",
            "retainedCpApplyElapsedSec": 4.5,
            "retainedCpBestObj": 1210,
            "retainedCpDispatchObj": 1215,
            "retainedCpPostDispatchObj": 1215,
            "retainedCpSelectedDispatchVariant": "cp_band_preferred_priority_release",
            "retainedCpPostSelectedDispatchVariant": "cp_band_preferred_priority_release",
            "retainedCpDispatchAnchorStages": "i1 i2 i3",
            "retainedCpDispatchElapsedSec": 1.2,
            "retainedCpDispatchUpdatedIncumbent": True,
            "retainedCpDispatchKeptIncumbent": False,
        },
    )

    summary.save(output_path)
    df = pd.read_csv(output_path)

    assert list(df["mipLbBound"]) == [1234]
    assert list(df["mipLbStatus"]) == ["TIME_LIMIT"]
    assert list(df["mipLbApplyElapsedSec"]) == [12.3]
    assert list(df["retainedCpBound"]) == [1200]
    assert list(df["retainedCpCallCount"]) == [2]
    assert list(df["retainedCpBounds"]) == ["1180;1200"]
    assert list(df["retainedCpBestBound"]) == [1200]
    assert list(df["retainedCpBestMode"]) == ["first_topk_bottlenecks_last"]
    assert list(df["retainedCpBestBoundObj"]) == [1210]
    assert list(df["retainedCpStatus"]) == ["FEASIBLE"]
    assert list(df["retainedCpApplyElapsedSec"]) == [4.5]
    assert list(df["retainedCpDispatchObj"]) == [1215]
    assert list(df["retainedCpPostDispatchObj"]) == [1215]
    assert list(df["retainedCpSelectedDispatchVariant"]) == [
        "cp_band_preferred_priority_release"
    ]
    assert list(df["retainedCpPostSelectedDispatchVariant"]) == [
        "cp_band_preferred_priority_release"
    ]
    assert list(df["retainedCpDispatchAnchorStages"]) == ["i1 i2 i3"]
    assert list(df["retainedCpDispatchElapsedSec"]) == [1.2]
    assert list(df["retainedCpDispatchUpdatedIncumbent"]) == [True]
    assert list(df["retainedCpDispatchKeptIncumbent"]) == [False]


def test_create_summary_row_includes_mip_lb_columns(tmp_path: Path) -> None:
    summary_path = tmp_path / "1_summary.csv"
    pd.DataFrame(
        [
            {
                "insName": "1",
                "foundFeasibleSol": True,
                "totalElapsedTime": 50.0,
                "initObj": 1000,
                "initBound": 800,
                "bestObj": 900,
                "bestBound": 850,
                "improvementRatio": 0.1,
                "methodCallCounts": "{}",
                "reportCount": 5,
                "mipLbBound": 870,
                "mipLbStatus": "TIME_LIMIT",
                "mipLbDelta": 400,
                "mipLbSolverRuntimeSec": 7.5,
                "mipLbApplyElapsedSec": 8.1,
                "mipLbDispatchCmax": 980,
                "mipLbSelectedDispatchVariant": "best_of_mixed_dispatches",
                "retainedCpBound": 860,
                "retainedCpCallCount": 2,
                "retainedCpBounds": "840;860",
                "retainedCpModes": "first_last;first_bottleneck_last",
                "retainedCpBestBound": 860,
                "retainedCpBestStatus": "FEASIBLE",
                "retainedCpBestMode": "first_bottleneck_last",
                "retainedCpBestStages": "i0 i1 i2",
                "retainedCpBestBottleneckStage": "i1",
                "retainedCpBestSolverRuntimeSec": 3.5,
                "retainedCpBestApplyElapsedSec": 3.9,
                "retainedCpBestBoundObj": 875,
                "retainedCpStatus": "FEASIBLE",
                "retainedCpMode": "first_bottleneck_last",
                "retainedCpStages": "i0 i1 i2",
                "retainedCpBottleneckStage": "i1",
                "retainedCpSolverRuntimeSec": 3.5,
                "retainedCpApplyElapsedSec": 3.9,
                "retainedCpBestObj": 875,
                "retainedCpDispatchObj": 878,
                "retainedCpPostDispatchObj": 878,
                "retainedCpSelectedDispatchVariant": "best_of_mixed_dispatches_cp_aggregate_start_slack_rank",
                "retainedCpPostSelectedDispatchVariant": "best_of_mixed_dispatches_cp_aggregate_start_slack_rank",
                "retainedCpDispatchAnchorStages": "i0 i1 i2",
                "retainedCpDispatchElapsedSec": 1.1,
                "retainedCpDispatchUpdatedIncumbent": False,
                "retainedCpDispatchKeptIncumbent": False,
            }
        ]
    ).to_csv(summary_path, index=False)

    runner = object.__new__(HfsSingleInstanceRunner)
    runner.name = "1"
    runner.summary_path = summary_path
    runner.instance = SimpleNamespace(
        name="1",
        job_count=20,
        stage_count=5,
        machine_count_per_stage=[2],
    )
    runner.stopping_criteria = SimpleNamespace(timelimit=100.0)

    summary_row = runner._create_summary_row()

    assert summary_row is not None
    assert summary_row["mipLbBound"] == 870
    assert summary_row["mipLbStatus"] == "TIME_LIMIT"
    assert summary_row["mipLbDelta"] == 400
    assert summary_row["mipLbSolverRuntimeSec"] == 7.5
    assert summary_row["mipLbApplyElapsedSec"] == 8.1
    assert summary_row["mipLbDispatchCmax"] == 980
    assert (
        summary_row["mipLbSelectedDispatchVariant"]
        == "best_of_mixed_dispatches"
    )
    assert summary_row["retainedCpBound"] == 860
    assert summary_row["retainedCpCallCount"] == 2
    assert summary_row["retainedCpBounds"] == "840;860"
    assert summary_row["retainedCpModes"] == "first_last;first_bottleneck_last"
    assert summary_row["retainedCpBestBound"] == 860
    assert summary_row["retainedCpBestStatus"] == "FEASIBLE"
    assert summary_row["retainedCpBestMode"] == "first_bottleneck_last"
    assert summary_row["retainedCpBestStages"] == "i0 i1 i2"
    assert summary_row["retainedCpBestBottleneckStage"] == "i1"
    assert summary_row["retainedCpBestSolverRuntimeSec"] == 3.5
    assert summary_row["retainedCpBestApplyElapsedSec"] == 3.9
    assert summary_row["retainedCpBestBoundObj"] == 875
    assert summary_row["retainedCpStatus"] == "FEASIBLE"
    assert summary_row["retainedCpMode"] == "first_bottleneck_last"
    assert summary_row["retainedCpStages"] == "i0 i1 i2"
    assert summary_row["retainedCpBottleneckStage"] == "i1"
    assert summary_row["retainedCpSolverRuntimeSec"] == 3.5
    assert summary_row["retainedCpApplyElapsedSec"] == 3.9
    assert summary_row["retainedCpBestObj"] == 875
    assert summary_row["retainedCpDispatchObj"] == 878
    assert summary_row["retainedCpPostDispatchObj"] == 878
    assert (
        summary_row["retainedCpSelectedDispatchVariant"]
        == "best_of_mixed_dispatches_cp_aggregate_start_slack_rank"
    )
    assert (
        summary_row["retainedCpPostSelectedDispatchVariant"]
        == "best_of_mixed_dispatches_cp_aggregate_start_slack_rank"
    )
    assert summary_row["retainedCpDispatchAnchorStages"] == "i0 i1 i2"
    assert summary_row["retainedCpDispatchElapsedSec"] == 1.1
    assert summary_row["retainedCpDispatchUpdatedIncumbent"] is False
    assert summary_row["retainedCpDispatchKeptIncumbent"] is False


def test_build_extra_summary_fields_keeps_retained_cp_history() -> None:
    runner = object.__new__(HfsSingleInstanceRunner)
    runner.ctrlr = SimpleNamespace(
        last_mip_lb_result=None,
        last_mip_lb_solution_payload=None,
        last_retained_cp_lb_result=SimpleNamespace(
            certified_final_lb=100.0,
            status_name="FEASIBLE",
            retained_stage_mode="first_last",
            retained_stage_ids=["i0", "i4"],
            bottleneck_stage_id=None,
            solver_runtime_sec=8.0,
            objective_ub=130.0,
        ),
        last_retained_cp_lb_apply_elapsed_sec=8.5,
        retained_cp_lb_records=[
            {
                "bound": 100.0,
                "status": "FEASIBLE",
                "mode": "first_last",
                "stage_ids": ["i0", "i4"],
                "bottleneck_stage_id": None,
                "solver_runtime_sec": 8.0,
                "apply_elapsed_sec": 8.5,
                "objective_ub": 130.0,
            },
            {
                "bound": 112.0,
                "status": "FEASIBLE",
                "mode": "first_topk_bottlenecks_last",
                "stage_ids": ["i0", "i2", "i4"],
                "bottleneck_stage_id": "i2",
                "solver_runtime_sec": 9.0,
                "apply_elapsed_sec": 9.5,
                "objective_ub": 125.0,
            },
        ],
        best_retained_cp_lb_record={
            "bound": 112.0,
            "status": "FEASIBLE",
            "mode": "first_topk_bottlenecks_last",
            "stage_ids": ["i0", "i2", "i4"],
            "bottleneck_stage_id": "i2",
            "solver_runtime_sec": 9.0,
            "apply_elapsed_sec": 9.5,
            "objective_ub": 125.0,
        },
    )

    fields = runner._build_extra_summary_fields()

    assert fields["retainedCpBound"] == 100.0
    assert fields["retainedCpCallCount"] == 2
    assert fields["retainedCpBounds"] == "100.0;112.0"
    assert fields["retainedCpModes"] == "first_last;first_topk_bottlenecks_last"
    assert fields["retainedCpBestBound"] == 112.0
    assert fields["retainedCpBestMode"] == "first_topk_bottlenecks_last"
    assert fields["retainedCpBestStages"] == "i0 i2 i4"
    assert fields["retainedCpBestBottleneckStage"] == "i2"
    assert fields["retainedCpBestSolverRuntimeSec"] == 9.0
    assert fields["retainedCpBestApplyElapsedSec"] == 9.5
    assert fields["retainedCpBestBoundObj"] == 125.0
    assert fields["retainedCpBestObj"] == 130.0
