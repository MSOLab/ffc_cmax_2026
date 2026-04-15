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
        },
    )

    summary.save(output_path)
    df = pd.read_csv(output_path)

    assert list(df["mipLbBound"]) == [1234]
    assert list(df["mipLbStatus"]) == ["TIME_LIMIT"]
    assert list(df["mipLbApplyElapsedSec"]) == [12.3]


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
