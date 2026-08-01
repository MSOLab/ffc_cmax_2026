from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from routix.constants import SubroutineReportStatisticsKeys
from routix.type_defs import RunMode

from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.controller.controller_core import (
    HybridFlowShopCpLnsControllerCore,
)


class _FailingController:
    def set_working_dir(self, _working_dir) -> None:
        pass

    def run(self) -> None:
        raise ValueError("controller failed")


def test_single_instance_run_propagates_controller_error_without_post_processing() -> (
    None
):
    runner = object.__new__(HfsSingleInstanceRunner)
    runner.mode = RunMode.FULL_RUN
    runner.working_dir = None
    runner.get_controller = lambda: _FailingController()
    runner.post_run_process = MagicMock()

    with pytest.raises(ValueError, match="controller failed"):
        runner.run()

    runner.post_run_process.assert_not_called()


def test_single_instance_run_does_not_swallow_keyboard_interrupt() -> None:
    runner = object.__new__(HfsSingleInstanceRunner)
    runner.mode = RunMode.FULL_RUN
    runner.working_dir = None
    controller = _FailingController()
    controller.run = MagicMock(side_effect=KeyboardInterrupt)
    runner.get_controller = lambda: controller
    runner.post_run_process = MagicMock()

    with pytest.raises(KeyboardInterrupt):
        runner.run()

    runner.post_run_process.assert_not_called()


def test_call_method_preserves_original_traceback() -> None:
    controller = object.__new__(HybridFlowShopCpLnsControllerCore)
    controller.final_time_reserve_is_reached = lambda: False
    controller._method_context_mgr = SimpleNamespace(
        push=MagicMock(),
        pop=MagicMock(),
    )
    controller._get_call_context_of_current_method = lambda: "0-explode"
    controller._record_method_context_start = MagicMock()
    controller._record_method_context_end = MagicMock()
    controller.timer = SimpleNamespace(elapsed_sec=0.0)
    controller.method_call_counts = defaultdict(int)

    def origin() -> None:
        raise RuntimeError("boom")

    def explode() -> None:
        origin()

    controller.explode = explode

    with pytest.raises(RuntimeError, match="boom") as exc_info:
        controller._call_method("explode")

    frame_names = []
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        frame_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next

    assert frame_names.count("_call_method") == 1
    assert frame_names[-2:] == ["explode", "origin"]


def test_sequential_runner_records_failure_and_logs_traceback() -> None:
    instance_runner = SimpleNamespace(
        ins_name="broken-instance",
        run=MagicMock(side_effect=ValueError("instance failed")),
    )
    runner = object.__new__(HfsMultiInstanceRunner)
    runner.runners = [instance_runner]
    runner.append_result = MagicMock()
    runner.post_run_process = MagicMock(return_value="done")

    result = runner._run_sequential()

    assert result == "done"
    failure_row = runner.append_result.call_args.args[0]
    assert (
        failure_row[SubroutineReportStatisticsKeys.INSTANCE_NAME] == "broken-instance"
    )
    assert failure_row[SubroutineReportStatisticsKeys.FOUND_FEASIBLE_SOL] is False
    assert failure_row["error"] == "instance failed"


def test_append_result_keeps_mixed_success_and_failure_rows_valid(tmp_path) -> None:
    runner = object.__new__(HfsMultiInstanceRunner)
    runner.results = []
    runner.working_dir = tmp_path

    runner.append_result({"insName": "ok", "bestObj": 10})
    runner.append_result(
        runner._create_failure_result("failed", ValueError("instance failed"))
    )

    result_df = pd.read_csv(tmp_path / "multi_instance_summary.csv")
    assert result_df["insName"].tolist() == ["ok", "failed"]
    assert pd.isna(result_df.loc[0, "error"])
    assert result_df.loc[1, "error"] == "instance failed"


def test_append_result_preserves_rows_already_persisted_to_csv(tmp_path) -> None:
    summary_path = tmp_path / "multi_instance_summary.csv"
    existing_row = HfsSingleInstanceRunner.normalize_summary_row(
        {"insName": "existing", "bestObj": 20}
    )
    pd.DataFrame([existing_row]).to_csv(summary_path, index=False)
    runner = object.__new__(HfsMultiInstanceRunner)
    runner.results = []
    runner.working_dir = tmp_path

    runner.append_result({"insName": "new", "bestObj": 10, "error": None})

    result_df = pd.read_csv(summary_path)
    assert result_df["insName"].tolist() == ["existing", "new"]
