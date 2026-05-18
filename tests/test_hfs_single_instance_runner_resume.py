import datetime
from types import SimpleNamespace

import numpy as np

from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.solution_manager import HfsSolutionManager


class _FakeTimer:
    def __init__(self) -> None:
        self.start_dt = datetime.datetime(2026, 1, 1)

    def set_start_time(self, value) -> None:
        self.start_dt = value


def test_try_apply_resume_ignores_nan_init_bound_and_restores_best_bound() -> None:
    runner = object.__new__(HfsSingleInstanceRunner)
    runner.name = "demo"
    runner.resume_start_time_map = {("j0", "i0", "m0"): 0}
    runner.resume_end_time_map = {("j0", "i0", "m0"): 5}
    runner.resume_obj_store = object()
    runner.resume_summary_dict = {
        "initObj": np.nan,
        "initBound": np.nan,
        "totalElapsedTime": 12.5,
        "bestObj": 5.0,
        "bestBound": 4.0,
        "retainedCpPostDispatchObj": 105.0,
        "retainedCpDispatchCpUb": 100.0,
        "retainedCpDispatchCpLb": 90.0,
    }
    runner.ctrlr = SimpleNamespace(
        obj_store=None,
        solution_manager=HfsSolutionManager(),
        instance=SimpleNamespace(
            job_id_list=["j0"],
            stage_id_list=["i0"],
            stage_2_machines_map={"i0": ["m0"]},
        ),
        timer=_FakeTimer(),
    )

    runner._try_apply_resume()

    assert runner.ctrlr.solution_manager.best_obj_value == 5.0
    assert runner.ctrlr.solution_manager.best_obj_bound == 4.0
    assert runner.ctrlr.last_retained_cp_post_dispatch_obj == 105.0
    assert runner.ctrlr.last_retained_cp_dispatch_cp_ub == 100.0
    assert runner.ctrlr.last_retained_cp_dispatch_cp_lb == 90.0
