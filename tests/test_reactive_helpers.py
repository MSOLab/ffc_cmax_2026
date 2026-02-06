import csv
from types import SimpleNamespace
from typing import cast

from hybridflowshop.controller.controller_core import HybridFlowShopCpLnsControllerCore
from hybridflowshop.controller.reactive.reactive_loop_report import (
    ReactiveLoopReportEntry,
)
from hybridflowshop.controller.reactive.reactive_looper import ReactiveLooper
from hybridflowshop.controller.reactive.reactive_param_tuner import (
    ReactiveParamTuner,
    TunerParams,
)


def test_report_entry_row_and_header():
    entry = ReactiveLoopReportEntry(
        iter_count=1,
        subroutine_name="op",
        kwargs={"rho": 0.2, "computational_time": 5},
        time_start=0.0,
        time_elapsed=0.1,
        prev_obj_value=999.0,
        obj_value=123.0,
        timelimit_reached=False,
        is_optimal=False,
        is_improved=True,
    )
    row = entry.get_row_dict()
    assert row["rho"] == 0.2
    assert row["timelimit"] == 5
    hdr = ReactiveLoopReportEntry.get_header()
    assert "rho" in hdr and "timelimit" in hdr


def test_tuner_basic_behavior():
    # dummy method records last kwargs into a namespace for inspection
    ctx = SimpleNamespace()

    def dummy_method(rho: float = 0.1, computational_time: float = 1.0):
        ctx.rho = rho
        ctx.tim = computational_time
        return {"obj_value": rho}

    tuner = ReactiveParamTuner(
        method=dummy_method,
        opening_kwargs={"rho": 0.1, "computational_time": 1.0},
        tuner_param_dict={
            "rho": TunerParams(step_size=0.1, min=0.0, max=1.0),
            "computational_time": TunerParams(step_size=1.0, min=0.1, max=10.0),
        },
    )

    tuner.call_method()
    assert ctx.rho == 0.1
    assert ctx.tim == 1.0
    tuner.increment("rho")
    assert tuner.get_current_value("rho") == 0.2
    tuner.decrement("rho")
    assert tuner.get_current_value("rho") == 0.1


def test_reactive_looper_writes_reports(tmp_path):
    # Minimal fake controller with needed interfaces
    class FakeSolutionManager:
        def __init__(self):
            self.best_obj_value = None

        def get_last_report(self):
            # Simulate an HfsCpsatSolverReport-like object
            return SimpleNamespace(
                obj_value=1.0, status=0, is_feasible=True, elapsed_time=0.01
            )

    class FakeTimer:
        def get_elapsed_sec(self):
            return 0.0

        def get_remaining_sec(self, t):
            return 100.0

    class FakeCtrl:
        def __init__(self):
            self.solution_manager = FakeSolutionManager()
            self.timer = FakeTimer()
            self.stopping_criteria = SimpleNamespace(timelimit=1000)
            self.obj_store = SimpleNamespace(get_last_gap=lambda: None)

        def is_stopping_condition(self):
            return False

        # context manager stub
        def temporarily_extended_context(self, name):
            class Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Ctx()

    ctrl = FakeCtrl()

    # configure looper expecting 1 subroutine named 'dummy'
    # create a dummy method on controller
    def dummy_method(rho: float = 0.1, computational_time: float = 1.0):
        return {"obj_value": 1.0}

    setattr(ctrl, "dummy", dummy_method)

    subroutine_names = ["dummy"]
    opening_kwargs = {"rho": 0.1, "computational_time": 1.0}
    reactive_param_tuner_dict = {
        "rho": {"step_size": 0.1, "min": 0.0, "max": 1.0},
        "computational_time": {"step_size": 1.0, "min": 0.1, "max": 10.0},
    }
    stopping_criteria = {"max_loop_count": 1}

    looper = ReactiveLooper(
        cast(HybridFlowShopCpLnsControllerCore, ctrl),
        subroutine_names,
        opening_kwargs,
        reactive_param_tuner_dict,
        stopping_criteria,
    )
    # Ensure stopping criteria attributes exist on looper.stopping_criteria
    looper.stopping_criteria.max_loop_count = 1
    looper.stopping_criteria.stop_at_global_timelimit_minus = None
    # call one subroutine
    looper.loop_count = 1
    # instead of calling into CP solver report flow, insert a synthetic report entry
    from hybridflowshop.controller.reactive.reactive_loop_report import (
        ReactiveLoopReportEntry,
    )

    looper.report_entries.append(
        ReactiveLoopReportEntry(
            iter_count=1,
            subroutine_name="dummy",
            kwargs=opening_kwargs,
            time_start=0.0,
            time_elapsed=0.01,
            prev_obj_value=1.0,
            obj_value=1.0,
            timelimit_reached=False,
            is_optimal=False,
            is_improved=False,
        )
    )

    # write reports
    yaml_path = tmp_path / "rep.yaml"
    csv_path = tmp_path / "rep.csv"
    looper.write_report_yaml(yaml_path)
    looper.write_report_csv(csv_path)

    assert yaml_path.exists()
    text = yaml_path.read_text()
    assert "iterCount" in text or "subroutineName" in text

    assert csv_path.exists()
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) >= 1
