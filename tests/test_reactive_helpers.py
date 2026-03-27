import csv
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

from mbls.cpsat import CpsatStatus
from routix import DynamicDataObject

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
        kwargs={"rho": 0.2, "job_count": 3, "computational_time": 5},
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
    assert row["job_count"] == 3
    assert row["timelimit"] == 5
    hdr = ReactiveLoopReportEntry.get_header()
    assert "rho" in hdr and "job_count" in hdr and "timelimit" in hdr


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


def test_reactive_looper_supports_job_count_subroutines():
    class FakeCtrl:
        def __init__(self):
            self.solution_manager = SimpleNamespace(
                _a_is_better_obj_value=lambda a, b: a < b
            )
            self.timer = SimpleNamespace(
                get_elapsed_sec=lambda: 0.0,
                get_remaining_sec=lambda _t: 100.0,
            )
            self.stopping_criteria = SimpleNamespace(timelimit=1000)
            self.obj_store = SimpleNamespace(get_last_gap=lambda: None)

        def is_stopping_condition(self):
            return False

        def critical_job_ns(self, job_count: int = 1, computational_time: float = 1.0):
            return {"obj_value": job_count}

        def job_block_ns(self, rho: float = 0.1, computational_time: float = 1.0):
            return {"obj_value": rho}

    ctrl = FakeCtrl()
    looper = ReactiveLooper(
        cast(HybridFlowShopCpLnsControllerCore, ctrl),
        [
            {
                "method": "critical_job_ns",
                "job_count": 1,
                "computational_time": 1.0,
            },
            {
                "method": "job_block_ns",
                "rho": 0.1,
                "computational_time": 1.0,
            },
        ],
        {
            "job_count": {"step_size": 1, "min": 1, "max": 5},
            "rho": {"step_size": 0.1, "min": 0.1, "max": 1.0},
            "computational_time": {"step_size": 1.0, "min": 0.1, "max": 10.0},
        },
        {"max_loop_count": 1},
    )

    assert looper._get_size_param_name("critical_job_ns") == "job_count"
    assert looper._get_size_param_name("job_block_ns") == "rho"

    looper.obj_value_before_step = 1.0
    looper.no_improvement_step_series_lth = 0
    looper._update_reactive_params(
        "critical_job_ns",
        SimpleNamespace(
            status=CpsatStatus.OPTIMAL,
            obj_value=1.0,
            is_feasible=True,
        ),
    )
    assert (
        looper.reactive_param_tuner_dict["critical_job_ns"].get_current_value(
            "job_count"
        )
        == 2
    )


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

    routine_name = "dummy"
    opening_kwargs = {"rho": 0.1, "computational_time": 1.0}
    routine_data = {routine_name: opening_kwargs}
    reactive_param_tuner_dict = {
        "rho": {"step_size": 0.1, "min": 0.0, "max": 1.0},
        "computational_time": {"step_size": 1.0, "min": 0.1, "max": 10.0},
    }
    stopping_criteria = {"max_loop_count": 1}

    looper = ReactiveLooper(
        cast(HybridFlowShopCpLnsControllerCore, ctrl),
        routine_data,
        reactive_param_tuner_dict,
        stopping_criteria,
    )
    looper.initialize_states()
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


def test_repeat_while_improvement_repeats_until_first_non_improvement():
    ctrl = HybridFlowShopCpLnsControllerCore.__new__(HybridFlowShopCpLnsControllerCore)
    incumbent = SimpleNamespace(makespan=10)
    recorded_contexts = []
    seen_routines = []

    @contextmanager
    def fake_context(name):
        recorded_contexts.append(name)
        yield

    def fake_run_flow(routine_data):
        seen_routines.append(routine_data.to_obj())
        if len(seen_routines) == 1:
            incumbent.makespan = 9
        else:
            incumbent.makespan = 9

    ctrl.solution_manager = SimpleNamespace(get_incumbent=lambda: incumbent)
    ctrl.is_stopping_condition = lambda: False
    ctrl.temporarily_extended_context = fake_context
    ctrl._run_flow = fake_run_flow

    ctrl.repeat_while_improvement(
        routine_data=DynamicDataObject.from_obj({"method": "pw_cp", "batch_size": 3}),
        n_repeats=None,
        max_no_improve=0,
    )

    assert recorded_contexts == ["reps_001", "reps_002"]
    assert seen_routines == [
        {"method": "pw_cp", "batch_size": 3},
        {"method": "pw_cp", "batch_size": 3},
    ]


def test_repeat_while_improvement_stops_at_repeat_limit():
    ctrl = HybridFlowShopCpLnsControllerCore.__new__(HybridFlowShopCpLnsControllerCore)
    incumbent = SimpleNamespace(makespan=10)
    recorded_contexts = []

    @contextmanager
    def fake_context(name):
        recorded_contexts.append(name)
        yield

    def fake_run_flow(routine_data):
        incumbent.makespan -= 1

    ctrl.solution_manager = SimpleNamespace(get_incumbent=lambda: incumbent)
    ctrl.is_stopping_condition = lambda: False
    ctrl.temporarily_extended_context = fake_context
    ctrl._run_flow = fake_run_flow

    ctrl.repeat_while_improvement(
        routine_data=DynamicDataObject.from_obj({"method": "pw_cp", "batch_size": 2}),
        n_repeats=2,
        max_no_improve=0,
    )

    assert recorded_contexts == ["reps_001", "reps_002"]
