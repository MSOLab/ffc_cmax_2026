from types import SimpleNamespace

import hybridflowshop.controller.hfs_cp_lns as hfs_cp_lns_module
from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule
from lb_bucket.cp.search import RetainedStageCpResult


class _FakeSchedule:
    def __init__(
        self,
        makespan: int,
        *,
        label: str | None = None,
        seq: list[str] | None = None,
    ) -> None:
        self.makespan = makespan
        self.label = label or f"obj_{makespan}"
        self.seq = list(seq or [])

    def get_jik_2_start_time_map(self):
        return {}

    def get_jik_2_end_time_map(self):
        return {}

    def make_semi_active(self, *_args, **_kwargs) -> None:
        return None

    def deepcopy(self, *_args, **_kwargs):
        return _FakeSchedule(self.makespan, label=self.label, seq=self.seq)


class _RecorderSolutionManager:
    def __init__(self, incumbent=None, best_obj_bound=None) -> None:
        self.incumbent = incumbent
        self.best_obj_value = (
            float(incumbent.makespan) if incumbent is not None else None
        )
        self.best_obj_bound = best_obj_bound
        self.registered = []

    def get_incumbent(self):
        return self.incumbent

    def _a_is_better_obj_value(self, value_a, value_b):
        return value_b is None or value_a < value_b

    def register(self, report, solution, update_if_equal_obj=False):
        self.registered.append((report, solution))
        if solution is None:
            return False
        obj_value = float(solution.makespan)
        if obj_value != report.obj_value:
            raise ValueError(
                f"Inconsistent objective value: {obj_value} != {report.obj_value}"
            )
        if self._a_is_better_obj_value(obj_value, self.best_obj_value):
            self.incumbent = solution
            self.best_obj_value = obj_value
            return True
        if update_if_equal_obj and obj_value == self.best_obj_value:
            self.incumbent = solution
            return True
        return False


def _wire_common_controller(ctrl: HybridFlowShopCpLnsController) -> None:
    ctrl.timer = SimpleNamespace(elapsed_sec=0.0)
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None,
    )
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "test-call"
    ctrl._make_subroutine_report = lambda **kwargs: SimpleNamespace(**kwargs)
    ctrl.set_cp_model_as_base_cp_model = lambda: None
    ctrl.is_stopping_condition = lambda *args, **kwargs: False


def _make_retained_cp_result(objective_ub: float) -> RetainedStageCpResult:
    return RetainedStageCpResult(
        ins_name="demo",
        input_lb=10.0,
        input_ub=50.0,
        retained_stage_mode="first_topk_bottlenecks_last",
        retained_stage_ids=("i0", "i1", "i2"),
        bottleneck_stage_id="i1",
        selected_bottleneck_stage_ids=("i1",),
        bottleneck_band_radius=None,
        middle_band_radius=None,
        retained_stage_ratios=(),
        quantile_count=None,
        job_count=3,
        stage_count=3,
        machine_count_per_stage="1 1 1",
        objective_ub=objective_ub,
        objective_lb=10.0,
        certified_final_lb=10.0,
        status_name="FEASIBLE",
        time_limit_sec_used=1.0,
        solver_runtime_sec=0.8,
        wall_runtime_sec=0.8,
        model_build_wall_sec=0.1,
    )


def test_bound_gap_guarded_incremental_sw_cp_skips_when_gap_is_small(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(1050), best_obj_bound=1000.0
    )
    calls = []
    monkeypatch.setattr(
        ctrl, "incremental_sw_cp", lambda **kwargs: calls.append(kwargs)
    )

    ctrl.bound_gap_guarded_incremental_sw_cp(
        solver_thread_cnt=16,
        min_incumbent_bound_gap_ratio=0.07,
        batch_size_ratio=0.05,
    )

    assert calls == []


def test_bound_gap_guarded_incremental_sw_cp_runs_when_gap_is_large(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(1080), best_obj_bound=1000.0
    )
    calls = []
    monkeypatch.setattr(
        ctrl, "incremental_sw_cp", lambda **kwargs: calls.append(kwargs)
    )

    ctrl.bound_gap_guarded_incremental_sw_cp(
        solver_thread_cnt=16,
        min_incumbent_bound_gap_ratio=0.07,
        batch_size_ratio=0.05,
        unfixed_batch_count_max=8,
    )

    assert len(calls) == 1
    assert calls[0]["solver_thread_cnt"] == 16
    assert calls[0]["batch_size_ratio"] == 0.05
    assert calls[0]["unfixed_batch_count_max"] == 8


def test_workload_guarded_neh_cp_can_skip_on_small_bound_gap(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(1050), best_obj_bound=1000.0
    )
    calls = []
    monkeypatch.setattr(ctrl, "neh_cp", lambda **kwargs: calls.append(kwargs))

    ctrl.workload_guarded_neh_cp(
        solver_thread_cnt=16,
        added_batch_size=25,
        min_incumbent_bound_gap_ratio=0.07,
    )

    assert calls == []


def test_workload_guarded_neh_cp_can_skip_on_large_restore_loss(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(3400), best_obj_bound=3100.0
    )
    ctrl.last_retained_cp_dispatch_cp_ub = 3200.0
    ctrl.last_retained_cp_post_dispatch_obj = 3400.0
    calls = []
    monkeypatch.setattr(ctrl, "neh_cp", lambda **kwargs: calls.append(kwargs))

    ctrl.workload_guarded_neh_cp(
        solver_thread_cnt=16,
        added_batch_size=25,
        min_incumbent_bound_gap_ratio=0.07,
        max_retained_cp_restore_loss_ratio=0.04,
    )

    assert calls == []


def test_workload_guarded_neh_cp_runs_inside_restore_loss_guard(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(3317), best_obj_bound=3100.0
    )
    ctrl.last_retained_cp_dispatch_cp_ub = 3270.0
    ctrl.last_retained_cp_post_dispatch_obj = 3317.0
    calls = []
    monkeypatch.setattr(ctrl, "neh_cp", lambda **kwargs: calls.append(kwargs))

    ctrl.workload_guarded_neh_cp(
        solver_thread_cnt=16,
        added_batch_size=25,
        min_incumbent_bound_gap_ratio=0.06,
        max_retained_cp_restore_loss_ratio=0.04,
    )

    assert len(calls) == 1
    assert calls[0]["added_batch_size"] == 25


def test_workload_scaled_guarded_neh_cp_selects_medium_budget(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=160, stage_count=15)
    calls = []
    monkeypatch.setattr(
        ctrl,
        "workload_guarded_neh_cp",
        lambda **kwargs: calls.append(kwargs),
    )

    ctrl.workload_scaled_guarded_neh_cp(
        solver_thread_cnt=16,
        min_incumbent_bound_gap_ratio=0.035,
        max_retained_cp_restore_loss_ratio=0.06,
    )

    assert len(calls) == 1
    assert calls[0]["added_batch_size"] == 10
    assert calls[0]["cp_tl_nc_multiplier"] == 0.030
    assert calls[0]["cp_tl_nc_multiplier_2nd_obj"] == 0.004
    assert calls[0]["max_added_batch_count"] == 6
    assert calls[0]["min_remaining_nc_after_neh"] == 0.55
    assert calls[0]["min_incumbent_bound_gap_ratio"] == 0.035
    assert calls[0]["max_retained_cp_restore_loss_ratio"] == 0.06


def test_workload_scaled_guarded_neh_cp_selects_large_budget(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=200, stage_count=20)
    calls = []
    monkeypatch.setattr(
        ctrl,
        "workload_guarded_neh_cp",
        lambda **kwargs: calls.append(kwargs),
    )

    ctrl.workload_scaled_guarded_neh_cp(
        solver_thread_cnt=16,
        minimize_sum_ci_lex=True,
        use_lns_only=True,
    )

    assert len(calls) == 1
    assert calls[0]["added_batch_size"] == 20
    assert calls[0]["cp_tl_nc_multiplier"] == 0.010
    assert calls[0]["cp_tl_nc_multiplier_2nd_obj"] == 0.0025
    assert calls[0]["max_added_batch_count"] == 4
    assert calls[0]["min_remaining_nc_after_neh"] == 0.75
    assert calls[0]["minimize_sum_ci_lex"] is True
    assert calls[0]["use_lns_only"] is True


def test_neh_cp_branch_portfolio_runs_lanes_from_same_prefix_and_keeps_best(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=5)
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.solution_manager = _RecorderSolutionManager(_FakeSchedule(100, label="prefix"))
    _wire_common_controller(ctrl)

    starts = []
    returns = iter([98, 97, 95, 99])

    class _FakeObjSeries:
        def __init__(self, obj):
            self._obj = obj

        def items(self):
            return [(0.0, float(self._obj))]

    class _FakeObjStore:
        def __init__(self, obj):
            self.obj_value_series = _FakeObjSeries(obj)

    class _FakeResult:
        def __init__(self, obj):
            self.schedule = _FakeSchedule(obj)
            self.sub_obj_store = _FakeObjStore(obj)
            self.last_obj_value = obj

    class _FakeNehConstructor:
        def __init__(self, _ctx):
            pass

        def run(self, ref_schedule, *_args, **_kwargs):
            starts.append(ref_schedule.makespan)
            return _FakeResult(next(returns))

    monkeypatch.setattr(hfs_cp_lns_module, "NehCpConstructor", _FakeNehConstructor)

    ctrl.neh_cp_branch_portfolio(
        solver_thread_cnt=16,
        lanes=[
            {
                "name": "A",
                "steps": [
                    {"added_batch_size": 10},
                    {"added_batch_size": 15},
                ],
            },
            {
                "name": "B",
                "steps": [
                    {"added_batch_size": 20},
                ],
            },
            {
                "name": "C",
                "steps": [
                    {"added_batch_size": 25},
                ],
            },
        ],
    )

    assert starts == [100, 98, 100, 100]
    assert ctrl.solution_manager.best_obj_value == 95
    assert ctrl.solution_manager.get_incumbent().makespan == 95
    assert len(ctrl.solution_manager.registered) == 1
    assert ctrl.solution_manager.registered[0][0].subroutine_name == (
        "neh_cp_branch_portfolio"
    )


def test_neh_cp_branch_portfolio_can_race_original_and_pre_cp_lane(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=5)
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.solution_manager = _RecorderSolutionManager(_FakeSchedule(100, label="prefix"))
    _wire_common_controller(ctrl)

    base_cp_starts = []
    neh_starts = []

    def _fake_base_cp_branch_step(
        self,
        ref_schedule,
        step_config,
        *,
        default_solver_thread_cnt,
    ):
        base_cp_starts.append(
            (
                ref_schedule.makespan,
                step_config["tl_nc_multiplier"],
                default_solver_thread_cnt,
            )
        )
        return SimpleNamespace(
            schedule=_FakeSchedule(94, label="pre_cp"),
            sub_obj_store=None,
            last_obj_value=94,
        )

    class _FakeObjSeries:
        def __init__(self, obj):
            self._obj = obj

        def items(self):
            return [(0.0, float(self._obj))]

    class _FakeObjStore:
        def __init__(self, obj):
            self.obj_value_series = _FakeObjSeries(obj)

    class _FakeResult:
        def __init__(self, obj):
            self.schedule = _FakeSchedule(obj)
            self.sub_obj_store = _FakeObjStore(obj)
            self.last_obj_value = obj

    class _FakeNehConstructor:
        def __init__(self, _ctx):
            pass

        def run(self, ref_schedule, *_args, **_kwargs):
            neh_starts.append(ref_schedule.makespan)
            return _FakeResult(92 if ref_schedule.makespan == 100 else 90)

    monkeypatch.setattr(
        HybridFlowShopCpLnsController,
        "_run_base_cp_branch_step",
        _fake_base_cp_branch_step,
    )
    monkeypatch.setattr(hfs_cp_lns_module, "NehCpConstructor", _FakeNehConstructor)

    ctrl.neh_cp_branch_portfolio(
        solver_thread_cnt=16,
        lanes=[
            {
                "name": "post_dispatch_original",
                "steps": [{"method": "neh_cp", "added_batch_size": 10}],
            },
            {
                "name": "pre_cp_then_neh",
                "steps": [
                    {
                        "method": "solve_base_cp_model",
                        "tl_nc_multiplier": 0.05,
                    },
                    {"method": "neh_cp", "added_batch_size": 10},
                ],
            },
        ],
    )

    assert base_cp_starts == [(100, 0.05, 16)]
    assert neh_starts == [100, 94]
    assert ctrl.solution_manager.best_obj_value == 90
    assert ctrl.solution_manager.get_incumbent().makespan == 90
    assert len(ctrl.solution_manager.registered) == 1


def test_base_cp_branch_step_accepts_tighten_ranges(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=5)
    ctrl.params = SimpleNamespace()
    ctrl.vars = SimpleNamespace(makespan="makespan")
    ctrl.base_cp_model_is_set = False
    ctrl._base_cp_model_options = None

    model_options = []
    solve_calls = []

    class _FakeCpModel:
        def clear_hints(self):
            return None

        def add_hint(self, *_args, **_kwargs):
            return None

        def delete_added_constraints(self):
            return None

    def _set_base_model(**kwargs):
        model_options.append(kwargs)
        ctrl.cp_model = _FakeCpModel()
        ctrl.base_cp_model_is_set = True
        ctrl._base_cp_model_options = {
            "tighten_ranges": bool(kwargs.get("tighten_ranges", False)),
            "link_job_completion": bool(kwargs.get("link_job_completion", False)),
        }

    ctrl.set_cp_model_as_base_cp_model = _set_base_model
    ctrl._resolve_tl_nc_computational_time = (
        lambda *, computational_time, tl_nc_multiplier: 1.5
    )

    def _solve_current(*args, **kwargs):
        solve_calls.append((args, kwargs))
        return (
            SimpleNamespace(
                is_feasible=True,
                obj_bound=90,
                status="FEASIBLE",
            ),
            _FakeSchedule(91),
        )

    ctrl.solve_current_cp_remaining_time_limit = _solve_current
    monkeypatch.setattr(
        hfs_cp_lns_module.BaseModelBuilder,
        "apply_start_hints_from_start_time_map",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        hfs_cp_lns_module.BaseModelBuilder,
        "apply_end_hints_from_end_time_map",
        staticmethod(lambda *_args, **_kwargs: None),
    )

    result = ctrl._run_base_cp_branch_step(
        _FakeSchedule(100),
        {
            "method": "solve_base_cp_model",
            "tl_nc_multiplier": 0.05,
            "solver_thread_cnt": 8,
            "use_lns_only": True,
            "tighten_ranges": True,
            "link_job_completion": True,
        },
        default_solver_thread_cnt=16,
    )

    assert result.schedule.makespan == 91
    assert model_options == [
        {
            "tighten_ranges": True,
            "link_job_completion": True,
        }
    ]
    assert solve_calls
    assert solve_calls[0][0][1] == 8
    assert solve_calls[0][1]["use_lns_only"] is True


def test_neh_cp_branch_portfolio_can_fallback_when_retained_cp_sequence_is_missing(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=5)
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.solution_manager = _RecorderSolutionManager(_FakeSchedule(100, label="prefix"))
    _wire_common_controller(ctrl)
    ctrl._get_last_retained_cp_consensus_job_sequence = lambda: (_ for _ in ()).throw(
        ValueError("missing retained rows")
    )

    seen_kwargs = []

    class _FakeResult:
        def __init__(self):
            self.schedule = _FakeSchedule(99)
            self.sub_obj_store = None
            self.last_obj_value = 99

    class _FakeNehConstructor:
        def __init__(self, _ctx):
            pass

        def run(self, _ref_schedule, *_args, **kwargs):
            seen_kwargs.append(kwargs)
            return _FakeResult()

    monkeypatch.setattr(hfs_cp_lns_module, "NehCpConstructor", _FakeNehConstructor)

    ctrl.neh_cp_branch_portfolio(
        solver_thread_cnt=16,
        lanes=[
            {
                "name": "C",
                "steps": [
                    {
                        "added_batch_size": 15,
                        "job_seq_by_last_retained_cp": True,
                        "retained_cp_sequence_fallback": "bottleneck",
                    },
                ],
            },
        ],
    )

    assert len(seen_kwargs) == 1
    assert seen_kwargs[0]["job_sequence_override"] is None
    assert seen_kwargs[0]["job_seq_by_bottleneck_stage"] is True
    assert ctrl.solution_manager.best_obj_value == 99


def test_suffix_probe_portfolio_selects_probe_winner_and_runs_only_its_suffix(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=5)
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.solution_manager = _RecorderSolutionManager(_FakeSchedule(100, label="prefix"))
    _wire_common_controller(ctrl)

    probe_starts = []
    probe_returns = iter([96, 93, 91])

    class _FakeObjSeries:
        def __init__(self, obj):
            self._obj = obj

        def items(self):
            return [(0.0, float(self._obj))]

    class _FakeObjStore:
        def __init__(self, obj):
            self.obj_value_series = _FakeObjSeries(obj)

    class _FakeResult:
        def __init__(self, obj):
            self.schedule = _FakeSchedule(obj)
            self.sub_obj_store = _FakeObjStore(obj)
            self.last_obj_value = obj

    class _FakeNehConstructor:
        def __init__(self, _ctx):
            pass

        def run(self, ref_schedule, *_args, **_kwargs):
            probe_starts.append(ref_schedule.makespan)
            return _FakeResult(next(probe_returns))

    suffix_calls = []

    def fake_neh_cp(**kwargs):
        suffix_calls.append(("neh_cp", kwargs["added_batch_size"]))
        current = ctrl.solution_manager.get_incumbent()
        next_obj = current.makespan - kwargs["added_batch_size"]
        report = ctrl._make_subroutine_report(
            elapsed_time=0.0,
            obj_value=next_obj,
            obj_bound=None,
            is_init=False,
            subroutine_name="neh_cp",
            progress_obj_value_records=((0.0, next_obj),),
        )
        ctrl.solution_manager.register(report, _FakeSchedule(next_obj))

    def fake_incremental_sw_cp(**kwargs):
        suffix_calls.append(("incremental_sw_cp", kwargs["batch_size_ratio"]))

    monkeypatch.setattr(hfs_cp_lns_module, "NehCpConstructor", _FakeNehConstructor)
    monkeypatch.setattr(ctrl, "neh_cp", fake_neh_cp)
    monkeypatch.setattr(ctrl, "incremental_sw_cp", fake_incremental_sw_cp)

    ctrl.suffix_probe_portfolio(
        solver_thread_cnt=16,
        lanes=[
            {
                "name": "0504_suffix",
                "probe_steps": [
                    {"method": "neh_cp", "added_batch_size": 10},
                ],
                "suffix_steps": [
                    {"method": "neh_cp", "added_batch_size": 20},
                ],
            },
            {
                "name": "0510_suffix",
                "probe_steps": [
                    {"method": "neh_cp", "added_batch_size": 10},
                    {"method": "neh_cp", "added_batch_size": 15},
                ],
                "suffix_steps": [
                    {"method": "neh_cp", "added_batch_size": 30},
                    {"method": "incremental_sw_cp", "batch_size_ratio": 0.09},
                ],
            },
        ],
    )

    assert probe_starts == [100, 100, 93]
    assert ctrl.solution_manager.registered[0][0].subroutine_name == (
        "suffix_probe_portfolio"
    )
    assert ctrl.solution_manager.registered[0][1].makespan == 91
    assert suffix_calls == [("neh_cp", 30), ("incremental_sw_cp", 0.09)]
    assert ctrl.solution_manager.get_incumbent().makespan == 61


def test_suffix_probe_portfolio_can_commit_equal_probe_schedule_for_suffix(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=5)
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.solution_manager = _RecorderSolutionManager(_FakeSchedule(100, label="prefix"))
    _wire_common_controller(ctrl)

    class _FakeResult:
        def __init__(self):
            self.schedule = _FakeSchedule(100, label="equal_lane")
            self.sub_obj_store = None
            self.last_obj_value = 100

    class _FakeNehConstructor:
        def __init__(self, _ctx):
            pass

        def run(self, _ref_schedule, *_args, **_kwargs):
            return _FakeResult()

    suffix_start_labels = []

    def fake_neh_cp(**_kwargs):
        suffix_start_labels.append(ctrl.solution_manager.get_incumbent().label)

    monkeypatch.setattr(hfs_cp_lns_module, "NehCpConstructor", _FakeNehConstructor)
    monkeypatch.setattr(ctrl, "neh_cp", fake_neh_cp)

    ctrl.suffix_probe_portfolio(
        solver_thread_cnt=16,
        lanes=[
            {
                "name": "equal_probe",
                "probe_steps": [
                    {"method": "neh_cp", "added_batch_size": 10},
                ],
                "suffix_steps": [
                    {"method": "neh_cp", "added_batch_size": 20},
                ],
            },
        ],
    )

    assert ctrl.solution_manager.get_incumbent().label == "equal_lane"
    assert suffix_start_labels == ["equal_lane"]


def test_suffix_probe_portfolio_workload_guard_runs_fallback(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=40, stage_count=20)
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.solution_manager = _RecorderSolutionManager(_FakeSchedule(100))
    _wire_common_controller(ctrl)

    suffix_calls = []
    monkeypatch.setattr(
        ctrl,
        "neh_cp",
        lambda **kwargs: suffix_calls.append(kwargs["added_batch_size"]),
    )

    ctrl.suffix_probe_portfolio(
        solver_thread_cnt=16,
        min_workload_size=900,
        lanes=[
            {
                "name": "guarded_lane",
                "probe_steps": [{"method": "neh_cp", "added_batch_size": 10}],
                "suffix_steps": [{"method": "neh_cp", "added_batch_size": 15}],
            },
        ],
        fallback_steps=[
            {"method": "neh_cp", "added_batch_size": 10},
            {"method": "neh_cp", "added_batch_size": 15},
        ],
    )

    assert suffix_calls == [10, 15]


def test_suffix_probe_portfolio_lane_workload_guard_skips_one_lane(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.solution_manager = _RecorderSolutionManager(_FakeSchedule(100, label="prefix"))
    _wire_common_controller(ctrl)

    lane_starts = []

    class _FakeResult:
        def __init__(self, obj):
            self.schedule = _FakeSchedule(obj)
            self.sub_obj_store = None
            self.last_obj_value = obj

    class _FakeNehConstructor:
        def __init__(self, _ctx):
            pass

        def run(self, ref_schedule, *_args, **_kwargs):
            lane_starts.append(ref_schedule.makespan)
            return _FakeResult(91)

    suffix_calls = []
    monkeypatch.setattr(hfs_cp_lns_module, "NehCpConstructor", _FakeNehConstructor)
    monkeypatch.setattr(
        ctrl,
        "neh_cp",
        lambda **kwargs: suffix_calls.append(kwargs["added_batch_size"]),
    )

    ctrl.suffix_probe_portfolio(
        solver_thread_cnt=16,
        lanes=[
            {
                "name": "too_large_min_workload",
                "min_workload_size": 2400,
                "probe_steps": [{"method": "neh_cp", "added_batch_size": 10}],
                "suffix_steps": [{"method": "neh_cp", "added_batch_size": 15}],
            },
            {
                "name": "allowed_lane",
                "min_workload_size": 900,
                "probe_steps": [{"method": "neh_cp", "added_batch_size": 10}],
                "suffix_steps": [{"method": "neh_cp", "added_batch_size": 20}],
            },
        ],
    )

    assert lane_starts == [100]
    assert suffix_calls == [20]


def test_workload_guarded_retained_cp_bottleneck_band_stage_ns_skips_on_small_gap(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(1030), best_obj_bound=1000.0
    )
    calls = []
    monkeypatch.setattr(
        ctrl,
        "retained_cp_bottleneck_band_stage_ns",
        lambda **kwargs: calls.append(kwargs),
    )

    ctrl.workload_guarded_retained_cp_bottleneck_band_stage_ns(
        solver_thread_cnt=16,
        radius=2,
        tl_nc_multiplier=0.01,
        min_incumbent_bound_gap_ratio=0.04,
    )

    assert calls == []


def test_workload_guarded_retained_cp_bottleneck_band_stage_ns_skips_on_restore_loss(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(3400), best_obj_bound=3100.0
    )
    ctrl.last_retained_cp_dispatch_cp_ub = 3200.0
    ctrl.last_retained_cp_post_dispatch_obj = 3400.0
    calls = []
    monkeypatch.setattr(
        ctrl,
        "retained_cp_bottleneck_band_stage_ns",
        lambda **kwargs: calls.append(kwargs),
    )

    ctrl.workload_guarded_retained_cp_bottleneck_band_stage_ns(
        solver_thread_cnt=16,
        radius=2,
        min_incumbent_bound_gap_ratio=0.04,
        max_retained_cp_restore_loss_ratio=0.04,
    )

    assert calls == []


def test_workload_guarded_retained_cp_bottleneck_band_stage_ns_runs_inside_guards(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(1060), best_obj_bound=1000.0
    )
    calls = []
    monkeypatch.setattr(
        ctrl,
        "retained_cp_bottleneck_band_stage_ns",
        lambda **kwargs: calls.append(kwargs),
    )

    ctrl.workload_guarded_retained_cp_bottleneck_band_stage_ns(
        solver_thread_cnt=16,
        radius=2,
        tl_nc_multiplier=0.01,
        min_workload_size=1200,
        min_instance_stage_count=15,
        min_incumbent_bound_gap_ratio=0.04,
        max_retained_cp_restore_loss_ratio=0.06,
        use_lns_only=True,
    )

    assert len(calls) == 1
    assert calls[0]["solver_thread_cnt"] == 16
    assert calls[0]["radius"] == 2
    assert calls[0]["tl_nc_multiplier"] == 0.01
    assert calls[0]["use_lns_only"] is True


def test_workload_guarded_dispatch_from_retained_cp_skips_outside_stage_guard(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(1060), best_obj_bound=1000.0
    )
    calls = []
    monkeypatch.setattr(
        ctrl,
        "dispatch_from_retained_cp",
        lambda **kwargs: calls.append(kwargs),
    )

    ctrl.workload_guarded_dispatch_from_retained_cp(
        min_instance_stage_count=20,
        min_incumbent_bound_gap_ratio=0.04,
        retained_cp_snapshot_top_k=2,
    )

    assert calls == []


def test_workload_guarded_dispatch_from_retained_cp_runs_inside_guards(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=240, stage_count=20)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(6260), best_obj_bound=5735.0
    )
    calls = []
    monkeypatch.setattr(
        ctrl,
        "dispatch_from_retained_cp",
        lambda **kwargs: calls.append(kwargs) or {"ok": True},
    )

    result = ctrl.workload_guarded_dispatch_from_retained_cp(
        min_workload_size=1800,
        min_instance_stage_count=20,
        min_incumbent_bound_gap_ratio=0.06,
        use_retained_cp_snapshot_portfolio=True,
        retained_cp_snapshot_top_k=2,
        include_extended_rank_variants=True,
    )

    assert result == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["use_retained_cp_snapshot_portfolio"] is True
    assert calls[0]["retained_cp_snapshot_top_k"] == 2
    assert calls[0]["include_extended_rank_variants"] is True


def test_workload_guarded_retained_stage_cp_lb_dispatch_skips_when_post_gain_is_large(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(3317), best_obj_bound=3101.0
    )
    ctrl.last_retained_cp_dispatch_obj = 3403.0
    cp_calls = []
    dispatch_calls = []
    monkeypatch.setattr(
        ctrl,
        "apply_retained_stage_cp_lb",
        lambda **kwargs: cp_calls.append(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        ctrl,
        "dispatch_from_retained_cp",
        lambda **kwargs: dispatch_calls.append(kwargs) or {"dispatch": True},
    )

    result = ctrl.workload_guarded_retained_stage_cp_lb_dispatch(
        threads=16,
        tl_nc_multiplier=0.05,
        retained_stage_mode="first_bottleneck_band_last",
        bottleneck_band_radius=2,
        min_instance_stage_count=15,
        max_instance_stage_count=15,
        min_incumbent_bound_gap_ratio=0.06,
        max_post_dispatch_improvement_ratio=0.02,
        run_if_post_dispatch_missing=False,
    )

    assert result is None
    assert cp_calls == []
    assert dispatch_calls == []


def test_workload_guarded_retained_stage_cp_lb_dispatch_runs_inside_guards(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=160, stage_count=15)
    ctrl.solution_manager = _RecorderSolutionManager(
        _FakeSchedule(4177), best_obj_bound=3890.0
    )
    ctrl.last_retained_cp_dispatch_obj = 4230.0
    cp_calls = []
    dispatch_calls = []
    monkeypatch.setattr(
        ctrl,
        "apply_retained_stage_cp_lb",
        lambda **kwargs: cp_calls.append(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        ctrl,
        "dispatch_from_retained_cp",
        lambda **kwargs: dispatch_calls.append(kwargs) or {"dispatch": True},
    )

    result = ctrl.workload_guarded_retained_stage_cp_lb_dispatch(
        threads=16,
        tl_nc_multiplier=0.05,
        retained_stage_mode="first_bottleneck_band_last",
        bottleneck_band_radius=2,
        min_instance_job_count=120,
        min_instance_stage_count=15,
        max_instance_stage_count=15,
        min_incumbent_bound_gap_ratio=0.06,
        max_post_dispatch_improvement_ratio=0.02,
        use_retained_cp_snapshot_portfolio=True,
        retained_cp_snapshot_top_k=3,
        include_extended_rank_variants=True,
    )

    assert result == {"dispatch": True}
    assert len(cp_calls) == 1
    assert cp_calls[0]["threads"] == 16
    assert cp_calls[0]["tl_nc_multiplier"] == 0.05
    assert cp_calls[0]["retained_stage_mode"] == "first_bottleneck_band_last"
    assert cp_calls[0]["bottleneck_band_radius"] == 2
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0]["use_retained_cp_snapshot_portfolio"] is True
    assert dispatch_calls[0]["retained_cp_snapshot_top_k"] == 3
    assert dispatch_calls[0]["include_extended_rank_variants"] is True


def test_initialize_by_dispatch_portfolio_registers_best_candidate(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.solution_manager = _RecorderSolutionManager()

    def fake_get_candidates(**config):
        score = 100
        if config["head_for_all_stages"]:
            score -= 15
        if not config["machine_then_job"]:
            score -= 5
        if "stage_agg_2" in config["method_list"]:
            score -= 10
        if config["mi_agg_method"] == "min":
            score -= 3
        return {"fake": _FakeSchedule(score)}

    monkeypatch.setattr(
        ctrl,
        "_get_selected_dispatch_candidate_schedules",
        fake_get_candidates,
    )

    ctrl.initialize_by_dispatch_portfolio(portfolio="balanced", include_stage_agg=True)

    report, solution = ctrl.solution_manager.registered[-1]
    assert report.subroutine_name == "initialize_by_dispatch_portfolio"
    assert report.is_init is True
    assert solution.makespan == 67
    assert ctrl.last_selected_dispatch_config["mi_agg_method"] == "min"
    assert "stage_agg_2" in ctrl.last_selected_dispatch_config["method_list"]


def test_initialize_by_dispatch_portfolio_can_select_earlier_near_best(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.solution_manager = _RecorderSolutionManager()

    def fake_get_candidates(**config):
        score = 100
        if config["randomize_mid_all"]:
            score = 90 - config.get("_fake_trial_offset", 0)
        if config["head_for_all_stages"]:
            score -= 1
        return {"fake": _FakeSchedule(score)}

    call_idx = {"value": 0}

    call_configs = []

    def fake_get_candidates_with_trial_score(**config):
        call_configs.append(dict(config))
        call_idx["value"] += 1
        if config["randomize_mid_all"]:
            config["_fake_trial_offset"] = call_idx["value"]
        return fake_get_candidates(**config)

    monkeypatch.setattr(
        ctrl,
        "_get_selected_dispatch_candidate_schedules",
        fake_get_candidates_with_trial_score,
    )

    ctrl.initialize_by_dispatch_portfolio(
        portfolio="compact",
        include_stage_agg=False,
        include_mid_order_variants=True,
        randomized_mid_trials=2,
        selection_strategy="earliest_within_slack",
        selection_obj_slack=1,
    )

    report, solution = ctrl.solution_manager.registered[-1]
    assert report.subroutine_name == "initialize_by_dispatch_portfolio"
    assert solution.makespan == 83
    assert ctrl.last_selected_dispatch_config["randomize_mid_all"] is True
    mid_order_calls = [
        config
        for config in call_configs
        if config["randomize_mid_all"]
        or config["reverse_mid_even"]
        or config["reverse_mid_all"]
    ]
    assert mid_order_calls
    assert all(
        config["method_list"] == ["bn2d_all_stages"] for config in mid_order_calls
    )


def test_initialize_by_dispatch_portfolio_can_randomize_non_bn2d_tie_breaks(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.instance = SimpleNamespace(job_id_list=["A", "B", "C"])
    ctrl.solution_manager = _RecorderSolutionManager()
    call_configs = []

    def fake_get_candidates(**config):
        call_configs.append(dict(config))
        if config.get("job_tiebreak_rank"):
            return {"fake": _FakeSchedule(80)}
        return {"fake": _FakeSchedule(100)}

    monkeypatch.setattr(
        ctrl,
        "_get_selected_dispatch_candidate_schedules",
        fake_get_candidates,
    )

    ctrl.initialize_by_dispatch_portfolio(
        portfolio="compact",
        include_stage_agg=False,
        randomized_tiebreak_trials=2,
    )

    random_rank_calls = [
        config for config in call_configs if config.get("job_tiebreak_rank")
    ]
    assert len(random_rank_calls) == 4
    assert all(
        config["method_list"] == ["best_of_mixed_dispatches"]
        for config in random_rank_calls
    )
    assert {config["machine_then_job"] for config in random_rank_calls} == {
        True,
        False,
    }
    assert ctrl.solution_manager.get_incumbent().makespan == 80


def test_post_mip_config_restores_mixed_dispatch_after_bn2d_only_init() -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.last_selected_dispatch_config = {"method_list": ["bn2d_all_stages"]}

    config = ctrl._get_selected_dispatch_config_for_post_mip()

    assert config["method_list"] == ["bn2d_all_stages", "best_of_mixed_dispatches"]


def test_initialize_by_sequence_insertion_portfolio_registers_best_candidate(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.instance = SimpleNamespace(
        job_id_list=["A", "B", "C"],
        stage_id_list=["S1", "S2"],
    )
    ctrl.job_2_stage_2_p_dict = {
        "A": {"S1": 4, "S2": 1},
        "B": {"S1": 3, "S2": 3},
        "C": {"S1": 1, "S2": 5},
    }
    ctrl.solution_manager = _RecorderSolutionManager()

    def fake_score_sequence(sequence):
        rank = {"A": 3, "B": 1, "C": 2}
        return _FakeSchedule(sum((idx + 1) * rank[j] for idx, j in enumerate(sequence)))

    def fake_dispatch_sequence(sequence, *, machine_then_job, head_for_all_stages):
        score = fake_score_sequence(sequence).makespan
        if machine_then_job:
            score -= 2
        if head_for_all_stages:
            score -= 1
        return _FakeSchedule(score)

    monkeypatch.setattr(ctrl, "_from_job_sequence_get_schedule", fake_score_sequence)
    monkeypatch.setattr(
        ctrl,
        "_get_best_mixed_schedule_from_job_sequence",
        fake_dispatch_sequence,
    )

    ctrl.initialize_by_sequence_insertion_portfolio(
        order_rules=["total_desc"],
        beam_width=2,
        max_insert_positions=None,
        machine_then_job_options=[True, False],
        head_for_all_stages_options=[True, False],
    )

    report, solution = ctrl.solution_manager.registered[-1]
    assert report.subroutine_name == "initialize_by_sequence_insertion_portfolio"
    assert report.is_init is True
    assert solution.makespan == 7


def test_critical_schedule_repair_ls_registers_best_repair(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.instance = SimpleNamespace(job_count=10, stage_count=4)
    ctrl.stage_2_job_2_p_dict = {"S1": {"J1": 1}}
    incumbent = HybridFlowshopLiteSchedule(
        jobs=["J1"],
        stages=["S1"],
        machines_per_stage={"S1": ["M1"]},
    )
    incumbent.add_ops_times_2_mc("S1", "M1", "J1", start_time=0, end_time=100)
    ctrl.solution_manager = _RecorderSolutionManager(incumbent)
    ctrl.get_remaining_time_limit = lambda subroutine_time_limit: subroutine_time_limit
    ctrl._resolve_target_stage_ids_for_critical_machine_ls = lambda **_kwargs: None
    ctrl.create_empty_schedule_from_ins = lambda: _FakeSchedule(100)

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.improve_schedule_by_critical_stage_sequence_insertions",
        lambda *_args, **_kwargs: _FakeSchedule(95),
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.improve_schedule_by_critical_cross_machine_insertions",
        lambda *_args, **_kwargs: _FakeSchedule(90),
    )
    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.improve_schedule_by_critical_adjacent_swaps",
        lambda schedule, *_args, **_kwargs: _FakeSchedule(schedule.makespan - 1),
    )

    ctrl.critical_schedule_repair_ls(
        max_rounds=1,
        tl_nc_multiplier=0.5,
        target_stage_mode="all",
    )

    report, solution = ctrl.solution_manager.registered[-1]
    assert report.subroutine_name == "critical_schedule_repair_ls"
    assert solution.makespan == 89


def test_dispatch_from_retained_cp_keeps_better_incumbent(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    obj_log = []
    _wire_common_controller(ctrl)
    ctrl.add_obj_value_log = lambda *args, **kwargs: obj_log.append((args, kwargs))
    incumbent = _FakeSchedule(3159)
    ctrl.solution_manager = _RecorderSolutionManager(incumbent)
    ctrl.instance = SimpleNamespace()
    ctrl.last_retained_cp_lb_retained_solution_rows = [{"stage_id": "i1"}]
    ctrl.last_retained_cp_lb_result = SimpleNamespace(
        objective_ub=2968.0,
        certified_final_lb=2894.0,
    )

    captured_kwargs = {}

    def fake_run_post_retained_cp_dispatch(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            dispatched_schedule=_FakeSchedule(3211),
            selected_dispatch_variant="mixed_cp_consensus",
            dispatched_schedules={"mixed_cp_consensus": _FakeSchedule(3211)},
            variant_2_anchor_stage_ids={"mixed_cp_consensus": []},
        )

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.run_post_retained_cp_dispatch",
        fake_run_post_retained_cp_dispatch,
    )

    result = ctrl.dispatch_from_retained_cp(
        randomized_mixed_rank_trials=7,
        save_cp_dispatch_artifacts=False,
    )

    assert result is not None
    assert captured_kwargs["randomized_mixed_rank_trials"] == 7
    assert result["schedule"] is incumbent
    assert result["selected_dispatch_variant"] == "incumbent_before_retained_cp"
    assert result["post_cp_selected_dispatch_variant"] == "mixed_cp_consensus"
    assert result["post_cp_selected_obj"] == 3211
    assert result["kept_incumbent"] is True
    assert ctrl.solution_manager.get_incumbent() is incumbent
    assert ctrl.last_retained_cp_dispatch_obj == 3159
    assert ctrl.last_retained_cp_post_dispatch_obj == 3211
    assert ctrl.last_retained_cp_dispatch_kept_incumbent is True
    assert set(ctrl.last_retained_cp_dispatch_candidate_schedules) == {
        "mixed_cp_consensus",
        "incumbent_before_retained_cp",
    }
    assert (
        ctrl.last_retained_cp_dispatch_candidate_schedules[
            "incumbent_before_retained_cp"
        ]
        is incumbent
    )
    report, registered_solution = ctrl.solution_manager.registered[-1]
    assert report.obj_value == 3159
    assert registered_solution is incumbent
    assert obj_log[-1][0][1] == 3159


def test_neh_cp_sequence_beam_runs_diverse_candidates_and_registers_best(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    incumbent = _FakeSchedule(100, label="inc", seq=["A", "B", "C", "D"])
    ctrl.solution_manager = _RecorderSolutionManager(incumbent)
    ctrl.instance = SimpleNamespace(job_count=4, stage_count=2, stage_id_list=["S1"])
    ctrl.job_2_stage_2_p_dict = {}
    ctrl.stage_2_job_2_p_dict = {}
    ctrl.last_retained_cp_dispatch_candidate_schedules = {
        "best_same_sequence": _FakeSchedule(95, label="same", seq=["A", "B", "C", "D"]),
        "diverse_candidate": _FakeSchedule(
            98, label="diverse", seq=["D", "C", "B", "A"]
        ),
        "duplicate_but_promising": _FakeSchedule(
            99, label="duplicate", seq=["A", "B", "C", "D"]
        ),
    }
    calls = []

    monkeypatch.setattr(
        ctrl,
        "_get_neh_reference_sequence",
        lambda schedule, **_kwargs: list(schedule.seq),
    )

    class FakeNehConstructor:
        def __init__(self, _ctx) -> None:
            return None

        def run(self, ref_schedule, *_args, **_kwargs):
            calls.append(ref_schedule.label)
            output_by_label = {"same": 90, "diverse": 80, "inc": 100}
            output = _FakeSchedule(
                output_by_label[ref_schedule.label],
                label=f"{ref_schedule.label}_neh",
                seq=ref_schedule.seq,
            )
            return SimpleNamespace(
                schedule=output,
                sub_obj_store=None,
                last_obj_value=output.makespan,
            )

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.NehCpConstructor",
        FakeNehConstructor,
    )

    ctrl.neh_cp_sequence_beam(
        solver_thread_cnt=16,
        candidate_top_k=2,
        candidate_pool_top_k=4,
        min_sequence_position_diff_ratio=0.50,
        include_incumbent_candidate=True,
        added_batch_size=10,
        cp_tl_nc_multiplier=0.05,
    )

    assert calls == ["same", "diverse"]
    assert ctrl.solution_manager.get_incumbent().makespan == 80
    report, solution = ctrl.solution_manager.registered[-1]
    assert report.subroutine_name == "neh_cp_sequence_beam"
    assert solution.label == "diverse_neh"


def test_workload_guarded_neh_cp_sequence_beam_skips_outside_workload_band(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=80, stage_count=15)

    captured = {"called": False}

    def fake_neh_cp_sequence_beam(**_kwargs):
        captured["called"] = True

    monkeypatch.setattr(ctrl, "neh_cp_sequence_beam", fake_neh_cp_sequence_beam)

    ctrl.workload_guarded_neh_cp_sequence_beam(
        solver_thread_cnt=16,
        min_workload_size=1800,
    )

    assert captured["called"] is False


def test_workload_guarded_neh_cp_sequence_beam_runs_inside_workload_band(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(job_count=160, stage_count=15)

    captured = {}

    def fake_neh_cp_sequence_beam(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ctrl, "neh_cp_sequence_beam", fake_neh_cp_sequence_beam)

    ctrl.workload_guarded_neh_cp_sequence_beam(
        solver_thread_cnt=16,
        candidate_top_k=3,
        cp_tl_nc_multiplier=0.02,
        max_instance_job_count=200,
        min_workload_size=2200,
        max_workload_size=2600,
    )

    assert captured["solver_thread_cnt"] == 16
    assert captured["candidate_top_k"] == 3
    assert captured["cp_tl_nc_multiplier"] == 0.02


def test_workload_adaptive_retained_cp_lb_selects_large_topk(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.instance = SimpleNamespace(job_count=120, stage_count=15)
    captured = {}

    def fake_apply_retained_stage_cp_lb(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        ctrl,
        "apply_retained_stage_cp_lb",
        fake_apply_retained_stage_cp_lb,
    )

    result = ctrl.apply_workload_adaptive_retained_stage_cp_lb(
        retained_stage_mode="first_topk_bottlenecks_last",
        small_extra_bottleneck_count=3,
        large_extra_bottleneck_count=4,
        large_workload_threshold=1800,
    )

    assert result == {"ok": True}
    assert captured["extra_bottleneck_count"] == 4
    assert captured["retained_stage_mode"] == "first_topk_bottlenecks_last"


def test_workload_adaptive_retained_cp_lb_can_select_by_stage_count(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    captured = []

    def fake_apply_retained_stage_cp_lb(**kwargs):
        captured.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        ctrl,
        "apply_retained_stage_cp_lb",
        fake_apply_retained_stage_cp_lb,
    )

    ctrl.instance = SimpleNamespace(job_count=160, stage_count=10)
    ctrl.apply_workload_adaptive_retained_stage_cp_lb(
        small_extra_bottleneck_count=3,
        large_extra_bottleneck_count=6,
        adaptive_basis="stage_count",
        large_stage_count_threshold=15,
    )

    ctrl.instance = SimpleNamespace(job_count=40, stage_count=15)
    ctrl.apply_workload_adaptive_retained_stage_cp_lb(
        small_extra_bottleneck_count=3,
        large_extra_bottleneck_count=6,
        adaptive_basis="stage_count",
        large_stage_count_threshold=15,
    )

    assert [call["extra_bottleneck_count"] for call in captured] == [3, 6]


def test_workload_adaptive_retained_cp_lb_can_select_by_workload_tier(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    captured = []

    def fake_apply_retained_stage_cp_lb(**kwargs):
        captured.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        ctrl,
        "apply_retained_stage_cp_lb",
        fake_apply_retained_stage_cp_lb,
    )

    for job_count, stage_count in [(40, 20), (80, 20), (120, 15)]:
        ctrl.instance = SimpleNamespace(job_count=job_count, stage_count=stage_count)
        ctrl.apply_workload_adaptive_retained_stage_cp_lb(
            adaptive_basis="workload_tier",
            workload_tier_thresholds=[1200, 1800],
            workload_tier_extra_bottleneck_counts=[3, 5, 6],
        )

    assert [call["extra_bottleneck_count"] for call in captured] == [3, 5, 6]


def test_adaptive_neh_preserved_head_selects_small_workload(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.instance = SimpleNamespace(job_count=80, stage_count=10)
    captured = {}

    def fake_neh_cp(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ctrl, "neh_cp", fake_neh_cp)

    ctrl.neh_cp_adaptive_preserved_head(
        solver_thread_cnt=16,
        added_batch_size=20,
        job_seq_by_bottleneck_stage=True,
        small_preserved_head_job_portion=0.25,
        large_preserved_head_job_portion=0.40,
        large_workload_threshold=1800,
    )

    assert captured["preserved_head_job_portion"] == 0.25
    assert captured["job_seq_by_bottleneck_stage"] is True


def test_dispatch_from_retained_cp_can_choose_earliest_snapshot_within_slack(
    monkeypatch,
) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    _wire_common_controller(ctrl)
    ctrl.solution_manager = _RecorderSolutionManager()
    ctrl.instance = SimpleNamespace()
    ctrl.last_retained_cp_lb_retained_solution_rows = [{"source": "final"}]
    ctrl.last_retained_cp_lb_result = _make_retained_cp_result(100.0)
    ctrl.last_retained_cp_lb_solution_snapshots = [
        {
            "objective_ub": 101.0,
            "objective_lb": 10.0,
            "runtime_sec": 0.1,
            "retained_solution_rows": [{"source": "snapshot1"}],
        },
        {
            "objective_ub": 102.0,
            "objective_lb": 10.0,
            "runtime_sec": 0.2,
            "retained_solution_rows": [{"source": "snapshot2"}],
        },
        {
            "objective_ub": 103.0,
            "objective_lb": 10.0,
            "runtime_sec": 0.3,
            "retained_solution_rows": [{"source": "snapshot3"}],
        },
    ]
    ub_2_makespan = {100.0: 108, 101.0: 107, 102.0: 104, 103.0: 100}

    def fake_run_post_retained_cp_dispatch(**kwargs):
        objective_ub = float(kwargs["retained_cp_result"].objective_ub)
        variant = f"variant_{objective_ub:g}"
        schedule = _FakeSchedule(ub_2_makespan[objective_ub])
        return SimpleNamespace(
            dispatched_schedule=schedule,
            selected_dispatch_variant=variant,
            dispatched_schedules={variant: schedule},
            variant_2_anchor_stage_ids={variant: []},
        )

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.run_post_retained_cp_dispatch",
        fake_run_post_retained_cp_dispatch,
    )

    result = ctrl.dispatch_from_retained_cp(
        use_retained_cp_snapshot_portfolio=True,
        retained_cp_snapshot_top_k=3,
        retained_cp_dispatch_selection_strategy=(
            "earliest_snapshot_within_makespan_slack"
        ),
        retained_cp_dispatch_makespan_slack=4,
        save_cp_dispatch_artifacts=False,
    )

    assert result is not None
    assert result["schedule"].makespan == 104
    assert result["selected_dispatch_source"] == "snapshot_2_ub_102"
    assert result["selected_dispatch_variant"] == "snapshot_2_ub_102:variant_102"


def test_best_of_mixed_dispatches_can_filter_sequence_methods(monkeypatch) -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    calls = []

    def make_dispatch_method(name: str, makespan: int):
        def dispatch_method(**_kwargs):
            calls.append(name)
            return _FakeSchedule(makespan)

        dispatch_method.__name__ = f"_get_schedule_by_{name}"
        return dispatch_method

    monkeypatch.setattr(
        ctrl,
        "_get_schedule_by_cds",
        make_dispatch_method("cds", 100),
    )
    monkeypatch.setattr(
        ctrl,
        "_get_schedule_by_gupta",
        make_dispatch_method("gupta", 80),
    )
    monkeypatch.setattr(
        ctrl,
        "_get_schedule_by_palmer",
        make_dispatch_method("palmer", 90),
    )

    schedule = ctrl._get_schedule_by_best_of_mixed_dispatches(
        mixed_dispatch_methods=["cds", "palmer"],
    )

    assert calls == ["cds", "palmer"]
    assert schedule is not None
    assert schedule.makespan == 90
