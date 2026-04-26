from types import SimpleNamespace

from lb_bucket.cp.search import RetainedStageCpResult

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class _FakeSchedule:
    def __init__(self, makespan: int) -> None:
        self.makespan = makespan

    def get_jik_2_start_time_map(self):
        return {}

    def make_semi_active(self, *_args, **_kwargs) -> None:
        return None


class _RecorderSolutionManager:
    def __init__(self, incumbent=None) -> None:
        self.incumbent = incumbent
        self.best_obj_value = (
            float(incumbent.makespan) if incumbent is not None else None
        )
        self.registered = []

    def get_incumbent(self):
        return self.incumbent

    def _a_is_better_obj_value(self, value_a, value_b):
        return value_b is None or value_a < value_b

    def register(self, report, solution):
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
        return False


def _wire_common_controller(ctrl: HybridFlowShopCpLnsController) -> None:
    ctrl.timer = SimpleNamespace(elapsed_sec=0.0)
    ctrl.obj_store = SimpleNamespace(
        add_last_timestamp_note=lambda *args, **kwargs: None,
    )
    ctrl.add_obj_value_log = lambda *args, **kwargs: None
    ctrl._get_call_context_of_current_method = lambda: "test-call"
    ctrl._make_subroutine_report = lambda **kwargs: SimpleNamespace(**kwargs)


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

    def fake_get_candidates_with_trial_score(**config):
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

    monkeypatch.setattr(
        "hybridflowshop.controller.hfs_cp_lns.run_post_retained_cp_dispatch",
        lambda **_kwargs: SimpleNamespace(
            dispatched_schedule=_FakeSchedule(3211),
            selected_dispatch_variant="mixed_cp_consensus",
            dispatched_schedules={"mixed_cp_consensus": _FakeSchedule(3211)},
            variant_2_anchor_stage_ids={"mixed_cp_consensus": []},
        ),
    )

    result = ctrl.dispatch_from_retained_cp(save_cp_dispatch_artifacts=False)

    assert result is not None
    assert result["schedule"] is incumbent
    assert result["selected_dispatch_variant"] == "incumbent_before_retained_cp"
    assert result["post_cp_selected_dispatch_variant"] == "mixed_cp_consensus"
    assert result["post_cp_selected_obj"] == 3211
    assert result["kept_incumbent"] is True
    assert ctrl.solution_manager.get_incumbent() is incumbent
    assert ctrl.last_retained_cp_dispatch_obj == 3159
    assert ctrl.last_retained_cp_post_dispatch_obj == 3211
    assert ctrl.last_retained_cp_dispatch_kept_incumbent is True
    report, registered_solution = ctrl.solution_manager.registered[-1]
    assert report.obj_value == 3159
    assert registered_solution is incumbent
    assert obj_log[-1][0][1] == 3159


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
