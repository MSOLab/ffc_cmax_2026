from types import SimpleNamespace

import pytest

from hybridflowshop.controller import HybridFlowShopCpLnsController


def _make_controller() -> HybridFlowShopCpLnsController:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = SimpleNamespace(
        stage_id_list=["i0", "i1", "i2", "i3", "i4", "i5"],
    )
    ctrl.last_retained_cp_lb_result = SimpleNamespace(
        retained_stage_ids=("i0", "i1", "i3", "i4", "i5"),
        retained_stage_mode="first_topk_bottlenecks_last",
        bottleneck_stage_id="i3",
        selected_bottleneck_stage_ids=("i3", "i1"),
        bottleneck_band_radius=None,
    )
    return ctrl


def test_resolve_retained_completion_stage_ids_supports_anchor_scopes() -> None:
    ctrl = _make_controller()

    assert ctrl._resolve_retained_completion_stage_ids(
        retained_stage_scope="all",
        retained_stage_ids=None,
    ) == ["i0", "i1", "i3", "i4", "i5"]
    assert ctrl._resolve_retained_completion_stage_ids(
        retained_stage_scope="preferred_anchor",
        retained_stage_ids=None,
    ) == ["i3", "i4"]
    assert ctrl._resolve_retained_completion_stage_ids(
        retained_stage_scope="bottlenecks",
        retained_stage_ids=None,
    ) == ["i3", "i1"]
    assert ctrl._resolve_retained_completion_stage_ids(
        retained_stage_scope="first_last",
        retained_stage_ids=None,
    ) == ["i0", "i5"]
    assert ctrl._resolve_retained_completion_stage_ids(
        retained_stage_scope="first_bottlenecks_last",
        retained_stage_ids=None,
    ) == ["i0", "i3", "i1", "i5"]


def test_resolve_retained_completion_stage_ids_validates_explicit_ids() -> None:
    ctrl = _make_controller()

    assert ctrl._resolve_retained_completion_stage_ids(
        retained_stage_scope="all",
        retained_stage_ids=["i2", "i4"],
    ) == ["i2", "i4"]

    with pytest.raises(ValueError, match="Unknown retained_stage_ids"):
        ctrl._resolve_retained_completion_stage_ids(
            retained_stage_scope="all",
            retained_stage_ids=["missing"],
        )


def test_retained_cp_integer_completion_ladder_runs_hard_and_hint_steps() -> None:
    ctrl = object.__new__(HybridFlowShopCpLnsController)
    ctrl.last_retained_cp_lb_retained_solution_rows = [
        {"stage_id": "i0", "job_id": "j0", "start": 1, "end": 2}
    ]
    ctrl.solution_manager = SimpleNamespace(best_obj_value=100.0)
    ctrl.is_stopping_condition = lambda *args, **kwargs: False
    ctrl.base_cp_model_is_set = True
    cleanup_calls = []
    ctrl.cp_model = SimpleNamespace(
        delete_added_constraints=lambda: cleanup_calls.append("delete"),
        clear_hints=lambda: cleanup_calls.append("clear_hints"),
    )

    calls = []

    def complete_from_retained_cp(**kwargs):
        calls.append(("complete", kwargs))
        ctrl.solution_manager.best_obj_value = 98.0

    def solve_base_cp_model_with_retained_hint(**kwargs):
        calls.append(("hint", kwargs))
        ctrl.solution_manager.best_obj_value = 97.0

    ctrl.complete_from_retained_cp = complete_from_retained_cp
    ctrl.solve_base_cp_model_with_retained_hint = solve_base_cp_model_with_retained_hint

    ctrl.retained_cp_integer_completion_ladder(
        solver_thread_cnt=16,
        steps=[
            {
                "name": "hard_probe",
                "mode": "complete",
                "retained_stage_scope": "preferred_anchor",
                "time_slack": 4,
            },
            {
                "name": "soft_probe",
                "mode": "hint",
                "retained_stage_scope": "all",
                "prefer_incumbent_hints": False,
            },
        ],
    )

    assert [call[0] for call in calls] == ["complete", "hint"]
    assert calls[0][1]["solver_thread_cnt"] == 16
    assert calls[0][1]["retained_stage_scope"] == "preferred_anchor"
    assert calls[0][1]["time_slack"] == 4
    assert calls[1][1]["prefer_incumbent_hints"] is False
    assert calls[1][1]["computational_time"] is None
    assert ctrl.solution_manager.best_obj_value == 97.0
    assert cleanup_calls == ["delete", "clear_hints"]
