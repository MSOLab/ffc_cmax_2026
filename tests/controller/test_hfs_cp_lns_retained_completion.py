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
