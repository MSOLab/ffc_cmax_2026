import re

import pytest

from hybridflowshop.controller.hfs_cp_lns import HybridFlowShopCpLnsController


class FakeSchedule:
    def __init__(self, makespan: int, reversed_result=None):
        self.makespan = makespan
        self._reversed_result = reversed_result if reversed_result is not None else self
        self.make_semi_active_calls = []

    def as_reversed(self):
        return self._reversed_result

    def make_semi_active(self, *args, **kwargs):
        self.make_semi_active_calls.append((args, kwargs))


def _make_controller() -> HybridFlowShopCpLnsController:
    ctrl = HybridFlowShopCpLnsController.__new__(HybridFlowShopCpLnsController)
    ctrl.instance = "orig_instance"
    ctrl.get_file_path_for_subroutine = lambda *_args, **_kwargs: "dummy-path"
    ctrl.stage_2_job_2_p_dict = {"s1": {"j1": 1}}  # Dummy duration dict
    return ctrl


def _patch_mixed_dispatcher(
    monkeypatch, module, method_name: str, orig_schedule, rev_schedule
):
    call_log = []

    class FakeMixedDispatcher:
        def __init__(self, instance):
            self.instance = instance

    def _method(self, **kwargs):
        call_log.append((self.instance, kwargs))
        if self.instance == "rev_instance":
            return rev_schedule
        return orig_schedule

    setattr(FakeMixedDispatcher, method_name, _method)
    monkeypatch.setattr(module, "MixedDispatcher", FakeMixedDispatcher)
    return call_log


def _patch_machine_dispatcher(
    monkeypatch, module, method_name: str, orig_schedule, rev_schedule
):
    class FakeMachineDispatcher:
        def __init__(self, instance):
            self.instance = instance

    def _method(self):
        if self.instance == "rev_instance":
            return rev_schedule
        return orig_schedule

    setattr(FakeMachineDispatcher, method_name, _method)
    monkeypatch.setattr(module, "MachineDispatcher", FakeMachineDispatcher)


def _patch_bn2d_dispatcher(monkeypatch, module, orig_schedule, rev_schedule):
    call_log = []

    class FakeBN2DDispatcher:
        def __init__(self, instance):
            self.instance = instance

        def get_schedule_by_bn2d_all_stages(self, **kwargs):
            call_log.append((self.instance, kwargs))
            if self.instance == "rev_instance":
                return rev_schedule
            return orig_schedule

    monkeypatch.setattr(module, "BN2DDispatcher", FakeBN2DDispatcher)
    return call_log


@pytest.mark.parametrize(
    "method_name,dispatcher_method",
    [
        ("_get_schedule_by_gupta", "get_schedule_by_gupta"),
        ("_get_schedule_by_palmer", "get_schedule_by_palmer"),
    ],
)
def test_mixed_methods_choose_reversed_when_better(
    monkeypatch, method_name, dispatcher_method
):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = _make_controller()
    expected = FakeSchedule(10)
    reversed_schedule = FakeSchedule(10, reversed_result=expected)
    orig_schedule = FakeSchedule(20)

    call_log = _patch_mixed_dispatcher(
        monkeypatch,
        module,
        dispatcher_method,
        orig_schedule=orig_schedule,
        rev_schedule=reversed_schedule,
    )
    monkeypatch.setattr(
        module,
        "reverse_stages",
        lambda instance: "rev_instance",
    )

    method = getattr(ctrl, method_name)
    result = method(
        machine_then_job=True,
        head_for_all_stages=True,
        use_palmer_index=True,
        draw_gantt_per_step=True,
    )

    assert result is expected
    assert len(call_log) == 2
    assert call_log[0][0] == "orig_instance"
    assert call_log[1][0] == "rev_instance"
    assert call_log[0][1] == call_log[1][1]
    assert call_log[0][1]["machine_then_job"] is True
    assert call_log[0][1]["head_for_all_stages"] is True
    assert call_log[0][1]["use_palmer_index"] is True
    assert call_log[0][1]["draw_gantt_per_step"] is True
    assert (
        call_log[0][1]["get_file_path_for_subroutine"]
        is ctrl.get_file_path_for_subroutine
    )


@pytest.mark.parametrize(
    "method_name,dispatcher_method",
    [
        ("_get_schedule_by_gupta", "get_schedule_by_gupta"),
        ("_get_schedule_by_palmer", "get_schedule_by_palmer"),
    ],
)
def test_mixed_methods_keep_original_on_tie(
    monkeypatch, method_name, dispatcher_method
):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = _make_controller()
    orig_schedule = FakeSchedule(15)
    rev_schedule = FakeSchedule(15, reversed_result=FakeSchedule(99))

    _patch_mixed_dispatcher(
        monkeypatch,
        module,
        dispatcher_method,
        orig_schedule=orig_schedule,
        rev_schedule=rev_schedule,
    )
    monkeypatch.setattr(
        module,
        "reverse_stages",
        lambda instance: "rev_instance",
    )

    method = getattr(ctrl, method_name)
    result = method()

    assert result is orig_schedule


def test_bn2d_all_stages_choose_reversed_when_better(monkeypatch):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = _make_controller()
    option = object()
    expected = FakeSchedule(10)
    reversed_schedule = FakeSchedule(10, reversed_result=expected)
    orig_schedule = FakeSchedule(20)

    call_log = _patch_bn2d_dispatcher(
        monkeypatch,
        module,
        orig_schedule=orig_schedule,
        rev_schedule=reversed_schedule,
    )
    monkeypatch.setattr(module, "reverse_stages", lambda instance: "rev_instance")

    result = ctrl._get_schedule_by_bn2d_all_stages(option=option, draw_gantt=False)

    assert result is expected
    assert len(call_log) == 2
    assert call_log[0][0] == "orig_instance"
    assert call_log[1][0] == "rev_instance"
    assert call_log[0][1] == call_log[1][1]
    assert call_log[0][1]["option"] is option
    assert call_log[0][1]["gantt_draw_func"] is None
    assert expected.make_semi_active_calls == [((ctrl.stage_2_job_2_p_dict,), {})]


def test_bn2d_all_stages_keep_original_on_tie(monkeypatch):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = _make_controller()
    option = object()
    orig_schedule = FakeSchedule(15)
    rev_schedule = FakeSchedule(15, reversed_result=FakeSchedule(99))

    _patch_bn2d_dispatcher(
        monkeypatch,
        module,
        orig_schedule=orig_schedule,
        rev_schedule=rev_schedule,
    )
    monkeypatch.setattr(module, "reverse_stages", lambda instance: "rev_instance")

    result = ctrl._get_schedule_by_bn2d_all_stages(option=option, draw_gantt=False)

    assert result is orig_schedule


@pytest.mark.parametrize(
    "method_name,dispatcher_method",
    [
        ("_get_schedule_by_dm_cds", "get_schedule_by_cds"),
        ("_get_schedule_by_dm_gupta", "get_schedule_by_gupta"),
        ("_get_schedule_by_dm_palmer", "get_schedule_by_palmer"),
    ],
)
def test_dm_methods_choose_reversed_when_better(
    monkeypatch, method_name, dispatcher_method
):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = _make_controller()
    expected = FakeSchedule(7)
    rev_schedule = FakeSchedule(7, reversed_result=expected)
    orig_schedule = FakeSchedule(11)

    _patch_machine_dispatcher(
        monkeypatch,
        module,
        dispatcher_method,
        orig_schedule=orig_schedule,
        rev_schedule=rev_schedule,
    )
    monkeypatch.setattr(
        module,
        "reverse_stages",
        lambda instance: "rev_instance",
    )

    method = getattr(ctrl, method_name)
    result = method()

    assert result is expected


@pytest.mark.parametrize(
    "method_name,dispatcher_method,expected_error",
    [
        (
            "_get_schedule_by_dm_cds",
            "get_schedule_by_cds",
            "No schedule found after applying DM(CDS).",
        ),
        (
            "_get_schedule_by_dm_gupta",
            "get_schedule_by_gupta",
            "No schedule found after applying DM(Gupta).",
        ),
        (
            "_get_schedule_by_dm_palmer",
            "get_schedule_by_palmer",
            "No schedule found after applying DM(Palmer).",
        ),
    ],
)
def test_dm_methods_raise_when_both_none(
    monkeypatch, method_name, dispatcher_method, expected_error
):
    import hybridflowshop.controller.hfs_cp_lns as module

    ctrl = _make_controller()

    _patch_machine_dispatcher(
        monkeypatch,
        module,
        dispatcher_method,
        orig_schedule=None,
        rev_schedule=None,
    )
    monkeypatch.setattr(
        module,
        "reverse_stages",
        lambda instance: "rev_instance",
    )

    method = getattr(ctrl, method_name)
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        method()
