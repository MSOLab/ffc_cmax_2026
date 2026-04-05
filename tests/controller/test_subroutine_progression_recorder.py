from unittest.mock import MagicMock

from hybridflowshop.controller.controller_core import HybridFlowShopCpLnsControllerCore


class TestSubroutineProgressionRecorder:
    def test_start_subroutine_call_increments_counter(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._call_counter = 0
        controller._active_call_index = None
        controller._active_subroutine_name = None
        controller._active_call_global_start = None
        controller._subroutine_call_progress_map = {}
        controller._subroutine_call_meta_list = []
        controller.timer = MagicMock()
        controller.timer.elapsed_sec = 10.0

        HybridFlowShopCpLnsControllerCore._start_subroutine_call(
            controller, "test_method"
        )

        assert controller._call_counter == 1
        assert controller._active_call_index == 1
        assert controller._active_subroutine_name == "test_method"
        assert controller._active_call_global_start == 10.0
        assert "1-test_method" in controller._subroutine_call_progress_map
        assert len(controller._subroutine_call_meta_list) == 1
        meta = controller._subroutine_call_meta_list[0]
        assert meta["call_index"] == 1
        assert meta["subroutine_name"] == "test_method"
        assert meta["global_start_sec"] == 10.0

    def test_end_subroutine_call_creates_marker(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._call_counter = 1
        controller._active_call_index = 1
        controller._active_subroutine_name = "test_method"
        controller._active_call_global_start = 5.0
        controller._subroutine_call_progress_map = {
            "1-test_method": [
                {
                    "global_sec": 7.0,
                    "obj_value": 100.0,
                    "call_index": 1,
                    "prefixed_subroutine_name": "1-test_method",
                    "local_sec": 2.0,
                }
            ]
        }
        controller._subroutine_call_meta_list = [
            {
                "call_index": 1,
                "subroutine_name": "test_method",
                "prefixed_subroutine_name": "1-test_method",
                "global_start_sec": 5.0,
            }
        ]
        controller._subroutine_end_marker_list = []
        controller._combined_progress_list = []
        controller.timer = MagicMock()
        controller.timer.elapsed_sec = 15.0

        HybridFlowShopCpLnsControllerCore._end_subroutine_call(
            controller, "test_method"
        )

        assert len(controller._subroutine_end_marker_list) == 1
        marker = controller._subroutine_end_marker_list[0]
        assert marker["call_index"] == 1
        assert marker["subroutine_name"] == "test_method"
        assert marker["global_end_sec"] == 15.0
        assert controller._active_call_index is None
        meta = controller._subroutine_call_meta_list[0]
        assert meta["global_end_sec"] == 15.0
        assert meta["elapsed_sec"] == 10.0

    def test_record_objective_point_adds_to_active_call(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._active_call_index = 2
        controller._active_subroutine_name = "repeat"
        controller._active_call_global_start = 10.0
        controller._subroutine_call_progress_map = {"2-repeat": []}
        controller._combined_progress_list = []

        HybridFlowShopCpLnsControllerCore._record_objective_point(
            controller, 12.0, 95.0
        )

        assert len(controller._subroutine_call_progress_map["2-repeat"]) == 1
        point = controller._subroutine_call_progress_map["2-repeat"][0]
        assert point["global_sec"] == 12.0
        assert point["obj_value"] == 95.0
        assert point["local_sec"] == 2.0
        assert len(controller._combined_progress_list) == 1

    def test_record_objective_point_ignores_when_no_active_call(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._active_call_index = None
        controller._subroutine_call_progress_map = {}
        controller._combined_progress_list = []

        HybridFlowShopCpLnsControllerCore._record_objective_point(
            controller, 5.0, 100.0
        )

        assert len(controller._combined_progress_list) == 0

    def test_get_progression_data_returns_expected_structure(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._subroutine_call_progress_map = {
            "1-init": [
                {
                    "global_sec": 1.0,
                    "obj_value": 110.0,
                    "call_index": 1,
                    "prefixed_subroutine_name": "1-init",
                    "local_sec": 1.0,
                }
            ],
            "2-repeat": [
                {
                    "global_sec": 5.0,
                    "obj_value": 100.0,
                    "call_index": 2,
                    "prefixed_subroutine_name": "2-repeat",
                    "local_sec": 2.0,
                }
            ],
        }
        controller._subroutine_call_meta_list = [
            {
                "call_index": 1,
                "subroutine_name": "init",
                "prefixed_subroutine_name": "1-init",
                "global_start_sec": 0.0,
                "global_end_sec": 2.0,
                "elapsed_sec": 2.0,
            },
            {
                "call_index": 2,
                "subroutine_name": "repeat",
                "prefixed_subroutine_name": "2-repeat",
                "global_start_sec": 3.0,
                "global_end_sec": 7.0,
                "elapsed_sec": 4.0,
            },
        ]
        controller._combined_progress_list = [
            {
                "global_sec": 1.0,
                "obj_value": 110.0,
                "call_index": 1,
                "prefixed_subroutine_name": "1-init",
                "local_sec": 1.0,
            },
            {
                "global_sec": 5.0,
                "obj_value": 100.0,
                "call_index": 2,
                "prefixed_subroutine_name": "2-repeat",
                "local_sec": 2.0,
            },
        ]
        controller._subroutine_end_marker_list = [
            {
                "global_end_sec": 2.0,
                "call_index": 1,
                "prefixed_subroutine_name": "1-init",
                "subroutine_name": "init",
            },
            {
                "global_end_sec": 7.0,
                "call_index": 2,
                "prefixed_subroutine_name": "2-repeat",
                "subroutine_name": "repeat",
            },
        ]
        controller.instance = MagicMock()
        controller.instance.name = "test_instance"
        controller.stopping_criteria = MagicMock()
        controller.stopping_criteria.timelimit = 50.0

        data = HybridFlowShopCpLnsControllerCore.get_progression_data(controller)

        assert data["artifact_version"] == 1
        assert data["instance_id"] == "test_instance"
        assert data["timelimit_sec"] == 50.0
        assert len(data["subroutine_calls"]) == 2
        assert len(data["combined_progress_list"]) == 2
        assert len(data["subroutine_end_marker_list"]) == 2
        assert data["subroutine_calls"][0]["call_index"] == 1
        assert data["subroutine_calls"][0]["global_start_sec"] == 0.0
        assert data["subroutine_calls"][0]["global_end_sec"] == 2.0
        assert data["subroutine_calls"][0]["elapsed_sec"] == 2.0
        assert data["subroutine_calls"][1]["call_index"] == 2
        assert data["subroutine_calls"][1]["global_start_sec"] == 3.0
        assert data["subroutine_calls"][1]["elapsed_sec"] == 4.0
