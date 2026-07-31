import datetime
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from hybridflowshop.controller.controller_core import HybridFlowShopCpLnsControllerCore
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


def _make_progression_controller() -> HybridFlowShopCpLnsControllerCore:
    controller = HybridFlowShopCpLnsControllerCore.__new__(
        HybridFlowShopCpLnsControllerCore
    )
    controller._subroutine_call_progress_map = {}
    controller._subroutine_call_meta_list = []
    controller._combined_progress_list = []
    controller._subroutine_end_marker_list = []
    controller._method_context_meta_map = {}
    controller.solution_manager = SimpleNamespace(history=[])
    controller.instance = SimpleNamespace(name="test_instance")
    controller.stopping_criteria = SimpleNamespace(timelimit=50.0)
    return controller


def _make_checkpoint_schedule() -> HybridFlowshopLiteSchedule:
    schedule = HybridFlowshopLiteSchedule(
        jobs=["j1"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1"]},
    )
    schedule.add_ops_times_2_mc("s1", "m1", "j1", 0, 3)
    return schedule


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

    def test_step_checkpoint_writes_resume_compatible_files(self, tmp_path) -> None:
        controller = _make_progression_controller()
        schedule = _make_checkpoint_schedule()
        controller._working_dir_path = tmp_path / "scenario" / "ins1"
        controller._working_dir_path.mkdir(parents=True)
        controller.solution_manager = SimpleNamespace(
            get_incumbent=lambda: schedule,
            best_obj_value=3.0,
            best_obj_bound=1.0,
        )
        controller.obj_store = SimpleNamespace(
            save_yaml=lambda path, encoding="utf-8": path.write_text(
                "obj_value: {}\n",
                encoding=encoding,
            )
        )
        controller.timer = SimpleNamespace(elapsed_sec=12.5)
        controller.instance = SimpleNamespace(name="ins1")
        controller.save_step_checkpoints_enabled = True
        controller._subroutine_flow = [
            {"method": "set_random_seed"},
            {"method": "neh_cp"},
            {"method": "incremental_sw_cp"},
        ]
        controller.stopping_criteria = {"timelimit": 100}

        controller._try_save_step_checkpoint("2-neh_cp", "neh_cp")

        checkpoint_dir = (
            tmp_path / "scenario" / "checkpoints" / "2-neh_cp" / "ins1" / "results"
        )
        assert (checkpoint_dir / "ins1_solution.yaml").is_file()
        assert (checkpoint_dir / "ins1_obj_log.yaml").is_file()
        summary_path = checkpoint_dir / "ins1_summary.csv"
        assert summary_path.is_file()
        assert "bestObj" in summary_path.read_text(encoding="utf-8")
        assert (
            tmp_path / "scenario" / "checkpoints" / "2-neh_cp" / "subroutine_flow.yaml"
        ).is_file()

    def test_step_checkpoint_skips_nested_contexts(self, tmp_path) -> None:
        controller = _make_progression_controller()
        controller._working_dir_path = tmp_path / "scenario" / "ins1"
        controller._working_dir_path.mkdir(parents=True)
        controller.solution_manager = SimpleNamespace(
            get_incumbent=_make_checkpoint_schedule,
            best_obj_value=3.0,
            best_obj_bound=1.0,
        )
        controller.obj_store = SimpleNamespace(save_yaml=lambda *_args, **_kwargs: None)
        controller.timer = SimpleNamespace(elapsed_sec=12.5)
        controller.instance = SimpleNamespace(name="ins1")
        controller.save_step_checkpoints_enabled = True

        controller._try_save_step_checkpoint("14-incremental_sw_cp.1-sw_cp", "sw_cp")

        assert not (tmp_path / "scenario" / "checkpoints").exists()

    def test_record_objective_point_skips_equal_value_in_same_call(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._active_call_index = 2
        controller._active_subroutine_name = "repeat"
        controller._active_call_global_start = 10.0
        controller._subroutine_call_progress_map = {
            "2-repeat": [
                {
                    "global_sec": 12.0,
                    "obj_value": 95.0,
                    "call_index": 2,
                    "prefixed_subroutine_name": "2-repeat",
                    "local_sec": 2.0,
                }
            ]
        }
        controller._combined_progress_list = list(
            controller._subroutine_call_progress_map["2-repeat"]
        )

        HybridFlowShopCpLnsControllerCore._record_objective_point(
            controller, 13.0, 95.0, is_maximize=False
        )

        assert len(controller._subroutine_call_progress_map["2-repeat"]) == 1
        assert len(controller._combined_progress_list) == 1

    def test_record_objective_point_skips_worse_value_for_minimize(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._active_call_index = 2
        controller._active_subroutine_name = "repeat"
        controller._active_call_global_start = 10.0
        controller._subroutine_call_progress_map = {
            "2-repeat": [
                {
                    "global_sec": 12.0,
                    "obj_value": 95.0,
                    "call_index": 2,
                    "prefixed_subroutine_name": "2-repeat",
                    "local_sec": 2.0,
                }
            ]
        }
        controller._combined_progress_list = list(
            controller._subroutine_call_progress_map["2-repeat"]
        )

        HybridFlowShopCpLnsControllerCore._record_objective_point(
            controller, 13.0, 96.0, is_maximize=False
        )

        assert len(controller._subroutine_call_progress_map["2-repeat"]) == 1
        assert len(controller._combined_progress_list) == 1

    def test_record_objective_point_adds_strict_improvement_for_minimize(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._active_call_index = 2
        controller._active_subroutine_name = "repeat"
        controller._active_call_global_start = 10.0
        controller._subroutine_call_progress_map = {
            "2-repeat": [
                {
                    "global_sec": 12.0,
                    "obj_value": 95.0,
                    "call_index": 2,
                    "prefixed_subroutine_name": "2-repeat",
                    "local_sec": 2.0,
                }
            ]
        }
        controller._combined_progress_list = list(
            controller._subroutine_call_progress_map["2-repeat"]
        )

        HybridFlowShopCpLnsControllerCore._record_objective_point(
            controller, 14.0, 94.0, is_maximize=False
        )

        assert len(controller._subroutine_call_progress_map["2-repeat"]) == 2
        point = controller._subroutine_call_progress_map["2-repeat"][-1]
        assert point["global_sec"] == 14.0
        assert point["obj_value"] == 94.0
        assert point["local_sec"] == 4.0
        assert len(controller._combined_progress_list) == 2

    def test_record_objective_point_treats_new_call_first_point_as_recordable(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._active_call_index = 3
        controller._active_subroutine_name = "repeat"
        controller._active_call_global_start = 20.0
        controller._subroutine_call_progress_map = {
            "2-repeat": [
                {
                    "global_sec": 14.0,
                    "obj_value": 94.0,
                    "call_index": 2,
                    "prefixed_subroutine_name": "2-repeat",
                    "local_sec": 4.0,
                }
            ],
            "3-repeat": [],
        }
        controller._combined_progress_list = [
            {
                "global_sec": 14.0,
                "obj_value": 94.0,
                "call_index": 2,
                "prefixed_subroutine_name": "2-repeat",
                "local_sec": 4.0,
            }
        ]

        HybridFlowShopCpLnsControllerCore._record_objective_point(
            controller, 20.0, 94.0, is_maximize=False
        )

        assert len(controller._subroutine_call_progress_map["3-repeat"]) == 1
        point = controller._subroutine_call_progress_map["3-repeat"][0]
        assert point["global_sec"] == 20.0
        assert point["obj_value"] == 94.0
        assert point["local_sec"] == 0.0
        assert len(controller._combined_progress_list) == 2

    def test_record_objective_point_adds_strict_improvement_for_maximize(self) -> None:
        controller = MagicMock(spec=HybridFlowShopCpLnsControllerCore)
        controller._active_call_index = 2
        controller._active_subroutine_name = "repeat"
        controller._active_call_global_start = 10.0
        controller._subroutine_call_progress_map = {
            "2-repeat": [
                {
                    "global_sec": 12.0,
                    "obj_value": 95.0,
                    "call_index": 2,
                    "prefixed_subroutine_name": "2-repeat",
                    "local_sec": 2.0,
                }
            ]
        }
        controller._combined_progress_list = list(
            controller._subroutine_call_progress_map["2-repeat"]
        )

        HybridFlowShopCpLnsControllerCore._record_objective_point(
            controller, 13.0, 96.0, is_maximize=True
        )

        assert len(controller._subroutine_call_progress_map["2-repeat"]) == 2
        point = controller._subroutine_call_progress_map["2-repeat"][-1]
        assert point["global_sec"] == 13.0
        assert point["obj_value"] == 96.0
        assert point["local_sec"] == 3.0
        assert len(controller._combined_progress_list) == 2

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
        controller = _make_progression_controller()
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

    def test_get_progression_data_collects_nested_incremental_sw_cp_reports(self) -> None:
        controller = _make_progression_controller()
        controller._subroutine_call_meta_list = [
            {
                "call_index": 1,
                "subroutine_name": "incremental_sw_cp",
                "prefixed_subroutine_name": "1-incremental_sw_cp",
                "global_start_sec": 10.0,
                "global_end_sec": 11.0,
                "elapsed_sec": 1.0,
            }
        ]
        controller._subroutine_end_marker_list = [
            {
                "global_end_sec": 11.0,
                "call_index": 1,
                "prefixed_subroutine_name": "1-incremental_sw_cp",
                "subroutine_name": "incremental_sw_cp",
            }
        ]
        controller._method_context_meta_map = {
            "1-incremental_sw_cp.1-unfixed_batch_count_002": {
                "global_start_sec": 10.2
            },
            "1-incremental_sw_cp.2-unfixed_batch_count_003": {
                "global_start_sec": 10.7
            },
        }
        controller.solution_manager.history = [
            SimpleNamespace(
                report=SimpleNamespace(
                    call_context="1-incremental_sw_cp.1-unfixed_batch_count_002",
                    progress_obj_value_records=((0.01, 982.0), (0.22, 980.0)),
                    progress_time_basis="local",
                )
            ),
            SimpleNamespace(
                report=SimpleNamespace(
                    call_context="1-incremental_sw_cp.2-unfixed_batch_count_003",
                    progress_obj_value_records=(
                        (0.05, 978.0),
                        (0.08, 977.0),
                        (0.15, 976.0),
                    ),
                    progress_time_basis="local",
                )
            ),
        ]

        data = HybridFlowShopCpLnsControllerCore.get_progression_data(controller)

        incremental_call = data["subroutine_calls"][0]
        assert [p["obj_value"] for p in incremental_call["local_progress_list"]] == [
            982.0,
            980.0,
            978.0,
            977.0,
            976.0,
        ]
        assert [p["obj_value"] for p in data["combined_progress_list"]] == [
            982.0,
            980.0,
            978.0,
            977.0,
            976.0,
        ]
        assert [round(p["global_sec"], 2) for p in incremental_call["local_progress_list"]] == [
            10.21,
            10.42,
            10.75,
            10.78,
            10.85,
        ]

    def test_get_progression_data_collects_nested_repeat_reports(self) -> None:
        controller = _make_progression_controller()
        controller._subroutine_call_meta_list = [
            {
                "call_index": 1,
                "subroutine_name": "repeat_while_improvement",
                "prefixed_subroutine_name": "1-repeat_while_improvement",
                "global_start_sec": 5.0,
                "global_end_sec": 7.0,
                "elapsed_sec": 2.0,
            }
        ]
        controller._subroutine_end_marker_list = [
            {
                "global_end_sec": 7.0,
                "call_index": 1,
                "prefixed_subroutine_name": "1-repeat_while_improvement",
                "subroutine_name": "repeat_while_improvement",
            }
        ]
        controller._method_context_meta_map = {
            "1-repeat_while_improvement.1-reps_001.1-sw_cp": {
                "global_start_sec": 5.25
            },
            "1-repeat_while_improvement.2-reps_002.1-sw_cp": {
                "global_start_sec": 6.05
            },
        }
        controller.solution_manager.history = [
            SimpleNamespace(
                report=SimpleNamespace(
                    call_context="1-repeat_while_improvement.1-reps_001.1-sw_cp",
                    progress_obj_value_records=((0.1, 98.0), (0.3, 95.0)),
                    progress_time_basis="local",
                )
            ),
            SimpleNamespace(
                report=SimpleNamespace(
                    call_context="1-repeat_while_improvement.2-reps_002.1-sw_cp",
                    progress_obj_value_records=((0.2, 94.0), (0.4, 93.0)),
                    progress_time_basis="local",
                )
            ),
        ]

        data = HybridFlowShopCpLnsControllerCore.get_progression_data(controller)

        repeat_call = data["subroutine_calls"][0]
        assert [p["obj_value"] for p in repeat_call["local_progress_list"]] == [
            98.0,
            95.0,
            94.0,
            93.0,
        ]
        assert [round(p["local_sec"], 2) for p in repeat_call["local_progress_list"]] == [
            0.35,
            0.55,
            1.25,
            1.45,
        ]
        assert [p["obj_value"] for p in data["combined_progress_list"]] == [
            98.0,
            95.0,
            94.0,
            93.0,
        ]

    def test_get_progression_data_falls_back_to_recorded_points_when_report_has_no_progress(
        self,
    ) -> None:
        controller = _make_progression_controller()
        controller._subroutine_call_meta_list = [
            {
                "call_index": 1,
                "subroutine_name": "apply_mip_lb",
                "prefixed_subroutine_name": "1-apply_mip_lb",
                "global_start_sec": 10.0,
                "global_end_sec": 20.0,
                "elapsed_sec": 10.0,
            }
        ]
        controller._subroutine_end_marker_list = [
            {
                "global_end_sec": 20.0,
                "call_index": 1,
                "prefixed_subroutine_name": "1-apply_mip_lb",
                "subroutine_name": "apply_mip_lb",
            }
        ]
        controller._subroutine_call_progress_map = {
            "1-apply_mip_lb": [
                {
                    "global_sec": 18.0,
                    "obj_value": 6408.0,
                    "call_index": 1,
                    "prefixed_subroutine_name": "1-apply_mip_lb",
                    "local_sec": 8.0,
                }
            ]
        }
        controller._combined_progress_list = list(
            controller._subroutine_call_progress_map["1-apply_mip_lb"]
        )
        controller.solution_manager.history = [
            SimpleNamespace(
                report=SimpleNamespace(
                    call_context="1-apply_mip_lb",
                    progress_obj_value_records=(),
                    progress_time_basis="local",
                    obj_value_records=((1.0, 6406.0),),
                )
            )
        ]

        data = HybridFlowShopCpLnsControllerCore.get_progression_data(controller)

        apply_mip_lb_call = data["subroutine_calls"][0]
        assert [p["obj_value"] for p in apply_mip_lb_call["local_progress_list"]] == [
            6408.0
        ]
        assert [p["obj_value"] for p in data["combined_progress_list"]] == [6408.0]

    def test_add_file_handler_lowers_root_level_to_capture_info(
        self, tmp_path
    ) -> None:
        controller = HybridFlowShopCpLnsControllerCore.__new__(
            HybridFlowShopCpLnsControllerCore
        )
        controller._working_dir_path = tmp_path
        controller.log_handlers = []

        root_logger = logging.getLogger()
        original_level = root_logger.level

        try:
            root_logger.setLevel(logging.WARNING)
            controller.add_file_handler(level=logging.INFO)
            logging.info("progression-log-smoke-test")
            controller.release_log_handlers()

            log_path = tmp_path / "subroutine_controller.log"
            assert log_path.exists()
            assert "progression-log-smoke-test" in log_path.read_text(
                encoding="utf-8"
            )
        finally:
            controller.log_handlers = []
            root_logger.setLevel(original_level)

    def test_run_tracks_top_level_subroutine_calls_for_main_loop(self) -> None:
        controller = HybridFlowShopCpLnsControllerCore.__new__(
            HybridFlowShopCpLnsControllerCore
        )
        controller._subroutine_flow = [
            {"method": "initialize"},
            {"method": "incremental_sw_cp"},
        ]
        controller.method_names_to_run_before_resume = set()
        controller._run_flow = MagicMock()
        controller._start_subroutine_call = MagicMock()
        controller._end_subroutine_call = MagicMock()
        controller.post_run_process = MagicMock()
        controller.timer = SimpleNamespace(
            start_dt=datetime.datetime(2026, 4, 8, 12, 0, 0),
            set_start_time=MagicMock(),
        )

        HybridFlowShopCpLnsControllerCore.run(controller, flow_resume_idx=0)

        assert controller._start_subroutine_call.call_args_list == [
            (("initialize",),),
            (("incremental_sw_cp",),),
        ]
        assert controller._end_subroutine_call.call_args_list == [
            (("initialize",),),
            (("incremental_sw_cp",),),
        ]
        assert controller._run_flow.call_count == 2
        controller.post_run_process.assert_called_once_with()
