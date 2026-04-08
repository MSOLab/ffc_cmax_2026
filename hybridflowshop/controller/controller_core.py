import datetime
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

from mbls.cpsat import (
    CpsatSolverReport,
    CpsatStatus,
    CpSubroutineController,
    CustomCpModel,
    ObjectiveBoundRecorder,
    ObjectiveValueRecorder,
)
from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from routix.util.comparison import float_a_leq_b, float_a_stl_b, float_equals
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from cpsat_solver_config import SolveConfig, configure_solver
from hybridflowshop.cpsat_model_2.cumulative import (
    BaseModelBuilder,
    CumulativeVars,
    OperationVars,
)
from hybridflowshop.cpsat_model_2.params import Params
from hybridflowshop.report import HfsCpsatSolverReport, HfsSubroutineReport
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule

from ..painter.gantt import GanttPlotter
from ..solution_manager import HfsSolutionManager


class HybridFlowShopCpLnsControllerCore(
    CpSubroutineController[HybridFlowshopParameters, CustomCpModel, StoppingCriteria]
):
    """
    Controller for solving Hybrid Flow Shop problems using CP-based algorithms.
    """

    # Start controller state

    solution_manager: HfsSolutionManager
    """Solution manager for Hybrid Flow Shop scheduling solutions."""

    total_elapsed_time: float
    """Total elapsed time for the controller."""

    # End controller state

    # Start controller pre-defined values

    method_names_to_run_before_resume: set[str]
    """Name of methods to run before resuming from a paused state."""

    # End controller pre-defined values

    def __init__(
        self,
        instance: HybridFlowshopParameters,
        shared_param_dict: dict,
        subroutine_flow: DynamicDataObject,
        stopping_criteria: StoppingCriteria,
    ):
        super().__init__(
            instance,
            shared_param_dict,
            CustomCpModel,
            subroutine_flow,
            stopping_criteria,
        )
        self.solution_manager = HfsSolutionManager()

        self.method_names_to_run_before_resume = {
            "set_random_seed",
            "set_cp_model_as_base_cp_model",
        }
        assert "" not in self.method_names_to_run_before_resume

        # Frequently used parameters
        self.job_2_stage_2_p_dict = self.instance.job_2_stage_2_p_map
        """Job name -> stage name -> processing time map"""
        self.stage_2_job_2_p_dict = self.instance.stage_2_job_2_p_map
        """Stage name -> job name -> processing time map"""

        self._subroutine_call_progress_map: dict[str, list[dict]] = {}
        self._subroutine_call_meta_list: list[dict] = []
        self._combined_progress_list: list[dict] = []
        self._subroutine_end_marker_list: list[dict] = []
        self._method_context_meta_map: dict[str, dict[str, Any]] = {}
        self._active_call_index: int | None = None
        self._active_subroutine_name: str | None = None
        self._active_call_global_start: float | None = None
        self._call_counter: int = 0

        logging.info(
            f"Start solving {self.instance.name} using CP model class:"
            f" {self.cp_model_class.__module__}.{self.cp_model_class.__name__}",
        )

    # Start abstract getters

    def create_base_cp_model(self, **kwargs) -> CustomCpModel:
        builder = BaseModelBuilder()
        horizon = self.get_horizon()
        tighten_ranges = kwargs.get("tighten_ranges", False)
        link_job_completion = kwargs.get("link_job_completion", False)
        mdl, params, variables = builder.build(
            self.instance,
            horizon,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        )

        self.params: Params = params
        self.vars: CumulativeVars = variables
        mdl.minimize(variables.makespan)
        return mdl

    # End abstract getters

    def get_horizon(self) -> int:
        """
        Get the horizon for the CP model.

        - If the best objective value is available from the solution manager,
          use its ceiling as the horizon.
        - Otherwise, retrieve the horizon from the shared parameters.

        Raises:
            ValueError: If no best objective value from the solution manager
                and the horizon is not found in shared parameters.

        Returns:
            int: The horizon value for the CP model.
        """
        if self.solution_manager.best_obj_value:
            return math.ceil(self.solution_manager.best_obj_value)
        if not isinstance(self.shared_param_dict, dict):
            raise ValueError("Shared parameters is not a dictionary.")
        if "horizon" not in self.shared_param_dict:
            raise ValueError("Horizon not found in shared parameters.")
        return self.shared_param_dict["horizon"]

    def set_working_dir(self, dir_path: Path | str):
        super().set_working_dir(dir_path)
        self.log_handlers: list[logging.StreamHandler] = []
        self.add_file_handler()

    def add_file_handler(
        self,
        log_filename: Optional[str] = None,
        level=logging.INFO,
        fmt="%(asctime)s - %(levelname)s - %(message)s",
    ):
        logger = logging.getLogger()
        _log_filename = log_filename or "subroutine_controller.log"
        if self._working_dir_path is not None:
            log_path = self._working_dir_path / _log_filename
            # 이미 같은 파일 핸들러가 등록되어 있는지 확인 (중복 방지)
            for handler in logger.handlers:
                if isinstance(
                    handler, logging.FileHandler
                ) and handler.baseFilename == str(log_path):
                    return  # 이미 등록되어 있으면 추가하지 않음

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(file_handler)
            self.log_handlers = [file_handler]

    def release_log_handlers(self) -> None:
        logger = logging.getLogger()
        for handler in self.log_handlers:
            logger.removeHandler(handler)
            handler.close()

    # Start stopping condition

    def is_stopping_condition(self, log_reason_if_true: bool = True, **kwargs) -> bool:
        return self.ub_equals_lb(log_reason_if_true) or self.time_is_up(
            log_reason_if_true
        )

    def ub_equals_lb(self, log_reason_if_true: bool = True) -> bool:
        """Checks if the current best objective equals the best objective bound.

        Raises:
            ValueError: If the current best objective is better than the best objective bound.

        Returns:
            bool: True if the current best objective equals the best objective bound,
                False otherwise.
        """
        best_obj_value = self.solution_manager.best_obj_value
        best_obj_bound = self.solution_manager.best_obj_bound

        if best_obj_value is None or best_obj_bound is None:
            # Case 1: Either ObjValue or ObjBound is None
            return False
        # best_obj_value is not None and best_obj_bound is not None

        # Case 2: ObjValue equals ObjBound
        # Considered equal if close enough (considering floating point precision)
        if float_equals(best_obj_value, best_obj_bound):
            if log_reason_if_true:
                logging.info(
                    f"Stop by UB == LB: best objective value ({best_obj_value}) "
                    f"equals best objective bound ({best_obj_bound})."
                )
            return True
        # Case 3: ObjValue is strictly better than ObjBound
        if self.solution_manager._a_is_better_obj_value(best_obj_value, best_obj_bound):
            if log_reason_if_true:
                raise ValueError(
                    f"Inconsistent state: best objective value ({best_obj_value}) "
                    f"is strictly better than best objective bound ({best_obj_bound})."
                )
        # Case 4: ObjValue is worse than ObjBound
        return False

    def time_is_up(self, log_reason_if_true: bool = True) -> bool:
        if self.stopping_criteria.timelimit is None:
            return False
        # If total elapsed time exceeds the stopping criteria
        if float_a_leq_b(self.stopping_criteria.timelimit, self.timer.elapsed_sec):
            if log_reason_if_true:
                logging.info("Stop by timelimit")
            return True
        return False

    def get_remaining_sec(self) -> float:
        return self.timer.get_remaining_sec(self.stopping_criteria.timelimit)

    def get_remaining_time_limit(self, subroutine_time_limit: float | None) -> float:
        """Get the remaining time limit for the subroutine.

        Args:
            subroutine_time_limit (float | None, optional): The time limit for the subroutine in seconds.
                If None, the remaining time limit is used.

        Returns:
            float: The minimum of the subroutine time limit and the remaining time limit.
        """
        if subroutine_time_limit is None:
            return self.get_remaining_sec()
        return min(subroutine_time_limit, self.get_remaining_sec())

    # End stopping condition

    # Start report creation

    def _make_subroutine_report(
        self,
        elapsed_time: float,
        obj_value: float | None,
        obj_bound: float | None,
        is_init: bool,
        subroutine_name: str = "",
        progress_obj_value_records: Sequence[tuple[float, float]] = (),
        progress_time_basis: str = "local",
    ) -> HfsSubroutineReport:
        call_context = self._get_call_context_of_current_method()
        return HfsSubroutineReport(
            elapsed_time=elapsed_time,
            obj_value=obj_value,
            obj_bound=obj_bound,
            is_init=is_init,
            subroutine_name=subroutine_name,
            call_context=call_context,
            progress_obj_value_records=tuple(progress_obj_value_records),
            progress_time_basis=progress_time_basis,
        )

    # End report creation

    # Start visualization

    def draw_gantt(
        self,
        schedule: HybridFlowshopLiteSchedule,
        output_path: Path | None = None,
        stage_list: Sequence[str] | None = None,
        force_start: int | None = None,
        force_end: int | None = None,
        highlight_op_set: set[tuple[str, str]] | None = None,
    ):
        """Draws the Gantt chart of the given schedule.

        Args:
            schedule (HybridFlowshopLiteSchedule): The schedule to draw.
            output_path (Path | None, optional): The output path for the Gantt chart image. Defaults to None.
        """
        if output_path is None:
            output_path = self.get_file_path_for_subroutine("_gantt.png")
        if isinstance(schedule, HybridFlowshopLiteSchedule):
            plotter = GanttPlotter()
            plotter.export_hybrid_flowshop_plot(
                output_path,
                schedule.get_jik_2_start_time_map(),
                schedule.get_jik_2_end_time_map(),
                self.instance.job_id_list,
                stage_list=stage_list,
                highlight_op_set=highlight_op_set,
                force_start=force_start,
                force_end=force_end,
            )

    def draw_incumbent_gantt(
        self,
        output_path: Path | None = None,
        highlight_op_set: set[tuple[str, str]] | None = None,
    ) -> None:
        """Draws the Gantt chart of the incumbent solution.

        Args:
            output_path (Path | None, optional): The output path for the Gantt chart image. Defaults to None.
        """
        incumbent_solution = self.solution_manager.get_incumbent()
        if isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            self.draw_gantt(
                incumbent_solution,
                output_path=output_path,
                highlight_op_set=highlight_op_set,
            )
        else:
            logging.warning("No incumbent solution available to draw Gantt chart.")

    # End visualization

    # Start subroutine progression recorder

    def _record_method_context_start(self, call_context: str) -> None:
        self._method_context_meta_map.setdefault(
            call_context,
            {
                "call_context": call_context,
                "global_start_sec": self.timer.elapsed_sec,
            },
        )

    def _record_method_context_end(self, call_context: str) -> None:
        meta = self._method_context_meta_map.setdefault(
            call_context,
            {"call_context": call_context},
        )
        global_end = self.timer.elapsed_sec
        meta["global_end_sec"] = global_end
        global_start = meta.get("global_start_sec")
        if global_start is not None:
            meta["elapsed_sec"] = global_end - global_start

    def _get_context_start_sec(self, call_context: str) -> float | None:
        meta = self._method_context_meta_map.get(call_context)
        if meta is not None:
            global_start = meta.get("global_start_sec")
            if isinstance(global_start, (float, int)):
                return float(global_start)

        for subroutine_meta in self._subroutine_call_meta_list:
            if subroutine_meta["prefixed_subroutine_name"] == call_context:
                return float(subroutine_meta["global_start_sec"])
        return None

    @staticmethod
    def _get_call_context_depth(call_context: str) -> int:
        if not call_context or call_context == "ROOT":
            return 0
        return len(call_context.split("."))

    def _build_progress_point_list(
        self,
        progress_records: Sequence[tuple[float, float]],
        *,
        call_index: int,
        prefixed_name: str,
        global_start_sec: float,
        progress_time_basis: str = "local",
    ) -> list[dict]:
        progress_list: list[dict] = []
        for timestamp, obj_value in progress_records:
            if progress_time_basis == "global":
                global_sec = timestamp
                local_sec = timestamp - global_start_sec
            else:
                global_sec = global_start_sec + timestamp
                local_sec = timestamp

            progress_list.append(
                {
                    "global_sec": global_sec,
                    "obj_value": obj_value,
                    "call_index": call_index,
                    "prefixed_subroutine_name": prefixed_name,
                    "local_sec": local_sec,
                }
            )
        return progress_list

    @staticmethod
    def _build_combined_progress_list(subroutine_calls: Sequence[dict]) -> list[dict]:
        combined_progress_list: list[dict] = []
        for call in subroutine_calls:
            for point in call.get("local_progress_list", []):
                combined_progress_list.append(dict(point))

        combined_progress_list.sort(
            key=lambda point: (
                point.get("global_sec", math.inf),
                point.get("call_index", math.inf),
                point.get("local_sec", math.inf),
            )
        )
        return combined_progress_list

    def _start_subroutine_call(self, subroutine_name: str) -> None:
        self._call_counter += 1
        call_index = self._call_counter
        prefixed_name = f"{call_index}-{subroutine_name}"
        global_start = self.timer.elapsed_sec

        self._active_call_index = call_index
        self._active_subroutine_name = subroutine_name
        self._active_call_global_start = global_start

        self._subroutine_call_progress_map[prefixed_name] = []
        self._subroutine_call_meta_list.append(
            {
                "call_index": call_index,
                "subroutine_name": subroutine_name,
                "prefixed_subroutine_name": prefixed_name,
                "global_start_sec": global_start,
            }
        )

    def _end_subroutine_call(self, subroutine_name: str) -> None:
        if self._active_call_index is None:
            return
        call_index = self._active_call_index
        prefixed_name = f"{call_index}-{subroutine_name}"
        global_end = self.timer.elapsed_sec

        self._subroutine_end_marker_list.append(
            {
                "global_end_sec": global_end,
                "call_index": call_index,
                "prefixed_subroutine_name": prefixed_name,
                "subroutine_name": subroutine_name,
            }
        )

        for meta in self._subroutine_call_meta_list:
            if meta["call_index"] == call_index:
                meta["global_end_sec"] = global_end
                meta["elapsed_sec"] = global_end - meta["global_start_sec"]
                break

        self._active_call_index = None
        self._active_subroutine_name = None
        self._active_call_global_start = None

    def _call_method(self, method_name: str, **kwargs: dict[str, Any]):
        if not hasattr(self, method_name):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {method_name}"
            )

        self._method_context_mgr.push(method_name)
        call_context = self._get_call_context_of_current_method()
        self._record_method_context_start(call_context)

        start_sec = self.timer.elapsed_sec
        self.method_call_counts[method_name] += 1

        log_entry: dict[str, Any] = {
            "method": method_name,
            "call_context": call_context,
            "start_sec": start_sec,
            "kwargs": kwargs,
        }
        try:
            getattr(self, method_name)(**kwargs)
        except Exception as e:
            end_sec = self.timer.elapsed_sec
            elapsed_sec = end_sec - start_sec
            log_entry["elapsed_sec"] = elapsed_sec
            log_entry["error"] = str(e)
            logging.error(str(log_entry))
            self._record_method_context_end(call_context)
            self._method_context_mgr.pop()
            raise e

        end_sec = self.timer.elapsed_sec
        elapsed_sec = end_sec - start_sec
        log_entry["elapsed_sec"] = elapsed_sec
        logging.info(str(log_entry))

        self._record_method_context_end(call_context)
        self._method_context_mgr.pop()

    @contextmanager
    def temporarily_extended_context(self, appended_name: str):
        self._method_context_mgr.push(appended_name)
        call_context = self._get_call_context_of_current_method()
        self._record_method_context_start(call_context)
        try:
            yield
        finally:
            self._record_method_context_end(call_context)
            self._method_context_mgr.pop()

    def _record_objective_point(
        self,
        global_sec: float,
        obj_value: float,
        is_maximize: bool | None = None,
    ) -> None:
        if self._active_call_index is None:
            return
        call_index = self._active_call_index
        subroutine_name = self._active_subroutine_name or "unknown"
        prefixed_name = f"{call_index}-{subroutine_name}"
        global_start = self._active_call_global_start or global_sec
        local_sec = global_sec - global_start
        recorded_points = self._subroutine_call_progress_map.setdefault(
            prefixed_name, []
        )

        if recorded_points:
            last_obj_value = recorded_points[-1]["obj_value"]
            if is_maximize is None:
                cp_model = getattr(self, "cp_model", None)
                if cp_model is not None and hasattr(cp_model, "is_maximize"):
                    is_maximize = cp_model.is_maximize()
                else:
                    is_maximize = False

            is_improved = (
                float_a_stl_b(last_obj_value, obj_value)
                if is_maximize
                else float_a_stl_b(obj_value, last_obj_value)
            )
            if not is_improved:
                return

        point = {
            "global_sec": global_sec,
            "obj_value": obj_value,
            "call_index": call_index,
            "prefixed_subroutine_name": prefixed_name,
            "local_sec": local_sec,
        }

        recorded_points.append(point)
        self._combined_progress_list.append(point)

    def _get_report_for_call(self, prefixed_name: str) -> dict | None:
        """Get the subroutine report from solution_manager.history by call context.

        Args:
            prefixed_name: The prefixed subroutine name (e.g., "4-pw_cp")

        Returns:
            The report dict if found, None otherwise
        """
        for record in self.solution_manager.history:
            report = getattr(record, "report", None)
            if report is None:
                continue
            # Check if report has call_context attribute and it matches
            call_context = getattr(report, "call_context", None)
            if call_context == prefixed_name:
                # Convert report to dict format for consistent access
                return {
                    "progress_obj_value_records": getattr(
                        report, "progress_obj_value_records", ()
                    ),
                    "progress_time_basis": getattr(
                        report, "progress_time_basis", "local"
                    ),
                }
        return None

    def _collect_nested_reports(
        self, parent_prefixed_name: str, parent_call_index: int
    ) -> list[tuple[float, float]]:
        """Collect progress records from nested subroutine calls within a parent call.

        This handles cases like:
        - incremental_pw_cp -> multiple unfixed_batch_count_* calls -> pw_cp calls
        - repeat_while_improvement -> multiple reps_* calls

        Args:
            parent_prefixed_name: Parent's prefixed name (e.g., "2-incremental_pw_cp")
            parent_call_index: Parent's call index in the meta list

        Returns:
            Combined list of (local_sec, obj_value) tuples from nested calls,
            sorted by local time
        """
        # Find parent meta
        parent_meta = None

        for i, meta in enumerate(self._subroutine_call_meta_list):
            if meta["call_index"] == parent_call_index:
                parent_meta = meta
                break

        if not parent_meta:
            return []

        parent_start = float(parent_meta["global_start_sec"])
        nested_prefix = f"{parent_prefixed_name}."

        # Collect reports whose global_start falls within this parent's range
        progress_records = []
        for record in self.solution_manager.history:
            report = getattr(record, "report", None)
            if report is None:
                continue

            call_context = getattr(report, "call_context", None)
            if call_context is None:
                continue

            if not isinstance(call_context, str):
                continue

            if not call_context.startswith(nested_prefix):
                continue

            parent_depth = self._get_call_context_depth(parent_prefixed_name)
            call_context_depth = self._get_call_context_depth(call_context)
            if call_context_depth <= parent_depth:
                continue

            nested_records = getattr(report, "progress_obj_value_records", ())
            if nested_records:
                nested_global_start = self._get_context_start_sec(call_context)
                if nested_global_start is not None:
                    progress_time_basis = getattr(
                        report, "progress_time_basis", "local"
                    )
                    for local_time, obj_value in nested_records:
                        if progress_time_basis == "global":
                            global_time = local_time
                        else:
                            global_time = nested_global_start + local_time
                        parent_relative_time = global_time - parent_start
                        progress_records.append((parent_relative_time, obj_value))

        return sorted(progress_records, key=lambda x: x[0])

    def get_progression_data(self) -> dict:
        subroutine_calls = []
        for meta in self._subroutine_call_meta_list:
            prefixed_name = meta["prefixed_subroutine_name"]
            call_index = meta["call_index"]

            # First, try direct report match
            report_data = self._get_report_for_call(prefixed_name)

            # If no direct report or it's a container-like subroutine, collect nested
            local_list = []
            if report_data and report_data["progress_obj_value_records"]:
                # Has its own progress records
                local_list = self._build_progress_point_list(
                    report_data["progress_obj_value_records"],
                    call_index=call_index,
                    prefixed_name=prefixed_name,
                    global_start_sec=float(meta["global_start_sec"]),
                    progress_time_basis=report_data["progress_time_basis"],
                )
            else:
                # Try to collect from nested calls
                nested_records = self._collect_nested_reports(prefixed_name, call_index)
                if nested_records:
                    local_list = self._build_progress_point_list(
                        nested_records,
                        call_index=call_index,
                        prefixed_name=prefixed_name,
                        global_start_sec=float(meta["global_start_sec"]),
                    )
                else:
                    # Fallback to existing map
                    local_list = self._subroutine_call_progress_map.get(
                        prefixed_name, []
                    )

            subroutine_calls.append(
                {
                    "call_index": call_index,
                    "subroutine_name": meta["subroutine_name"],
                    "prefixed_subroutine_name": prefixed_name,
                    "global_start_sec": meta["global_start_sec"],
                    "global_end_sec": meta.get("global_end_sec"),
                    "elapsed_sec": meta.get("elapsed_sec"),
                    "local_progress_list": local_list,
                }
            )

        subroutine_calls.sort(key=lambda x: x["call_index"])

        return {
            "artifact_version": 1,
            "instance_id": self.instance.name,
            "timelimit_sec": getattr(self.stopping_criteria, "timelimit", None),
            "subroutine_calls": subroutine_calls,
            "combined_progress_list": self._build_combined_progress_list(
                subroutine_calls
            ),
            "subroutine_end_marker_list": self._subroutine_end_marker_list,
        }

    # End subroutine progression recorder

    def add_obj_value_log(
        self, elapsed: float, value: float, is_maximize: bool | None = None
    ) -> None:
        super().add_obj_value_log(elapsed, value, is_maximize)
        self._record_objective_point(elapsed, value, is_maximize)

    def extend_obj_value_log(
        self,
        value_log: Sequence[tuple[float, float]],
        is_maximize: bool | None = None,
    ) -> None:
        super().extend_obj_value_log(value_log, is_maximize)
        for elapsed_time, obj_value in value_log:
            self._record_objective_point(elapsed_time, obj_value, is_maximize)

    def run(self, flow_resume_idx: int = -1) -> None:
        """Overrides the run method to execute the subroutine flow.

        Args:
            flow_resume_idx (int, optional): The index to resume the flow from. Defaults to -1.
        """
        # Detect if this is a resume run (flow_resume_idx != -1 means resuming from a specific point)
        is_resume_run = flow_resume_idx != -1

        if isinstance(self._subroutine_flow, Sequence) and not isinstance(
            self._subroutine_flow, (str, bytes)
        ):
            # Accumulate elapsed time from all pre-resume methods
            total_pre_resume_elapsed = 0.0

            # First pass: run all methods before flow_resume_idx
            for idx, subroutine_data in enumerate(self._subroutine_flow):
                method_name = subroutine_data.get("method", "")
                if idx < flow_resume_idx:
                    e_timer = ElapsedTimer()
                    if (
                        subroutine_data.get("method", "")
                        in self.method_names_to_run_before_resume
                    ):
                        self._run_flow(subroutine_data)
                    else:
                        self._run_flow(subroutine_data, skip_method_call=True)
                    total_pre_resume_elapsed += e_timer.elapsed_sec

            # Adjust timer once after all pre-resume methods complete
            if is_resume_run:
                # Resume run: _try_apply_resume already set the baseline,
                # so we add pre-resume time to the existing baseline
                current_start_dt = self.timer.start_dt
                new_start_dt = current_start_dt - datetime.timedelta(
                    seconds=total_pre_resume_elapsed
                )
                self.timer.set_start_time(new_start_dt)

            # Second pass: run methods from flow_resume_idx onwards
            for idx, subroutine_data in enumerate(self._subroutine_flow):
                if idx >= flow_resume_idx:
                    self._run_flow(subroutine_data)
                    self._end_subroutine_call(method_name)
        else:
            logging.warning(
                "Subroutine flow is not a sequence; running as a single step."
            )
            if isinstance(self._subroutine_flow, dict):
                method_name = self._subroutine_flow.get("method", "unknown")
            else:
                method_name = "unknown"
            self._start_subroutine_call(method_name)
            self._run_flow(self._subroutine_flow)
            self._end_subroutine_call(method_name)
        self.post_run_process()

    # Start post-run process

    def post_run_process(self) -> None:
        """
        Finalizes the run by checking the feasibility of the incumbent solution
        and releasing log handlers.
        """
        incumbent = self.solution_manager.get_incumbent()
        if incumbent:
            self.check_feasibility(incumbent.get_jik_2_start_time_map())
        self.release_log_handlers()
        self.total_elapsed_time = self.timer.elapsed_sec

    def check_feasibility(
        self, start_time_map: dict[tuple[str, str, str], int]
    ) -> float:
        """Check the feasibility of the given start times.

        Args:
            start_time_map (dict[tuple[str, str, str], int]): A mapping of (job, stage, machine) to start time.

        Raises:
            ValueError: If any start time is negative or invalid.
            RuntimeError: If the feasibility check fails while solving the model.
            ValueError: If the feasibility check fails with an unexpected status.

        Returns:
            float: The objective value of the solution if feasible.
        """
        logging.info("Feasibility check starts")
        for (j, i, k), start_time in start_time_map.items():
            if start_time < 0:
                raise ValueError(
                    f"Invalid start time for job {j}, stage {i}, machine {k}: {start_time}"
                )
        makespan = 0
        end_time_map: dict[tuple[str, str, str], int] = {}
        for (j, i, k), start_time in start_time_map.items():
            end_time = start_time + self.job_2_stage_2_p_dict[j][i]
            end_time_map[(j, i, k)] = end_time
            if end_time > makespan:
                makespan = end_time

        from ..schedule_lite import (
            validate_duration,
            validate_no_overlap,
            validate_precedence,
        )

        validate_duration(start_time_map, end_time_map, self.stage_2_job_2_p_dict)
        validate_precedence(start_time_map, end_time_map, self.instance.stage_id_list)
        validate_no_overlap(
            start_time_map,
            end_time_map,
            self.instance.stage_id_list,
            self.instance.stage_2_machines_map,
        )

        logging.info("Feasibility check passed")
        return makespan

    # End post-run process

    # Start solver call methods

    def solve_cp_model_2(
        self,
        mdl: CustomCpModel,
        computational_time: float,
        solver_thread_cnt: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        keep_all_feasible_solutions_in_presolve: bool | None = None,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
        e_timer: ElapsedTimer | None = None,
        log_search_progress: bool = False,
        print_on_obj_value_update: bool = False,
        print_on_obj_bound_update: bool = False,
        log_level_obj_value: int = logging.INFO,
        log_level_obj_bound: int = logging.INFO,
        last_timestamp_note: Any | None = None,
    ) -> CpsatSolverReport:
        start_time = self.timer.elapsed_sec
        if e_timer is None:
            # If no external timer is provided, create a new ElapsedTimer for this solve call.
            e_timer = ElapsedTimer()

        solve_cfg = SolveConfig(
            log_search_progress=log_search_progress,
            log_to_stdout=False if log_search_progress else None,
            log_to_response=True if log_search_progress else None,
            max_time_in_seconds=computational_time,
            num_workers=solver_thread_cnt,
            keep_all_feasible_solutions_in_presolve=keep_all_feasible_solutions_in_presolve,
            random_seed=self.random_seed,
            encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
            expand_reservoir_constraints=expand_reservoir_constraints,
            expand_reservoir_using_circuit=expand_reservoir_using_circuit,
            interleave_search=interleave_search,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
        )
        self.solver = configure_solver(solve_cfg)
        obj_value_recorder = ObjectiveValueRecorder(
            e_timer,
            print_on_record=print_on_obj_value_update,
            log_level_on_record=log_level_obj_value,
        )

        obj_bound_recorder = ObjectiveBoundRecorder(
            e_timer,
            print_on_record=print_on_obj_bound_update,
            log_level_on_record=log_level_obj_bound,
        )
        self.solver.best_bound_callback = obj_bound_recorder

        cp_solver_status = self.solver.solve(mdl, solution_callback=obj_value_recorder)
        if log_search_progress:
            solve_log = self.solver.response_proto.solve_log
            if solve_log:
                try:
                    filename_suffix = "_cp_sat_search.log"
                    if last_timestamp_note and isinstance(last_timestamp_note, str):
                        filename_suffix = f"_cp_sat_search_{last_timestamp_note}.log"
                    solve_log_path = self.get_file_path_for_subroutine(filename_suffix)
                    with solve_log_path.open("a", encoding="utf-8") as fp:
                        fp.write(
                            f"\n=== {self._get_call_context_of_current_method()} "
                            f"at {datetime.datetime.now().isoformat()} ===\n"
                        )
                        fp.write(solve_log)
                        if not solve_log.endswith("\n"):
                            fp.write("\n")
                except Exception as err:
                    logging.warning("Failed to write CP-SAT search log: %s", err)
        cpsat_status = CpsatStatus.from_cp_solver_status(cp_solver_status)
        elapsed_time = self.solver.wall_time
        if cpsat_status.is_feasible:
            obj_value = self.solver.objective_value
            if cpsat_status == CpsatStatus.OPTIMAL:
                obj_bound = obj_value
            else:
                obj_bound = self.solver.best_objective_bound
        else:
            obj_value, obj_bound = CpsatStatus.get_obj_value_and_bound_for_infeasible(
                False
            )

        last_timestamp = e_timer.elapsed_sec

        # Store the objective value and bound logs

        obj_value_records: list[tuple[float, float]] = []
        for entry in obj_value_recorder.entries:
            obj_value_records.append((entry[0], entry[1].value))
        if cpsat_status.is_feasible:
            obj_value_records.append((last_timestamp, obj_value))

        if obj_value_is_valid:
            old_obj_value_records = [
                (start_time + timestamp, value)
                for timestamp, value in obj_value_records
            ]
            self.extend_obj_value_log(
                old_obj_value_records, is_maximize=self.cp_model.is_maximize()
            )
            # Record value for the last timestamp if it is the same as the last value
            # and is not recorded for the last timestamp
            if (
                obj_value == self.obj_store.get_last_obj_value()
                and (start_time + last_timestamp, obj_value) not in obj_value_records
            ):
                self.add_obj_value_log(
                    start_time + last_timestamp, obj_value, is_maximize=None
                )

        def get_obj_bound_records() -> list[tuple[float, float]]:
            """Returns the recorded objective bounds and elapsed times.

            Returns:
                list[tuple[float, float]]: A list of tuples containing (elapsed time, objective bound).
            """
            timestamp_2_bound_map: dict[float, float] = {}

            for b_entry in obj_bound_recorder.elapsed_time_and_bound:
                timestamp = b_entry[0]
                bound = b_entry[1]
                if timestamp not in timestamp_2_bound_map:
                    timestamp_2_bound_map[timestamp] = bound

            for v_entry in obj_value_recorder.entries:
                timestamp = v_entry[0]
                bound = v_entry[1].bound
                if timestamp not in timestamp_2_bound_map:
                    timestamp_2_bound_map[timestamp] = bound

            return [
                (timestamp, timestamp_2_bound_map[timestamp])
                for timestamp in sorted(timestamp_2_bound_map.keys())
            ]

        obj_bound_records = get_obj_bound_records()
        if cpsat_status.is_feasible:
            obj_bound_records.append((last_timestamp, obj_bound))

        if obj_bound_is_valid:
            old_obj_bound_records = [
                (start_time + timestamp, bound)
                for timestamp, bound in obj_bound_records
            ]
            self.extend_obj_bound_log(old_obj_bound_records, is_maximize=None)
            # Record bound for the last timestamp if it is the same as the last bound
            # and is not recorded for the last timestamp
            if (
                obj_bound == self.obj_store.get_last_obj_bound()
                and (start_time + last_timestamp, obj_bound) not in obj_bound_records
            ):
                self.add_obj_bound_log(
                    start_time + last_timestamp, obj_bound, is_maximize=None
                )

        _last_timestamp_note = (
            last_timestamp_note or self._get_call_context_of_current_method()
        )
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

        solver_report = CpsatSolverReport(
            elapsed_time,
            obj_value,
            obj_bound,
            cpsat_status,
            obj_value_records,
            obj_bound_records,
        )
        return solver_report

    def extract_stage_2_job_2_start_time_map(
        self, params: Params, variables: OperationVars
    ) -> dict[str, dict[str, int]]:
        start_time_map: dict[str, dict[str, int]] = {}
        """stage ID -> job ID -> start time"""
        for i in params.i_list:
            start_time_map[i] = {}
            for j in params.j_list:
                start_value = self.solver.Value(variables.op_start[j, i])
                start_time_map[i][j] = start_value
        return start_time_map

    def extract_stage_2_job_2_end_time_map(
        self, params: Params, variables: OperationVars
    ) -> dict[str, dict[str, int]]:
        end_time_map: dict[str, dict[str, int]] = {}
        """stage ID -> job ID -> end time"""
        for i in params.i_list:
            end_time_map[i] = {}
            for j in params.j_list:
                end_value = self.solver.Value(variables.op_end[j, i])
                end_time_map[i][j] = end_value
        return end_time_map

    def create_empty_schedule_from_ins(
        self, instance: HybridFlowshopParameters | None = None
    ) -> HybridFlowshopLiteSchedule:
        """Creates an empty HybridFlowshopLiteSchedule for the problem instance.

        Args:
            instance (HybridFlowshopParameters | None, optional): the problem instance.
                If None, uses self.instance. Defaults to None.

        Returns:
            HybridFlowshopLiteSchedule: An empty schedule object.
        """
        if instance is None:
            instance = self.instance
        return HybridFlowshopLiteSchedule(
            instance.job_id_list,
            instance.stage_id_list,
            instance.stage_2_machines_map,
        )

    def create_schedule(
        self, params: Params, variables: OperationVars, make_semi_active: bool = False
    ) -> HybridFlowshopLiteSchedule:
        """
        Constructs a full HybridFlowshopLiteSchedule from the solved CP model.

        - This method first extracts the start and end times for each operation
        (job, stage) from the CP solver.
        - It then uses a greedy approach to assign each operation
        to a specific machine within its stage.
          - Operations are assigned in the order of their start times
          to the earliest available machine.

        Raises:
            RuntimeError: If an operation cannot be scheduled due to timing conflicts
                          or if the machine assignment fails.
            RuntimeError: If an operation fails to be scheduled on any machine,
                          indicating a potential inconsistency or issue.

        Returns:
            HybridFlowshopLiteSchedule: A complete schedule object with all operations
                                    assigned to specific machines and time slots.
        """
        start_time_map = self.extract_stage_2_job_2_start_time_map(params, variables)

        schedule = self.create_empty_schedule_from_ins()
        for i in params.i_list:
            # For greedy machine assignment,
            # Sort operations at stage i by 1) their start time 2) their job index in self.j_list
            # This ensures that operations are assigned to machines in a consistent order.
            # This is important for the greedy assignment to work correctly.
            sorted_j_list = sorted(
                params.j_list,
                key=lambda j: (start_time_map[i][j], params.j_list.index(j)),
            )
            for j in sorted_j_list:
                start_time = start_time_map[i][j]
                schedule.add_operation_2_stage(
                    i, j, params.p[j, i], release_t=start_time
                )
        if make_semi_active:
            schedule.make_semi_active(self.stage_2_job_2_p_dict)

        return schedule

    def extract_start_end_time_map(
        self,
        params: Params,
        variables: CumulativeVars,
    ) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
        """
        Extracts the final start and end time maps from the solved model.

        This method orchestrates the post-solution process by first calling
        `create_schedule()` to build a complete, feasible schedule with specific
        machine assignments. It then extracts and returns the detailed start
        and end time dictionaries from that schedule.

        Returns:
            tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
                A tuple containing two dictionaries:
                - The first maps (job, stage, machine) to the operation start time.
                - The second maps (job, stage, machine) to the operation end time.
        """
        schedule = self.create_schedule(params, variables)
        return schedule.get_jik_2_start_time_map(), schedule.get_jik_2_end_time_map()

    def solve_current_cp_remaining_time_limit(
        self,
        computational_time: float | None,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        make_semi_active_after_cp: bool = False,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        is_initial_solution: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> tuple[HfsCpsatSolverReport, HybridFlowshopLiteSchedule | None]:
        """Solves the current CP model, creates a schedule, and registers the result.

        Args:
            computational_time (float | None): The maximum computational time in seconds.
                If None, uses the remaining time limit.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            no_improvement_timelimit (float | None, optional): If there is no improvement in this
                amount of time, the search will be stopped. If None, no timeout is set.
                Defaults to None.
            obj_value_is_valid (bool, optional): If True, adds the objective value log.
                Defaults to False.
            obj_bound_is_valid (bool, optional): If True, adds the objective bound log.
                Defaults to False.
            is_initial_solution (bool, optional): If True, indicates that this is an initial solution.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
            encode_cumulative_as_reservoir (bool | None, optional): Whether to encode cumulative constraints as reservoir constraints.
                Defaults to None.
            expand_reservoir_constraints (bool | None, optional): Whether to expand reservoir constraints.
                Defaults to None.
            expand_reservoir_using_circuit (bool | None, optional): Whether to expand reservoir constraints using a circuit.
                Defaults to None.
            interleave_search (bool | None, optional): Whether to interleave the search.
                Defaults to None.
            use_lns_only (bool | None, optional): Whether to use LNS-only mode.
                Defaults to None.
            cp_model_probing_level (int | None, optional): The level of probing for the CP model.
                Defaults to None.
            log_search_progress (bool, optional): If True, logs the search progress during solving.
                Defaults to False.

        Returns:
            tuple[HfsCpsatSolverReport, HybridFlowshopLiteSchedule | None]:
                A tuple containing the solver report and the created schedule (if a feasible solution is found).
        """
        sub_timer = ElapsedTimer()
        # Utilize the objective bound if available
        if (
            obj_value_is_valid
            and self.solution_manager.best_obj_bound is not None
            and not math.isnan(self.solution_manager.best_obj_bound)
        ):
            BaseModelBuilder.set_obj_lower_bound(
                self.cp_model, self.vars, self.solution_manager.best_obj_bound
            )

        # mdl_txt_path = self.get_file_path_for_subroutine("_cp_sat_model.txt")
        # self.cp_model.export_to_file(str(mdl_txt_path))

        _timelimit = self.get_remaining_time_limit(computational_time)
        solver_report = self.solve_cp_model_2(
            self.cp_model,
            _timelimit,
            solver_thread_cnt,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
            expand_reservoir_constraints=expand_reservoir_constraints,
            expand_reservoir_using_circuit=expand_reservoir_using_circuit,
            interleave_search=interleave_search,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
            e_timer=sub_timer,
            log_search_progress=log_search_progress,
            log_level_obj_bound=logging.INFO if obj_bound_is_valid else logging.DEBUG,
        )

        hfs_solver_report = HfsCpsatSolverReport.from_other(
            solver_report, is_init=is_initial_solution
        )

        subroutine_name = self._method_context_mgr.peek()
        call_context = self._get_call_context_of_current_method()
        hfs_solver_report = hfs_solver_report.copy(
            subroutine_name=subroutine_name,
            call_context=call_context,
        )

        # If the objective value or bound is not valid, use the best known values.
        report_updates: dict[str, Any] = {}
        if obj_value_is_valid:
            report_updates["obj_value"] = hfs_solver_report.obj_value
        else:
            report_updates["obj_value"] = self.solution_manager.best_obj_value
        if obj_bound_is_valid:
            report_updates["obj_bound"] = hfs_solver_report.obj_bound
        else:
            report_updates["obj_bound"] = self.solution_manager.best_obj_bound

        if report_updates:
            hfs_solver_report = hfs_solver_report.copy(
                obj_value=report_updates.get("obj_value"),
                obj_bound=report_updates.get("obj_bound"),
            )

        solution: HybridFlowshopLiteSchedule | None = None
        if hfs_solver_report.obj_value is None:
            if obj_value_is_valid:
                logging.warning("Failed to find a valid objective value.")
        else:
            if hfs_solver_report.is_feasible:
                solution = self.create_schedule(
                    self.params, self.vars, make_semi_active=make_semi_active_after_cp
                )
                if error_if_infeasible:
                    self.check_feasibility(solution.get_jik_2_start_time_map())
                # Ensure consistency between report and solution
                if solution.makespan != hfs_solver_report.obj_value:
                    # solution.makespan may be better than the reported objective value
                    # due to the way the CP solver reports values
                    # (e.g., due to presolve or how it handles bounds).
                    # In such cases, we update the report to reflect the actual solution value.
                    logging.info(
                        f"Objective value in report ({hfs_solver_report.obj_value}) "
                        f"does not match the makespan of the created solution ({solution.makespan}). "
                        f"Updating the report to reflect the solution's makespan."
                    )
                    hfs_solver_report = hfs_solver_report.copy(
                        elapsed_time=sub_timer.elapsed_sec, obj_value=solution.makespan
                    )
            else:
                logging.warning(
                    "No feasible solution found in the current CP model solving."
                )

        return hfs_solver_report, solution

    def solve_with_initial_solution(
        self,
        computational_time: float | None,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        make_semi_active_after_cp: bool = False,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> tuple[HfsCpsatSolverReport, HybridFlowshopLiteSchedule | None]:
        """Solves the current CP model using the incumbent solution as a hint.

        Args:
            computational_time (float | None): The maximum computational time in seconds.
                If None, uses the remaining time limit.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            no_improvement_timelimit (float | None, optional): If there is no improvement in this
                amount of time, the search will be stopped. If None, no timeout is set.
                Defaults to None.
            obj_value_is_valid (bool, optional): If True, adds the objective value log.
                Defaults to False.
            obj_bound_is_valid (bool, optional): If True, adds the objective bound log.
                Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
            encode_cumulative_as_reservoir (bool | None, optional): Whether to encode cumulative constraints as reservoir constraints.
                Defaults to None.
            expand_reservoir_constraints (bool | None, optional): Whether to expand reservoir constraints.
                Defaults to None.
            expand_reservoir_using_circuit (bool | None, optional): Whether to expand reservoir constraints using a circuit.
                Defaults to None.
            interleave_search (bool | None, optional): Whether to interleave the search.
                Defaults to None.
            use_lns_only (bool | None, optional): Whether to use LNS-only mode.
                Defaults to None.
            cp_model_probing_level (int | None, optional): The level of probing for the CP model.
                Defaults to None.
            log_search_progress (bool, optional): If True, logs the search progress during solving.
                Defaults to False.

        Returns:
            tuple[HfsCpsatSolverReport, HybridFlowshopLiteSchedule | None]:
                A tuple containing the solver report and the created schedule (if a feasible solution is found).
        """
        incumbent_solution = self.solution_manager.get_incumbent()
        is_initial_run = incumbent_solution is None

        if incumbent_solution:
            self.cp_model.clear_hints()
            logging.info(
                "Applying incumbent solution with objValue "
                f"{incumbent_solution.makespan} as a hint."
            )
            BaseModelBuilder.apply_start_hints_from_start_time_map(
                self.cp_model,
                self.params,
                self.vars,
                incumbent_solution.get_jik_2_start_time_map(),
            )
            BaseModelBuilder.apply_end_hints_from_end_time_map(
                self.cp_model,
                self.params,
                self.vars,
                incumbent_solution.get_jik_2_end_time_map(),
            )
            self.cp_model.add_hint(self.vars.makespan, incumbent_solution.makespan)

        return self.solve_current_cp_remaining_time_limit(
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            make_semi_active_after_cp=make_semi_active_after_cp,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            is_initial_solution=is_initial_run,
            encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
            expand_reservoir_constraints=expand_reservoir_constraints,
            expand_reservoir_using_circuit=expand_reservoir_using_circuit,
            interleave_search=interleave_search,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
            log_search_progress=log_search_progress,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    # End solver call methods

    # Start repeat

    def repeat_while_improvement(
        self,
        routine_data: DynamicDataObject,
        n_repeats: int | None = None,
        max_no_improve: int | None = None,
    ):
        """
        Repeats the execution of a routine a specified number of times.

        Args:
            routine_data (DynamicDataObject): The routine data to be executed.
            n_repeats (int | None, optional): Number of times to repeat the routine.
                If None, repeats until the stopping condition or the no-improvement
                limit is met. Defaults to None.
            max_no_improve (int | None, optional): Maximum number of consecutive
                non-improving iterations before stopping.
                If 0, stops after the first non-improving iteration.
                If None or negative, treated as 0.
                Defaults to None.
        """
        _max_no_improve: int = (
            0 if max_no_improve is None or max_no_improve < 0 else max_no_improve
        )

        incumbent_sol = self.solution_manager.get_incumbent()
        if incumbent_sol is None:
            obj_before = math.inf
        else:
            obj_before = incumbent_sol.makespan

        no_improve_count = 0
        i = 0
        while n_repeats is None or i < n_repeats:
            if self.is_stopping_condition():
                logging.info(
                    "[Repeat] Stopping condition met at iteration %d/%s.",
                    i + 1,
                    n_repeats if n_repeats is not None else "inf",
                )
                break
            logging.info(
                "[Repeat] Starting repeat %d/%s.",
                i + 1,
                n_repeats if n_repeats is not None else "inf",
            )

            subroutine_name = f"reps_{i + 1:03d}"
            with self.temporarily_extended_context(subroutine_name):
                self._run_flow(DynamicDataObject.from_obj(routine_data))

            incumbent_sol = self.solution_manager.get_incumbent()
            if incumbent_sol is None:
                obj_after = math.inf
            else:
                obj_after = incumbent_sol.makespan

            if float_a_stl_b(obj_after, obj_before):
                no_improve_count = 0
                logging.info(
                    f"[Repeat] Improvement observed ({obj_before} -> {obj_after}). Continuing."
                )
                obj_before = obj_after
            else:
                logging.info(
                    f"[Repeat] No improvement observed ({obj_before} -> {obj_after})."
                )
                no_improve_count += 1
                if no_improve_count > _max_no_improve:
                    logging.info(
                        f"[Repeat] Max no-improve reached ({_max_no_improve}). Stopping repeats."
                    )
                    break
            i += 1
