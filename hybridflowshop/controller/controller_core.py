import datetime
import logging
import math
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
from mbls.cpsat.callbacks import ValueBoundPair
from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from routix.util.comparison import float_a_leq_b, float_equals
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder, CumulativeVars
from hybridflowshop.cpsat_model_2.params import Params
from hybridflowshop.report import HfsCpsatSolverReport
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

    log_solver_level: int
    """Logging level for the solver output."""

    log_search_progress: bool
    """Whether to log search progress during CP solving."""

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
        self.log_solver_level = logging.INFO  # TODO: make it configurable
        self.log_search_progress = False  # TODO: make it configurable

        # Frequently used parameters
        self.job_2_stage_2_p_dict = self.instance.p_manager.job_2_stage_2_value_map(
            self.instance.job_id_list, self.instance.stage_id_list
        )
        """Job name -> stage name -> processing time map"""
        self.stage_2_job_2_p_dict = self.instance.p_manager.stage_2_job_2_value_map(
            self.instance.stage_id_list, self.instance.job_id_list
        )
        """Stage name -> job name -> processing time map"""

        logging.info(
            f"Start solving {self.instance.name} using CP model class:"
            f" {self.cp_model_class.__module__}.{self.cp_model_class.__name__}",
        )

    # Start abstract getters

    def create_base_cp_model(self, **kwargs) -> CustomCpModel:
        builder = BaseModelBuilder()
        horizon = self.get_horizon()
        mdl, params, variables = builder.build(self.instance, horizon)
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

    # Start visualization

    def draw_gantt(
        self, schedule: HybridFlowshopLiteSchedule, output_path: Path | None = None
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
                schedule.get_start_time_map(),
                schedule.get_end_time_map(),
                self.instance.job_id_list,
            )

    def draw_incumbent_gantt(self, output_path: Path | None = None) -> None:
        """Draws the Gantt chart of the incumbent solution.

        Args:
            output_path (Path | None, optional): The output path for the Gantt chart image. Defaults to None.
        """
        incumbent_solution = self.solution_manager.get_incumbent()
        if isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            self.draw_gantt(incumbent_solution, output_path=output_path)
        else:
            logging.warning("No incumbent solution available to draw Gantt chart.")

    # End visualization

    def run(self, flow_resume_idx: int = -1) -> None:
        """Overrides the run method to execute the subroutine flow.

        Args:
            flow_resume_idx (int, optional): The index to resume the flow from. Defaults to -1.
        """
        if isinstance(self._subroutine_flow, Sequence) and not isinstance(
            self._subroutine_flow, (str, bytes)
        ):
            for idx, subroutine_data in enumerate(self._subroutine_flow):
                if idx < flow_resume_idx:
                    if (
                        subroutine_data.get("method", "")
                        in self.method_names_to_run_before_resume
                    ):
                        e_timer = ElapsedTimer()
                        self._run_flow(subroutine_data)
                        virtual_dt = datetime.datetime.now() - datetime.timedelta(
                            seconds=e_timer.elapsed_sec
                        )
                        self.timer.set_start_time(virtual_dt)
                    else:
                        self._run_flow(subroutine_data, skip_method_call=True)
                else:
                    self._run_flow(subroutine_data)
        else:
            logging.warning(
                "Subroutine flow is not a sequence; running as a single step."
            )
            self._run_flow(self._subroutine_flow)
        self.post_run_process()

    # Start post-run process

    def post_run_process(self) -> None:
        """
        Finalizes the run by checking the feasibility of the incumbent solution
        and releasing log handlers.
        """
        incumbent = self.solution_manager.get_incumbent()
        if incumbent:
            self.check_feasibility(incumbent.get_start_time_map())
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

        validate_duration(start_time_map, end_time_map, self.job_2_stage_2_p_dict)
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
        e_timer: ElapsedTimer | None = None,
        print_on_obj_value_update: bool = False,
        print_on_obj_bound_update: bool = False,
        log_level_obj_value: int = logging.INFO,
        log_level_obj_bound: int = logging.INFO,
        last_timestamp_note: Any | None = None,
    ) -> CpsatSolverReport:
        from ..cpsat_model_2.solver import SolveConfig, configure_solver

        if e_timer is None:
            e_timer = self.timer

        solve_cfg = SolveConfig(
            log_search_progress=self.log_search_progress,
            time_limit_s=computational_time,
            num_workers=solver_thread_cnt,
            keep_all_feasible_solutions_in_presolve=keep_all_feasible_solutions_in_presolve,
            random_seed=self.random_seed,
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

        def get_obj_value_records() -> list[tuple[float, float]]:
            """Returns the recorded objective values and elapsed times.

            Returns:
                list[tuple[float, float]]: A list of tuples containing (elapsed time, objective value).
            """
            return_list: list[tuple[float, float]] = []
            list_by_value_recorder: list[tuple[float, ValueBoundPair]] = (
                obj_value_recorder.entries
            )
            for entry in list_by_value_recorder:
                return_list.append((entry[0], entry[1].value))
            return return_list

        obj_value_records: list[tuple[float, float]] = []
        if obj_value_is_valid:
            obj_value_records = get_obj_value_records()
            if cpsat_status.is_feasible:
                obj_value_records.append((last_timestamp, obj_value))
            self.extend_obj_value_log(
                obj_value_records, is_maximize=self.cp_model.is_maximize()
            )
            # Record value for the last timestamp if it is the same as the last value
            # and is not recorded for the last timestamp
            if (
                obj_value == self.obj_store.get_last_obj_value()
                and (last_timestamp, obj_value) not in obj_value_records
            ):
                self.add_obj_value_log(last_timestamp, obj_value, is_maximize=None)

        def get_obj_bound_records() -> list[tuple[float, float]]:
            """Returns the recorded objective bounds and elapsed times.

            Returns:
                list[tuple[float, float]]: A list of tuples containing (elapsed time, objective bound).
            """
            timestamp_list = []
            timestamp_2_bound_map: dict[float, float] = {}

            list_by_bound_recorder: list[tuple[float, float]] = (
                obj_bound_recorder.elapsed_time_and_bound
            )
            for b_entry in list_by_bound_recorder:
                timestamp = b_entry[0]
                bound = b_entry[1]
                if timestamp not in timestamp_list:
                    timestamp_list.append(timestamp)
                timestamp_2_bound_map[timestamp] = bound

            list_by_value_recorder: list[tuple[float, ValueBoundPair]] = (
                obj_value_recorder.entries
            )
            for v_entry in list_by_value_recorder:
                timestamp = v_entry[0]
                bound = v_entry[1].bound
                if timestamp not in timestamp_list:
                    timestamp_list.append(timestamp)
                if timestamp not in timestamp_2_bound_map:
                    timestamp_2_bound_map[timestamp] = bound

            timestamp_list.sort()
            return [
                (timestamp, timestamp_2_bound_map[timestamp])
                for timestamp in timestamp_list
            ]

        obj_bound_records: list[tuple[float, float]] = []
        if obj_bound_is_valid:
            obj_bound_records = get_obj_bound_records()
            if cpsat_status.is_feasible:
                obj_bound_records.append((last_timestamp, obj_bound))
            self.extend_obj_bound_log(obj_bound_records, is_maximize=False)
            # Record bound for the last timestamp if it is the same as the last bound
            # and is not recorded for the last timestamp
            if (
                obj_bound == self.obj_store.get_last_obj_bound()
                and (last_timestamp, obj_bound) not in obj_bound_records
            ):
                self.add_obj_bound_log(last_timestamp, obj_bound, is_maximize=None)

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
        self, params: Params, variables: CumulativeVars
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
        self, params: Params, variables: CumulativeVars
    ) -> dict[str, dict[str, int]]:
        end_time_map: dict[str, dict[str, int]] = {}
        """stage ID -> job ID -> end time"""
        for i in params.i_list:
            end_time_map[i] = {}
            for j in params.j_list:
                end_value = self.solver.Value(variables.op_end[j, i])
                end_time_map[i][j] = end_value
        return end_time_map

    def create_empty_schedule_from_ins(self) -> HybridFlowshopLiteSchedule:
        """
        Creates an empty HybridFlowshopLiteSchedule for the problem instance.

        Returns:
            HybridFlowshopLiteSchedule: An empty schedule object.
        """
        return HybridFlowshopLiteSchedule(
            self.instance.job_id_list,
            self.instance.stage_id_list,
            self.instance.stage_2_machines_map,
        )

    def create_schedule(
        self, params: Params, variables: CumulativeVars
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
        return schedule.get_start_time_map(), schedule.get_end_time_map()

    def solve_current_cp_remaining_time_limit(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        is_initial_solution: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> tuple[HfsCpsatSolverReport, HybridFlowshopLiteSchedule | None]:
        """Solves the current CP model, creates a schedule, and registers the result.

        Args:
            computational_time (float): The maximum computational time in seconds.
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
        """
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
            log_level_obj_bound=logging.INFO if obj_bound_is_valid else logging.DEBUG,
        )

        hfs_solver_report = HfsCpsatSolverReport.from_other(
            solver_report, is_init=is_initial_solution
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
                solution = self.create_schedule(self.params, self.vars)
                if error_if_infeasible:
                    self.check_feasibility(solution.get_start_time_map())
                # Ensure consistency between report and solution
                if solution.makespan != hfs_solver_report.obj_value:
                    raise ValueError(
                        "Objective value mismatch between solver report and schedule makespan."
                    )
            else:
                logging.warning(
                    "No feasible solution found in the current CP model solving."
                )

        return hfs_solver_report, solution

    def solve_with_initial_solution(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> tuple[HfsCpsatSolverReport, HybridFlowshopLiteSchedule | None]:
        """Solves the current CP model using the incumbent solution as a hint.

        Args:
            computational_time (float): The maximum computational time in seconds.
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

        Raises:
            TypeError: If the incumbent solution is not compatible with the CP model.
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
                incumbent_solution.get_start_time_map(),
            )

        return self.solve_current_cp_remaining_time_limit(
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            is_initial_solution=is_initial_run,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    # End solver call methods
