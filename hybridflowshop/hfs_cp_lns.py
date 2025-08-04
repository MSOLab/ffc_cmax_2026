import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from mbls.cpsat import CpsatStatus, CpSubroutineController, ObjValueBoundStore
from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from routix.io import object_to_yaml
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.utils import tuple_to_pyyaml_key

from .cp_2023_naderi_cumulative import CP2023NaderiCumulative
from .cp_2023_naderi_optional_interval import CP2023NaderiOptionalInterval
from .cp_cumulative_optional_hybrid import CPCumulativeOptionalHybrid
from .cp_optional_interval_master_timevar import CPOptionalIntervalMasterTimevar
from .painter.gantt import GanttPlotter
from .report import HfsCpsatSolverReport, HfsSubroutineReport
from .scheduling.hybrid_flowshop_schedule import HybridFlowshopSchedule
from .solution_manager import HfsSolutionManager

HfsMathModel = CP2023NaderiCumulative


class HybridFlowShopCpLnsController(
    CpSubroutineController[HybridFlowshopParameters, HfsMathModel, StoppingCriteria]
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
            HfsMathModel,
            subroutine_flow,
            stopping_criteria,
        )
        self.solution_manager = HfsSolutionManager()

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
            "Using CP model class: %s.%s",
            self.cp_model_class.__module__,
            self.cp_model_class.__name__,
        )

    # Start abstract getters

    def create_base_cp_model(self) -> HfsMathModel:
        if "horizon" not in self.shared_param_dict:
            raise ValueError("Horizon not found in shared parameters.")
        horizon = self.shared_param_dict["horizon"]
        return self.cp_model_class.from_instance(self.instance, horizon)

    # End abstract getters

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

    def is_stopping_condition(self) -> bool:
        return self.time_is_up()

    def time_is_up(self) -> bool:
        # If total elapsed time exceeds the stopping criteria
        if self.timer.elapsed_sec >= self.stopping_criteria.timelimit:
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
        self, schedule: HybridFlowshopSchedule, output_path: Path | None = None
    ):
        """Draws the Gantt chart of the given schedule.

        Args:
            schedule (HybridFlowshopSchedule): The schedule to draw.
            output_path (Path | None, optional): The output path for the Gantt chart image. Defaults to None.
        """
        if output_path is None:
            output_path = self.get_file_path_for_subroutine("_gantt.png")
        if isinstance(schedule, HybridFlowshopSchedule):
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
        if isinstance(incumbent_solution, HybridFlowshopSchedule):
            self.draw_gantt(incumbent_solution, output_path=output_path)
        else:
            logging.warning("No incumbent solution available to draw Gantt chart.")

    # End visualization

    # Start subroutine definition

    def solve_current_cp_remaining_time_limit(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        is_initial_solution: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Solves the current CP model, creates a schedule, and registers the result.

        Args:
            computational_time (float): The maximum computational time in seconds.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
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
        _timelimit = self.get_remaining_time_limit(computational_time)

        # Utilize the objective bound if available
        if obj_value_is_valid and self.solution_manager.best_obj_bound is not None:
            self.cp_model.set_obj_lower_bound(self.solution_manager.best_obj_bound)

        # mdl_txt_path = self.get_file_path_for_subroutine("_cp_sat_model.txt")
        # self.cp_model.export_to_file(str(mdl_txt_path))

        solver_report = self.solve_current_cp_model(
            _timelimit,
            solver_thread_cnt,
            random_seed=self.random_seed,
            e_timer=self.timer,
            log_level_obj_value=logging.INFO,
            log_level_obj_bound=logging.INFO,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

        hfs_solver_report = HfsCpsatSolverReport.from_other(
            solver_report, is_init=is_initial_solution
        )

        # Create a new report with None values if the objective or bound is not valid.
        # This is necessary because the report object is frozen.
        report_updates: dict[str, Any] = {}
        if not obj_value_is_valid:
            report_updates["obj_value"] = None
        if not obj_bound_is_valid:
            report_updates["obj_bound"] = None

        if report_updates:
            hfs_solver_report = hfs_solver_report.replace(**report_updates)

        solution: HybridFlowshopSchedule | None = None
        if hfs_solver_report.is_feasible:
            solution = self.cp_model.create_schedule()
            if error_if_infeasible:
                self.check_feasibility(solution.get_start_time_map())
            # Ensure consistency between report and solution
            if solution.makespan != hfs_solver_report.obj_value:
                raise ValueError(
                    "Objective value mismatch between solver report and schedule makespan."
                )
        # Register the solution
        was_updated = self.solution_manager.register(hfs_solver_report, solution)
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def solve_with_initial_solution(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Solves the current CP model using the incumbent solution as a hint.
        """
        incumbent_solution = self.solution_manager.get_incumbent()
        is_initial_run = incumbent_solution is None

        if incumbent_solution:
            self.cp_model.clear_hints()
            logging.info(
                "Applying incumbent solution with objValue "
                f"{incumbent_solution.makespan} as a hint."
            )
            if isinstance(self.cp_model, CP2023NaderiCumulative):
                self.cp_model.add_start_hints_from_start_time_map(
                    incumbent_solution.get_start_time_map(),
                    ignore_integrity_check=True,
                )
            elif isinstance(
                self.cp_model,
                (
                    CP2023NaderiOptionalInterval,
                    CPOptionalIntervalMasterTimevar,
                    CPCumulativeOptionalHybrid,
                ),
            ):
                self.cp_model.add_start_and_present_hints_from_start_time_map(
                    incumbent_solution.get_start_time_map(),
                    ignore_integrity_check=True,
                )
            else:
                raise TypeError(f"Unsupported CP model type: {type(self.cp_model)}")

        self.solve_current_cp_remaining_time_limit(
            computational_time,
            solver_thread_cnt,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            is_initial_solution=is_initial_run,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    # Subroutine: solve base CP model

    def solve_base_cp_model(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        is_initial_solution: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Solve the base CP model for the hybrid flow shop problem.

        This method resets the CP model and solves it with the given computational time and number of workers.
        - If `is_initial_solution` is True, the solution is treated as the initial solution (e.g., for logging or summary purposes).
        - If `is_initial_solution` is False, the incumbent solution (if it exists) is applied as a hint to the CP model before solving.
        - If `draw_gantt` is True, a Gantt chart of the solution is generated after solving.

        Args:
            computational_time (float): The maximum computational time in seconds for solving the CP model.
            solver_thread_cnt (int): The number of parallel workers (threads) to use during search.
            is_initial_solution (bool, optional): If True, marks this run as producing the initial solution (affects summary/logging). Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution after solving. Defaults to False.
        """
        self.cp_model.delete_added_constraints()
        if is_initial_solution:
            self.solve_current_cp_remaining_time_limit(
                computational_time,
                solver_thread_cnt,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                is_initial_solution=True,
                error_if_infeasible=True,
                draw_gantt=draw_gantt,
            )
        else:
            # If it is not an initial solution, apply the incumbent solution as a hint
            self.solve_with_initial_solution(
                computational_time,
                solver_thread_cnt,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                error_if_infeasible=True,
                draw_gantt=draw_gantt,
            )

    # Helper method for LNS-CP

    def freeze_solve_reset(
        self,
        freeze_method: Callable,
        computational_time: float,
        solver_thread_cnt: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Apply the freeze method, solve, and reset the model.

        Args:
            freeze_method (Callable): A callable that applies the freeze method to the CP model.
            computational_time (float): The maximum computational time in seconds.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            obj_value_is_valid (bool, optional): If True, adds the objective value log.
                Defaults to False.
            obj_bound_is_valid (bool, optional): If True, adds the objective bound log.
                Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        freeze_method()
        self.solve_with_initial_solution(
            computational_time,
            solver_thread_cnt,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )
        self.cp_model.delete_added_constraints()

    # Subroutine: Time window operator

    def time_window_search(
        self,
        rho: float,
        computational_time: float,
        solver_thread_cnt: int,
        error_if_infeasible=False,
        draw_gantt: bool = False,
    ):
        """Time window search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of makespan to define the window size.
                For example, 0.2 means 20% of makespan.
            computational_time (float): The maximum computational time in seconds.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        self.freeze_solve_reset(
            lambda: self.apply_time_window_operator(rho),
            computational_time,
            solver_thread_cnt,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_time_window_operator(self, rho: float):
        """Apply the Time Window Operator to the current CP model.

        Args:
            rho (float): Fraction of makespan to define the window size.
                For example, 0.2 means 20% of makespan.

        Raises:
            ValueError: If no incumbent solution is available.
            ValueError: If the incumbent solution is not a valid HybridFlowshopSchedule instance.
            ValueError: If no end times are available for Time Window Operator.
            ValueError: If the window length is not positive.
        """
        logging.info(f"Applying time window operator with rho={rho}")
        if not self.solution_manager.has_incumbent():
            raise ValueError(
                "No incumbent solution available for Time Window Operator."
            )
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopSchedule instance."
            )
        start_time_map = incumbent_solution.get_start_time_map()
        end_time_map = incumbent_solution.get_end_time_map()

        # 1. Calculate makespan (C_max)
        all_end_time_map = list(end_time_map.values())
        if not all_end_time_map:
            raise ValueError("No end times available for Time Window Operator.")
        C_max = max(all_end_time_map)

        # 2. Select random time window
        window_length = int(rho * C_max)
        if window_length <= 0:
            raise ValueError("Window length must be positive.")

        window_start = random.randint(0, max(0, C_max - window_length))
        window_end = window_start + window_length

        logging.info(
            f"[Time Window] Selected window: [{window_start}, {window_end}] (C_max={C_max})"
        )

        # 3. Classify operations
        out_of_window_ops = set()

        for key in start_time_map:
            s_time = start_time_map[key]
            e_time = end_time_map[key]
            if not self.is_within_window(
                s_time, window_start, window_end
            ) and not self.is_within_window(e_time, window_start, window_end):
                out_of_window_ops.add(key)

        # 4. Fix machine assignment and precedence for out-of-window operations
        stage_mc_to_jobs: dict[tuple[str, str], list[str]] = defaultdict(list)

        for j, i, k in out_of_window_ops:
            # self.cp_model.add_fixed_machine_assignment_constraint(j, i, k)
            stage_mc_to_jobs[(i, k)].append(j)

        for (i, k), jobs in stage_mc_to_jobs.items():
            # Start time 기준 정렬
            jobs_sorted = sorted(jobs, key=lambda j: start_time_map[(j, i, k)])
            for j1, j2 in zip(jobs_sorted[:-1], jobs_sorted[1:]):
                if isinstance(self.cp_model, CP2023NaderiCumulative):
                    self.cp_model.add_operation_weak_precedence_constraint(j1, j2, i)
                elif isinstance(
                    self.cp_model,
                    (
                        CP2023NaderiOptionalInterval,
                        CPOptionalIntervalMasterTimevar,
                        CPCumulativeOptionalHybrid,
                    ),
                ):
                    self.cp_model.add_fixed_operation_precedence_constraint(
                        j1, j2, i, k
                    )
                else:
                    raise TypeError(f"Unsupported CP model type: {type(self.cp_model)}")

    @staticmethod
    def is_within_window(time: int, window_start: int, window_end: int) -> bool:
        """Check if a given time is within the specified window."""
        return window_start <= time <= window_end

    # Subroutine: Block operator

    def block_search(
        self,
        rho: float,
        computational_time: float,
        solver_thread_cnt: int,
        error_if_infeasible=False,
        draw_gantt: bool = False,
    ):
        """Block search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            computational_time (float): The maximum computational time in seconds.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """

        self.freeze_solve_reset(
            lambda: self.apply_block_operator(rho),
            computational_time,
            solver_thread_cnt,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_block_operator(self, rho: float):
        """Apply the Block Operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.

        Raises:
            ValueError: If no incumbent solution is available.
            ValueError: If the incumbent solution is not a valid HybridFlowshopSchedule instance.
            ValueError: If no end times are available for Time Window Operator.
            ValueError: If the window length is not positive.
        """
        logging.info(f"Applying block operator with rho={rho}")
        if not self.solution_manager.has_incumbent():
            raise ValueError(
                "No incumbent solution available for Time Window Operator."
            )
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopSchedule instance."
            )
        start_time_map = incumbent_solution.get_start_time_map()
        end_time_map = incumbent_solution.get_end_time_map()

        if not start_time_map or not end_time_map:
            raise ValueError("No solution available for block operator.")

        all_ops = list(start_time_map.keys())  # TODO: 순서 유지되는지 확인
        total_ops = len(all_ops)
        num_to_select = max(1, int(rho * total_ops))

        # Step 1: Start from a random operation
        seed_op = random.choice(all_ops)
        selected_ops = set([seed_op])
        queue = [seed_op]

        # Step 2: Expand to overlapping operations
        while queue and len(selected_ops) < num_to_select:
            current_op = queue.pop(0)
            cs, ce = start_time_map[current_op], end_time_map[current_op]
            for op in all_ops:
                if op in selected_ops:
                    continue
                os, oe = start_time_map[op], end_time_map[op]
                if self.is_overlap(cs, ce, os, oe):
                    selected_ops.add(op)
                    queue.append(op)
                if len(selected_ops) >= num_to_select:
                    break

        logging.info(
            f"[Block Operator] Selected {len(selected_ops)} overlapping ops (target={num_to_select})"
        )

        # Step 3: Out-of-block 작업들에 대해 고정 제약 추가
        out_of_block_ops = set(all_ops) - selected_ops
        stage_mc_to_jobs = defaultdict(list)

        for j, i, k in out_of_block_ops:
            # self.cp_model.add_fixed_machine_assignment_constraint(j, i, k)
            stage_mc_to_jobs[(i, k)].append(j)

        for (i, k), jobs in stage_mc_to_jobs.items():
            jobs_sorted = sorted(jobs, key=lambda j: start_time_map[(j, i, k)])
            for j1, j2 in zip(jobs_sorted[:-1], jobs_sorted[1:]):
                if isinstance(self.cp_model, CP2023NaderiCumulative):
                    self.cp_model.add_operation_weak_precedence_constraint(j1, j2, i)
                elif isinstance(
                    self.cp_model,
                    (
                        CP2023NaderiOptionalInterval,
                        CPOptionalIntervalMasterTimevar,
                        CPCumulativeOptionalHybrid,
                    ),
                ):
                    self.cp_model.add_fixed_operation_precedence_constraint(
                        j1, j2, i, k
                    )
                else:
                    raise TypeError(f"Unsupported CP model type: {type(self.cp_model)}")

    @staticmethod
    def is_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
        """Check if two time intervals overlap."""
        return not (e1 <= s2 or e2 <= s1)

    # Subroutine: Johnson-based Heuristic for initialization

    def dispatch_by_job_stage_time(
        self,
        job_sequence: list[str],
        schedule: HybridFlowshopSchedule,
        draw_gantt: bool = False,
    ):
        """
        Dispatches jobs according to the given sequence and registers the resulting
        solution with the solution manager.

        This is the classic job-major (job -> stage -> time) dispatching strategy.

        Args:
            job_sequence (list[str]): The sequence of job IDs to be dispatched (dispatch order).
            schedule (HybridFlowshopSchedule): The schedule to which jobs are dispatched.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        for idx, j in enumerate(job_sequence):
            schedule.dispatch_job_by_stages(
                j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
            )
            # TODO: uncomment only for debug purpose
            # output_path = self.get_file_path_for_subroutine(f"_gantt_{idx}_{j}.png")
            # self.draw_gantt(schedule, output_path=output_path)

        # Create report and register the new solution
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(schedule.makespan),
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log and draw Gantt chart if the solution is an improvement
        if was_updated:
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(
                log_time, float(schedule.makespan), is_maximize=False
            )
            if draw_gantt:
                self.draw_incumbent_gantt()

    def dispatch_by_stage_time_job(
        self,
        job_sequence: list[str],
        schedule: HybridFlowshopSchedule,
        draw_gantt: bool = False,
    ):
        """
        For each stage (in order), dispatch jobs in the given job_sequence order.
        For each job in the sequence, schedule its operation in the current stage to the earliest available machine.

        This is the stage-major (stage -> job -> time) dispatching strategy.
        In hybrid flowshop, this is equivalent to dispatching jobs by stage, then by job, then by earliest time.

        Args:
            job_sequence (list[str]): The sequence of job IDs to be dispatched (dispatch order).
            schedule (HybridFlowshopSchedule): The schedule to which jobs are dispatched.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        for idx, i in enumerate(self.instance.stage_id_list):
            schedule.dispatch_stage_by_jobs(
                i, job_sequence, self.stage_2_job_2_p_dict[i]
            )
            # TODO: uncomment only for debug purpose
            # output_path = self.get_file_path_for_subroutine(f"_gantt_{idx}_{j}.png")
            # self.draw_gantt(schedule, output_path=output_path)

        # Create report and register the new solution
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(schedule.makespan),
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log and draw Gantt chart if the solution is an improvement
        if was_updated:
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(
                log_time, float(schedule.makespan), is_maximize=False
            )
            if draw_gantt:
                self.draw_incumbent_gantt()

    def construct_solution_by_incremental_cp(
        self,
        job_sequence: list[str],
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        max_time_per_add: float | None = None,
        no_improvement_timelimit: float | None = None,
        is_init: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Constructs a solution by incrementally solving CP submodels and registers
        the final result.

        This method builds a feasible schedule by adding jobs one by one
        according to the provided job sequence. At each step, a sub-CP model
        is constructed for the current subset of jobs and solved with the
        given time limit. Previously scheduled jobs are frozen to guide the solver.

        Args:
            job_sequence (list[str]): The sequence of job IDs to be added.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            added_batch_size (int, optional): The number of jobs to add in each iteration.
                Defaults to 1.
            max_time_per_add (float | None, optional): Time limit (in seconds) for solving each incremental subproblem.
                If None, uses the remaining time limit. Defaults to None.
            no_improvement_timelimit (float | None, optional): If there is no improvement for this
                amount of time, the search will be stopped. If None, no timeout is set.
                Defaults to None.
            is_init (bool, optional): If True, indicates that this is an initial solution.
                Defaults to False.
            error_if_infeasible (bool, optional): If True, raises an error if the solution is infeasible.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws a Gantt chart of the solution.
                Defaults to False.

        Raises:
            TypeError: If this method is called on a CP model that does not support incremental solving.
        """
        sub_timer = ElapsedTimer()
        last_solution: HybridFlowshopSchedule | None = None

        sub_obj_store = ObjValueBoundStore[float]()
        """Subroutine-specific objective store"""
        sub_obj_store.obj_value_series.name = "ObjVal after dispatch"
        sub_obj_store.obj_bound_series.name = "ObjVal before dispatch"

        job_cnt = len(job_sequence)
        sequence_of_job_sublist = [
            job_sequence[i : i + added_batch_size]
            for i in range(0, len(job_sequence), added_batch_size)
        ]

        job_subset: set[str] = set()
        for job_sublist in sequence_of_job_sublist:
            job_subset.update(job_sublist)
            job_subset_cnt = len(job_subset)
            all_jobs_are_included = job_subset_cnt == job_cnt

            # Solution of dispatching job_sublist by jobs to the schedule of last_solution
            partial_sol_dj: HybridFlowshopSchedule = (
                HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
                    self.instance.stage_2_machines_map
                )
                if last_solution is None
                else last_solution.deepcopy()
            )
            for j in job_sublist:
                partial_sol_dj.dispatch_job_by_stages(
                    j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
                )

            # Solution of dispatching job_sublist by stages to the schedule of last_solution
            partial_sol_ds: HybridFlowshopSchedule = (
                HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
                    self.instance.stage_2_machines_map
                )
                if last_solution is None
                else last_solution.deepcopy()
            )
            for i in self.instance.stage_id_list:
                partial_sol_ds.dispatch_stage_by_jobs(
                    i, job_sublist, self.stage_2_job_2_p_dict[i]
                )

            # Select the best partial solution
            partial_sol_best = (
                partial_sol_dj
                if partial_sol_dj.makespan <= partial_sol_ds.makespan
                else partial_sol_ds
            )

            sub_cp_mdl = self.cp_model.create_problem_of_job_subset(job_subset)
            if last_solution is not None:
                if isinstance(sub_cp_mdl, CP2023NaderiCumulative):
                    # Freeze operation precedences
                    # sub_cp_mdl.add_stage_ops_weak_precedence_constraints_from_start_time_map(
                    #     last_solution.get_start_time_map(), ignore_integrity_check=True
                    # )
                    sub_cp_mdl.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
                        last_solution, ignore_integrity_check=True
                    )
                    # Apply hint
                    sub_cp_mdl.add_start_hints_from_start_time_map(
                        partial_sol_best.get_start_time_map(),
                        ignore_integrity_check=True,
                    )
                elif isinstance(
                    self.cp_model,
                    (
                        CP2023NaderiOptionalInterval,
                        CPOptionalIntervalMasterTimevar,
                        CPCumulativeOptionalHybrid,
                    ),
                ):
                    # Freeze operation precedences
                    sub_cp_mdl.add_fixed_machine_and_ops_precedence_constraints_from_start_time_map(
                        last_solution.get_start_time_map(), ignore_integrity_check=True
                    )
                    # Apply hint
                    sub_cp_mdl.add_start_and_present_hints_from_start_time_map(
                        partial_sol_best.get_start_time_map(),
                        ignore_integrity_check=True,
                    )
                else:
                    raise TypeError(f"Unsupported CP model type: {type(self.cp_model)}")

            # mdl_txt_path = self.get_file_path_for_subroutine(
            #     f"_{job_subset_cnt}_cp_sat_model.txt"
            # )
            # sub_cp_mdl.export_to_file(str(mdl_txt_path))

            _timelimit = self.get_remaining_time_limit(max_time_per_add)
            iter_report = self.solve_cp_model(
                sub_cp_mdl,
                _timelimit,
                solver_thread_cnt,
                random_seed=self.random_seed,
                no_improvement_timelimit=no_improvement_timelimit,
                e_timer=sub_timer,
                obj_value_is_valid=all_jobs_are_included,
            )
            last_timestamp = sub_timer.elapsed_sec

            if iter_report.is_feasible:
                # Update the last solution
                last_solution = sub_cp_mdl.create_schedule()
                # If last_solution is not better than partial_dispatched_sol,
                if last_solution is None:
                    last_solution = partial_sol_best
                elif last_solution.makespan >= partial_sol_best.makespan:
                    # Use the partial dispatched solution
                    last_solution = partial_sol_best

                # Dispatch remaining jobs to create a schedule feasible to the original problem
                all_dispatched_sol_dj = last_solution.deepcopy()
                remaining_jobs = [j for j in job_sequence if j not in job_subset]
                for j in remaining_jobs:
                    all_dispatched_sol_dj.dispatch_job_by_stages(
                        j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
                    )

                all_dispatched_sol_ds = last_solution.deepcopy()
                remaining_jobs = [j for j in job_sequence if j not in job_subset]
                for i in self.instance.stage_id_list:
                    all_dispatched_sol_ds.dispatch_stage_by_jobs(
                        i, remaining_jobs, self.stage_2_job_2_p_dict[i]
                    )

                all_dispatched_sol_best = (
                    all_dispatched_sol_dj
                    if all_dispatched_sol_dj.makespan <= all_dispatched_sol_ds.makespan
                    else all_dispatched_sol_ds
                )

                # TODO: uncomment only for debug purpose
                output_path = self.get_file_path_for_subroutine(
                    f"_gantt_{job_subset_cnt}_1_partial_dispatched_solution.yaml"
                )
                solution_dict = {
                    "start_times": tuple_to_pyyaml_key(
                        partial_sol_best.get_start_time_map()
                    ),
                    "end_times": tuple_to_pyyaml_key(
                        partial_sol_best.get_end_time_map()
                    ),
                }
                object_to_yaml(solution_dict, output_path)
                if last_solution.makespan < partial_sol_best.makespan:
                    output_path = self.get_file_path_for_subroutine(
                        f"_gantt_{job_subset_cnt}_2_partial_CP_solution.yaml"
                    )
                    solution_dict = {
                        "start_times": tuple_to_pyyaml_key(
                            last_solution.get_start_time_map()
                        ),
                        "end_times": tuple_to_pyyaml_key(
                            last_solution.get_end_time_map()
                        ),
                    }
                    object_to_yaml(solution_dict, output_path)
                if remaining_jobs:
                    output_path = self.get_file_path_for_subroutine(
                        f"_gantt_{job_subset_cnt}_3_all_dispatched_solution.yaml"
                    )
                    solution_dict = {
                        "start_times": tuple_to_pyyaml_key(
                            all_dispatched_sol_best.get_start_time_map()
                        ),
                        "end_times": tuple_to_pyyaml_key(
                            all_dispatched_sol_best.get_end_time_map()
                        ),
                    }
                    object_to_yaml(solution_dict, output_path)

                # Store the objective value logs

                # Obj. value of dispatched solution as a value
                sub_obj_store.add_obj_value(
                    last_timestamp, all_dispatched_sol_best.makespan, is_maximize=None
                )

                # Obj. values of Un-dispatched solution as bounds
                undispatched_obj_value_records = sub_cp_mdl.get_obj_value_records()
                for elapsed, value in undispatched_obj_value_records:
                    sub_obj_store.add_obj_bound(elapsed, value, is_maximize=None)
                if (
                    last_timestamp,
                    last_solution.makespan,
                ) not in undispatched_obj_value_records:
                    sub_obj_store.add_obj_bound(
                        last_timestamp, last_solution.makespan, is_maximize=None
                    )
                _last_timestamp_note = f"{job_subset_cnt}/{job_cnt}"
                sub_obj_store.add_last_timestamp_note(
                    _last_timestamp_note,
                    obj_value_is_valid=True,
                    obj_bound_is_valid=True,
                )

        if last_solution is None:
            logging.warning("Incremental CP construction failed to find a solution.")
            report = HfsSubroutineReport(
                elapsed_time=sub_timer.elapsed_sec,
                obj_value=None,
                obj_bound=None,
                is_init=is_init,
            )
            self.solution_manager.register(report, None)
            return

        if error_if_infeasible:
            self.check_feasibility(last_solution.get_start_time_map())

        # Create report for the final solution and register it
        final_report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(last_solution.makespan),
            obj_bound=None,
            is_init=is_init,
        )
        was_updated = self.solution_manager.register(final_report, last_solution)

        if was_updated:
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(
                log_time, float(last_solution.makespan), is_maximize=False
            )
            if draw_gantt:
                self.draw_incumbent_gantt()

        # Write the objective store to a YAML file
        # TODO: suffix from output_metadata
        if sub_obj_store:
            sub_obj_store.save_yaml(self.get_file_path_for_subroutine("_obj_log.yaml"))

    @staticmethod
    def get_johnsons_rule_sequence(
        job_name_2_p1_map: dict[str, int], job_name_2_p2_map: dict[str, int]
    ) -> list[str]:
        """
        Apply Johnson's rule to determine the job sequence.
        This method takes two dictionaries representing aggregated processing times
        for two stages and returns a job sequence based on the Johnson's rule.

        Args:
            job_name_2_p1_map (dict[str, int]): job ID -> processing time for the 1st stage
            job_name_2_p2_map (dict[str, int]): job ID -> processing time for the 2nd stage

        Returns:
            list[str]: A list of job IDs ordered according to Johnson's rule.
        """
        jobs = list(job_name_2_p1_map.keys())

        l1: list[str] = []
        l2: list[str] = []

        for job in jobs:
            if job_name_2_p1_map[job] <= job_name_2_p2_map[job]:
                l1.append(job)
            else:
                l2.append(job)
            # logging.info(
            #     f"Job {job} with p1={job_name_2_p1_map[job]}, p2={job_name_2_p2_map[job]}"
            #     f"; Min={min(job_name_2_p1_map[job], job_name_2_p2_map[job])}"
            #     f"; p1<=p2: {job_name_2_p1_map[job] <= job_name_2_p2_map[job]}"
            # )

        # Sort by increasing order of p_1j, tie-breaking by job-ID (ascending)
        l1.sort(key=lambda j: (job_name_2_p1_map[j], j))

        # Sort by decreasing order of p_2j, tie-breaking by job-ID (descending)
        l2.sort(key=lambda j: (job_name_2_p2_map[j], j), reverse=True)

        # logging.info(f"Job sequence by Johnson's rule: {l1}+{l2}")

        sequence = l1 + l2
        return sequence

    def get_cds_sequence(self, k: int) -> list[str]:
        """
        Get Campbell-Dudek-Smith (CDS) sequence given $k$.

        The sequence is originally for the permultation flow shop problem
        to minimize the makespan.

        When k=1, result is the same as get_sequence1().

        Args:
            k (int): Index between 1 and m-1, where m is the number of stages.

        Returns:
            list[str]: A list of job IDs ordered according to the CDS rule.
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        m = self.instance.stage_count
        p1_stages = stages[:k]  # Sum over stages 0 ~ k-1 (1~k on paper)
        p2_stages = stages[m - k :]  # Sum over stages m-k ~ m-1 (m-k+1 ~ m on paper)
        p1 = {j: sum(p_dict[j, i] for i in p1_stages) for j in jobs}
        p2 = {j: sum(p_dict[j, i] for i in p2_stages) for j in jobs}

        return self.get_johnsons_rule_sequence(p1, p2)

    def get_tp_sequence(self, k: int) -> list[str]:
        """
        Get two-partition sequence given $k$.

        - For each job, consider the sum of processing times of stages 0 to k-1
          as the first part (p1),
        - and the sum of processing times of stages k to m-1 as the second part
          (p2).
        - Create a job sequence by applying Johnson's rule for F2||C_max.

        When k = num_stages // 2, result is the same as get_sequence2().

        Args:
            k (int): Index between 1 and m-1, where m is the number of stages.

        Returns:
            list[str]: A list of job IDs ordered according to the TP rule.
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        p1_stages = stages[:k]  # Sum over stages 0 ~ k-1
        p2_stages = stages[k:]  # Sum over stages k ~ m-1
        p1 = {j: sum(p_dict[j, i] for i in p1_stages) for j in jobs}
        p2 = {j: sum(p_dict[j, i] for i in p2_stages) for j in jobs}

        return self.get_johnsons_rule_sequence(p1, p2)

    def get_gupta_sequence(self) -> list[str]:
        """
        Return the job sequence according to Gupta's functional heuristic algorithm.

        - For each job, calculate:
            f(j) = A / min_{1 <= m <= M-1} (p_{j,m} + p_{j,m+1})
            where
                A = 1 if p_{j,M} <= p_{j,1}
                A = -1 otherwise
        - Sort jobs in ascending order of f(j).
        (Break ties by total processing time, then by job ID)

        Returns:
            list[str]: A list of job IDs in Gupta heuristic order.
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        m = len(stages)
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        gupta_score = {}
        total_p = {}

        for j in jobs:
            # min over all adjacent pairs (m=0 to m-2)
            min_sum = min(
                p_dict[j, stages[m1]] + p_dict[j, stages[m1 + 1]] for m1 in range(m - 1)
            )
            A = 1 if p_dict[j, stages[-1]] <= p_dict[j, stages[0]] else -1
            f_j = A / min_sum if min_sum != 0 else float("inf")  # avoid div by zero
            gupta_score[j] = f_j
            total_p[j] = sum(p_dict[j, s] for s in stages)

        # Sort by ascending order: (f_j, total processing time, job id)
        sorted_jobs = sorted(jobs, key=lambda j: (gupta_score[j], total_p[j], j))
        return sorted_jobs

    def get_palmer_sequence(self) -> list[str]:
        """
        Return the job sequence according to Palmer's slope index heuristic.

        Returns:
            list[str]: Job ID list in Palmer slope order (descending s_i).
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        m = len(stages)
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        palmer_score = {}
        for j in jobs:
            s = sum(
                (m - 2 * (stage_idx + 1) + 1) * p_dict[j, stages[stage_idx]]
                for stage_idx in range(m)
            )
            palmer_score[j] = s

        # Sort by ascending order: (Palmer score, job id)
        sorted_jobs = sorted(jobs, key=lambda j: (palmer_score[j], j))
        return sorted_jobs

    def initialize_by_cjqp(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        max_time_per_add: float | None = None,
        no_improvement_timelimit: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Build a CP-guided solution using Palmer sequence.

        This method incrementally constructs a feasible schedule by solving sub-CP models
        for each job prefix in the Palmer sequence.

        Args:
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            added_batch_size (int, optional): The number of jobs to add in each iteration.
                Defaults to 1.
            max_time_per_add (float | None, optional): Time limit (in seconds) for solving each incremental subproblem.
                If None, uses the remaining time limit. Defaults to None.
            no_improvement_timelimit (float | None, optional): If there is no improvement for this
                amount of time, the search will be stopped. If None, no timeout is set.
                Defaults to None.
            error_if_infeasible (bool, optional): If True, raises an error if the solution is infeasible.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws a Gantt chart of the solution.
                Defaults to False.
        """

        self.construct_solution_by_incremental_cp(
            self.get_palmer_sequence(),
            solver_thread_cnt,
            added_batch_size=added_batch_size,
            max_time_per_add=max_time_per_add,
            no_improvement_timelimit=no_improvement_timelimit,
            is_init=True,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def initialize_by_cjqg(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        max_time_per_add: float | None = None,
        no_improvement_timelimit: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Build a CP-guided solution using Gupta sequence.

        This method incrementally constructs a feasible schedule by solving sub-CP models
        for each job prefix in the Gupta sequence.

        Args:
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            added_batch_size (int, optional): The number of jobs to add in each iteration.
                Defaults to 1.
            max_time_per_add (float | None, optional): Time limit (in seconds) for solving each incremental subproblem.
                If None, uses the remaining time limit. Defaults to None.
            no_improvement_timelimit (float | None, optional): If there is no improvement for this
                amount of time, the search will be stopped. If None, no timeout is set.
                Defaults to None.
            error_if_infeasible (bool, optional): If True, raises an error if the solution is infeasible.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws a Gantt chart of the solution.
                Defaults to False.
        """

        self.construct_solution_by_incremental_cp(
            self.get_gupta_sequence(),
            solver_thread_cnt,
            added_batch_size=added_batch_size,
            max_time_per_add=max_time_per_add,
            no_improvement_timelimit=no_improvement_timelimit,
            is_init=True,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_shdlb(self) -> None:
        """
        Compute the global lower bound for the Hybrid Flow Shop instance using the method
        described by Santos et al. (1995) and update the global lower bound.
        """
        sub_timer = ElapsedTimer()
        instance = self.instance
        jobs: list[str] = instance.job_id_list
        stages: list[str] = instance.stage_id_list
        stage_2_mc_count_map: dict[str, int] = {
            i: len(instance.stage_2_machines_map[i]) for i in stages
        }
        p: dict[tuple[str, str], int] = instance.p_manager.job_stage_2_value_map(
            jobs, stages
        )
        m = len(stages)

        def LB0() -> int:
            return max(sum(p[j, i] for i in stages) for j in jobs)

        def RS(j: str, stage_idx: int) -> int:
            return sum(p[j, stages[s]] for s in range(stage_idx + 1, m))

        def LS(j: str, stage_idx: int) -> int:
            return sum(p[j, stages[s]] for s in range(0, stage_idx))

        def LBj(stage_idx: int) -> float:
            i = stages[stage_idx]
            M_i = stage_2_mc_count_map[i]

            # LSA and RSA: ascending sorted LS and RS values
            LSA = sorted([LS(j, stage_idx) for j in jobs])
            RSA = sorted([RS(j, stage_idx) for j in jobs])
            total_processing = sum(p[j, i] for j in jobs)
            lhs_sum = sum(LSA[y] for y in range(min(M_i, len(LSA))))
            rhs_sum = sum(RSA[y] for y in range(min(M_i, len(RSA))))
            return math.ceil((lhs_sum + total_processing + rhs_sum) / M_i)

        lb0 = LB0()
        stage_bounds = [LBj(stage_idx) for stage_idx in range(m)]
        obj_bound = max([lb0] + stage_bounds)

        logging.info(
            f"[SHD LB] LB(0) = {lb0}, LB(j) = {stage_bounds}, LB_MAX = {obj_bound}"
        )

        # Log
        if self.solution_manager.current_obj_bound_is_worse_than(obj_bound):
            log_time = self.timer.elapsed_sec
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=False)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_bound_is_valid=True
            )

        # Create report and register
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=None,
            obj_bound=obj_bound,
            is_init=False,
        )
        self.solution_manager.register(report, None)

    def initialize_by_dj_cds(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the Campbell-Dudek-Smith (CDS) sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        # Subroutine states
        best_makespan = float("inf")
        best_schedule: HybridFlowshopSchedule | None = None
        best_k = -1

        for k in range(1, self.instance.stage_count):
            # Create an empty schedule
            schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
                self.instance.stage_2_machines_map
            )
            # Dispatch
            job_sequence = self.get_cds_sequence(k)
            for j in job_sequence:
                schedule.dispatch_job_by_stages(
                    j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
                )
            # Update subroutine states
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        if error_if_infeasible:
            self.check_feasibility(best_schedule.get_start_time_map())
        logging.info(f"Best schedule found with k={best_k}, makespan={best_makespan}")

        # Create report and register the new solution
        obj_value = float(best_schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, best_schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_dj_tp(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the two-partition sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        # Subroutine states
        best_makespan = float("inf")
        best_schedule: HybridFlowshopSchedule | None = None
        best_k = -1

        for k in range(0, self.instance.stage_count):
            # Create an empty schedule
            schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
                self.instance.stage_2_machines_map
            )
            # Dispatch
            job_sequence = self.get_tp_sequence(k)
            for j in job_sequence:
                schedule.dispatch_job_by_stages(
                    j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
                )
            # Update subroutine states
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        if error_if_infeasible:
            self.check_feasibility(best_schedule.get_start_time_map())
        logging.info(f"Best schedule found with k={best_k}, makespan={best_makespan}")

        # Create report and register the new solution
        obj_value = float(best_schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, best_schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_dj_gupta(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the gupta sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        # Create an empty schedule
        schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
            self.instance.stage_2_machines_map
        )
        # Dispatch
        job_sequence = self.get_gupta_sequence()
        for j in job_sequence:
            schedule.dispatch_job_by_stages(
                j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
            )

        if error_if_infeasible:
            self.check_feasibility(schedule.get_start_time_map())
        logging.info(f"Schedule found with makespan={schedule.makespan}")

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_dj_palmer(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the gupta sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        # Create an empty schedule
        schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
            self.instance.stage_2_machines_map
        )
        # Dispatch
        job_sequence = self.get_palmer_sequence()
        for j in job_sequence:
            schedule.dispatch_job_by_stages(
                j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
            )

        if error_if_infeasible:
            self.check_feasibility(schedule.get_start_time_map())
        logging.info(f"Schedule found with makespan={schedule.makespan}")

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_ds_cds(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the Campbell-Dudek-Smith (CDS) sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        best_makespan = float("inf")
        best_schedule: HybridFlowshopSchedule | None = None
        best_k = -1
        for k in range(1, self.instance.stage_count):
            # Create an empty schedule
            schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
                self.instance.stage_2_machines_map
            )
            job_sequence = self.get_cds_sequence(k)
            for i in self.instance.stage_id_list:
                schedule.dispatch_stage_by_jobs(
                    i, job_sequence, self.stage_2_job_2_p_dict[i]
                )
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        if error_if_infeasible:
            self.check_feasibility(best_schedule.get_start_time_map())
        logging.info(f"Best schedule found with k={best_k}, makespan={best_makespan}")

        # Create report and register the new solution
        obj_value = float(best_schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, best_schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_ds_tp(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the two-partition sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        best_makespan = float("inf")
        best_schedule: HybridFlowshopSchedule | None = None
        best_k = -1
        for k in range(0, self.instance.stage_count):
            # Create an empty schedule
            schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
                self.instance.stage_2_machines_map
            )
            job_sequence = self.get_tp_sequence(k)
            for i in self.instance.stage_id_list:
                schedule.dispatch_stage_by_jobs(
                    i, job_sequence, self.stage_2_job_2_p_dict[i]
                )
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        if error_if_infeasible:
            self.check_feasibility(best_schedule.get_start_time_map())
        logging.info(f"Best schedule found with k={best_k}, makespan={best_makespan}")

        # Create report and register the new solution
        obj_value = float(best_schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, best_schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_ds_gupta(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the two-partition sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        # Create an empty schedule
        schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
            self.instance.stage_2_machines_map
        )
        # Dispatch
        job_sequence = self.get_gupta_sequence()
        for i in self.instance.stage_id_list:
            schedule.dispatch_stage_by_jobs(
                i, job_sequence, self.stage_2_job_2_p_dict[i]
            )

        if error_if_infeasible:
            self.check_feasibility(schedule.get_start_time_map())
        logging.info(f"Schedule found with makespan={schedule.makespan}")

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_ds_palmer(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the two-partition sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        # Create an empty schedule
        schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
            self.instance.stage_2_machines_map
        )
        # Dispatch
        job_sequence = self.get_palmer_sequence()
        for i in self.instance.stage_id_list:
            schedule.dispatch_stage_by_jobs(
                i, job_sequence, self.stage_2_job_2_p_dict[i]
            )

        if error_if_infeasible:
            self.check_feasibility(schedule.get_start_time_map())
        logging.info(f"Schedule found with makespan={schedule.makespan}")

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        total_elapsed_time = self.timer.elapsed_sec
        self.add_obj_value_log(total_elapsed_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def get_incumbent_midpoint_sequence(self) -> list[str]:
        """
        Returns a job sequence based on the incumbent solution, sorted in ascending order by:

        1. midpoint = (start_time at first stage + start_time at last stage) / 2
        2. tie-break by first stage start_time
        3. tie-break by original job index

        Raises:
            ValueError: if no incumbent solution is available.

        Returns:
            list[str]: A list of job IDs representing the midpoint sequence.
        """
        incumbent = self.solution_manager.get_incumbent()
        if incumbent is None:
            raise ValueError(
                "No incumbent solution available to build midpoint sequence."
            )

        start_map = incumbent.get_start_time_map()
        jobs = self.instance.job_id_list
        idx_map = {j: idx for idx, j in enumerate(jobs)}
        first_stage = self.instance.stage_id_list[0]
        last_stage = self.instance.stage_id_list[-1]

        seq_info: list[tuple[float, int, int, str]] = []
        for j in jobs:
            # find any machine k for first and last stage
            s_first = next(
                t
                for (job, stage, _), t in start_map.items()
                if job == j and stage == first_stage
            )
            s_last = next(
                t
                for (job, stage, _), t in start_map.items()
                if job == j and stage == last_stage
            )
            midpoint = (s_first + s_last) / 2
            seq_info.append((midpoint, s_first, idx_map[j], j))

        seq_info.sort(key=lambda x: (x[0], x[1], x[2]))
        return [info[3] for info in seq_info]

    def initialize_by_cjims(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        max_time_per_add: float | None = None,
        no_improvement_timelimit: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Builds a CP-guided solution using a midpoint sequence from the incumbent solution.

        Args:
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            added_batch_size (int, optional): The number of jobs to add in each iteration.
                Defaults to 1.
            max_time_per_add (float | None, optional): Time limit (in seconds) for solving each incremental subproblem.
                If None, uses the remaining time limit. Defaults to None.
            no_improvement_timelimit (float | None, optional): If there is no improvement for this
                amount of time, the search will be stopped. If None, no timeout is set.
                Defaults to None.
            error_if_infeasible (bool, optional): If True, raises an error if the solution is infeasible.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws a Gantt chart of the solution.
                Defaults to False.
        """
        self.construct_solution_by_incremental_cp(
            self.get_incumbent_midpoint_sequence(),
            solver_thread_cnt,
            added_batch_size=added_batch_size,
            max_time_per_add=max_time_per_add,
            no_improvement_timelimit=no_improvement_timelimit,
            is_init=True,  # TODO: remove this line
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    # End subroutine definition

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
    ) -> None:
        """Check the feasibility of the given start times.

        Args:
            start_time_map (dict[tuple[str, str, str], int]): _description_

        Raises:
            ValueError: If any start time is negative or invalid.
            RuntimeError: If the feasibility check fails while solving the model.
            ValueError: If the feasibility check fails with an unexpected status.
        """
        logging.info("Feasibility check starts")
        for (j, i, k), start_time in start_time_map.items():
            if start_time < 0:
                raise ValueError(
                    f"Invalid start time for job {j}, stage {i}, machine {k}: {start_time}"
                )
        base_cp = self.create_base_cp_model()

        # Freeze operation start times and machine assignments
        for (j, i, k), start_time in start_time_map.items():
            if isinstance(self.cp_model, CP2023NaderiCumulative):
                base_cp.add(self.cp_model.var_op_start[j, i] == start_time)
            elif isinstance(
                self.cp_model,
                (
                    CP2023NaderiOptionalInterval,
                    CPOptionalIntervalMasterTimevar,
                    CPCumulativeOptionalHybrid,
                ),
            ):
                base_cp.add(self.cp_model.var_op_is_present[j, i, k] == 1)
                base_cp.add(self.cp_model.var_op_start[j, i, k] == start_time)
            else:
                raise TypeError(f"Unsupported CP model type: {type(self.cp_model)}")

        # Solve with tight time limit
        try:
            summary = self.solve_cp_model(base_cp, 1.0, 1)
        except Exception as e:
            raise RuntimeError(f"Feasibility check FAILED: {e}") from e

        if summary.status != CpsatStatus.OPTIMAL:
            raise ValueError(f"Feasibility check FAILED with status: {summary.status}")

        logging.info("Feasibility check passed")
