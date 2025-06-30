import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from mbls import DynamicDataObject, ElapsedTimer, SolverOutputSummary, SolverStatus
from mbls.cpsat.cp_subroutine_controller import CpSubroutineController
from schore.hybridflowshop import HybridFlowShopProblem

from .hfs_experiment_summary import HfsExperimentSummary
from .hfs_solver_output_summary import HfsSolverOutputSummary
from .pure_cp_2023_naderi import PureCP2023Naderi
from .scheduling.hybrid_flowshop_schedule import HybridFlowshopSchedule
from .solution_manager import SolutionManager
from .stopping_criteria import StoppingCriteria


class HybridFlowShopCpLnsController(
    CpSubroutineController[HybridFlowShopProblem, PureCP2023Naderi, StoppingCriteria]
):
    """
    Controller for solving Hybrid Flow Shop problems using CP-based LNS.

    This controller manages the interaction between the problem, the CP solver,
    and the solution manager, allowing for efficient search and solution management.
    """

    stopping_criteria: StoppingCriteria

    last_solution_manager: SolutionManager
    """Manages the last(most recent) solution."""
    incumbent_solution_manager: SolutionManager
    """Manages the incumbent solution."""

    def __init__(
        self,
        instance: HybridFlowShopProblem,
        shared_param_dict: dict,
        subroutine_flow: DynamicDataObject,
        stopping_criteria: StoppingCriteria,
    ):
        super().__init__(
            instance,
            shared_param_dict,
            PureCP2023Naderi,
            subroutine_flow,
            stopping_criteria,
        )

        self.experiment_summary = HfsExperimentSummary(instance.name)

        self.obj_lower_bound: float | None = None
        """Best objective bound found so far."""

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
        if self.timer.get_elapsed_sec() >= self.stopping_criteria.timelimit:
            logging.info("Stop by timelimit")
            return True
        return False

    def get_remaining_sec(self) -> float:
        return self.timer.get_remaining_sec(self.stopping_criteria.timelimit)

    def get_remaining_time_limit(self, subroutine_time_limit: float) -> float:
        """Get the remaining time limit for the subroutine.

        Args:
            subroutine_time_limit (float): The time limit for the subroutine in seconds.

        Returns:
            float: The minimum of the subroutine time limit and the remaining time limit.
        """
        return min(subroutine_time_limit, self.get_remaining_sec())

    # End stopping condition

    def create_base_cp_model(self) -> PureCP2023Naderi:
        if "horizon" not in self.shared_param_dict:
            raise ValueError("Horizon not found in shared parameters.")
        horizon = self.shared_param_dict["horizon"]
        return self.cp_model_class.from_instance(self.instance, horizon)

    # Start solution management

    def set_last_solution_as_incumbent(self, draw_gantt: bool = False) -> None:
        """Set the incumbent solution.

        Args:
            draw_gantt (bool, optional): If True, draws the Gantt chart of the incumbent solution.
                Defaults to False.
        """
        self.incumbent_solution_manager = self.last_solution_manager
        if draw_gantt:
            self.draw_incumbent_gantt()

    def draw_incumbent_gantt(self, output_path: Optional[Path] = None) -> None:
        if output_path is None:
            output_path = self.get_file_path_for_subroutine("_gantt.png")
        self.incumbent_solution_manager.save_gantt_as_png(output_path)

    def update_incumbent_solution(self, draw_gantt: bool = False) -> None:
        """
        Update the incumbent solution if the last solution is better.
        - This method checks if the last solution is better than the incumbent solution.
        - If it is, it sets the last solution as the incumbent solution and optionally draws the Gantt chart.

        Args:
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution if it is better.
                Defaults to False.
        """
        if self.last_solution_is_better_than_incumbent():
            self.set_last_solution_as_incumbent(draw_gantt=draw_gantt)

    def last_solution_is_better_than_incumbent(self) -> bool:
        """
        Check if the last solution is better than the incumbent solution.
        - If no incumbent solution exists, the last solution is considered better.
        """
        if not hasattr(self, "last_solution_manager"):
            raise ValueError("No last solution available to compare.")
        # If no incumbent solution exists, the last solution is considered better
        if not hasattr(self, "incumbent_solution_manager"):
            return True
        return (
            self.last_solution_manager.summary.objective_value is not None
            and self.incumbent_solution_manager.summary.objective_value is not None
            and self.last_solution_manager.summary.objective_value
            < self.incumbent_solution_manager.summary.objective_value
        )

    def get_incumbent_solution_dict(self, for_pyyaml: bool = False) -> dict[str, Any]:
        """
        Get the incumbent solution as a dictionary.

        Args:
            for_pyyaml (bool, optional): If true, create start time and end time dictionary for PyYAML.
                Defaults to False.
        """
        if not hasattr(self, "incumbent_solution_manager"):
            raise ValueError("No incumbent solution available.")
        return self.incumbent_solution_manager.get_solution_dict(for_pyyaml=for_pyyaml)

    # End solution management

    # Start subroutine definition

    def solve_current_cp_remaining_time_limit(
        self,
        computational_time: float,
        num_workers: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        is_initial_solution: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Solve the current CP model with the remaining time limit.
        - Updates the experiment summary with the run summary of the current solution.
        - Updates the last solution manager with the start and end times extracted from the CP model.
        - Checks the feasibility of the solution if required.
        - If the objective value is valid, it updates the incumbent solution.

        Args:
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
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

        # If LB for the objective bound is valid for the model, set it in the CP model
        if obj_value_is_valid:
            if self.obj_lower_bound is not None:
                self.cp_model.set_obj_lower_bound(self.obj_lower_bound)

        summary = self.solve_current_cp_model(
            _timelimit,
            num_workers,
            random_seed=self.random_seed,
            timer=self.timer,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

        _summary = HfsSolverOutputSummary.from_other(
            summary, is_init=is_initial_solution
        )

        start_times, end_times = self.cp_model.extract_start_end_times()

        if error_if_infeasible:
            self.check_feasibility(start_times)
        self.experiment_summary.add_run_summary(_summary)
        self.last_solution_manager = SolutionManager(
            start_times=start_times,
            end_times=end_times,
            summary=summary,
        )

        if obj_value_is_valid:
            self.update_incumbent_solution(draw_gantt=draw_gantt)
        if obj_bound_is_valid:
            if summary.best_objective_bound is not None:
                if self.obj_lower_bound is None:
                    self.obj_lower_bound = summary.best_objective_bound
                else:
                    self.obj_lower_bound = max(
                        self.obj_lower_bound, summary.best_objective_bound
                    )

    def feasible_incumbent_solution_exists(self) -> bool:
        """Check if the incumbent solution is feasible."""
        return (
            hasattr(self, "incumbent_solution_manager")
            and self.incumbent_solution_manager.is_feasible
        )

    def solve_with_initial_solution(
        self,
        computational_time: float,
        num_workers: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Solve the current CP model with the incumbent solution as the initial solution.

        - If a feasible incumbent solution exists, it clears the hints in the current CP model
          and applies the incumbent solution as a hint.
        - Then, it solves the CP model with the given computational time and number of workers.

        Args:
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            obj_value_is_valid (bool, optional): If True, adds the objective value log.
                Defaults to False.
            obj_bound_is_valid (bool, optional): If True, adds the objective bound log.
                Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        cannot_apply_hint = not self.feasible_incumbent_solution_exists()
        if not cannot_apply_hint:
            self.cp_model.clear_hints()
            self.incumbent_solution_manager.apply_start_and_present_hints(self.cp_model)
        self.solve_current_cp_remaining_time_limit(
            computational_time,
            num_workers,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            is_initial_solution=cannot_apply_hint,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    # Subroutine: solve base CP model

    def solve_base_cp_model(
        self,
        computational_time: float,
        num_workers: int,
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
            num_workers (int): The number of parallel workers (threads) to use during search.
            is_initial_solution (bool, optional): If True, marks this run as producing the initial solution (affects summary/logging). Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution after solving. Defaults to False.
        """
        self.cp_model.delete_added_constraints()
        if is_initial_solution:
            self.solve_current_cp_remaining_time_limit(
                computational_time,
                num_workers,
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
                num_workers,
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
        num_workers: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Apply the freeze method, solve, and reset the model.

        Args:
            freeze_method (Callable): A callable that applies the freeze method to the CP model.
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
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
            num_workers,
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
        num_workers: int,
        error_if_infeasible=False,
        draw_gantt: bool = False,
    ):
        """Time window search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        self.freeze_solve_reset(
            lambda: self.apply_time_window_operator(rho),
            computational_time,
            num_workers,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_time_window_operator(self, rho: float):
        """
        Apply the Time Window Operator to the current CP model.

        Args:
            current_start_times (dict[tuple[str, str, str], int]): (job_name, stage_name, mc_name) -> current start_time
            current_end_times (dict[tuple[str, str, str], int]): (job_name, stage_name, mc_name) -> current end_time
            rho (float, optional): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
        """
        logging.info(f"Applying time window operator with rho={rho}")

        start_times = self.incumbent_solution_manager.start_times
        end_times = self.incumbent_solution_manager.end_times

        # 1. Calculate makespan (C_max)
        all_end_times = list(end_times.values())
        if not all_end_times:
            raise ValueError("No end times available for Time Window Operator.")
        C_max = max(all_end_times)

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

        for key in start_times:
            s_time = start_times[key]
            e_time = end_times[key]
            if not self.is_within_window(
                s_time, window_start, window_end
            ) and not self.is_within_window(e_time, window_start, window_end):
                out_of_window_ops.add(key)

        # 4. Fix machine assignment and precedence for out-of-window operations
        stage_mc_to_jobs: dict[tuple[str, str], list[str]] = defaultdict(list)

        for j, i, k in out_of_window_ops:
            self.cp_model.add_fixed_machine_assignment_constraint(j, i, k)
            stage_mc_to_jobs[(i, k)].append(j)

        for (i, k), jobs in stage_mc_to_jobs.items():
            # Start time 기준 정렬
            jobs_sorted = sorted(jobs, key=lambda j: start_times[(j, i, k)])
            for j1, j2 in zip(jobs_sorted[:-1], jobs_sorted[1:]):
                self.cp_model.add_fixed_operation_precedence_constraint(j1, j2, i, k)

    @staticmethod
    def is_within_window(time: int, window_start: int, window_end: int) -> bool:
        """Check if a given time is within the specified window."""
        return window_start <= time <= window_end

    # Subroutine: Block operator

    def block_search(
        self,
        rho: float,
        computational_time: float,
        num_workers: int,
        error_if_infeasible=False,
        draw_gantt: bool = False,
    ):
        """Block search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """

        self.freeze_solve_reset(
            lambda: self.apply_block_operator(rho),
            computational_time,
            num_workers,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_block_operator(self, rho: float):
        """
        Apply the Block Operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
        """
        logging.info(f"Applying block operator with rho={rho}")

        start_times = self.incumbent_solution_manager.start_times
        end_times = self.incumbent_solution_manager.end_times

        if not start_times or not end_times:
            raise ValueError("No solution available for block operator.")

        all_ops = list(start_times.keys())  # TODO: 순서 유지되는지 확인
        total_ops = len(all_ops)
        num_to_select = max(1, int(rho * total_ops))

        # Step 1: Start from a random operation
        seed_op = random.choice(all_ops)
        selected_ops = set([seed_op])
        queue = [seed_op]

        # Step 2: Expand to overlapping operations
        while queue and len(selected_ops) < num_to_select:
            current_op = queue.pop(0)
            cs, ce = start_times[current_op], end_times[current_op]
            for op in all_ops:
                if op in selected_ops:
                    continue
                os, oe = start_times[op], end_times[op]
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
            self.cp_model.add_fixed_machine_assignment_constraint(j, i, k)
            stage_mc_to_jobs[(i, k)].append(j)

        for (i, k), jobs in stage_mc_to_jobs.items():
            jobs_sorted = sorted(jobs, key=lambda j: start_times[(j, i, k)])
            for j1, j2 in zip(jobs_sorted[:-1], jobs_sorted[1:]):
                self.cp_model.add_fixed_operation_precedence_constraint(j1, j2, i, k)

    @staticmethod
    def is_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
        """Check if two time intervals overlap."""
        return not (e1 <= s2 or e2 <= s1)

    # Subroutine: Johnson-based Heuristic for initialization

    def dispatch_sequentially(self, job_sequence: list[str], draw_gantt: bool = False):
        e_timer = ElapsedTimer()

        # Create an empty schedule
        schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
            self.instance.stage_2_machines_map
        )
        # Job name -> stage name -> processing time map
        job_2_stage_2_p_dict = self.instance.p_manager.job_2_stage_2_value_map(
            self.instance.job_id_list, self.instance.stage_id_list
        )

        for idx, j in enumerate(job_sequence):
            schedule.dispatch_job_earliest(
                j, self.instance.stage_id_list, job_2_stage_2_p_dict[j]
            )
            # TODO: uncomment only for debug purpose
            # start_times = schedule.get_start_time_map()
            # end_times = schedule.get_end_time_map()
            # obj_value = float(schedule.makespan)
            # sol_mgr = SolutionManager(
            #     start_times,
            #     end_times,
            #     SolverOutputSummary(
            #         SolverStatus.FEASIBLE,
            #         0.0,
            #         objective_value=obj_value,
            #         best_objective_bound=None,
            #         progress_log=None,
            #     ),
            # )
            # output_path = self.get_file_path_for_subroutine(f"_gantt_{idx}_{j}.png")
            # sol_mgr.save_gantt_as_png(output_path)

        log_time = self.timer.get_elapsed_sec()
        obj_value = float(schedule.makespan)
        progress_log = [(log_time, obj_value, 0.0)]
        self.append_obj_log(
            progress_log,
            is_maximize=False,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
        )

        elapsed_time = e_timer.get_elapsed_sec()
        summary_for_sol_mgr = SolverOutputSummary(
            SolverStatus.FEASIBLE,
            elapsed_time,
            objective_value=obj_value,
            best_objective_bound=None,
            progress_log=progress_log,
        )
        summary_for_run = HfsSolverOutputSummary.from_other(
            summary_for_sol_mgr, is_init=True
        )
        self.experiment_summary.add_run_summary(summary_for_run)

        start_times = schedule.get_start_time_map()
        end_times = schedule.get_end_time_map()
        self.last_solution_manager = SolutionManager(
            start_times, end_times, summary_for_sol_mgr
        )

        self.update_incumbent_solution(draw_gantt=draw_gantt)

    def construct_solution_by_incremental_cp(
        self,
        job_sequence: list[str],
        max_time_per_add: float,
        num_workers: int,
        added_batch_size: int = 1,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Construct a complete solution by incrementally solving CP submodels.

        This method builds a feasible schedule by adding jobs one by one
        according to the provided job sequence. At each step, a sub-CP model
        is constructed for the current subset of jobs and solved with the
        given time limit. Previously scheduled jobs are frozen to guide the solver.

        Args:
            job_sequence (list[str]): The sequence of job IDs to be added.
            max_time_per_add (float): The time limit (in seconds) for solving each incremental subproblem.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            added_batch_size (int, optional): The number of jobs to add in each iteration.
                Defaults to 1.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """

        e_timer = ElapsedTimer()

        _last_summary: SolverOutputSummary | None = None
        _last_sol_manager: SolutionManager | None = None

        sequence_of_job_sublist = [
            job_sequence[i : i + added_batch_size]
            for i in range(0, len(job_sequence), added_batch_size)
        ]

        job_subset: set[str] = set()
        for job_sublist in sequence_of_job_sublist:
            logging.info(f"Add jobs {job_sublist} into {len(job_subset)}-job subset")
            job_subset.update(job_sublist)  # Add new jobs to the subset

            # Create CP model with the job subset
            sub_cp_mdl = self.cp_model.create_problem_of_job_subset(job_subset)
            # If this is not the first iteration, freeze jobs in the previous model
            if _last_sol_manager is not None:
                _last_sol_manager.apply_fixed_machine_and_ops_precedence_constraints(
                    sub_cp_mdl
                )

            _timelimit = self.get_remaining_time_limit(max_time_per_add)
            _last_summary = self.solve_cp_model(
                sub_cp_mdl,
                _timelimit,
                num_workers,
                random_seed=self.random_seed,
                timer=self.timer,
            )

            start_times, end_times = sub_cp_mdl.extract_start_end_times()
            _last_sol_manager = SolutionManager(
                start_times=start_times,
                end_times=end_times,
                summary=_last_summary,
            )

        assert _last_summary is not None, "No summary available after solving CP model."
        assert _last_summary.objective_value is not None, (
            "No objective value available after solving."
        )
        assert _last_sol_manager is not None, "No solution available after solving."

        if error_if_infeasible:
            self.check_feasibility(_last_sol_manager.start_times)

        log_time = self.timer.get_elapsed_sec()
        obj_value = _last_summary.objective_value
        progress_log = [(log_time, obj_value, 0.0)]
        self.append_obj_log(
            progress_log,
            is_maximize=False,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
        )
        elapsed_time = e_timer.get_elapsed_sec()
        subroutine_summary = HfsSolverOutputSummary(
            _last_summary.status,
            elapsed_time,
            objective_value=obj_value,
            best_objective_bound=None,
            progress_log=progress_log,
            is_init=True,
        )
        self.experiment_summary.add_run_summary(subroutine_summary)
        self.last_solution_manager = _last_sol_manager

        self.update_incumbent_solution(draw_gantt=draw_gantt)

    @staticmethod
    def johnson_rule_permutation(
        aggregated_p1: dict[str, int], aggregated_p2: dict[str, int]
    ) -> list[str]:
        """
        Apply Johnson's rule to determine the job sequence.
        This method takes two dictionaries representing aggregated processing times
        for two stages and returns a job sequence based on the Johnson's rule.

        Args:
            aggregated_p1 (dict[str, int]): job ID -> aggregated processing time for the 1st stage
            aggregated_p2 (dict[str, int]): job ID -> aggregated processing time for the 2nd stage

        Returns:
            list[str]: A list of job IDs ordered according to Johnson's rule.
        """
        jobs = list(aggregated_p1.keys())
        # n = len(jobs)
        # sequence: list[str] = [""] * n  # Initialize sequence with empty strings
        # left = 0
        # right = n - 1

        # # Pre-sort jobs based on aggregated processing times
        # jobs.sort(
        #     key=lambda j: (min(aggregated_p1[j], aggregated_p2[j]), j)
        # )  # Sort by min processing time first, then by job ID

        # while jobs:
        #     job = jobs.pop(0)  # Remove the first job from the sorted list
        #     if aggregated_p1[job] <= aggregated_p2[job]:
        #         sequence[left] = job  # Last of the front
        #         left += 1
        #     else:
        #         sequence[right] = job  # First of the back
        #         right -= 1
        l1: list[str] = []
        l2: list[str] = []

        for job in jobs:
            if aggregated_p1[job] <= aggregated_p2[job]:
                l1.append(job)
            else:
                l2.append(job)
            # logging.info(
            #     f"Job {job} with p1={aggregated_p1[job]}, p2={aggregated_p2[job]}"
            #     f"; Min={min(aggregated_p1[job], aggregated_p2[job])}"
            #     f"; p1<=p2: {aggregated_p1[job] <= aggregated_p2[job]}"
            # )

        # Sort by increasing order of p_1j, tie-breaking by job-ID (ascending)
        l1.sort(key=lambda j: (aggregated_p1[j], j))

        # Sort by decreasing order of p_2j, tie-breaking by job-ID (descending)
        l2.sort(key=lambda j: (aggregated_p2[j], j), reverse=True)

        logging.info(f"Job sequence by Johnson's rule: {l1}+{l2}")

        sequence = l1 + l2
        return sequence

    def get_jh1_sequence(self) -> list[str]:
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )
        p1 = {j: p_dict[j, stages[0]] for j in jobs}  # First stage processing times
        p2 = {j: p_dict[j, stages[-1]] for j in jobs}  # Last stage processing times
        return self.johnson_rule_permutation(p1, p2)

    def get_jh2_sequence(self) -> list[str]:
        jobs = self.instance.job_id_list
        num_stages = self.instance.stage_count
        stages = self.instance.stage_id_list
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )
        mid = num_stages // 2
        # First half stage list
        first_half_stages = stages[:mid]
        # Second half stage list
        second_half_stages = stages[mid:]
        p1 = {
            j: sum(p_dict[j, s] for s in first_half_stages) for j in jobs
        }  # Aggregated processing times for first half stages
        p2 = {
            j: sum(p_dict[j, s] for s in second_half_stages) for j in jobs
        }  # Aggregated processing times for second half stages
        return self.johnson_rule_permutation(p1, p2)

    def build_jh1_solution(
        self,
        max_time_per_add: float,
        num_workers: int,
        added_batch_size: int = 1,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Build a CP-guided solution using the Johnson-based Heuristic 1 (jh1) sequence.

        This method computes a job sequence by aggregating processing times from
        the first and last stages (jh1 rule), then incrementally constructs a feasible
        schedule by solving sub-CP models for each job prefix in the sequence.

        Args:
            max_time_per_add (float): The maximum computational time per addition in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """

        self.construct_solution_by_incremental_cp(
            self.get_jh1_sequence(),
            max_time_per_add,
            num_workers,
            added_batch_size=added_batch_size,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def build_jh2_solution(
        self,
        max_time_per_add: float,
        num_workers: int,
        added_batch_size: int = 1,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Build a CP-guided solution using the Johnson-based Heuristic 2 (jh2) sequence.

        This method computes a job sequence by aggregating processing times from
        the first half and second half stages (jh2 rule), then incrementally constructs a feasible
        schedule by solving sub-CP models for each job prefix in the sequence.

        Args:
            max_time_per_add (float): The maximum computational time per addition in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """

        self.construct_solution_by_incremental_cp(
            self.get_jh2_sequence(),
            max_time_per_add,
            num_workers,
            added_batch_size=added_batch_size,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def init_shdlb(self) -> None:
        """
        Compute the global lower bound for the Hybrid Flow Shop instance using the method
        described by Santos et al. (1995) and assign it to `self.obj_bound`.
        """
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

        self.obj_lower_bound = max([lb0] + stage_bounds)
        logging.info(
            f"[Lower Bound] LB(0) = {lb0}, LB(j) = {stage_bounds}, LB_MAX = {self.obj_lower_bound}"
        )

    def dispatch_by_jh1(self, draw_gantt: bool = False) -> None:
        """
        Dispatch jobs in the order determined by Johnson's rule for the first and last stages.
        This method computes the job sequence using the Johnson-based Heuristic 1 (jh1) and
        dispatches jobs sequentially to create a feasible schedule.
        """
        job_sequence = self.get_jh1_sequence()
        self.dispatch_sequentially(job_sequence, draw_gantt=draw_gantt)

    def dispatch_by_jh2(self, draw_gantt: bool = False) -> None:
        """
        Dispatch jobs in the order determined by Johnson's rule for the first half and second half stages.
        This method computes the job sequence using the Johnson-based Heuristic 2 (jh2) and
        dispatches jobs sequentially to create a feasible schedule.
        """
        job_sequence = self.get_jh2_sequence()
        self.dispatch_sequentially(job_sequence, draw_gantt=draw_gantt)

    # End subroutine definition

    def post_run_process(self) -> None:
        self.check_feasibility(self.incumbent_solution_manager.start_times)
        self.release_log_handlers()

    def check_feasibility(self, start_times: dict[tuple[str, str, str], int]) -> None:
        """Check the feasibility of the given start times.

        Args:
            start_times (dict[tuple[str, str, str], int]): _description_

        Raises:
            ValueError: If any start time is negative or invalid.
            RuntimeError: If the feasibility check fails while solving the model.
            ValueError: If the feasibility check fails with an unexpected status.
        """
        logging.info("Feasibility check starts")
        for (j, i, k), start_time in start_times.items():
            if start_time < 0:
                raise ValueError(
                    f"Invalid start time for job {j}, stage {i}, machine {k}: {start_time}"
                )
        base_cp = self.create_base_cp_model()

        # Freeze operation start times and machine assignments
        for (j, i, k), start_time in start_times.items():
            base_cp.add(self.cp_model.var_op_is_present[j, i, k] == 1)
            base_cp.add(self.cp_model.var_op_start[j, i, k] == start_time)

        # Solve with tight time limit
        try:
            summary = self.solve_cp_model(base_cp, 1.0, 1)
        except Exception as e:
            raise RuntimeError(f"Feasibility check FAILED: {e}") from e

        if not SolverStatus.is_optimal_solution(summary.status):
            raise ValueError(f"Feasibility check FAILED with status: {summary.status}")

        logging.info("Feasibility check passed")
