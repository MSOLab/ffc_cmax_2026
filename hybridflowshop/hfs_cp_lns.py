import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from mbls import DynamicDataObject, SolverStatus
from mbls.cpsat.cp_subroutine_controller import CpSubroutineController
from schore.hybridflowshop import HybridFlowShopProblem

from hybridflowshop.solution_manager import SolutionManager

from .pure_cp_2023_naderi import PureCP2023Naderi
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

    def set_last_solution_as_incumbent(self) -> None:
        """Set the incumbent solution."""
        self.incumbent_solution_manager = self.last_solution_manager
        self.draw_incumbent_gantt()

    def draw_incumbent_gantt(self, output_path: Optional[Path] = None) -> None:
        if output_path is None:
            output_path = self.get_file_path_for_subroutine("_gantt.png")
        self.incumbent_solution_manager.save_gantt_as_png(output_path)

    def update_incumbent_solution(self) -> None:
        if self.last_solution_is_better_than_incumbent():
            self.set_last_solution_as_incumbent()

    def last_solution_is_better_than_incumbent(self) -> bool:
        """Check if the last solution is better than the incumbent solution."""
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

    def solve_cp(
        self,
        computational_time: float,
        num_workers: int,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
    ):
        """Solve the current CP model.

        Args:
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            obj_value_is_valid (bool, optional): If True, adds the objective value log.
                Defaults to False.
            obj_bound_is_valid (bool, optional): If True, adds the objective bound log.
                Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
        """
        _timelimit = self.get_remaining_time_limit(computational_time)
        summary = self.solve_current_cp_model(
            _timelimit,
            num_workers,
            random_seed=self.random_seed,
            timer=self.timer,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

        start_times, end_times = self.cp_model.extract_start_end_times()

        if error_if_infeasible:
            self.check_feasibility(start_times)
        self.experiment_summary.add_run_summary(summary)
        self.last_solution_manager = SolutionManager(
            start_times=start_times,
            end_times=end_times,
            summary=summary,
        )

        if obj_value_is_valid:
            self.update_incumbent_solution()

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
    ):
        """Solve the current CP model with the incumbent solution as the initial solution.

        Args:
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            obj_value_is_valid (bool, optional): If True, adds the objective value log.
                Defaults to False.
            obj_bound_is_valid (bool, optional): If True, adds the objective bound log.
                Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
        """
        if self.feasible_incumbent_solution_exists():
            self.incumbent_solution_manager.apply_start_and_present_hint_to(
                self.cp_model
            )
        self.solve_cp(
            computational_time,
            num_workers,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            error_if_infeasible=error_if_infeasible,
        )

    # Subroutine: solve base CP model

    def solve_base_cp_model(
        self,
        computational_time: float,
        num_workers: int,
        hint_from_incumbent: bool = False,
    ):
        """
        Solve the base CP model.
        This method resets the CP model, applies the incumbent solution as a hint if available,
        and solves the model with the given computational time and number of workers.

        Args:
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            hint_from_incumbent (bool, optional): If True, uses the incumbent solution as a hint.
                Defaults to False.
        """
        self.cp_model.delete_added_constraints()
        if hint_from_incumbent and self.feasible_incumbent_solution_exists():
            self.incumbent_solution_manager.apply_start_and_present_hint_to(
                self.cp_model
            )
        self.solve_cp(
            computational_time,
            num_workers,
            obj_value_is_valid=True,
            obj_bound_is_valid=True,
            error_if_infeasible=True,
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
        """
        freeze_method()
        self.solve_with_initial_solution(
            computational_time,
            num_workers,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            error_if_infeasible=error_if_infeasible,
        )
        self.cp_model.delete_added_constraints()

    # Subroutine: Time window operator

    def time_window_search(
        self,
        rho: float,
        computational_time: float,
        num_workers: int,
        error_if_infeasible=False,
    ):
        """Time window search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
        """
        self.freeze_solve_reset(
            lambda: self.apply_time_window_operator(rho),
            computational_time,
            num_workers,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
        )

    def apply_time_window_operator(self, rho: float):
        """
        Apply the Time Window Operator to the current CP model.

        Args:
            current_start_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current start_time
            current_end_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current end_time
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
        stage_mc_to_jobs = defaultdict(list)

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
    ):
        """Block search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            computational_time (float): The maximum computational time in seconds.
            num_workers (int): The number of parallel workers (i.e. threads) to use during search.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
        """

        self.freeze_solve_reset(
            lambda: self.apply_block_operator(rho),
            computational_time,
            num_workers,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
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

    # End subroutine definition

    def post_run_process(self) -> None:
        self.check_feasibility(self.incumbent_solution_manager.start_times)
        self.release_log_handlers()

    def check_feasibility(self, start_times: dict[tuple[str, str, str], int]) -> None:
        """Check the feasibility of the given start times.

        Args:
            start_times (dict[tuple[str, str, str], int]): _description_

        Raises:
            ValueError: _description_
            RuntimeError: _description_
            ValueError: _description_
        """
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
            raise RuntimeError(f"Feasibility check failed: {e}") from e

        if not SolverStatus.is_optimal_solution(summary.status):
            raise ValueError(f"Feasibility check failed with status: {summary.status}")

        logging.info("Feasibility check passed.")
