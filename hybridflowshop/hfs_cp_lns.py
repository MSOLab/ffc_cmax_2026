import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from mbls import (
    DynamicDataObject,
    ExperimentSummary,
    SolverOutputSummary,
    SolverStatus,
    SubroutineController,
)
from schore.hybridflowshop import HybridFlowShopProblem

from .plotter import ObjectiveProgressPlotter
from .pure_cp_2023_naderi import PureCP2023Naderi
from .solution_manager import SolutionManager
from .stopping_criteria import StoppingCriteria


class HybridFlowShopCpLnsController(SubroutineController):
    stopping_criteria: StoppingCriteria

    hfs_instance: HybridFlowShopProblem
    cp_model: PureCP2023Naderi
    last_solution_manager: SolutionManager
    """Manages the last(most recent) solution."""
    incumbent_solution_manager: SolutionManager
    """Manages the incumbent solution."""

    def __init__(
        self,
        hfs_instance: HybridFlowShopProblem,
        subroutine_flow: DynamicDataObject,
        stopping_criteria: StoppingCriteria,
        horizon: int,
    ):
        super().__init__(hfs_instance.name, subroutine_flow, stopping_criteria)
        self.hfs_instance = hfs_instance
        self.cp_model = PureCP2023Naderi(hfs_instance, horizon)
        self.cp_model.freeze_base_constraints()

    # Start stopping condition

    def is_stopping_condition(self) -> bool:
        return self.time_is_up()

    def time_is_up(self) -> bool:
        # If total elapsed time exceeds the stopping criteria
        if self.timer.get_elapsed_sec() >= self.stopping_criteria.timelimit:
            print("Stop by timelimit")
            return True
        return False

    # End stopping condition

    # Start solution management

    def set_last_solution_as_incumbent(self) -> None:
        """Set the incumbent solution."""
        self.incumbent_solution_manager = self.last_solution_manager
        self.draw_incumbent_gantt()

    def draw_incumbent_gantt(self, output_path: Path | None = None) -> None:
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
        if not hasattr(self, "incumbent_solution_manager"):
            return True
        return (
            self.last_solution_manager.summary.objective_value is not None
            and self.incumbent_solution_manager.summary.objective_value is not None
            and self.last_solution_manager.summary.objective_value
            < self.incumbent_solution_manager.summary.objective_value
        )

    # End solution management

    # Start experiment summary methods

    def get_experiment_summary(self) -> ExperimentSummary:
        """Get the experiment summary.

        Returns:
            ExperimentSummary: The experiment summary object.
        """
        return self.experiment_summary

    # End experiment summary methods

    # Start subroutine definition

    def solve_cp(
        self,
        computational_time: float,
        n_threads: int,
        check_feasibility=False,
        set_incumbent_solution=False,
        update_incumbent_solution=False,
    ):
        """Solve current CP model.

        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
            check_feasibility (bool, optional): If True, check feasibility of the solution. Defaults to False.
            set_incumbent_solution (bool, optional): If True, set the solution as the incumbent. Defaults to False.
            update_incumbent_solution (bool, optional): If True, update the incumbent solution. Defaults to False.
        """

        (solver_status, elapsed_time, obj_value, obj_bound) = self.cp_model.solve(
            computational_time, n_threads, self.timer
        )
        progress_log = self.cp_model.get_progress_log()

        summary = SolverOutputSummary(
            solver_status, elapsed_time, obj_value, obj_bound, progress_log
        )
        start_times, end_times = self.cp_model.extract_start_end_times()

        if check_feasibility:
            self.check_feasibility(start_times)
        self.experiment_summary.add_run_summary(summary)
        self.last_solution_manager = SolutionManager(start_times, end_times, summary)

        if set_incumbent_solution:
            self.set_last_solution_as_incumbent()
        elif update_incumbent_solution:
            self.update_incumbent_solution()

    def solve_with_initial_solution(
        self,
        computational_time: float,
        n_threads: int,
        check_feasibility=False,
        update_incumbent_solution=False,
    ):
        """Solve CP model with the incumbent solution as the initial solution.

        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
            check_feasibility (bool, optional): If True, check feasibility of the solution. Defaults to False.
            update_incumbent_solution (bool, optional): If True, update the incumbent solution. Defaults to False.
        """
        self.incumbent_solution_manager.apply_hint_to(self.cp_model)
        self.solve_cp(
            computational_time,
            n_threads,
            check_feasibility=check_feasibility,
            update_incumbent_solution=update_incumbent_solution,
        )

    def freeze_solve_reset(
        self,
        freeze_method: Callable,
        computational_time: float,
        n_threads: int,
        check_feasibility=False,
        update_incumbent_solution=False,
    ):
        """Apply the freeze method, solve, and reset the model.

        Args:
            freeze_method (Callable): A callable that applies the freeze method to the CP model.
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
            check_feasibility (bool, optional): If True, check feasibility of the solution. Defaults to False.
            update_incumbent_solution (bool, optional): If True, update the incumbent solution. Defaults to False.
        """
        freeze_method()
        self.solve_with_initial_solution(
            computational_time,
            n_threads,
            check_feasibility=check_feasibility,
            update_incumbent_solution=update_incumbent_solution,
        )
        self.cp_model.delete_added_constraints()

    # Time window operator
    def apply_time_window_search(
        self,
        rho: float,
        computational_time: float,
        n_threads: int,
        check_feasibility=False,
        update_incumbent_solution=False,
    ):
        """Time window search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
            check_feasibility (bool, optional): If True, check feasibility of the solution. Defaults to False.
            update_incumbent_solution (bool, optional): If True, update the incumbent solution. Defaults to False.
        """

        self.freeze_solve_reset(
            lambda: self.apply_time_window_operator(rho),
            computational_time,
            n_threads,
            check_feasibility=check_feasibility,
            update_incumbent_solution=update_incumbent_solution,
        )

    def apply_time_window_operator(self, rho: float):
        """
        Apply the Time Window Operator to the current CP model.

        Args:
            current_start_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current start_time
            current_end_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current end_time
            rho (float, optional): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
        """
        print(f"Applying time window operator with rho={rho}")

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

        print(
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

    def is_within_window(self, time: int, window_start: int, window_end: int) -> bool:
        """Check if a given time is within the specified window."""
        return window_start <= time <= window_end

    def check_feasibility(self, start_times: dict[tuple[str, str, str], int]) -> None:
        """check feasibility of the given solution.

        Args:
            start_times (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> start time (int)

        Raises:
            RuntimeError: If error occurred during feasibility check CP solving.
            ValueError: If given schedule is not feasible.
        """
        feasibility_cp = PureCP2023Naderi(self.hfs_instance, self.cp_model.horizon)

        # Freeze operation start times and machine assignments
        for (j, i, k), start_time in start_times.items():
            feasibility_cp.add(self.cp_model.var_op_is_present[j, i, k] == 1)
            feasibility_cp.add(self.cp_model.var_op_start[j, i, k] == start_time)

        # Solve with tight time limit
        try:
            solver_status, _, _, _ = feasibility_cp.solve(
                computational_time=1.0, n_threads=1, timer=self.timer
            )
        except Exception as e:
            raise RuntimeError(f"Feasibility check failed: {e}")

        if not SolverStatus.is_optimal_solution(solver_status):
            raise ValueError(
                f"Given schedule is not feasible. Solver status: {solver_status}"
            )

    # Block operator
    def apply_block_search(
        self,
        rho: float,
        computational_time: float,
        n_threads: int,
        check_feasibility=False,
        update_incumbent_solution=False,
    ):
        """Block search with incumbent solution as the hint.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
            check_feasibility (bool, optional): If True, check feasibility of the solution. Defaults to False.
            update_incumbent_solution (bool, optional): If True, update the incumbent solution. Defaults to False.
        """

        self.freeze_solve_reset(
            lambda: self.apply_block_operator(rho),
            computational_time,
            n_threads,
            check_feasibility=check_feasibility,
            update_incumbent_solution=update_incumbent_solution,
        )

    def apply_block_operator(self, rho: float):
        """
        Apply the Block Operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
        """
        print(f"Applying block operator with rho={rho}")

        start_times = self.incumbent_solution_manager.start_times
        end_times = self.incumbent_solution_manager.end_times

        if not start_times or not end_times:
            raise ValueError("No solution available for block operator.")

        all_ops = list(start_times.keys())
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

        print(
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
        experiment_summary_filename = "experiment_summary.yaml"
        solution_progress_fig_filename = "solution_progress.png"
        if self._working_dir_path is None:
            raise AttributeError("Working directory path is not set.")
        result_dir = self._working_dir_path / "result"
        # Check feasibility of the incumbent solution
        self.check_feasibility(self.incumbent_solution_manager.start_times)

        # Experiment summary -> YAML file
        self.experiment_summary.save_as_yaml(result_dir / experiment_summary_filename)

        # Plot solution progress
        ObjectiveProgressPlotter.plot_solution_progress(
            self.get_log(), result_dir / solution_progress_fig_filename
        )

    def get_log(self) -> list[tuple[float, float, float]]:
        return_list: list[tuple[float, float, float]] = []
        for run in self.experiment_summary.runs:
            if run.progress_log:
                return_list.extend(run.progress_log)
        return return_list
