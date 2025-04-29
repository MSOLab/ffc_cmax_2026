import random
from collections import defaultdict

from clad import (
    DynamicDataObject,
    SolverOutputSummary,
    SolverStatus,
    SubroutineController,
)
from plotter import ObjectiveProgressPlotter
from pure_cp_2023_naderi import PureCP2023Naderi
from schore.hybridflowshop.problem import HybridFlowShopProblem
from solution_manager import SolutionManager
from stopping_criteria import StoppingCriteria


class HybridFlowShopCpLnsController(SubroutineController):
    stopping_criteria: StoppingCriteria

    hfs_instance: HybridFlowShopProblem
    cp_model: PureCP2023Naderi
    incumbent_solution_manager: SolutionManager

    # the latest solution by solve_cp
    start_times_latest_sol: dict[tuple[str, str, str], int]
    """(job, stage, machine) -> start time (int)"""
    end_times_latest_sol: dict[tuple[str, str, str], int]
    """(job, stage, machine) -> end time (int)"""
    summary_latest_sol: SolverOutputSummary
    """summary object from the last call of solve_cp"""

    progress_log: list[tuple[float, float, float]]
    """List of tuples (elapsed_time, objective_value, best_objective_bound)"""

    def __init__(
        self,
        hfs_instance: HybridFlowShopProblem,
        stopping_criteria: StoppingCriteria,
        subroutine_flow: DynamicDataObject,
        horizon: int,
    ):
        super().__init__(hfs_instance.name, stopping_criteria, subroutine_flow)
        self.hfs_instance = hfs_instance
        self.cp_model = PureCP2023Naderi(hfs_instance, horizon)
        self.cp_model.freeze_base_constraints()

        self.progress_log = []

    def is_stopping_condition(self) -> bool:
        # If total elapsed time exceeds the stopping criteria
        if self.timer.get_elapsed_sec() >= self.stopping_criteria.timelimit:
            print("Stop by timelimit")
            return True
        return False

    def set_incumbent_solution(self) -> None:
        """Set the incumbent solution."""
        self.incumbent_solution_manager = SolutionManager(
            self.start_times_latest_sol,
            self.end_times_latest_sol,
            self.summary_latest_sol,
        )
        if self.experiment_summary.initial_obj is None:
            # 최초 solve_cp일 경우, 기록
            self.experiment_summary.record_initial_solution(
                self.summary_latest_sol.objective_value,
                self.summary_latest_sol.best_objective_bound,
            )

    def update_incumbent_solution(self) -> None:
        if not hasattr(self, "incumbent_solution_manager"):
            raise ValueError("No incumbent solution available to update.")
        new_solution_manager = SolutionManager(
            self.start_times_latest_sol,
            self.end_times_latest_sol,
            self.summary_latest_sol,
        )
        if (
            self.incumbent_solution_manager.summary.objective_value
            >= self.summary_latest_sol.objective_value
        ):
            self.incumbent_solution_manager = new_solution_manager

    def get_result_summary(self):
        return self.incumbent_solution_manager.get_result_summary()

    def save_incumbent_gantt_as_png(self, filename_format: str):
        filename = filename_format.format(ins_name=self.hfs_instance.name)
        self.incumbent_solution_manager.save_gantt_as_png(
            filename, self._working_dir_path
        )

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
        """  # noqa: E501

        self.summary_latest_sol = self.cp_model.solve_with_summary(
            computational_time, n_threads, self.timer
        )
        self.start_times_latest_sol, self.end_times_latest_sol = (
            self.cp_model.extract_start_end_times()
        )
        if check_feasibility:
            self.check_feasibility(self.start_times_latest_sol)
        if set_incumbent_solution:
            self.set_incumbent_solution()
            self.progress_log.extend(self.summary_latest_sol.progress_log)
        elif update_incumbent_solution:
            self.update_incumbent_solution()
            self.progress_log.extend(self.summary_latest_sol.progress_log)

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
        """  # noqa: E501

        self.apply_time_window_operator(rho)
        self.incumbent_solution_manager.apply_hint_to(self.cp_model)
        self.solve_cp(
            computational_time,
            n_threads,
            check_feasibility=check_feasibility,
            update_incumbent_solution=update_incumbent_solution,
        )
        # Remove added constraints
        self.cp_model.delete_added_constraints()

    def apply_time_window_operator(self, rho: float):
        """
        Apply the Time Window Operator to the current CP model.

        Args:
            current_start_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current start_time
            current_end_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current end_time
            rho (float, optional): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
        """  # noqa: E501
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

    def check_latest_feasibility(self) -> None:
        """Check feasibility of the latest solution."""
        self.check_feasibility(self.start_times_latest_sol)

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
            feasibility_cp.add(self.cp_model.var_op_is_present[j][i][k] == 1)
            feasibility_cp.add(self.cp_model.var_op_start[j][i][k] == start_time)

        # Solve with tight time limit
        try:
            summary = feasibility_cp.solve_with_summary(
                computational_time=1.0, n_threads=1, timer=self.timer
            )
        except Exception as e:
            raise RuntimeError(f"Feasibility check failed: {e}")

        SolverStatus.raise_if_not_feasible(summary.status)

    def post_run_process(self):
        # Check feasibility of the incumbent solution
        self.check_feasibility(self.incumbent_solution_manager.start_times)

        # Experiment summary -> YAML file
        self.experiment_summary.record_final_solution(
            self.incumbent_solution_manager.get_obj_value(),
            self.incumbent_solution_manager.get_obj_bound(),
        )
        self.experiment_summary.record_total_elapsed_time(self.timer.get_elapsed_sec())
        self.experiment_summary.record_feasibility(True)  # assume checked already
        experiment_summary_path = (
            self._working_dir_path
            / f"{self.experiment_summary.name}_experiment_summary.yaml"
        )
        self.experiment_summary.save_as_yaml(experiment_summary_path)

        # Plot solution progress
        ObjectiveProgressPlotter.plot_solution_progress(
            self.progress_log,
            save_path=self._working_dir_path
            / f"{self.experiment_summary.name}_solution_progress.png",
        )
