import random
from collections import defaultdict

from clad import DynamicDataObject, SolverStatus, SubroutineController
from pure_cp_2023_naderi import PureCP2023Naderi
from schore.hybridflowshop.problem import HybridFlowShopProblem
from stopping_criteria import StoppingCriteria


class HybridFlowShopCpLnsController(SubroutineController):
    _stopping_criteria: StoppingCriteria

    hfs_instance: HybridFlowShopProblem
    cp_model: PureCP2023Naderi

    def __init__(
        self,
        hfs_instance: HybridFlowShopProblem,
        stopping_criteria: StoppingCriteria,
        subroutine_flow: DynamicDataObject,
        horizon: int,
    ):
        super().__init__(stopping_criteria, subroutine_flow)
        self.hfs_instance = hfs_instance
        self.cp_model = PureCP2023Naderi(hfs_instance, horizon)

    def is_stopping_condition(self) -> bool:
        # If total elapsed time exceeds the stopping criteria
        if self._timer.get_elapsed_sec() >= self._stopping_criteria.timelimit:
            print("Stop by timelimit")
            return True
        return False

    def get_result_summary(self):
        return self.cp_model.summary

    def solve_cp(
        self, computational_time: float, n_threads: int, check_feasibility: bool = False
    ):
        """Solve current CP model.

        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
        """
        print(
            f"Solving CP model with computational_time={computational_time}"
            f", n_threads={n_threads}"
        )
        self.cp_model.solve_with_summary(computational_time, n_threads, self.timer)
        self.cp_model.delete_added_constraints()
        start_times, _ = self.cp_model.extract_start_end_times()
        if check_feasibility:
            self.check_feasibility(start_times)

    def apply_time_window_search(
        self, rho: float, computational_time: float, n_threads: int
    ):
        """Time window search with incumbent solution

        Args:
            rho (float): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
        """
        start_times, end_times = self.cp_model.extract_start_end_times()
        self.apply_time_window_operator(start_times, end_times, rho)
        self.solve_cp(computational_time, n_threads)

    def apply_time_window_operator(
        self,
        current_start_times: dict[tuple[str, str, str], int],
        current_end_times: dict[tuple[str, str, str], int],
        rho: float,
    ):
        """
        Apply the Time Window Operator to the current CP model.

        Args:
            current_start_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current start_time
            current_end_times (dict[tuple[str, str, str], int]): (job_name, stage_name, machine_id) -> current end_time
            rho (float, optional): Fraction of makespan to define the window size (e.g., 0.2 means 20% of makespan)
        """  # noqa: E501
        print(f"Applying time window operator with rho={rho}")

        # 1. Calculate makespan (C_max)
        all_end_times = list(current_end_times.values())
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

        for key in current_start_times:
            s_time = current_start_times[key]
            e_time = current_end_times[key]
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
            jobs_sorted = sorted(jobs, key=lambda j: current_start_times[(j, i, k)])
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
            feasibility_cp.add(self.cp_model.var_op_is_present[j][i][k] == 1)
            feasibility_cp.add(self.cp_model.var_op_start[j][i][k] == start_time)

        # Solve with tight time limit
        try:
            summary = feasibility_cp.solve_with_summary(
                computational_time=1.0, n_threads=1, timer=self.timer
            )
        except Exception as e:
            raise RuntimeError(f"Feasibility check failed: {e}")

        if not SolverStatus.found_feasible_solution(summary.status):
            raise ValueError(
                "Feasibility check failed: Given schedule is not feasible."
            )
