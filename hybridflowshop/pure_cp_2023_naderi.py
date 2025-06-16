from __future__ import annotations

from collections import defaultdict
from typing import Optional

from mbls import ElapsedTimer
from mbls.cpsat import CpModelWithOptionalFixedInterval
from schore.hybridflowshop import HybridFlowShopProblem


class PureCP2023Naderi(CpModelWithOptionalFixedInterval):
    # Indices & Parameters

    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    M_of: dict[str, list[str]]
    """$M_i$: machine index (k) list for stage i"""

    p: dict[tuple[str, str], int]
    """$P_{ji}$: processing time of job j at stage i"""

    def __init__(self, horizon: int):
        super().__init__(horizon)

    @classmethod
    def from_instance(
        cls, hfs_instance: HybridFlowShopProblem, horizon: int
    ) -> "PureCP2023Naderi":
        """Creates a PureCP2023Naderi model from a HybridFlowShopProblem instance.

        Args:
            hfs_instance (HybridFlowShopProblem): The hybrid flow shop problem instance.
            horizon (int): The time horizon for the scheduling problem.

        Returns:
            PureCP2023Naderi: An instance of the PureCP2023Naderi model.
        """
        result = cls(horizon)
        result.define_model(hfs_instance)
        return result

    def define_model(self, hfs_instance: HybridFlowShopProblem):
        self.define_parameters(hfs_instance)
        self.define_variables()
        self.define_makespan_objective()
        self.define_constraints()

    def solve(
        self,
        computational_time: float,
        n_threads: int,
        random_seed: Optional[int] = None,
        timer: Optional[ElapsedTimer] = None,
    ) -> tuple[str, float, float, float]:
        """Solve the CP model.

        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
            timer (Optional[ElapsedTimer], optional): Timer to be passed to solver callback. Defaults to None.

        Returns:
            tuple[str, float, float, float]: A tuple containing
            - the solver status as a string defined in SolverStatus,
            - elapsed time in seconds,
            - the upper bound of the objective function, and
            - the lower bound of the objective function.
        """  # noqa: E501
        return super().solve_with_prog_logger(
            computational_time, n_threads, random_seed, timer
        )

    def get_progress_log(self) -> list[tuple[float, float, float]]:
        """Returns the log list.

        Returns:
            list[tuple[float, float, float]]: a list of tuples
                containing (elapsed time, objective value, best bound)
        """
        return self.sol_prog_logger.get_log()

    # Parameters

    def define_parameters(self, hfs_instance: HybridFlowShopProblem):
        self.j_list = hfs_instance.job_id_list
        self.i_list = hfs_instance.stage_id_list
        self.M_of = hfs_instance.stage_2_machines_map
        _p = hfs_instance.p_manager.job_stage_2_value_map(self.j_list, self.i_list)
        self.p = {
            (j, i): int(float(_p[j, i])) for j in self.j_list for i in self.i_list
        }

    # Variables

    def define_variables(self):
        # Define variables for each operation in each job
        for j in self.j_list:
            for i in self.i_list:
                for k in self.M_of[i]:
                    self.define_optional_fixed_interval_var(j, i, k, self.p[j, i])

    # Objective

    def define_makespan_objective(self):
        # alias for readability
        j_list = self.j_list
        i_list = self.i_list
        M_of = self.M_of

        makespan = self.new_int_var(0, self.horizon, "makespan")
        self.add_max_equality(
            makespan,
            [self.var_op_end[j, i, k] for j in j_list for i in i_list for k in M_of[i]],
        )

        self.minimize(makespan)

    # Constraints

    def define_constraints(self):
        # Alias for readability
        j_list = self.j_list
        i_list = self.i_list
        M_of = self.M_of

        # Constraints: NoOverlap

        for i in i_list:
            for k in M_of[i]:
                self.add_no_overlap([self.var_op_intvl[j, i, k] for j in j_list])

        # Constraints: Alternative

        for j in j_list:
            for i in i_list:
                self.add(sum(self.var_op_is_present[j, i, k] for k in M_of[i]) == 1)

        # Constraints: EndBeforeStart

        consecutive_stage_pairs = []
        for stage_idx, i in enumerate(i_list[:-1]):
            next_i = i_list[stage_idx + 1]
            consecutive_stage_pairs.append((i, next_i))

        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                for k in M_of[i]:
                    for next_k in M_of[next_i]:
                        self.add(
                            self.var_op_end[j, i, k]
                            <= self.var_op_start[j, next_i, next_k]
                            # ).only_enforce_if(
                            #     [
                            #         self.var_op_is_present[j, i, k],
                            #         self.var_op_is_present[j, next_i, next_k],
                            #     ]
                        )

    # Subproblem generation

    def create_problem_of_job_subset(
        self,
        job_subset: set[str],
    ) -> PureCP2023Naderi:
        """Creates a new problem instance with a subset of jobs.

        Args:
            job_subset (set[str]): A set of job indices to include in the new problem.

        Raises:
            ValueError: If the job subset is not a subset of the original job list.

        Returns:
            PureCP2023Naderi: A new instance of the PureCP2023Naderi model
                with the specified job subset.
        """
        if not job_subset.issubset(self.j_list):
            raise ValueError("Job subset must be a subset of the original job list.")
        # Create a new instance of the model
        new_model = PureCP2023Naderi(self.horizon)

        # Filter parameters based on the job subset
        new_model.j_list = [j for j in self.j_list if j in job_subset]
        new_model.i_list = self.i_list
        new_model.M_of = {i: [k for k in self.M_of[i]] for i in self.i_list}
        new_model.p = {
            (j, i): self.p[j, i] for j in new_model.j_list for i in new_model.i_list
        }
        # Define variables, objective, and constraints for the new model
        new_model.define_variables()
        new_model.define_makespan_objective()
        new_model.define_constraints()

        return new_model

    # extraction method for the solution

    def extract_start_end_times(
        self,
    ) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
        """Extracts start and end times from a solved CP model.

        Returns:
            tuple:
                - start_times (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> start time (int)
                - end_times (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> end time (int)
        """
        start_times: dict[tuple[str, str, str], int] = {}
        end_times: dict[tuple[str, str, str], int] = {}

        for j in self.j_list:
            for i in self.i_list:
                for k in self.M_of[i]:
                    start_var = self.var_op_start[j, i, k]
                    end_var = self.var_op_end[j, i, k]
                    is_present_var = self.var_op_is_present[j, i, k]

                    # Check if this operation is selected (is_present == 1)
                    if self.solver.Value(is_present_var):
                        start_value = self.solver.Value(start_var)
                        end_value = self.solver.Value(end_var)
                        start_times[(j, i, k)] = start_value
                        end_times[(j, i, k)] = end_value

        return start_times, end_times

    # methods to add constraints for LNS

    def add_fixed_machine_assignment_constraint(
        self, j: str, i: str, k: str, ignore_integrity_check: bool = True
    ) -> None:
        """Adds a constraint to fix a specific job and to a stage's machine.

        Args:
            j (str): job index
            i (str): stage index
            k (str): machine index
            ignore_integrity_check (bool, optional): Skip data integrity check. Defaults to True.
        """
        if not ignore_integrity_check:
            assert j in self.j_list, f"Job {j} not in job list."
            assert i in self.i_list, f"Stage {i} not in stage list."
            assert k in self.M_of[i], f"Machine {k} not in machine list for stage {i}."

        self.add(self.var_op_is_present[j, i, k] == 1)

    def add_fixed_operation_precedence_constraint(
        self, j1: str, j2: str, i: str, k: str, ignore_integrity_check: bool = True
    ) -> None:
        """Adds a precedence constraint between two operations on the same machine.
        The operation of job j1 must finish before the operation of job j2 starts.

        Args:
            j1 (str): preceding job index
            j2 (str): succeeding job index
            i (str): stage index
            k (str): machine index
            ignore_integrity_check (bool, optional): Skip data integrity check. Defaults to True.
        """  # noqa: E501
        if not ignore_integrity_check:
            assert j1 in self.j_list, f"Job {j1} not in job list."
            assert j2 in self.j_list, f"Job {j2} not in job list."
            assert i in self.i_list, f"Stage {i} not in stage list."
            assert k in self.M_of[i], f"Machine {k} not in machine list for stage {i}."

        self.add(self.var_op_end[j1, i, k] <= self.var_op_start[j2, i, k])

    def add_fixed_machine_and_ops_precedence_constraints_from_start_times(
        self,
        start_times: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        """Fixes the operations based on provided start times.

        Args:
            start_times (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> start time (int)
            ignore_integrity_check (bool, optional): Skip data integrity check. Defaults to True.
        """
        stage_mc_to_jobs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for j, i, k in start_times:
            self.add_fixed_machine_assignment_constraint(
                j, i, k, ignore_integrity_check
            )
            stage_mc_to_jobs[(i, k)].append(j)

        for (i, k), jobs in stage_mc_to_jobs.items():
            jobs_sorted = sorted(jobs, key=lambda j: start_times[(j, i, k)])
            for j1, j2 in zip(jobs_sorted[:-1], jobs_sorted[1:]):
                self.add_fixed_operation_precedence_constraint(
                    j1, j2, i, k, ignore_integrity_check
                )

    # methods to add hints
    def add_start_and_present_hints_from_start_times(
        self,
        start_times: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        for (j, i, k), s_time in start_times.items():
            if not ignore_integrity_check:
                assert j in self.j_list, f"Job {j} not in job list."
                assert i in self.i_list, f"Stage {i} not in stage list."
                assert k in self.M_of[i], (
                    f"Machine {k} not in machine list for stage {i}."
                )
            self.add_hint(self.var_op_start[j, i, k], s_time)
            self.add_hint(self.var_op_is_present[j, i, k], 1)
