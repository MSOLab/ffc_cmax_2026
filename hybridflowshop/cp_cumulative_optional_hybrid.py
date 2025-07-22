from __future__ import annotations

import math
from collections import defaultdict

from mbls.cpsat import CpModelWithOptionalFixedInterval
from ortools.sat.python.cp_model import IntVar
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)


class CPCumulativeOptionalHybrid(CpModelWithOptionalFixedInterval):
    """
    A hybrid CP model for the Hybrid Flowshop problem that combines the precision
    of Optional Interval variables for machine assignment with the powerful
    propagation of the Cumulative constraint to break symmetries on identical
    parallel machines.

    - It uses Optional Interval variables to decide the exact machine for each operation.
    - It introduces master variables (B_ji, C_ji) for the start/end time of an
      operation (j, i) regardless of the machine.
    - It adds a redundant Cumulative constraint for each stage. This global constraint
      acts on the master variables, stating that the number of concurrent operations
      cannot exceed the number of machines in that stage. This significantly
      improves performance by reducing the search space and breaking symmetries.
    """

    # Indices & Parameters

    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    M_of: dict[str, list[str]]
    """$M_i$: machine index (k) list for stage i"""

    p: dict[tuple[str, str], int]
    """$P_{ji}$: processing time of job j at stage i"""

    # Variable aliases

    B_ji: dict[tuple[str, str], IntVar]
    """$B_{ji}$: beginning time of job j at stage i"""

    C_ji: dict[tuple[str, str], IntVar]
    """$C_{ji}$: completion time of job j at stage i"""

    # Objective
    obj_var: IntVar
    """Defines the makespan objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__(horizon)

    @classmethod
    def from_instance(
        cls, instance: HybridFlowshopParameters, horizon: int
    ) -> "CPCumulativeOptionalHybrid":
        """Creates a CPCumulativeOptionalHybrid model from a HybridFlowshopParameters instance."""
        result = cls(horizon)
        result.define_model(instance)
        return result

    def define_model(self, instance: HybridFlowshopParameters) -> None:
        self.define_parameters(instance)
        self.define_variables()
        self.define_makespan_objective()
        self.define_constraints()

    # Parameters

    def define_parameters(self, instance: HybridFlowshopParameters) -> None:
        self.j_list = instance.job_id_list
        self.i_list = instance.stage_id_list
        self.M_of = instance.stage_2_machines_map
        _p = instance.p_manager.job_stage_2_value_map(self.j_list, self.i_list)
        self.p = {
            (j, i): int(float(_p[j, i])) for j in self.j_list for i in self.i_list
        }

    # Variables

    def define_variables(self) -> None:
        # Define optional interval variables for each specific operation (job, stage, machine)
        for j in self.j_list:
            for i in self.i_list:
                for k in self.M_of[i]:
                    self.define_optional_fixed_interval_var((j, i, k), self.p[j, i])

        # Define master interval variables for each (job, stage) pairs
        self.B_ji = {}
        self.C_ji = {}
        for j in self.j_list:
            for i in self.i_list:
                self.define_fixed_interval_var((j, i), self.p[j, i])
                # Aliases for readability
                self.B_ji[j, i] = self.var_op_start[j, i]
                self.C_ji[j, i] = self.var_op_end[j, i]

    # Objective

    def define_makespan_objective(self) -> None:
        # alias for readability
        last_i = self.i_list[-1]

        makespan = self.new_int_var(0, self.horizon, "makespan")
        # Define the makespan as the maximum end time of master interval variables at the last stage
        self.add_max_equality(makespan, [self.C_ji[j, last_i] for j in self.j_list])

        self.minimize(makespan)
        self.obj_var = makespan

    def set_obj_lower_bound(self, bound: float) -> None:
        """Sets a lower bound on the objective variable, handling potential float precision issues."""
        if self.obj_var is None:
            return
        epsilon = 1e-9
        if abs(bound - round(bound)) < epsilon:
            int_bound = round(bound)
        else:
            int_bound = math.ceil(bound)
        self.add(self.obj_var >= int_bound)

    # Constraints

    def define_constraints(self) -> None:
        # Alias for readability
        j_list = self.j_list
        i_list = self.i_list
        M_of = self.M_of

        # --- Base Constraints for Optional Interval Formulation ---

        # One machine must be selected for each operation
        for j in j_list:
            for i in i_list:
                self.add(sum(self.var_op_is_present[j, i, k] for k in M_of[i]) == 1)

        # NoOverlap on each specific machine
        for i in i_list:
            for k in M_of[i]:
                self.add_no_overlap([self.var_op_intvl[j, i, k] for j in j_list])

        # --- Channeling Constraints: Link master variables with optional variables ---
        for j in j_list:
            for i in i_list:
                # Link start times
                for k in M_of[i]:
                    self.add(
                        self.B_ji[j, i] == self.var_op_start[j, i, k]
                    ).only_enforce_if(self.var_op_is_present[j, i, k])
                # Link end times (master end = master start + processing time)
                self.add(self.C_ji[j, i] == self.B_ji[j, i] + self.p[j, i])

        # --- Redundant Global Constraint for Symmetry Breaking ---
        # Capacity constraints for each stage
        for i in i_list:
            # Use master intervals
            intervals = [self.var_op_intvl[j, i] for j in j_list]
            demands = [1] * len(j_list)
            capacity = len(self.M_of[i])
            self.add_cumulative(intervals, demands, capacity)

        # --- Precedence Constraints between Stages ---
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                self.add(self.C_ji[j, i] <= self.B_ji[j, next_i])

    # Subproblem generation

    def create_problem_of_job_subset(
        self, job_subset: set[str]
    ) -> "CPCumulativeOptionalHybrid":
        """Creates a new problem instance with a subset of jobs."""
        if not job_subset.issubset(self.j_list):
            raise ValueError("Job subset must be a subset of the original job list.")

        new_model = CPCumulativeOptionalHybrid(self.horizon)
        new_model.j_list = [j for j in self.j_list if j in job_subset]
        new_model.i_list = self.i_list
        new_model.M_of = {i: [k for k in self.M_of[i]] for i in self.i_list}
        new_model.p = {
            (j, i): self.p[j, i] for j in new_model.j_list for i in new_model.i_list
        }
        new_model.define_variables()
        new_model.define_makespan_objective()
        new_model.define_constraints()
        return new_model

    # extraction method for the solution

    def extract_start_end_time_map(
        self,
    ) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
        """Extracts start and end times from a solved CP model."""
        start_time_map: dict[tuple[str, str, str], int] = {}
        end_time_map: dict[tuple[str, str, str], int] = {}

        for j in self.j_list:
            for i in self.i_list:
                for k in self.M_of[i]:
                    is_present_var = self.var_op_is_present[j, i, k]
                    if self.solver.Value(is_present_var):
                        start_var = self.var_op_start[j, i, k]
                        end_var = self.var_op_end[j, i, k]
                        start_time_map[(j, i, k)] = self.solver.Value(start_var)
                        end_time_map[(j, i, k)] = self.solver.Value(end_var)
        return start_time_map, end_time_map

    # methods to add constraints for LNS - optional interval variables

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
        """
        if not ignore_integrity_check:
            assert j1 in self.j_list, f"Job {j1} not in job list."
            assert j2 in self.j_list, f"Job {j2} not in job list."
            assert i in self.i_list, f"Stage {i} not in stage list."
            assert k in self.M_of[i], f"Machine {k} not in machine list for stage {i}."

        self.add(self.var_op_end[j1, i, k] <= self.var_op_start[j2, i, k])

    def add_fixed_machine_and_ops_precedence_constraints_from_start_time_map(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        """Fixes the operations based on provided start times.

        Args:
            start_time_map (dict[tuple[str, str, str], int]): Mapping (j, i, k) -> start time (int)
            ignore_integrity_check (bool, optional): Skip data integrity check. Defaults to True.
        """
        stage_mc_to_jobs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for j, i, k in start_time_map:
            self.add_fixed_machine_assignment_constraint(
                j, i, k, ignore_integrity_check
            )
            stage_mc_to_jobs[(i, k)].append(j)

        for (i, k), jobs in stage_mc_to_jobs.items():
            jobs_sorted = sorted(jobs, key=lambda j: start_time_map[(j, i, k)])
            for j1, j2 in zip(jobs_sorted[:-1], jobs_sorted[1:]):
                self.add_fixed_operation_precedence_constraint(
                    j1, j2, i, k, ignore_integrity_check
                )

    # methods to add constraints for LNS - cumulative

    def add_operation_weak_precedence_constraint(
        self, j1: str, j2: str, i: str, ignore_integrity_check: bool = True
    ) -> None:
        """
        Adds a *weak* precedence constraint between two operations at a specific stage.

        - This method enforces that operation j1 must *start* before operation j2 starts at stage i.
        - It does not fix machine assignments but ensures the relative order of operations.

        Args:
            j1 (str): Job ID of the first operation.
            j2 (str): Job ID of the second operation.
            i (str): Stage ID where the precedence constraint applies.
            ignore_integrity_check (bool, optional): Skip data integrity check. Defaults to True.
        """
        if not ignore_integrity_check:
            assert j1 in self.j_list, f"Job {j1} not in job list."
            assert j2 in self.j_list, f"Job {j2} not in job list."
            assert i in self.i_list, f"Stage {i} not in stage list."

        self.add(self.var_op_start[j1, i] <= self.var_op_start[j2, i])

    def add_stage_ops_weak_precedence_constraints_from_start_time_map(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        """
        Adds weak precedence constraints for operations within each stage based on their start times.

        This method does not fix machine assignments but enforces the relative order of operations
        on each stage as determined by the provided start time map.

        Args:
            start_time_map (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> start time (int)
            ignore_integrity_check (bool, optional): Skip data integrity check. Defaults to True.
        """
        stage_to_job_mc_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        """i -> list of jobs j in stage i"""
        for j, i, k in start_time_map:
            if j not in stage_to_job_mc_pairs[i]:
                stage_to_job_mc_pairs[i].append((j, k))

        for i, j_k_list in stage_to_job_mc_pairs.items():
            # start time -> list of j
            s_time_to_jobs: dict[int, list[str]] = defaultdict(list)
            for j, k in j_k_list:
                s_time_to_jobs[start_time_map[j, i, k]].append(j)
            # Sort start times
            sorted_start_times = sorted(s_time_to_jobs.keys())
            # Add weak precedence constraints
            for idx in range(len(sorted_start_times) - 1):
                j1_list = s_time_to_jobs[sorted_start_times[idx]]
                j2_list = s_time_to_jobs[sorted_start_times[idx + 1]]
                for j1 in j1_list:
                    for j2 in j2_list:
                        self.add_operation_weak_precedence_constraint(
                            j1, j2, i, ignore_integrity_check=ignore_integrity_check
                        )

    # methods to add hints

    def add_start_and_present_hints_from_start_time_map(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
        fix_machine_assignment: bool = False,
    ) -> None:
        """Adds hints for start times and presence indicators based on a provided start time map.

        Args:
            start_time_map (dict[tuple[str, str, str], int]): Mapping of (j, i, k) to start time.
            ignore_integrity_check (bool, optional): Whether to skip integrity checks. Defaults to True.
            fix_machine_assignment (bool, optional): Whether to fix machine assignments. Defaults to False.
        """
        for (j, i, k), s_time in start_time_map.items():
            if not ignore_integrity_check:
                assert j in self.j_list, f"Job {j} not in job list."
                assert i in self.i_list, f"Stage {i} not in stage list."
                assert k in self.M_of[i], (
                    f"Machine {k} not in machine list for stage {i}."
                )

            if fix_machine_assignment:
                self.add_hint(self.var_op_start[j, i, k], s_time)
                self.add_hint(self.var_op_is_present[j, i, k], 1)
            else:
                self.add_hint(self.B_ji[j, i], s_time)
