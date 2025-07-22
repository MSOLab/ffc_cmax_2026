from __future__ import annotations

import math
from collections import defaultdict

from mbls.cpsat.cp_model_with_fixed_interval import CpModelWithFixedInterval
from ortools.sat.python.cp_model import IntVar
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from .scheduling.hybrid_flowshop_operation import HybridFlowshopOperation
from .scheduling.hybrid_flowshop_schedule import HybridFlowshopSchedule


class CP2023NaderiCumulative(CpModelWithFixedInterval):
    """
    A specific implementation of the cumulative (pulse in IBM CP Optimizer) CP model
    for the Hybrid Flowshop problem.

    Reference:

    - Naderi, B., Ruiz, R., & Roshanaei, V. (2023).
      Mixed-integer programming vs. constraint programming for shop scheduling problems: new results and outlook.
      INFORMS Journal on Computing, 35(4), 817-843.
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

    # Objective
    obj_var: IntVar
    """Defines the makespan objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__(horizon)

    @classmethod
    def from_instance(
        cls, instance: HybridFlowshopParameters, horizon: int
    ) -> "CP2023NaderiCumulative":
        """
        Create a CP2023NaderiIntervalCumulative model from a HybridFlowshopParameters instance.

        Args:
            instance (HybridFlowshopParameters): The hybrid flow shop problem instance.
            horizon (int): The time horizon for the scheduling problem.

        Returns:
            CP2023NaderiIntervalCumulative: An instance of the model.
        """
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
        """
        Define the parameters for the model based on the HybridFlowshopParameters instance.

        Args:
            hfs_instance (HybridFlowshopParameters): The hybrid flow shop problem instance.
        """
        self.j_list = instance.job_id_list
        self.i_list = instance.stage_id_list
        self.M_of = instance.stage_2_machines_map
        _p = instance.p_manager.job_stage_2_value_map(self.j_list, self.i_list)
        self.p = {
            (j, i): int(float(_p[j, i])) for j in self.j_list for i in self.i_list
        }

    # Variables

    def define_variables(self) -> None:
        # Interval variables
        for j in self.j_list:
            for i in self.i_list:
                self.define_fixed_interval_var((j, i), self.p[j, i])

    # Objective

    def define_makespan_objective(self) -> None:
        # alias for readability
        j_list = self.j_list
        last_i = self.i_list[-1]

        makespan = self.new_int_var(0, self.horizon, "makespan")
        self.add_max_equality(makespan, [self.var_op_end[j, last_i] for j in j_list])

        self.minimize(makespan)
        self.obj_var = makespan

    def set_obj_lower_bound(self, bound: float) -> None:
        """
        Sets a lower bound on the objective variable, handling potential float precision issues.

        Args:
            bound (float): The lower bound to set for the objective variable.
        """
        if self.obj_var is None:
            return

        # A small tolerance to handle floating point inaccuracies
        epsilon = 1e-9

        # If the bound is very close to an integer, treat it as such.
        # Otherwise, use ceiling to ensure we don't cut off valid integer solutions.
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

        # Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                self.add(self.var_op_end[j, i] <= self.var_op_start[j, next_i])

        # Capacity constraints for each stage
        for i in i_list:
            intervals = [self.var_op_intvl[j, i] for j in j_list]
            demands = [1] * len(j_list)
            capacity = len(self.M_of[i])
            self.add_cumulative(intervals, demands, capacity)

    # Subproblem generation

    def create_problem_of_job_subset(
        self, job_subset: set[str]
    ) -> CP2023NaderiCumulative:
        if not job_subset.issubset(self.j_list):
            raise ValueError("Job subset must be a subset of the original job list.")
        # Create a new instance
        new_model = CP2023NaderiCumulative(self.horizon)

        # Filter parameters based on the job subset
        new_model.j_list = [j for j in self.j_list if j in job_subset]
        new_model.i_list = self.i_list  # Stages remain the same
        new_model.M_of = {i: [k for k in self.M_of[i]] for i in self.i_list}
        new_model.p = {
            (j, i): self.p[j, i] for j in new_model.j_list for i in new_model.i_list
        }
        # Define variables, objective, and constraints
        new_model.define_variables()
        new_model.define_makespan_objective()
        new_model.define_constraints()

        return new_model

    # extraction method for the solution

    def create_schedule(self) -> HybridFlowshopSchedule:
        """
        Constructs a full HybridFlowshopSchedule from the solved CP model.

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
            HybridFlowshopSchedule: A complete schedule object with all operations
                                    assigned to specific machines and time slots.
        """
        start_time_map: dict[str, dict[str, int]] = {}
        """stage ID -> job ID -> start time"""
        end_time_map: dict[str, dict[str, int]] = {}
        """stage ID -> job ID -> end time"""
        for i in self.i_list:
            start_time_map[i] = {}
            end_time_map[i] = {}
            for j in self.j_list:
                start_value = self.solver.Value(self.var_op_start[j, i])
                end_value = self.solver.Value(self.var_op_end[j, i])
                start_time_map[i][j] = start_value
                end_time_map[i][j] = end_value

        schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(self.M_of)

        for i in self.i_list:
            stage = schedule.get_stage_by_name(i)
            # For greedy machine assignment,
            # Sort operations at stage i by 1) their start time 2) their job index in self.j_list
            # This ensures that operations are assigned to machines in a consistent order.
            # This is important for the greedy assignment to work correctly.
            sorted_j_list = sorted(
                self.j_list,
                key=lambda j: (start_time_map[i][j], self.j_list.index(j)),
            )

            for j in sorted_j_list:
                start_time = start_time_map[i][j]
                end_time = end_time_map[i][j]
                # Select the machine for the operation based on the start time
                mc_name, earliest_feasible_st = stage.select_machine_by_start_idle_idx(
                    self.p[j, i], start_time
                )
                if earliest_feasible_st > start_time:
                    raise RuntimeError(
                        f"Operation {j} at stage {i} cannot start at {start_time} "
                        f"because the earliest feasible start time on machine {mc_name} is {earliest_feasible_st}."
                    )
                operation = stage.add_operation(
                    HybridFlowshopOperation(
                        job_name=j,
                        stage_name=i,
                        mc_name=mc_name,
                        start=start_time,
                        end=end_time,
                    )
                )
                if operation is None:
                    raise RuntimeError(
                        f"Failed to schedule operation {j} at stage {i}  during extraction "
                        f"on machine {mc_name} with start time {start_time} and end time {end_time}."
                    )
        return schedule

    def extract_start_end_time_map(
        self,
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
        schedule = self.create_schedule()
        return schedule.get_start_time_map(), schedule.get_end_time_map()

    # methods to add constraints for LNS

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

    def add_start_hints_from_start_time_map(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        for (j, i, _), s_time in start_time_map.items():
            if not ignore_integrity_check:
                assert j in self.j_list, f"Job {j} not in job list."
                assert i in self.i_list, f"Stage {i} not in stage list."
            self.add_hint(self.var_op_start[j, i], s_time)
