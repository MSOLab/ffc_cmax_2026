from __future__ import annotations

import math
from collections import defaultdict

from mbls.cpsat import CpModelWithOptionalFixedInterval
from ortools.sat.python.cp_model import IntVar
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)
from schore.schedule_examples.parallel_shop.identical_flow import (
    HybridFlowshopOperation,
    HybridFlowshopSchedule,
)


class CPOptionalIntervalMasterTimevar(CpModelWithOptionalFixedInterval):
    # Indices & Parameters

    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    M_of: dict[str, list[str]]
    """$M_i$: machine index (k) list for stage i"""

    p: dict[tuple[str, str], int]
    """$P_{ji}$: processing time of job j at stage i"""

    # Master time variables

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
    ) -> CPOptionalIntervalMasterTimevar:
        """Creates a model from a HybridFlowshopParameters instance.

        Args:
            instance (HybridFlowshopParameters): The hybrid flow shop problem instance.
            horizon (int): The time horizon for the scheduling problem.

        Returns:
            CPOptionalIntervalMasterTimevar: An instance of the model.
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
        self.j_list = instance.job_id_list
        self.i_list = instance.stage_id_list
        self.M_of = instance.stage_2_machines_map
        _p = instance.p_manager.job_stage_2_value_map(self.j_list, self.i_list)
        self.p = {
            (j, i): int(float(_p[j, i])) for j in self.j_list for i in self.i_list
        }

    # Variables

    def define_variables(self) -> None:
        # Define variables for each operation in each job
        for j in self.j_list:
            for i in self.i_list:
                for k in self.M_of[i]:
                    self.define_optional_fixed_interval_var((j, i, k), self.p[j, i])

        # Define master variables for each (job, stage)
        self.B_ji = {}
        self.C_ji = {}
        for j in self.j_list:
            for i in self.i_list:
                self.B_ji[j, i] = self.new_int_var(0, self.horizon, f"B_{j}_{i}")
                self.C_ji[j, i] = self.new_int_var(0, self.horizon, f"C_{j}_{i}")

    # Objective

    def define_makespan_objective(self) -> None:
        # alias for readability
        last_i = self.i_list[-1]

        makespan = self.new_int_var(0, self.horizon, "makespan")
        # The makespan is the maximum of the end times of the operations in the last stage.
        self.add_max_equality(makespan, [self.C_ji[j, last_i] for j in self.j_list])

        self.minimize(makespan)
        self.obj_var = makespan

    def set_obj_lower_bound(self, bound: float) -> None:
        """Sets a lower bound on the objective variable, handling potential float precision issues."""
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
        M_of = self.M_of

        # One machine must be selected for each operation
        for j in j_list:
            for i in i_list:
                self.add(sum(self.var_op_is_present[j, i, k] for k in M_of[i]) == 1)

        # Link master variables with optional interval variables
        for j in j_list:
            for i in i_list:
                # Link start times
                for k in M_of[i]:
                    self.add(
                        self.B_ji[j, i] == self.var_op_start[j, i, k]
                    ).only_enforce_if(self.var_op_is_present[j, i, k])
                # Link end times
                self.add(self.C_ji[j, i] == self.B_ji[j, i] + self.p[j, i])

        # NoOverlap on each machine
        for i in i_list:
            for k in M_of[i]:
                self.add_no_overlap([self.var_op_intvl[j, i, k] for j in j_list])

        # Constraints: Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                self.add(self.C_ji[j, i] <= self.B_ji[j, next_i])

    # Subproblem generation

    def create_problem_of_job_subset(
        self, job_subset: set[str]
    ) -> CPOptionalIntervalMasterTimevar:
        """Creates a new problem instance with a subset of jobs.

        Args:
            job_subset (set[str]): A set of job indices to include in the new problem.

        Raises:
            ValueError: If the job subset is not a subset of the original job list.

        Returns:
            CPOptionalIntervalMasterTimevar: A new instance of the model with the specified job subset.
        """
        if not job_subset.issubset(self.j_list):
            raise ValueError("Job subset must be a subset of the original job list.")
        new_model = self.__class__(self.horizon)

        # Filter parameters based on the job subset
        new_model.j_list = [j for j in self.j_list if j in job_subset]
        new_model.i_list = self.i_list
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

    def extract_start_end_time_map(
        self,
    ) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
        """Extracts start and end times from a solved CP model.

        Returns:
            tuple:
                - start_time_map (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> start time (int)
                - end_time_map (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> end time (int)
        """
        start_time_map: dict[tuple[str, str, str], int] = {}
        end_time_map: dict[tuple[str, str, str], int] = {}

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
                        start_time_map[(j, i, k)] = start_value
                        end_time_map[(j, i, k)] = end_value

        return start_time_map, end_time_map

    def create_schedule(self) -> HybridFlowshopSchedule:
        """Creates a HybridFlowshopSchedule from the current model's solution.

        Returns:
            HybridFlowshopSchedule: A schedule object containing the operations and their timings.
        """
        start_time_map, end_time_map = self.extract_start_end_time_map()

        schedule = HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(self.M_of)

        for (j, i, k), start_time in start_time_map.items():
            stage = schedule.get_stage_by_name(i)
            end_time = end_time_map[(j, i, k)]
            if start_time is not None and end_time is not None:
                operation = stage.add_operation(
                    HybridFlowshopOperation(
                        job_name=j,
                        stage_name=i,
                        mc_name=k,
                        start=start_time,
                        end=end_time,
                    )
                )
                if operation is None:
                    raise RuntimeError(
                        f"Failed to schedule operation of job {j} at stage {i} on machine {k}"
                        f" with start time {start_time} and end time {end_time}."
                    )
            else:
                raise ValueError(
                    f"Start or end time for operation of job {j} at stage {i} on machine {k} is None."
                )
        return schedule

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
            start_time_map (dict[tuple[str, str, str], int]): Mapping (job, stage, machine) -> start time (int)
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

    # methods to add hints

    def add_start_and_present_hints_from_start_time_map(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        for (j, i, k), s_time in start_time_map.items():
            if not ignore_integrity_check:
                assert j in self.j_list, f"Job {j} not in job list."
                assert i in self.i_list, f"Stage {i} not in stage list."
                assert k in self.M_of[i], (
                    f"Machine {k} not in machine list for stage {i}."
                )
            self.add_hint(self.var_op_start[j, i, k], s_time)
            self.add_hint(self.var_op_is_present[j, i, k], 1)
