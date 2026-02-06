from __future__ import annotations

import math
from dataclasses import dataclass

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntervalVar, IntVar
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule

from .params import Params


@dataclass
class CumulativeVars:
    op_start: dict[tuple[str, str], IntVar]
    """
    (j,i) -> start time variables for each operation in a job.
    """

    op_end: dict[tuple[str, str], IntVar]
    """
    (j,i) -> end time variables for each operation in a job.
    """

    op_intvl: dict[tuple[str, str], IntervalVar]
    """
    (j,i) -> interval variables for each operation in a job.
    """

    makespan: IntVar
    """Makespan variable"""


class BaseModelBuilder:
    def build(
        self, instance: HybridFlowshopParameters, horizon: int
    ) -> tuple[CustomCpModel, Params, CumulativeVars]:
        mdl = CustomCpModel()
        params: Params = self._make_params(instance)
        variables: CumulativeVars = self._make_vars(mdl, params, horizon)
        self._add_structural_constraints(mdl, params, variables)
        self._define_objective(mdl, params, variables)
        mdl.set_num_base_constraints()

        return mdl, params, variables

    @staticmethod
    def _make_params(instance: HybridFlowshopParameters) -> Params:
        j_list = instance.job_id_list
        i_list = instance.stage_id_list
        M_of = instance.stage_2_machines_map
        _p = instance.p_manager.job_stage_2_value_map(j_list, i_list)
        p = {(j, i): int(float(_p[j, i])) for j in j_list for i in i_list}
        return Params(
            j_list=j_list,
            i_list=i_list,
            M_of=M_of,
            p=p,
        )

    @staticmethod
    def _make_vars(mdl: CustomCpModel, params: Params, horizon: int) -> CumulativeVars:
        op_start: dict[tuple[str, str], IntVar] = {}
        op_end: dict[tuple[str, str], IntVar] = {}
        op_intvl: dict[tuple[str, str], IntervalVar] = {}

        for j in params.j_list:
            for i in params.i_list:
                start_var = mdl.new_int_var(0, horizon, f"start_{j}_{i}")
                end_var = mdl.new_int_var(0, horizon, f"end_{j}_{i}")
                interval_var = mdl.new_interval_var(
                    start_var,
                    params.p[j, i],
                    end_var,
                    f"interval_{j}_{i}",
                )

                op_start[(j, i)] = start_var
                op_end[(j, i)] = end_var
                op_intvl[(j, i)] = interval_var

        makespan = mdl.new_int_var(0, horizon, "makespan")

        return CumulativeVars(
            op_start=op_start,
            op_end=op_end,
            op_intvl=op_intvl,
            makespan=makespan,
        )

    @staticmethod
    def _add_structural_constraints(
        mdl: CustomCpModel, params: Params, variables: CumulativeVars
    ) -> None:
        # Alias for readability
        j_list = params.j_list
        i_list = params.i_list

        # Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                mdl.add(variables.op_end[j, i] <= variables.op_start[j, next_i])

        # Capacity constraints for each stage
        for i in i_list:
            intervals = [variables.op_intvl[j, i] for j in j_list]
            demands = [1] * len(j_list)
            capacity = len(params.M_of[i])
            mdl.add_cumulative(intervals, demands, capacity)

    @staticmethod
    def _define_objective(
        mdl: CustomCpModel, params: Params, variables: CumulativeVars
    ) -> None:
        # Alias for readability
        j_list = params.j_list
        i_list = params.i_list
        last_i = i_list[-1]

        # Makespan definition
        mdl.add_max_equality(
            variables.makespan, [variables.op_end[j, last_i] for j in j_list]
        )

        # Set objective to minimize makespan
        mdl.minimize(variables.makespan)

    # Additional constraints

    @staticmethod
    def set_obj_lower_bound(
        mdl: CustomCpModel,
        variables: CumulativeVars,
        lower_bound: int | float,
    ) -> None:
        """Sets a lower bound on the makespan objective.

        Args:
            lower_bound (int | float): The lower bound value to set.
        """
        # A small tolerance to handle floating point inaccuracies
        epsilon = 1e-9

        # If the bound is very close to an integer, treat it as such.
        # Otherwise, use ceiling to ensure we don't cut off valid integer solutions.
        if abs(lower_bound - round(lower_bound)) < epsilon:
            int_bound = round(lower_bound)
        else:
            int_bound = math.ceil(lower_bound)

        mdl.add(variables.makespan >= int_bound)

    @staticmethod
    def add_fixed_operation_precedence_constraint(
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        j1: str,
        j2: str,
        i: str,
        ignore_integrity_check: bool = True,
    ) -> None:
        """Adds a precedence constraint between two operations.
        The operation of job j1 must finish before the operation of job j2 starts.

        Args:
            j1 (str): preceding job index
            j2 (str): succeeding job index
            i (str): stage index
            ignore_integrity_check (bool, optional): Skip data integrity check. Defaults to True.
        """
        if not ignore_integrity_check:
            assert j1 in params.j_list, f"Job {j1} not in job list."
            assert j2 in params.j_list, f"Job {j2} not in job list."
            assert i in params.i_list, f"Stage {i} not in stage list."

        mdl.add(variables.op_end[j1, i] <= variables.op_start[j2, i])

    @staticmethod
    def add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        current_schedule: HybridFlowshopLiteSchedule,
    ) -> None:
        start_time_map = current_schedule.get_start_time_map()
        end_time_map = current_schedule.get_end_time_map()
        for i in params.i_list:
            current_j_set = {j for j, ip, _ in start_time_map if ip == i}
            current_j_list = [j for j in params.j_list if j in current_j_set]
            stage_job_2_index_map = {j: idx for idx, j in enumerate(current_j_list)}
            # Extract start and end times for jobs at stage i
            # Map of job -> start time at stage i
            j_2_start_time_map = {
                j: start_time_map[j, i, k]
                for j in current_j_list
                for k in params.M_of[i]
                if (j, i, k) in start_time_map
            }
            # Map of job -> end time at stage i
            j_2_end_time_map = {
                j: end_time_map[j, i, k]
                for j in current_j_list
                for k in params.M_of[i]
                if (j, i, k) in end_time_map
            }
            # List of jobs sorted by their 1) end times 2) start times 3) job index
            sorted_j_list = sorted(
                current_j_list,
                key=lambda j: (
                    j_2_end_time_map.get(j, float("inf")),
                    j_2_start_time_map.get(j, float("inf")),
                    stage_job_2_index_map.get(j, float("inf")),
                ),
            )
            for idx, j1 in enumerate(sorted_j_list):
                remaining_jobs = sorted_j_list[idx + 1 :]
                if not remaining_jobs:
                    continue
                j1_end_time = j_2_end_time_map.get(j1, float("inf"))
                # Find (the number of machines of stage i) jobs that start earliest after j1_end_time.
                max_candidates = min(len(params.M_of[i]), len(remaining_jobs))
                # Only consider up to the number of machines, since at most that many jobs can start in parallel at this stage.
                j2_list = sorted(
                    (
                        j2
                        for j2 in remaining_jobs
                        if j_2_start_time_map.get(j2, float("inf")) >= j1_end_time
                    ),
                    key=lambda j2: (
                        j_2_start_time_map[j2],
                        j_2_end_time_map[j2],
                        stage_job_2_index_map[j2],
                    ),
                )[:max_candidates]
                # Add precedence constraints: j1 must finish before each j2 starts
                for j2 in j2_list:
                    BaseModelBuilder.add_fixed_operation_precedence_constraint(
                        mdl, params, variables, j1, j2, i
                    )

    @staticmethod
    def add_start_time_freezed_operation_constraints(
        mdl: CustomCpModel,
        variables: CumulativeVars,
        start_time_map: dict[tuple[str, str, str], int],
    ) -> None:
        for (j, i, k), s_time in start_time_map.items():
            mdl.add(variables.op_start[j, i] == s_time)

    # Hints

    @staticmethod
    def apply_start_hints_from_start_time_map(
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        start_time_map: dict[tuple[str, str, str], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        """Applies start time hints to the model from a given start time map.

        Args:
            start_time_map (dict[tuple[str, str, str], int]): A mapping from (job_id, stage_id, machine_id) to start time.
        """
        for (j, i, _), s_time in start_time_map.items():
            if not ignore_integrity_check:
                assert j in params.j_list, f"Job {j} not in job list."
                assert i in params.i_list, f"Stage {i} not in stage list."
            mdl.add_hint(variables.op_start[j, i], s_time)
