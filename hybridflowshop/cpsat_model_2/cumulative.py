from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Mapping

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntervalVar, IntVar
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    McIdType,
    StageIdType,
)

from .params import Params


@dataclass
class OperationVars:
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


@dataclass
class CumulativeVars(OperationVars):
    makespan: IntVar
    """Makespan variable"""


class BaseModelBuilder:
    def build(
        self,
        instance: HybridFlowshopParameters,
        horizon: int,
        minimize_sum_ci: bool = False,
        minimize_makespan_plus_sum_other_stages: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
    ) -> tuple[CustomCpModel, Params, CumulativeVars]:
        mdl = CustomCpModel()
        params: Params = self.make_params(instance)
        variables: CumulativeVars = self._make_vars(
            mdl, params, horizon, tighten_ranges=tighten_ranges
        )
        self._add_structural_constraints(mdl, params, variables)
        if link_job_completion:
            self._add_job_completion_link_constraints(mdl, params, variables)
        # self._add_inter_stage_structural_constraints(mdl, params, variables)
        self._define_objective(
            mdl,
            params,
            variables,
            minimize_sum_ci=minimize_sum_ci,
            minimize_makespan_plus_sum_other_stages=minimize_makespan_plus_sum_other_stages,
            horizon=horizon,
        )
        mdl.set_num_base_constraints()

        return mdl, params, variables

    def build_horizon_per_stage(
        self,
        instance: HybridFlowshopParameters,
        stage_2_mc_2_horizon: Mapping[str, Mapping[str, int]],
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
    ) -> tuple[CustomCpModel, Params, CumulativeVars]:
        mdl = CustomCpModel()
        params: Params = self.make_params(instance)
        variables: CumulativeVars = self._make_vars_horizon_per_stage(
            mdl, params, stage_2_mc_2_horizon, tighten_ranges=tighten_ranges
        )
        self._add_structural_constraints(mdl, params, variables, stage_2_mc_2_horizon)
        if link_job_completion:
            self._add_job_completion_link_constraints(mdl, params, variables)
        # self._add_inter_stage_structural_constraints(mdl, params, variables)
        self._define_objective(mdl, params, variables)
        mdl.set_num_base_constraints()

        return mdl, params, variables

    @staticmethod
    def make_params(instance: HybridFlowshopParameters) -> Params:
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
    def _compute_head(params: Params) -> dict[tuple[str, str], int]:
        j_i_2_head: dict[tuple[str, str], int] = {}
        for j in params.j_list:
            acc = 0
            for i in params.i_list:
                j_i_2_head[j, i] = acc
                acc += params.p[j, i]

        return j_i_2_head

    @staticmethod
    def _compute_tail(params: Params) -> dict[tuple[str, str], int]:
        j_i_2_tail: dict[tuple[str, str], int] = {}
        for j in params.j_list:
            acc = 0
            for i in reversed(params.i_list):
                j_i_2_tail[j, i] = acc
                acc += params.p[j, i]

        return j_i_2_tail

    @staticmethod
    def _make_vars(
        mdl: CustomCpModel, params: Params, horizon: int, tighten_ranges: bool = False
    ) -> CumulativeVars:
        op_start: dict[tuple[str, str], IntVar] = {}
        op_end: dict[tuple[str, str], IntVar] = {}
        op_intvl: dict[tuple[str, str], IntervalVar] = {}

        if tighten_ranges:
            j_i_2_head = BaseModelBuilder._compute_head(params)
            j_i_2_tail = BaseModelBuilder._compute_tail(params)
        else:
            j_i_2_head = {(j, i): 0 for j in params.j_list for i in params.i_list}
            j_i_2_tail = {(j, i): 0 for j in params.j_list for i in params.i_list}

        for j in params.j_list:
            for i in params.i_list:
                p = params.p[j, i]

                assert j_i_2_head[j, i] <= horizon - j_i_2_tail[j, i] - p
                assert j_i_2_head[j, i] + p <= horizon - j_i_2_tail[j, i]

                start_var = mdl.new_int_var(
                    j_i_2_head[j, i], horizon - j_i_2_tail[j, i] - p, f"start_{j}_{i}"
                )
                end_var = mdl.new_int_var(
                    j_i_2_head[j, i] + p, horizon - j_i_2_tail[j, i], f"end_{j}_{i}"
                )
                interval_var = mdl.new_interval_var(
                    start_var, p, end_var, f"interval_{j}_{i}"
                )

                op_start[(j, i)] = start_var
                op_end[(j, i)] = end_var
                op_intvl[(j, i)] = interval_var

        job_lb = max(sum(params.p[j, i] for i in params.i_list) for j in params.j_list)
        stage_load_lb = max(
            math.ceil(sum(params.p[j, i] for j in params.j_list) / len(params.M_of[i]))
            for i in params.i_list
        )
        makespan_lb = max(job_lb, stage_load_lb)
        makespan = mdl.new_int_var(makespan_lb, horizon, "makespan")

        return CumulativeVars(
            op_start=op_start,
            op_end=op_end,
            op_intvl=op_intvl,
            makespan=makespan,
        )

    @staticmethod
    def _make_vars_horizon_per_stage(
        mdl: CustomCpModel,
        params: Params,
        stage_2_mc_2_horizon: Mapping[str, Mapping[str, int]],
        tighten_ranges: bool = False,
    ) -> CumulativeVars:
        op_start: dict[tuple[str, str], IntVar] = {}
        op_end: dict[tuple[str, str], IntVar] = {}
        op_intvl: dict[tuple[str, str], IntervalVar] = {}

        stage_2_horizon: dict[str, int] = {
            stage: max(mc_2_horizon.values())
            for stage, mc_2_horizon in stage_2_mc_2_horizon.items()
        }

        # Compute total horizon for tightening calculation
        total_horizon = sum(
            params.p[j, i] for j in params.j_list for i in params.i_list
        )

        if tighten_ranges:
            j_i_2_head = BaseModelBuilder._compute_head(params)
            j_i_2_tail = BaseModelBuilder._compute_tail(params)
        else:
            j_i_2_head = {(j, i): 0 for j in params.j_list for i in params.i_list}
            j_i_2_tail = {(j, i): 0 for j in params.j_list for i in params.i_list}

        for j in params.j_list:
            for i in params.i_list:
                stage_horizon = stage_2_horizon[i]
                p = params.p[j, i]

                # Apply tightening with min() rule for upper bounds
                if tighten_ranges:
                    # start upper: min(total_horizon - tail - p, stage_horizon - p)
                    start_upper = min(
                        total_horizon - j_i_2_tail[j, i] - p,
                        stage_horizon - p,
                    )
                    # end upper: min(total_horizon - tail, stage_horizon)
                    end_upper = min(total_horizon - j_i_2_tail[j, i], stage_horizon)
                else:
                    start_upper = stage_horizon - p
                    end_upper = stage_horizon

                start_var = mdl.new_int_var(
                    j_i_2_head[j, i], start_upper, f"start_{j}_{i}"
                )
                end_var = mdl.new_int_var(
                    j_i_2_head[j, i] + p, end_upper, f"end_{j}_{i}"
                )
                interval_var = mdl.new_interval_var(
                    start_var, p, end_var, f"interval_{j}_{i}"
                )

                op_start[(j, i)] = start_var
                op_end[(j, i)] = end_var
                op_intvl[(j, i)] = interval_var

        makespan = mdl.new_int_var(0, stage_2_horizon[params.i_list[-1]], "makespan")

        return CumulativeVars(
            op_start=op_start,
            op_end=op_end,
            op_intvl=op_intvl,
            makespan=makespan,
        )

    @staticmethod
    def _add_precedence_constraints(
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
    ) -> None:
        """Add consecutive-stage precedence constraints for each job."""
        j_list = params.j_list
        i_list = params.i_list

        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                mdl.add(variables.op_end[j, i] <= variables.op_start[j, next_i])

    @staticmethod
    def _add_base_capacity_constraints(
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        stage_2_mc_2_horizon: Mapping[str, Mapping[str, int]] = {},
    ) -> None:
        """Add the default stage-capacity cumulative constraints."""
        i_list = params.i_list
        last_i = i_list[-1]

        stage_2_horizon: dict[str, int] = {
            stage: max(mc_2_horizon.values())
            for stage, mc_2_horizon in stage_2_mc_2_horizon.items()
        }

        for i in i_list:
            intervals = [variables.op_intvl[j, i] for j in params.j_list]
            demands = [1] * len(params.j_list)
            # Additional dummy intervals based on stage_2_mc_2_horizon
            if i in stage_2_mc_2_horizon and i != last_i:
                stage_horizon = stage_2_horizon[i]
                dummy_idx = 0
                for mc_horizon in stage_2_mc_2_horizon[i].values():
                    if mc_horizon < stage_horizon:
                        dummy_interval = mdl.new_interval_var(
                            mc_horizon,
                            stage_horizon - mc_horizon,
                            stage_horizon,
                            f"dummy_{i}_{dummy_idx}",
                        )
                        intervals.append(dummy_interval)
                        demands.append(1)
                        dummy_idx += 1

            capacity = len(params.M_of[i])
            mdl.add_cumulative(intervals, demands, capacity)

    @staticmethod
    def _add_structural_constraints(
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        stage_2_mc_2_horizon: Mapping[str, Mapping[str, int]] = {},
    ) -> None:
        """Add precedence and default stage-capacity constraints."""
        BaseModelBuilder._add_precedence_constraints(mdl, params, variables)
        BaseModelBuilder._add_base_capacity_constraints(
            mdl,
            params,
            variables,
            stage_2_mc_2_horizon,
        )

    @staticmethod
    def _add_job_completion_link_constraints(
        mdl: CustomCpModel, params: Params, variables: CumulativeVars
    ) -> None:
        j_i_2_tail = BaseModelBuilder._compute_tail(params)
        j_list = params.j_list
        i_list = params.i_list
        last_i = i_list[-1]

        for j in j_list:
            for i in i_list:
                mdl.add(
                    variables.op_end[j, i] + j_i_2_tail[j, i]
                    <= variables.op_end[j, last_i]
                )

    # @staticmethod
    # def _add_inter_stage_structural_constraints(
    #     mdl: CustomCpModel, params: Params, variables: CumulativeVars
    # ) -> None:
    #     for j in params.j_list:
    #         for a_idx, i in enumerate(params.i_list):
    #             transfer = 0
    #             for b_idx in range(a_idx + 1, len(params.i_list)):
    #                 k = params.i_list[b_idx]
    #                 if b_idx > a_idx + 1:
    #                     prev_stage = params.i_list[b_idx - 1]
    #                     transfer += params.p[j, prev_stage]
    #                 mdl.add(
    #                     variables.op_end[j, i] + transfer <= variables.op_start[j, k]
    #                 )

    @staticmethod
    def _define_objective(
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        minimize_sum_ci: bool = False,
        minimize_makespan_plus_sum_other_stages: bool = False,
        horizon: int = 0,
    ) -> None:
        """Define the objective function for the CP-SAT model.

        Supports three optimization objectives:

        1. **Makespan minimization** (default):
           Minimizes the maximum end time across all jobs at the last stage.
           ``minimize max_j(op_end[j, last_i])``

        2. **Sum of stage end times** (`minimize_sum_ci=True`):
           Minimizes the sum of end times for each stage, where Ci represents
           the end time of the last job at stage i.
           ``minimize sum(Ci for i in stages)``
           where ``Ci = max_j(op_end[j, i])``

        3. **Linear combination (weighted) objective** (`minimize_makespan_plus_sum_other_stages=True`):
           Minimizes a weighted sum that prioritizes makespan while also
           considering other stage end times. This is useful for balanced
           optimization where finishing all stages promptly is important.
           ``minimize (stage_cnt * makespan) + sum(Ci for i in stages[:-1])``

        Args:
            mdl (CustomCpModel): The CP-SAT model to which the objective will be added.
            params (Params): Parameters containing job and stage index sets.
            variables (CumulativeVars): Decision variables including operation end
                times and makespan.
            minimize_sum_ci (bool, optional): If True, minimizes the sum of all stage
                end times.
            Defaults to False.
            minimize_makespan_plus_sum_other_stages (bool, optional): If True,
                minimizes a weighted sum of makespan and other stage end times.
                Defaults to False.
            horizon (int, optional): The upper bound for stage end times. If 0 or
                negative, computed as the sum of all processing times. Defaults to 0.
        """
        # Alias for readability
        j_list = params.j_list
        i_list = params.i_list
        last_i = i_list[-1]

        if not minimize_sum_ci and not minimize_makespan_plus_sum_other_stages:
            # Makespan definition
            job_completion = [variables.op_end[j, last_i] for j in j_list]
            mdl.add_max_equality(variables.makespan, job_completion)
            # Set objective to minimize makespan
            mdl.minimize(variables.makespan)
        else:
            if horizon <= 0:
                horizon = sum(params.p[j, i] for j in j_list for i in i_list)
            # Define objective to minimize the sum of all stage's (last) end times
            stage_end_vars = []
            for i in i_list:
                Ci = mdl.new_int_var(0, horizon, f"Ci_{i}")
                mdl.add_max_equality(Ci, [variables.op_end[j, i] for j in j_list])
                stage_end_vars.append(Ci)

            if minimize_sum_ci:
                sum_Ci = mdl.new_int_var(0, len(i_list) * horizon, "stage_end_time_sum")
                mdl.add(sum_Ci == sum(stage_end_vars))
                mdl.minimize(sum_Ci)
            else:
                # minimize (stage_cnt * makespan) + (sum of all other stage's end times)
                stage_cnt = len(i_list)
                makespan = variables.makespan
                sum_other_stage_end_times = mdl.new_int_var(
                    0, (stage_cnt - 1) * horizon, "sum_other_stage_end_times"
                )
                other_stage_end_vars = [
                    stage_end_vars[i]
                    for i in range(len(stage_end_vars))
                    if i != len(stage_end_vars) - 1
                ]
                mdl.add(sum_other_stage_end_times == sum(other_stage_end_vars))
                mdl.minimize(stage_cnt * makespan + sum_other_stage_end_times)

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
        variables: OperationVars,
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
        variables: OperationVars,
        current_schedule: HybridFlowshopLiteSchedule,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        profile_fix_job_ids: set[JobIdType] | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
    ) -> None:
        """
        Add precedence constraints from a reference dispatch schedule.

        Adds constraints of the form ``op_end[j1, i] <= op_start[j2, i]`` to
        preserve ordering information observed in ``current_schedule``.

        Modes:
            - ``profile_fix_by_machine=True``: preserve machine-sequence order
            with a configurable stride per machine at each stage.
            - ``profile_fix_by_machine=False``: use stage-level start/end times
            to select successor candidates and add a bounded number of arcs.

        Args:
            mdl (CustomCpModel): Target CP-SAT model.
            params (Params): Index sets and processing parameters.
            variables (CumulativeVars): Decision variables used in constraints.
            current_schedule (HybridFlowshopLiteSchedule): Reference schedule
                providing start/end times and machine-level sequences.
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                sequence; otherwise apply stage-level time-based selection.
                Defaults to False.
            machine_precedence_stride (int, optional): Gap between predecessor and
                successor positions when ``profile_fix_by_machine=True``.
                - 1: adjacent precedence (default), e.g. 1->2->3->4->5
                - 2: every-other precedence, e.g. 1->3->5 and 2->4
                Ignored when ``profile_fix_by_machine=False``.
                Defaults to 1.
            profile_fix_job_ids (set[JobIdType] | None, optional): If provided,
                add precedence arcs only when both jobs are in this set. Defaults
                to None, which keeps all jobs in the reference profile eligible.
            stage_precedence_min_processing_time_diff (int | None, optional): When
                using stage-level time-based selection, add an arc only if the two
                jobs' processing times on the stage differ by at least this amount.
                Defaults to None, which keeps the current behavior.
            stage_precedence_min_processing_time_diff_ratio (float | None, optional):
                When using stage-level time-based selection, add an arc only if
                ``abs(p1 - p2) / max(p1, p2)`` is at least this value. Defaults to
                None, which keeps the current behavior.
        """
        if machine_precedence_stride < 1:
            raise ValueError("machine_precedence_stride must be >= 1")
        if (
            stage_precedence_min_processing_time_diff is not None
            and stage_precedence_min_processing_time_diff < 0
        ):
            raise ValueError("stage_precedence_min_processing_time_diff must be >= 0")
        if (
            stage_precedence_min_processing_time_diff_ratio is not None
            and stage_precedence_min_processing_time_diff_ratio < 0
        ):
            raise ValueError("stage_precedence_min_processing_time_diff_ratio must be >= 0")

        start_time_map = current_schedule.get_jik_2_start_time_map()
        end_time_map = current_schedule.get_jik_2_end_time_map()

        for i in params.i_list:
            if profile_fix_by_machine:
                for m in params.M_of[i]:
                    job_tuple_seq = current_schedule.get_job_sequence(i, m)
                    seq_len = len(job_tuple_seq)

                    for idx in range(seq_len - machine_precedence_stride):
                        j1 = job_tuple_seq[idx][2]
                        j2 = job_tuple_seq[idx + machine_precedence_stride][2]
                        if profile_fix_job_ids is not None and (
                            j1 not in profile_fix_job_ids
                            or j2 not in profile_fix_job_ids
                        ):
                            continue
                        BaseModelBuilder.add_fixed_operation_precedence_constraint(
                            mdl, params, variables, j1, j2, i
                        )
            else:
                current_j_set = {j for j, ip, _ in start_time_map if ip == i}
                if profile_fix_job_ids is not None:
                    current_j_set &= profile_fix_job_ids
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
                sorted_by_end = sorted(
                    current_j_list,
                    key=lambda j: (
                        j_2_end_time_map.get(j, float("inf")),
                        j_2_start_time_map.get(j, float("inf")),
                        stage_job_2_index_map.get(j, float("inf")),
                    ),
                )

                # List of jobs sorted by their 1) start times 2) end times 3) job index
                sorted_by_start = sorted(
                    current_j_list,
                    key=lambda j: (
                        j_2_start_time_map.get(j, float("inf")),
                        j_2_end_time_map.get(j, float("inf")),
                        stage_job_2_index_map.get(j, float("inf")),
                    ),
                )

                # Index-based selection of precedence arcs:
                # for each job j1 in end-time order
                for idx, j1 in enumerate(sorted_by_end):
                    j1_end_time = j_2_end_time_map.get(j1, float("inf"))
                    max_candidates = min(
                        len(params.M_of[i]), len(sorted_by_end) - idx - 1
                    )

                    # Find the position in the start-time sorted list where jobs start
                    # after j1 ends; use bisect_left to find the insertion point
                    # for j1_end_time in the sorted_by_start list
                    start_idx = bisect_left(
                        sorted_by_start,
                        j1_end_time,
                        key=lambda j: j_2_start_time_map.get(j, float("inf")),
                    )

                    # Add precedence constraints from j1 to a bounded number
                    # of successor candidates in start-time order
                    j2_list: list[JobIdType] = sorted_by_start[
                        start_idx : start_idx + max_candidates
                    ]
                    for j2 in j2_list:
                        p_diff = abs(params.p[j1, i] - params.p[j2, i])
                        if (
                            stage_precedence_min_processing_time_diff is not None
                            and p_diff < stage_precedence_min_processing_time_diff
                        ):
                            continue
                        p_max = max(params.p[j1, i], params.p[j2, i])
                        p_diff_ratio = p_diff / p_max if p_max > 0 else 0.0
                        if (
                            stage_precedence_min_processing_time_diff_ratio is not None
                            and p_diff_ratio
                            < stage_precedence_min_processing_time_diff_ratio
                        ):
                            continue
                        BaseModelBuilder.add_fixed_operation_precedence_constraint(
                            mdl, params, variables, j1, j2, i
                        )

    @staticmethod
    def add_start_time_freezed_operation_constraints(
        mdl: CustomCpModel,
        variables: OperationVars,
        start_time_map: dict[tuple[JobIdType, StageIdType, McIdType], int],
    ) -> None:
        for (j, i, k), s_time in start_time_map.items():
            mdl.add(variables.op_start[j, i] == s_time)

    # Hints

    @staticmethod
    def apply_start_hints_from_start_time_map(
        mdl: CustomCpModel,
        params: Params,
        variables: OperationVars,
        start_time_map: dict[tuple[JobIdType, StageIdType, McIdType], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        """Applies start time hints to the model from a given start time map.

        Args:
            start_time_map (dict[tuple[JobIdType, StageIdType, McIdType], int]): A mapping from (job_id, stage_id, machine_id) to start time.
        """
        for (j, i, _), s_time in start_time_map.items():
            if not ignore_integrity_check:
                assert j in params.j_list, f"Job {j} not in job list."
                assert i in params.i_list, f"Stage {i} not in stage list."
            mdl.add_hint(variables.op_start[j, i], s_time)

    @staticmethod
    def apply_end_hints_from_end_time_map(
        mdl: CustomCpModel,
        params: Params,
        variables: OperationVars,
        end_time_map: dict[tuple[JobIdType, StageIdType, McIdType], int],
        ignore_integrity_check: bool = True,
    ) -> None:
        """Applies end time hints to the model from a given end time map.

        Args:
            end_time_map (dict[tuple[JobIdType, StageIdType, McIdType], int]): A mapping from (job_id, stage_id, machine_id) to end time.
        """
        for (j, i, _), e_time in end_time_map.items():
            if not ignore_integrity_check:
                assert j in params.j_list, f"Job {j} not in job list."
                assert i in params.i_list, f"Stage {i} not in stage list."
            mdl.add_hint(variables.op_end[j, i], e_time)
