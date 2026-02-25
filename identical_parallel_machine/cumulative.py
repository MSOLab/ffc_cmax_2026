from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntervalVar, IntVar

from .params import ParallelMcParams


@dataclass
class ParallelMcVars:
    op_start: dict[str, IntVar]
    """j -> start time variables for each operation in a job."""

    op_end: dict[str, IntVar]
    """j -> end time variables for each operation in a job."""

    op_intvl: dict[str, IntervalVar]
    """j -> interval variables for each operation in a job."""

    makespan: IntVar
    """Makespan variable"""


class ParallelMcModelBuilder:
    @staticmethod
    def build(
        j_list: Sequence[str],
        i_list: Sequence[str],
        p: Mapping[str, int],
        r: Mapping[str, int] | None,
        tr: Mapping[str, int] | None,
        horizon: int,
    ) -> tuple[CustomCpModel, ParallelMcParams, ParallelMcVars]:
        mdl = CustomCpModel()
        params: ParallelMcParams = ParallelMcModelBuilder._make_params(
            j_list, i_list, p, r, tr
        )
        variables: ParallelMcVars = ParallelMcModelBuilder._make_vars(
            mdl, params, horizon
        )
        ParallelMcModelBuilder._add_structural_constraints(mdl, params, variables)
        ParallelMcModelBuilder._define_objective(mdl, params, variables)
        mdl.set_num_base_constraints()

        return mdl, params, variables

    @staticmethod
    def _make_params(
        j_list: Sequence[str],
        i_list: Sequence[str],
        p: Mapping[str, int],
        r: Mapping[str, int] | None,
        tr: Mapping[str, int] | None,
    ) -> ParallelMcParams:
        return ParallelMcParams(j_list, i_list, p, r, tr)

    @staticmethod
    def _make_vars(
        mdl: CustomCpModel, params: ParallelMcParams, horizon: int
    ) -> ParallelMcVars:
        op_start: dict[str, IntVar] = {}
        op_end: dict[str, IntVar] = {}
        op_intvl: dict[str, IntervalVar] = {}

        for j in params.j_list:
            start_var = mdl.new_int_var(0, horizon, f"start_{j}")
            end_var = mdl.new_int_var(0, horizon, f"end_{j}")
            interval_var = mdl.new_interval_var(
                start_var,
                params.p[j],
                end_var,
                f"interval_{j}",
            )

            op_start[j] = start_var
            op_end[j] = end_var
            op_intvl[j] = interval_var

        makespan = mdl.new_int_var(0, horizon, "makespan")

        return ParallelMcVars(op_start, op_end, op_intvl, makespan)

    @staticmethod
    def _add_structural_constraints(
        mdl: CustomCpModel, params: ParallelMcParams, variables: ParallelMcVars
    ) -> None:
        intervals = [variables.op_intvl[j] for j in params.j_list]
        demands = [1] * len(params.j_list)
        capacity = len(params.i_list)
        mdl.add_cumulative(intervals, demands, capacity)

        # Release time constraints
        if params.r is not None:
            for j in params.j_list:
                mdl.add(variables.op_start[j] >= params.r[j])

    @staticmethod
    def _define_objective(
        mdl: CustomCpModel, params: ParallelMcParams, variables: ParallelMcVars
    ) -> None:
        # Alias for readability
        j_list = params.j_list
        if params.tr is None:
            mdl.add_max_equality(
                variables.makespan, [variables.op_end[j] for j in j_list]
            )
        else:
            # If transition times are defined, we need to consider them in the makespan
            expr_list = []
            for j in j_list:
                if j in params.tr:
                    expr_list.append(variables.op_end[j] + params.tr[j])
                else:
                    expr_list.append(variables.op_end[j])
            mdl.add_max_equality(variables.makespan, expr_list)

        # Set objective to minimize makespan
        mdl.minimize(variables.makespan)
