from __future__ import annotations

from mbls.cpsat.cp_model_with_fixed_interval import CpModelWithFixedInterval
from ortools.sat.python.cp_model import IntVar


class CPCumulative(CpModelWithFixedInterval):
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
    """$I$: machine index (i) list"""

    p: dict[str, int]
    """$p_j$: processing time of job j"""

    r: dict[str, int] | None
    """$r_j$: release time of job j"""

    tr: dict[str, int] | None
    """${tr}_j$: transition time of job j"""

    # Objective
    obj_var: IntVar
    """Defines the makespan objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__(horizon)

    @classmethod
    def from_parameters(
        cls,
        j_list: list[str],
        i_list: list[str],
        p_dict: dict[str, int],
        horizon: int,
        r_dict: dict[str, int] | None = None,
        tr_dict: dict[str, int] | None = None,
    ) -> CPCumulative:
        result = cls(horizon)
        result.define_model(j_list, i_list, p_dict, r_dict, tr_dict)
        return result

    def define_model(
        self,
        j_list: list[str],
        i_list: list[str],
        p_dict: dict[str, int],
        r_dict: dict[str, int] | None,
        tr_dict: dict[str, int] | None,
    ) -> None:
        self.define_parameters(j_list, i_list, p_dict, r_dict, tr_dict)
        self.define_variables()
        self.define_makespan_objective()
        self.define_constraints()

    # Parameters

    def define_parameters(
        self,
        j_list: list[str],
        i_list: list[str],
        p_dict: dict[str, int],
        r_dict: dict[str, int] | None,
        tr_dict: dict[str, int] | None,
    ) -> None:
        self.j_list = j_list
        self.i_list = i_list
        self.p = p_dict
        self.r = r_dict
        self.tr = tr_dict

    # Variables

    def define_variables(self) -> None:
        # Interval variables
        for j in self.j_list:
            self.define_fixed_interval_var(j, self.p[j])

    # Objective

    def define_makespan_objective(self) -> None:
        # alias for readability
        j_list = self.j_list

        makespan = self.new_int_var(0, self.horizon, "makespan")
        if self.tr is None:
            self.add_max_equality(makespan, [self.var_op_end[j] for j in j_list])
        else:
            # If transition times are defined, we need to consider them in the makespan
            expr_list = []
            for j in j_list:
                if j in self.tr:
                    expr_list.append(self.var_op_end[j] + self.tr[j])
                else:
                    expr_list.append(self.var_op_end[j])
            self.add_max_equality(makespan, expr_list)

        self.minimize(makespan)
        self.obj_var = makespan

    def set_obj_lower_bound(self, lb: int) -> None:
        """Apply a lower bound to the objective variable.

        Args:
            lb (int): The lower bound to apply.
        """
        self.add(self.obj_var >= lb)

    # Constraints

    def define_constraints(self, impose_all_stage_capacity_constr: bool = True) -> None:
        # Capacity constraints
        self.add_capacity_constraint()

        # Release time constraints
        if self.r is not None:
            for j in self.j_list:
                self.add(self.var_op_start[j] >= self.r[j])

    def add_capacity_constraint(self) -> None:
        intervals = [self.var_op_intvl[j] for j in self.j_list]
        demands = [1] * len(self.j_list)
        capacity = len(self.i_list)
        self.add_cumulative(intervals, demands, capacity)
