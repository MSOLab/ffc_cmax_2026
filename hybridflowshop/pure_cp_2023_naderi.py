from collections import defaultdict

from clad.cpsat import CustomCpModel, Utils
from clad.solver_output_summary import SolverOutputSummary
from ortools.sat.python.cp_model import IntervalVar, IntVar
from schore.hybridflowshop import HybridFlowShopProblem


class PureCP2023Naderi(CustomCpModel):

    # Parameters

    j_list: list[str]
    """$J$: job index list"""

    i_list: list[str]
    """$I$: stage index list"""

    M_of: dict[str, list[str]]
    """$M_i$: machine index list for stage i"""

    p: dict[str, dict[str, int]]
    """$P_{ji}$: processing time of job j at stage i"""

    # Variables

    var_op_start: dict[str, dict[str, dict[str, IntVar]]]
    """
    Dictionary to store start time variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """
    var_op_end: dict[str, dict[str, dict[str, IntVar]]]
    """
    Dictionary to store end time variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """
    var_op_is_present: dict[str, dict[str, dict[str, IntVar]]]
    """
    Dictionary to store presence indicator variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """
    var_op_intvl: dict[str, dict[str, dict[str, IntervalVar]]]
    """
    Dictionary to store interval variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """
    horizon: int
    """
    The horizon for the scheduling problem, which is the maximum time
    that any operation can start or end.
    This is used to define the domain of the start and end time variables.
    """

    # Objective function

    obj_func: IntVar
    """The objective function for the scheduling problem."""

    summary: SolverOutputSummary

    def __init__(self, hfs_instance: HybridFlowShopProblem):
        super().__init__()
        self.define_model(hfs_instance)

    def define_model(self, hfs_instance: HybridFlowShopProblem):
        self.define_parameters(hfs_instance)
        self.define_variables()
        self.define_makespan_objective()
        self.define_constraints()

    def solve(self, computational_time: float, n_threads: int):
        self.summary = self.run_and_summarize(computational_time, n_threads)

    def run_and_summarize(
        self, computational_time: float, n_threads: int
    ) -> SolverOutputSummary:
        """Solve the CP model with the specified computational time and number of threads.

        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.

        Returns:
            SolverOutputSummary: _description_
        """
        solver_status, elapsed_time, ub, lb = super().solve_and_get_status(
            computational_time, n_threads
        )

        return SolverOutputSummary(
            Utils.get_status_string(solver_status),
            ub,
            lb,
            elapsed_time,
        )

    # Parameters

    def define_parameters(self, hfs_instance: HybridFlowShopProblem):
        self.horizon = 100000
        self.j_list = hfs_instance.get_job_id_list()
        self.i_list = hfs_instance.get_stage_id_list()
        self.M_of = hfs_instance.get_stage_2_machines_map()
        self.p = hfs_instance.p_manager.job_2_stage_2_value_map(
            self.j_list, self.i_list
        )

    # Variables

    def define_variables(self):
        # Initialize dictionaries to store variables
        self.var_op_start = defaultdict(lambda: defaultdict(dict))
        self.var_op_end = defaultdict(lambda: defaultdict(dict))
        self.var_op_is_present = defaultdict(lambda: defaultdict(dict))
        self.var_op_intvl = defaultdict(lambda: defaultdict(dict))

        # Define variables for each operation in each job
        for j in self.j_list:
            for i in self.i_list:
                for k in self.M_of[i]:
                    self.define_optional_interval_var(j, i, k, self.p[j][i])

    def define_optional_interval_var(
        self, j: str, i: str, k: str, processing_time: int
    ):
        suffix = f"_{j}_{i}_{k}"
        start_var = self.new_int_var(0, self.horizon, f"start{suffix}")
        end_var = self.new_int_var(0, self.horizon, f"end{suffix}")
        is_present_var = self.new_bool_var(f"is_present{suffix}")
        interval_var = self.new_optional_interval_var(
            start_var,
            processing_time,
            end_var,
            is_present_var,
            f"interval{suffix}",
        )
        self.var_op_start[j][i][k] = start_var
        self.var_op_end[j][i][k] = end_var
        self.var_op_is_present[j][i][k] = is_present_var
        self.var_op_intvl[j][i][k] = interval_var

    # Objective

    def define_makespan_objective(self):
        # alias for readability
        j_list = self.j_list
        i_list = self.i_list
        M_of = self.M_of

        makespan = self.new_int_var(0, self.horizon, "makespan")
        self.add_max_equality(
            makespan,
            [self.var_op_end[j][i][k] for j in j_list for i in i_list for k in M_of[i]],
        )

        self.obj_func = makespan
        self.minimize(self.obj_func)

    # Constraints

    def define_constraints(self):
        # Alias for readability
        j_list = self.j_list
        i_list = self.i_list
        M_of = self.M_of

        # Constraints: NoOverlap

        for i in i_list:
            for k in M_of[i]:
                self.add_no_overlap([self.var_op_intvl[j][i][k] for j in j_list])

        # Constraints: Alternative

        for j in j_list:
            for i in i_list:
                self.add(sum(self.var_op_is_present[j][i][k] for k in M_of[i]) == 1)

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
                            self.var_op_end[j][i][k]
                            <= self.var_op_start[j][next_i][next_k]
                        )
