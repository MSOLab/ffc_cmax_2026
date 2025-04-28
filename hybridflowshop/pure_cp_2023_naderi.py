from clad.cpsat import CpModelWithOptionalInterval, Utils
from clad.solver_output_summary import SolverOutputSummary
from schore.hybridflowshop import HybridFlowShopProblem


class PureCP2023Naderi(CpModelWithOptionalInterval):

    # Indices & Parameters

    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    M_of: dict[str, list[str]]
    """$M_i$: machine index (k) list for stage i"""

    p: dict[str, dict[str, int]]
    """$P_{ji}$: processing time of job j at stage i"""

    # Result

    summary: SolverOutputSummary

    def __init__(self, hfs_instance: HybridFlowShopProblem, horizon: int):
        super().__init__(horizon)
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
            SolverOutputSummary
        """  # noqa: E501
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
        self.j_list = hfs_instance.get_job_id_list()
        self.i_list = hfs_instance.get_stage_id_list()
        self.M_of = hfs_instance.get_stage_2_machines_map()
        self.p = hfs_instance.p_manager.job_2_stage_2_value_map(
            self.j_list, self.i_list
        )

    # Variables

    def define_variables(self):
        # Define variables for each operation in each job
        for j in self.j_list:
            for i in self.i_list:
                for k in self.M_of[i]:
                    self.define_optional_interval_var(j, i, k, self.p[j][i])

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
