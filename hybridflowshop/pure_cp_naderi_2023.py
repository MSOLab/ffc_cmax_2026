from collections import defaultdict

from clad.cpsat import CustomCpModel, Utils
from hfs_summary import HFSOutputSummary
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL, CpSolver, IntervalVar, IntVar
from schore.hybridflowshop import HybridFlowShopProblem


class PureCPSolver:
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
    mdl: CustomCpModel
    """
    The CP model used for solving the hybrid flow shop problem.
    This model contains all the variables and constraints for the problem.
    """
    horizon: int
    """
    The horizon for the scheduling problem, which is the maximum time
    that any operation can start or end.
    This is used to define the domain of the start and end time variables.
    """

    summary: HFSOutputSummary

    def __init__(self, hfs_instance: HybridFlowShopProblem):
        self.hfs_instance = hfs_instance

    def solve(self, computational_time: float, n_threads: int):
        self.build_model()
        self.add_makespan_objective()
        self.summary = self.run_solver(computational_time, n_threads)

    def build_model(self):
        """
        Builds the CP model for the hybrid flow shop problem.
        This method creates variables for each operation in the jobs,
        including start and end times, presence indicators, and intervals.
        It also adds constraints to ensure that:
        - No two operations overlap on the same machine.
        - Each job is processed on exactly one machine at each stage.
        - Operations are processed in the correct order across stages.
        The model is built using the Google OR-Tools CP-SAT solver.
        The model is stored in the `mdl` attribute, and the variables are stored
        in the `var_op_start`, `var_op_end`, `var_op_is_present`, and `var_op_intvl`
        attributes.
        The horizon for the scheduling is set to 100000.
        The method uses the `CustomCpModel` class to create a custom CP model.
        The `CustomCpModel` class extends the `CpModel` class from OR-Tools
        and allows for additional functionalities such as changing variable domains
        and adding linear constraints.
        The method uses dictionaries to store the variables for each operation,
        indexed by job name, stage name, and machine number.
        """
        self.mdl = CustomCpModel()
        self.horizon = 100000
        mdl = self.mdl  # alias for readability

        # Parameters
        j_list = self.hfs_instance.get_job_id_list()
        i_list = self.hfs_instance.get_stage_id_list()
        M_of = self.hfs_instance.get_stage_2_machines_map()
        p = self.hfs_instance.p_manager.job_2_stage_2_value_map(j_list, i_list)

        # Variables
        self.var_op_start = defaultdict(lambda: defaultdict(dict))
        self.var_op_end = defaultdict(lambda: defaultdict(dict))
        self.var_op_is_present = defaultdict(lambda: defaultdict(dict))
        self.var_op_intvl = defaultdict(lambda: defaultdict(dict))
        for j in j_list:
            for i in i_list:
                for k in M_of[i]:
                    self.define_optional_interval_var(j, i, k, p[j][i])

        # Constraints: NoOverlap

        for i in i_list:
            for k in M_of[i]:
                mdl.add_no_overlap([self.var_op_intvl[j][i][k] for j in j_list])

        # Constraints: Alternative

        for j in j_list:
            for i in i_list:
                mdl.add(sum(self.var_op_is_present[j][i][k] for k in M_of[i]) == 1)

        # Constraints: EndBeforeStart

        consecutive_stage_pairs = []
        for stage_idx, i in enumerate(i_list[:-1]):
            next_i = i_list[stage_idx + 1]
            consecutive_stage_pairs.append((i, next_i))

        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                for k in M_of[i]:
                    for next_k in M_of[next_i]:
                        mdl.add(
                            self.var_op_end[j][i][k]
                            <= self.var_op_start[j][next_i][next_k]
                        )

    def define_optional_interval_var(
        self, j: str, i: str, k: str, processing_time: int
    ):
        mdl = self.mdl
        horizon = self.horizon

        suffix = f"_{j}_{i}_{k}"
        start_var = mdl.new_int_var(0, horizon, f"start{suffix}")
        end_var = mdl.new_int_var(0, horizon, f"end{suffix}")
        is_present_var = mdl.new_bool_var(f"is_present{suffix}")
        interval_var = mdl.new_optional_interval_var(
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

    def add_makespan_objective(self):
        """Adds the makespan objective to the CP model.
        The makespan is defined as the maximum end time of all operations
        across all jobs and stages. The objective is to minimize the makespan.
        """
        # Parameters
        j_list = self.hfs_instance.get_job_id_list()
        i_list = self.hfs_instance.get_stage_id_list()
        M_of = self.hfs_instance.get_stage_2_machines_map()

        # Variable
        makespan = self.mdl.new_int_var(0, self.horizon, "makespan")

        self.mdl.add_max_equality(
            makespan,
            [self.var_op_end[j][i][k] for j in j_list for i in i_list for k in M_of[i]],
        )
        self.mdl.minimize(makespan)

    def run_solver(self, computational_time: float, n_threads: int) -> HFSOutputSummary:
        """Runs the CP solver with the specified computational time and number of threads.
        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
        Returns:
            tuple[int, int | None, int | None]: A tuple containing the status of the solver,
            the objective value (if found), and the best objective bound (if found).
        """
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = computational_time
        solver.parameters.num_workers = n_threads
        solver_status = solver.Solve(self.mdl)
        elapsed_time = solver.wall_time
        if solver_status == OPTIMAL or solver_status == FEASIBLE:
            return HFSOutputSummary(
                Utils.get_status_string(solver_status),
                round(solver.objective_value),
                round(solver.best_objective_bound),
                elapsed_time,
            )
        else:
            return HFSOutputSummary(
                Utils.get_status_string(solver_status),
                None,
                None,
                elapsed_time,
            )
