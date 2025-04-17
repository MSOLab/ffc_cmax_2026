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
        """Builds the CP model for the hybrid flow shop problem.
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
        mdl = CustomCpModel()

        self.var_op_start = defaultdict(lambda: defaultdict(dict))
        self.var_op_end = defaultdict(lambda: defaultdict(dict))
        self.var_op_is_present = defaultdict(lambda: defaultdict(dict))
        self.var_op_intvl = defaultdict(lambda: defaultdict(dict))

        horizon = 100000
        for j in self.hfs_instance.jobs:
            for stage_idx, i in enumerate(self.hfs_instance.stage_names):
                for k in self.hfs_instance.stage_2_machines_map[i]:
                    suffix = f"_{j.name}_{i}_{k}"
                    start_var = mdl.new_int_var(0, horizon, f"start{suffix}")
                    end_var = mdl.new_int_var(0, horizon, f"end{suffix}")
                    is_present_var = mdl.new_bool_var(f"is_present{suffix}")
                    interval_var = mdl.new_optional_interval_var(
                        start_var,
                        j.operations[stage_idx].processing_time,
                        end_var,
                        is_present_var,
                        f"interval{suffix}",
                    )
                    self.var_op_start[j.name][i][k] = start_var
                    self.var_op_end[j.name][i][k] = end_var
                    self.var_op_is_present[j.name][i][k] = is_present_var
                    self.var_op_intvl[j.name][i][k] = interval_var

        for stage_idx, i in enumerate(self.hfs_instance.stage_names):
            for k in self.hfs_instance.stage_2_machines_map[i]:
                suffix = f"_{i}_{k}"
                mdl.add_no_overlap(
                    [
                        self.var_op_intvl[j.name][i][k]
                        for j in self.hfs_instance.jobs
                        if k in j.operations[stage_idx].eligible_mc_set
                    ]
                )

        for j in self.hfs_instance.jobs:
            for i in self.hfs_instance.stage_names:
                mdl.add(
                    sum(
                        self.var_op_is_present[j.name][i][k]
                        for k in self.hfs_instance.stage_2_machines_map[i]
                    )
                    == 1
                )

        consecutive_stage_pairs = [
            (self.hfs_instance.stage_names[i], self.hfs_instance.stage_names[i + 1])
            for i in range(self.hfs_instance.num_stages - 1)
        ]

        for j in self.hfs_instance.jobs:
            for i, next_i in consecutive_stage_pairs:
                for k in self.hfs_instance.stage_2_machines_map[i]:
                    for next_k in self.hfs_instance.stage_2_machines_map[next_i]:
                        mdl.add(
                            self.var_op_end[j.name][i][k]
                            <= self.var_op_start[j.name][next_i][next_k]
                        )

        self.mdl = mdl
        self.horizon = horizon

    def add_makespan_objective(self):
        """Adds the makespan objective to the CP model.
        The makespan is defined as the maximum end time of all operations
        across all jobs and stages. The objective is to minimize the makespan.
        """
        makespan = self.mdl.new_int_var(0, self.horizon, "makespan")
        self.mdl.add_max_equality(
            makespan,
            [
                self.var_op_end[j.name][i][k]
                for j in self.hfs_instance.jobs
                for i in self.hfs_instance.stage_names
                for k in self.hfs_instance.stage_2_machines_map[i]
            ],
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
