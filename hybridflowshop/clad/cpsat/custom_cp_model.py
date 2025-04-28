from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from ortools.sat.cp_model_pb2 import ConstraintProto, CpSolverStatus
from ortools.sat.python.cp_model import (
    Constraint,
    CpModel,
    CpSolver,
    IntVar,
    LinearExpr,
)

from .utils import Utils


class CustomCpModel(CpModel):
    r"""A custom CpModel class that extends the ortools CpModel class."""

    solver: CpSolver
    """CpSolver object for solving the model."""

    num_base_constraints: int
    """Number of base constraints in the model."""

    # Added constraints

    added_constraints: list[Constraint]
    """List of added constraints."""
    idx_added_constraints: list[tuple[int, int]]
    """List of tuples representing the indices of the added constraints."""

    def __init__(self):
        super().__init__()
        self.num_base_constraints = 0
        self.added_constraints = []
        self.idx_added_constraints = []

    def solve_and_get_status(
        self, computational_time: float, n_threads: int
    ) -> tuple[CpSolverStatus, float, float, float]:
        """Solve the CP model with the specified computational time and number of threads.

        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.

        Returns:
            tuple[CpSolverStatus, float, float, float]: A tuple containing
            - the solver status,
            - elapsed time,
            - the upper bound of the objective function, and
            - the lower bound of the objective function.
        """  # noqa: E501
        self.init_solver(computational_time, n_threads)

        solver_status = self.solver.Solve(self)
        elapsed_time = self.solver.wall_time

        if Utils.found_feasible_solution(solver_status):
            ub = self.solver.objective_value
            lb = self.solver.best_objective_bound
        else:
            ub, lb = Utils.get_ub_and_lb_for_infeasible(self.is_maximize())

        return solver_status, elapsed_time, ub, lb

    def init_solver(self, computational_time: float, n_threads: int) -> None:
        """Initializes the solver with the given computational time and number of threads.

        Args:
            computational_time (float): The maximum computational time in seconds.
            n_threads (int): The number of threads to use for solving.
        """  # noqa: E501
        self.solver = CpSolver()
        self.solver.parameters.max_time_in_seconds = computational_time
        self.solver.parameters.num_workers = n_threads

    # variable functions

    def change_domain(self, var: IntVar, domain: list[int]) -> None:
        """Changes the domain of a variable.

        Args:
            var (IntVar)
            domain (list[int]): A list of two integers representing the new domain.
        """
        assert (
            len(domain) == 2
        ), f"Domain must be a list of two integers; {domain} given."

        var.Proto().domain[:] = domain

    # objective functions

    def is_maximize(self) -> bool:
        """
        Returns:
            bool: True if the objective is maximize, False if minimize.
        """
        return self._CpModel__model.objective.maximize

    # constraint functions

    def _get_constraints(self) -> RepeatedCompositeFieldContainer[ConstraintProto]:
        return self._CpModel__model.constraints

    def get_next_constr_idx(self) -> int:
        """Returns the index of the next constraint.

        Returns:
            int: The index of the next constraint.
        """
        return len(self._get_constraints())

    def freeze_base_constraints(self) -> None:
        """Sets the base number of constraints to the current number of constraints."""
        self.num_base_constraints = self.get_next_constr_idx()

    # methods to delete constraints

    def delete_constraints(self, idx_start: int, idx_end: int) -> None:
        del self._get_constraints()[idx_start:idx_end]

    def delete_added_constraints(self):
        """Deletes all constraints added after base model was built.

        Raises:
            ValueError: If no constraints were added after the base model was built.
        """

        if self.num_base_constraints == 0:
            raise ValueError("No base model constraints defined.")
        current_num_constraints = self.get_next_constr_idx()
        if current_num_constraints > self.num_base_constraints:
            self.delete_constraints(self.num_base_constraints, current_num_constraints)
