from ortools.sat.python import cp_model

from ..elapsed_timer import ElapsedTimer


class SolutionProgressLogger(cp_model.CpSolverSolutionCallback):
    _timer: ElapsedTimer
    _log: list[tuple[float, float, float]]
    """List of tuples containing (elapsed time, objective value, best bound)"""
    _print_on_solution_callback: bool

    def __init__(self, timer: ElapsedTimer, print_on_solution_callback: bool = False):
        super().__init__()
        self._timer = timer
        self._log = []
        self._print_on_solution_callback = print_on_solution_callback

    def on_solution_callback(self):
        elapsed = self._timer.get_elapsed_sec()
        objective = self.ObjectiveValue()
        best_bound = self.BestObjectiveBound()
        self._log.append((elapsed, objective, best_bound))
        if self._print_on_solution_callback:
            print(
                f"Time: {elapsed:.2f} sec"
                f", Objective: {objective}, Best Bound: {best_bound}"
            )

    def get_log(self):
        return self._log.copy()
