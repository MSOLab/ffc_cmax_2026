"""
Defines the concrete implementation of the solution manager for the Hybrid Flowshop Scheduling problem.
"""

import math

from routix.solution_manager import SolutionManager

from .report import HfsSubroutineReport
from .scheduling.hybrid_flowshop_schedule import (
    HybridFlowshopSchedule,
)


class HfsSolutionManager(SolutionManager[HfsSubroutineReport, HybridFlowshopSchedule]):
    """
    A concrete solution manager for Hybrid Flowshop Scheduling.

    This class specializes the abstract manager by implementing the comparison
    logic specific to HFS, which is based on minimizing the makespan.
    """

    # --- Abstract Methods Implementation ---

    def _get_obj_value(self, solution: HybridFlowshopSchedule) -> float:
        return float(solution.makespan)

    def _a_is_better_obj_value(self, value_a: float, value_b: float | None) -> bool:
        if value_b is None:
            return True
        # False if close enough (considering floating point precision)
        if math.isclose(value_a, value_b, rel_tol=1e-9, abs_tol=1e-12):
            return False
        # A smaller makespan is better (minimization).
        return value_a < value_b

    def _a_is_better_obj_bound(self, bound_a: float, bound_b: float | None) -> bool:
        if bound_b is None:
            return True
        # False if close enough (considering floating point precision)
        if math.isclose(bound_a, bound_b, rel_tol=1e-9, abs_tol=1e-12):
            return False
        # For a minimization problem, a higher lower bound is better.
        return bound_a > bound_b
