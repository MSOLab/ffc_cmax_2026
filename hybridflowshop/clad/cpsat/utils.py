from ortools.sat.python.cp_model import (
    FEASIBLE,
    INFEASIBLE,
    INT_MAX,
    INT_MIN,
    MODEL_INVALID,
    OPTIMAL,
    UNKNOWN,
)


class Utils:
    status_dict = {
        UNKNOWN: "UNKNOWN",
        MODEL_INVALID: "MODEL_INVALID",
        FEASIBLE: "FEASIBLE",
        INFEASIBLE: "INFEASIBLE",
        OPTIMAL: "OPTIMAL",
    }
    """Map: ortools.sat.python.cp_model status codes -> string"""

    feasible_status_set = frozenset({FEASIBLE, OPTIMAL})
    """Set of status codes indicating a feasible solution was found."""

    @staticmethod
    def get_status_string(status: int) -> str:
        """Returns the status string corresponding to the given status code."""
        return Utils.status_dict.get(status, "UNKNOWN")

    @staticmethod
    def found_feasible_solution(status: int) -> bool:
        """Checks if a feasible solution was found based on the status code."""
        return status in Utils.feasible_status_set

    @staticmethod
    def get_ub_and_lb_for_infeasible_maximize() -> tuple[int, int]:
        """Returns the upper and lower bounds for maximization problems."""
        return INT_MIN, INT_MIN

    @staticmethod
    def get_ub_and_lb_for_infeasible_minimize() -> tuple[int, int]:
        """Returns the upper and lower bounds for minimization problems."""
        return INT_MAX, INT_MAX

    @staticmethod
    def get_ub_and_lb_for_infeasible(is_maximize: bool) -> tuple[int, int]:
        """Returns the upper and lower bounds for infeasible problems."""
        if is_maximize:
            return Utils.get_ub_and_lb_for_infeasible_maximize()
        else:
            return Utils.get_ub_and_lb_for_infeasible_minimize()
