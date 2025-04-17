from ortools.sat.python.cp_model import (
    FEASIBLE,
    INFEASIBLE,
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

    @staticmethod
    def get_status_string(status: int) -> str:
        """Returns the status string corresponding to the given status code."""
        return Utils.status_dict.get(status, "UNKNOWN")
