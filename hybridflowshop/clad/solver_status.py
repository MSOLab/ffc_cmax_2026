class SolverStatus:
    MODEL_INVALID = "MODEL_INVALID"
    INFEASIBLE = "INFEASIBLE"
    FEASIBLE = "FEASIBLE"
    OPTIMAL = "OPTIMAL"
    UNKNOWN = "UNKNOWN"

    @staticmethod
    def all_statuses() -> set[str]:
        """Returns a set of all defined solver statuses."""
        return {
            SolverStatus.MODEL_INVALID,
            SolverStatus.INFEASIBLE,
            SolverStatus.FEASIBLE,
            SolverStatus.OPTIMAL,
            SolverStatus.UNKNOWN,
        }

    @staticmethod
    def is_model_invalid(status: str) -> bool:
        """Checks if the given status indicates an invalid model."""
        return status == SolverStatus.MODEL_INVALID

    @staticmethod
    def is_infeasible(status: str) -> bool:
        """Checks if the given status represents an infeasible solution."""
        return status == SolverStatus.INFEASIBLE

    @staticmethod
    def found_feasible_solution(status: str) -> bool:
        """Checks if a feasible solution was found based on the status string."""
        return status in {SolverStatus.FEASIBLE, SolverStatus.OPTIMAL}

    # @staticmethod
    # def is_optimal_solution(status: str) -> bool:
    #     """Checks if the given status represents an optimal solution."""
    #     return status == SolverStatus.OPTIMAL

    @staticmethod
    def is_unknown(status: str) -> bool:
        """Checks if the given status is unknown."""
        return status == SolverStatus.UNKNOWN

    @staticmethod
    def raise_if_not_feasible(status: str) -> None:
        """Raises an exception if the status indicates infeasibility."""
        if status not in SolverStatus.all_statuses():
            raise ValueError(f"Unknown status: {status}")
        elif SolverStatus.is_model_invalid(status):
            raise ValueError("The model is invalid.")
        elif SolverStatus.is_infeasible(status):
            raise ValueError("The problem is infeasible.")
        elif SolverStatus.is_unknown(status):
            raise ValueError("The status is unknown.")
