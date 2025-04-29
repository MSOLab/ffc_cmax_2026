class SolverStatus:
    UNKNOWN = "UNKNOWN"
    MODEL_INVALID = "MODEL_INVALID"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    OPTIMAL = "OPTIMAL"

    # @staticmethod
    # def all_statuses() -> set[str]:
    #     """Returns a set of all defined solver statuses."""
    #     return {
    #         SolverStatus.UNKNOWN,
    #         SolverStatus.MODEL_INVALID,
    #         SolverStatus.FEASIBLE,
    #         SolverStatus.INFEASIBLE,
    #         SolverStatus.OPTIMAL,
    #     }

    @staticmethod
    def found_feasible_solution(status: str) -> bool:
        """Checks if a feasible solution was found based on the status string."""
        return status in {SolverStatus.FEASIBLE, SolverStatus.OPTIMAL}

    # @staticmethod
    # def is_optimal_solution(status: str) -> bool:
    #     """Checks if the given status represents an optimal solution."""
    #     return status == SolverStatus.OPTIMAL

    # @staticmethod
    # def is_infeasible_solution(status: str) -> bool:
    #     """Checks if the given status represents an infeasible solution."""
    #     return status == SolverStatus.INFEASIBLE
