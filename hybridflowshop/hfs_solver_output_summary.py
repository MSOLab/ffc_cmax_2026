from __future__ import annotations

from mbls import SolverOutputSummary


class HfsSolverOutputSummary(SolverOutputSummary):
    """
    Custom output summary for hybrid flow shop problems.

    Inherits from SolverOutputSummary and can be extended with additional methods or properties
    specific to hybrid flow shop problem solving.
    """

    is_init: bool
    """Indicates if this summary is for an initial solution."""

    def __init__(
        self,
        status,
        elapsed_time,
        objective_value=None,
        best_objective_bound=None,
        progress_log=None,
        is_init=False,
    ):
        super().__init__(
            status, elapsed_time, objective_value, best_objective_bound, progress_log
        )
        self.is_init = is_init

    @classmethod
    def from_other(
        cls, other: SolverOutputSummary, is_init: bool = False
    ) -> HfsSolverOutputSummary:
        """
        Create a new instance from another HfsSolverOutputSummary instance.

        Args:
            other (SolverOutputSummary): The instance to copy from.
            is_init (bool): Whether this summary is for an initial solution.

        Returns:
            HfsSolverOutputSummary: A new instance with the same properties as `other`.
        """
        return cls(
            status=other.status,
            elapsed_time=other.elapsed_time,
            objective_value=other.objective_value,
            best_objective_bound=other.best_objective_bound,
            progress_log=other.progress_log,
            is_init=is_init,
        )
