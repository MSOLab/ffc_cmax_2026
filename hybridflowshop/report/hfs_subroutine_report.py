from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from mbls.cpsat import CpsatSolverReport, CpsatStatus
from routix.report import SubroutineReport


@dataclass(frozen=True)
class HfsSubroutineReport(SubroutineReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution.

    - is_init: If this report corresponds to initialization, not improvement.
    """

    is_init: bool
    """True if this report corresponds to solution initialization, False otherwise."""

    def to_string_dict(self) -> dict[str, str]:
        """
        Return a dictionary with string representations of each field, suitable for CSV export.

        Returns:
            dict[str, str]: String representations of all report fields.
                - "elapsed_time"
                - "obj_value"
                - "obj_bound"
                - "obj_progress_log"
                - "is_init"
        """
        result = super().to_string_dict()
        result["is_init"] = str(self.is_init)
        return result


@dataclass(frozen=True)
class HfsCpsatSolverReport(HfsSubroutineReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution, specifically using CP-SAT solver.
    """

    status: CpsatStatus
    """Solver status as a CpsatStatus enum."""

    obj_value_records: list[tuple[float, float]]
    """
    List of (elapsed time, objective value)

    - Each entry records the state of the solver at a given time.
    - The list may not have the last entry.
    """

    obj_bound_records: list[tuple[float, float]]
    """
    List of (elapsed time, objective bound)

    - Each entry records the state of the solver at a given time.
    - The list may not have the last entry.
    """

    def to_string_dict(self) -> dict[str, str]:
        """
        Return a dictionary with string representations of each field, suitable for CSV export.

        - All values are converted to strings.
        - The status is exported as the standardized status string
          (e.g., "OPTIMAL"), not the enum representation.
        - Progress logs are wrapped in double quotes to ensure they are treated as strings in CSV.
          - If the log is empty, the string is empty.

        Returns:
            dict[str, str]: String representations of all report fields.
                - "elapsed_time"
                - "obj_value"
                - "obj_bound"
                - "status"
                - "obj_value_records"
                - "obj_bound_records"
        """
        d = super().to_string_dict()
        d["status"] = self.status.to_solver_status_enum().value
        d["obj_value_records"] = (
            f'"{self.obj_value_records}"' if self.obj_value_records else ""
        )
        d["obj_bound_records"] = (
            f'"{self.obj_bound_records}"' if self.obj_bound_records else ""
        )
        return d

    @classmethod
    def from_other(
        cls, other: CpsatSolverReport, is_init: bool = False
    ) -> HfsCpsatSolverReport:
        """
        Create an instance of HfsCpsatSolverReport from another CpsatSolverReport.

        Args:
            other (CpsatSolverReport): Another CpsatSolverReport instance to copy from.
            is_init (bool, optional): If this report corresponds to initialization, not improvement.
                Defaults to False.

        Returns:
            HfsCpsatSolverReport: A new instance of HfsCpsatSolverReport created from another CpsatSolverReport.
        """
        return cls(
            elapsed_time=other.elapsed_time,
            obj_value=other.obj_value,
            obj_bound=other.obj_bound,
            obj_value_records=other.obj_value_records,
            obj_bound_records=other.obj_bound_records,
            status=other.status,
            is_init=is_init,
        )

    @property
    def is_feasible(self) -> bool:
        """Check if the solution is feasible.

        Returns:
            bool: True if an objective value is available and the status indicates feasibility.
        """
        if self.obj_value is None:
            return False
        return self.status.is_feasible


HfsSubroutineReportT = TypeVar("HfsSubroutineReportT", bound=HfsSubroutineReport)
"""
Type variable for HfsSubroutineReport, allowing methods to specify
that they return or accept an instance of HfsSubroutineReport or its subclasses.
"""
