from __future__ import annotations

from dataclasses import dataclass, KW_ONLY
from typing import Sequence, TypeVar

from mbls.cpsat import CpsatSolverReport, CpsatStatus
from routix.report import SubroutineReport


@dataclass(frozen=True)
class HfsSubroutineReport(SubroutineReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution.

    - is_init: If this report corresponds to initialization, not improvement.
    - subroutine_name: Name of the subroutine that produced this report.
    - call_context: Unique call-site identifier (e.g. "4-pw_cp") to disambiguate
      repeated invocations of the same subroutine name within a flow.
    - progress_obj_value_records: Normalized (elapsed_time, obj_value) pairs
      representing intermediate progress. For subroutines with dedicated result
      objects (PwCpResult, NehCpResult, PrTsResult), these come from
      sub_obj_store.obj_value_series.items(). For CP-SAT subroutines, they
      mirror obj_value_records. For subroutines without a dedicated store but
      that call add_obj_value_log(), a singleton tuple is synthesized.
    - progress_time_basis: "local" if timestamps are subroutine-local (must be
      shifted by start_time in post-processing), "global" if timestamps already
      live on the controller/global axis (no shift needed).
    """

    is_init: bool
    """True if this report corresponds to solution initialization, False otherwise."""

    _: KW_ONLY

    subroutine_name: str = ""
    """Name of the subroutine that produced this report."""

    call_context: str = ""
    """Unique call-site identifier to disambiguate repeated subroutine invocations."""

    progress_obj_value_records: tuple[tuple[float, float], ...] = ()
    """Normalized (elapsed_time, obj_value) pairs representing intermediate progress."""

    progress_time_basis: str = "local"
    """'local' if timestamps need start_time shifting, 'global' if already on controller axis."""

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
                - "subroutine_name"
                - "call_context"
                - "progress_obj_value_records"
                - "progress_time_basis"
        """
        result = super().to_string_dict()
        result["is_init"] = str(self.is_init)
        result["subroutine_name"] = self.subroutine_name
        result["call_context"] = self.call_context
        result["progress_obj_value_records"] = (
            f'"{self.progress_obj_value_records}"'
            if self.progress_obj_value_records
            else ""
        )
        result["progress_time_basis"] = self.progress_time_basis
        return result


@dataclass(frozen=True)
class HfsCpsatSolverReport(HfsSubroutineReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution, specifically using CP-SAT solver.
    """

    _: KW_ONLY

    status: CpsatStatus
    """Solver status as a CpsatStatus enum."""

    obj_value_records: Sequence[tuple[float, float]] = ()
    """
    List of (elapsed time, objective value)

    - Each entry records the state of the solver at a given time.
    - The sequence may not have the last entry.
    """

    obj_bound_records: Sequence[tuple[float, float]] = ()
    """
    List of (elapsed time, objective bound)

    - Each entry records the state of the solver at a given time.
    - The sequence may not have the last entry.
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
        obj_value_records = other.obj_value_records
        return cls(
            elapsed_time=other.elapsed_time,
            obj_value=other.obj_value,
            obj_bound=other.obj_bound,
            obj_value_records=obj_value_records,
            obj_bound_records=other.obj_bound_records,
            status=other.status,
            is_init=is_init,
            progress_obj_value_records=tuple(obj_value_records),
            progress_time_basis="local",
        )

    def copy(self, **kwargs) -> HfsCpsatSolverReport:
        """Create a copy of the report, optionally updating fields with new values.

        Args:
            **kwargs: Keyword arguments to update specific fields.

        Returns:
            HfsCpsatSolverReport: A new instance of HfsCpsatSolverReport with copied or updated fields.
        """
        return HfsCpsatSolverReport(
            elapsed_time=kwargs.get("elapsed_time", self.elapsed_time),
            obj_value=kwargs.get("obj_value", self.obj_value),
            obj_bound=kwargs.get("obj_bound", self.obj_bound),
            obj_value_records=kwargs.get("obj_value_records", self.obj_value_records),
            obj_bound_records=kwargs.get("obj_bound_records", self.obj_bound_records),
            status=kwargs.get("status", self.status),
            is_init=kwargs.get("is_init", self.is_init),
            subroutine_name=kwargs.get("subroutine_name", self.subroutine_name),
            call_context=kwargs.get("call_context", self.call_context),
            progress_obj_value_records=kwargs.get(
                "progress_obj_value_records", self.progress_obj_value_records
            ),
            progress_time_basis=kwargs.get(
                "progress_time_basis", self.progress_time_basis
            ),
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
