from typing import Any

from routix.report import SubroutineReportStatistics

from .hfs_subroutine_report import HfsCpsatSolverReport, HfsSubroutineReportT


class HfsSubroutineReportStatistics(SubroutineReportStatistics[HfsSubroutineReportT]):
    def get_init_obj_value_report(
        self, is_maximize: bool = False
    ) -> HfsSubroutineReportT | None:
        # Find valid reports
        valid_reports = [r for r in self.reports if r.obj_value is not None]
        # If no valid reports, return None
        if not valid_reports:
            return None

        # Find reports with is_init=True
        init_reports = [r for r in valid_reports if r.is_init]
        if not init_reports:
            # If no initial reports are found, return the first valid run
            return valid_reports[0]

        # Find the best run
        if is_maximize:
            return max(
                init_reports,
                key=lambda r: r.obj_value if r.obj_value is not None else float("-inf"),
            )
        return min(
            init_reports,
            key=lambda r: r.obj_value if r.obj_value is not None else float("inf"),
        )

    def get_improvement_ratio(self, is_maximize: bool = False) -> float | None:
        init = self.get_init_obj_value_report()
        best = self.get_best_report(is_maximize=is_maximize)

        if not (
            init and best and init.obj_value is not None and best.obj_value is not None
        ):
            return None
        if init.obj_value == 0:
            return None

        if is_maximize:
            return (best.obj_value - init.obj_value) / init.obj_value
        return (init.obj_value - best.obj_value) / init.obj_value

    def get_init_obj_bound_report(
        self, is_maximize: bool = False
    ) -> HfsSubroutineReportT | None:
        # Find valid reports
        valid_reports = [r for r in self.reports if r.obj_bound is not None]
        # If no valid reports, return None
        if not valid_reports:
            return None

        # Find reports with is_init=True
        init_reports = [r for r in valid_reports if r.is_init]
        if not init_reports:
            # If no initial reports are found, return the first valid run
            return valid_reports[0]

        # Find the best run
        if is_maximize:
            return min(
                init_reports,
                key=lambda r: r.obj_bound if r.obj_bound is not None else float("inf"),
            )
        return max(
            init_reports,
            key=lambda r: r.obj_bound if r.obj_bound is not None else float("-inf"),
        )

    def to_dict(self, is_maximize: bool = False) -> dict[str, Any]:
        """Return a dictionary representation of the statistics.

        Args:
            is_maximize (bool, optional): True if the objective is to maximize, False if to minimize.
                Defaults to False.

        Returns:
            dict[str, Any]: A dictionary representation of the statistics.
        """
        return_dict = super().to_dict(is_maximize=is_maximize)

        best_obj_value_report = self.get_best_report(is_maximize=is_maximize)

        init_obj_value_report = self.get_init_obj_value_report(is_maximize=is_maximize)
        if init_obj_value_report:
            init_obj_value = init_obj_value_report.obj_value
        else:
            init_obj_value = (
                best_obj_value_report.obj_value if best_obj_value_report else None
            )

        best_obj_bound_report = self.get_best_bound_report(is_maximize=is_maximize)

        init_obj_bound_report = self.get_init_obj_bound_report(is_maximize=is_maximize)
        if init_obj_bound_report:
            init_obj_bound = init_obj_bound_report.obj_bound
        else:
            init_obj_bound = (
                best_obj_bound_report.obj_bound if best_obj_bound_report else None
            )

        # Remove "firstObj" from the return dictionary
        return_dict.pop("firstObj")
        return_dict["initObj"] = init_obj_value
        if init_obj_bound is not None:
            # Remove "firstBound" from the return dictionary
            return_dict.pop("firstBound", None)
            return_dict["initBound"] = init_obj_bound
        if (
            best_obj_bound_report is not None
            and best_obj_bound_report.obj_bound is not None
        ):
            # Override bestBound with the best obj_bound report, not the best obj_value report
            return_dict["bestBound"] = best_obj_bound_report.obj_bound
        if type(best_obj_bound_report) is HfsCpsatSolverReport:
            return_dict["status"] = (
                best_obj_bound_report.status.to_solver_status_enum().value
            )

        return return_dict

    # TODO: move methods to the base class

    @property
    def min_obj_bound_report(self) -> HfsSubroutineReportT | None:
        """
        Returns:
            HfsSubroutineReportT | None: The report with the minimum objective bound.
                - If no valid reports exist, returns None.
                - Tie-breaker: prefers later reports in case of equal objective bounds.
        """
        fea = [r for r in self.reports if r.obj_bound is not None]
        if not fea:
            return None
        idx, _ = min(enumerate(fea), key=lambda t: (t[1].obj_bound, -t[0]))
        return fea[idx]

    @property
    def max_obj_bound_report(self) -> HfsSubroutineReportT | None:
        """
        Returns:
            HfsSubroutineReportT | None: The report with the maximum objective bound.
                - If no valid reports exist, returns None.
                - Tie-breaker: prefers later reports in case of equal objective bounds.
        """
        fea = [r for r in self.reports if r.obj_bound is not None]
        if not fea:
            return None
        idx, _ = max(enumerate(fea), key=lambda t: (t[1].obj_bound, t[0]))
        return fea[idx]

    def get_best_bound_report(
        self, is_maximize: bool = False
    ) -> HfsSubroutineReportT | None:
        """Get the best report based on the objective bound.

        Args:
            is_maximize (bool): True if the objective is to maximize, False if to minimize.

        Returns:
            HfsSubroutineReportT | None: The best report based on objective bound.
                - If is_maximize is True, returns the report with the maximum objective bound.
                - If is_maximize is False, returns the report with the minimum objective bound.
                - If no valid reports exist, returns None.
        """
        return self.min_obj_bound_report if is_maximize else self.max_obj_bound_report
