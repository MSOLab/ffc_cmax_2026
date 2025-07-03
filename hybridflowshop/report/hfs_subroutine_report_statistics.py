from typing import Any

from routix.report import SubroutineReportStatistics

from .hfs_subroutine_report import HfsCpsatSolverReport, HfsSubroutineReportT


class HfsSubroutineReportStatistics(SubroutineReportStatistics[HfsSubroutineReportT]):
    def get_init_summary(
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
        init = self.get_init_summary()
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

    def to_dict(self, is_maximize: bool = False) -> dict[str, Any]:
        """Return a dictionary representation of the statistics.

        Args:
            is_maximize (bool, optional): True if the objective is to maximize, False if to minimize.
                Defaults to False.

        Returns:
            dict[str, Any]: A dictionary representation of the statistics.
        """
        return_dict = super().to_dict(is_maximize=is_maximize)

        best = self.get_best_report(is_maximize=is_maximize)

        init_summary = self.get_init_summary(is_maximize=is_maximize)
        if init_summary:
            init_obj = init_summary.obj_value
        else:
            init_obj = best.obj_value if best else None

        # Remove "firstObj" from the return dictionary
        return_dict.pop("firstObj")
        return_dict["initObj"] = init_obj
        if type(best) is HfsCpsatSolverReport:
            return_dict["status"] = best.status

        return return_dict
