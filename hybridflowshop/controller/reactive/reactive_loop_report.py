from dataclasses import dataclass
from typing import Any


@dataclass
class ReactiveLoopReportEntry:
    iter_count: int
    subroutine_name: str
    kwargs: dict[str, Any]
    time_start: float
    time_elapsed: float
    obj_value: float
    # Special fields
    is_optimal: bool
    is_improved: bool

    def get_row_dict(self) -> dict[str, Any]:
        row = {
            "iterCount": self.iter_count,
            "subroutineName": self.subroutine_name,
            "timeStart": self.time_start,
            "timeElapsed": self.time_elapsed,
            "objValue": self.obj_value,
            "isOptimal": self.is_optimal,
            "isImproved": self.is_improved,
        }
        if "rho" in self.kwargs:
            row["rho"] = self.kwargs["rho"]
        if "computational_time" in self.kwargs:
            row["timelimit"] = self.kwargs["computational_time"]
        return row

    @staticmethod
    def get_header() -> list[str]:
        return [
            "iterCount",
            "subroutineName",
            "rho",
            "timelimit",
            "timeStart",
            "timeElapsed",
            "objValue",
            "isOptimal",
            "isImproved",
        ]
