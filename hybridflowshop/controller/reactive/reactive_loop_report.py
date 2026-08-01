from dataclasses import dataclass
from typing import Any


@dataclass
class ReactiveLoopReportEntry:
    iter_count: int
    subroutine_name: str
    kwargs: dict[str, Any]
    time_start: float
    time_elapsed: float
    prev_obj_value: float
    obj_value: float
    # Special fields
    timelimit_reached: bool
    is_optimal: bool
    is_improved: bool

    def get_row_dict(self) -> dict[str, Any]:
        row = {
            "iterCount": self.iter_count,
            "subroutineName": self.subroutine_name,
            "timeStart": self.time_start,
            "timeElapsed": self.time_elapsed,
            "prevObjValue": self.prev_obj_value,
            "objValue": self.obj_value,
            "timelimitReached": self.timelimit_reached,
            "isOptimal": self.is_optimal,
            "isImproved": self.is_improved,
        }
        if "rho" in self.kwargs:
            row["rho"] = self.kwargs["rho"]
        if "job_count" in self.kwargs:
            row["job_count"] = self.kwargs["job_count"]
        if "radius" in self.kwargs:
            row["radius"] = self.kwargs["radius"]
        if "bottleneck_band_radius" in self.kwargs:
            row["bottleneck_band_radius"] = self.kwargs["bottleneck_band_radius"]
        if "computational_time" in self.kwargs:
            row["timelimit"] = self.kwargs["computational_time"]
        if "tl_nc_multiplier" in self.kwargs:
            row["tl_nc_multiplier"] = self.kwargs["tl_nc_multiplier"]
        return row

    @staticmethod
    def get_header() -> list[str]:
        return [
            "iterCount",
            "subroutineName",
            "rho",
            "job_count",
            "radius",
            "bottleneck_band_radius",
            "timelimit",
            "tl_nc_multiplier",
            "timeStart",
            "timeElapsed",
            "prevObjValue",
            "objValue",
            "timelimitReached",
            "isOptimal",
            "isImproved",
        ]
