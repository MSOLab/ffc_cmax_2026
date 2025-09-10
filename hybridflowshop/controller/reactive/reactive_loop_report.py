from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routix.concurrent_util import append_data_to_yaml, append_data_to_csv


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
            "iter_count": self.iter_count,
            "subroutine_name": self.subroutine_name,
            "time_start": self.time_start,
            "time_elapsed": self.time_elapsed,
            "obj_value": self.obj_value,
            "is_optimal": self.is_optimal,
            "is_improved": self.is_improved,
        }
        if "rho" in self.kwargs:
            row["rho"] = self.kwargs["rho"]
        if "computational_time" in self.kwargs:
            row["computational_time"] = self.kwargs["computational_time"]
        return row

    @staticmethod
    def get_header() -> list[str]:
        return [
            "iter_count",
            "subroutine_name",
            "time_start",
            "time_elapsed",
            "obj_value",
            "is_optimal",
            "is_improved",
            "rho",
            "computational_time",
        ]


def append_entry_to_yaml(
    yaml_path: Path, entry: ReactiveLoopReportEntry, encoding: str = "utf-8"
) -> None:
    row = entry.get_row_dict()
    append_data_to_yaml(yaml_path, row, encoding=encoding)


def append_entry_to_csv(
    csv_path: Path, entry: ReactiveLoopReportEntry, encoding: str = "utf-8-sig"
) -> None:
    row = entry.get_row_dict()
    header = ReactiveLoopReportEntry.get_header()
    append_data_to_csv(csv_path, row, header, encoding=encoding)
