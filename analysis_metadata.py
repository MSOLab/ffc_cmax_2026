import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisMetadata:
    name: str
    result_dir_path_str: str
    reactive_loop_report_rel_path: str = "6-run_reactive_loop_report.csv"
    reactive_loop_report_required_cols: frozenset[str] = frozenset(
        {
            "iterCount",
            "rho",
            "timelimit",
            "subroutineName",
            "isImproved",
        }
    )

    def get_analysis_dir_path(self) -> Path:
        expanded = os.path.expandvars(self.result_dir_path_str)
        return Path(expanded).expanduser()

    def assert_reactive_loop_report_columns(self, columns: set[str]) -> None:
        missing = set(self.reactive_loop_report_required_cols) - columns
        if missing:
            raise ValueError(
                f"Missing columns in reactive loop report for {self.name}: {missing}"
            )
