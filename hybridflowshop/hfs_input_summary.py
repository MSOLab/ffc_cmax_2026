from dataclasses import dataclass

from .constants import INPUT_SUMMARY_HEADER


@dataclass
class HfsInputSummary:
    name: str
    job_count: int
    stage_count: int
    machines_per_stage: int
    timelimit: float

    def comma_separated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return f"{self.name},{self.job_count},{self.stage_count},{self.machines_per_stage},{self.timelimit}"

    @staticmethod
    def header() -> str:
        """Returns the header for the comma-separated values."""
        return INPUT_SUMMARY_HEADER
