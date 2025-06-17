from dataclasses import dataclass


@dataclass
class HfsInputSummary:
    name: str
    job_count: int
    stage_count: int
    timelimit: float

    def comma_separated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return f"{self.name},{self.job_count},{self.stage_count},{self.timelimit}"

    @staticmethod
    def header() -> str:
        """Returns the header for the comma-separated values."""
        return "name,job_count,stage_count,timelimit"
