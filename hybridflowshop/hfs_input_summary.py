from dataclasses import dataclass


@dataclass
class HFSInputSummary:
    name: str
    num_jobs: int
    num_stages: int
    timelimit: float

    def comma_separated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return f"{self.name},{self.num_jobs},{self.num_stages},{self.timelimit}"
