from dataclasses import dataclass


@dataclass
class HFSInputSummary:
    name: str
    num_jobs: int
    num_stages: int
    computational_time: float
    n_threads: int

    def comma_seperated_values(self) -> str:
        """Returns a string with comma-separated values of the summary."""
        return (
            f"{self.name},{self.num_jobs},{self.num_stages}"
            f",{self.computational_time},{self.n_threads}"
        )
