from typing import TextIO

from ..manager.processing_time_manager import JobStageProcessingTimeManager
from ..util.text_data_parser import TextDataParser


class HybridFlowShopProblem:
    """
    Represents a hybrid flow shop problem instance with multiple jobs and stages,
    where each stage may have multiple parallel machines.

    This class assumes all machines at a given stage are eligible for any operation at that stage.
    """  # noqa: E501

    num_jobs: int
    """Number of jobs in the problem instance."""
    num_stages: int
    """Number of stages in the problem instance."""
    machines_per_stage: list[int]
    """List of the number of parallel machines at each stage."""
    p_manager: JobStageProcessingTimeManager
    """Manager for processing times of jobs at each stage."""

    def __init__(
        self,
        num_jobs: int,
        num_stages: int,
        machines_per_stage: list[int],
        p_manager: JobStageProcessingTimeManager,
    ):
        self.num_jobs = num_jobs
        self.num_stages = num_stages
        self.machines_per_stage = machines_per_stage  # e.g., [2, 3, 2]
        self.p_manager = p_manager

    def __repr__(self):
        return (
            f"HybridFlowShopProblem(num_jobs={self.num_jobs},"
            f" num_stages={self.num_stages})"
        )

    @classmethod
    def from_pra_data(cls, stream: TextIO) -> "HybridFlowShopProblem":
        """
        Parse hybrid flow shop problem instance from a text stream in PRA-style format.

        Expected format:
            <num_jobs>
            <num_stages>
            <machines_per_stage>  # space-separated list
            <processing_time_row_0>
            <processing_time_row_1>
            ...
            <processing_time_row_n-1>

        Args:
            stream (TextIO): Input stream (e.g., open file or StringIO) containing instance data.

        Returns:
            HybridFlowShopProblem: Parsed problem instance.
        """  # noqa: E501
        num_jobs = TextDataParser.strip_a_typed_value(stream, int)
        num_stages = TextDataParser.strip_a_typed_value(stream, int)
        machines_per_stage = TextDataParser.strip_a_typed_list(stream, int)

        cls._validate_pra_structure(num_stages, machines_per_stage)

        processing_times = JobStageProcessingTimeManager.from_text_stream(
            stream, num_jobs, dtype=int
        )

        cls._validate_processing_times(num_stages, processing_times)

        return cls(
            num_jobs=num_jobs,
            num_stages=num_stages,
            machines_per_stage=machines_per_stage,
            p_manager=processing_times,
        )

    @staticmethod
    def _validate_pra_structure(num_stages: int, machines_per_stage: list[int]):
        if len(machines_per_stage) != num_stages:
            raise ValueError(
                f"Stage count mismatch; num_stages={num_stages};"
                f" by machines_per_stage={len(machines_per_stage)}"
            )

    @staticmethod
    def _validate_processing_times(
        num_stages: int, processing_times: JobStageProcessingTimeManager
    ):
        if processing_times.col_count() != num_stages:
            raise ValueError(
                f"Expected {num_stages} processing times per job,"
                f" got {processing_times.col_count()}."
            )
        if processing_times.df.isnull().values.any():
            raise ValueError("Null value exists in the processing time data.")

    def get_job_id_list(self) -> list[str]:
        """Generate a list of job IDs with zero-padded numbers."""
        num_digits = len(str(self.num_jobs - 1))
        return [f"j{str(j).zfill(num_digits)}" for j in range(self.num_jobs)]

    def get_stage_id_list(self) -> list[str]:
        """Generate a list of stage IDs with zero-padded numbers."""
        num_digits = len(str(self.num_stages - 1))
        return [f"i{str(s).zfill(num_digits)}" for s in range(self.num_stages)]

    def get_stage_2_machines_map(self) -> dict[str, list[str]]:
        """Generate a mapping from stage IDs to lists of machine IDs."""
        result: dict[str, list[str]] = {}
        for stage_idx, stage_id in enumerate(self.get_stage_id_list()):
            num_digits = len(str(self.machines_per_stage[stage_idx] - 1))
            machine_ids = [
                f"{stage_id}_{str(m).zfill(num_digits)}"
                for m in range(self.machines_per_stage[stage_idx])
            ]
            result[stage_id] = machine_ids
        return result
