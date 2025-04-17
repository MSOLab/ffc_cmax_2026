from typing import TextIO

from .fixed_operation import HybridFlowShopFixedOperation
from .job import HybridFlowShopJob


class HybridFlowShopProblem:
    """
    Represents a hybrid flow shop problem instance with multiple jobs and stages,
    where each stage may have multiple parallel machines.

    This class assumes all machines at a given stage are eligible for any operation at that stage.
    """  # noqa: E501

    stage_names: list[str]
    """List of stage names (e.g., ["S1", "S2", ...])."""
    stage_2_machines_map: dict[str, list[str]]
    """Mapping from stage names to machine names (e.g., {"S1": ["S1M1", "S1M2"]})."""
    machine_names: list[str]
    """List of all machine names (e.g., ["S1M1", "S1M2", "S2M1", ...])."""
    jobs: list[HybridFlowShopJob]
    """List of jobs in the problem instance."""

    num_jobs: int
    """Number of jobs in the problem instance."""
    num_stages: int
    """Number of stages in the problem instance."""

    def __init__(
        self,
        num_jobs: int,
        num_stages: int,
        machines_per_stage: list[int],
        processing_times: list[list[int]],
    ):
        """
        Initialize the problem instance.

        Args:
            num_jobs (int): Number of jobs.
            num_stages (int): Number of stages.
            machines_per_stage (list[int]): Number of parallel machines per stage.
            processing_times (list[list[int]]): Matrix of shape [num_jobs][num_stages] representing
                                                processing time of each job at each stage.
        """  # noqa: E501
        self.num_jobs = num_jobs
        self.num_stages = num_stages
        self.machines_per_stage = machines_per_stage  # e.g., [2, 3, 2]

        self.stage_names: list[str] = [f"S{i+1}" for i in range(num_stages)]
        self.stage_2_machines_map: dict[str, list[str]] = {}
        for stage, i in enumerate(self.stage_names):
            self.stage_2_machines_map[i] = [
                i + f"M{m+1}" for m in range(machines_per_stage[stage])
            ]
        self.machine_names: list[str] = [
            k for i in self.stage_names for k in self.stage_2_machines_map[i]
        ]

        self.jobs: list[HybridFlowShopJob] = []

        for j in range(num_jobs):
            job_name = f"J{j}"
            operations = []
            for stage, i in enumerate(self.stage_names):
                pt = processing_times[j][stage]
                # Assume all machines at this stage are eligible: "M1", "M2", ...
                eligible = self.stage_2_machines_map[i]
                op_name = job_name + i
                operation = HybridFlowShopFixedOperation(
                    processing_time=pt, eligible_mc_set=eligible, name=op_name
                )
                operations.append(operation)

            job = HybridFlowShopJob(operations, name=job_name)
            self.jobs.append(job)

    def __repr__(self):
        return (
            f"HybridFlowShopProblem(num_jobs={self.num_jobs},"
            f" num_stages={self.num_stages})"
        )

    def __str__(self):
        return f"Hybrid Flow Shop with {self.num_jobs} jobs, {self.num_stages} stages"

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
        num_jobs = int(stream.readline().strip())
        num_stages = int(stream.readline().strip())
        machines_per_stage = [int(x) for x in stream.readline().strip().split()]

        # Validate the number of stages and machines
        if len(machines_per_stage) != num_stages:
            raise ValueError(
                f"Stage count mismatch; num_stages={num_stages};"
                f" by machines_per_stage={len(machines_per_stage)}"
            )

        processing_times = []
        for _ in range(num_jobs):
            row = [int(x) for x in stream.readline().strip().split()]
            if len(row) != num_stages:
                raise ValueError(
                    f"Expected {num_stages} processing times per job, got {len(row)}."
                )
            processing_times.append(row)

        return cls(
            num_jobs=num_jobs,
            num_stages=num_stages,
            machines_per_stage=machines_per_stage,
            processing_times=processing_times,
        )
