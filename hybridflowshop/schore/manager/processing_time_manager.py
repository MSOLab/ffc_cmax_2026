from typing import Self, TextIO, Type, TypeVar

import pandas as pd

from .base.table_2d_manager import Table2DManager

T = TypeVar("T", bound=int)


class JobStageProcessingTimeManager(Table2DManager):
    """colums for stages, rows for jobs"""

    def __init__(self, name: str, df: pd.DataFrame) -> None:
        super().__init__(name, df)

    @classmethod
    def from_text_stream(
        cls,
        stream: TextIO,
        row_count: int,
        dtype: Type[T] | None = None,
        sep: str | None = None,
        name: str = "JobStageProcessingTimeManager",
    ) -> Self:
        return super().from_text_stream(stream, row_count, dtype, sep, name)

    def stage_2_job_2_value_map(
        self, stage_id_list: list[str], job_id_list: list[str]
    ) -> dict[str, dict[str, T]]:
        """
        Create a mapping from stage ID to job ID and value.

        Args:
            stage_id_list (list[str]): List of stage IDs (columns).
            job_id_list (list[str]): List of job IDs (rows).

        Returns:
            dict[str, dict[str, T]]: Stage ID -> Job ID -> Value.
        """
        assert len(stage_id_list) == self.col_count(), "stage count mismatch"
        assert len(job_id_list) == self.row_count(), "job count mismatch"

        return {
            stage_id: {
                job_id: self.df.iat[row_idx, col_idx]
                for row_idx, job_id in enumerate(job_id_list)
            }
            for col_idx, stage_id in enumerate(stage_id_list)
        }

    def job_2_stage_2_value_map(
        self, job_id_list: list[str], stage_id_list: list[str]
    ) -> dict[str, dict[str, T]]:
        """
        Create a mapping from job ID to stage ID and value.

        Args:
            job_id_list (list[str]): List of job IDs (rows).
            stage_id_list (list[str]): List of stage IDs (columns).

        Returns:
            dict[str, dict[str, T]]: Job ID -> Stage ID -> Value.
        """
        assert len(job_id_list) == self.row_count(), "job count mismatch"
        assert len(stage_id_list) == self.col_count(), "stage count mismatch"

        return {
            job_id: {
                stage_id: self.df.iat[row_idx, col_idx]
                for col_idx, stage_id in enumerate(stage_id_list)
            }
            for row_idx, job_id in enumerate(job_id_list)
        }
