from __future__ import annotations

from schore.schedule.abstract import Activity


class HybridFlowshopOperation(Activity):
    def __init__(
        self, job_name: str, stage_name: str, mc_name: str, start: int, end: int
    ) -> None:
        """Initialize a HybridFlowshopOperation.

        Args:
            job_name (str): The name of the job this operation belongs to.
            stage_name (str): The name of the stage this operation belongs to.
            mc_name (str): The name of the machine that processes this operation.
            start (int): The start time of the operation.
            end (int): The end time of the operation.
        """
        super().__init__()

        # Inputs
        self._job_name: str = job_name
        """The name of the job this operation belongs to."""
        self._stage_name: str = stage_name
        """The name of the stage this operation belongs to."""
        self._mc_name: str = mc_name
        """The name of the machine that processes this operation."""
        self._start: int = start
        """The start time of the operation."""
        self._end: int = end
        """The end time of the operation."""

    def copy(self) -> HybridFlowshopOperation:
        """
        Returns a deep (hard) copy of this HybridFlowshopOperation.
        """
        return HybridFlowshopOperation(
            self._job_name, self._stage_name, self._mc_name, self._start, self._end
        )

    # Start required getters

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the operation in the format "job_name.stage_name".
        """
        return f"{self._job_name}.{self._stage_name}"

    @property
    def start(self) -> int:
        """
        Returns:
            int: The start time of the operation.
        """
        return self._start

    @property
    def end(self) -> int:
        """
        Returns:
            int: The end time of the operation.
        """
        return self._end

    # End required getters

    # Start getters

    @property
    def job_name(self) -> str:
        """
        Returns:
            str: The name of the job this operation belongs to.
        """
        return self._job_name

    @property
    def stage_name(self) -> str:
        """
        Returns:
            str: The name of the stage this operation belongs to.
        """
        return self._stage_name

    @property
    def mc_name(self) -> str:
        """
        Returns:
            str: The name of the machine that processes this operation.
        """
        return self._mc_name

    # End getters
