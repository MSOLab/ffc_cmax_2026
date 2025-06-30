from .base.activity import Activity


class HybridFlowshopOperation(Activity):
    def __init__(
        self, job_name: str, stage_name: str, mc_name: str, start: int, end: int
    ) -> None:
        """
        Initialize a HybridFlowshopOperation.

        Args:
            job_name (str): The name of the job this operation belongs to.
            stage_name (str): The name of the stage this operation belongs to.
            eligible_mc_name_list (list[str]): List of machine IDs that can process this operation.

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

    # Required getters

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

    # Getters

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
