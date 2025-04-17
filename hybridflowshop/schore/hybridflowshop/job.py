from typing import Sequence, cast

from ..abc.abst_job_in_shop import AbstractJobInShop
from .fixed_operation import HybridFlowShopFixedOperation


class HybridFlowShopJob(AbstractJobInShop):
    """
    Concrete job for hybrid flow shop scheduling environments.

    Each job consists of a sequence of HybridFlowShopFixedOperation instances and a unique name.
    Provides readable string representations for both developers and users.
    """  # noqa: E501

    def __init__(
        self, operations: Sequence[HybridFlowShopFixedOperation], name: str = ""
    ):
        """
        Initialize the hybrid flow shop job.

        Args:
            operations (Sequence[HybridFlowShopFixedOperation]): The ordered operations for this job.
            name (str, optional): The name of the job. Defaults to an empty string.
        """  # noqa: E501
        super().__init__(operations, name)

    def __repr__(self) -> str:
        return (
            f"HybridFlowShopJob(name='{self.name}',"
            f" num_operations={len(self.operations)})"
        )

    def __str__(self) -> str:
        return f"[HybridFlowShopJob] {self.name}: {len(self.operations)} operations"

    @property
    def operations(self) -> Sequence[HybridFlowShopFixedOperation]:
        """The sequence of operations associated with the job."""
        return cast(Sequence[HybridFlowShopFixedOperation], super().operations)
