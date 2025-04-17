from collections.abc import Iterable

from ..abc.fixed_operation import FixedOperation


class HybridFlowShopFixedOperation(FixedOperation):
    """
    Concrete operation for hybrid flow shop environments with parallel machines.

    Extends FixedOperation with a set of eligible machines. Both processing time
    and machine eligibility are immutable once initialized.

    Attributes:
        eligible_mc_set (frozenset[str]): Machines eligible to process this operation.
    """

    def __init__(
        self, processing_time: int, eligible_mc_set: Iterable[str], name: str = ""
    ):
        super().__init__(processing_time, name)
        self.__eligible_mc_set = frozenset(eligible_mc_set)

    def __repr__(self):
        return (
            f"HybridFlowShopFixedOperation(name='{self.name}', "
            f"processing_time={self.processing_time}, "
            f"eligible_mc_set={sorted(self.eligible_mc_set)})"
        )

    def __str__(self):
        return (
            f"[HybridFlowShopFixedOperation] {self.name} — {self.processing_time} units"
        )

    @property
    def eligible_mc_set(self) -> frozenset[str]:
        """The set of machines eligible to process this operation."""
        return self.__eligible_mc_set

    @eligible_mc_set.setter
    def eligible_mc_set(self, *args, **kwargs):
        """This attribute is read-only."""
        raise AttributeError("Cannot modify the set of eligible machines.")
