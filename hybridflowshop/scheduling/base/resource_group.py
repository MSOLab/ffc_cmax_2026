from abc import ABC, abstractmethod
from typing import Generic

from .resource import ResourceT


class ResourceGroup(Generic[ResourceT], ABC):
    """
    Abstract base class representing a group of resources in a scheduling system.

    This class defines the common interface for any group of resources that can
    be managed collectively in a scheduling context. A resource group may represent
    a set of machines, workers, or any entities capable of processing activities,
    and can be configured for parallel or sequential operation depending on subclass.

    Subclasses must implement the group name, the resource list, and resource addition.

    Example:
        class MachineGroup(ResourceGroup[Machine]):
            def __init__(self, name):
                self._name = name
                self._resources = []

            @property
            def name(self):
                return self._name

            @property
            def resources(self):
                return self._resources

            def add_resource(self, res: Machine) -> None:
                self._resources.append(res)
    """

    # Abstract getters

    @property
    @abstractmethod
    def name(self) -> str:
        """The name or unique identifier of the resource group."""
        ...

    @property
    @abstractmethod
    def resources(self) -> list[ResourceT]:
        """
        The list of resources contained in this group.

        Returns:
            list[ResourceT]: All resources currently managed by this group.
        """
        ...

    # Abstract setters

    @abstractmethod
    def add_resource(self, res: ResourceT) -> None:
        """Add a resource to the group.

        Args:
            res (ResourceT): The resource to add.

        Side effects:
            - Modifies the internal resource list.
        """
        ...

    # Getters

    @property
    def makespan(self) -> int:
        """
        Returns:
            int: The largest makespan value among the group's resources,
                or 0 if the group is empty.
        """
        return max((res.makespan for res in self.resources), default=0)
