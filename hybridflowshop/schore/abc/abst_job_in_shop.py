from abc import ABC, abstractmethod
from typing import Sequence

from .abst_operation import AbstractOperation


class AbstractJobInShop(ABC):
    """
    Abstract base class for jobs in flow shop, job shop, or open shop scheduling environments.

    Defines the common structure and required behaviors for job instances, including:
    - A sequence of operations (`AbstractOperation` objects)
    - A unique job name
    - Abstract string representations (__repr__, __str__)

    Attributes:
        operations (Sequence[AbstractOperation]): The sequence of operations that make up this job.
        name (str): The name of the job (read-only).

    Notes:
        This class cannot be instantiated directly. Subclasses must implement all abstract methods and properties.
    """  # noqa: E501

    def __init__(self, operations: Sequence[AbstractOperation], name: str = ""):
        """
        Initialize the job with a sequence of operations and an optional name.

        Args:
            operations (Sequence[AbstractOperation]): The sequence of operations to be performed by this job.
            name (str, optional): The name of the job. Defaults to an empty string.

        Raises:
            TypeError: If this abstract base class is instantiated directly.
        """  # noqa: E501
        if type(self) is AbstractJobInShop:
            raise TypeError(
                "AbstractJobInShop is an abstract class "
                "and cannot be instantiated directly."
            )

        # Make a mutable internal copy to prevent external modification
        self.__operations = list(operations)
        self.__name = name

    @property
    def operations(self) -> Sequence[AbstractOperation]:
        """The sequence of operations associated with the job."""
        return self.__operations

    @property
    def name(self) -> str:
        """The name of the job."""
        return self.__name

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the job,
        e.g., 'AbstractJobInShop(name=JobA)'."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the job,
        e.g., 'Job A: 3 operations'.
        """
        pass
