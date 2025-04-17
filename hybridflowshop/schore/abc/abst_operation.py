from abc import ABC, abstractmethod


class AbstractOperation(ABC):
    """
    Abstract base class for operations in scheduling problems.

    Each operation has an immutable name and must implement the following:
    - A developer-friendly string representation via `__repr__`
    - A user-friendly string representation via `__str__`
    - A read-only `processing_time` property indicating its duration

    Attributes:
        name (str): The name of the operation (read-only).

    Notes:
        This class cannot be instantiated directly. Subclasses must implement all abstract methods and properties.
    """  # noqa: E501

    def __init__(self, name: str):
        """
        Initialize an operation with the specified name.

        Args:
            name (str): The name of the operation.

        Raises:
            TypeError: If this abstract base class is instantiated directly.
        """
        if type(self) is AbstractOperation:
            raise TypeError(
                "AbstractOperation is an abstract class "
                "and cannot be instantiated directly."
            )
        self.__name: str = name

    @property
    def name(self) -> str:
        """Returns the name of the operation."""
        return self.__name

    @name.setter
    def name(self, *args, **kwargs):
        """This attribute is read-only."""
        raise AttributeError("Cannot modify the name of an operation.")

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a developer-friendly string representation of the operation."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns a user-friendly string representation of the operation."""
        pass

    @property
    @abstractmethod
    def processing_time(self) -> int:
        """Returns the (representative) processing time of the operation."""
        pass
