from abc import ABC

from .abst_operation import AbstractOperation


class FixedOperation(AbstractOperation, ABC):
    """
    Abstract base class for operations with fixed, non-overridable processing times.

    This class guarantees that all operations have:
    - a non-negative, immutable processing time
    - an immutable name (inherited from AbstractOperation)

    Subclasses may extend this with additional metadata (e.g., machine ID, job ID),
    but cannot change core time behavior.

    Attributes:
        rocessing_time (int): The fixed, non-negative processing time of the operation (read-only).

    Notes:
        This class cannot be instantiated directly. Subclasses must implement all abstract methods and properties.
    """  # noqa: E501

    # Note:
    # - Subclasses of FixedOperation must still implement:
    #     - __repr__: for developer/debugging representation
    #     - __str__: for user-friendly string representation

    def __init__(self, processing_time: int, name: str = ""):
        """
        Initialize the fixed operation with a name and processing time.

        Args:
            processing_time (int): Must be a non-negative integer.
            name (str, optional): The name of the operation. Defaults to "".

        Raises:
            TypeError: If this abstract base class is instantiated directly,
                or if processing_time is not an integer (bools are also rejected).
            ValueError: If processing_time is negative.
        """
        if type(self) is FixedOperation:
            raise TypeError(
                "FixedOperation is an abstract class "
                "and cannot be instantiated directly."
            )
        if not isinstance(processing_time, int) or isinstance(processing_time, bool):
            raise TypeError("Processing time must be an integer.")
        if processing_time < 0:
            raise ValueError("Processing time must be non-negative.")

        super().__init__(name)

        self.__processing_time: int = processing_time

    @property
    def processing_time(self) -> int:
        """
        Return the fixed, non-negative processing time.

        This property should not be overridden by subclasses.
        """
        return self.__processing_time

    @processing_time.setter
    def processing_time(self, *args, **kwargs):
        """
        Prevent modification of processing time after initialization.

        Raises:
            AttributeError: Always raised when attempting to modify the value.
        """
        raise AttributeError("Cannot modify the processing time of a fixed operation.")
