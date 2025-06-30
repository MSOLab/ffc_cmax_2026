from abc import ABC, abstractmethod
from typing import TypeVar


class Activity(ABC):
    """
    Abstract base class representing a schedulable activity.

    This class defines the required interface for any activity
    that will be used in scheduling algorithms or systems.
    Each activity has a name, start time, and end time.
    The duration is computed from the difference between end and start.

    Subclasses must implement the `name`, `start`, and `end` properties.

    Typical usage includes representing jobs, operations, or tasks
    that are subject to scheduling, sequencing, or resource allocation.

    Example:
        class MyActivity(Activity):
            def __init__(self, name, start, end):
                self._name = name
                self._start = start
                self._end = end

            @property
            def name(self): return self._name

            @property
            def start(self): return self._start

            @property
            def end(self): return self._end
    """

    # Abstract getters

    @property
    @abstractmethod
    def name(self) -> str:
        """The name or unique identifier of the activity."""
        ...

    @property
    @abstractmethod
    def start(self) -> int:
        """The start time (inclusive) of the activity, as an integer timestamp."""
        ...

    @property
    @abstractmethod
    def end(self) -> int:
        """The end time (exclusive) of the activity, as an integer timestamp."""
        ...

    # Getters

    @property
    def duration(self) -> int:
        """
        The duration of the activity, computed as `end - start`.

        Returns:
            int: The positive duration of the activity.

        Raises:
            ValueError: If the duration is zero or negative,
                        indicating an error in the activity's start or end times.

        Notes:
            - If `end < start`, this signals a logical error in the activity's timing.
            - If `end == start`, zero-duration activities are considered invalid.
        """
        val = self.end - self.start
        if val < 0:
            raise ValueError(
                f"Activity {self.name} has invalid duration: end ({self.end}) is less than start ({self.start})"
            )
        if val == 0:
            raise ValueError(f"Activity {self.name} has zero duration")
        return val


ActivityT = TypeVar("ActivityT", bound=Activity)
"""
Type variable for generics involving Activity subclasses.

`ActivityT` is used to indicate a generic activity type in type annotations,
ensuring that only classes derived from `Activity` can be used.
This supports type-safe collections, functions, or classes that operate on any Activity subtype.

Example:
```
    @property
    def activity_list(self) -> list[ActivityT]:
        # Operate on any list of activities, as long as they inherit from Activity
        pass
```
"""
