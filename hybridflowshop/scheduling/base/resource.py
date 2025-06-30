import bisect
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .activity import ActivityT


class Resource(Generic[ActivityT], ABC):
    """
    Abstract base class representing a resource that can be assigned activities in a scheduling system.

    Each resource maintains a (time-sorted) list of activities assigned to it, and provides
    common scheduling utilities such as checking for time conflicts, finding available time slots,
    and computing the makespan. Subclasses should implement the `name` property.

    Example:
        class Machine(Resource[Activity]):
            @property
            def name(self) -> str:
                return self._machine_name
    """

    def __init__(self) -> None:
        self._activity_list: list[ActivityT] = []
        """
        Internal list of activities assigned to this resource.
        This list is kept sorted by activity start time.
        Note: Not thread-safe; synchronize externally if used in concurrent contexts.
        """

    # Abstract getters

    @property
    @abstractmethod
    def name(self) -> str:
        """The name or unique identifier of the resource."""
        ...

    # Getters

    @property
    def activity_list(self) -> list[ActivityT]:
        """
        Returns:
            list[ActivityT]: A copy of the activity list, sorted by start time.
                Modifying the returned list does not affect the internal state.
        """
        return self._activity_list.copy()

    def _get_index_of_1st_activity_after(self, time: int) -> int:
        """Find the index of the first activity starting at or after the specified time.

        Args:
            time (int): The reference time.

        Returns:
            int: Index of the first activity with start >= time, or len(activity_list) if none.
        """
        starts = [acty.start for acty in self._activity_list]
        # Use bisect for efficient search since _activity_list is sorted by start time
        return bisect.bisect_right(starts, time - 1)

    def get_activity_count_started_before(self, time: int) -> int:
        """Count activities that started before the given time.

        Args:
            time (int): Time threshold.

        Returns:
            int: The number of activities with start < time.
        """
        return self._get_index_of_1st_activity_after(time)

    def check_for_time_conflict(self, start_time: int, end_time: int) -> bool:
        """Check if any assigned activity overlaps with the given time interval.

        Args:
            start_time (int): Start of the interval (inclusive).
            end_time (int): End of the interval (exclusive).

        Returns:
            bool: True if there is any overlap, False otherwise.

        Note:
            Assumes activities are non-overlapping and activity_list is sorted by start time.
        """
        # Find the 1st activity that starts at or after start_time
        idx = self._get_index_of_1st_activity_after(start_time)

        # Check if the activity right before the start time overlaps
        if idx > 0:
            before_acty = self._activity_list[idx - 1]
            if start_time < before_acty.end:
                return True

        # Check if the activity at or right after the start time overlaps
        if idx < len(self._activity_list):
            after_acty = self._activity_list[idx]
            if after_acty.start < end_time:
                return True

        return False

    @property
    def makespan(self) -> int:
        """
        Returns:
            int: End time of the last activity, or 0 if none.
        """
        if not self._activity_list:
            return 0
        return self._activity_list[-1].end

    def get_earliest_start_time(self, duration: int, release_t: int = 0) -> int:
        """
        Find the earliest feasible start time >= release_t for a new activity of given duration.

        The method returns the first available time slot, after release_t,
        such that the new activity would not overlap with any existing one.

        Args:
            duration (int): Required duration for the new activity (must be positive).
            release_t (int, optional): Earliest possible start time. Defaults to 0.

        Returns:
            int: Earliest feasible start time for the activity.

        Raises:
            ValueError: If duration is not positive.

        Example:
            If current activities are [0-10], [12-14], and duration=2, release_t=10,
            the method would return 10.
        """
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        prev_end = max(0, release_t)

        # Start from the 1st activity that starts at or after prev_end
        start_idx = self._get_index_of_1st_activity_after(prev_end)

        # Check if there's an activity before start_idx that might affect prev_end
        if start_idx > 0:
            before_acty = self._activity_list[start_idx - 1]
            if before_acty.end > prev_end:
                prev_end = before_acty.end

        # Check gaps starting from start_idx
        for i in range(start_idx, len(self._activity_list)):
            acty = self._activity_list[i]
            # If the gap before acty is big enough
            if prev_end + duration <= acty.start:
                return prev_end
            # Update prev_end
            if prev_end < acty.end:
                prev_end = acty.end
        return prev_end

    # Setters

    def clear(self) -> None:
        """
        Remove all activities from the resource.

        Side effects:
            - Empties the internal activity list.
        """
        self._activity_list.clear()

    def add_activity(
        self, acty: ActivityT, force_add: bool = False
    ) -> ActivityT | None:
        """Add an activity to the resource.

        By default, adds the activity only if it does not conflict with existing activities.
        If `force_add` is True, the activity is inserted regardless of conflicts.

        Args:
            acty (ActivityT): The activity to add.
            force_add (bool, optional): If True, add even if conflicting. Defaults to False.

        Returns:
            ActivityT | None: The activity if added, None if not added due to conflict.

        Side effects:
            - Modifies the internal activity list (inserts at the correct position).

        Note:
            It is recommended to avoid using `force_add=True` unless for debugging
            or special use cases, as this may result in overlapping activities.
        """
        # Index of activity in resource that is the 1st to start after added activity ends.
        index = self._get_index_of_1st_activity_after(acty.end)
        if not force_add and self.check_for_time_conflict(acty.start, acty.end):
            return None
        self._activity_list.insert(index, acty)
        return acty


ResourceT = TypeVar("ResourceT", bound=Resource)
"""
Type variable for use with Resource subclasses.

Ensures type safety in generic scheduling code involving resources.
Only classes derived from Resource are allowed.

Example:
    def assign(resource: ResourceT, activity: ActivityT) -> None:
        resource.add_activity(activity)
"""
