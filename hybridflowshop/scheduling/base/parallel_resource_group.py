from .resource import ResourceT
from .resource_group import ResourceGroup


class ParallelResourceGroup(ResourceGroup[ResourceT]):
    # Getters

    def get_earliest_start_resource_name_and_time(
        self, duration: int, release_t: int = 0
    ) -> tuple[str, int]:
        """
        Find the resource in the group that can start a new activity at the earliest possible time.

        - Considers the duration of the activity and the earliest time it can start.
        - If multiple resources can start at the same earliest time,
        the first such resource in the list is returned.

        This is typically used in scheduling scenarios to assign tasks to resources
        such that tasks can be started as early as possible.

        Args:
            duration (int): The duration of the new activity.
            release_t (int, optional): The earliest possible start time for the activity.
                Defaults to 0.

        Raises:
            ValueError: If `duration` is not positive.
            ValueError: If no resource is available to start the activity.

        Returns:
            tuple[str, int]: A tuple containing the name of the resource and the earliest start time.

        Example:
            >>> group.get_earliest_start_resource_name_and_time(5, release_t=10)
            ('ResourceA', 12)
        """
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        earliest_start = float("inf")  # Use float("inf") instead of int("inf")
        earliest_resource_name: str | None = None

        for res in self.resources:
            start_time = res.get_earliest_start_time(duration, release_t)
            if start_time < earliest_start:
                earliest_start = start_time
                earliest_resource_name = res.name

        if earliest_resource_name is not None:
            return earliest_resource_name, int(earliest_start)
        raise ValueError("No resource available to start the activity")
