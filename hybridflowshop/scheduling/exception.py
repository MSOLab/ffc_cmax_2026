class SchedulingFailureException(Exception):
    def __init__(
        self,
        activity_name: str,
        p: int,
        resource_name: str,
        start_time: int,
        message: str | None = None,
    ):
        if message is None:
            message = (
                f"Failed to schedule {activity_name} of length {p}"
                f" in {resource_name} at time {start_time}"
            )
        super().__init__(message)
        self.activity_name = activity_name
        self.resource_name = resource_name
        self.start_time = start_time
