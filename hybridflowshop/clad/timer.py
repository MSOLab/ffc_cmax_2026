import datetime as dt


class Timer:
    # the date and time at which the handler was initiated
    _start_dt: dt.datetime
    _start_time_in_seconds: float

    def set_start_time(self, start_dt: dt.datetime):
        self._start_dt = start_dt
        self._start_time_in_seconds = start_dt.timestamp()

    def set_start_time_as_now(self):
        self.set_start_time(dt.datetime.now())

    def get_start_time_in_seconds(self) -> float:
        return self._start_time_in_seconds

    def get_formatted_start_dt(self) -> str:
        return (
            self._start_dt.strftime("%Y-%m-%d %H:%M:%S.")
            + f"{int(self._start_dt.microsecond / 10000):02d}"
        )

    @staticmethod
    def get_current_time_in_seconds() -> float:
        return dt.datetime.now().timestamp()

    def get_elapsed_time(self) -> float:
        """
        Returns:
            float: Seconds elapsed since the handler was initiated
        """
        return self.get_current_time_in_seconds() - self._start_time_in_seconds
