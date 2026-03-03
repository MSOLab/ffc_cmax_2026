import logging
from typing import Any

from routix import StoppingCriteria


class LocalStoppingCriteria(StoppingCriteria):
    def get_subroutine_timelimit(
        self,
        global_remaining_sec: float,
        global_timelimit: float | None = None,
    ) -> float:
        """Calculate subroutine timelimit by reserving time from global remaining.

        Args:
            global_remaining_sec: Remaining time from global timelimit.
            global_timelimit: Original global timelimit (needed for percent calculation).

        Returns:
            Calculated timelimit for subroutine after applying reserve.
        """
        # Calculate reserve from absolute offset (if set)
        reserve = 0.0
        if self.stop_at_global_timelimit_minus is not None:
            reserve = max(reserve, self.stop_at_global_timelimit_minus)
        # Calculate reserve from percent-based offset (if set) and take the larger one
        if (
            self.stop_at_global_timelimit_minus_percent is not None
            and global_timelimit is not None
        ):
            percent_reserve = (
                global_timelimit * self.stop_at_global_timelimit_minus_percent
            )
            reserve = max(reserve, percent_reserve)
        return global_remaining_sec - reserve

    def __init__(self, param_dict: dict[str, Any]):
        # Loop stopping criteria
        self.max_loop_count: int | None = None
        """Maximum number of loops for the local subroutine."""
        self.max_no_improvement_steps: int | None = None
        """Stop if no improvement in the last N steps."""
        self.stop_at_global_timelimit_minus: float | None = None
        """Stop local subroutine this many seconds before the global timelimit."""
        self.stop_at_global_timelimit_minus_percent: float | None = None
        """Stop local subroutine when remaining time is <= this fraction of global timelimit."""
        self.lb_gap_is_leq: float | None = None
        """Stop if the lower bound gap is less than or equal to this value."""
        self.on_exception: str = "stop"
        """Action to take on exception during loop."""

        # Subroutine-wise stopping criteria
        self.rho_hits_ub: bool = False
        """
        If rho value for a subroutine is greater than or equal to its upper bound,
        exclude the subroutine from further consideration.
        """
        self.tl_hits_ub: bool = False
        """
        If computational_time value for a subroutine is greater than or equal to its upper bound,
        exclude the subroutine from further consideration.
        """

        super().__init__(param_dict)

    def is_loop_stopping_condition(
        self,
        loop_count: int,
        no_improvement_steps: int,
        global_remaining_sec: float,
        lb_gap: float | None,
        global_timelimit: float | None = None,
        log_reason_if_true: bool = True,
    ) -> bool:
        if self.max_loop_count is not None and loop_count == self.max_loop_count:
            if log_reason_if_true:
                logging.info("Stopping local subroutine: reached max_loop_count.")
            return True

        # Check absolute timelimit threshold
        if (
            self.stop_at_global_timelimit_minus is not None
            and global_remaining_sec <= self.stop_at_global_timelimit_minus
        ):
            if log_reason_if_true:
                logging.info(
                    "Stopping local subroutine: reached global timelimit minus threshold."
                )
            return True

        # Check relative timelimit threshold (percent of global timelimit)
        if (
            self.stop_at_global_timelimit_minus_percent is not None
            and global_timelimit is not None
            and global_remaining_sec
            <= self.stop_at_global_timelimit_minus_percent * global_timelimit
        ):
            if log_reason_if_true:
                logging.info(
                    "Stopping local subroutine: reached global timelimit minus percent threshold."
                )
            return True

        if (
            self.max_no_improvement_steps is not None
            and no_improvement_steps == self.max_no_improvement_steps
        ):
            if log_reason_if_true:
                logging.info(
                    "Stopping local subroutine: reached max_no_improvement_steps."
                )
            return True
        if (
            self.lb_gap_is_leq is not None
            and lb_gap is not None
            and lb_gap <= self.lb_gap_is_leq
        ):
            if log_reason_if_true:
                logging.info(
                    "Stopping local subroutine: reached lb_gap_is_leq threshold."
                )
            return True

        return False
