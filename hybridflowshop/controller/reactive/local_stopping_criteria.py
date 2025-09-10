import logging
from typing import Any

from routix import StoppingCriteria


class LocalStoppingCriteria(StoppingCriteria):
    def __init__(self, param_dict: dict[str, Any]):
        self.max_loop_count: int | None = None
        """Maximum number of loops for the local subroutine."""
        self.max_no_improvement_steps: int | None = None
        """Stop if no improvement in the last N steps."""
        self.rho_is_geq: float | None = None
        """Stop if the rho value is greater than or equal to this value."""
        self.stop_at_global_timelimit_minus: float | None = None
        """Stop local subroutine this many seconds before the global timelimit."""
        self.lb_gap_is_leq: float | None = None
        """Stop if the lower bound gap is less than or equal to this value."""
        self.on_exception: str = "stop"
        """Action to take on exception during loop."""

        super().__init__(param_dict)

    def is_stopping_condition(
        self,
        loop_count: int,
        no_improvement_steps: int,
        min_rho: float,
        global_remaining_sec: float,
        lb_gap: float | None,
    ) -> bool:
        if self.max_loop_count is not None and loop_count == self.max_loop_count:
            logging.info("Stopping local subroutine: reached max_loop_count.")
            return True
        if (
            self.stop_at_global_timelimit_minus is not None
            and global_remaining_sec <= self.stop_at_global_timelimit_minus
        ):
            logging.info(
                "Stopping local subroutine: reached global timelimit minus threshold."
            )
            return True
        if self.rho_is_geq is not None and min_rho >= self.rho_is_geq:
            logging.info("Stopping local subroutine: reached rho_is_geq threshold.")
            return True
        if (
            self.max_no_improvement_steps is not None
            and no_improvement_steps == self.max_no_improvement_steps
        ):
            logging.info("Stopping local subroutine: reached max_no_improvement_steps.")
            return True
        if (
            self.lb_gap_is_leq is not None
            and lb_gap is not None
            and lb_gap <= self.lb_gap_is_leq
        ):
            logging.info("Stopping local subroutine: reached lb_gap_is_leq threshold.")
            return True
        return False
