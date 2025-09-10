from typing import Any

from routix import StoppingCriteria


class LocalStoppingCriteria(StoppingCriteria):
    max_loop_count: int
    """Maximum number of loops for the local subroutine."""
    max_no_improvement_steps: int
    """Stop if no improvement in the last N steps."""
    rho_is_geq: float
    """Stop if the rho value is greater than or equal to this value."""
    stop_at_global_timelimit_minus: float
    """Stop local subroutine this many seconds before the global timelimit."""
    lb_gap_is_leq: float
    """Stop if the lower bound gap is less than or equal to this value."""
    on_exception: str
    """Action to take on exception during loop."""

    def __init__(self, param_dict: dict[str, Any]):
        super().__init__(param_dict)

    def is_stopping_condition(
        self,
        loop_count: int,
        no_improvement_steps: int,
        min_rho: float,
        remaining_sec: float,
        lb_gap: float | None,
    ) -> bool:
        if loop_count == self.max_loop_count:
            return True
        if remaining_sec <= self.stop_at_global_timelimit_minus:
            return True
        if min_rho >= self.rho_is_geq:
            return True
        if no_improvement_steps == self.max_no_improvement_steps:
            return True
        if lb_gap is not None and lb_gap <= self.lb_gap_is_leq:
            return True
        return False
