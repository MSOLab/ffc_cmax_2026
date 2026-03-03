from dataclasses import dataclass


@dataclass
class BN2DOption:
    left_cap_multiplier: int | None = None
    right_cap_multiplier: int | None = None
    left_cap_portion: float | None = None
    right_cap_portion: float | None = None
    normalize_by_stage_cnt: bool = False

    randomize_mid_all: bool = False
    """
    If True, the order of mid-jobs is randomized.
    `reverse_mid_even` and `reverse_mid_all` are ignored if this is True.
    """

    reverse_mid_even: bool = False
    """
    If True, the order of mid-jobs is reversed at even positions (1-based).
    For example, [A,B,C,D,E,F,G,H] -> [A,H,C,F,E,D,G,B]
    `reverse_mid_all` is ignored if this is True.
    """

    reverse_mid_all: bool = False
    """
    If True, the order of mid-jobs is reversed.
    """

    mixed_schedule_for_former_stages: bool = False
    """
    If True, use mixed schedule for stages before bottleneck stage.
    """

    mixed_schedule_for_later_stages: bool = False
    """
    If True, use mixed schedule for stages after bottleneck stage.
    """

    machine_then_job: bool = False
    """
    If True, dispatch by stage - machine - job priority.
    If False, dispatch by stage - job - machine priority.
    """
