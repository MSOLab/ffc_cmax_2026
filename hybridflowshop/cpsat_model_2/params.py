from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # Indices & Parameters

    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    M_of: dict[str, list[str]]
    """$M_i$: machine index (k) list for stage i"""

    p: dict[tuple[str, str], int]
    """$P_{ji}$: processing time of job j at stage i"""
