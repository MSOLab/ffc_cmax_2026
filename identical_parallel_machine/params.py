from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelMcParams:
    # Indices & Parameters

    j_list: list[str]
    """$J$: job id (j) list"""

    i_list: list[str]
    """$I$: machine id (i) list"""

    p: dict[str, int]
    """$p_j$: processing time of job j"""

    r: dict[str, int] | None
    """$r_j$: release time of job j"""

    tr: dict[str, int] | None
    """${tr}_j$: transition time of job j"""
