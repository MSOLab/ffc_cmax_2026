from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ParallelMcParams:
    # Indices & Parameters

    j_list: Sequence[str]
    """$J$: job id (j) list"""

    i_list: Sequence[str]
    """$I$: machine id (i) list"""

    p: Mapping[str, int]
    """$p_j$: processing time of job j"""

    r: Mapping[str, int] | None
    """$r_j$: release time of job j"""

    tr: Mapping[str, int] | None
    """${tr}_j$: transition time of job j"""
