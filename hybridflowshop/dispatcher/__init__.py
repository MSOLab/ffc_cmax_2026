"""
Dispatcher package for Hybrid Flow Shop scheduling.

This package provides dispatching heuristics to generate initial schedules
using various rules like CDS, Gupta, Palmer, BN2D, and Machine-first dispatch.

Usage:
    from hybridflowshop.dispatcher.machine import MachineDispatcher

    dispatcher = MachineDispatcher(instance)
    schedule = dispatcher.get_schedule_by_cds()
"""

from hybridflowshop.dispatcher.base import BaseDispatcher
from hybridflowshop.dispatcher.bn2d import BN2DDispatcher
from hybridflowshop.dispatcher.bn2d_option import BN2DOption
from hybridflowshop.dispatcher.job import JobDispatcher
from hybridflowshop.dispatcher.machine import MachineDispatcher
from hybridflowshop.dispatcher.mixed import MixedDispatcher
from hybridflowshop.dispatcher.stage import StageDispatcher

from .utils import (
    build_schedule_from_stage_job_sequences_priority_score,
    build_schedule_from_stage_job_sequences_strict_call_order,
    dispatch_job_sequence_by_stages,
    dispatch_stage_job_sequences_priority_score,
    dispatch_stage_job_sequences_strict_call_order,
    dispatch_stages_by_job_sequence,
    from_job_sequence_get_schedule_mixed,
    get_bottleneck_anchor_stage_from_solution_payload,
    get_job_sequence_from_dispatch_windows_aggregate,
    get_job_sequence_from_dispatch_windows_anchor_stage,
    get_job_tiebreak_rank_from_job_sequence,
    get_job_tiebreak_rank_from_stage_job_sequences,
    get_stage_job_sequences_from_dispatch_windows,
    get_stage_job_sequences_from_schedule,
    improve_schedule_by_critical_adjacent_swaps,
    improve_schedule_by_critical_cross_machine_insertions,
    improve_schedule_by_critical_stage_sequence_insertions,
    reverse_even_positions,
)

__all__ = [
    "BaseDispatcher",
    "StageDispatcher",
    "JobDispatcher",
    "MixedDispatcher",
    "BN2DDispatcher",
    "BN2DOption",
    "MachineDispatcher",
    "build_schedule_from_stage_job_sequences_priority_score",
    "build_schedule_from_stage_job_sequences_strict_call_order",
    "dispatch_stage_job_sequences_priority_score",
    "dispatch_stage_job_sequences_strict_call_order",
    "dispatch_job_sequence_by_stages",
    "dispatch_stages_by_job_sequence",
    "from_job_sequence_get_schedule_mixed",
    "get_bottleneck_anchor_stage_from_solution_payload",
    "get_job_sequence_from_dispatch_windows_aggregate",
    "get_job_sequence_from_dispatch_windows_anchor_stage",
    "get_job_tiebreak_rank_from_job_sequence",
    "get_job_tiebreak_rank_from_stage_job_sequences",
    "get_stage_job_sequences_from_schedule",
    "get_stage_job_sequences_from_dispatch_windows",
    "improve_schedule_by_critical_stage_sequence_insertions",
    "improve_schedule_by_critical_adjacent_swaps",
    "improve_schedule_by_critical_cross_machine_insertions",
    "reverse_even_positions",
]
