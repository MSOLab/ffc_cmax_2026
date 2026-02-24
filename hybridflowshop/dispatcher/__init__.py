"""
Dispatcher package for Hybrid Flow Shop scheduling.

This package provides dispatching heuristics to generate initial schedules
using various rules like CDS, Gupta, Palmer, and BN2D.

Usage:
    from hybridflowshop.dispatcher import Dispatcher

    dispatcher = Dispatcher(instance)
    schedule = dispatcher.get_schedule_by_best_of_selected_dispatches()
"""

from hybridflowshop.dispatcher.base import BaseDispatcher
from hybridflowshop.dispatcher.bn2d import BN2DDispatcher
from hybridflowshop.dispatcher.bn2d_option import BN2DOption
from hybridflowshop.dispatcher.job import JobDispatcher
from hybridflowshop.dispatcher.mixed import MixedDispatcher
from hybridflowshop.dispatcher.stage import StageDispatcher

from .utils import (
    dispatch_job_sequence_by_stages,
    dispatch_stages_by_job_sequence,
    from_job_sequence_get_schedule_mixed,
    reverse_even_positions,
)

__all__ = [
    "BaseDispatcher",
    "StageDispatcher",
    "JobDispatcher",
    "MixedDispatcher",
    "BN2DDispatcher",
    "BN2DOption",
    "dispatch_job_sequence_by_stages",
    "dispatch_stages_by_job_sequence",
    "from_job_sequence_get_schedule_mixed",
    "reverse_even_positions",
]
