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
from hybridflowshop.dispatcher.job import JobDispatcher
from hybridflowshop.dispatcher.mixed import MixedDispatcher
from hybridflowshop.dispatcher.stage import StageDispatcher

__all__ = [
    "BaseDispatcher",
    "StageDispatcher",
    "JobDispatcher",
    "MixedDispatcher",
]
