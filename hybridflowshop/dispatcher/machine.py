"""MachineDispatcher class for machine-centric dispatch methods (DM)."""

from typing import Mapping
import logging
from schore.parameters_examples import HybridFlowshopParameters

from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    StageIdType,
)

from .base import BaseDispatcher
from .utils import dispatch_stages_by_job_sequence


class MachineDispatcher(BaseDispatcher):
    """Machine-centric dispatch methods (DM)."""

    def __init__(
        self,
        instance: HybridFlowshopParameters,
        last_stage_dispatch_rule: str = "spt",
        spt_on_last_stage: bool = False,
    ):
        super().__init__(instance)
        self.spt_on_last_stage = spt_on_last_stage

    def get_schedule_by_cds(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        from_stage: StageIdType | None = None,
        job_2_release_t: dict[JobIdType, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """Get schedule using DM (Machine-first dispatch) with CDS sequence.

        Args:
            schedule (HybridFlowshopLiteSchedule | None, optional): Given schedule,
                may be empty or partially scheduled. Defaults to None.
            from_stage (StageIdType | None, optional): The first stage to dispatch jobs.
                If not provided, defaults to the first stage in schedule.
            job_2_release_t (dict[JobIdType, int] | None, optional): The release time
                for each job. Defaults to None.

        Returns:
            HybridFlowshopLiteSchedule | None: The generated schedule, or None if infeasible.
        """
        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None
        best_k = -1

        for k in range(1, self.stage_count):
            _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=False)
            job_sequence = self.get_cds_sequence(k)
            dispatch_stages_by_job_sequence(
                _schedule,
                job_sequence,
                self.stage_2_job_2_p,
                from_stage=from_stage,
                job_2_release_t=job_2_release_t,
                machine_then_job=True,
                spt_on_last_stage=self.spt_on_last_stage,
            )

            makespan = _schedule.makespan
            if best_obj is None or best_obj > makespan:
                best_obj = makespan
                best_sch = _schedule
                best_k = k

        logging.debug(
            f"Best DM(CDS) schedule found with k={best_k}, makespan={best_obj}"
        )
        return best_sch

    def get_schedule_by_gupta(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        from_stage: StageIdType | None = None,
        job_2_release_t: dict[JobIdType, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """Get schedule using DM (Machine-first dispatch) with Gupta sequence.

        Args:
            schedule (HybridFlowshopLiteSchedule | None, optional): Given schedule,
                may be empty or partially scheduled. Defaults to None.
            from_stage (StageIdType | None, optional): The first stage to dispatch jobs.
                If not provided, defaults to the first stage in schedule.
            job_2_release_t (dict[JobIdType, int] | None, optional): The release time
                for each job. Defaults to None.

        Returns:
            HybridFlowshopLiteSchedule | None: The generated schedule, or None if infeasible.
        """
        _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=False)
        job_sequence = self.get_gupta_sequence()
        dispatch_stages_by_job_sequence(
            _schedule,
            job_sequence,
            self.stage_2_job_2_p,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
            machine_then_job=True,
            spt_on_last_stage=self.spt_on_last_stage,
        )
        return _schedule

    def get_schedule_by_palmer(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        from_stage: StageIdType | None = None,
        job_2_release_t: Mapping[JobIdType, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """Get schedule using DM (Machine-first dispatch) with Palmer sequence.

        Uses Palmer's slope index heuristic to create job sequence,
        then dispatches using machine-first strategy.
        The Palmer sequence is used as the priority for the first stage only.
        For subsequent stages, LRT-based priority is used.

        Priority: (1) stage order, (2) earliest available machine,
                  (3) job order from Palmer sequence (first stage only)

        Args:
            schedule: Given schedule, may be empty or partially scheduled.
                If None, creates a new empty schedule.
            from_stage: The first stage to dispatch jobs.
                If None, starts from the first stage.
            job_2_release_t: The release time for each job.

        Returns:
            The generated schedule, or None if infeasible.
        """
        _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=False)
        job_sequence = self.get_palmer_sequence()
        dispatch_stages_by_job_sequence(
            _schedule,
            job_sequence,
            self.stage_2_job_2_p,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
            machine_then_job=True,
            spt_on_last_stage=self.spt_on_last_stage,
        )
        return _schedule
