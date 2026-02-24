"""
StageDispatcher class for stage-sequence dispatch method (DS).
"""

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule

from .base import BaseDispatcher, JobIdType, StageIdType
from .utils import dispatch_stages_by_job_sequence


class StageDispatcher(BaseDispatcher):
    """
    Stage-sequence dispatch methods (DS).

    These methods dispatch all jobs through one stage at a time,
    rather than dispatching jobs one-by-one through all stages.
    """

    def get_schedule_by_ds_cds(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        from_stage: StageIdType | None = None,
        job_2_release_t: dict[JobIdType, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """Get schedule using DS (Stage-first dispatch) with CDS sequence.

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
            )

            makespan = _schedule.makespan
            if best_obj is None or best_obj > makespan:
                best_obj = makespan
                best_sch = _schedule
                best_k = k

        print(f"Best DS(CDS) schedule found with k={best_k}, makespan={best_obj}")
        return best_sch

    def get_schedule_by_ds_gupta(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        from_stage: StageIdType | None = None,
        job_2_release_t: dict[JobIdType, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """Get schedule using DS (Stage-first dispatch) with Gupta sequence.

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
        )
        return _schedule

    def get_schedule_by_ds_palmer(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        from_stage: StageIdType | None = None,
        job_2_release_t: dict[JobIdType, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """Get schedule using DS (Stage-first dispatch) with Palmer sequence.

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
        job_sequence = self.get_palmer_sequence()
        dispatch_stages_by_job_sequence(
            _schedule,
            job_sequence,
            self.stage_2_job_2_p,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
        )
        return _schedule
