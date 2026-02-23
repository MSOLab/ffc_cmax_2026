"""
JobDispatcher class for job-sequence dispatch methods (DJ).
"""

from typing import Sequence

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule

from .base import BaseDispatcher


class JobDispatcher(BaseDispatcher):
    """
    Job-sequence dispatch methods (DJ).

    These methods dispatch jobs one-by-one through all stages,
    rather than dispatching all jobs through one stage at a time.
    """

    def dispatch_job_sequence_by_stages(
        self,
        schedule: HybridFlowshopLiteSchedule,
        job_sequence: Sequence[str],
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> None:
        """Dispatch a sequence of jobs through all stages.

        Args:
            schedule (HybridFlowshopLiteSchedule): The schedule to modify in-place.
            job_sequence (Sequence[str]): The sequence of job IDs to dispatch.
            from_stage (str | None, optional): The first stage to dispatch jobs through.
                Defaults to None.
            job_2_release_t (dict[str, int] | None, optional): The release time for each job.
                Defaults to None.
        """
        for job_id in job_sequence:
            schedule.dispatch_job_by_stages(
                job_id,
                self.job_2_stage_2_p[job_id],
                from_stage=from_stage,
                release_t=job_2_release_t[job_id] if job_2_release_t else None,
            )

    def get_schedule_by_dj_cds(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """Get schedule using DJ (Job-Sequence) with CDS sequence.

        Args:
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.

        Returns:
            The generated schedule, or None if infeasible.
        """
        best_obj_val: int | None = None
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_k = -1

        for k in range(1, self.stage_count):
            if schedule is not None:
                _schedule = schedule.deepcopy()
            else:
                _schedule = self._create_empty_schedule()
            job_sequence = self.get_cds_sequence(k)
            self.dispatch_job_sequence_by_stages(
                _schedule,
                job_sequence,
                from_stage=from_stage,
                job_2_release_t=job_2_release_t,
            )

            makespan = _schedule.makespan
            if best_obj_val is None or best_obj_val > makespan:
                best_obj_val = makespan
                best_schedule = _schedule
                best_k = k

        print(f"Best CDS schedule found with k={best_k}, makespan={best_obj_val}")
        if in_place and best_schedule is not None:
            schedule = best_schedule
            return None
        else:
            return best_schedule

    def get_schedule_by_dj_gupta(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using DJ (Job-Sequence) with Gupta sequence.

        Args:
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.

        Returns:
            The generated schedule, or None if infeasible.
        """
        _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=in_place)
        job_sequence = self.get_gupta_sequence()
        self.dispatch_job_sequence_by_stages(
            _schedule,
            job_sequence,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
        )

        if in_place:
            return None
        return _schedule

    def get_schedule_by_dj_palmer(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using DJ (Job-Sequence) with Palmer sequence.

        Args:
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.

        Returns:
            The generated schedule, or None if infeasible.
        """
        _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=in_place)
        job_sequence = self.get_palmer_sequence()
        self.dispatch_job_sequence_by_stages(
            _schedule,
            job_sequence,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
        )

        if in_place:
            return None
        return _schedule
