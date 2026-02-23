"""
StageDispatcher class for stage-sequence dispatch methods (DS and DWB).
"""

from typing import Sequence

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule

from .base import BaseDispatcher


class StageDispatcher(BaseDispatcher):
    """
    Stage-sequence dispatch methods (DS and DWB).

    These methods dispatch all jobs through one stage at a time,
    rather than dispatching jobs one-by-one through all stages.
    """

    def dispatch_stages_by_job_sequence(
        self,
        schedule: HybridFlowshopLiteSchedule,
        job_sequence: Sequence[str],
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> None:
        _stage_id_list = self.stage_id_list
        if from_stage is not None:
            from_stage_index = self.stage_id_list.index(from_stage)
            _stage_id_list = self.stage_id_list[from_stage_index:]
        for stage_id in _stage_id_list:
            schedule.dispatch_stage_by_jobs(
                stage_id,
                job_sequence,
                self.stage_2_job_2_p[stage_id],
                job_2_release=job_2_release_t,
            )

    def get_schedule_by_ds_cds(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using DS (Stage-Sequence) with CDS sequence.

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
            if schedule is not None and in_place:
                _schedule = schedule.deepcopy()
            else:
                _schedule = self._create_empty_schedule()
            job_sequence = self.get_cds_sequence(k)
            self.dispatch_stages_by_job_sequence(
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

    def get_schedule_by_ds_gupta(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using DS (Stage-Sequence) with Gupta sequence.

        Args:
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.

        Returns:
            The generated schedule, or None if infeasible.
        """
        _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=in_place)
        job_sequence = self.get_gupta_sequence()
        self.dispatch_stages_by_job_sequence(
            _schedule,
            job_sequence,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
        )

        if in_place:
            return None
        return _schedule

    def get_schedule_by_ds_palmer(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using DS (Stage-Sequence) with Palmer sequence.

        Args:
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.

        Returns:
            The generated schedule, or None if infeasible.
        """
        _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=in_place)
        job_sequence = self.get_palmer_sequence()
        self.dispatch_stages_by_job_sequence(
            _schedule,
            job_sequence,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
        )

        if in_place:
            return None
        return _schedule
