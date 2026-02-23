"""
MixedDispatcher class for mixed dispatch methods with head/tail concept.
"""

import math
from typing import Callable, Sequence

from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    from_job_sequence_get_schedule_mixed,
)

from .base import BaseDispatcher


class MixedDispatcher(BaseDispatcher):
    """
    Mixed dispatch methods with head/tail concept.

    The mixed dispatch strategy uses the concept of "head" (priority jobs)
    and "tail" (non-priority jobs). The head jobs are dispatched through
    all remaining stages using dispatch_job_by_stages, while the tail jobs
    are scheduled using dispatch_stage_by_jobs with priority rule.
    """

    def get_schedule_by_cds(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        head_for_all_stages: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
        draw_gantt_per_step: bool = False,
        get_file_path_for_subroutine: Callable | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using mixed dispatch with CDS sequence.

        Args:
            schedule: If provided and in_place=True, modify directly. Otherwise copy.
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.
            head_for_all_stages: If True, apply head to all stages.
                                If False, only apply head to the first stage.
            from_stage: Optional stage to start dispatching from.
            job_2_release_t: Optional mapping from job ID to release time.
            draw_gantt_per_step: If True, draw Gantt chart for each step of dispatching.
            get_file_path_for_subroutine: Optional callable to get file path for subroutine.

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
            dispatched_schedule = self.get_best_mixed_schedule_by_sequence(
                job_sequence,
                schedule=_schedule,
                from_stage=from_stage,
                job_2_release_t=job_2_release_t,
                head_for_all_stages=head_for_all_stages,
                draw_gantt_per_step=draw_gantt_per_step,
                get_file_path_for_subroutine=get_file_path_for_subroutine,
            )
            if dispatched_schedule is None:
                continue

            makespan = dispatched_schedule.makespan
            if best_obj_val is None or best_obj_val > makespan:
                best_obj_val = makespan
                best_schedule = dispatched_schedule
                best_k = k

        print(f"Best CDS schedule found with k={best_k}, makespan={best_obj_val}")
        if in_place and best_schedule is not None:
            schedule = best_schedule
            return None
        else:
            return best_schedule

    def get_schedule_by_gupta(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        head_for_all_stages: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
        draw_gantt_per_step: bool = False,
        get_file_path_for_subroutine: Callable | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using mixed dispatch with Gupta sequence.

        Args:
            schedule: If provided and in_place=True, modify directly. Otherwise copy.
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.
            head_for_all_stages: If True, apply head to all stages.
                                If False, only apply head to the first stage.
            from_stage: Optional stage to start dispatching from.
            job_2_release_t: Optional mapping from job ID to release time.
            draw_gantt_per_step: If True, draw Gantt chart for each step of dispatching.
            get_file_path_for_subroutine: Optional callable to get file path for subroutine.

        Returns:
            The generated schedule, or None if infeasible.
        """
        return self.get_best_mixed_schedule_by_sequence(
            self.get_gupta_sequence(),
            schedule=schedule,
            head_for_all_stages=head_for_all_stages,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=get_file_path_for_subroutine,
        )

    def get_schedule_by_palmer(
        self,
        schedule: HybridFlowshopLiteSchedule | None = None,
        in_place: bool = False,
        head_for_all_stages: bool = False,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
        draw_gantt_per_step: bool = False,
        get_file_path_for_subroutine: Callable | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get schedule using mixed dispatch with Palmer sequence.

        Args:
            schedule: If provided and in_place=True, modify directly. Otherwise copy.
            in_place: If True, modify the incumbent schedule directly.
                     If False, work on a deepcopy and return new schedule.
            head_for_all_stages: If True, apply head to all stages.
                                If False, only apply head to the first stage.
            from_stage: Optional stage to start dispatching from.
            job_2_release_t: Optional mapping from job ID to release time.
            draw_gantt_per_step: If True, draw Gantt chart for each step of dispatching.
            get_file_path_for_subroutine: Optional callable to get file path for subroutine.

        Returns:
            The generated schedule, or None if infeasible.
        """
        return self.get_best_mixed_schedule_by_sequence(
            self.get_palmer_sequence(),
            schedule=schedule,
            head_for_all_stages=head_for_all_stages,
            from_stage=from_stage,
            job_2_release_t=job_2_release_t,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=get_file_path_for_subroutine,
        )

    def _get_np_candidates(self) -> list[int]:
        """
        Generate candidate values for the number of priority jobs (np) for mixed dispatch.

        Generates a sequence of decreasing np values by halving (ceiling) starting from
        the total job count, ending with 0.

        Returns:
            List of np candidates in descending order.
        """
        np = self.job_count
        np_list = [np]
        while np > 1:
            np = math.ceil(np / 2)
            np_list.append(np)
        np_list.append(0)
        return np_list

    def get_best_mixed_schedule_by_sequence(
        self,
        job_sequence: Sequence[str],
        schedule: HybridFlowshopLiteSchedule | None = None,
        from_stage: str | None = None,
        job_2_release_t: dict[str, int] | None = None,
        head_for_all_stages: bool = False,
        draw_gantt_per_step: bool = False,
        get_file_path_for_subroutine: Callable | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        """
        Get best mixed schedule by trying multiple np (number of priority jobs) values.

        Args:
            job_sequence: The job sequence to use for dispatching.
            schedule: If provided and in_place=True, modify directly. Otherwise copy.
            head_for_all_stages: If True, apply head to all stages.
                                If False, only apply head to the first stage.
            from_stage: Optional stage to start dispatching from.
            job_2_release_t: Optional mapping from job ID to release time.

        Returns:
            The best generated schedule, or None if infeasible.
        """
        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None

        np_list = self._get_np_candidates()
        np_2_stage_2_head: dict[int, dict[str, int]] = {}

        for np in np_list:
            if head_for_all_stages:
                np_2_stage_2_head[np] = {
                    stage_id: np for stage_id in self.stage_id_list
                }
            else:
                if from_stage is not None:
                    if from_stage not in self.stage_id_list:
                        raise ValueError(f"from_stage {from_stage} not in stage list")
                    np_2_stage_2_head[np] = {from_stage: np}
                else:
                    np_2_stage_2_head[np] = {self.stage_id_list[0]: np}

        for np in np_list:
            _schedule = self._prepare_schedule_for_dispatch(schedule, in_place=False)
            from_job_sequence_get_schedule_mixed(
                _schedule,
                job_sequence,
                self.stage_2_job_2_p,
                np_2_stage_2_head[np],
                from_stage=from_stage,
                job_2_release=job_2_release_t,
                draw_gantt_per_step=draw_gantt_per_step,
                get_file_path_for_subroutine=get_file_path_for_subroutine,
            )
            if _schedule is None:
                continue
            if best_obj is None or _schedule.makespan < best_obj:
                best_obj = _schedule.makespan
                best_sch = _schedule

        if schedule is not None and schedule is best_sch:
            return None
        return best_sch
