"""
BN2DDispatcher class for BN2D (Bottleneck-based Two-Way Dispatching) methods.
"""

import math
import random
from typing import Callable

from schore.parameters_examples import HybridFlowshopParameters
from schore.parameters_examples.parallel_shop.identical_flow.hybrid_flowshop import (
    create_instance_of_stage_subset,
)

from hybridflowshop.dispatcher.mixed import MixedDispatcher
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    StageIdType,
)

from .base import BaseDispatcher
from .bn2d_option import (
    BN2DOption,
)
from .utils import (
    dispatch_job_sequence_by_stages,
    dispatch_stages_by_job_sequence,
    reverse_even_positions,
)


class BN2DDispatcher(BaseDispatcher):
    """
    BN2D (Bottleneck-based Two-Way Dispatching) dispatcher.

    This dispatcher implements the BN2D heuristic for hybrid flow shop scheduling,
    which focuses on the bottleneck stage and uses two-way dispatching:
    1. Dispatch later stages (after bottleneck) based on bottleneck completion times
    2. Dispatch former stages (before bottleneck) using reversed instance with release times
    """

    def __init__(
        self,
        instance: HybridFlowshopParameters,
    ):
        super().__init__(instance)
        self.instance = instance
        self.mixed_dispatcher = MixedDispatcher(instance)

    def _get_bottleneck_stage_schedule_heuristic(
        self,
        bottleneck_stage_id: StageIdType,
        option: BN2DOption,
        instance: HybridFlowshopParameters | None = None,
        gantt_draw_func: Callable | None = None,
    ) -> tuple[HybridFlowshopLiteSchedule, int]:
        """Get schedule for the bottleneck stage using BN2D heuristic.

        Args:
            bottleneck_stage_id: The bottleneck stage ID.
            option: Heuristic configuration options.
            instance: Optional instance override.
            gantt_draw_func: Optional function to draw Gantt chart for the bottleneck stage.

        Returns:
            Tuple of (schedule, makespan).
        """
        if instance is None:
            instance = self.instance
            job_2_stage_2_p = self.job_2_stage_2_p
            stage_2_job_2_p = self.stage_2_job_2_p
        else:
            job_2_stage_2_p = instance.job_2_stage_2_p_map
            stage_2_job_2_p = instance.stage_2_job_2_p_map

        # From hybrid flow shop problem define parallel machine scheduling problem for the bottleneck stage
        bottleneck_stage_index = instance.stage_id_list.index(bottleneck_stage_id)
        before_stage_id_list = instance.stage_id_list[:bottleneck_stage_index]
        after_stage_id_list = instance.stage_id_list[bottleneck_stage_index + 1 :]
        before_stage_cnt = len(before_stage_id_list)
        after_stage_cnt = len(after_stage_id_list)

        r_dict: dict[JobIdType, int] = {
            j: sum(job_2_stage_2_p[j][s] for s in before_stage_id_list)
            for j in instance.job_id_list
        }
        if option.normalize_by_stage_cnt and before_stage_cnt > 0:
            r_dict = {j: math.ceil(r / before_stage_cnt) for j, r in r_dict.items()}

        p_dict: dict[JobIdType, int] = stage_2_job_2_p[bottleneck_stage_id]

        tr_dict: dict[JobIdType, int] = {
            j: sum(job_2_stage_2_p[j][s] for s in after_stage_id_list)
            for j in instance.job_id_list
        }
        if option.normalize_by_stage_cnt and after_stage_cnt > 0:
            tr_dict = {j: math.ceil(tr / after_stage_cnt) for j, tr in tr_dict.items()}

        machine_cnt = len(instance.stage_2_machines_map[bottleneck_stage_id])
        job_cnt = instance.job_count

        left_cap_op_cnt = 0
        if option.left_cap_multiplier is not None:
            left_cap_op_cnt = option.left_cap_multiplier * machine_cnt
        elif option.left_cap_portion is not None:
            left_cap_op_cnt = int(option.left_cap_portion * job_cnt)

        right_cap_op_cnt = 0
        if option.right_cap_multiplier is not None:
            right_cap_op_cnt = option.right_cap_multiplier * machine_cnt
        elif option.right_cap_portion is not None:
            right_cap_op_cnt = int(option.right_cap_portion * job_cnt)

        left_cap_job_id_list: list[str] = []
        right_cap_job_id_list: list[str] = []

        if left_cap_op_cnt > 0 or right_cap_op_cnt > 0:
            from hybridflowshop.select_and_assign import solve_selection_problem

            result = solve_selection_problem(
                jobs=instance.job_id_list,
                r=r_dict,
                t=tr_dict,
                K_L=left_cap_op_cnt,
                K_R=right_cap_op_cnt,
            )

            if result["status"] in ("OPTIMAL", "FEASIBLE"):
                left_cap_job_id_list = result["L_set"]
                # Sort by r_j in ascending order
                left_cap_job_id_list.sort(key=lambda j: r_dict[j])
                right_cap_job_id_list = result["R_set"]
                # Sort by tr_j in descending order
                right_cap_job_id_list.sort(key=lambda j: tr_dict[j], reverse=True)
            else:
                # Fallback to greedy selection
                if left_cap_op_cnt > 0:
                    sorted_by_r = sorted(r_dict.items(), key=lambda x: x[1])
                    left_cap_job_id_list = [j for j, _ in sorted_by_r[:left_cap_op_cnt]]

                if right_cap_op_cnt > 0:
                    sorted_by_tr = sorted(tr_dict.items(), key=lambda x: x[1])
                    # Exclude those in head_job_id_list
                    sorted_by_tr = [
                        (j, t) for j, t in sorted_by_tr if j not in left_cap_job_id_list
                    ]
                    right_cap_job_id_list = [
                        j for j, _ in sorted_by_tr[:right_cap_op_cnt]
                    ]
            for j in left_cap_job_id_list:
                self.logger.debug(
                    f"Left cap job {j}: r={r_dict[j]}, p={p_dict[j]}, tr={tr_dict[j]}"
                )
            for j in right_cap_job_id_list:
                self.logger.debug(
                    f"Right cap job {j}: r={r_dict[j]}, p={p_dict[j]}, tr={tr_dict[j]}"
                )

        # Update mid_job_id_list to only include jobs that are not in head or tail job lists
        mid_job_id_list = [
            j
            for j in instance.job_id_list
            if j not in left_cap_job_id_list and j not in right_cap_job_id_list
        ]

        if option.randomize_mid_all:
            random.shuffle(mid_job_id_list)
        else:
            # Sort mid jobs by (r_j - tr_j, tie-break by original job index)
            mid_job_id_list.sort(
                key=lambda j: (r_dict[j] - tr_dict[j], instance.job_id_list.index(j))
            )
            if option.reverse_mid_even:
                reverse_even_positions(mid_job_id_list, in_place=True)
            elif option.reverse_mid_all:
                mid_job_id_list.reverse()

        sorted_j_list = left_cap_job_id_list + mid_job_id_list + right_cap_job_id_list

        dispatched_schedule = self._create_empty_schedule(instance=instance)
        dispatched_schedule.dispatch_stage_by_jobs(
            bottleneck_stage_id,
            sorted_j_list,
            p_dict,
            job_2_release=r_dict,
        )

        end_time_dict = dispatched_schedule.get_jik_2_end_time_map()
        makespan = 0
        for op, end_time in end_time_dict.items():
            last_stage_completion = end_time + tr_dict[op[0]]
            if last_stage_completion > makespan:
                makespan = last_stage_completion

        self.logger.debug(f"Bottleneck parallel MC: partial_obj={makespan}")
        if gantt_draw_func is not None:
            gantt_draw_func(dispatched_schedule, force_start=0, force_end=makespan)
        return dispatched_schedule, makespan

    def _create_reversed_instance_for_former_stages(
        self,
        before_stage_list: list[StageIdType],
        job_2_bottleneck_start_time: dict[JobIdType, int],
        bcmax: int,
    ) -> tuple[HybridFlowshopParameters, dict[JobIdType, int]]:
        """Create a reverse scheduling instance for former stages.

        Args:
            before_stage_list (list[StageIdType]): List of stage IDs for the former stages
                (i.e. stages before the bottleneck stage).
            bottleneck_stage_start_time_map (dict[JobIdType, int]): job ID -> start time at
                bottleneck stage
            bcmax (int): The makespan of the bottleneck stage schedule.

        Returns:
            Tuple of (reversed instance, job -> release time mapping).
        """
        stage_list = list(reversed(before_stage_list))
        job_2_release_t: dict[JobIdType, int] = {}
        for j in self.job_id_list:
            bottleneck_start_time = job_2_bottleneck_start_time[j]
            job_2_release_t[j] = bcmax - bottleneck_start_time

        # Create a new instance for former stages with reversed time
        reversed_instance = create_instance_of_stage_subset(
            self.instance,
            stage_list,
        )

        return reversed_instance, job_2_release_t

    def _dispatch_former_stages(
        self,
        instance_for_former_stages: HybridFlowshopParameters,
        job_2_release_t: dict[JobIdType, int],
        get_mixed_schedule: bool = False,
        machine_then_job: bool = False,
    ) -> HybridFlowshopLiteSchedule:
        sorted_j_list = sorted(
            instance_for_former_stages.job_id_list,
            key=lambda j: (
                job_2_release_t[j],
                instance_for_former_stages.job_id_list.index(j),
            ),
        )
        if get_mixed_schedule:
            dispatcher = MixedDispatcher(instance_for_former_stages)
            schedule = dispatcher.get_best_mixed_schedule_by_sequence(
                sorted_j_list,
                job_2_release_t=job_2_release_t,
                machine_then_job=machine_then_job,
                draw_gantt_per_step=False,
            )
            if schedule is None:
                raise ValueError("Failed to get schedule for former stages")
            return schedule

        ds_schedule = self._create_empty_schedule(instance_for_former_stages)
        stage_2_job_2_p_sub = {
            stage_id: {j: self.job_2_stage_2_p[j][stage_id] for j in sorted_j_list}
            for stage_id in instance_for_former_stages.stage_id_list
        }
        dispatch_stages_by_job_sequence(
            ds_schedule,
            sorted_j_list,
            stage_2_job_2_p_sub,
            job_2_release_t=job_2_release_t,
        )

        dj_schedule = self._create_empty_schedule(instance_for_former_stages)
        job_2_stage_2_p_sub = {
            j: {
                stage_id: self.job_2_stage_2_p[j][stage_id]
                for stage_id in instance_for_former_stages.stage_id_list
            }
            for j in sorted_j_list
        }
        dispatch_job_sequence_by_stages(
            dj_schedule,
            sorted_j_list,
            job_2_stage_2_p_sub,
            job_2_release_t=job_2_release_t,
        )

        if ds_schedule.makespan < dj_schedule.makespan:
            return ds_schedule
        return dj_schedule

    def _get_bottleneck_stage(self) -> StageIdType:
        """Identify the bottleneck stage.

        Returns:
            StageIdType: The bottleneck stage ID.
        """
        stage_id_2_total_p: dict[StageIdType, int] = {}
        for stage_id in self.stage_id_list:
            stage_id_2_total_p[stage_id] = sum(
                self.stage_2_job_2_p[stage_id][job_id] for job_id in self.job_id_list
            )
        stage_id_2_bottleneck_index: dict[StageIdType, float] = {
            stage_id: total_p / len(self.machines_per_stage[stage_id])
            for stage_id, total_p in stage_id_2_total_p.items()
        }
        bottleneck_stage_id = max(
            stage_id_2_bottleneck_index, key=lambda s: stage_id_2_bottleneck_index[s]
        )
        return bottleneck_stage_id

    def _get_schedule_from_bottleneck_stage(
        self,
        bottleneck_stage_id: str,
        option: BN2DOption,
        gantt_draw_func: Callable | None = None,
    ) -> HybridFlowshopLiteSchedule:
        """Schedule the entire hybrid flow shop from a single bottleneck stage.

        Uses two-way dispatching:
        1. Dispatch later stages (after bottleneck)
            based on bottleneck completion times
        2. Dispatch former stages (before bottleneck)
            using reversed instance with release times

        Args:
            bottleneck_stage_id: The bottleneck stage ID to schedule from
            option: The bottleneck stage schedule heuristic option
            draw_gantt: Whether to draw Gantt chart for the bottleneck stage only

        Returns:
            Complete schedule for all stages
        """
        # Get bottleneck schedule and makespan
        bottleneck_schedule, bcmax = self._get_bottleneck_stage_schedule_heuristic(
            bottleneck_stage_id, option, gantt_draw_func=gantt_draw_func
        )

        # Get later stages (after bottleneck)
        bottleneck_stage_index = self.stage_id_list.index(bottleneck_stage_id)
        later_stage_list = self.stage_id_list[bottleneck_stage_index + 1 :]
        if later_stage_list:
            self.logger.debug(f"Later stages: {later_stage_list}")
            # Sort jobs by end time at bottleneck stage
            job_2_bottleneck_end_time = self._get_job_2_end_time_map(
                bottleneck_schedule, bottleneck_stage_id
            )
            sorted_j_list = sorted(
                self.job_id_list,
                key=lambda j: (
                    job_2_bottleneck_end_time.get(j, 0),
                    self.job_id_list.index(j),
                ),
            )

            if option.mixed_schedule_for_later_stages:
                # Use mixed schedule for later stages
                schedule = self.mixed_dispatcher.get_best_mixed_schedule_by_sequence(
                    sorted_j_list,
                    schedule=bottleneck_schedule.deepcopy(),
                    from_stage=later_stage_list[0],
                    job_2_release_t=job_2_bottleneck_end_time,
                    machine_then_job=option.machine_then_job,
                    draw_gantt_per_step=False,
                )
                if schedule is None:
                    raise ValueError("Failed to get mixed schedule for later stages")
            else:
                # Try both DS and DJ for later stages, pick better
                later_ds_schedule = bottleneck_schedule.deepcopy()
                for stage_id in later_stage_list:
                    later_ds_schedule.dispatch_stage_by_jobs(
                        stage_id, sorted_j_list, self.stage_2_job_2_p[stage_id]
                    )

                later_dj_schedule = bottleneck_schedule.deepcopy()
                for job_id in sorted_j_list:
                    later_dj_schedule.dispatch_job_by_stages(
                        job_id,
                        self.job_2_stage_2_p[job_id],
                        from_stage=later_stage_list[0],
                    )

                if later_ds_schedule.makespan <= later_dj_schedule.makespan:
                    schedule = later_ds_schedule
                else:
                    schedule = later_dj_schedule
        else:
            schedule = bottleneck_schedule.deepcopy()
        if gantt_draw_func is not None:
            gantt_draw_func(schedule, force_start=0)

        # Handle former stages (before bottleneck)
        before_stage_list = self.stage_id_list[:bottleneck_stage_index]
        if before_stage_list:
            self.logger.debug(f"Before stages: {before_stage_list}")
            job_2_bottleneck_start_time = self._get_job_2_start_time_map(
                bottleneck_schedule, bottleneck_stage_id
            )
            instance_for_former_stages, job_2_release_t = (
                self._create_reversed_instance_for_former_stages(
                    before_stage_list, job_2_bottleneck_start_time, bcmax
                )
            )
            former_schedule = self._dispatch_former_stages(
                instance_for_former_stages,
                job_2_release_t,
                get_mixed_schedule=option.mixed_schedule_for_former_stages,
                machine_then_job=option.machine_then_job,
            )

            former_schedule_makespan = former_schedule.makespan
            discrepancy = former_schedule_makespan - bcmax
            self.logger.debug(
                f"Former stages makespan: {former_schedule_makespan}, "
                f"discrepancy with bottleneck schedule: {discrepancy}"
            )

            # Right-shift original schedule by discrepancy
            schedule.right_shift(discrepancy)

            # Add former stage operations (reversed time)
            former_schedule_end_time_map = former_schedule.get_jik_2_end_time_map()
            for op, end_time in former_schedule_end_time_map.items():
                job_id, stage_id, mc_id = op
                start_time = former_schedule_makespan - end_time
                duration = self.job_2_stage_2_p[job_id][stage_id]
                schedule.add_ops_times_2_mc(
                    stage_id, mc_id, job_id, start_time, start_time + duration
                )

        schedule.make_semi_active(self.stage_2_job_2_p)
        return schedule

    # Public methods

    def get_schedule_by_bn2d_all_stages(
        self, option: BN2DOption, gantt_draw_func: Callable | None = None
    ) -> HybridFlowshopLiteSchedule | None:
        """Get BN2D schedule trying all stages as bottleneck.

        Args:
            schedule: The schedule to populate.
            option: BN2DOption with all configuration parameters.

        Returns:
            HybridFlowshopLiteSchedule | None: The best schedule found, or None if infeasible.
        """
        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None

        for bottleneck_stage_id in self.stage_id_list:
            _schedule = self._get_schedule_from_bottleneck_stage(
                bottleneck_stage_id, option, gantt_draw_func=gantt_draw_func
            )
            if _schedule is None:
                continue

            makespan = _schedule.makespan
            if best_obj is None or makespan < best_obj:
                best_obj = makespan
                best_sch = _schedule

        return best_sch

    def get_schedule_by_bn2d_single_stage(
        self, option: BN2DOption, gantt_draw_func: Callable | None = None
    ) -> HybridFlowshopLiteSchedule | None:
        """Get BN2D schedule for a single bottleneck stage.

        Args:
            schedule: The schedule to populate.
            option: BN2DOption with all configuration parameters.
            gantt_draw_func: Optional function to draw Gantt chart for the bottleneck stage.

        Returns:
            HybridFlowshopLiteSchedule | None: The populated schedule, or None if infeasible.
        """
        bottleneck_stage_id = self._get_bottleneck_stage()

        return self._get_schedule_from_bottleneck_stage(
            bottleneck_stage_id, option, gantt_draw_func=gantt_draw_func
        )

    def _get_job_2_start_time_map(
        self, schedule: HybridFlowshopLiteSchedule, stage_id: str
    ) -> dict[str, int]:
        """Get job -> start time map for a specific stage from schedule."""
        result: dict[str, int] = {}
        for (job_id, sid, _), start_time in schedule.get_jik_2_start_time_map().items():
            if sid == stage_id:
                result[job_id] = start_time
        return result

    def _get_job_2_end_time_map(
        self, schedule: HybridFlowshopLiteSchedule, stage_id: str
    ) -> dict[str, int]:
        """Get job -> end time map for a specific stage from schedule."""
        result: dict[str, int] = {}
        for (job_id, sid, _), end_time in schedule.get_jik_2_end_time_map().items():
            if sid == stage_id:
                result[job_id] = end_time
        return result
