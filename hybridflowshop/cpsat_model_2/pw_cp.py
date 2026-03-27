from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import CpSolver, IntervalVar, IntVar

from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    McIdType,
    StageIdType,
)

from .cumulative import BaseModelBuilder, OperationVars
from .params import Params

JobMcType = tuple[JobIdType, McIdType]  # (job_id, mc_id)

# Parameters


@dataclass(frozen=True)
class OperationPartition:
    """
    Encapsulates the operation partition of PW-CP subproblems.

    Operations are grouped into time-fixed, profile-fixed, unfixed, and
    right-time-fixed regions around the current batch.
    """

    left_time_fixed: tuple[JobMcType, ...]
    left_profile_fixed: tuple[JobMcType, ...]
    unfixed: tuple[JobMcType, ...]
    right_profile_fixed: tuple[JobMcType, ...]
    right_time_fixed: tuple[JobMcType, ...]

    @property
    def all_operations(self) -> tuple[JobMcType, ...]:
        """Return all operations in the partition."""
        return (
            self.left_time_fixed
            + self.left_profile_fixed
            + self.unfixed
            + self.right_profile_fixed
            + self.right_time_fixed
        )

    @property
    def time_fixed(self) -> tuple[JobMcType, ...]:
        """Return all operations that should have fixed start times in CP model."""
        return self.left_time_fixed + self.right_time_fixed

    @property
    def non_time_fixed(self) -> tuple[JobMcType, ...]:
        """Return all operations except time-fixed ones."""
        return self.left_profile_fixed + self.unfixed + self.right_profile_fixed

    @property
    def profile_fixed(self) -> tuple[JobMcType, ...]:
        """Return all operations that keep precedence but not start times fixed."""
        return self.left_profile_fixed + self.right_profile_fixed

    @property
    def non_profile_fixed(self) -> tuple[JobMcType, ...]:
        """Return all operations except profile-fixed ones."""
        return self.left_time_fixed + self.unfixed + self.right_time_fixed

    @property
    def slack_occupying(self) -> tuple[JobMcType, ...]:
        """Return all operations except right-time-fixed ones."""
        return (
            self.left_time_fixed
            + self.left_profile_fixed
            + self.unfixed
            + self.right_profile_fixed
        )

    @property
    def non_left_time_fixed(self) -> tuple[JobMcType, ...]:
        """Return all operations except left-time-fixed ones."""
        return (
            self.left_profile_fixed
            + self.unfixed
            + self.right_profile_fixed
            + self.right_time_fixed
        )

    @property
    def non_time_fixed_jobs(self) -> frozenset[JobIdType]:
        """Return the set of job IDs that have at least one non-time-fixed operation."""
        return frozenset(job_id for job_id, _ in self.non_time_fixed)

    @property
    def unfixed_jobs(self) -> frozenset[JobIdType]:
        """Return the set of job IDs that have at least one unfixed operation."""
        return frozenset(job_id for job_id, _ in self.unfixed)

    def promote_job_contained_ops(
        self, promoted_job_id_set: set[JobIdType]
    ) -> OperationPartition:
        """Promote profile-fixed operations of unfixed jobs into the unfixed set."""
        if not promoted_job_id_set:
            return self

        promoted_left = tuple(
            sorted(
                op for op in self.left_profile_fixed if op[0] not in promoted_job_id_set
            )
        )
        promoted_right = tuple(
            sorted(
                op
                for op in self.right_profile_fixed
                if op[0] not in promoted_job_id_set
            )
        )
        promoted_unfixed = tuple(
            sorted(
                self.unfixed
                + tuple(
                    op
                    for op in self.left_profile_fixed + self.right_profile_fixed
                    if op[0] in promoted_job_id_set
                )
            )
        )
        return OperationPartition(
            left_time_fixed=self.left_time_fixed,
            left_profile_fixed=promoted_left,
            unfixed=promoted_unfixed,
            right_profile_fixed=promoted_right,
            right_time_fixed=self.right_time_fixed,
        )


# Variables


@dataclass
class DummyBarVars:
    """
    Dummy bar variables representing machine-level boundaries.

    Dummy bars are used to model time-fixed operation boundaries:
    - Left bar: fully fixed interval
        - Represents the region occupied by left-time-fixed operations
    - Right bar: variable start interval (start = right_boundary - common_spacing)
        - Represents the region from right boundary to horizon, with common spacing gap
    """

    left_bar_interval: dict[StageIdType, dict[McIdType, IntervalVar]]
    """
    stage_id -> mc_id -> IntervalVar for left boundary bar
    Start, length, & end are constants
    """

    left_bar_end: dict[StageIdType, dict[McIdType, int]]
    """
    stage_id -> mc_id -> end time for left boundary bar
    This is the fixed end time representing the boundary of left-time-fixed operations
    """

    right_bar_interval: dict[StageIdType, dict[McIdType, IntervalVar]]
    """
    stage_id -> mc_id -> IntervalVar for right boundary bar
    Start is variable (= right_boundary_time - common_spacing)
    Length is variable (horizon - right_boundary + common_spacing)
    End is constant (horizon)
    """

    right_bar_init_start: dict[StageIdType, dict[McIdType, int]]
    """
    stage_id -> mc_id -> initial start time for right boundary bar
    """

    common_spacing: IntVar
    """
    Common spacing variable (shared across all machines)
    This is the gap that will be created before right boundary
    """


@dataclass
class PwCpVars(OperationVars, DummyBarVars):
    makespan: IntVar | None = None


class PwCpModelBuilder(BaseModelBuilder):
    @staticmethod
    def make_non_time_fixed_ops_vars(
        mdl: CustomCpModel,
        params: Params,
        horizon: int,
        stage_2_partition: Mapping[str, OperationPartition],
        tighten_ranges: bool = False,
    ) -> OperationVars:
        """Create CP variables for unfixed operations."""
        op_start: dict[tuple[str, str], IntVar] = {}
        op_end: dict[tuple[str, str], IntVar] = {}
        op_intvl: dict[tuple[str, str], IntervalVar] = {}

        if tighten_ranges:
            j_i_2_head = BaseModelBuilder._compute_head(params)
            j_i_2_tail = BaseModelBuilder._compute_tail(params)
        else:
            j_i_2_head = {(j, i): 0 for j in params.j_list for i in params.i_list}
            j_i_2_tail = {(j, i): 0 for j in params.j_list for i in params.i_list}

        for i in params.i_list:
            partition = stage_2_partition[i]
            for j, _ in partition.non_time_fixed:
                p = params.p[j, i]

                assert j_i_2_head[j, i] <= horizon - j_i_2_tail[j, i] - p
                assert j_i_2_head[j, i] + p <= horizon - j_i_2_tail[j, i]

                start_var = mdl.new_int_var(
                    j_i_2_head[j, i], horizon - j_i_2_tail[j, i] - p, f"start_{j}_{i}"
                )
                end_var = mdl.new_int_var(
                    j_i_2_head[j, i] + p, horizon - j_i_2_tail[j, i], f"end_{j}_{i}"
                )
                interval_var = mdl.new_interval_var(
                    start_var, p, end_var, f"interval_{j}_{i}"
                )

                op_start[j, i] = start_var
                op_end[j, i] = end_var
                op_intvl[j, i] = interval_var

        return OperationVars(op_start=op_start, op_end=op_end, op_intvl=op_intvl)

    @staticmethod
    def make_dummy_bar_vars(
        mdl: CustomCpModel,
        params: Params,
        horizon: int,
        stage_2_mc_2_window: dict[StageIdType, dict[McIdType, tuple[int, int]]],
    ) -> DummyBarVars:
        l_dummy_intervals: dict[StageIdType, dict[McIdType, IntervalVar]] = {
            stage_id: {} for stage_id in params.i_list
        }
        l_dummy_end: dict[StageIdType, dict[McIdType, int]] = {
            stage_id: {} for stage_id in params.i_list
        }
        r_dummy_intervals: dict[StageIdType, dict[McIdType, IntervalVar]] = {
            stage_id: {} for stage_id in params.i_list
        }
        r_dummy_init_start: dict[StageIdType, dict[McIdType, int]] = {
            stage_id: {} for stage_id in params.i_list
        }

        # Common spacing variable (shared across all machines)
        common_spacing = mdl.new_int_var(0, horizon, "common_spacing")

        for stage_id, mc_2_window in stage_2_mc_2_window.items():
            for mc_id, (left_boundary_time, right_boundary_time) in mc_2_window.items():
                # Left dummy: fully fixed interval from 0 to left_boundary_time
                if left_boundary_time > 0:
                    l_dummy_interval = mdl.new_interval_var(
                        0,
                        left_boundary_time,
                        left_boundary_time,
                        f"l_dummy_{stage_id}_{mc_id}",
                    )
                    l_dummy_intervals[stage_id][mc_id] = l_dummy_interval
                    l_dummy_end[stage_id][mc_id] = left_boundary_time

                # Right dummy: end fixed as horizon
                r_dummy_end = horizon
                # Length is declared as horizon - right_boundary_time + common_spacing
                r_dummy_lth = horizon - right_boundary_time + common_spacing
                # Start is variable with range [0, right_boundary_time]
                r_dummy_start = mdl.new_int_var(
                    0, right_boundary_time, f"r_dummy_start_{stage_id}_{mc_id}"
                )
                r_dummy_interval = mdl.new_interval_var(
                    r_dummy_start,
                    r_dummy_lth,
                    r_dummy_end,
                    f"r_dummy_{stage_id}_{mc_id}",
                )
                r_dummy_intervals[stage_id][mc_id] = r_dummy_interval
                r_dummy_init_start[stage_id][mc_id] = right_boundary_time

        return DummyBarVars(
            left_bar_interval=l_dummy_intervals,
            left_bar_end=l_dummy_end,
            right_bar_interval=r_dummy_intervals,
            right_bar_init_start=r_dummy_init_start,
            common_spacing=common_spacing,
        )

    @staticmethod
    def add_non_fixed_job_precedence_constraints(
        mdl: CustomCpModel,
        params: Params,
        stage_2_partition: Mapping[StageIdType, OperationPartition],
        right_justified_schedule: HybridFlowshopLiteSchedule,
        stage_2_mc_2_window: dict[StageIdType, dict[McIdType, tuple[int, int]]],
        variables: OperationVars,
    ) -> None:
        """
        Add explicit precedence constraints for non-time-fixed jobs.

        Ensures flowshop structure is preserved even with dummy bar simplification.
        Only adds constraints for jobs in non_time_fixed_operations.
        """
        consecutive_stage_pairs = list(zip(params.i_list[:-1], params.i_list[1:]))

        for i, next_i in consecutive_stage_pairs:
            i_partition = stage_2_partition[i]
            i_non_time_fixed_job_set = set(j for j, _ in i_partition.non_time_fixed)
            next_i_partition = stage_2_partition[next_i]

            # Invariant: each job has at most one operation in non_time_fixed_operations,
            # so we can directly compare job sets
            next_i_non_time_fixed_job_set = set(
                j for j, _ in next_i_partition.non_time_fixed
            )
            assert len(next_i_non_time_fixed_job_set) == len(
                next_i_partition.non_time_fixed
            ), (
                "All jobs in next_i_partition.non_time_fixed_operations have only one operation"
            )

            next_i_est = min(
                window[0] for window in stage_2_mc_2_window[next_i].values()
            )
            for op in next_i_partition.non_time_fixed:
                j = op[0]
                if j in i_non_time_fixed_job_set:
                    # Add precedence constraint for non-time-fixed operation pair:
                    # op in stage i must end before op in stage next_i starts
                    mdl.add(variables.op_end[j, i] <= variables.op_start[j, next_i])
                else:
                    # Add start time lower bound constraint for jobs
                    # that are time-fixed in i & non-time-fixed in next_i
                    i_end_time = right_justified_schedule.get_job_end_time(i, j)
                    # Force only if the end time of the time-fixed operation in stage i
                    # is greater than the earliest start time in next_i,
                    # otherwise it is already guaranteed by the left-time-fixed operation
                    if i_end_time > next_i_est:
                        mdl.add(variables.op_start[j, next_i] >= i_end_time)

            i_lct = max(window[1] for window in stage_2_mc_2_window[i].values())
            for op in i_partition.non_time_fixed:
                j = op[0]
                if j not in next_i_non_time_fixed_job_set:
                    # Add end time upper bound constraint for jobs
                    # that are non-time-fixed in i & time-fixed in next_i
                    next_i_start_time = (
                        right_justified_schedule.get_job_end_time(next_i, j)
                        - params.p[j, next_i]
                    )
                    # Force only if the start time of the time-fixed operation in next_i
                    # is less than the latest end time in stage i,
                    # otherwise it is already guaranteed by the right-time-fixed operation
                    if next_i_start_time < i_lct:
                        mdl.add(variables.op_end[j, i] <= next_i_start_time)

    @staticmethod
    def add_capacity_with_dummy_bar_constraints(
        mdl: CustomCpModel,
        params: Params,
        stage_2_partition: Mapping[StageIdType, OperationPartition],
        variables: OperationVars,
        dummy_bar_vars: DummyBarVars,
    ) -> None:
        """Add the stage-capacity cumulative constraints, including dummy bars."""
        i_list = params.i_list

        for i in i_list:
            partition = stage_2_partition[i]
            intervals: list[IntervalVar] = [
                variables.op_intvl[op[0], i] for op in partition.non_time_fixed
            ]
            l_dummy_intervals_map: dict[str, IntervalVar] = (
                dummy_bar_vars.left_bar_interval.get(i, {})
            )
            if l_dummy_intervals_map:
                intervals.extend(l_dummy_intervals_map.values())
            r_dummy_intervals_map: dict[str, IntervalVar] = (
                dummy_bar_vars.right_bar_interval.get(i, {})
            )
            if r_dummy_intervals_map:
                intervals.extend(r_dummy_intervals_map.values())

            demands: list[int] = [1] * len(intervals)

            capacity = len(params.M_of[i])
            mdl.add_cumulative(intervals, demands, capacity)

    @staticmethod
    def add_common_spacing_objective(
        mdl: CustomCpModel,
        dummy_bar_vars: DummyBarVars,
    ) -> IntVar:
        """
        Add objective to maximize common spacing.

        Args:
            mdl: CP-SAT model
            dummy_bar_vars: Dummy bar variables containing common_spacing

        Returns:
            The common_spacing variable being maximized
        """
        mdl.maximize(dummy_bar_vars.common_spacing)
        return dummy_bar_vars.common_spacing

    @staticmethod
    def add_makespan_objective(
        mdl: CustomCpModel,
        horizon: int,
        last_stage_id: StageIdType,
        last_stage_partition: OperationPartition,
        variables: OperationVars,
    ) -> IntVar:
        """
        Add objective to minimize makespan.

        Args:
            mdl: CP-SAT model
            last_stage_id: The ID of the last stage
            last_stage_partition: The partition of operations in the last stage
            variables: Operation variables containing end time variables

        Returns:
            An IntVar representing the makespan to be minimized
        """
        makespan_var = mdl.new_int_var(0, horizon, "makespan")
        job_completion: list[IntVar] = [
            variables.op_end[job_mc_id[0], last_stage_id]
            for job_mc_id in last_stage_partition.non_time_fixed
        ]
        mdl.add_max_equality(makespan_var, job_completion)
        mdl.minimize(makespan_var)
        return makespan_var


def extract_stage_2_job_2_time_map(
    params: Params,
    stage_2_partition: Mapping[StageIdType, OperationPartition],
    variables: OperationVars,
    solver: CpSolver,
) -> dict[StageIdType, dict[JobIdType, tuple[int, int]]]:
    return_map: dict[StageIdType, dict[JobIdType, tuple[int, int]]] = {}
    for i in params.i_list:
        partition = stage_2_partition[i]
        job_2_time_range: dict[JobIdType, tuple[int, int]] = {}
        for job_id, _ in partition.non_time_fixed:
            start_time = solver.Value(variables.op_start[job_id, i])
            end_time = solver.Value(variables.op_end[job_id, i])
            job_2_time_range[job_id] = (start_time, end_time)
        return_map[i] = job_2_time_range
    return return_map


def create_pw_cp_schedule(
    solver: CpSolver,
    params: Params,
    stage_2_partition: Mapping[StageIdType, OperationPartition],
    right_justified_schedule: HybridFlowshopLiteSchedule,
    variables: PwCpVars,
) -> HybridFlowshopLiteSchedule:
    """Create a schedule from the CP model solution."""
    stage_2_job_2_time_map = extract_stage_2_job_2_time_map(
        params, stage_2_partition, variables, solver
    )
    schedule = HybridFlowshopLiteSchedule(params.j_list, params.i_list, params.M_of)

    for i in params.i_list:
        partition = stage_2_partition[i]

        # Phase 1: Left-time-fixed operations dispatch
        # Group left_time_fixed operations by machine
        mc_2_ltf_ops: dict[McIdType, list[tuple[int, int, JobIdType]]] = {}
        """mc_id -> list of (start_time, end_time, job_id) for left-time-fixed operations"""

        for job_id, mc_id in partition.left_time_fixed:
            if mc_id not in mc_2_ltf_ops:
                mc_2_ltf_ops[mc_id] = []
            # Get end time from right_justified_schedule
            end_time = right_justified_schedule.get_job_end_time(i, job_id)
            start_time = end_time - params.p[job_id, i]
            mc_2_ltf_ops[mc_id].append((start_time, end_time, job_id))

        # Dispatch left-time-fixed operations to their designated machines
        for mc_id, ltf_ops in mc_2_ltf_ops.items():
            if not ltf_ops:
                continue
            # Sort by start time
            ltf_ops.sort(key=lambda x: x[0])
            for start_time, end_time, job_id in ltf_ops:
                schedule.add_ops_times_2_mc(i, mc_id, job_id, start_time, end_time)

            # Assertion: check min start and max end match left_bar_interval
            if (
                i in variables.left_bar_interval
                and mc_id in variables.left_bar_interval[i]
            ):
                left_interval_end = variables.left_bar_end[i][mc_id]
                # left_bar_interval ends at left_boundary_time
                assert ltf_ops[-1][1] == left_interval_end, (
                    f"Left-time-fixed assertion failed for {i}.{mc_id}: "
                    f"last op end={ltf_ops[-1][1]}, expected {left_interval_end}"
                )

        # Phase 2: Non-time-fixed operations dispatch
        # Collect Non-time-fixed operations with their CP solution times
        non_tf_ops: list[tuple[int, int, JobIdType]] = []
        for job_id, mc_id in partition.non_time_fixed:
            start_time, end_time = stage_2_job_2_time_map[i][job_id]
            non_tf_ops.append((start_time, end_time, job_id))

        if non_tf_ops:
            # Sort by start_time ascending, then end_time descending
            non_tf_ops.sort(key=lambda x: (x[0], -x[1]))

            # Dispatch using add_operation_2_stage (auto machine selection)
            for start_time, end_time, job_id in non_tf_ops:
                # Get duration from params
                duration = params.p[job_id, i]
                schedule.add_operation_2_stage(
                    i, job_id, duration, release_t=start_time
                )
                if schedule.get_job_end_time(i, job_id) != end_time:
                    raise AssertionError(
                        f"Non-time-fixed operation dispatch assertion failed for {i}.{job_id}: "
                        f"scheduled end={schedule.get_job_end_time(i, job_id)}, expected {end_time}"
                    )

        # Phase 3: Right-time-fixed operations dispatch
        if i in variables.right_bar_init_start:
            mc_2_rtf_ops: dict[McIdType, list[tuple[int, int, JobIdType]]] = {}
            """mc_id -> list of (start_time, end_time, job_id) for right-time-fixed operations"""

            for job_id, mc_id in partition.right_time_fixed:
                if mc_id not in mc_2_rtf_ops:
                    mc_2_rtf_ops[mc_id] = []
                # Get end time from right_justified_schedule
                end_time = right_justified_schedule.get_job_end_time(i, job_id)
                start_time = end_time - params.p[job_id, i]
                mc_2_rtf_ops[mc_id].append((start_time, end_time, job_id))

            # Sort operations within each machine by start time
            for mc_id in mc_2_rtf_ops:
                mc_2_rtf_ops[mc_id].sort(key=lambda x: x[0])

            # Sort right_bar_init_start by initial start time (ascending)
            # to determine dispatch order
            mc_2_right_bar_init_start = variables.right_bar_init_start[i]
            sorted_mcs_by_start = sorted(
                mc_2_right_bar_init_start.keys(),
                key=lambda mc: mc_2_right_bar_init_start[mc],
            )

            # Track which machines have received right-time-fixed operations
            right_fixed_dispatched: set[McIdType] = set()

            for source_mc in sorted_mcs_by_start:
                # Get the right-time-fixed operation from the source machine's definition
                if source_mc not in mc_2_rtf_ops:
                    continue

                # Find the target machine with minimum latest end time among those not yet dispatched
                target_mc = None
                min_latest_end = float("inf")

                for candidate_mc in schedule.machines_per_stage[i]:
                    if candidate_mc in right_fixed_dispatched:
                        continue
                    latest_end = schedule.get_machine_latest_end_time(i, candidate_mc)
                    if latest_end < min_latest_end:
                        min_latest_end = latest_end
                        target_mc = candidate_mc

                if target_mc is None:
                    continue

                # Dispatch right-time-fixed operation to the selected target machine
                rtf_ops = mc_2_rtf_ops[source_mc]
                for start_time, end_time, job_id in rtf_ops:
                    schedule.add_ops_times_2_mc(
                        i, target_mc, job_id, start_time, end_time
                    )
                    if schedule.get_job_end_time(i, job_id) != end_time:
                        raise AssertionError(
                            f"Right-time-fixed operation dispatch assertion failed for {i}.{job_id}: "
                            f"scheduled end={schedule.get_job_end_time(i, job_id)}, expected {end_time}"
                        )
                right_fixed_dispatched.add(target_mc)

    return schedule
