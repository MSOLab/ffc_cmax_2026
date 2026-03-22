from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from mbls.cpsat import CpsatSolverReport, CustomCpModel, ObjValueBoundStore
from ortools.sat.python import cp_model
from routix import ElapsedTimer
from routix.io.yaml import dump_yaml
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.cpsat_model_2.cumulative import CumulativeVars, OperationVars
from hybridflowshop.cpsat_model_2.params import Params
from hybridflowshop.cpsat_model_2.pw_cp import (
    DummyBarVars,
    JobMcType,
    OperationPartition,
    PwCpModelBuilder,
    PwCpVars,
    create_pw_cp_schedule,
)
from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    McIdType,
    OperationType,
    StageIdType,
)

HighlightOperationType = tuple[JobIdType, StageIdType]
StageBoundaryProfile = dict[StageIdType, list[int | None]]
SolvedSlackIntervals = dict[StageIdType, list[tuple[JobIdType, int, int]]]


class PwCpContext(Protocol):
    """
    Minimal dependency interface.
    (HybridFlowShopCpLnsController is effectively designed to satisfy this interface.)
    """

    solver: cp_model.CpSolver

    def get_remaining_time_limit(
        self, subroutine_time_limit: float | None
    ) -> float: ...

    def solve_cp_model_2(
        self,
        mdl: CustomCpModel,
        computational_time: float,
        solver_thread_cnt: int,
        e_timer: ElapsedTimer | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        use_lns_only: bool | None = None,
        log_level_obj_value: int = logging.INFO,
        log_level_obj_bound: int = logging.INFO,
        log_search_progress: bool = False,
        last_timestamp_note: Any | None = None,
    ) -> CpsatSolverReport: ...

    def create_schedule(
        self, params: Params, variables: CumulativeVars
    ) -> HybridFlowshopLiteSchedule: ...

    def check_feasibility(self, start_time_map: dict[OperationType, int]) -> float: ...

    def get_file_path_for_subroutine(self, suffix: str) -> Path: ...


@dataclass(frozen=True)
class PwCpSubproblemSpec:
    batch_idx: int
    subproblem_idx: int
    stage_2_partition: Mapping[StageIdType, OperationPartition]
    stage_2_mc_2_window: dict[StageIdType, dict[McIdType, tuple[int, int]]]
    init_schedule: HybridFlowshopLiteSchedule
    is_last_batch: bool

    @property
    def left_time_fixed_op_set(self) -> set[JobMcType]:
        return set(
            op
            for partition in self.stage_2_partition.values()
            for op in partition.left_time_fixed
        )

    @property
    def left_profile_fixed_op_set(self) -> set[JobMcType]:
        return set(
            op
            for partition in self.stage_2_partition.values()
            for op in partition.left_profile_fixed
        )

    @property
    def unfixed_op_set(self) -> set[JobMcType]:
        return set(
            op
            for partition in self.stage_2_partition.values()
            for op in partition.unfixed
        )

    @property
    def right_profile_fixed_op_set(self) -> set[JobMcType]:
        return set(
            op
            for partition in self.stage_2_partition.values()
            for op in partition.right_profile_fixed
        )

    @property
    def right_time_fixed_op_set(self) -> set[JobMcType]:
        return set(
            op
            for partition in self.stage_2_partition.values()
            for op in partition.right_time_fixed
        )

    @property
    def is_right_time_fixed_empty(self) -> bool:
        """True if no right-time-fixed operations exist."""
        return len(self.right_time_fixed_op_set) == 0

    @property
    def time_fixed_op_set(self) -> set[JobMcType]:
        return set(
            op
            for partition in self.stage_2_partition.values()
            for op in partition.time_fixed
        )

    @property
    def profile_fixed_op_set(self) -> set[JobMcType]:
        return set(
            op
            for partition in self.stage_2_partition.values()
            for op in partition.profile_fixed
        )

    @property
    def non_time_fixed_op_count(self) -> int:
        return sum(
            len(partition.non_time_fixed)
            for partition in self.stage_2_partition.values()
        )


@dataclass(frozen=True)
class PwCpSubproblemLog:
    batch_idx: int
    subproblem_idx: int
    is_last_batch: bool
    objective_name: str
    time_limit_sec: float
    elapsed_time_sec: float
    status: str
    incumbent_makespan_before: int
    candidate_makespan: int | None
    accepted: bool
    obj_value_records: tuple[tuple[float, float], ...]
    obj_bound_records: tuple[tuple[float, float], ...]

    @property
    def improved(self) -> bool:
        if self.candidate_makespan is None:
            return False
        return self.candidate_makespan < self.incumbent_makespan_before

    @property
    def improvement_amount(self) -> int | None:
        if self.candidate_makespan is None:
            return None
        return self.incumbent_makespan_before - self.candidate_makespan


@dataclass
class PwCpRunState:
    timer: ElapsedTimer
    incumbent: HybridFlowshopLiteSchedule
    sub_obj_store: ObjValueBoundStore[int]
    subproblem_idx: int
    subproblem_logs: list[PwCpSubproblemLog]
    max_time_per_batch: float | None


@dataclass
class PwCpResult:
    schedule: HybridFlowshopLiteSchedule
    sub_obj_store: ObjValueBoundStore[int]
    last_obj_value: int
    subproblem_logs: tuple[PwCpSubproblemLog, ...]
    total_pw_cp_elapsed_sec: float
    max_time_per_batch: float | None

    def save_yaml(self, output_path: Path) -> None:
        """Saves the PW-CP result to a YAML file.

        Note: This method should only be called when debug_export=True
        to avoid unnecessary I/O operations.
        """
        solution_dict = self.sub_obj_store.to_dict()
        solution_dict.pop("obj_bound", None)

        accepted_count = sum(1 for log in self.subproblem_logs if log.accepted)
        improving_subproblem_count = sum(
            1 for log in self.subproblem_logs if log.improved
        )
        last_obj_improvement_time_sec = None
        for log in reversed(self.subproblem_logs):
            if log.improved and log.obj_value_records:
                last_obj_improvement_time_sec = log.obj_value_records[-1][0]
                break

        solution_dict["pw_cp_metadata"] = {
            "cp_sat_subproblems": [
                {
                    **asdict(log),
                    "improved": log.improved,
                    "improvement_amount": log.improvement_amount,
                }
                for log in self.subproblem_logs
            ],
            "summary": {
                "subproblem_count": len(self.subproblem_logs),
                "accepted_count": accepted_count,
                "improving_subproblem_count": improving_subproblem_count,
                "last_batch_idx": max(
                    (log.batch_idx for log in self.subproblem_logs),
                    default=None,
                ),
                "total_pw_cp_elapsed_sec": self.total_pw_cp_elapsed_sec,
                "last_obj_improvement_time_sec": last_obj_improvement_time_sec,
                "final_incumbent_makespan": self.last_obj_value,
                "max_time_per_batch": self.max_time_per_batch,
            },
        }

        dump_yaml(PwCpConstructor._normalize_for_yaml(solution_dict), output_path)


class PwCpConstructor:
    def __init__(self, ctx: PwCpContext):
        self.ctx = ctx
        self.builder = PwCpModelBuilder
        self._st: PwCpRunState | None = None

    def _require_state(self) -> PwCpRunState:
        if self._st is None:
            raise RuntimeError("PwCpConstructor.run() is not active; state is missing.")
        return self._st

    @staticmethod
    def _resolve_batch_time_limit(
        non_time_fixed_op_count: int,
        non_time_fixed_op_time_limit_multiplier: float | None,
        max_time_per_batch: float | None,
    ) -> float | None:
        if non_time_fixed_op_time_limit_multiplier is not None:
            if non_time_fixed_op_time_limit_multiplier <= 0:
                raise ValueError("non_time_fixed_op_time_limit_multiplier must be > 0")
            return non_time_fixed_op_count * non_time_fixed_op_time_limit_multiplier
        return max_time_per_batch

    def run(
        self,
        ref_schedule: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        batch_size: int = 1,
        left_profile_fixed_batch_count: int = 0,
        right_profile_fixed_batch_count: int = 0,
        enable_promotion_profile_fixed: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        non_time_fixed_op_time_limit_multiplier: float | None = None,
        max_time_per_batch: float | None = None,
        solver_thread_cnt: int | None = None,
        use_lns_only: bool = False,
        debug_export: bool = False,
        tighten_ranges: bool = False,
        error_if_infeasible: bool = False,
    ) -> PwCpResult:
        timer = ElapsedTimer()
        if solver_thread_cnt is None:
            solver_thread_cnt = 1
        if (
            non_time_fixed_op_time_limit_multiplier is not None
            and non_time_fixed_op_time_limit_multiplier <= 0
        ):
            raise ValueError("non_time_fixed_op_time_limit_multiplier must be > 0")

        sub_obj_store = ObjValueBoundStore[int]()
        sub_obj_store.obj_value_series.name = "ObjVal after PW-CP batch"
        sub_obj_store.add_obj_value(0.0, int(ref_schedule.makespan), None)
        sub_obj_store.add_last_timestamp_note(
            "initial_schedule",
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
        )

        self._st = PwCpRunState(
            timer=timer,
            incumbent=ref_schedule,
            sub_obj_store=sub_obj_store,
            subproblem_idx=0,
            subproblem_logs=[],
            max_time_per_batch=(
                None
                if non_time_fixed_op_time_limit_multiplier is not None
                else max_time_per_batch
            ),
        )

        try:
            if debug_export:
                self._draw_schedule_gantt(
                    schedule=ref_schedule,
                    output_suffix="_batch_000_initial_gantt.png",
                    highlight_op_set=None,
                    machine_list_per_stage=None,
                    force_start=None,
                    force_end=None,
                )
            initial_batches = self._build_stage_batches(
                ref_schedule, batch_size=batch_size
            )
            max_batch_cnt = self._validate_and_get_batch_count(initial_batches)
            for batch_idx in range(max_batch_cnt):
                st = self._require_state()
                current_batches = self._build_stage_batches(
                    st.incumbent, batch_size=batch_size
                )
                current_max_batch_cnt = self._validate_and_get_batch_count(
                    current_batches
                )
                if current_max_batch_cnt != max_batch_cnt:
                    raise AssertionError(
                        "PW-CP batch count changed during run: "
                        f"initial={max_batch_cnt}, current={current_max_batch_cnt}."
                    )

                # Step 1: Build partition
                stage_2_partition: dict[str, OperationPartition] = {}
                for stage_id in ref_schedule.stages:
                    current_batch = current_batches[stage_id]
                    stage_2_partition[stage_id] = self._build_operation_partition(
                        current_batch,
                        stage_id,
                        batch_idx,
                        left_profile_fixed_batch_count=left_profile_fixed_batch_count,
                        right_profile_fixed_batch_count=right_profile_fixed_batch_count,
                    )
                if enable_promotion_profile_fixed:
                    unfixed_job_set = set(
                        job_id
                        for partition in stage_2_partition.values()
                        for job_id in partition.unfixed_jobs
                    )
                    for stage_id, partition in stage_2_partition.items():
                        stage_2_partition[stage_id] = partition.promote_job_contained_ops(
                            promoted_job_id_set=unfixed_job_set
                        )

                # Determine if this is a makespan batch (no right-time-fixed ops)
                has_right_time_fixed = any(
                    len(partition.right_time_fixed) > 0
                    for partition in stage_2_partition.values()
                )
                spec = self._build_batch_spec(
                    incumbent=st.incumbent,
                    stage_2_partition=stage_2_partition,
                    stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                    batch_idx=batch_idx,
                    max_batch_cnt=max_batch_cnt,
                )

                batch_time_limit = self._resolve_batch_time_limit(
                    spec.non_time_fixed_op_count,
                    max_time_per_batch=max_time_per_batch,
                    non_time_fixed_op_time_limit_multiplier=non_time_fixed_op_time_limit_multiplier,
                )

                if not has_right_time_fixed:
                    logging.info(
                        "Processing makespan batch %d/%d: no right-time-fixed operations, "
                        "switching to makespan minimization.",
                        batch_idx + 1,
                        max_batch_cnt,
                    )
                    candidate = self._solve_makespan_batch(
                        spec=spec,
                        instance=instance,
                        stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                        profile_fix_by_machine=profile_fix_by_machine,
                        machine_precedence_stride=machine_precedence_stride,
                        max_time_per_batch=batch_time_limit,
                        solver_thread_cnt=solver_thread_cnt,
                        use_lns_only=use_lns_only,
                        tighten_ranges=tighten_ranges,
                        debug_export=debug_export,
                    )
                else:
                    candidate = self._solve_slack_batch(
                        spec=spec,
                        instance=instance,
                        stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                        profile_fix_by_machine=profile_fix_by_machine,
                        machine_precedence_stride=machine_precedence_stride,
                        max_time_per_batch=batch_time_limit,
                        solver_thread_cnt=solver_thread_cnt,
                        use_lns_only=use_lns_only,
                        tighten_ranges=tighten_ranges,
                        debug_export=debug_export,
                    )

                st.incumbent, accepted = self._accept_candidate_or_repair_incumbent(
                    candidate=candidate,
                    incumbent=st.incumbent,
                    stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                )

                if debug_export and batch_idx < max_batch_cnt:
                    self._save_solution_dict(spec, st.incumbent, accepted=accepted)

                ts = st.timer.elapsed_sec
                st.sub_obj_store.add_obj_value(ts, int(st.incumbent.makespan), None)
                st.sub_obj_store.add_last_timestamp_note(
                    f"batch={batch_idx + 1}",
                    obj_value_is_valid=True,
                    obj_bound_is_valid=False,
                )

            if error_if_infeasible:
                self.ctx.check_feasibility(
                    self._require_state().incumbent.get_jik_2_start_time_map()
                )

            return PwCpResult(
                schedule=self._require_state().incumbent,
                sub_obj_store=self._require_state().sub_obj_store,
                last_obj_value=self._require_state().incumbent.makespan,
                subproblem_logs=tuple(self._require_state().subproblem_logs),
                total_pw_cp_elapsed_sec=self._require_state().timer.elapsed_sec,
                max_time_per_batch=self._require_state().max_time_per_batch,
            )
        finally:
            self._st = None

    def _build_stage_batches(
        self,
        schedule: HybridFlowshopLiteSchedule,
        batch_size: int,
        sort_by_start_time: bool = False,
    ) -> dict[StageIdType, list[tuple[JobMcType, ...]]]:
        """Builds batches of operations for each stage.

        Operations within each stage are sorted before batching. The sorting
        criterion depends on the ``sort_by_start_time`` flag:

        - If ``True``: sorted by (start_time, end_time, machine_id, job_id).
        - If ``False``: sorted by (midpoint, start_time, machine_id, job_id),
          where midpoint = (start_time + end_time) / 2. This groups operations
          that are centered around similar time points, which can be useful when
          the optimization focus is on the middle of the processing window.

        Args:
            schedule: The current schedule containing operation timings.
            batch_size: Maximum number of operations per batch. At least 1.
            sort_by_start_time: If True, sort by start time primarily.
                If False, sort by midpoint (average of start and end) primarily.
                Defaults to False.

        Returns:
            A dictionary mapping stage_id to a list of batches, where each batch
            is a tuple of operation references (job_id, machine_id).

        Example:
            With operations:
                - A: start=0, end=100 (midpoint=50)
                - B: start=40, end=50 (midpoint=45)

            sort_by_start_time=True  -> order: A, B (by start: 0 < 40)
            sort_by_start_time=False -> order: B, A (by midpoint: 45 < 50)
        """
        _batch_size = max(1, batch_size)
        stage_2_batches: dict[StageIdType, list[tuple[JobMcType, ...]]] = {}
        for stage_id in schedule.stages:
            ops = sorted(
                (
                    (job_id, stage_id, mc_id, start_time, end_time)
                    for mc_id, start_time, end_time, job_id in schedule.iter_operations_on_stage(
                        stage_id
                    )
                ),
                key=lambda op: (
                    (op[3], op[4], op[2], op[0])
                    if sort_by_start_time
                    else ((op[3] + op[4]) / 2, op[3], op[2], op[0])
                ),
            )
            stage_2_batches[stage_id] = [
                tuple(
                    (job_id, mc_id)
                    for job_id, _, mc_id, _, _ in ops[idx : idx + _batch_size]
                )
                for idx in range(0, len(ops), _batch_size)
            ]
        return stage_2_batches

    def _build_operation_partition(
        self,
        batches: list[tuple[JobMcType, ...]],
        stage_id: StageIdType,
        current_batch_idx: int,
        left_profile_fixed_batch_count: int = 0,
        right_profile_fixed_batch_count: int = 0,
    ) -> OperationPartition:
        """
        Build operation partition based on batch indices.

        Partitioning rule:
        - left_time_fixed: sufficiently earlier batches
        - left_profile_fixed: immediately preceding batches
        - unfixed: operations from batch at current_batch_idx
        - right_profile_fixed: immediately following batches
        - right_time_fixed: sufficiently later batches

        This leverages the existing time-based sorting in _build_stage_batches,
        where batch_idx=0 contains earliest operations and higher indices contain
        later operations.
        """
        l_tf_ops: list[JobMcType] = []
        l_pf_ops: list[JobMcType] = []
        unfixed_ops: list[JobMcType] = []
        r_pf_ops: list[JobMcType] = []
        r_tf_ops: list[JobMcType] = []

        for idx, batch in enumerate(batches):
            if idx < current_batch_idx - left_profile_fixed_batch_count:
                l_tf_ops.extend(batch)
            elif idx < current_batch_idx:
                l_pf_ops.extend(batch)
            elif idx == current_batch_idx:
                unfixed_ops.extend(batch)
            elif idx <= current_batch_idx + right_profile_fixed_batch_count:
                r_pf_ops.extend(batch)
            else:
                r_tf_ops.extend(batch)

        return OperationPartition(
            left_time_fixed=tuple(sorted(l_tf_ops)),
            left_profile_fixed=tuple(sorted(l_pf_ops)),
            unfixed=tuple(sorted(unfixed_ops)),
            right_profile_fixed=tuple(sorted(r_pf_ops)),
            right_time_fixed=tuple(sorted(r_tf_ops)),
        )

    def _build_window_map(
        self,
        right_justified_schedule: HybridFlowshopLiteSchedule,
        stage_2_partition: Mapping[StageIdType, OperationPartition],
        stage_2_job_2_p_dict: dict[StageIdType, dict[JobIdType, int]],
    ) -> dict[StageIdType, dict[McIdType, tuple[int, int]]]:
        """
        Builds the window map for each machine
        based on the right-justified schedule and operation partition.

        Returns:
            dict[str, dict[str, tuple[int, int]]]: stage_id -> mc_id -> (left_boundary, right_boundary)
        """
        start_map = right_justified_schedule.get_jik_2_start_time_map()
        end_map = right_justified_schedule.get_jik_2_end_time_map()

        # Machine boundaries (nested dict: stage_id -> machine_idx -> (left, right))
        horizon = int(right_justified_schedule.makespan)
        stage_2_mc_2_window: dict[str, dict[str, tuple[int, int]]] = {}

        for stage_id in right_justified_schedule.stages:
            partition = stage_2_partition[stage_id]
            stage_2_mc_2_window[stage_id] = {}
            for mc_id in right_justified_schedule.machines_per_stage[stage_id]:
                # right_boundary: min start time of right_time_fixed ops on this machine
                stage_r_tf_ops = [
                    (job_id, stage_id, op_mc_id)
                    for job_id, op_mc_id in partition.right_time_fixed
                    if op_mc_id == mc_id
                ]
                machine_right_starts = [start_map[op] for op in stage_r_tf_ops]
                right_boundary = (
                    min(machine_right_starts) if machine_right_starts else horizon
                )

                # left_boundary: max end time of left_time_fixed ops on this machine
                stage_l_tf_ops = [
                    (job_id, stage_id, op_mc_id)
                    for job_id, op_mc_id in partition.left_time_fixed
                    if op_mc_id == mc_id
                ]
                left_boundary = (
                    max(end_map[op] for op in stage_l_tf_ops) if stage_l_tf_ops else 0
                )

                stage_2_mc_2_window[stage_id][mc_id] = (left_boundary, right_boundary)

        return stage_2_mc_2_window

    def _build_subproblem_spec(
        self,
        stage_2_partition: Mapping[str, OperationPartition],
        stage_2_mc_2_window: dict[str, dict[str, tuple[int, int]]],
        init_schedule: HybridFlowshopLiteSchedule,
        batch_idx: int,
        max_batch_cnt: int,
    ) -> PwCpSubproblemSpec:
        st = self._require_state()
        st.subproblem_idx += 1

        return PwCpSubproblemSpec(
            batch_idx=batch_idx,
            subproblem_idx=st.subproblem_idx,
            stage_2_partition=stage_2_partition,
            stage_2_mc_2_window=stage_2_mc_2_window,
            init_schedule=init_schedule,
            is_last_batch=(batch_idx == max_batch_cnt - 1),
        )

    def _build_batch_spec(
        self,
        incumbent: HybridFlowshopLiteSchedule,
        stage_2_partition: Mapping[str, OperationPartition],
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        batch_idx: int,
        max_batch_cnt: int,
    ) -> PwCpSubproblemSpec:
        # Check if right-time-fixed is empty - if so, no window needed (makespan batch)
        has_right_time_fixed = any(
            len(partition.right_time_fixed) > 0
            for partition in stage_2_partition.values()
        )

        init_schedule = incumbent.deepcopy()
        if has_right_time_fixed:
            # Slack batch: proceed with right-justification
            non_ltf_op_set: set[tuple[JobIdType, StageIdType, McIdType]] = set()
            for stage_id, partition in stage_2_partition.items():
                for job_id, mc_id in partition.non_left_time_fixed:
                    non_ltf_op_set.add((job_id, stage_id, mc_id))
            init_schedule.make_right_justified(
                stage_2_job_2_p_dict,
                operation_set=non_ltf_op_set,
            )
        else:
            # Makespan batch: no need for right-justification
            pass

        stage_2_mc_2_window = self._build_window_map(
            init_schedule,
            stage_2_partition,
            stage_2_job_2_p_dict,
        )
        return self._build_subproblem_spec(
            stage_2_partition=stage_2_partition,
            stage_2_mc_2_window=stage_2_mc_2_window,
            init_schedule=init_schedule,
            batch_idx=batch_idx,
            max_batch_cnt=max_batch_cnt,
        )

    @staticmethod
    def _normalize_for_yaml(value):
        if isinstance(value, dict):
            return {k: PwCpConstructor._normalize_for_yaml(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PwCpConstructor._normalize_for_yaml(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value

    @staticmethod
    def _build_highlight_ops(
        stage_2_partition: Mapping[str, OperationPartition],
    ) -> list[HighlightOperationType]:
        highlight_ops: list[HighlightOperationType] = []
        seen: set[HighlightOperationType] = set()
        for stage_id, partition in stage_2_partition.items():
            for job_id, mc_id in partition.unfixed:
                op_ref = (job_id, stage_id)
                if op_ref not in seen:
                    seen.add(op_ref)
                    highlight_ops.append(op_ref)
        return highlight_ops

    def _prepare_pw_cp_model(
        self,
        spec: PwCpSubproblemSpec,
        instance: HybridFlowshopParameters,
        profile_fix_by_machine: bool,
        machine_precedence_stride: int,
        tighten_ranges: bool = False,
    ) -> tuple[CustomCpModel, Params, PwCpVars]:
        horizon = spec.init_schedule.makespan
        mdl = CustomCpModel()

        # Parameters
        params: Params = self.builder.make_params(instance)

        # Variables
        non_time_fixed_vars: OperationVars = self.builder.make_non_time_fixed_ops_vars(
            mdl,
            params,
            spec.init_schedule.makespan,
            spec.stage_2_partition,
            tighten_ranges=tighten_ranges,
        )
        dummy_bar_vars: DummyBarVars = self.builder.make_dummy_bar_vars(
            mdl, params, horizon, spec.stage_2_mc_2_window
        )
        # Constraints
        self.builder.add_non_fixed_job_precedence_constraints(
            mdl,
            params,
            spec.stage_2_partition,
            spec.init_schedule,
            spec.stage_2_mc_2_window,
            non_time_fixed_vars,
        )
        self.builder.add_capacity_with_dummy_bar_constraints(
            mdl, params, spec.stage_2_partition, non_time_fixed_vars, dummy_bar_vars
        )

        # Hints
        hint_schedule = spec.init_schedule
        ntf_op_set: set[OperationType] = set()
        for stage_id, partition in spec.stage_2_partition.items():
            for job_mc_id in partition.non_time_fixed:
                ntf_op_set.add((job_mc_id[0], stage_id, job_mc_id[1]))

        all_start_time_map = hint_schedule.get_jik_2_start_time_map()
        ntf_start_time_map = {op: all_start_time_map[op] for op in ntf_op_set}
        all_end_time_map = hint_schedule.get_jik_2_end_time_map()
        ntf_end_time_map = {op: all_end_time_map[op] for op in ntf_op_set}

        self.builder.apply_start_hints_from_start_time_map(
            mdl, params, non_time_fixed_vars, ntf_start_time_map
        )
        self.builder.apply_end_hints_from_end_time_map(
            mdl, params, non_time_fixed_vars, ntf_end_time_map
        )

        # Profile-fixed precedence constraints
        profile_fixed_op_set: set[JobMcType] = spec.profile_fixed_op_set
        if profile_fixed_op_set:
            removed_ops: set[OperationType] = set()
            for stage_id, partition in spec.stage_2_partition.items():
                for job_mc_id in partition.non_profile_fixed:
                    removed_ops.add((job_mc_id[0], stage_id, job_mc_id[1]))

            profile_fixed_schedule = spec.init_schedule.deepcopy()
            profile_fixed_schedule.remove_operations(removed_ops)
            self.builder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
                mdl,
                params,
                non_time_fixed_vars,
                profile_fixed_schedule,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            )

        # Objective
        self.builder.add_common_spacing_objective(mdl, dummy_bar_vars)

        pw_cp_vars = PwCpVars(
            op_start=non_time_fixed_vars.op_start,
            op_end=non_time_fixed_vars.op_end,
            op_intvl=non_time_fixed_vars.op_intvl,
            left_bar_interval=dummy_bar_vars.left_bar_interval,
            left_bar_end=dummy_bar_vars.left_bar_end,
            right_bar_interval=dummy_bar_vars.right_bar_interval,
            right_bar_init_start=dummy_bar_vars.right_bar_init_start,
            common_spacing=dummy_bar_vars.common_spacing,
        )
        return mdl, params, pw_cp_vars

    def _prepare_makespan_batch_model(
        self,
        spec: PwCpSubproblemSpec,
        instance: HybridFlowshopParameters,
        profile_fix_by_machine: bool,
        machine_precedence_stride: int,
        tighten_ranges: bool = False,
    ) -> tuple[CustomCpModel, Params, PwCpVars]:
        """Prepare CP model for makespan minimization batch.

        Unlike slack batches:
        - Uses left dummy bars (to respect left-time-fixed)
        - Does NOT use right dummy bars or window constraints
        - Minimizes makespan instead of maximizing common_spacing

        Unlike full problem:
        - Only optimizes non-time-fixed operations
        - Profile-fixed operations provide precedence constraints
        """
        horizon = spec.init_schedule.makespan
        mdl = CustomCpModel()

        # Parameters
        params: Params = self.builder.make_params(instance)
        last_stage_id = params.i_list[-1]

        # Validate: must have non-time-fixed operations in last stage
        if not spec.stage_2_partition[last_stage_id].non_time_fixed:
            raise ValueError(
                "Makespan batch requires at least one non-time-fixed operation in the last stage. "
                f"Got {len(spec.stage_2_partition[last_stage_id].non_time_fixed)} ops. Check partition logic."
            )

        # Variables
        non_time_fixed_vars: OperationVars = self.builder.make_non_time_fixed_ops_vars(
            mdl,
            params,
            horizon,
            spec.stage_2_partition,
            tighten_ranges=tighten_ranges,
        )
        dummy_bar_vars: DummyBarVars = self.builder.make_dummy_bar_vars(
            mdl, params, horizon, spec.stage_2_mc_2_window
        )
        # Constraints
        self.builder.add_non_fixed_job_precedence_constraints(
            mdl,
            params,
            spec.stage_2_partition,
            spec.init_schedule,
            spec.stage_2_mc_2_window,
            non_time_fixed_vars,
        )
        self.builder.add_capacity_with_dummy_bar_constraints(
            mdl, params, spec.stage_2_partition, non_time_fixed_vars, dummy_bar_vars
        )

        # Hints
        hint_schedule = spec.init_schedule
        ntf_op_set: set[OperationType] = set()
        for stage_id, partition in spec.stage_2_partition.items():
            for job_mc_id in partition.non_time_fixed:
                ntf_op_set.add((job_mc_id[0], stage_id, job_mc_id[1]))

        all_start_time_map = hint_schedule.get_jik_2_start_time_map()
        ntf_start_time_map = {op: all_start_time_map[op] for op in ntf_op_set}
        all_end_time_map = hint_schedule.get_jik_2_end_time_map()
        ntf_end_time_map = {op: all_end_time_map[op] for op in ntf_op_set}

        self.builder.apply_start_hints_from_start_time_map(
            mdl, params, non_time_fixed_vars, ntf_start_time_map
        )
        self.builder.apply_end_hints_from_end_time_map(
            mdl, params, non_time_fixed_vars, ntf_end_time_map
        )

        # Profile-fixed precedence constraints (from initial schedule)
        profile_fixed_op_set: set[JobMcType] = spec.profile_fixed_op_set
        if profile_fixed_op_set:
            removed_ops: set[OperationType] = set()
            for stage_id, partition in spec.stage_2_partition.items():
                for job_mc_id in partition.non_profile_fixed:
                    removed_ops.add((job_mc_id[0], stage_id, job_mc_id[1]))

            profile_fixed_schedule = spec.init_schedule.deepcopy()
            profile_fixed_schedule.remove_operations(removed_ops)
            self.builder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
                mdl,
                params,
                non_time_fixed_vars,
                profile_fixed_schedule,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            )

        # Objective
        makespan_var = self.builder.add_makespan_objective(
            mdl,
            horizon,
            last_stage_id,
            spec.stage_2_partition[last_stage_id],
            non_time_fixed_vars,
        )

        # Build PwCpVars (exclude right_bar vars as they're not used)
        pw_cp_vars = PwCpVars(
            op_start=non_time_fixed_vars.op_start,
            op_end=non_time_fixed_vars.op_end,
            op_intvl=non_time_fixed_vars.op_intvl,
            left_bar_interval=dummy_bar_vars.left_bar_interval,
            left_bar_end=dummy_bar_vars.left_bar_end,
            right_bar_interval={},
            right_bar_init_start={},
            common_spacing=dummy_bar_vars.common_spacing,
            makespan=makespan_var,
        )

        return mdl, params, pw_cp_vars

    def _solve_slack_batch(
        self,
        spec: PwCpSubproblemSpec,
        instance: HybridFlowshopParameters,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        profile_fix_by_machine: bool,
        machine_precedence_stride: int,
        max_time_per_batch: float | None,
        solver_thread_cnt: int,
        use_lns_only: bool,
        tighten_ranges: bool,
        debug_export: bool,
    ) -> HybridFlowshopLiteSchedule | None:
        _timer = ElapsedTimer()
        incumbent_obj = int(spec.init_schedule.makespan)
        logging.info(
            "PW-CP slack subproblem (batch=%d, subproblem=%d) starting. "
            "Incumbent makespan=%d, unfixed ops=%d, time_fixed ops=%d.",
            spec.batch_idx + 1,
            spec.subproblem_idx,
            incumbent_obj,
            len(spec.unfixed_op_set),
            len(spec.time_fixed_op_set),
        )
        mdl, params, variables = self._prepare_pw_cp_model(
            spec=spec,
            instance=instance,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            tighten_ranges=tighten_ranges,
        )
        timelimit = self.ctx.get_remaining_time_limit(max_time_per_batch)
        logging.info(
            "Solving non-final batch with common spacing maximization (timelimit=%.2fs).",
            timelimit,
        )
        report = self.ctx.solve_cp_model_2(
            mdl,
            timelimit,
            solver_thread_cnt,
            e_timer=_timer,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            use_lns_only=use_lns_only,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
            log_search_progress=debug_export,
            last_timestamp_note=f"batch={spec.batch_idx + 1}",
        )

        if not getattr(report, "is_feasible", False):
            self._append_subproblem_log(
                spec=spec,
                report=report,
                time_limit_sec=timelimit,
                incumbent_makespan_before=incumbent_obj,
                candidate_makespan=None,
                accepted=False,
            )
            logging.info(
                "Non-final batch: no feasible solution found. "
                "Incumbent makespan=%d remains.",
                incumbent_obj,
            )
            return None

        candidate_schedule = create_pw_cp_schedule(
            self.ctx.solver,
            params,
            spec.stage_2_partition,
            spec.init_schedule,
            variables,
        )
        candidate_schedule.make_semi_active(stage_2_job_2_p_dict)
        candidate_obj = int(candidate_schedule.makespan)
        self._append_subproblem_log(
            spec=spec,
            report=report,
            time_limit_sec=timelimit,
            incumbent_makespan_before=incumbent_obj,
            candidate_makespan=candidate_obj,
            accepted=candidate_obj < incumbent_obj,
        )
        logging.info(
            "Non-final batch: CP solution found. Incumbent makespan=%d, "
            "CP solution makespan=%d, diff=%d (%.2f%%).",
            incumbent_obj,
            candidate_obj,
            candidate_obj - incumbent_obj,
            100.0 * (candidate_obj - incumbent_obj) / incumbent_obj
            if incumbent_obj > 0
            else 0,
        )
        return candidate_schedule

    def _solve_makespan_batch(
        self,
        spec: PwCpSubproblemSpec,
        instance: HybridFlowshopParameters,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        profile_fix_by_machine: bool,
        machine_precedence_stride: int,
        max_time_per_batch: float | None,
        solver_thread_cnt: int,
        use_lns_only: bool,
        tighten_ranges: bool,
        debug_export: bool,
    ) -> HybridFlowshopLiteSchedule | None:
        """Solve a batch with makespan minimization.

        Called when right-time-fixed operations are empty.
        Optimizes non-time-fixed operations while respecting:
        - Left-time-fixed operations (via left dummy bars)
        - Profile-fixed precedence constraints
        """
        _timer = ElapsedTimer()
        incumbent_obj = int(spec.init_schedule.makespan)
        logging.info(
            "PW-CP makespan batch (batch=%d, subproblem=%d) starting. "
            "Incumbent makespan=%d, non-time-fixed ops=%d.",
            spec.batch_idx + 1,
            spec.subproblem_idx,
            incumbent_obj,
            len(spec.unfixed_op_set | spec.profile_fixed_op_set),
        )

        mdl, params, variables = self._prepare_makespan_batch_model(
            spec=spec,
            instance=instance,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            tighten_ranges=tighten_ranges,
        )

        timelimit = self.ctx.get_remaining_time_limit(max_time_per_batch)
        logging.info(
            "Solving makespan batch with makespan minimization (timelimit=%.2fs).",
            timelimit,
        )

        report = self.ctx.solve_cp_model_2(
            mdl,
            timelimit,
            solver_thread_cnt,
            e_timer=_timer,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            use_lns_only=use_lns_only,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
            log_search_progress=debug_export,
            last_timestamp_note=f"batch={spec.batch_idx + 1}",
        )

        if not getattr(report, "is_feasible", False):
            self._append_subproblem_log(
                spec=spec,
                report=report,
                time_limit_sec=timelimit,
                incumbent_makespan_before=incumbent_obj,
                candidate_makespan=None,
                accepted=False,
            )
            logging.info(
                "Makespan batch: no feasible solution found. "
                "Incumbent makespan=%d remains.",
                incumbent_obj,
            )
            return None

        candidate_schedule = create_pw_cp_schedule(
            self.ctx.solver,
            params,
            spec.stage_2_partition,
            spec.init_schedule,
            variables,
        )
        candidate_schedule.make_semi_active(stage_2_job_2_p_dict)
        candidate_obj = int(candidate_schedule.makespan)

        self._append_subproblem_log(
            spec=spec,
            report=report,
            time_limit_sec=timelimit,
            incumbent_makespan_before=incumbent_obj,
            candidate_makespan=candidate_obj,
            accepted=candidate_obj < incumbent_obj,
        )

        logging.info(
            "Makespan batch: CP solution found. Incumbent makespan=%d, "
            "CP solution makespan=%d, improvement=%d (%.2f%%).",
            incumbent_obj,
            candidate_obj,
            incumbent_obj - candidate_obj,
            100.0 * (incumbent_obj - candidate_obj) / incumbent_obj
            if incumbent_obj > 0
            else 0,
        )

        return candidate_schedule

    def _accept_candidate_or_repair_incumbent(
        self,
        candidate: HybridFlowshopLiteSchedule | None,
        incumbent: HybridFlowshopLiteSchedule,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
    ) -> tuple[HybridFlowshopLiteSchedule, bool]:
        if candidate is not None and candidate.makespan < incumbent.makespan:
            try:
                self.ctx.check_feasibility(candidate.get_jik_2_start_time_map())
            except Exception:
                logging.exception("PW-CP candidate failed feasibility check.")
            else:
                return candidate, True
        incumbent.make_semi_active(stage_2_job_2_p_dict)
        return incumbent, False

    def _append_subproblem_log(
        self,
        spec: PwCpSubproblemSpec,
        report: CpsatSolverReport,
        time_limit_sec: float,
        incumbent_makespan_before: int,
        candidate_makespan: int | None,
        accepted: bool,
    ) -> None:
        self._require_state().subproblem_logs.append(
            PwCpSubproblemLog(
                batch_idx=spec.batch_idx,
                subproblem_idx=spec.subproblem_idx,
                is_last_batch=spec.is_last_batch,
                objective_name=("makespan" if spec.is_last_batch else "common_spacing"),
                time_limit_sec=time_limit_sec,
                elapsed_time_sec=report.elapsed_time,
                status=report.status.to_solver_status_enum().value,
                incumbent_makespan_before=incumbent_makespan_before,
                candidate_makespan=candidate_makespan,
                accepted=accepted,
                obj_value_records=tuple(report.obj_value_records),
                obj_bound_records=tuple(report.obj_bound_records),
            )
        )

    def _validate_and_get_batch_count(
        self, stage_2_batches: dict[StageIdType, list[tuple[JobMcType, ...]]]
    ) -> int:
        batch_counts: dict[StageIdType, int] = {
            stage_id: len(batches) for stage_id, batches in stage_2_batches.items()
        }
        unique_counts = set(batch_counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"PW-CP requires identical batch counts across stages, got {batch_counts}."
            )
        return next(iter(unique_counts), 0)

    def _draw_schedule_gantt(
        self,
        schedule: HybridFlowshopLiteSchedule,
        output_suffix: str,
        highlight_op_set: set[tuple[str, str]] | None,
        machine_list_per_stage: dict[str, list[str]] | None,
        force_start: int | None,
        force_end: int | None,
    ) -> None:
        output_path = self.ctx.get_file_path_for_subroutine(output_suffix)
        plotter = GanttPlotter()
        plotter.export_hybrid_flowshop_plot(
            output_path,
            schedule.get_jik_2_start_time_map(),
            schedule.get_jik_2_end_time_map(),
            job_list=None,
            stage_list=None,
            machine_list_per_stage=machine_list_per_stage,
            all_job_list=None,
            highlight_op_set=highlight_op_set,
            force_start=force_start,
            force_end=force_end,
        )

    def _draw_candidate_retiming_gantts(
        self,
        spec: PwCpSubproblemSpec,
        before_retiming: HybridFlowshopLiteSchedule,
        after_successor_reassign: HybridFlowshopLiteSchedule | None,
        after_predecessor_reassign: HybridFlowshopLiteSchedule | None,
        after_semi_active: HybridFlowshopLiteSchedule | None,
        solved_slack_intervals: SolvedSlackIntervals,
    ) -> None:
        """Draw candidate schedules for each reassignment and retiming step."""
        highlight_op_set: set[tuple[JobIdType, StageIdType]] = set()
        for stage_id, partition in spec.stage_2_partition.items():
            highlight_op_set.update(
                (job_id, stage_id) for job_id, _ in partition.unfixed
            )
        schedule_items: list[tuple[str, HybridFlowshopLiteSchedule]] = [
            ("01_before_retiming", before_retiming)
        ]
        if after_successor_reassign is not None:
            schedule_items.append(
                ("02_after_successor_reassign", after_successor_reassign)
            )
        if after_predecessor_reassign is not None:
            schedule_items.append(
                ("03_after_predecessor_reassign", after_predecessor_reassign)
            )
        if after_semi_active is not None:
            schedule_items.append(("04_after_semi_active", after_semi_active))

        force_end = max(int(schedule.makespan) for _, schedule in schedule_items)

        for suffix, schedule in schedule_items:
            start_map, end_map, machine_list_per_stage = (
                self._build_plot_inputs_with_slack_intervals(
                    schedule,
                    solved_slack_intervals,
                )
            )
            output_path = self.ctx.get_file_path_for_subroutine(
                f"_batch_{spec.batch_idx + 1:03d}_candidate_{suffix}.png"
            )
            plotter = GanttPlotter()
            plotter.export_hybrid_flowshop_plot(
                output_path,
                start_map,
                end_map,
                job_list=None,
                stage_list=None,
                machine_list_per_stage=machine_list_per_stage,
                all_job_list=None,
                highlight_op_set=highlight_op_set,
                force_start=0,
                force_end=force_end,
            )

    @staticmethod
    def _build_plot_inputs_with_slack_intervals(
        schedule: HybridFlowshopLiteSchedule,
        solved_slack_intervals: SolvedSlackIntervals,
    ) -> tuple[
        dict[tuple[str, str, str], int],
        dict[tuple[str, str, str], int],
        dict[str, list[str]],
    ]:
        """Build plotting inputs with solved slack intervals on separate lanes."""
        start_map = dict(schedule.get_jik_2_start_time_map())
        end_map = dict(schedule.get_jik_2_end_time_map())
        machine_list_per_stage = {
            stage_id: list(machine_ids)
            for stage_id, machine_ids in schedule.machines_per_stage.items()
        }

        for stage_id, stage_slacks in solved_slack_intervals.items():
            stage_machine_ids = machine_list_per_stage.setdefault(stage_id, [])
            for machine_id, slack_start, slack_end in stage_slacks:
                if slack_end <= slack_start:
                    continue
                slack_machine_id = f"{machine_id}__cp_slack"
                if slack_machine_id not in stage_machine_ids:
                    stage_machine_ids.append(slack_machine_id)
                slack_job_id = f"cp-slack-{stage_id}-{machine_id}"
                start_map[(slack_job_id, stage_id, slack_machine_id)] = slack_start
                end_map[(slack_job_id, stage_id, slack_machine_id)] = slack_end

        return start_map, end_map, machine_list_per_stage

    def _save_solution_dict(
        self,
        spec: PwCpSubproblemSpec,
        incumbent: HybridFlowshopLiteSchedule,
        *,
        accepted: bool,
    ) -> None:
        output_path = self.ctx.get_file_path_for_subroutine(
            f"_batch_{spec.batch_idx + 1:03d}_solution.yaml"
        )
        solution_dict = {
            "start_time_map": incumbent.get_jik_2_start_time_map(),
            "end_time_map": incumbent.get_jik_2_end_time_map(),
            "highlight_ops": self._build_highlight_ops(spec.stage_2_partition),
            "batch_idx": spec.batch_idx,
            "subproblem_idx": spec.subproblem_idx,
            "accepted": accepted,
            "partition": {
                "left_time_fixed": list(spec.left_time_fixed_op_set),
                "left_profile_fixed": list(spec.left_profile_fixed_op_set),
                "unfixed": list(spec.unfixed_op_set),
                "right_profile_fixed": list(spec.right_profile_fixed_op_set),
                "right_time_fixed": list(spec.right_time_fixed_op_set),
            },
        }
        dump_yaml(self._normalize_for_yaml(solution_dict), output_path)
