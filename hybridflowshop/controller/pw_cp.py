from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from mbls.cpsat import CpsatSolverReport, CustomCpModel, ObjValueBoundStore
from ortools.sat.python import cp_model
from routix import ElapsedTimer
from routix.io.yaml import dump_yaml
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.cpsat_model_2.cumulative import (
    BaseModelBuilder,
    CumulativeVars,
)
from hybridflowshop.cpsat_model_2.params import Params
from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
)

OperationRef = tuple[str, str, str]
HighlightedOperationRef = tuple[str, str]
StageBoundaryProfile = dict[str, list[int | None]]
SolvedSlackIntervals = dict[str, list[tuple[str, int, int]]]


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

    def check_feasibility(
        self, start_time_map: dict[tuple[str, str, str], int]
    ) -> float: ...

    def get_file_path_for_subroutine(self, suffix: str) -> Path: ...


@dataclass(frozen=True)
class OperationPartition:
    """
    Encapsulates the operation partition of PW-CP subproblems.

    Operations are grouped into time-fixed, profile-fixed, unfixed, and
    right-time-fixed regions around the current batch.
    """

    left_time_fixed: tuple[OperationRef, ...]
    left_profile_fixed: tuple[OperationRef, ...]
    unfixed: tuple[OperationRef, ...]
    right_profile_fixed: tuple[OperationRef, ...]
    right_time_fixed: tuple[OperationRef, ...]

    # Boundary profiles (computed during partition creation)
    right_boundary_profile: StageBoundaryProfile | None = None

    @property
    def all_operations(self) -> tuple[OperationRef, ...]:
        """Return all operations in the partition."""
        return (
            self.left_time_fixed
            + self.left_profile_fixed
            + self.unfixed
            + self.right_profile_fixed
            + self.right_time_fixed
        )

    @property
    def time_fixed_operations(self) -> tuple[OperationRef, ...]:
        """Return all operations that should have fixed start times in CP model."""
        return self.left_time_fixed + self.right_time_fixed

    @property
    def profile_fixed_operations(self) -> tuple[OperationRef, ...]:
        """Return all operations that keep precedence but not start times fixed."""
        return self.left_profile_fixed + self.right_profile_fixed

    @property
    def slack_occupying_operations(self) -> tuple[OperationRef, ...]:
        """Return all operations except right-time-fixed ones."""
        return (
            self.left_time_fixed
            + self.left_profile_fixed
            + self.unfixed
            + self.right_profile_fixed
        )

    def promote_job_contained_ops(self) -> OperationPartition:
        """Promote profile-fixed operations of unfixed jobs into the unfixed set."""
        unfixed_job_ids = {job_id for job_id, _stage_id, _mc_id in self.unfixed}
        if not unfixed_job_ids:
            return self

        promoted_left = tuple(
            sorted(op for op in self.left_profile_fixed if op[0] not in unfixed_job_ids)
        )
        promoted_right = tuple(
            sorted(
                op for op in self.right_profile_fixed if op[0] not in unfixed_job_ids
            )
        )
        promoted_unfixed = tuple(
            sorted(
                self.unfixed
                + tuple(
                    op
                    for op in self.left_profile_fixed + self.right_profile_fixed
                    if op[0] in unfixed_job_ids
                )
            )
        )
        return OperationPartition(
            left_time_fixed=self.left_time_fixed,
            left_profile_fixed=promoted_left,
            unfixed=promoted_unfixed,
            right_profile_fixed=promoted_right,
            right_time_fixed=self.right_time_fixed,
            right_boundary_profile=self.right_boundary_profile,
        )


@dataclass(frozen=True)
class PwCpSubproblemSpec:
    batch_idx: int
    subproblem_idx: int
    partition: OperationPartition
    right_boundary_profile: StageBoundaryProfile
    is_last_batch: bool


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
        self.builder = BaseModelBuilder()
        self._st: PwCpRunState | None = None

    def _require_state(self) -> PwCpRunState:
        if self._st is None:
            raise RuntimeError("PwCpConstructor.run() is not active; state is missing.")
        return self._st

    @staticmethod
    def _resolve_batch_time_limit(
        partition: OperationPartition,
        unfixed_op_time_limit_multiplier: float | None,
        max_time_per_batch: float | None,
    ) -> float | None:
        if unfixed_op_time_limit_multiplier is not None:
            if unfixed_op_time_limit_multiplier <= 0:
                raise ValueError("unfixed_op_time_limit_multiplier must be > 0")
            return len(partition.unfixed) * unfixed_op_time_limit_multiplier
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
        unfixed_op_time_limit_multiplier: float | None = None,
        max_time_per_batch: float | None = None,
        solver_thread_cnt: int | None = None,
        use_lns_only: bool = False,
        debug_export: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        error_if_infeasible: bool = False,
    ) -> PwCpResult:
        timer = ElapsedTimer()
        if solver_thread_cnt is None:
            solver_thread_cnt = 1
        if (
            unfixed_op_time_limit_multiplier is not None
            and unfixed_op_time_limit_multiplier <= 0
        ):
            raise ValueError("unfixed_op_time_limit_multiplier must be > 0")

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
                if unfixed_op_time_limit_multiplier is not None
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

                partition, right_justified_sched = self._build_operation_partition(
                    current_batches,
                    ref_schedule.stages,
                    batch_idx,
                    incumbent=st.incumbent,
                    stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                    left_profile_fixed_batch_count=left_profile_fixed_batch_count,
                    right_profile_fixed_batch_count=right_profile_fixed_batch_count,
                )
                if enable_promotion_profile_fixed:
                    partition = partition.promote_job_contained_ops()

                batch_time_limit = self._resolve_batch_time_limit(
                    partition=partition,
                    max_time_per_batch=max_time_per_batch,
                    unfixed_op_time_limit_multiplier=unfixed_op_time_limit_multiplier,
                )
                timelimit = self.ctx.get_remaining_time_limit(batch_time_limit)
                if timelimit <= 0:
                    logging.info(
                        "PW-CP time limit exhausted before batch %d.",
                        batch_idx + 1,
                    )
                    break

                spec = self._build_subproblem_spec(
                    incumbent=st.incumbent,
                    stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                    partition=partition,
                    batch_idx=batch_idx,
                    max_batch_cnt=max_batch_cnt,
                )
                candidate = self._solve_subproblem(
                    spec=spec,
                    incumbent=st.incumbent,
                    right_justified_schedule=right_justified_sched,
                    instance=instance,
                    stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                    profile_fix_by_machine=profile_fix_by_machine,
                    machine_precedence_stride=machine_precedence_stride,
                    max_time_per_batch=batch_time_limit,
                    solver_thread_cnt=solver_thread_cnt,
                    use_lns_only=use_lns_only,
                    tighten_ranges=tighten_ranges,
                    link_job_completion=link_job_completion,
                    debug_export=debug_export,
                )

                accepted = False
                if candidate is not None and candidate.makespan < st.incumbent.makespan:
                    try:
                        self.ctx.check_feasibility(candidate.get_jik_2_start_time_map())
                    except Exception:
                        logging.exception("PW-CP candidate failed feasibility check.")
                    else:
                        st.incumbent = candidate
                        accepted = True
                else:
                    st.incumbent.make_semi_active(stage_2_job_2_p_dict)

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
    ) -> dict[str, list[tuple[OperationRef, ...]]]:
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
            is a tuple of operation references (job_id, stage_id, machine_id).

        Example:
            With operations:
                - A: start=0, end=100 (midpoint=50)
                - B: start=40, end=50 (midpoint=45)

            sort_by_start_time=True  -> order: A, B (by start: 0 < 40)
            sort_by_start_time=False -> order: B, A (by midpoint: 45 < 50)
        """
        _batch_size = max(1, batch_size)
        stage_2_batches: dict[str, list[tuple[OperationRef, ...]]] = {}
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
                    (job_id, stage_id, mc_id)
                    for job_id, _, mc_id, _, _ in ops[idx : idx + _batch_size]
                )
                for idx in range(0, len(ops), _batch_size)
            ]
        return stage_2_batches

    def _build_subproblem_spec(
        self,
        incumbent: HybridFlowshopLiteSchedule,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        partition: OperationPartition,
        batch_idx: int,
        max_batch_cnt: int,
    ) -> PwCpSubproblemSpec:
        st = self._require_state()
        st.subproblem_idx += 1

        # Profiles are now computed during partition creation
        assert partition.right_boundary_profile is not None, "Right profile must be set"
        return PwCpSubproblemSpec(
            batch_idx=batch_idx,
            subproblem_idx=st.subproblem_idx,
            partition=partition,
            right_boundary_profile=partition.right_boundary_profile,
            is_last_batch=(batch_idx == max_batch_cnt - 1),
        )

    def _build_right_boundary_profile(
        self,
        incumbent: HybridFlowshopLiteSchedule,
        all_ops: list[OperationRef],
        l_tf_ops: list[OperationRef],
        r_tf_ops: list[OperationRef],
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
    ) -> tuple[StageBoundaryProfile, HybridFlowshopLiteSchedule]:
        shifted = incumbent.deepcopy()
        r_justified_op_set = set(all_ops) - set(l_tf_ops)
        shifted.make_right_justified(
            stage_2_job_2_p_dict,
            operation_set=r_justified_op_set,
        )
        start_map = shifted.get_jik_2_start_time_map()
        right_boundary_profile: StageBoundaryProfile = {}
        for stage_id in shifted.stages:
            stage_right_ops = [
                (job_id, mc_id)
                for job_id, op_stage_id, mc_id in r_tf_ops
                if op_stage_id == stage_id
            ]
            stage_boundaries: list[int] = []

            for mc_id in shifted.machines_per_stage[stage_id]:
                machine_starts = [
                    int(start_map[job_id, stage_id, op_mc_id])
                    for job_id, op_mc_id in stage_right_ops
                    if op_mc_id == mc_id
                ]
                stage_boundaries.append(
                    min(machine_starts) if machine_starts else None
                )

            right_boundary_profile[stage_id] = stage_boundaries
        return right_boundary_profile, shifted

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
        unfixed_ops: tuple[OperationRef, ...],
    ) -> list[HighlightedOperationRef]:
        highlight_ops: list[HighlightedOperationRef] = []
        seen: set[HighlightedOperationRef] = set()
        for job_id, stage_id, _machine_id in unfixed_ops:
            op_ref = (job_id, stage_id)
            if op_ref not in seen:
                seen.add(op_ref)
                highlight_ops.append(op_ref)
        return highlight_ops

    def _solve_subproblem(
        self,
        spec: PwCpSubproblemSpec,
        incumbent: HybridFlowshopLiteSchedule,
        right_justified_schedule: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        profile_fix_by_machine: bool,
        machine_precedence_stride: int,
        max_time_per_batch: float | None,
        solver_thread_cnt: int,
        use_lns_only: bool,
        tighten_ranges: bool,
        link_job_completion: bool,
        debug_export: bool,
    ) -> HybridFlowshopLiteSchedule | None:
        _timer = ElapsedTimer()
        incumbent_obj = int(incumbent.makespan)
        logging.info(
            "PW-CP subproblem (batch=%d, subproblem=%d) starting. "
            "Incumbent makespan=%d, unfixed ops=%d, time_fixed ops=%d.",
            spec.batch_idx + 1,
            spec.subproblem_idx,
            incumbent_obj,
            len(spec.partition.unfixed),
            len(spec.partition.time_fixed_operations),
        )
        mdl = CustomCpModel()
        params: Params = BaseModelBuilder._make_params(instance)
        variables: CumulativeVars = BaseModelBuilder._make_vars(
            mdl, params, incumbent.makespan, tighten_ranges=tighten_ranges
        )
        if link_job_completion:
            BaseModelBuilder._add_job_completion_link_constraints(
                mdl, params, variables
            )

        hint_schedule = incumbent if spec.is_last_batch else right_justified_schedule
        BaseModelBuilder.apply_start_hints_from_start_time_map(
            mdl, params, variables, hint_schedule.get_jik_2_start_time_map()
        )
        BaseModelBuilder.apply_end_hints_from_end_time_map(
            mdl, params, variables, hint_schedule.get_jik_2_end_time_map()
        )

        right_justified_start_map = right_justified_schedule.get_jik_2_start_time_map()
        frozen_start_time_map = {
            op: right_justified_start_map[op]
            for op in spec.partition.time_fixed_operations
        }
        BaseModelBuilder.add_start_time_freezed_operation_constraints(
            mdl, variables, frozen_start_time_map
        )
        if spec.partition.profile_fixed_operations:
            profile_fixed_schedule = incumbent.deepcopy()
            profile_fixed_schedule.remove_operations(
                set(profile_fixed_schedule.get_jik_2_start_time_map())
                - set(spec.partition.profile_fixed_operations)
            )
            BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
                mdl,
                params,
                variables,
                profile_fixed_schedule,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            )

        timelimit = self.ctx.get_remaining_time_limit(max_time_per_batch)

        if spec.is_last_batch:
            # Final batch: solve original problem
            BaseModelBuilder._add_structural_constraints(mdl, params, variables)
            BaseModelBuilder._define_objective(mdl, params, variables)
            mdl.add_hint(variables.makespan, incumbent_obj)
            logging.info(
                "Solving final batch with makespan minimization (timelimit=%.2fs).",
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
                    "Final batch: no feasible solution found. "
                    "Incumbent makespan=%d remains.",
                    incumbent_obj,
                )
                return None

            candidate_schedule = self.ctx.create_schedule(params, variables)
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
                "Final batch: CP solution found. Incumbent makespan=%d, "
                "CP solution makespan=%d, diff=%d (%.2f%%).",
                incumbent_obj,
                candidate_obj,
                candidate_obj - incumbent_obj,
                100.0 * (candidate_obj - incumbent_obj) / incumbent_obj
                if incumbent_obj > 0
                else 0,
            )
            return candidate_schedule
        else:
            # Non-final batch: maximize right slack
            BaseModelBuilder._add_precedence_constraints(mdl, params, variables)
            slack_vars = BaseModelBuilder.add_right_slack_variables(
                mdl,
                params,
                slack_occupying_ops=spec.partition.slack_occupying_operations,
                right_boundary_profile=spec.right_boundary_profile,
                horizon=incumbent.makespan,
            )
            BaseModelBuilder.add_right_slack_constraints(
                mdl,
                params,
                variables,
                slack_occupying_ops=spec.partition.slack_occupying_operations,
                right_time_fixed_ops=spec.partition.right_time_fixed,
                right_boundary_profile=spec.right_boundary_profile,
                slack_vars=slack_vars,
            )
            BaseModelBuilder.add_right_slack_objective(
                mdl,
                slack_occupying_ops=spec.partition.slack_occupying_operations,
                slack_vars=slack_vars,
            )
            self._apply_right_slack_hints(
                mdl=mdl,
                schedule=right_justified_schedule,
                slack_occupying_ops=spec.partition.slack_occupying_operations,
                right_boundary_profile=spec.right_boundary_profile,
                params=params,
            )
            mdl.add_hint(variables.makespan, incumbent_obj)
            logging.info(
                "Solving non-final batch with right slack maximization (timelimit=%.2fs).",
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

            candidate_schedule = self.ctx.create_schedule(params, variables)
            candidate_before_retiming = candidate_schedule.deepcopy()
            candidate_after_successor_reassign: HybridFlowshopLiteSchedule | None = None
            candidate_after_predecessor_reassign: HybridFlowshopLiteSchedule | None = None
            candidate_after_semi_active: HybridFlowshopLiteSchedule | None = None
            solved_slack_intervals: SolvedSlackIntervals = {}
            try:
                if report.obj_value is None:
                    raise ValueError(
                        "Non-final batch feasible solution must report a right-slack objective value."
                    )
                if report.obj_value < 0:
                    raise ValueError(
                        f"Non-final batch right-slack objective must be >= 0, got {report.obj_value}."
                    )
                if report.obj_value > 0:
                    solved_slack_intervals = self._extract_solved_slack_intervals(
                        mdl=mdl,
                        schedule=candidate_schedule,
                        right_boundary_profile=spec.right_boundary_profile,
                    )
                    self._apply_successor_machine_reassignment(
                        candidate_schedule=candidate_schedule,
                        solved_slack_intervals=solved_slack_intervals,
                        right_boundary_profile=spec.right_boundary_profile,
                        partition=spec.partition,
                        stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                    )
                    candidate_after_successor_reassign = candidate_schedule.deepcopy()
                    self._apply_predecessor_machine_reassignment(
                        candidate_schedule=candidate_schedule,
                        solved_slack_intervals=solved_slack_intervals,
                        partition=spec.partition,
                        stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                    )
                    candidate_after_predecessor_reassign = candidate_schedule.deepcopy()

                candidate_schedule.make_semi_active(stage_2_job_2_p_dict)
                candidate_after_semi_active = candidate_schedule.deepcopy()
            finally:
                if debug_export:
                    self._draw_candidate_retiming_gantts(
                        spec=spec,
                        before_retiming=candidate_before_retiming,
                        after_successor_reassign=candidate_after_successor_reassign,
                        after_predecessor_reassign=candidate_after_predecessor_reassign,
                        after_semi_active=candidate_after_semi_active,
                        solved_slack_intervals=solved_slack_intervals,
                    )

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

    @staticmethod
    def _compute_right_slack_values(
        schedule: HybridFlowshopLiteSchedule,
        slack_occupying_ops: tuple[OperationRef, ...],
        right_boundary_profile: StageBoundaryProfile,
        params: Params,
    ) -> dict[str, dict[str, int] | dict[str, list[tuple[str, int, int]]]]:
        """Compute right-slack summaries from a schedule and boundary profile."""
        stage_2_machine_2_intervals: dict[str, dict[str, list[tuple[int, int]]]] = {}
        start_time_map = schedule.get_jik_2_start_time_map()
        end_time_map = schedule.get_jik_2_end_time_map()

        for job_id, stage_id, machine_id in slack_occupying_ops:
            stage_2_machine_2_intervals.setdefault(stage_id, {}).setdefault(
                machine_id, []
            ).append(
                (
                    int(start_time_map[job_id, stage_id, machine_id]),
                    int(end_time_map[job_id, stage_id, machine_id]),
                )
            )

        hint_values: dict[str, int] = {}
        slack_intervals_by_stage: dict[str, list[tuple[str, int, int]]] = {}
        actual_slack_end_by_stage_machine: dict[tuple[str, int], int] = {}
        slack_lengths: list[int] = []

        for stage_id, machine_ids in params.M_of.items():
            machine_2_intervals = stage_2_machine_2_intervals.get(stage_id, {})
            stage_slack_intervals: list[tuple[str, int, int]] = []
            for machine_idx, (machine_id, slack_end) in enumerate(
                zip(machine_ids, right_boundary_profile.get(stage_id, []), strict=False),
                start=1,
            ):
                if slack_end is None:
                    continue
                latest_end_before_slack = 0
                for start, end in machine_2_intervals.get(machine_id, []):
                    if start < slack_end:
                        latest_end_before_slack = max(
                            latest_end_before_slack, min(end, slack_end)
                        )
                slack_length = max(0, slack_end - latest_end_before_slack)
                slack_lengths.append(slack_length)
                actual_slack_end_by_stage_machine[stage_id, machine_idx] = slack_end

        common_slack_length = min(slack_lengths) if slack_lengths else 0

        for stage_id, machine_ids in params.M_of.items():
            stage_slack_intervals = []
            for machine_idx, (machine_id, slack_end) in enumerate(
                zip(machine_ids, right_boundary_profile.get(stage_id, []), strict=False),
                start=1,
            ):
                del slack_end
                actual_slack_end = actual_slack_end_by_stage_machine.get(
                    (stage_id, machine_idx)
                )
                if actual_slack_end is None:
                    continue
                slack_start = max(0, actual_slack_end - common_slack_length)
                hint_values[f"slack_start_{stage_id}_{machine_idx}"] = slack_start
                if common_slack_length > 0:
                    stage_slack_intervals.append(
                        (machine_id, slack_start, actual_slack_end)
                    )

            if stage_slack_intervals:
                slack_intervals_by_stage[stage_id] = stage_slack_intervals

        hint_values["slack_length"] = common_slack_length
        return {
            "hint_values": hint_values,
            "slack_intervals_by_stage": slack_intervals_by_stage,
        }

    @staticmethod
    def _compute_right_slack_hint_values(
        schedule: HybridFlowshopLiteSchedule,
        slack_occupying_ops: tuple[OperationRef, ...],
        right_boundary_profile: StageBoundaryProfile,
        params: Params,
    ) -> dict[str, int]:
        """Compute hint values for right-slack objective auxiliaries."""
        return PwCpConstructor._compute_right_slack_values(
            schedule=schedule,
            slack_occupying_ops=slack_occupying_ops,
            right_boundary_profile=right_boundary_profile,
            params=params,
        )["hint_values"]

    @staticmethod
    def _apply_right_slack_hints(
        mdl: CustomCpModel,
        schedule: HybridFlowshopLiteSchedule,
        slack_occupying_ops: tuple[OperationRef, ...],
        right_boundary_profile: StageBoundaryProfile,
        params: Params,
    ) -> None:
        """Apply schedule-derived hints for right-slack auxiliaries by variable name."""
        hint_values = PwCpConstructor._compute_right_slack_hint_values(
            schedule=schedule,
            slack_occupying_ops=slack_occupying_ops,
            right_boundary_profile=right_boundary_profile,
            params=params,
        )
        proto = mdl.Proto()
        name_to_index = {
            var.name: var_idx for var_idx, var in enumerate(proto.variables) if var.name
        }
        for var_name, value in hint_values.items():
            var_idx = name_to_index.get(var_name)
            if var_idx is None:
                continue
            mdl.add_hint(mdl.get_int_var_from_proto_index(var_idx), value)

    def _extract_solved_slack_intervals(
        self,
        mdl: CustomCpModel,
        schedule: HybridFlowshopLiteSchedule,
        right_boundary_profile: StageBoundaryProfile,
    ) -> SolvedSlackIntervals:
        """Extract solved slack intervals from the current CP-SAT solution."""
        try:
            _ = self.ctx.solver.ResponseProto()
        except RuntimeError:
            logging.debug(
                "Skipping solved slack extraction because the CP solver response "
                "is unavailable in the current context."
            )
            return {}

        proto = mdl.Proto()
        name_to_index = {
            var.name: var_idx for var_idx, var in enumerate(proto.variables) if var.name
        }
        solved: SolvedSlackIntervals = {}

        for stage_id, machine_ids in schedule.machines_per_stage.items():
            slack_end_times = right_boundary_profile.get(stage_id, [])
            stage_slacks: list[tuple[str, int, int]] = []
            for machine_idx, machine_id in enumerate(machine_ids, start=1):
                if machine_idx > len(slack_end_times):
                    continue
                slack_end = slack_end_times[machine_idx - 1]
                if slack_end is None:
                    continue
                start_var_idx = name_to_index.get(
                    f"slack_start_{stage_id}_{machine_idx}"
                )
                length_var_idx = name_to_index.get(
                    "slack_length"
                )
                if start_var_idx is None or length_var_idx is None:
                    continue

                slack_start = int(
                    self.ctx.solver.Value(
                        mdl.get_int_var_from_proto_index(start_var_idx)
                    )
                )
                slack_length = int(
                    self.ctx.solver.Value(
                        mdl.get_int_var_from_proto_index(length_var_idx)
                    )
                )
                if slack_length <= 0:
                    continue

                stage_slacks.append((machine_id, slack_start, int(slack_end)))

            if stage_slacks:
                solved[stage_id] = stage_slacks

        return solved

    @staticmethod
    def _find_job_machine_on_stage(
        schedule: HybridFlowshopLiteSchedule,
        stage_id: str,
        job_id: str,
    ) -> str | None:
        for machine_id in schedule.machines_per_stage[stage_id]:
            for _start_time, _end_time, scheduled_job_id in schedule.get_job_sequence(
                stage_id, machine_id
            ):
                if scheduled_job_id == job_id:
                    return machine_id
        return None

    @staticmethod
    def _find_first_job_on_machine_starting_at_or_after(
        schedule: HybridFlowshopLiteSchedule,
        stage_id: str,
        machine_id: str,
        time_point: int,
    ) -> str | None:
        for start_time, _end_time, job_id in schedule.get_job_sequence(
            stage_id, machine_id
        ):
            if start_time >= time_point:
                return job_id
        return None

    def _apply_successor_machine_reassignment(
        self,
        candidate_schedule: HybridFlowshopLiteSchedule,
        solved_slack_intervals: SolvedSlackIntervals,
        right_boundary_profile: StageBoundaryProfile,
        partition: OperationPartition,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
    ) -> None:
        """Align stage-local machine assignment with solved CP slack intervals."""
        del right_boundary_profile

        if not solved_slack_intervals:
            return

        for stage_id, stage_slacks in solved_slack_intervals.items():
            sorted_stage_slacks = sorted(
                stage_slacks,
                key=lambda slack: (slack[1], slack[2], slack[0]),
            )
            for target_machine_id, slack_start, slack_end in sorted_stage_slacks:
                if slack_end <= slack_start:
                    continue

                boundary_job_ids: list[str] = []
                for (
                    job_id,
                    op_stage_id,
                    intended_machine_id,
                ) in partition.right_time_fixed:
                    if (
                        op_stage_id != stage_id
                        or intended_machine_id != target_machine_id
                    ):
                        continue
                    current_machine_id = self._find_job_machine_on_stage(
                        candidate_schedule, stage_id, job_id
                    )
                    if current_machine_id is None:
                        continue
                    for (
                        start_time,
                        _end_time,
                        scheduled_job_id,
                    ) in candidate_schedule.get_job_sequence(
                        stage_id, current_machine_id
                    ):
                        if scheduled_job_id == job_id and start_time == slack_end:
                            boundary_job_ids.append(job_id)
                            break

                if not boundary_job_ids:
                    if slack_end != candidate_schedule.makespan:
                        logging.warning(
                            "PW-CP slack reassignment skipped for %s.%s [%d, %d): "
                            "no right-boundary job starts at slack_end. "
                            "Possible mismatch between CP solution and current schedule "
                            "(slack_end=%d, makespan=%d).",
                            stage_id,
                            target_machine_id,
                            slack_start,
                            slack_end,
                            slack_end,
                            candidate_schedule.makespan,
                        )
                    continue

                boundary_job_id = boundary_job_ids[0]
                current_machine_id = self._find_job_machine_on_stage(
                    candidate_schedule, stage_id, boundary_job_id
                )
                if current_machine_id is None:
                    raise ValueError(
                        f"Boundary job {boundary_job_id} not found on stage {stage_id}"
                    )
                if current_machine_id == target_machine_id:
                    logging.debug(
                        "PW-CP slack reassignment not needed for %s.%s [%d, %d): "
                        "boundary job %s is already on the intended machine.",
                        stage_id,
                        target_machine_id,
                        slack_start,
                        slack_end,
                        boundary_job_id,
                    )
                    continue

                target_start_job_id = (
                    self._find_first_job_on_machine_starting_at_or_after(
                        candidate_schedule,
                        stage_id,
                        target_machine_id,
                        slack_end,
                    )
                )
                if target_start_job_id is None:
                    logging.debug(
                        "PW-CP slack reassignment skipped for %s.%s [%d, %d): "
                        "target machine has no suffix to exchange.",
                        stage_id,
                        target_machine_id,
                        slack_start,
                        slack_end,
                    )
                    continue

                logging.debug(
                    "PW-CP swapping suffixes on stage %s: %s.%s -> %s.%s "
                    "(boundary_job=%s, target_start_job=%s, slack=[%d, %d))",
                    stage_id,
                    current_machine_id,
                    boundary_job_id,
                    target_machine_id,
                    target_start_job_id,
                    boundary_job_id,
                    target_start_job_id,
                    slack_start,
                    slack_end,
                )
                from_suffix_job_ids = candidate_schedule.collect_stage_machine_suffix_job_ids(
                    stage_id, current_machine_id, boundary_job_id
                )
                to_suffix_job_ids = candidate_schedule.collect_stage_machine_suffix_job_ids(
                    stage_id, target_machine_id, target_start_job_id
                )
                candidate_schedule.swap_stage_machine_operation_sets(
                    stage_id=stage_id,
                    from_machine_id=current_machine_id,
                    from_job_ids=from_suffix_job_ids,
                    to_machine_id=target_machine_id,
                    to_job_ids=to_suffix_job_ids,
                    stage_2_job_2_duration=stage_2_job_2_p_dict,
                    do_make_semi_active=False,
                )

    def _apply_predecessor_machine_reassignment(
        self,
        candidate_schedule: HybridFlowshopLiteSchedule,
        solved_slack_intervals: SolvedSlackIntervals,
        partition: OperationPartition,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
    ) -> None:
        if not solved_slack_intervals:
            return

        right_time_fixed_job_ids_by_stage: dict[str, set[str]] = {}
        for job_id, stage_id, _machine_id in partition.right_time_fixed:
            right_time_fixed_job_ids_by_stage.setdefault(stage_id, set()).add(job_id)

        for stage_id in reversed(candidate_schedule.stages):
            stage_slacks = solved_slack_intervals.get(stage_id)
            if not stage_slacks:
                continue

            right_time_fixed_job_ids = right_time_fixed_job_ids_by_stage.get(stage_id, set())
            target_machine_ids = {machine_id for machine_id, _slack_start, _slack_end in stage_slacks}
            stage_job_infos: list[tuple[str, int, int, str]] = []
            removed_ops: set[tuple[str, str, str]] = set()
            for machine_id, start_time, end_time, job_id in candidate_schedule.iter_operations_on_stage(
                stage_id
            ):
                del start_time
                if machine_id not in target_machine_ids:
                    continue
                if job_id in right_time_fixed_job_ids:
                    continue
                stage_job_infos.append(
                    (
                        job_id,
                        end_time,
                        stage_2_job_2_p_dict[stage_id][job_id],
                        machine_id,
                    )
                )
                removed_ops.add((job_id, stage_id, machine_id))

            if not stage_job_infos:
                logging.debug(
                    "PW-CP predecessor reassignment skipped for %s: "
                    "no non-right-time-fixed operations found.",
                    stage_id,
                )
                continue

            # Sort by end_time primarily (descending), then by duration (descending), then by job_id (ascending)
            job_id_seq = [
                job_id
                for job_id, _end_time, _duration, _ in sorted(
                    stage_job_infos,
                    key=lambda item: (-item[1], -item[2], item[0]),
                )
            ]
            job_2_deadline = {
                job_id: end_time
                for job_id, end_time, _, _ in stage_job_infos
            }
            mc_2_lct = {
                machine_id: slack_start
                for machine_id, slack_start, _slack_end in stage_slacks
            }

            candidate_schedule.remove_operations(removed_ops)
            candidate_schedule.dispatch_stage_reversed_by_jobs(
                stage_id=stage_id,
                job_id_seq=job_id_seq,
                job_2_duration=stage_2_job_2_p_dict[stage_id],
                mc_2_lct=mc_2_lct,
                job_2_deadline=job_2_deadline,
            )

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
                objective_name=("makespan" if spec.is_last_batch else "right_slack"),
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
        self, stage_2_batches: dict[str, list[tuple[OperationRef, ...]]]
    ) -> int:
        batch_counts = {
            stage_id: len(batches) for stage_id, batches in stage_2_batches.items()
        }
        unique_counts = set(batch_counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"PW-CP requires identical batch counts across stages, got {batch_counts}."
            )
        return next(iter(unique_counts), 0)

    def _build_operation_partition(
        self,
        stage_2_batches: dict[str, list[tuple[OperationRef, ...]]],
        stage_ids: list[str],
        current_batch_idx: int,
        incumbent: HybridFlowshopLiteSchedule,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        left_profile_fixed_batch_count: int = 0,
        right_profile_fixed_batch_count: int = 0,
    ) -> tuple[OperationPartition, HybridFlowshopLiteSchedule]:
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

        Boundary profile is computed during partition creation:
        - right_boundary_profile: machine-order slack end times of right-justified ops
        """
        all_ops: list[OperationRef] = []
        l_tf_ops: list[OperationRef] = []
        l_pf_ops: list[OperationRef] = []
        unfixed_ops: list[OperationRef] = []
        r_pf_ops: list[OperationRef] = []
        r_tf_ops: list[OperationRef] = []

        for stage_id in stage_ids:
            batches = stage_2_batches[stage_id]
            all_ops.extend([op for batch in batches for op in batch])
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

        # Compute boundary profile during partition creation
        right_profile, right_justified_sched = self._build_right_boundary_profile(
            incumbent,
            all_ops,
            l_tf_ops,
            r_tf_ops,
            stage_2_job_2_p_dict,
        )

        partition = OperationPartition(
            left_time_fixed=tuple(sorted(l_tf_ops)),
            left_profile_fixed=tuple(sorted(l_pf_ops)),
            unfixed=tuple(sorted(unfixed_ops)),
            right_profile_fixed=tuple(sorted(r_pf_ops)),
            right_time_fixed=tuple(sorted(r_tf_ops)),
            right_boundary_profile=right_profile,
        )
        return partition, right_justified_sched

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
        highlight_op_set: set[tuple[str, str]] = {
            (job_id, stage_id) for job_id, stage_id, _ in spec.partition.unfixed
        }
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
            "highlight_ops": self._build_highlight_ops(spec.partition.unfixed),
            "batch_idx": spec.batch_idx,
            "subproblem_idx": spec.subproblem_idx,
            "accepted": accepted,
            "partition": {
                "left_time_fixed": list(spec.partition.left_time_fixed),
                "left_profile_fixed": list(spec.partition.left_profile_fixed),
                "unfixed": list(spec.partition.unfixed),
                "right_profile_fixed": list(spec.partition.right_profile_fixed),
                "right_time_fixed": list(spec.partition.right_time_fixed),
            },
            "right_boundary_profile": spec.right_boundary_profile,
        }
        dump_yaml(self._normalize_for_yaml(solution_dict), output_path)
