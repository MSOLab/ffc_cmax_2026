import logging
from dataclasses import asdict, dataclass, field
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
    StageFixedIntervals,
)
from hybridflowshop.cpsat_model_2.params import Params
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
)

OperationRef = tuple[str, str, str]
HighlightedOperationRef = tuple[str, str]
StageBoundaryProfile = dict[str, list[int]]
MachineAvailabilityProfile = dict[str, dict[str, int]]


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
    Encapsulates the three-way partition of operations for PW-CP subproblems.

    Designed to be extended with additional operation categories (e.g.,
    profile-fixed operations) without changing the interface.
    """

    left_time_fixed_ops: tuple[OperationRef, ...]
    optimization: tuple[OperationRef, ...]
    right_time_fixed_ops: tuple[OperationRef, ...]
    boundary_profile_fixed: tuple[OperationRef, ...] = field(default_factory=tuple)

    # Boundary profiles (computed during partition creation)
    left_boundary_profile: MachineAvailabilityProfile | None = None
    right_boundary_profile: StageBoundaryProfile | None = None
    right_fixed_intervals: StageFixedIntervals | None = None

    @property
    def all_operations(self) -> tuple[OperationRef, ...]:
        """Return all operations in the partition."""
        return (
            self.left_time_fixed_ops
            + self.optimization
            + self.right_time_fixed_ops
            + self.boundary_profile_fixed
        )

    @property
    def time_fixed_operations(self) -> tuple[OperationRef, ...]:
        """Return all operations that should have fixed start times in CP model."""
        return (
            self.left_time_fixed_ops
            + self.right_time_fixed_ops
            + self.boundary_profile_fixed
        )


@dataclass(frozen=True)
class PwCpSubproblemSpec:
    batch_idx: int
    subproblem_idx: int
    partition: OperationPartition
    left_boundary_profile: MachineAvailabilityProfile
    right_boundary_profile: StageBoundaryProfile
    right_fixed_intervals: StageFixedIntervals
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

    def run(
        self,
        ref_schedule: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        batch_size: int = 1,
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

        sub_obj_store = ObjValueBoundStore[int]()
        sub_obj_store.obj_value_series.name = "ObjVal after PW-CP batch"
        sub_obj_store.obj_bound_series.name = "ObjVal before PW-CP batch"

        self._st = PwCpRunState(
            timer=timer,
            incumbent=ref_schedule,
            sub_obj_store=sub_obj_store,
            subproblem_idx=0,
            subproblem_logs=[],
            max_time_per_batch=max_time_per_batch,
        )

        try:
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

                timelimit = self.ctx.get_remaining_time_limit(max_time_per_batch)
                if timelimit <= 0:
                    logging.info(
                        "PW-CP time limit exhausted before batch %d.",
                        batch_idx + 1,
                    )
                    break

                partition = self._build_operation_partition(
                    current_batches,
                    ref_schedule.stages,
                    batch_idx,
                    incumbent=st.incumbent,
                    stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                )
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
                    instance=instance,
                    stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                    max_time_per_batch=max_time_per_batch,
                    solver_thread_cnt=solver_thread_cnt,
                    use_lns_only=use_lns_only,
                    tighten_ranges=tighten_ranges,
                    link_job_completion=link_job_completion,
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

                if debug_export:
                    self._save_solution_dict(spec, st.incumbent, accepted=accepted)

                ts = st.timer.elapsed_sec
                st.sub_obj_store.add_obj_value(ts, int(st.incumbent.makespan), None)
                st.sub_obj_store.add_obj_bound(ts, int(st.incumbent.makespan), None)
                st.sub_obj_store.add_last_timestamp_note(
                    f"batch={batch_idx + 1}",
                    obj_value_is_valid=True,
                    obj_bound_is_valid=True,
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
        assert partition.left_boundary_profile is not None, "Left profile must be set"
        assert partition.right_boundary_profile is not None, "Right profile must be set"
        assert partition.right_fixed_intervals is not None, "Right intervals must be set"
        return PwCpSubproblemSpec(
            batch_idx=batch_idx,
            subproblem_idx=st.subproblem_idx,
            partition=partition,
            left_boundary_profile=partition.left_boundary_profile,
            right_boundary_profile=partition.right_boundary_profile,
            right_fixed_intervals=partition.right_fixed_intervals,
            is_last_batch=(batch_idx == max_batch_cnt - 1),
        )

    def _compute_left_boundary_profile(
        self, incumbent: HybridFlowshopLiteSchedule, cutoff: int
    ) -> MachineAvailabilityProfile:
        profile: MachineAvailabilityProfile = {
            stage_id: {mc_id: 0 for mc_id in incumbent.machines_per_stage[stage_id]}
            for stage_id in incumbent.stages
        }
        for stage_id in incumbent.stages:
            for mc_id in incumbent.machines_per_stage[stage_id]:
                latest_end = 0
                for start_time, end_time, _job_id in incumbent.get_job_sequence(
                    stage_id, mc_id
                ):
                    if start_time < cutoff:
                        latest_end = end_time
                profile[stage_id][mc_id] = int(latest_end)
        return profile

    def _build_right_guard_profile(
        self,
        incumbent: HybridFlowshopLiteSchedule,
        right_time_fixed_ops: tuple[OperationRef, ...],
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
    ) -> tuple[StageBoundaryProfile, StageFixedIntervals]:
        shifted = incumbent.deepcopy()
        shifted.make_right_justified(
            stage_2_job_2_p_dict,
            operation_set=set(right_time_fixed_ops),
        )
        start_map = shifted.get_jik_2_start_time_map()
        right_boundary_profile: StageBoundaryProfile = {}
        right_fixed_intervals: StageFixedIntervals = {}
        for stage_id in shifted.stages:
            stage_right_ops = [
                (job_id, mc_id)
                for job_id, op_stage_id, mc_id in right_time_fixed_ops
                if op_stage_id == stage_id
            ]
            stage_boundaries: list[int] = []
            stage_intervals: list[tuple[int, int, int]] = []

            for mc_id in shifted.machines_per_stage[stage_id]:
                machine_starts = [
                    int(start_map[job_id, stage_id, op_mc_id])
                    for job_id, op_mc_id in stage_right_ops
                    if op_mc_id == mc_id
                ]
                stage_boundaries.append(
                    min(machine_starts) if machine_starts else int(incumbent.makespan)
                )

            for job_id, mc_id in stage_right_ops:
                start = int(start_map[job_id, stage_id, mc_id])
                duration = int(stage_2_job_2_p_dict[stage_id][job_id])
                stage_intervals.append((start, start + duration, duration))

            right_boundary_profile[stage_id] = stage_boundaries
            right_fixed_intervals[stage_id] = stage_intervals
        return right_boundary_profile, right_fixed_intervals

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
        optimization_ops: tuple[OperationRef, ...],
    ) -> list[HighlightedOperationRef]:
        highlight_ops: list[HighlightedOperationRef] = []
        seen: set[HighlightedOperationRef] = set()
        for job_id, stage_id, _machine_id in optimization_ops:
            op_ref = (job_id, stage_id)
            if op_ref not in seen:
                seen.add(op_ref)
                highlight_ops.append(op_ref)
        return highlight_ops

    def _solve_subproblem(
        self,
        spec: PwCpSubproblemSpec,
        incumbent: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        max_time_per_batch: float | None,
        solver_thread_cnt: int,
        use_lns_only: bool,
        tighten_ranges: bool,
        link_job_completion: bool,
    ) -> HybridFlowshopLiteSchedule | None:
        _timer = ElapsedTimer()
        incumbent_obj = int(incumbent.makespan)
        logging.info(
            "PW-CP subproblem (batch=%d, subproblem=%d) starting. "
            "Incumbent makespan=%d, optimization ops=%d, time_fixed ops=%d.",
            spec.batch_idx + 1,
            spec.subproblem_idx,
            incumbent_obj,
            len(spec.partition.optimization),
            len(spec.partition.time_fixed_operations),
        )

        mdl, params, variables = self.builder.build(
            instance,
            incumbent.makespan,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        )
        BaseModelBuilder.apply_start_hints_from_start_time_map(
            mdl, params, variables, incumbent.get_jik_2_start_time_map()
        )
        BaseModelBuilder.apply_end_hints_from_end_time_map(
            mdl, params, variables, incumbent.get_jik_2_end_time_map()
        )

        frozen_start_time_map = {
            op: incumbent.get_jik_2_start_time_map()[op]
            for op in spec.partition.time_fixed_operations
        }
        BaseModelBuilder.add_start_time_freezed_operation_constraints(
            mdl, variables, frozen_start_time_map
        )

        timelimit = self.ctx.get_remaining_time_limit(max_time_per_batch)

        if spec.is_last_batch:
            # Final batch: minimize makespan directly
            mdl.minimize(variables.makespan)
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
                log_search_progress=True,
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
            # Non-final batch: maximize right guard slack
            BaseModelBuilder.add_right_guard_objective(
                mdl,
                params,
                variables,
                optimization_ops=spec.partition.optimization,
                right_boundary_profile=spec.right_boundary_profile,
                right_fixed_intervals=spec.right_fixed_intervals,
                horizon=incumbent.makespan,
            )
            self._apply_right_guard_hints(
                mdl=mdl,
                incumbent=incumbent,
                optimization_ops=spec.partition.optimization,
                right_boundary_profile=spec.right_boundary_profile,
                params=params,
            )
            mdl.add_hint(variables.makespan, incumbent_obj)
            logging.info(
                "Solving non-final batch with right guard slack maximization (timelimit=%.2fs).",
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
                log_search_progress=True,
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

    @staticmethod
    def _compute_right_guard_hint_values(
        incumbent: HybridFlowshopLiteSchedule,
        optimization_ops: tuple[OperationRef, ...],
        right_boundary_profile: StageBoundaryProfile,
        params: Params,
    ) -> dict[str, int]:
        """Compute incumbent values for right-guard objective auxiliaries."""
        stage_2_machine_2_intervals: dict[str, dict[str, list[tuple[int, int]]]] = {}
        start_time_map = incumbent.get_jik_2_start_time_map()
        end_time_map = incumbent.get_jik_2_end_time_map()

        for job_id, stage_id, machine_id in optimization_ops:
            stage_2_machine_2_intervals.setdefault(stage_id, {}).setdefault(
                machine_id, []
            ).append(
                (
                    int(start_time_map[job_id, stage_id, machine_id]),
                    int(end_time_map[job_id, stage_id, machine_id]),
                )
            )

        hint_values: dict[str, int] = {}
        stage_guard_mins: list[int] = []
        for stage_id, machine_ids in params.M_of.items():
            stage_extras: list[int] = []
            machine_2_intervals = stage_2_machine_2_intervals.get(stage_id, {})
            for machine_idx, (machine_id, guard_end) in enumerate(
                zip(machine_ids, right_boundary_profile[stage_id], strict=False),
                start=1,
            ):
                latest_end_before_guard = 0
                for start, end in machine_2_intervals.get(machine_id, []):
                    if start < guard_end:
                        latest_end_before_guard = max(
                            latest_end_before_guard, min(end, guard_end)
                        )
                extra = max(0, guard_end - latest_end_before_guard)
                hint_values[f"guard_extra_{stage_id}_{machine_idx}"] = extra
                hint_values[f"guard_start_{stage_id}_{machine_idx}"] = guard_end - extra
                stage_extras.append(extra)

            stage_guard_min = min(stage_extras, default=0)
            hint_values[f"stage_guard_min_{stage_id}"] = stage_guard_min
            stage_guard_mins.append(stage_guard_min)

        hint_values["global_guard_min"] = min(stage_guard_mins, default=0)
        return hint_values

    @staticmethod
    def _apply_right_guard_hints(
        mdl: CustomCpModel,
        incumbent: HybridFlowshopLiteSchedule,
        optimization_ops: tuple[OperationRef, ...],
        right_boundary_profile: StageBoundaryProfile,
        params: Params,
    ) -> None:
        """Apply incumbent hints for right-guard auxiliaries by variable name."""
        hint_values = PwCpConstructor._compute_right_guard_hint_values(
            incumbent=incumbent,
            optimization_ops=optimization_ops,
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
                objective_name=("makespan" if spec.is_last_batch else "right_guard_slack"),
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

    def _add_right_guard_objective(
        self,
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        optimization_ops: tuple[OperationRef, ...],
        right_boundary_profile: StageBoundaryProfile,
        right_fixed_intervals: StageFixedIntervals,
        horizon: int,
    ) -> cp_model.IntVar:
        """Thin wrapper around BaseModelBuilder.add_right_guard_objective."""
        return BaseModelBuilder.add_right_guard_objective(
            mdl,
            params,
            variables,
            optimization_ops,
            right_boundary_profile,
            right_fixed_intervals,
            horizon,
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
    ) -> OperationPartition:
        """
        Build operation partition based on batch indices.

        Partitioning rule:
        - left_time_fixed_ops: operations from batches with idx < current_batch_idx
        - optimization: operations from batch at current_batch_idx
        - right_time_fixed_ops: operations from batches with idx > current_batch_idx

        This leverages the existing time-based sorting in _build_stage_batches,
        where batch_idx=0 contains earliest operations and higher indices contain
        later operations.

        Boundary profiles are computed during partition creation:
        - left_boundary_profile: machine availability based on cutoff time
        - right_boundary_profile: machine-order guard end times of right-justified ops
        """
        left_ops: list[OperationRef] = []
        optimization_ops: list[OperationRef] = []
        right_ops: list[OperationRef] = []

        for stage_id in stage_ids:
            batches = stage_2_batches[stage_id]
            for idx, batch in enumerate(batches):
                if idx < current_batch_idx:
                    left_ops.extend(batch)
                elif idx == current_batch_idx:
                    optimization_ops.extend(batch)
                else:
                    right_ops.extend(batch)

        # Compute boundary profiles during partition creation
        cutoff = min(
            incumbent.get_jik_2_start_time_map()[op] for op in optimization_ops
        )
        left_profile = self._compute_left_boundary_profile(incumbent, cutoff)
        right_profile, right_fixed_intervals = self._build_right_guard_profile(
            incumbent,
            tuple(right_ops),
            stage_2_job_2_p_dict,
        )

        return OperationPartition(
            left_time_fixed_ops=tuple(sorted(left_ops)),
            optimization=tuple(sorted(optimization_ops)),
            right_time_fixed_ops=tuple(sorted(right_ops)),
            left_boundary_profile=left_profile,
            right_boundary_profile=right_profile,
            right_fixed_intervals=right_fixed_intervals,
        )

    def _save_solution_dict(
        self,
        spec: PwCpSubproblemSpec,
        incumbent: HybridFlowshopLiteSchedule,
        *,
        accepted: bool,
    ) -> None:
        output_path = self.ctx.get_file_path_for_subroutine(
            f"_pw_cp_batch_{spec.batch_idx + 1:03d}_solution.yaml"
        )
        solution_dict = {
            "start_time_map": incumbent.get_jik_2_start_time_map(),
            "end_time_map": incumbent.get_jik_2_end_time_map(),
            "highlight_ops": self._build_highlight_ops(spec.partition.optimization),
            "batch_idx": spec.batch_idx,
            "subproblem_idx": spec.subproblem_idx,
            "accepted": accepted,
            "partition": {
                "left_time_fixed_ops": list(spec.partition.left_time_fixed_ops),
                "optimization": list(spec.partition.optimization),
                "right_time_fixed_ops": list(spec.partition.right_time_fixed_ops),
                "boundary_profile_fixed": list(spec.partition.boundary_profile_fixed),
            },
            "left_boundary_profile": spec.left_boundary_profile,
            "right_boundary_profile": spec.right_boundary_profile,
            "right_fixed_intervals": spec.right_fixed_intervals,
        }
        dump_yaml(self._normalize_for_yaml(solution_dict), output_path)
