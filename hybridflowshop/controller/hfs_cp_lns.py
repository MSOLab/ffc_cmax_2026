import csv
import logging
import math
import random
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lb_bucket.cp import (
    build_retained_stage_cp_model,
    build_retained_stage_cp_result,
    build_trace_rows,
    extract_retained_stage_solution_rows,
    sanitize_optional_float,
    select_bottleneck_stage_by_average_load,
    write_retained_stage_cp_artifacts,
)
from lb_bucket.cp.post_dispatch import (
    PostRetainedCpDispatchDependencies,
    PostRetainedCpDispatchRunResult,
    run_post_retained_cp_dispatch,
    write_post_retained_cp_dispatch_artifacts,
)
from lb_bucket.mip.dispatch_windows import build_dispatch_window_lookup
from lb_bucket.mip.post_dispatch import (
    PostMipDispatchDependencies,
    PostMipDispatchRunResult,
    run_post_mip_dispatch,
    write_post_mip_dispatch_artifacts,
)
from lb_bucket.mip.solution_io import read_solution_payload, write_solution_payload
from lb_bucket.mip.visualization import write_solution_payload_visualizations
from lb_bucket.mip.warm_start import from_start_end_time_maps_create_ub_schedule
from mbls.cpsat import CpsatStatus, ObjectiveValueRecorder
import pandas as pd
from routix import DynamicDataObject, ElapsedTimer
from schore.parameters import JobStageProcessingTimeManager
from schore.parameters_examples.parallel_shop.identical_flow.hybrid_flowshop import (
    HybridFlowshopParameters,
    reverse_stages,
)

from hybridflowshop.controller.neh_cp import NehCpConstructor, NehCpResult
from hybridflowshop.controller.pw_cp import PwCpConstructor, PwCpResult
from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder
from hybridflowshop.dispatcher import (
    BN2DDispatcher,
    BN2DOption,
    JobDispatcher,
    MachineDispatcher,
    MixedDispatcher,
    StageDispatcher,
)
from hybridflowshop.dispatcher.utils import (
    build_schedule_from_stage_job_sequences_priority_score,
    from_job_sequence_get_schedule_mixed,
    improve_schedule_by_critical_cross_machine_insertions,
    improve_schedule_by_critical_stage_sequence_insertions,
    improve_schedule_by_critical_adjacent_swaps,
)
from hybridflowshop.lower_bounds import (
    chen_lb4_lower_bound,
    chen_lb4_stage_lower_bounds,
    santos_lower_bound,
    santos_stage_lower_bound,
    simple_job_lower_bound,
)
from hybridflowshop.report import HfsCpsatSolverReport, HfsSubroutineReport
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    OperationType,
    StageIdType,
    get_bottleneck_stage_job_sequence,
    get_first_stage_start_sequence,
    get_midpoint_sequence,
    validate_schedule,
)
from lb_bucket.mip.search import run_bucket_search_for_instance
from lb_bucket.mip.shared import (
    ModelStrengtheningOptions,
    PrecedenceOptions,
    SummaryBoundRecord,
    TwoBucketInstance,
    import_gurobi,
)

from .controller_core import HybridFlowShopCpLnsControllerCore
from .reactive.reactive_looper import ReactiveLooper


def _schedule_sequence_signature(
    schedule: HybridFlowshopLiteSchedule,
) -> tuple[tuple[Any, ...], ...]:
    machine_parts: list[tuple[Any, ...]] = []
    stage_parts: list[tuple[Any, ...]] = []
    job_index = {job_id: idx for idx, job_id in enumerate(schedule.jobs)}

    for stage_id in schedule.stages:
        stage_ops: list[tuple[int, int, int, str]] = []
        for machine_id in schedule.machines_per_stage[stage_id]:
            machine_seq = tuple(
                job_id
                for _start_time, _end_time, job_id in schedule.get_job_sequence(
                    stage_id, machine_id
                )
            )
            machine_parts.append(("m", stage_id, machine_id, machine_seq))
            for start_time, end_time, job_id in schedule.get_job_sequence(
                stage_id, machine_id
            ):
                stage_ops.append(
                    (int(start_time), int(end_time), job_index[job_id], job_id)
                )
        stage_parts.append(
            (
                "s",
                stage_id,
                tuple(job_id for *_unused, job_id in sorted(stage_ops)),
            )
        )

    return tuple(machine_parts + stage_parts)


class _RetainedStageSnapshotRecorder(ObjectiveValueRecorder):
    def __init__(
        self,
        *,
        build: Any,
        snapshot_limit: int,
        e_timer: ElapsedTimer,
        print_on_record: bool = False,
        log_level_on_record: int | None = None,
    ) -> None:
        super().__init__(
            e_timer=e_timer,
            print_on_record=print_on_record,
            log_level_on_record=log_level_on_record,
        )
        self._build = build
        self._snapshot_limit = max(0, int(snapshot_limit))
        self._seen_objectives: set[float] = set()
        self.snapshots: list[dict[str, Any]] = []

    def on_solution_callback(self) -> None:
        super().on_solution_callback()
        if self._snapshot_limit <= 0:
            return

        objective_ub = sanitize_optional_float(self.objective_value)
        if objective_ub is None:
            return
        objective_key = round(objective_ub, 9)
        if objective_key in self._seen_objectives:
            return
        self._seen_objectives.add(objective_key)

        runtime_sec = None
        objective_lb = sanitize_optional_float(self.best_objective_bound)
        if self.entries:
            runtime_sec = sanitize_optional_float(self.entries[-1][0])
            objective_lb = sanitize_optional_float(self.entries[-1][1].bound)

        rows: list[dict[str, Any]] = []
        for stage_id in self._build.retained_stage_ids:
            for job_id in self._build.params.j_list:
                rows.append(
                    {
                        "stage_id": stage_id,
                        "job_id": job_id,
                        "start": int(
                            self.Value(self._build.variables.op_start[job_id, stage_id])
                        ),
                        "end": int(
                            self.Value(self._build.variables.op_end[job_id, stage_id])
                        ),
                        "processing_time": self._build.params.p[job_id, stage_id],
                        "head": self._build.head_by_job_stage[job_id, stage_id],
                        "tail": self._build.tail_by_job_stage[job_id, stage_id],
                    }
                )

        if len(self.snapshots) >= self._snapshot_limit:
            self.snapshots.pop(0)
        self.snapshots.append(
            {
                "snapshot_index": len(self.snapshots),
                "runtime_sec": runtime_sec,
                "objective_ub": objective_ub,
                "objective_lb": objective_lb,
                "retained_solution_rows": rows,
            }
        )


class _FullScheduleSnapshotRecorder(ObjectiveValueRecorder):
    def __init__(
        self,
        *,
        instance: HybridFlowshopParameters,
        params: Any,
        variables: Any,
        stage_2_job_2_p_dict: Mapping[str, Mapping[str, int]],
        make_semi_active: bool,
        snapshot_limit: int,
        e_timer: ElapsedTimer,
        print_on_record: bool = False,
        log_level_on_record: int | None = None,
    ) -> None:
        super().__init__(
            e_timer=e_timer,
            print_on_record=print_on_record,
            log_level_on_record=log_level_on_record,
        )
        self._instance = instance
        self._params = params
        self._variables = variables
        self._stage_2_job_2_p_dict = stage_2_job_2_p_dict
        self._make_semi_active = bool(make_semi_active)
        self._snapshot_limit = max(0, int(snapshot_limit))
        self._seen_signatures: set[tuple[tuple[Any, ...], ...]] = set()
        self._record_count = 0
        self.snapshots: list[dict[str, Any]] = []

    def _build_schedule_from_callback_values(self) -> HybridFlowshopLiteSchedule:
        start_time_map: dict[str, dict[str, int]] = {}
        for stage_id in self._params.i_list:
            start_time_map[stage_id] = {}
            for job_id in self._params.j_list:
                start_time_map[stage_id][job_id] = int(
                    self.Value(self._variables.op_start[job_id, stage_id])
                )

        schedule = HybridFlowshopLiteSchedule(
            self._instance.job_id_list,
            self._instance.stage_id_list,
            self._instance.stage_2_machines_map,
        )
        job_index = {job_id: idx for idx, job_id in enumerate(self._params.j_list)}
        for stage_id in self._params.i_list:
            sorted_job_ids = sorted(
                self._params.j_list,
                key=lambda job_id: (
                    start_time_map[stage_id][job_id],
                    job_index[job_id],
                ),
            )
            for job_id in sorted_job_ids:
                schedule.add_operation_2_stage(
                    stage_id,
                    job_id,
                    int(self._params.p[job_id, stage_id]),
                    release_t=start_time_map[stage_id][job_id],
                )

        if self._make_semi_active:
            schedule.make_semi_active(self._stage_2_job_2_p_dict)
        return schedule

    def on_solution_callback(self) -> None:
        super().on_solution_callback()
        if self._snapshot_limit <= 0:
            return

        objective_ub = sanitize_optional_float(self.objective_value)
        if objective_ub is None:
            return

        try:
            schedule = self._build_schedule_from_callback_values()
        except Exception:
            logging.exception(
                "[Tau coarsened CP] Failed to capture surrogate CP snapshot."
            )
            return

        signature = _schedule_sequence_signature(schedule)
        if signature in self._seen_signatures:
            return
        self._seen_signatures.add(signature)

        runtime_sec = None
        objective_lb = sanitize_optional_float(self.best_objective_bound)
        if self.entries:
            runtime_sec = sanitize_optional_float(self.entries[-1][0])
            objective_lb = sanitize_optional_float(self.entries[-1][1].bound)

        snapshot_index = self._record_count
        self._record_count += 1
        if len(self.snapshots) >= self._snapshot_limit:
            self.snapshots.pop(0)
        self.snapshots.append(
            {
                "snapshot_index": snapshot_index,
                "runtime_sec": runtime_sec,
                "objective_ub": objective_ub,
                "objective_lb": objective_lb,
                "schedule": schedule,
            }
        )


@dataclass
class _BranchPortfolioStepResult:
    schedule: HybridFlowshopLiteSchedule
    sub_obj_store: Any | None = None
    last_obj_value: float | None = None


@dataclass
class _TauScheduleCandidate:
    source: str
    schedule: HybridFlowshopLiteSchedule
    surrogate_obj: float | None = None
    surrogate_bound: float | None = None


def _extract_retained_cp_trace_records(
    trace_rows: Sequence[Mapping[str, Any]],
    objective_key: str,
) -> list[tuple[float, float]]:
    records: list[tuple[float, float]] = []
    for row in trace_rows:
        runtime_sec = sanitize_optional_float(row.get("runtime_sec"))
        objective_value = sanitize_optional_float(row.get(objective_key))
        if runtime_sec is None or objective_value is None:
            continue
        records.append((runtime_sec, objective_value))
    return records


class HybridFlowShopCpLnsController(HybridFlowShopCpLnsControllerCore):
    """
    Controller for solving Hybrid Flow Shop problems using CP-based algorithms.
    """

    # Override

    def set_cp_model_as_base_cp_model(
        self, tighten_ranges: bool = False, link_job_completion: bool = False
    ) -> None:
        base_cp_model_options = {
            "tighten_ranges": bool(tighten_ranges),
            "link_job_completion": bool(link_job_completion),
        }
        self.cp_model = self.create_base_cp_model(
            tighten_ranges=base_cp_model_options["tighten_ranges"],
            link_job_completion=base_cp_model_options["link_job_completion"],
        )
        self.cp_model.set_num_base_constraints()
        self.base_cp_model_is_set = True
        self._base_cp_model_options = base_cp_model_options

    def _base_cp_model_matches_options(
        self,
        *,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
    ) -> bool:
        requested_options = {
            "tighten_ranges": bool(tighten_ranges),
            "link_job_completion": bool(link_job_completion),
        }
        existing_options = getattr(self, "_base_cp_model_options", None)
        if existing_options is None:
            # Historical controller state means the base model was built with
            # both optional domain/constraint tightenings disabled.
            return requested_options == {
                "tighten_ranges": False,
                "link_job_completion": False,
            }
        return existing_options == requested_options

    # Start subroutine definition

    # Subroutine: solve base CP model

    def _record_retained_cp_lb_summary(
        self,
        result: Any,
        *,
        call_context: str,
        start_sec: float,
        apply_elapsed_sec: float | None,
    ) -> None:
        stage_ids = list(getattr(result, "retained_stage_ids", ()) or [])
        record = {
            "call_context": call_context,
            "start_sec": start_sec,
            "bound": getattr(result, "certified_final_lb", None),
            "status": getattr(result, "status_name", None),
            "mode": getattr(result, "retained_stage_mode", None),
            "stage_ids": stage_ids,
            "bottleneck_stage_id": getattr(result, "bottleneck_stage_id", None),
            "solver_runtime_sec": getattr(result, "solver_runtime_sec", None),
            "apply_elapsed_sec": apply_elapsed_sec,
            "objective_ub": getattr(result, "objective_ub", None),
        }
        records = list(getattr(self, "retained_cp_lb_records", ()) or [])
        records.append(record)
        self.retained_cp_lb_records = records

        bound = sanitize_optional_float(record["bound"])
        best_record = getattr(self, "best_retained_cp_lb_record", None)
        best_bound = (
            sanitize_optional_float(best_record.get("bound"))
            if isinstance(best_record, dict)
            else None
        )
        if bound is not None and self.solution_manager._a_is_better_obj_bound(
            bound,
            best_bound,
        ):
            self.best_retained_cp_lb_record = record
            self.best_retained_cp_lb_result = result

        retained_best = getattr(self, "best_retained_cp_lb_record", record)
        logging.info(
            "[CP LB] Recorded retained-stage CP LB #%d: mode=%s bound=%s "
            "best_retained_cp_bound=%s",
            len(records),
            record["mode"],
            record["bound"],
            retained_best.get("bound") if isinstance(retained_best, dict) else None,
        )

    def solve_base_cp_model(
        self,
        computational_time: float | None,
        solver_thread_cnt: int,
        tl_nc_multiplier: float | None = None,
        use_final_time_reserve: bool = False,
        consume_all_remaining_with_final_reserve: bool = False,
        make_semi_active_after_cp: bool = False,
        is_initial_solution: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        cp_model_probing_level: int | None = None,
        cp_sat_params: Mapping[str, Any] | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Solve the base CP model for the hybrid flow shop problem.

        This method resets the CP model and solves it with the given computational time and number of workers.
        - If `is_initial_solution` is True, the solution is treated as the initial solution (e.g., for logging or summary purposes).
        - If `is_initial_solution` is False, the incumbent solution (if it exists) is applied as a hint to the CP model before solving.
        - If `draw_gantt` is True, a Gantt chart of the solution is generated after solving.

        Args:
            computational_time (float | None): The maximum computational time in seconds for solving the CP model.
                If None, uses the remaining time limit.
            solver_thread_cnt (int): The number of parallel workers (threads) to use during search.
            is_initial_solution (bool, optional): If True, marks this run as producing the initial solution (affects summary/logging). Defaults to False.
            encode_cumulative_as_reservoir (bool | None, optional): Whether to encode cumulative constraints as reservoir constraints. Defaults to None.
            expand_reservoir_constraints (bool | None, optional): Whether to expand reservoir constraints. Defaults to None.
            expand_reservoir_using_circuit (bool | None, optional): Whether to expand reservoir constraints using a circuit. Defaults to None.
            interleave_search (bool | None, optional): Whether to interleave the search. Defaults to None.
            use_lns_only (bool | None, optional): Whether to use LNS-only mode. Defaults to None.
            tighten_ranges (bool, optional): Tighten operation start/end domains
                with job-chain head/tail processing-time sums. Defaults to False.
            link_job_completion (bool, optional): Add job completion link
                constraints. Defaults to False.
            cp_model_probing_level (int | None, optional): The level of probing for the CP model. Defaults to None.
            log_search_progress (bool, optional): If True, logs the search progress during solving. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution after solving. Defaults to False.
        """
        sub_timer = ElapsedTimer()
        if self.base_cp_model_is_set and self._base_cp_model_matches_options(
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        ):
            self.cp_model.delete_added_constraints()
        else:
            self.set_cp_model_as_base_cp_model(
                tighten_ranges=tighten_ranges,
                link_job_completion=link_job_completion,
            )

        _should_be_init: bool = self.solution_manager.get_incumbent() is None
        _is_init: bool = _should_be_init or is_initial_solution
        _computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        if use_final_time_reserve:
            _computational_time = self.consume_reserved_final_time_sec(
                fallback_sec=_computational_time,
                consume_all_remaining=consume_all_remaining_with_final_reserve,
            )
        if _computational_time is not None:
            # Subtract model handling time from subroutine time limit
            _computational_time = max(0.0, _computational_time - sub_timer.elapsed_sec)

        if _is_init:
            report, solution = self.solve_current_cp_remaining_time_limit(
                _computational_time,
                solver_thread_cnt,
                make_semi_active_after_cp=make_semi_active_after_cp,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
                expand_reservoir_constraints=expand_reservoir_constraints,
                expand_reservoir_using_circuit=expand_reservoir_using_circuit,
                interleave_search=interleave_search,
                use_lns_only=use_lns_only,
                cp_model_probing_level=cp_model_probing_level,
                cp_sat_params=cp_sat_params,
                is_initial_solution=True,
                log_search_progress=log_search_progress,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )
        else:
            # If it is not an initial solution, apply the incumbent solution as a hint
            report, solution = self.solve_with_initial_solution(
                _computational_time,
                solver_thread_cnt,
                make_semi_active_after_cp=make_semi_active_after_cp,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
                expand_reservoir_constraints=expand_reservoir_constraints,
                expand_reservoir_using_circuit=expand_reservoir_using_circuit,
                interleave_search=interleave_search,
                use_lns_only=use_lns_only,
                cp_model_probing_level=cp_model_probing_level,
                cp_sat_params=cp_sat_params,
                log_search_progress=log_search_progress,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )
        logging.info(
            "Solved base CP model: %s with objValue= %d & objBound= %d",
            report.status,
            report.obj_value,
            report.obj_bound,
        )

        # Register report & solution
        self.solution_manager.register(report, solution)

        # Log (time, objective value & bound)
        log_time = self.timer.elapsed_sec
        _last_timestamp_note = self._get_call_context_of_current_method()

        obj_value = self.obj_store.get_last_obj_value()
        obj_value_is_valid = False
        if obj_value is not None:
            self.add_obj_value_log(log_time, obj_value, is_maximize=None)
            obj_value_is_valid = True

        obj_bound = self.obj_store.get_last_obj_bound()
        obj_bound_is_valid = False
        if obj_bound is not None:
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=None)
            obj_bound_is_valid = True

        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

    def solve_full_schedule_stage_precedence_cp(
        self,
        computational_time: float | None,
        solver_thread_cnt: int,
        tl_nc_multiplier: float | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        use_final_time_reserve: bool = False,
        consume_all_remaining_with_final_reserve: bool = False,
        no_improvement_timelimit: float | None = None,
        make_semi_active_after_cp: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
        cp_sat_params: Mapping[str, Any] | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Solve full CP with incumbent-derived stage profile precedences.

        This is a whole-schedule neighborhood: no operations are time-fixed.
        The incumbent is used as a hint, and its stage-level order is preserved
        only through precedence arcs. When a processing-time difference threshold
        is supplied, arcs are added only for job pairs whose processing times on
        that stage differ enough, leaving similar-duration jobs free to swap.
        """
        sub_timer = ElapsedTimer()
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "No incumbent HybridFlowshopLiteSchedule available for "
                "solve_full_schedule_stage_precedence_cp."
            )

        if self.base_cp_model_is_set:
            self.cp_model.delete_added_constraints()
        else:
            self.set_cp_model_as_base_cp_model()

        computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        if use_final_time_reserve:
            computational_time = self.consume_reserved_final_time_sec(
                fallback_sec=computational_time,
                consume_all_remaining=consume_all_remaining_with_final_reserve,
            )
        if computational_time is not None:
            computational_time = max(0.0, computational_time - sub_timer.elapsed_sec)

        logging.info(
            "Full-schedule stage-precedence CP starts: incumbent=%s, "
            "profile_fix_by_machine=%s, machine_precedence_stride=%d, "
            "min_p_diff=%s, min_p_diff_ratio=%s, time_limit=%s.",
            incumbent_solution.makespan,
            profile_fix_by_machine,
            machine_precedence_stride,
            stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio,
            computational_time,
        )

        BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
            self.cp_model,
            self.params,
            self.vars,
            incumbent_solution,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
        )

        report: HfsCpsatSolverReport
        solution: HybridFlowshopLiteSchedule | None
        try:
            report, solution = self.solve_with_initial_solution(
                computational_time,
                solver_thread_cnt,
                no_improvement_timelimit=no_improvement_timelimit,
                make_semi_active_after_cp=make_semi_active_after_cp,
                obj_value_is_valid=True,
                obj_bound_is_valid=False,
                encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
                expand_reservoir_constraints=expand_reservoir_constraints,
                expand_reservoir_using_circuit=expand_reservoir_using_circuit,
                interleave_search=interleave_search,
                use_lns_only=use_lns_only,
                cp_model_probing_level=cp_model_probing_level,
                cp_sat_params=cp_sat_params,
                log_search_progress=log_search_progress,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )
        finally:
            self.cp_model.delete_added_constraints()

        logging.info(
            "Full-schedule stage-precedence CP finished: status=%s objValue=%s "
            "globalObjBound=%s restrictedObjBound=%s.",
            report.status,
            report.obj_value,
            report.obj_bound,
            getattr(getattr(self, "solver", None), "best_objective_bound", None),
        )

        was_updated = self.solution_manager.register(report, solution)
        if was_updated:
            self.set_cp_model_as_base_cp_model()
            if draw_gantt:
                self.draw_incumbent_gantt()

        log_time = self.timer.elapsed_sec
        _last_timestamp_note = self._get_call_context_of_current_method()

        obj_value = self.obj_store.get_last_obj_value()
        obj_value_is_valid = False
        if obj_value is not None:
            self.add_obj_value_log(log_time, obj_value, is_maximize=None)
            obj_value_is_valid = True

        obj_bound = self.obj_store.get_last_obj_bound()
        obj_bound_is_valid = False
        if obj_bound is not None:
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=None)
            obj_bound_is_valid = True

        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

    def critical_cone_cp(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        repeat_count: int = 1,
        min_improvement: int = 1,
        slack_tolerance: int = 0,
        tail_time_ratio: float = 0.20,
        seed_op_count: int = 4,
        stage_radius: int = 1,
        time_radius: int | None = None,
        time_radius_ratio: float | None = 0.05,
        machine_neighbor_depth: int = 2,
        max_selected_ops: int | None = 500,
        include_seed_jobs_all_stages: bool = False,
        include_deadline_jobs_all_stages: bool = True,
        fix_outside_start_times: bool = True,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        no_improvement_timelimit: float | None = None,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Solve a small CP focused on the incumbent's tail critical cone.

        The selected cone contains tail critical operations, nearby competitors
        in time and stage, same-job neighboring stages, and adjacent machine
        neighbors. Operations outside the cone can be start-time fixed, making
        this a direct makespan-improving feasibility search rather than a broad
        whole-schedule polish.
        """
        if repeat_count < 1:
            raise ValueError("repeat_count must be >= 1")
        if min_improvement < 1:
            raise ValueError("min_improvement must be >= 1")

        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        per_repeat_time = (
            None
            if resolved_computational_time is None
            else max(0.0, float(resolved_computational_time) / float(repeat_count))
        )

        logging.info(
            "Critical-cone CP starts: repeat_count=%d, per_repeat_time=%s, "
            "min_improvement=%d, slack_tolerance=%d, tail_time_ratio=%s, "
            "seed_op_count=%d, stage_radius=%d, time_radius=%s, "
            "time_radius_ratio=%s, machine_neighbor_depth=%d, max_selected_ops=%s, "
            "include_seed_jobs_all_stages=%s, include_deadline_jobs_all_stages=%s, "
            "fix_outside_start_times=%s.",
            repeat_count,
            per_repeat_time,
            min_improvement,
            slack_tolerance,
            tail_time_ratio,
            seed_op_count,
            stage_radius,
            time_radius,
            time_radius_ratio,
            machine_neighbor_depth,
            max_selected_ops,
            include_seed_jobs_all_stages,
            include_deadline_jobs_all_stages,
            fix_outside_start_times,
        )

        for repeat_idx in range(repeat_count):
            if self.is_stopping_condition(log_reason_if_true=False):
                logging.info(
                    "Critical-cone CP stops before repeat %d/%d because a stopping "
                    "condition is already met.",
                    repeat_idx + 1,
                    repeat_count,
                )
                break

            self._fix_profile_solve_reset(
                lambda repeat_idx=repeat_idx: self.apply_critical_cone_operator(
                    min_improvement=min_improvement,
                    slack_tolerance=slack_tolerance,
                    tail_time_ratio=tail_time_ratio,
                    seed_op_count=seed_op_count,
                    stage_radius=stage_radius,
                    time_radius=time_radius,
                    time_radius_ratio=time_radius_ratio,
                    machine_neighbor_depth=machine_neighbor_depth,
                    max_selected_ops=max_selected_ops,
                    include_seed_jobs_all_stages=include_seed_jobs_all_stages,
                    include_deadline_jobs_all_stages=include_deadline_jobs_all_stages,
                    fix_outside_start_times=fix_outside_start_times,
                    profile_fix_by_machine=profile_fix_by_machine,
                    machine_precedence_stride=machine_precedence_stride,
                    repeat_idx=repeat_idx,
                ),
                per_repeat_time,
                solver_thread_cnt,
                no_improvement_timelimit=no_improvement_timelimit,
                make_semi_active_after_cp=make_semi_active_after_cp,
                use_lns_only=use_lns_only,
                obj_value_is_valid=True,
                obj_bound_is_valid=False,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )

    def apply_critical_cone_operator(
        self,
        *,
        min_improvement: int,
        slack_tolerance: int,
        tail_time_ratio: float,
        seed_op_count: int,
        stage_radius: int,
        time_radius: int | None,
        time_radius_ratio: float | None,
        machine_neighbor_depth: int,
        max_selected_ops: int | None,
        include_seed_jobs_all_stages: bool = False,
        include_deadline_jobs_all_stages: bool = True,
        fix_outside_start_times: bool,
        profile_fix_by_machine: bool,
        machine_precedence_stride: int,
        repeat_idx: int = 0,
    ) -> set[OperationType]:
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "No incumbent HybridFlowshopLiteSchedule available for critical_cone_cp."
            )

        selected_ops = self._select_critical_cone_operations(
            incumbent_solution,
            target_makespan=int(
                math.floor(incumbent_solution.makespan - min_improvement)
            ),
            slack_tolerance=slack_tolerance,
            tail_time_ratio=tail_time_ratio,
            seed_op_count=seed_op_count,
            stage_radius=stage_radius,
            time_radius=time_radius,
            time_radius_ratio=time_radius_ratio,
            machine_neighbor_depth=machine_neighbor_depth,
            max_selected_ops=max_selected_ops,
            include_seed_jobs_all_stages=include_seed_jobs_all_stages,
            include_deadline_jobs_all_stages=include_deadline_jobs_all_stages,
            seed_offset=repeat_idx * seed_op_count,
        )
        if not selected_ops:
            raise ValueError("Critical-cone selection produced no operations.")

        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            fix_start_times=fix_outside_start_times,
        )

        target_makespan = int(math.floor(incumbent_solution.makespan - min_improvement))
        self.cp_model.add(self.vars.makespan <= target_makespan)

        selected_stage_count = len({op[1] for op in selected_ops})
        selected_job_count = len({op[0] for op in selected_ops})
        logging.info(
            "Critical-cone operator repeat=%d selected_ops=%d selected_jobs=%d "
            "selected_stages=%d incumbent=%d target_makespan<=%d.",
            repeat_idx + 1,
            len(selected_ops),
            selected_job_count,
            selected_stage_count,
            incumbent_solution.makespan,
            target_makespan,
        )
        return selected_ops

    def _select_critical_cone_operations(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        target_makespan: int | None = None,
        slack_tolerance: int,
        tail_time_ratio: float,
        seed_op_count: int,
        stage_radius: int,
        time_radius: int | None,
        time_radius_ratio: float | None,
        machine_neighbor_depth: int,
        max_selected_ops: int | None,
        include_seed_jobs_all_stages: bool = False,
        include_deadline_jobs_all_stages: bool = True,
        seed_offset: int = 0,
    ) -> set[OperationType]:
        if slack_tolerance < 0:
            raise ValueError("slack_tolerance must be >= 0")
        if not (0 < tail_time_ratio <= 1):
            raise ValueError("tail_time_ratio must satisfy 0 < value <= 1")
        if seed_op_count < 1:
            raise ValueError("seed_op_count must be >= 1")
        if stage_radius < 0:
            raise ValueError("stage_radius must be >= 0")
        if time_radius is not None and time_radius < 0:
            raise ValueError("time_radius must be >= 0")
        if time_radius_ratio is not None and time_radius_ratio < 0:
            raise ValueError("time_radius_ratio must be >= 0")
        if machine_neighbor_depth < 0:
            raise ValueError("machine_neighbor_depth must be >= 0")
        if max_selected_ops is not None and max_selected_ops < 1:
            raise ValueError("max_selected_ops must be >= 1")

        analysis_schedule = schedule.deepcopy()
        analysis_schedule.make_semi_active(self.stage_2_job_2_p_dict)
        start_map = analysis_schedule.get_jik_2_start_time_map()
        end_map = analysis_schedule.get_jik_2_end_time_map()
        all_ops = list(start_map)
        if not all_ops:
            return set()

        stage_list = list(analysis_schedule.stages)
        stage_2_idx = {stage_id: idx for idx, stage_id in enumerate(stage_list)}
        op_by_job_stage = {(op[0], op[1]): op for op in all_ops}
        slack_map = analysis_schedule.calculate_slack(self.stage_2_job_2_p_dict)
        resolved_time_radius = self._resolve_critical_cone_time_radius(
            analysis_schedule,
            time_radius=time_radius,
            time_radius_ratio=time_radius_ratio,
        )

        def op_processing_time(op: OperationType) -> int:
            return int(self.stage_2_job_2_p_dict[op[1]][op[0]])

        def op_slack(op: OperationType) -> int:
            return int(slack_map.get(op[1], {}).get(op[0], 10**9))

        tail_cutoff = int(
            max(0, math.floor(analysis_schedule.makespan * (1.0 - tail_time_ratio)))
        )
        critical_ops = [op for op in all_ops if op_slack(op) <= slack_tolerance]
        seed_candidates = [op for op in critical_ops if end_map[op] >= tail_cutoff]
        if not seed_candidates:
            seed_candidates = critical_ops
        if not seed_candidates:
            last_stage = stage_list[-1]
            seed_candidates = [
                op
                for op in all_ops
                if op[1] == last_stage and end_map[op] == analysis_schedule.makespan
            ]
        if not seed_candidates:
            seed_candidates = sorted(all_ops, key=lambda op: end_map[op], reverse=True)[
                :seed_op_count
            ]

        def seed_key(op: OperationType) -> tuple[int, int, int, int]:
            return (
                end_map[op],
                -op_slack(op),
                op_processing_time(op),
                start_map[op],
            )

        ordered_seed_candidates = sorted(seed_candidates, key=seed_key, reverse=True)
        if len(ordered_seed_candidates) <= seed_op_count:
            seeds = ordered_seed_candidates
        else:
            start_idx = int(seed_offset) % len(ordered_seed_candidates)
            rotated = (
                ordered_seed_candidates[start_idx:]
                + ordered_seed_candidates[:start_idx]
            )
            seeds = rotated[:seed_op_count]

        critical_blocks = analysis_schedule.find_critical_blocks(
            self.stage_2_job_2_p_dict,
            include_singletons=True,
        )
        block_by_op: dict[OperationType, list[OperationType]] = {}
        for block in critical_blocks:
            for op in block:
                block_by_op[op] = block

        selected_ops: set[OperationType] = set()
        protected_ops: set[OperationType] = set()

        def add_job_stage(job_id: JobIdType, stage_id: StageIdType) -> None:
            op = op_by_job_stage.get((job_id, stage_id))
            if op is not None:
                selected_ops.add(op)

        def add_job_stage_band(job_id: JobIdType, center_stage_id: StageIdType) -> None:
            center_idx = stage_2_idx[center_stage_id]
            left_idx = max(0, center_idx - stage_radius)
            right_idx = min(len(stage_list) - 1, center_idx + stage_radius)
            for stage_id in stage_list[left_idx : right_idx + 1]:
                add_job_stage(job_id, stage_id)

        def add_job_all_stages(job_id: JobIdType) -> None:
            for stage_id in stage_list:
                add_job_stage(job_id, stage_id)

        def add_machine_neighbors(op: OperationType) -> None:
            if machine_neighbor_depth <= 0:
                return
            job_id, stage_id, mc_id = op
            job_tuple_seq = analysis_schedule.get_job_sequence(stage_id, mc_id)
            position = None
            for idx, (_start_time, _end_time, seq_job_id) in enumerate(job_tuple_seq):
                if seq_job_id == job_id:
                    position = idx
                    break
            if position is None:
                return
            left_idx = max(0, position - machine_neighbor_depth)
            right_idx = min(len(job_tuple_seq) - 1, position + machine_neighbor_depth)
            for idx in range(left_idx, right_idx + 1):
                neighbor_job = job_tuple_seq[idx][2]
                add_job_stage(neighbor_job, stage_id)

        for seed in seeds:
            selected_ops.add(seed)
            protected_ops.add(seed)
            for block_op in block_by_op.get(seed, [seed]):
                selected_ops.add(block_op)
                protected_ops.add(block_op)
            if include_seed_jobs_all_stages:
                add_job_all_stages(seed[0])
            else:
                add_job_stage_band(seed[0], seed[1])
            add_machine_neighbors(seed)

            seed_stage_idx = stage_2_idx[seed[1]]
            seed_left = start_map[seed] - resolved_time_radius
            seed_right = end_map[seed] + resolved_time_radius
            for op in all_ops:
                if abs(stage_2_idx[op[1]] - seed_stage_idx) > stage_radius:
                    continue
                if start_map[op] <= seed_right and end_map[op] >= seed_left:
                    selected_ops.add(op)

        for op in list(selected_ops):
            add_job_stage_band(op[0], op[1])
            add_machine_neighbors(op)

        deadline_ops: list[OperationType] = []
        if target_makespan is not None:
            last_stage = stage_list[-1]
            deadline_ops = [
                op
                for op in all_ops
                if op[1] == last_stage and end_map[op] > target_makespan
            ]
            for op in deadline_ops:
                selected_ops.add(op)
                protected_ops.add(op)
                if include_deadline_jobs_all_stages:
                    add_job_all_stages(op[0])
                else:
                    add_job_stage_band(op[0], op[1])
                add_machine_neighbors(op)

        if max_selected_ops is not None and len(selected_ops) > max_selected_ops:
            selected_ops = self._trim_critical_cone_operations(
                selected_ops,
                protected_ops=protected_ops,
                seeds=seeds,
                start_map=start_map,
                end_map=end_map,
                slack_map=slack_map,
                max_selected_ops=max_selected_ops,
            )

        stage_counts = Counter(op[1] for op in selected_ops)
        logging.info(
            "Critical-cone selection: makespan=%d target_makespan=%s tail_cutoff=%d "
            "time_radius=%d critical_ops=%d deadline_ops=%d seed_offset=%d seeds=%s "
            "selected_ops=%d stage_counts=%s.",
            analysis_schedule.makespan,
            target_makespan,
            tail_cutoff,
            resolved_time_radius,
            len(critical_ops),
            len(deadline_ops),
            seed_offset,
            seeds,
            len(selected_ops),
            dict(stage_counts),
        )
        return selected_ops

    @staticmethod
    def _resolve_critical_cone_time_radius(
        schedule: HybridFlowshopLiteSchedule,
        *,
        time_radius: int | None,
        time_radius_ratio: float | None,
    ) -> int:
        if time_radius is not None:
            return int(time_radius)
        if time_radius_ratio is None:
            return 0
        return max(0, int(math.ceil(max(1, schedule.makespan) * time_radius_ratio)))

    def _trim_critical_cone_operations(
        self,
        selected_ops: set[OperationType],
        *,
        protected_ops: set[OperationType],
        seeds: Sequence[OperationType],
        start_map: Mapping[OperationType, int],
        end_map: Mapping[OperationType, int],
        slack_map: Mapping[StageIdType, Mapping[JobIdType, int]],
        max_selected_ops: int,
    ) -> set[OperationType]:
        if len(selected_ops) <= max_selected_ops:
            return selected_ops

        def op_slack(op: OperationType) -> int:
            return int(slack_map.get(op[1], {}).get(op[0], 10**9))

        def distance_to_seed(op: OperationType) -> int:
            op_mid = (start_map[op] + end_map[op]) // 2
            op_stage_idx = self.instance.stage_id_list.index(op[1])
            best_distance = 10**9
            for seed in seeds:
                seed_mid = (start_map[seed] + end_map[seed]) // 2
                seed_stage_idx = self.instance.stage_id_list.index(seed[1])
                distance = abs(op_mid - seed_mid) + 10 * abs(
                    op_stage_idx - seed_stage_idx
                )
                if distance < best_distance:
                    best_distance = distance
            return best_distance

        trimmed = set(protected_ops)
        remaining = [op for op in selected_ops if op not in trimmed]
        remaining.sort(
            key=lambda op: (
                distance_to_seed(op),
                op_slack(op),
                -end_map[op],
            )
        )
        for op in remaining:
            if len(trimmed) >= max_selected_ops:
                break
            trimmed.add(op)

        logging.info(
            "Critical-cone selection trimmed from %d to %d ops (protected=%d, max=%d).",
            len(selected_ops),
            len(trimmed),
            len(protected_ops),
            max_selected_ops,
        )
        return trimmed

    def _record_last_neh_improvement(
        self,
        *,
        method_name: str,
        input_obj: float | int | None,
        output_obj: float | int | None,
        was_updated: bool,
        extra_label: str | None = None,
    ) -> None:
        input_obj_value = sanitize_optional_float(input_obj)
        output_obj_value = sanitize_optional_float(output_obj)
        improvement = None
        improvement_ratio = None
        if input_obj_value is not None and output_obj_value is not None:
            improvement = float(input_obj_value) - float(output_obj_value)
            if abs(float(input_obj_value)) > 1e-9:
                improvement_ratio = improvement / abs(float(input_obj_value))

        self.last_neh_method_name = method_name
        self.last_neh_call_context = self._get_call_context_of_current_method()
        self.last_neh_input_obj = input_obj_value
        self.last_neh_output_obj = output_obj_value
        self.last_neh_improvement = improvement
        self.last_neh_improvement_ratio = improvement_ratio
        self.last_neh_updated_incumbent = bool(was_updated)

        logging.info(
            "[NEH Improvement] method=%s context=%s input_obj=%s output_obj=%s "
            "improvement=%s ratio=%s updated_incumbent=%s%s",
            method_name,
            self.last_neh_call_context,
            input_obj_value,
            output_obj_value,
            improvement,
            improvement_ratio,
            bool(was_updated),
            "" if extra_label is None else f" {extra_label}",
        )

    def _last_neh_improvement_guard_allows(
        self,
        *,
        context_label: str,
        require_last_neh_incumbent_update: bool = True,
        min_last_neh_improvement: float | None = None,
        min_last_neh_improvement_ratio: float | None = None,
        run_if_last_neh_missing: bool = False,
    ) -> bool:
        last_updated = getattr(self, "last_neh_updated_incumbent", None)
        last_method_name = getattr(self, "last_neh_method_name", None)
        last_context = getattr(self, "last_neh_call_context", None)
        improvement = sanitize_optional_float(
            getattr(self, "last_neh_improvement", None)
        )
        improvement_ratio = sanitize_optional_float(
            getattr(self, "last_neh_improvement_ratio", None)
        )

        if last_updated is None:
            if run_if_last_neh_missing:
                logging.info(
                    "[%s] Running because last NEH improvement info is unavailable "
                    "and run_if_last_neh_missing=true.",
                    context_label,
                )
                return True
            logging.info(
                "[%s] Skipping because last NEH improvement info is unavailable.",
                context_label,
            )
            return False

        if require_last_neh_incumbent_update and not bool(last_updated):
            logging.info(
                "[%s] Skipping because last NEH did not update incumbent "
                "(method=%s context=%s improvement=%s ratio=%s).",
                context_label,
                last_method_name,
                last_context,
                improvement,
                improvement_ratio,
            )
            return False

        if min_last_neh_improvement is not None:
            if improvement is None:
                if run_if_last_neh_missing:
                    logging.info(
                        "[%s] Running because last NEH improvement is unavailable "
                        "and run_if_last_neh_missing=true.",
                        context_label,
                    )
                    return True
                logging.info(
                    "[%s] Skipping because last NEH improvement is unavailable "
                    "(method=%s context=%s).",
                    context_label,
                    last_method_name,
                    last_context,
                )
                return False
            if improvement < float(min_last_neh_improvement):
                logging.info(
                    "[%s] Skipping because last NEH improvement %.6f is below min %.6f "
                    "(method=%s context=%s).",
                    context_label,
                    improvement,
                    float(min_last_neh_improvement),
                    last_method_name,
                    last_context,
                )
                return False

        if min_last_neh_improvement_ratio is not None:
            if improvement_ratio is None:
                if run_if_last_neh_missing:
                    logging.info(
                        "[%s] Running because last NEH improvement ratio is unavailable "
                        "and run_if_last_neh_missing=true.",
                        context_label,
                    )
                    return True
                logging.info(
                    "[%s] Skipping because last NEH improvement ratio is unavailable "
                    "(method=%s context=%s).",
                    context_label,
                    last_method_name,
                    last_context,
                )
                return False
            if improvement_ratio < float(min_last_neh_improvement_ratio):
                logging.info(
                    "[%s] Skipping because last NEH improvement ratio %.6f is below "
                    "min %.6f (improvement=%s method=%s context=%s).",
                    context_label,
                    improvement_ratio,
                    float(min_last_neh_improvement_ratio),
                    improvement,
                    last_method_name,
                    last_context,
                )
                return False

        logging.info(
            "[%s] Running because last NEH passes improvement guard "
            "(updated_incumbent=%s method=%s context=%s improvement=%s ratio=%s).",
            context_label,
            bool(last_updated),
            last_method_name,
            last_context,
            improvement,
            improvement_ratio,
        )
        return True

    def solve_base_cp_model_if_last_neh_improved(
        self,
        computational_time: float | None,
        solver_thread_cnt: int,
        tl_nc_multiplier: float | None = None,
        use_final_time_reserve: bool = False,
        consume_all_remaining_with_final_reserve: bool = False,
        make_semi_active_after_cp: bool = False,
        is_initial_solution: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
        cp_sat_params: Mapping[str, Any] | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        skip_if_last_neh_not_improved: bool = True,
        require_last_neh_incumbent_update: bool = True,
        min_last_neh_improvement: float | None = None,
        min_last_neh_improvement_ratio: float | None = None,
        run_if_last_neh_missing: bool = False,
    ) -> None:
        """Run base CP only when the immediately preceding NEH produced a real gain."""
        if skip_if_last_neh_not_improved:
            if not self._last_neh_improvement_guard_allows(
                context_label="Last-NEH guarded base CP",
                require_last_neh_incumbent_update=require_last_neh_incumbent_update,
                min_last_neh_improvement=min_last_neh_improvement,
                min_last_neh_improvement_ratio=min_last_neh_improvement_ratio,
                run_if_last_neh_missing=run_if_last_neh_missing,
            ):
                return
        else:
            logging.info("[Last-NEH guarded base CP] Guard disabled; running base CP.")

        self.solve_base_cp_model(
            computational_time=computational_time,
            solver_thread_cnt=solver_thread_cnt,
            tl_nc_multiplier=tl_nc_multiplier,
            use_final_time_reserve=use_final_time_reserve,
            consume_all_remaining_with_final_reserve=consume_all_remaining_with_final_reserve,
            make_semi_active_after_cp=make_semi_active_after_cp,
            is_initial_solution=is_initial_solution,
            encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
            expand_reservoir_constraints=expand_reservoir_constraints,
            expand_reservoir_using_circuit=expand_reservoir_using_circuit,
            interleave_search=interleave_search,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
            cp_sat_params=cp_sat_params,
            log_search_progress=log_search_progress,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def _last_neh_adaptive_time_uses_improved_budget(
        self,
        *,
        context_label: str,
        require_last_neh_incumbent_update: bool = True,
        min_last_neh_improvement: float | None = None,
        min_last_neh_improvement_ratio: float | None = None,
        use_improved_budget_if_last_neh_missing: bool = False,
    ) -> bool:
        last_updated = getattr(self, "last_neh_updated_incumbent", None)
        last_method_name = getattr(self, "last_neh_method_name", None)
        last_context = getattr(self, "last_neh_call_context", None)
        improvement = sanitize_optional_float(
            getattr(self, "last_neh_improvement", None)
        )
        improvement_ratio = sanitize_optional_float(
            getattr(self, "last_neh_improvement_ratio", None)
        )

        if last_updated is None:
            logging.info(
                "[%s] Last NEH info is unavailable; using %s budget.",
                context_label,
                (
                    "improved"
                    if use_improved_budget_if_last_neh_missing
                    else "not-improved"
                ),
            )
            return bool(use_improved_budget_if_last_neh_missing)

        uses_improved_budget = True
        reason = "last NEH passes improvement guard"
        if require_last_neh_incumbent_update and not bool(last_updated):
            uses_improved_budget = False
            reason = "last NEH did not update incumbent"
        elif min_last_neh_improvement is not None and (
            improvement is None or improvement < float(min_last_neh_improvement)
        ):
            uses_improved_budget = False
            reason = (
                "last NEH improvement is unavailable"
                if improvement is None
                else "last NEH improvement is below threshold"
            )
        elif min_last_neh_improvement_ratio is not None and (
            improvement_ratio is None
            or improvement_ratio < float(min_last_neh_improvement_ratio)
        ):
            uses_improved_budget = False
            reason = (
                "last NEH improvement ratio is unavailable"
                if improvement_ratio is None
                else "last NEH improvement ratio is below threshold"
            )

        logging.info(
            "[%s] %s; using %s budget "
            "(updated_incumbent=%s method=%s context=%s improvement=%s ratio=%s).",
            context_label,
            reason,
            "improved" if uses_improved_budget else "not-improved",
            bool(last_updated),
            last_method_name,
            last_context,
            improvement,
            improvement_ratio,
        )
        return uses_improved_budget

    def solve_base_cp_model_with_last_neh_adaptive_time(
        self,
        computational_time: float | None,
        solver_thread_cnt: int,
        tl_nc_multiplier: float | None = None,
        tl_nc_multiplier_if_last_neh_not_improved: float | None = None,
        use_final_time_reserve: bool = False,
        consume_all_remaining_with_final_reserve: bool = False,
        make_semi_active_after_cp: bool = False,
        is_initial_solution: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
        cp_sat_params: Mapping[str, Any] | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        require_last_neh_incumbent_update: bool = True,
        min_last_neh_improvement: float | None = None,
        min_last_neh_improvement_ratio: float | None = None,
        use_improved_budget_if_last_neh_missing: bool = False,
        small_workload_threshold: int | None = None,
        small_workload_tl_nc_multiplier: float | None = None,
    ) -> None:
        uses_improved_budget = self._last_neh_adaptive_time_uses_improved_budget(
            context_label="Last-NEH adaptive base CP",
            require_last_neh_incumbent_update=require_last_neh_incumbent_update,
            min_last_neh_improvement=min_last_neh_improvement,
            min_last_neh_improvement_ratio=min_last_neh_improvement_ratio,
            use_improved_budget_if_last_neh_missing=use_improved_budget_if_last_neh_missing,
        )
        selected_tl_nc_multiplier = (
            tl_nc_multiplier
            if uses_improved_budget
            else tl_nc_multiplier_if_last_neh_not_improved
        )
        workload_size = None
        if (
            small_workload_threshold is not None
            and small_workload_tl_nc_multiplier is not None
        ):
            workload_size = self._get_instance_workload_size()
            if workload_size < int(small_workload_threshold):
                logging.info(
                    "[Last-NEH adaptive base CP] workload_size=%d < %d; "
                    "overriding selected_tl_nc_multiplier=%s with small_workload_tl_nc_multiplier=%s.",
                    workload_size,
                    int(small_workload_threshold),
                    selected_tl_nc_multiplier,
                    small_workload_tl_nc_multiplier,
                )
                selected_tl_nc_multiplier = float(small_workload_tl_nc_multiplier)
        logging.info(
            "[Last-NEH adaptive base CP] selected_tl_nc_multiplier=%s "
            "(improved_budget=%s, fallback_budget=%s, workload_size=%s, "
            "small_workload_threshold=%s, small_workload_budget=%s).",
            selected_tl_nc_multiplier,
            tl_nc_multiplier,
            tl_nc_multiplier_if_last_neh_not_improved,
            workload_size,
            small_workload_threshold,
            small_workload_tl_nc_multiplier,
        )
        self.solve_base_cp_model(
            computational_time=computational_time,
            solver_thread_cnt=solver_thread_cnt,
            tl_nc_multiplier=selected_tl_nc_multiplier,
            use_final_time_reserve=use_final_time_reserve,
            consume_all_remaining_with_final_reserve=consume_all_remaining_with_final_reserve,
            make_semi_active_after_cp=make_semi_active_after_cp,
            is_initial_solution=is_initial_solution,
            encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
            expand_reservoir_constraints=expand_reservoir_constraints,
            expand_reservoir_using_circuit=expand_reservoir_using_circuit,
            interleave_search=interleave_search,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
            cp_sat_params=cp_sat_params,
            log_search_progress=log_search_progress,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def set_final_time_reserve(
        self,
        tl_nc_multiplier: float | None = None,
        computational_time: float | None = None,
        time_limit_sec: float | None = None,
    ) -> None:
        """Reserve remaining global time for a final CP call later in the flow."""
        if (
            tl_nc_multiplier is None
            and computational_time is None
            and time_limit_sec is None
        ):
            tl_nc_multiplier = 0.1
        provided_count = sum(
            value is not None
            for value in (tl_nc_multiplier, computational_time, time_limit_sec)
        )
        if provided_count != 1:
            raise ValueError(
                "Provide exactly one of tl_nc_multiplier, computational_time, "
                "or time_limit_sec for set_final_time_reserve."
            )
        if computational_time is None:
            computational_time = time_limit_sec
        reserve_sec = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        if reserve_sec is None:
            raise ValueError("Resolved final time reserve cannot be None.")
        self.set_reserved_final_time_sec(
            float(reserve_sec),
            label="final solve_base_cp_model",
        )
        logging.info(
            "[Final Reserve] Reserved %.3f sec for the final CP method "
            "(remaining_before_reserve=%.3f).",
            float(reserve_sec),
            self.get_remaining_sec(),
        )

    def set_workload_adaptive_final_time_reserve(
        self,
        medium_workload_threshold: int = 1200,
        large_workload_threshold: int = 1800,
        small_workload_tl_nc_multiplier: float = 0.75,
        medium_workload_tl_nc_multiplier: float = 0.40,
        large_workload_tl_nc_multiplier: float = 0.25,
    ) -> None:
        """Reserve final CP time using only total workload size as the proxy."""
        workload_size = self._get_instance_workload_size()
        if workload_size >= int(large_workload_threshold):
            selected_tl_nc_multiplier = float(large_workload_tl_nc_multiplier)
            selected_band = "large_workload"
        elif workload_size >= int(medium_workload_threshold):
            selected_tl_nc_multiplier = float(medium_workload_tl_nc_multiplier)
            selected_band = "medium_workload"
        else:
            selected_tl_nc_multiplier = float(small_workload_tl_nc_multiplier)
            selected_band = "small_workload"
        logging.info(
            "[Workload Final Reserve] workload_size=%d selected=%s "
            "tl_nc_multiplier=%.3f (small<%d: %.3f, medium>=%d: %.3f, large>=%d: %.3f).",
            workload_size,
            selected_band,
            selected_tl_nc_multiplier,
            int(medium_workload_threshold),
            float(small_workload_tl_nc_multiplier),
            int(medium_workload_threshold),
            float(medium_workload_tl_nc_multiplier),
            int(large_workload_threshold),
            float(large_workload_tl_nc_multiplier),
        )
        self.set_final_time_reserve(tl_nc_multiplier=selected_tl_nc_multiplier)

    def set_stage_workload_adaptive_final_time_reserve(
        self,
        large_workload_threshold: int = 1800,
        large_workload_tl_nc_multiplier: float = 0.25,
        high_stage_min_stage_count: int = 15,
        high_stage_min_workload_size: int = 1200,
        high_stage_tl_nc_multiplier: float = 0.40,
        default_tl_nc_multiplier: float = 0.75,
    ) -> None:
        """Reserve final CP time with a middle band for high-stage workloads."""
        workload_size = self._get_instance_workload_size()
        stage_count = int(self.instance.stage_count)
        if workload_size > int(large_workload_threshold):
            selected_tl_nc_multiplier = float(large_workload_tl_nc_multiplier)
            selected_band = "large_workload"
        elif stage_count >= int(high_stage_min_stage_count) and workload_size >= int(
            high_stage_min_workload_size
        ):
            selected_tl_nc_multiplier = float(high_stage_tl_nc_multiplier)
            selected_band = "high_stage_mid_workload"
        else:
            selected_tl_nc_multiplier = float(default_tl_nc_multiplier)
            selected_band = "default"
        logging.info(
            "[Stage/Workload Final Reserve] workload_size=%d stage_count=%d selected=%s "
            "tl_nc_multiplier=%.3f (large>%d: %.3f, high_stage>=%d and workload>=%d: %.3f, default=%.3f).",
            workload_size,
            stage_count,
            selected_band,
            selected_tl_nc_multiplier,
            int(large_workload_threshold),
            float(large_workload_tl_nc_multiplier),
            int(high_stage_min_stage_count),
            int(high_stage_min_workload_size),
            float(high_stage_tl_nc_multiplier),
            float(default_tl_nc_multiplier),
        )
        self.set_final_time_reserve(tl_nc_multiplier=selected_tl_nc_multiplier)

    def _get_incumbent_bound_gap_ratio(self) -> float | None:
        incumbent_obj = self.solution_manager.best_obj_value
        incumbent_bound = self.solution_manager.best_obj_bound
        if (
            incumbent_obj is None
            or incumbent_bound is None
            or not math.isfinite(float(incumbent_obj))
            or not math.isfinite(float(incumbent_bound))
            or abs(float(incumbent_bound)) <= 1e-9
        ):
            return None
        return (float(incumbent_obj) - float(incumbent_bound)) / abs(
            float(incumbent_bound)
        )

    def _bound_gap_guard_allows(
        self,
        *,
        context_label: str,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
    ) -> bool:
        if (
            min_incumbent_bound_gap_ratio is None
            and max_incumbent_bound_gap_ratio is None
        ):
            return True
        gap_ratio = self._get_incumbent_bound_gap_ratio()
        incumbent_obj = self.solution_manager.best_obj_value
        incumbent_bound = self.solution_manager.best_obj_bound
        if gap_ratio is None:
            if run_if_bound_missing:
                logging.info(
                    "[%s] Running because incumbent/bound gap is unavailable "
                    "(obj=%s, bound=%s) and run_if_bound_missing=true.",
                    context_label,
                    incumbent_obj,
                    incumbent_bound,
                )
                return True
            logging.info(
                "[%s] Skipping because incumbent/bound gap is unavailable "
                "(obj=%s, bound=%s).",
                context_label,
                incumbent_obj,
                incumbent_bound,
            )
            return False
        if min_incumbent_bound_gap_ratio is not None and gap_ratio < float(
            min_incumbent_bound_gap_ratio
        ):
            logging.info(
                "[%s] Skipping: gap_ratio=%.6f is below min %.6f (obj=%s, bound=%s).",
                context_label,
                gap_ratio,
                float(min_incumbent_bound_gap_ratio),
                incumbent_obj,
                incumbent_bound,
            )
            return False
        if max_incumbent_bound_gap_ratio is not None and gap_ratio > float(
            max_incumbent_bound_gap_ratio
        ):
            logging.info(
                "[%s] Skipping: gap_ratio=%.6f is above max %.6f (obj=%s, bound=%s).",
                context_label,
                gap_ratio,
                float(max_incumbent_bound_gap_ratio),
                incumbent_obj,
                incumbent_bound,
            )
            return False
        logging.info(
            "[%s] Running: gap_ratio=%.6f is inside [%s, %s] (obj=%s, bound=%s).",
            context_label,
            gap_ratio,
            (
                "-inf"
                if min_incumbent_bound_gap_ratio is None
                else f"{float(min_incumbent_bound_gap_ratio):.6f}"
            ),
            (
                "inf"
                if max_incumbent_bound_gap_ratio is None
                else f"{float(max_incumbent_bound_gap_ratio):.6f}"
            ),
            incumbent_obj,
            incumbent_bound,
        )
        return True

    def _post_dispatch_improvement_guard_allows(
        self,
        *,
        context_label: str,
        min_post_dispatch_improvement: float | None = None,
        max_post_dispatch_improvement: float | None = None,
        min_post_dispatch_improvement_ratio: float | None = None,
        max_post_dispatch_improvement_ratio: float | None = None,
        run_if_post_dispatch_missing: bool = True,
    ) -> bool:
        current_obj = sanitize_optional_float(self.solution_manager.best_obj_value)
        post_dispatch_obj = sanitize_optional_float(
            getattr(self, "last_retained_cp_dispatch_obj", None)
        )
        if current_obj is None or post_dispatch_obj is None:
            if run_if_post_dispatch_missing:
                logging.info(
                    "[%s] Running because post-dispatch/current improvement is unavailable "
                    "(post_dispatch=%s, current=%s) and run_if_post_dispatch_missing=true.",
                    context_label,
                    post_dispatch_obj,
                    current_obj,
                )
                return True
            logging.info(
                "[%s] Skipping because post-dispatch/current improvement is unavailable "
                "(post_dispatch=%s, current=%s).",
                context_label,
                post_dispatch_obj,
                current_obj,
            )
            return False

        improvement = float(post_dispatch_obj) - float(current_obj)
        ratio = None
        if abs(float(current_obj)) > 1e-9:
            ratio = improvement / abs(float(current_obj))

        if min_post_dispatch_improvement is not None and improvement < float(
            min_post_dispatch_improvement
        ):
            logging.info(
                "[%s] Skipping: post_dispatch_improvement=%.6f is below min %.6f "
                "(post_dispatch=%s, current=%s).",
                context_label,
                improvement,
                float(min_post_dispatch_improvement),
                post_dispatch_obj,
                current_obj,
            )
            return False
        if max_post_dispatch_improvement is not None and improvement > float(
            max_post_dispatch_improvement
        ):
            logging.info(
                "[%s] Skipping: post_dispatch_improvement=%.6f is above max %.6f "
                "(post_dispatch=%s, current=%s).",
                context_label,
                improvement,
                float(max_post_dispatch_improvement),
                post_dispatch_obj,
                current_obj,
            )
            return False

        if ratio is None:
            if (
                min_post_dispatch_improvement_ratio is not None
                or max_post_dispatch_improvement_ratio is not None
            ):
                if run_if_post_dispatch_missing:
                    logging.info(
                        "[%s] Running because post-dispatch/current improvement ratio is unavailable "
                        "(post_dispatch=%s, current=%s) and run_if_post_dispatch_missing=true.",
                        context_label,
                        post_dispatch_obj,
                        current_obj,
                    )
                    return True
                logging.info(
                    "[%s] Skipping because post-dispatch/current improvement ratio is unavailable "
                    "(post_dispatch=%s, current=%s).",
                    context_label,
                    post_dispatch_obj,
                    current_obj,
                )
                return False
        else:
            if min_post_dispatch_improvement_ratio is not None and ratio < float(
                min_post_dispatch_improvement_ratio
            ):
                logging.info(
                    "[%s] Skipping: post_dispatch_improvement_ratio=%.6f is below min %.6f "
                    "(improvement=%.6f, post_dispatch=%s, current=%s).",
                    context_label,
                    ratio,
                    float(min_post_dispatch_improvement_ratio),
                    improvement,
                    post_dispatch_obj,
                    current_obj,
                )
                return False
            if max_post_dispatch_improvement_ratio is not None and ratio > float(
                max_post_dispatch_improvement_ratio
            ):
                logging.info(
                    "[%s] Skipping: post_dispatch_improvement_ratio=%.6f is above max %.6f "
                    "(improvement=%.6f, post_dispatch=%s, current=%s).",
                    context_label,
                    ratio,
                    float(max_post_dispatch_improvement_ratio),
                    improvement,
                    post_dispatch_obj,
                    current_obj,
                )
                return False

        logging.info(
            "[%s] Running: post_dispatch_improvement=%.6f ratio=%s is inside configured guards "
            "(post_dispatch=%s, current=%s).",
            context_label,
            improvement,
            "None" if ratio is None else f"{ratio:.6f}",
            post_dispatch_obj,
            current_obj,
        )
        return True

    def _get_last_retained_cp_restore_loss_ratio(self) -> float | None:
        cp_ub = sanitize_optional_float(
            getattr(self, "last_retained_cp_dispatch_cp_ub", None)
        )
        if cp_ub is None:
            retained_result = getattr(self, "last_retained_cp_lb_result", None)
            cp_ub = sanitize_optional_float(
                getattr(retained_result, "objective_ub", None)
            )
        post_dispatch_obj = sanitize_optional_float(
            getattr(self, "last_retained_cp_post_dispatch_obj", None)
        )
        if (
            cp_ub is None
            or post_dispatch_obj is None
            or not math.isfinite(float(cp_ub))
            or abs(float(cp_ub)) <= 1e-9
        ):
            return None
        return (float(post_dispatch_obj) - float(cp_ub)) / abs(float(cp_ub))

    def _retained_cp_restore_loss_guard_allows(
        self,
        *,
        context_label: str,
        min_retained_cp_restore_loss_ratio: float | None = None,
        max_retained_cp_restore_loss_ratio: float | None = None,
        run_if_retained_cp_restore_missing: bool = True,
    ) -> bool:
        if (
            min_retained_cp_restore_loss_ratio is None
            and max_retained_cp_restore_loss_ratio is None
        ):
            return True
        restore_loss_ratio = self._get_last_retained_cp_restore_loss_ratio()
        if restore_loss_ratio is None:
            if run_if_retained_cp_restore_missing:
                logging.info(
                    "[%s] Running because retained-CP restore loss is unavailable "
                    "and run_if_retained_cp_restore_missing=true.",
                    context_label,
                )
                return True
            logging.info(
                "[%s] Skipping because retained-CP restore loss is unavailable.",
                context_label,
            )
            return False
        if (
            min_retained_cp_restore_loss_ratio is not None
            and restore_loss_ratio < float(min_retained_cp_restore_loss_ratio)
        ):
            logging.info(
                "[%s] Skipping: retained_cp_restore_loss_ratio=%.6f is below min %.6f.",
                context_label,
                restore_loss_ratio,
                float(min_retained_cp_restore_loss_ratio),
            )
            return False
        if (
            max_retained_cp_restore_loss_ratio is not None
            and restore_loss_ratio > float(max_retained_cp_restore_loss_ratio)
        ):
            logging.info(
                "[%s] Skipping: retained_cp_restore_loss_ratio=%.6f is above max %.6f.",
                context_label,
                restore_loss_ratio,
                float(max_retained_cp_restore_loss_ratio),
            )
            return False
        logging.info(
            "[%s] Running: retained_cp_restore_loss_ratio=%.6f is inside configured guards.",
            context_label,
            restore_loss_ratio,
        )
        return True

    def clear_final_time_reserve(self) -> None:
        reserve_sec = self.get_reserved_final_time_sec()
        self.clear_reserved_final_time_sec()
        logging.info(
            "[Final Reserve] Cleared %.3f sec final time reserve.", reserve_sec
        )

    def solve_base_cp_model_from_final_time_reserve(
        self,
        computational_time: float | None = None,
        solver_thread_cnt: int = 16,
        tl_nc_multiplier: float | None = None,
        make_semi_active_after_cp: bool = False,
        is_initial_solution: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = True,
        consume_all_remaining: bool = True,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        cp_model_probing_level: int | None = None,
        cp_sat_params: Mapping[str, Any] | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Run base CP using the time saved by set_final_time_reserve."""
        if (
            not self.final_time_reserve_is_active()
            and computational_time is None
            and tl_nc_multiplier is None
        ):
            tl_nc_multiplier = 0.1
        self.solve_base_cp_model(
            computational_time=computational_time,
            solver_thread_cnt=solver_thread_cnt,
            tl_nc_multiplier=tl_nc_multiplier,
            use_final_time_reserve=True,
            consume_all_remaining_with_final_reserve=consume_all_remaining,
            make_semi_active_after_cp=make_semi_active_after_cp,
            is_initial_solution=is_initial_solution,
            encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
            expand_reservoir_constraints=expand_reservoir_constraints,
            expand_reservoir_using_circuit=expand_reservoir_using_circuit,
            interleave_search=interleave_search,
            use_lns_only=use_lns_only,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
            cp_model_probing_level=cp_model_probing_level,
            cp_sat_params=cp_sat_params,
            log_search_progress=log_search_progress,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def solve_base_cp_model_with_retained_hint(
        self,
        computational_time: float | None,
        solver_thread_cnt: int,
        tl_nc_multiplier: float | None = None,
        retained_stage_scope: str = "all",
        retained_stage_ids: Sequence[str] | None = None,
        cp_lb_dir: str | None = None,
        cp_lb_source_scenario_dir: str | None = None,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
        cp_sat_params: Mapping[str, Any] | None = None,
        log_search_progress: bool = False,
        prefer_incumbent_hints: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Solve base CP with retained-stage CP times as soft solution hints."""
        sub_timer = ElapsedTimer()
        if cp_lb_dir is not None or cp_lb_source_scenario_dir is not None:
            from lb_bucket.cp.solution_io import read_retained_stage_cp_artifacts

            if cp_lb_dir is not None:
                source_dir = Path(cp_lb_dir)
            else:
                source_dir = (
                    Path(str(cp_lb_source_scenario_dir))
                    / str(self.instance.name)
                    / "cp_lb"
                )

            retained_result, retained_solution_rows = read_retained_stage_cp_artifacts(
                source_dir
            )
            if retained_result is None or not retained_solution_rows:
                logging.warning(
                    "[CP Hint] No saved retained-stage CP artifacts found under %s.",
                    source_dir,
                )
                return
            self.last_retained_cp_lb_result = retained_result
            self.last_retained_cp_lb_retained_solution_rows = retained_solution_rows
            self.last_retained_cp_lb_apply_elapsed_sec = None
            logging.info(
                "[CP Hint] Loaded saved retained-stage CP artifacts from %s.",
                source_dir,
            )

        retained_solution_rows = getattr(
            self, "last_retained_cp_lb_retained_solution_rows", None
        )
        if not retained_solution_rows:
            logging.warning("[CP Hint] No retained-stage CP solution rows available.")
            return

        selected_stage_ids = self._resolve_retained_completion_stage_ids(
            retained_stage_scope=retained_stage_scope,
            retained_stage_ids=retained_stage_ids,
        )
        if not selected_stage_ids:
            logging.warning("[CP Hint] No retained stages selected for hinting.")
            return

        if self.base_cp_model_is_set:
            self.cp_model.delete_added_constraints()
        else:
            self.set_cp_model_as_base_cp_model()

        selected_stage_id_set = set(selected_stage_ids)
        start_hint_by_ji: dict[tuple[str, str], int] = {}
        end_hint_by_ji: dict[tuple[str, str], int] = {}
        incumbent_solution = self.solution_manager.get_incumbent()
        if isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            for (
                job_id,
                stage_id,
                _,
            ), start_time in incumbent_solution.get_jik_2_start_time_map().items():
                start_hint_by_ji[str(job_id), str(stage_id)] = int(start_time)
            for (
                job_id,
                stage_id,
                _,
            ), end_time in incumbent_solution.get_jik_2_end_time_map().items():
                end_hint_by_ji[str(job_id), str(stage_id)] = int(end_time)

        retained_hint_count = 0
        skipped_retained_hint_count = 0
        for row in retained_solution_rows:
            stage_id = str(row["stage_id"])
            if stage_id not in selected_stage_id_set:
                continue
            job_id = str(row["job_id"])
            if (job_id, stage_id) not in self.vars.op_start:
                continue
            if (
                prefer_incumbent_hints
                and (job_id, stage_id) in start_hint_by_ji
                and (job_id, stage_id) in end_hint_by_ji
            ):
                skipped_retained_hint_count += 1
                continue
            start_hint_by_ji[job_id, stage_id] = int(row["start"])
            end_hint_by_ji[job_id, stage_id] = int(row["end"])
            retained_hint_count += 1

        self.cp_model.clear_hints()
        start_hint_count = 0
        for (job_id, stage_id), start_time in start_hint_by_ji.items():
            if (job_id, stage_id) in self.vars.op_start:
                self.cp_model.add_hint(self.vars.op_start[job_id, stage_id], start_time)
                start_hint_count += 1
        end_hint_count = 0
        for (job_id, stage_id), end_time in end_hint_by_ji.items():
            if (job_id, stage_id) in self.vars.op_end:
                self.cp_model.add_hint(self.vars.op_end[job_id, stage_id], end_time)
                end_hint_count += 1
        if isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            self.cp_model.add_hint(self.vars.makespan, incumbent_solution.makespan)

        logging.info(
            "[CP Hint] Solving base CP with retained hints: scope=%s stages=%s "
            "retained_hints=%d skipped_retained_hints=%d "
            "start_hints=%d end_hints=%d prefer_incumbent_hints=%s.",
            retained_stage_scope,
            selected_stage_ids,
            retained_hint_count,
            skipped_retained_hint_count,
            start_hint_count,
            end_hint_count,
            prefer_incumbent_hints,
        )

        _computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        if _computational_time is not None:
            _computational_time = max(0.0, _computational_time - sub_timer.elapsed_sec)

        report, solution = self.solve_current_cp_remaining_time_limit(
            _computational_time,
            solver_thread_cnt,
            make_semi_active_after_cp=make_semi_active_after_cp,
            obj_value_is_valid=True,
            obj_bound_is_valid=True,
            is_initial_solution=incumbent_solution is None,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
            cp_sat_params=cp_sat_params,
            log_search_progress=log_search_progress,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )
        logging.info(
            "Solved base CP model with retained hints: %s with objValue=%s "
            "& objBound=%s",
            report.status,
            report.obj_value,
            report.obj_bound,
        )
        if not report.is_feasible or solution is None:
            logging.warning(
                "[CP Hint] Retained-hint base CP did not produce a feasible "
                "solution; keeping the existing incumbent and skipping report "
                "registration."
            )
            return
        self.solution_manager.register(report, solution)

        log_time = self.timer.elapsed_sec
        _last_timestamp_note = self._get_call_context_of_current_method()
        obj_value = self.obj_store.get_last_obj_value()
        obj_value_is_valid = False
        if obj_value is not None:
            self.add_obj_value_log(log_time, obj_value, is_maximize=None)
            obj_value_is_valid = True
        obj_bound = self.obj_store.get_last_obj_bound()
        obj_bound_is_valid = False
        if obj_bound is not None:
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=None)
            obj_bound_is_valid = True
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

    def retained_cp_integer_completion_ladder(
        self,
        solver_thread_cnt: int,
        steps: Sequence[Mapping[str, Any]] | None = None,
        cp_lb_dir: str | None = None,
        cp_lb_source_scenario_dir: str | None = None,
        stop_after_first_improvement: bool = False,
        min_improvement: float = 1.0,
        cleanup_added_constraints: bool = True,
        draw_gantt: bool = False,
    ) -> None:
        """Try short full-integer completions guided by retained-stage CP rows.

        The retained-stage CP is a relaxation over a subset of stages. Each
        ``complete`` step fixes selected retained-stage start times within a
        small window and solves the full HFS CP model, so every omitted stage is
        scheduled with integer machine/resource constraints. ``hint`` steps use
        the retained rows only as CP-SAT hints.
        """
        if cp_lb_dir is not None or cp_lb_source_scenario_dir is not None:
            from lb_bucket.cp.solution_io import read_retained_stage_cp_artifacts

            if cp_lb_dir is not None:
                source_dir = Path(cp_lb_dir)
            else:
                source_dir = (
                    Path(str(cp_lb_source_scenario_dir))
                    / str(self.instance.name)
                    / "cp_lb"
                )

            retained_result, retained_solution_rows = read_retained_stage_cp_artifacts(
                source_dir
            )
            if retained_result is None or not retained_solution_rows:
                logging.warning(
                    "[CP Complete Ladder] No saved retained-stage CP artifacts "
                    "found under %s.",
                    source_dir,
                )
                return
            self.last_retained_cp_lb_result = retained_result
            self.last_retained_cp_lb_retained_solution_rows = retained_solution_rows
            self.last_retained_cp_lb_apply_elapsed_sec = None
            logging.info(
                "[CP Complete Ladder] Loaded saved retained-stage CP artifacts "
                "from %s.",
                source_dir,
            )

        retained_solution_rows = getattr(
            self, "last_retained_cp_lb_retained_solution_rows", None
        )
        if not retained_solution_rows:
            logging.warning(
                "[CP Complete Ladder] No retained-stage CP solution rows are available."
            )
            return

        step_configs: Sequence[Mapping[str, Any]] = steps or (
            {
                "name": "preferred_anchor_slack_4",
                "mode": "complete",
                "retained_stage_scope": "preferred_anchor",
                "time_slack": 4,
                "tl_nc_multiplier": 0.004,
                "use_lns_only": False,
                "cp_model_probing_level": 1,
            },
            {
                "name": "first_bottlenecks_last_slack_8",
                "mode": "complete",
                "retained_stage_scope": "first_bottlenecks_last",
                "time_slack": 8,
                "tl_nc_multiplier": 0.006,
                "use_lns_only": False,
                "cp_model_probing_level": 1,
            },
            {
                "name": "all_retained_soft_hint",
                "mode": "hint",
                "retained_stage_scope": "all",
                "tl_nc_multiplier": 0.004,
                "use_lns_only": False,
                "prefer_incumbent_hints": False,
                "cp_model_probing_level": 1,
            },
        )

        initial_obj = self.solution_manager.best_obj_value
        logging.info(
            "[CP Complete Ladder] Starting retained CP integer completion ladder "
            "with %d steps, incumbent=%s.",
            len(step_configs),
            initial_obj,
        )

        try:
            for step_idx, raw_step_config in enumerate(step_configs, start=1):
                if self.is_stopping_condition(log_reason_if_true=False):
                    logging.info(
                        "[CP Complete Ladder] Stopping before step %d because the "
                        "global stopping condition is met.",
                        step_idx,
                    )
                    break

                step = dict(self._branch_portfolio_to_plain_obj(raw_step_config))
                step_name = str(step.pop("name", step.pop("label", f"step_{step_idx}")))
                mode = str(step.pop("mode", step.pop("method", "complete"))).lower()
                step.setdefault("solver_thread_cnt", solver_thread_cnt)
                step.setdefault("draw_gantt", draw_gantt)
                before_obj = self.solution_manager.best_obj_value

                logging.info(
                    "[CP Complete Ladder] Step %d/%d %s mode=%s starts with "
                    "incumbent=%s kwargs=%s.",
                    step_idx,
                    len(step_configs),
                    step_name,
                    mode,
                    before_obj,
                    step,
                )

                if mode in {"complete", "hard", "hard_anchor"}:
                    self.complete_from_retained_cp(**step)
                elif mode in {"hint", "soft", "soft_hint"}:
                    step.setdefault("computational_time", None)
                    self.solve_base_cp_model_with_retained_hint(**step)
                else:
                    raise ValueError(
                        "retained_cp_integer_completion_ladder step mode must be "
                        f"'complete' or 'hint'. Received: {mode!r}."
                    )

                after_obj = self.solution_manager.best_obj_value
                improvement = (
                    float(before_obj) - float(after_obj)
                    if before_obj is not None and after_obj is not None
                    else 0.0
                )
                logging.info(
                    "[CP Complete Ladder] Step %s finished: incumbent_before=%s "
                    "incumbent_after=%s improvement=%s.",
                    step_name,
                    before_obj,
                    after_obj,
                    improvement,
                )
                if stop_after_first_improvement and improvement >= float(
                    min_improvement
                ):
                    logging.info(
                        "[CP Complete Ladder] Stop after first improvement triggered "
                        "by step %s.",
                        step_name,
                    )
                    break
        finally:
            cp_model = getattr(self, "cp_model", None)
            if cleanup_added_constraints and getattr(
                self, "base_cp_model_is_set", False
            ):
                cp_model.delete_added_constraints()
            if cleanup_added_constraints and hasattr(cp_model, "clear_hints"):
                cp_model.clear_hints()

        logging.info(
            "[CP Complete Ladder] Finished with incumbent=%s (initial=%s).",
            self.solution_manager.best_obj_value,
            initial_obj,
        )

    # Helper method for LNS-CP

    def _fix_profile_solve_reset(
        self,
        profile_fixing_method: Callable,
        computational_time: float | None,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        pre_solve_visualizer: Callable[[], None] | None = None,
        post_solve_visualizer: (
            Callable[[HybridFlowshopLiteSchedule | None], None] | None
        ) = None,
    ):
        """Apply the profile fixing method, solve, and reset the model.

        Args:
            profile_fixing_method (Callable): A callable that applies the profile fixing method to the CP model.
            computational_time (float | None): The maximum computational time in seconds.
                If None, uses the remaining time limit.
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            no_improvement_timelimit (float | None, optional): If there is no improvement in this
                amount of time, the search will be stopped. If None, no timeout is set.
                Defaults to None.
            swap_before_cp (bool, optional): If True, applies a swap operator before CP solving &
                temporarily fix swapped operation's profile. Defaults to False.
            obj_value_is_valid (bool, optional): If True, adds the objective value log.
                Defaults to False.
            obj_bound_is_valid (bool, optional): If True, adds the objective bound log.
                Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        if getattr(self, "base_cp_model_is_set", False):
            self.cp_model.delete_added_constraints()
        else:
            self.set_cp_model_as_base_cp_model()

        last_stage = self.instance.stage_id_list[-1]  # 또는 ref_schedule.stages[-1]
        if swap_before_cp:
            swap_timer = ElapsedTimer()
            ref_schedule = self.solution_manager.get_incumbent()
            if ref_schedule is None:
                raise ValueError("No incumbent solution available for swap operator.")
            if not isinstance(ref_schedule, HybridFlowshopLiteSchedule):
                raise ValueError(
                    "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
                )
            ref_obj_value = ref_schedule.makespan
            start_time_map = ref_schedule.get_jik_2_start_time_map()
            end_time_map = ref_schedule.get_jik_2_end_time_map()
            last_stage_op_cnt = sum(
                1 for _ in ref_schedule.iter_operations_on_stage(last_stage)
            )
            # Make reference schedule semi-active for finding critical blocks
            ref_schedule.make_semi_active(self.stage_2_job_2_p_dict)

            # Repeat until swapped schedule has the same or better objective
            max_trial_cnt = 1000
            swap_success = False
            for trial in range(1, max_trial_cnt + 1):
                # 1st operation: choose from critical blocks(prefer non-singletons) at random
                critical_blocks = ref_schedule.find_critical_blocks(
                    self.stage_2_job_2_p_dict, include_singletons=True
                )
                target_blocks = [b for b in critical_blocks if len(b) > 1]
                if not target_blocks:
                    logging.debug(
                        "No multi-operation critical blocks found, using all critical blocks"
                    )
                    target_blocks = critical_blocks
                if not target_blocks:
                    logging.warning(
                        "No critical blocks found; skipping swap. "
                        "trial=%d/%d, makespan=%s, |start|=%d, |end|=%d, last_stage=%s, last_stage_ops=%d",
                        trial,
                        max_trial_cnt,
                        ref_schedule.makespan,
                        len(start_time_map),
                        len(end_time_map),
                        last_stage,
                        last_stage_op_cnt,
                    )
                    stage_op_cnt = {
                        s: sum(1 for _ in ref_schedule.iter_operations_on_stage(s))
                        for s in self.instance.stage_id_list
                    }
                    logging.warning("stage_op_cnt=%s", stage_op_cnt)
                    break
                target_block = random.choice(target_blocks)
                op_1 = random.choice(target_block)
                target_stage = op_1[1]

                # 2nd operation: randomly choose operation of other job on the same stage
                op_2_candid_list: list[tuple[str, str, str]] = [
                    (
                        op_2_candid_info[3],  # job_id
                        target_stage,
                        op_2_candid_info[0],  # mc_id
                    )
                    for op_2_candid_info in ref_schedule.iter_operations_on_stage(
                        target_stage
                    )
                    if op_2_candid_info[3] != op_1[0]  # different job
                ]
                op_2 = random.choice(op_2_candid_list)

                # Swap
                swapped_schedule = ref_schedule.deepcopy()
                swapped_schedule.swap_two_operations_within_stage(
                    target_stage, op_1[0], op_2[0], self.stage_2_job_2_p_dict
                )

                # Check objective
                swapped_obj_value = swapped_schedule.makespan
                if swapped_obj_value <= ref_obj_value:
                    # Add to solution manager
                    report = self._make_subroutine_report(
                        elapsed_time=swap_timer.elapsed_sec,
                        obj_value=float(swapped_obj_value),
                        obj_bound=None,
                        is_init=False,
                        subroutine_name="_fix_profile_solve_reset",
                        progress_obj_value_records=[
                            (swap_timer.elapsed_sec, float(swapped_obj_value))
                        ],
                    )
                    self.solution_manager.register(report, swapped_schedule)
                    # If two operations does not overlap,
                    if not self.open_intervals_overlap(
                        start_time_map[op_1],
                        end_time_map[op_1],
                        start_time_map[op_2],
                        end_time_map[op_2],
                    ):
                        # Determine precedence based on original schedule's start times
                        op_1_start = start_time_map[op_1]
                        op_2_start = start_time_map[op_2]
                        if op_1_start <= op_2_start:
                            j_l, j_f = op_1[0], op_2[0]
                        else:
                            j_l, j_f = op_2[0], op_1[0]
                        # Add the precedence constraint to the CP model to enforce the swap
                        BaseModelBuilder.add_fixed_operation_precedence_constraint(
                            self.cp_model,
                            self.params,
                            self.vars,
                            j_l,
                            j_f,
                            target_stage,
                        )
                        logging.info(
                            f"Swap SUCCESS trial {trial}/{max_trial_cnt}: "
                            f"op1={op_1} op2={op_2} stage={target_stage} "
                            f"ref_obj={ref_obj_value} new_obj={swapped_obj_value} "
                            f"precedence={j_l} -> {j_f}"
                        )
                    else:
                        # Do not add precedence constraint if they overlap
                        logging.info(
                            f"Swap SUCCESS trial {trial}/{max_trial_cnt}: "
                            f"op1={op_1} op2={op_2} stage={target_stage} "
                            f"ref_obj={ref_obj_value} new_obj={swapped_obj_value} "
                        )
                    swap_success = True
                    break
                else:
                    logging.debug(
                        f"Swap trial {trial}/{max_trial_cnt}: Objective degraded "
                        f"(ref={ref_obj_value}, new={swapped_obj_value}), retrying..."
                    )

            if not swap_success:
                logging.warning(
                    f"Swap operator failed after {max_trial_cnt} trials. "
                    f"Proceeding without swap."
                )

        profile_fixing_method()
        if pre_solve_visualizer is not None:
            pre_solve_visualizer()
        report, solution = self.solve_with_initial_solution(
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )
        if post_solve_visualizer is not None:
            post_solve_visualizer(solution)
        self.cp_model.delete_added_constraints()

        # Register report & solution
        was_updated: bool = self.solution_manager.register(report, solution)
        if was_updated:
            # Re-define base CP model with the new makespan
            self.set_cp_model_as_base_cp_model()

        # Log (time, objective value & bound)
        log_time = self.timer.elapsed_sec
        _last_timestamp_note = self._get_call_context_of_current_method()

        obj_value = self.obj_store.get_last_obj_value()
        obj_value_is_valid = False
        if obj_value is not None:
            self.add_obj_value_log(log_time, obj_value, is_maximize=None)
            obj_value_is_valid = True

        obj_bound = self.obj_store.get_last_obj_bound()
        obj_bound_is_valid = False
        if obj_bound is not None:
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=None)
            obj_bound_is_valid = True

        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

    def _resolve_tl_nc_computational_time(
        self,
        *,
        computational_time: float | None,
        tl_nc_multiplier: float | None,
    ) -> float | None:
        if tl_nc_multiplier is None:
            return computational_time
        if tl_nc_multiplier <= 0:
            raise ValueError("tl_nc_multiplier must be > 0")
        return (
            float(tl_nc_multiplier)
            * float(self.instance.job_count)
            * float(self.instance.stage_count)
        )

    def _make_tau_coarsened_instance(self, tau: int) -> HybridFlowshopParameters:
        tau_int = int(tau)
        if tau_int <= 0:
            raise ValueError("tau must be a positive integer")

        job_ids = list(self.instance.job_id_list)
        stage_ids = list(self.instance.stage_id_list)
        rows: list[list[int]] = []
        for job_id in job_ids:
            rows.append(
                [
                    max(
                        1,
                        int(math.ceil(float(self.job_2_stage_2_p_dict[job_id][stage_id]) / tau_int)),
                    )
                    for stage_id in stage_ids
                ]
            )

        p_manager = JobStageProcessingTimeManager(
            name=f"{self.instance.name}_tau{tau_int}_P",
            df=pd.DataFrame(rows),
        )
        return HybridFlowshopParameters(
            name=f"{self.instance.name}_tau{tau_int}",
            job_id_list=job_ids,
            stage_id_list=stage_ids,
            stage_2_machines_map={
                stage_id: list(self.instance.stage_2_machines_map[stage_id])
                for stage_id in stage_ids
            },
            p_manager=p_manager,
        )

    @contextmanager
    def _temporary_instance_context(self, instance: HybridFlowshopParameters):
        old_instance = self.instance
        old_job_2_stage_2_p_dict = self.job_2_stage_2_p_dict
        old_stage_2_job_2_p_dict = self.stage_2_job_2_p_dict
        self.instance = instance
        self.job_2_stage_2_p_dict = instance.job_2_stage_2_p_map
        self.stage_2_job_2_p_dict = instance.stage_2_job_2_p_map
        try:
            yield
        finally:
            self.instance = old_instance
            self.job_2_stage_2_p_dict = old_job_2_stage_2_p_dict
            self.stage_2_job_2_p_dict = old_stage_2_job_2_p_dict

    def _get_tau_surrogate_dispatch_schedules(
        self,
        scaled_instance: HybridFlowshopParameters,
        *,
        cap_portions: Sequence[float] = (0.25, 0.30),
        method_list: Sequence[str] = ("bn2d_all_stages", "best_of_mixed_dispatches"),
        include_machine_then_job_variants: bool = True,
        candidate_top_k: int = 1,
    ) -> list[tuple[str, HybridFlowshopLiteSchedule]]:
        top_k = max(1, int(candidate_top_k))
        entries: list[tuple[int, str, HybridFlowshopLiteSchedule]] = []
        seen_signatures: set[tuple[tuple[Any, ...], ...]] = set()
        with self._temporary_instance_context(scaled_instance):
            for cap in cap_portions:
                machine_then_job_values = (
                    (True, False) if include_machine_then_job_variants else (False,)
                )
                for machine_then_job in machine_then_job_values:
                    candidate_schedules = self._get_selected_dispatch_candidate_schedules(
                        left_cap_portion=float(cap),
                        right_cap_portion=float(cap),
                        mixed_schedule_for_former_stages=True,
                        mixed_schedule_for_later_stages=True,
                        machine_then_job=machine_then_job,
                        head_for_all_stages=False,
                        normalize_by_stage_cnt=False,
                        method_list=list(method_list),
                        draw_gantt=False,
                    )
                    for method_name, schedule in candidate_schedules.items():
                        if schedule is None:
                            continue
                        signature = _schedule_sequence_signature(schedule)
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        obj = int(schedule.makespan)
                        cap_label = f"cap{float(cap):.3f}".replace(".", "p")
                        order_label = "mtj" if machine_then_job else "jtm"
                        label = f"{cap_label}_{order_label}_{method_name}"
                        entries.append((obj, label, schedule))
        entries.sort(key=lambda item: (item[0], item[1]))
        selected = [(label, schedule) for _obj, label, schedule in entries[:top_k]]
        if selected:
            logging.info(
                "[Tau coarsened CP] Selected %s/%s surrogate dispatch candidates "
                "for tau instance %s: best=%s.",
                len(selected),
                len(entries),
                scaled_instance.name,
                entries[0][0],
            )
        return selected

    def _get_tau_surrogate_dispatch_schedule(
        self,
        scaled_instance: HybridFlowshopParameters,
        *,
        cap_portions: Sequence[float] = (0.25, 0.30),
        method_list: Sequence[str] = ("bn2d_all_stages", "best_of_mixed_dispatches"),
        include_machine_then_job_variants: bool = True,
    ) -> HybridFlowshopLiteSchedule | None:
        schedules = self._get_tau_surrogate_dispatch_schedules(
            scaled_instance,
            cap_portions=cap_portions,
            method_list=method_list,
            include_machine_then_job_variants=include_machine_then_job_variants,
            candidate_top_k=1,
        )
        return schedules[0][1] if schedules else None

    @staticmethod
    def _sum_processing_time_horizon(instance: HybridFlowshopParameters) -> int:
        p_map = instance.job_2_stage_2_p_map
        horizon = 0
        for job_id in instance.job_id_list:
            for stage_id in instance.stage_id_list:
                horizon += int(p_map[job_id][stage_id])
        return max(1, horizon)

    def _create_schedule_from_solved_cp_vars_for_instance(
        self,
        *,
        instance: HybridFlowshopParameters,
        params: Any,
        variables: Any,
        stage_2_job_2_p_dict: Mapping[str, Mapping[str, int]],
        make_semi_active: bool,
    ) -> HybridFlowshopLiteSchedule:
        start_time_map: dict[str, dict[str, int]] = {}
        for stage_id in params.i_list:
            start_time_map[stage_id] = {}
            for job_id in params.j_list:
                start_time_map[stage_id][job_id] = int(
                    self.solver.Value(variables.op_start[job_id, stage_id])
                )

        schedule = HybridFlowshopLiteSchedule(
            instance.job_id_list,
            instance.stage_id_list,
            instance.stage_2_machines_map,
        )
        job_index = {job_id: idx for idx, job_id in enumerate(params.j_list)}
        for stage_id in params.i_list:
            sorted_job_ids = sorted(
                params.j_list,
                key=lambda job_id: (
                    start_time_map[stage_id][job_id],
                    job_index[job_id],
                ),
            )
            for job_id in sorted_job_ids:
                schedule.add_operation_2_stage(
                    stage_id,
                    job_id,
                    int(params.p[job_id, stage_id]),
                    release_t=start_time_map[stage_id][job_id],
                )

        if make_semi_active:
            schedule.make_semi_active(stage_2_job_2_p_dict)
        return schedule

    def _restore_original_schedule_from_tau_schedule(
        self,
        tau_schedule: HybridFlowshopLiteSchedule,
        *,
        restore_mode: str,
        make_semi_active: bool,
    ) -> HybridFlowshopLiteSchedule:
        mode = restore_mode.replace("-", "_").lower()
        schedule = self.create_empty_schedule_from_ins(self.instance)
        job_index = {job_id: idx for idx, job_id in enumerate(self.instance.job_id_list)}

        if mode in {"machine", "machine_sequence", "machine_profile"}:
            for stage_id in self.instance.stage_id_list:
                for machine_id in self.instance.stage_2_machines_map[stage_id]:
                    for _start, _end, job_id in tau_schedule.get_job_sequence(
                        stage_id, machine_id
                    ):
                        schedule.add_operation_2_mc(
                            stage_id,
                            machine_id,
                            job_id,
                            int(self.job_2_stage_2_p_dict[job_id][stage_id]),
                        )
        elif mode in {"stage", "stage_sequence", "stage_profile"}:
            start_time_map = tau_schedule.get_jik_2_start_time_map()
            end_time_map = tau_schedule.get_jik_2_end_time_map()
            for stage_id in self.instance.stage_id_list:
                stage_ops = [
                    (job_id, start_time, end_time_map[job_id, op_stage_id, machine_id])
                    for (job_id, op_stage_id, machine_id), start_time in start_time_map.items()
                    if op_stage_id == stage_id
                ]
                for job_id, _start, _end in sorted(
                    stage_ops,
                    key=lambda item: (item[1], item[2], job_index[item[0]]),
                ):
                    schedule.add_operation_2_stage(
                        stage_id,
                        job_id,
                        int(self.job_2_stage_2_p_dict[job_id][stage_id]),
                    )
        else:
            raise ValueError(
                "restore_mode must be one of machine_sequence or stage_sequence"
            )

        if make_semi_active:
            schedule.make_semi_active(self.stage_2_job_2_p_dict)
        validate_schedule(schedule, self.stage_2_job_2_p_dict)
        return schedule

    def _solve_local_base_cp_candidate(
        self,
        *,
        instance: HybridFlowshopParameters,
        computational_time: float,
        solver_thread_cnt: int,
        use_lns_only: bool | None,
        cp_model_probing_level: int | None,
        cp_sat_params: Mapping[str, Any] | None,
        make_semi_active: bool,
        reference_schedule: HybridFlowshopLiteSchedule | None = None,
        add_reference_precedence: bool = True,
        reference_profile_fix_by_machine: bool = False,
        stage_2_job_2_p_dict: Mapping[str, Mapping[str, int]] | None = None,
        snapshot_solution_limit: int = 0,
    ) -> tuple[Any, HybridFlowshopLiteSchedule | None, list[dict[str, Any]]]:
        if stage_2_job_2_p_dict is None:
            stage_2_job_2_p_dict = instance.stage_2_job_2_p_map

        builder = BaseModelBuilder()
        horizon = self._sum_processing_time_horizon(instance)
        if reference_schedule is not None:
            horizon = max(horizon, int(math.ceil(reference_schedule.makespan)))

        mdl, params, variables = builder.build(instance, horizon)
        mdl.minimize(variables.makespan)

        if reference_schedule is not None:
            if add_reference_precedence:
                BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
                    mdl,
                    params,
                    variables,
                    reference_schedule,
                    profile_fix_by_machine=reference_profile_fix_by_machine,
                )
            BaseModelBuilder.apply_start_hints_from_start_time_map(
                mdl,
                params,
                variables,
                reference_schedule.get_jik_2_start_time_map(),
            )
            BaseModelBuilder.apply_end_hints_from_end_time_map(
                mdl,
                params,
                variables,
                reference_schedule.get_jik_2_end_time_map(),
            )
            mdl.add_hint(variables.makespan, int(reference_schedule.makespan))

        solve_timer = ElapsedTimer()
        snapshot_recorder: _FullScheduleSnapshotRecorder | None = None
        if int(snapshot_solution_limit) > 0:
            snapshot_recorder = _FullScheduleSnapshotRecorder(
                instance=instance,
                params=params,
                variables=variables,
                stage_2_job_2_p_dict=stage_2_job_2_p_dict,
                make_semi_active=make_semi_active,
                snapshot_limit=int(snapshot_solution_limit),
                e_timer=solve_timer,
                print_on_record=False,
                log_level_on_record=logging.INFO,
            )

        solver_report = self.solve_cp_model_2(
            mdl,
            computational_time=computational_time,
            solver_thread_cnt=solver_thread_cnt,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
            cp_sat_params=cp_sat_params,
            e_timer=solve_timer,
            solution_callback=snapshot_recorder,
        )
        snapshots = snapshot_recorder.snapshots if snapshot_recorder is not None else []
        if not solver_report.status.is_feasible:
            return solver_report, None, snapshots

        schedule = self._create_schedule_from_solved_cp_vars_for_instance(
            instance=instance,
            params=params,
            variables=variables,
            stage_2_job_2_p_dict=stage_2_job_2_p_dict,
            make_semi_active=make_semi_active,
        )
        validate_schedule(schedule, stage_2_job_2_p_dict)
        valid_snapshots: list[dict[str, Any]] = []
        for snapshot in snapshots:
            try:
                validate_schedule(snapshot["schedule"], stage_2_job_2_p_dict)
            except Exception:
                logging.exception(
                    "[Tau coarsened CP] Dropping invalid surrogate CP snapshot."
                )
                continue
            valid_snapshots.append(snapshot)
        return solver_report, schedule, valid_snapshots

    def initialize_by_tau_coarsened_cp(
        self,
        tau_values: Sequence[int] = (5,),
        surrogate_computational_time: float | None = None,
        surrogate_tl_nc_multiplier: float | None = 0.02,
        polish_computational_time: float | None = None,
        polish_tl_nc_multiplier: float | None = 0.02,
        restore_modes: Sequence[str] = ("machine_sequence", "stage_sequence"),
        polish_profile_modes: Sequence[str] = ("restore",),
        surrogate_dispatch_before_cp: bool = False,
        include_surrogate_dispatch_candidate: bool = False,
        surrogate_dispatch_cap_portions: Sequence[float] = (0.25, 0.30),
        surrogate_dispatch_method_list: Sequence[str] = (
            "bn2d_all_stages",
            "best_of_mixed_dispatches",
        ),
        surrogate_dispatch_include_machine_then_job_variants: bool = True,
        surrogate_dispatch_candidate_top_k: int = 1,
        surrogate_neh_enabled: bool = False,
        surrogate_neh_position: str = "after_cp",
        surrogate_neh_tau_values: Sequence[int] | None = None,
        surrogate_neh_sources: Sequence[str] = ("dispatch_hint_cp",),
        surrogate_neh_added_batch_size: int = 20,
        surrogate_neh_added_batch_sizes: Sequence[int] | None = None,
        surrogate_neh_sequential: bool = False,
        surrogate_neh_also_solve_dispatch_cp: bool = False,
        surrogate_neh_cp_tl_nc_multiplier: float | None = 0.001,
        surrogate_neh_use_lns_only: bool = True,
        surrogate_neh_minimize_sum_ci_lex: bool = False,
        surrogate_neh_cp_tl_nc_multiplier_2nd_obj: float | None = None,
        surrogate_pw_cp_enabled: bool = False,
        surrogate_pw_cp_tau_values: Sequence[int] | None = None,
        surrogate_pw_cp_batch_size: int | None = None,
        surrogate_pw_cp_batch_size_ratio: float | None = 0.05,
        surrogate_pw_cp_unfixed_batch_count_min: int = 2,
        surrogate_pw_cp_unfixed_batch_count_max: int = 4,
        surrogate_pw_cp_step_size: int = 1,
        surrogate_pw_cp_lr_profile_fixed_batch_count: int = 1,
        surrogate_pw_cp_left_profile_fixed_batch_count: int = 0,
        surrogate_pw_cp_right_profile_fixed_batch_count: int = 0,
        surrogate_pw_cp_enable_promotion_profile_fixed: bool = True,
        surrogate_pw_cp_profile_fix_by_machine: bool = False,
        surrogate_pw_cp_machine_precedence_stride: int = 1,
        surrogate_pw_cp_stage_precedence_min_processing_time_diff: int | None = None,
        surrogate_pw_cp_stage_precedence_min_processing_time_diff_ratio: float
        | None = None,
        surrogate_pw_cp_non_time_fixed_op_time_limit_multiplier: float | None = 0.002,
        surrogate_pw_cp_max_time_per_batch: float | None = None,
        surrogate_pw_cp_use_lns_only: bool = False,
        surrogate_pw_cp_tighten_ranges: bool = False,
        surrogate_pw_cp_stop_on_no_improvement: bool = False,
        solver_thread_cnt: int = 16,
        surrogate_use_lns_only: bool | None = False,
        surrogate_cp_snapshot_solution_limit: int = 0,
        polish_use_lns_only: bool | None = True,
        cp_model_probing_level: int | None = 1,
        cp_sat_params: Mapping[str, Any] | None = None,
        make_semi_active: bool = True,
        save_candidate_artifacts: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Build an initial solution from tau-coarsened surrogate CP schedules.

        Each surrogate replaces p[j,i] by ceil(p[j,i] / tau).  A CP solve on the
        coarsened instance gives a stage/machine ordering, which is restored on
        the original processing times and optionally polished with the restored
        ordering used as precedence information.
        """
        sub_timer = ElapsedTimer()
        normalized_tau_values = [int(tau) for tau in tau_values]
        if not normalized_tau_values:
            logging.warning("[Tau coarsened CP] No tau values were provided.")
            return
        if not restore_modes:
            logging.warning("[Tau coarsened CP] No restore modes were provided.")
            return
        if not polish_profile_modes:
            logging.warning("[Tau coarsened CP] No polish profile modes were provided.")
            return
        surrogate_neh_tau_set = (
            {int(tau) for tau in surrogate_neh_tau_values}
            if surrogate_neh_tau_values is not None
            else None
        )
        surrogate_pw_cp_tau_set = (
            {int(tau) for tau in surrogate_pw_cp_tau_values}
            if surrogate_pw_cp_tau_values is not None
            else None
        )
        surrogate_neh_source_set = {str(source) for source in surrogate_neh_sources}
        surrogate_neh_position_norm = surrogate_neh_position.replace("-", "_").lower()
        if surrogate_neh_position_norm not in {"before_cp", "after_cp"}:
            raise ValueError(
                "surrogate_neh_position must be one of before_cp or after_cp."
            )
        surrogate_neh_batch_sizes = (
            [int(size) for size in surrogate_neh_added_batch_sizes]
            if surrogate_neh_added_batch_sizes is not None
            else [int(surrogate_neh_added_batch_size)]
        )
        surrogate_neh_batch_sizes = [
            size for size in surrogate_neh_batch_sizes if size > 0
        ]
        if surrogate_neh_enabled and not surrogate_neh_batch_sizes:
            raise ValueError(
                "surrogate NEH requires at least one positive batch size."
            )
        if surrogate_pw_cp_enabled:
            if surrogate_pw_cp_batch_size is None:
                if surrogate_pw_cp_batch_size_ratio is None:
                    raise ValueError(
                        "surrogate PW-CP requires either batch_size or "
                        "batch_size_ratio."
                    )
                if surrogate_pw_cp_batch_size_ratio <= 0:
                    raise ValueError(
                        "surrogate_pw_cp_batch_size_ratio must be > 0."
                    )
            elif surrogate_pw_cp_batch_size <= 0:
                raise ValueError("surrogate_pw_cp_batch_size must be positive.")
            if surrogate_pw_cp_unfixed_batch_count_min < 1:
                raise ValueError(
                    "surrogate_pw_cp_unfixed_batch_count_min must be >= 1."
                )
            if (
                surrogate_pw_cp_unfixed_batch_count_max
                < surrogate_pw_cp_unfixed_batch_count_min
            ):
                raise ValueError(
                    "surrogate_pw_cp_unfixed_batch_count_max must be >= min."
                )
            if surrogate_pw_cp_step_size < 1:
                raise ValueError("surrogate_pw_cp_step_size must be >= 1.")
            if surrogate_pw_cp_lr_profile_fixed_batch_count < 0:
                raise ValueError(
                    "surrogate_pw_cp_lr_profile_fixed_batch_count must be >= 0."
                )
            if surrogate_pw_cp_left_profile_fixed_batch_count < 0:
                raise ValueError(
                    "surrogate_pw_cp_left_profile_fixed_batch_count must be >= 0."
                )
            if surrogate_pw_cp_right_profile_fixed_batch_count < 0:
                raise ValueError(
                    "surrogate_pw_cp_right_profile_fixed_batch_count must be >= 0."
                )
            if surrogate_pw_cp_machine_precedence_stride < 1:
                raise ValueError(
                    "surrogate_pw_cp_machine_precedence_stride must be >= 1."
                )
            if (
                surrogate_pw_cp_non_time_fixed_op_time_limit_multiplier is not None
                and surrogate_pw_cp_non_time_fixed_op_time_limit_multiplier <= 0
            ):
                raise ValueError(
                    "surrogate_pw_cp_non_time_fixed_op_time_limit_multiplier "
                    "must be > 0."
                )
            if (
                surrogate_pw_cp_stage_precedence_min_processing_time_diff is not None
                and surrogate_pw_cp_stage_precedence_min_processing_time_diff < 0
            ):
                raise ValueError(
                    "surrogate_pw_cp_stage_precedence_min_processing_time_diff "
                    "must be >= 0."
                )
            if (
                surrogate_pw_cp_stage_precedence_min_processing_time_diff_ratio
                is not None
                and surrogate_pw_cp_stage_precedence_min_processing_time_diff_ratio < 0
            ):
                raise ValueError(
                    "surrogate_pw_cp_stage_precedence_min_processing_time_diff_ratio "
                    "must be >= 0."
                )

        raw_surrogate_time = self._resolve_tl_nc_computational_time(
            computational_time=surrogate_computational_time,
            tl_nc_multiplier=surrogate_tl_nc_multiplier,
        )
        raw_polish_time = self._resolve_tl_nc_computational_time(
            computational_time=polish_computational_time,
            tl_nc_multiplier=polish_tl_nc_multiplier,
        )

        candidate_rows: list[dict[str, Any]] = []
        progress_records: list[tuple[float, float]] = []
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_obj: float | None = None
        best_label = ""

        def maybe_record_candidate(
            *,
            tau: int,
            surrogate_source: str,
            mode: str,
            phase: str,
            schedule: HybridFlowshopLiteSchedule,
            surrogate_obj: float | None,
            surrogate_bound: float | None,
            expanded_surrogate_ub: float | None,
            elapsed_sec: float,
        ) -> None:
            nonlocal best_schedule, best_obj, best_label
            obj = float(schedule.makespan)
            candidate_rows.append(
                {
                    "tau": tau,
                    "surrogate_source": surrogate_source,
                    "restore_mode": mode,
                    "phase": phase,
                    "obj": obj,
                    "surrogate_obj": surrogate_obj,
                    "surrogate_bound": surrogate_bound,
                    "expanded_surrogate_ub": expanded_surrogate_ub,
                    "elapsed_sec": elapsed_sec,
                }
            )
            if best_obj is None or obj < best_obj:
                best_schedule = schedule
                best_obj = obj
                best_label = (
                    f"tau={tau} source={surrogate_source} mode={mode} phase={phase}"
                )
                progress_records.append((elapsed_sec, obj))
                logging.info(
                    "[Tau coarsened CP] New best original makespan=%s from %s "
                    "(surrogate_obj=%s expanded_ub=%s).",
                    obj,
                    best_label,
                    surrogate_obj,
                    expanded_surrogate_ub,
                )

        def add_tau_schedule_candidate(
            tau_candidates: list[_TauScheduleCandidate],
            seen_signatures: set[tuple[tuple[Any, ...], ...]],
            *,
            source: str,
            schedule: HybridFlowshopLiteSchedule,
            surrogate_obj: float | None = None,
            surrogate_bound: float | None = None,
        ) -> bool:
            signature = _schedule_sequence_signature(schedule)
            if signature in seen_signatures:
                return False
            seen_signatures.add(signature)
            tau_candidates.append(
                _TauScheduleCandidate(
                    source=source,
                    schedule=schedule,
                    surrogate_obj=surrogate_obj,
                    surrogate_bound=surrogate_bound,
                )
            )
            return True

        def add_cp_solution_candidates(
            tau_candidates: list[_TauScheduleCandidate],
            seen_signatures: set[tuple[tuple[Any, ...], ...]],
            *,
            source: str,
            report: Any,
            final_schedule: HybridFlowshopLiteSchedule | None,
            snapshots: Sequence[Mapping[str, Any]],
        ) -> None:
            if final_schedule is not None:
                add_tau_schedule_candidate(
                    tau_candidates,
                    seen_signatures,
                    source=source,
                    schedule=final_schedule,
                    surrogate_obj=(
                        float(report.obj_value)
                        if getattr(report, "obj_value", None) is not None
                        else float(final_schedule.makespan)
                    ),
                    surrogate_bound=(
                        float(report.obj_bound)
                        if getattr(report, "obj_bound", None) is not None
                        else None
                    ),
                )

            sorted_snapshots = sorted(
                snapshots,
                key=lambda snapshot: (
                    sanitize_optional_float(snapshot.get("objective_ub")) or math.inf,
                    int(snapshot.get("snapshot_index", 0)),
                ),
            )
            for snapshot in sorted_snapshots:
                snapshot_schedule = snapshot.get("schedule")
                if not isinstance(snapshot_schedule, HybridFlowshopLiteSchedule):
                    continue
                snapshot_index = int(snapshot.get("snapshot_index", 0))
                snapshot_source = f"{source}_snap{snapshot_index:03d}"
                add_tau_schedule_candidate(
                    tau_candidates,
                    seen_signatures,
                    source=snapshot_source,
                    schedule=snapshot_schedule,
                    surrogate_obj=sanitize_optional_float(
                        snapshot.get("objective_ub")
                    ),
                    surrogate_bound=sanitize_optional_float(
                        snapshot.get("objective_lb")
                    ),
                )

        def run_surrogate_pw_cp_chain(
            *,
            tau: int,
            source_label: str,
            reference_schedule: HybridFlowshopLiteSchedule,
            scaled_instance: HybridFlowshopParameters,
            tau_candidates: list[_TauScheduleCandidate],
            seen_signatures: set[tuple[tuple[Any, ...], ...]],
        ) -> tuple[HybridFlowshopLiteSchedule | None, str | None]:
            if not surrogate_pw_cp_enabled:
                return None, None
            if (
                surrogate_pw_cp_tau_set is not None
                and int(tau) not in surrogate_pw_cp_tau_set
            ):
                return None, None

            if surrogate_pw_cp_batch_size is not None:
                resolved_batch_size = max(1, int(surrogate_pw_cp_batch_size))
            else:
                assert surrogate_pw_cp_batch_size_ratio is not None
                resolved_batch_size = max(
                    1,
                    int(
                        round(
                            float(scaled_instance.job_count)
                            * float(surrogate_pw_cp_batch_size_ratio)
                        )
                    ),
                )

            pw_constructor = PwCpConstructor(self)
            stage_2_batch_list = pw_constructor.build_stage_2_batch_list(
                reference_schedule,
                batch_size=resolved_batch_size,
            )
            actual_batch_count = pw_constructor.validate_and_get_batch_count(
                stage_2_batch_list
            )
            if surrogate_pw_cp_unfixed_batch_count_min > actual_batch_count:
                logging.info(
                    "[Tau coarsened CP] Skipping surrogate PW-CP tau=%s "
                    "source=%s because min unfixed count %d exceeds available "
                    "batch count %d (batch_size=%d).",
                    tau,
                    source_label,
                    surrogate_pw_cp_unfixed_batch_count_min,
                    actual_batch_count,
                    resolved_batch_size,
                )
                return None, None

            max_unfixed_batch_count = min(
                int(surrogate_pw_cp_unfixed_batch_count_max),
                actual_batch_count,
            )
            if max_unfixed_batch_count < surrogate_pw_cp_unfixed_batch_count_max:
                logging.info(
                    "[Tau coarsened CP] Clamping surrogate PW-CP max unfixed "
                    "count from %d to %d for tau=%s source=%s.",
                    surrogate_pw_cp_unfixed_batch_count_max,
                    max_unfixed_batch_count,
                    tau,
                    source_label,
                )

            if surrogate_pw_cp_lr_profile_fixed_batch_count > 0:
                left_profile_fixed_batch_count = int(
                    surrogate_pw_cp_lr_profile_fixed_batch_count
                )
                right_profile_fixed_batch_count = int(
                    surrogate_pw_cp_lr_profile_fixed_batch_count
                )
            else:
                left_profile_fixed_batch_count = int(
                    surrogate_pw_cp_left_profile_fixed_batch_count
                )
                right_profile_fixed_batch_count = int(
                    surrogate_pw_cp_right_profile_fixed_batch_count
                )

            current_ref = reference_schedule
            best_pw_schedule: HybridFlowshopLiteSchedule | None = None
            best_pw_source_label: str | None = None
            for unfixed_batch_count in range(
                int(surrogate_pw_cp_unfixed_batch_count_min),
                max_unfixed_batch_count + 1,
            ):
                before_obj = int(current_ref.makespan)
                logging.info(
                    "[Tau coarsened CP] Running surrogate PW-CP tau=%s "
                    "source=%s batch_size=%d unfixed_count=%d/%d "
                    "before_obj=%d.",
                    tau,
                    source_label,
                    resolved_batch_size,
                    unfixed_batch_count,
                    max_unfixed_batch_count,
                    before_obj,
                )
                try:
                    with self._temporary_instance_context(scaled_instance):
                        pw_result = PwCpConstructor(self).run(
                            current_ref,
                            scaled_instance,
                            scaled_instance.stage_2_job_2_p_map,
                            batch_size=resolved_batch_size,
                            step_size=int(surrogate_pw_cp_step_size),
                            unfixed_batch_count=int(unfixed_batch_count),
                            left_profile_fixed_batch_count=left_profile_fixed_batch_count,
                            right_profile_fixed_batch_count=right_profile_fixed_batch_count,
                            enable_promotion_profile_fixed=bool(
                                surrogate_pw_cp_enable_promotion_profile_fixed
                            ),
                            profile_fix_by_machine=bool(
                                surrogate_pw_cp_profile_fix_by_machine
                            ),
                            machine_precedence_stride=int(
                                surrogate_pw_cp_machine_precedence_stride
                            ),
                            stage_precedence_min_processing_time_diff=surrogate_pw_cp_stage_precedence_min_processing_time_diff,
                            stage_precedence_min_processing_time_diff_ratio=surrogate_pw_cp_stage_precedence_min_processing_time_diff_ratio,
                            non_time_fixed_op_time_limit_multiplier=surrogate_pw_cp_non_time_fixed_op_time_limit_multiplier,
                            max_time_per_batch=surrogate_pw_cp_max_time_per_batch,
                            solver_thread_cnt=solver_thread_cnt,
                            use_lns_only=bool(surrogate_pw_cp_use_lns_only),
                            tighten_ranges=bool(surrogate_pw_cp_tighten_ranges),
                            error_if_infeasible=False,
                        )
                except Exception:
                    logging.exception(
                        "[Tau coarsened CP] Surrogate PW-CP failed tau=%s "
                        "source=%s unfixed_count=%d.",
                        tau,
                        source_label,
                        unfixed_batch_count,
                    )
                    continue

                pw_schedule = pw_result.schedule
                current_ref = pw_schedule
                pw_source_label = f"{source_label}_pw{int(unfixed_batch_count)}"
                add_tau_schedule_candidate(
                    tau_candidates,
                    seen_signatures,
                    source=pw_source_label,
                    schedule=pw_schedule,
                    surrogate_obj=float(pw_schedule.makespan),
                    surrogate_bound=None,
                )
                logging.info(
                    "[Tau coarsened CP] Surrogate PW-CP tau=%s source=%s "
                    "unfixed_count=%d makespan=%s improvement=%s.",
                    tau,
                    source_label,
                    unfixed_batch_count,
                    pw_schedule.makespan,
                    before_obj - int(pw_schedule.makespan),
                )
                if (
                    best_pw_schedule is None
                    or pw_schedule.makespan < best_pw_schedule.makespan
                ):
                    best_pw_schedule = pw_schedule
                    best_pw_source_label = pw_source_label

                if (
                    surrogate_pw_cp_stop_on_no_improvement
                    and int(pw_schedule.makespan) >= before_obj
                ):
                    logging.info(
                        "[Tau coarsened CP] Stopping surrogate PW-CP chain "
                        "for tau=%s source=%s after non-improving unfixed_count=%d.",
                        tau,
                        source_label,
                        unfixed_batch_count,
                    )
                    break

            return best_pw_schedule, best_pw_source_label

        for tau in normalized_tau_values:
            scaled_instance = self._make_tau_coarsened_instance(tau)
            tau_schedule_candidates: list[_TauScheduleCandidate] = []
            seen_tau_signatures: set[tuple[tuple[Any, ...], ...]] = set()
            surrogate_dispatch_schedule: HybridFlowshopLiteSchedule | None = None
            if surrogate_dispatch_before_cp or include_surrogate_dispatch_candidate:
                surrogate_dispatch_entries = self._get_tau_surrogate_dispatch_schedules(
                    scaled_instance,
                    cap_portions=surrogate_dispatch_cap_portions,
                    method_list=surrogate_dispatch_method_list,
                    include_machine_then_job_variants=surrogate_dispatch_include_machine_then_job_variants,
                    candidate_top_k=surrogate_dispatch_candidate_top_k,
                )
                if surrogate_dispatch_entries:
                    surrogate_dispatch_schedule = surrogate_dispatch_entries[0][1]
                if include_surrogate_dispatch_candidate:
                    for rank, (
                        dispatch_label,
                        dispatch_schedule,
                    ) in enumerate(surrogate_dispatch_entries, start=1):
                        source_label = (
                            "dispatch"
                            if rank == 1
                            else f"dispatch_top{rank}_{dispatch_label}"
                        )
                        add_tau_schedule_candidate(
                            tau_schedule_candidates,
                            seen_tau_signatures,
                            source=source_label,
                            schedule=dispatch_schedule,
                            surrogate_obj=float(dispatch_schedule.makespan),
                            surrogate_bound=None,
                        )
                    logging.info(
                        "[Tau coarsened CP] tau=%s retained %s dispatch "
                        "candidates for original restore.",
                        tau,
                        len(tau_schedule_candidates),
                    )

            cp_reference_schedule = surrogate_dispatch_schedule
            cp_source_label = (
                "dispatch_hint_cp"
                if surrogate_dispatch_schedule is not None
                and surrogate_dispatch_before_cp
                else "cp"
            )
            should_run_pre_neh_dispatch_cp = (
                surrogate_neh_enabled
                and bool(surrogate_neh_also_solve_dispatch_cp)
                and surrogate_neh_position_norm == "before_cp"
                and surrogate_dispatch_schedule is not None
                and surrogate_dispatch_before_cp
                and (
                    surrogate_neh_tau_set is None
                    or int(tau) in surrogate_neh_tau_set
                )
                and "dispatch" in surrogate_neh_source_set
            )
            if should_run_pre_neh_dispatch_cp:
                dispatch_cp_time = self.get_remaining_time_limit(
                    raw_surrogate_time
                )
                if dispatch_cp_time is None or dispatch_cp_time <= 0:
                    logging.warning(
                        "[Tau coarsened CP] No remaining time for plain "
                        "dispatch surrogate CP tau=%s.",
                        tau,
                    )
                else:
                    logging.info(
                        "[Tau coarsened CP] Solving plain dispatch surrogate "
                        "tau=%s with %.3f sec before NEH-CP hint.",
                        tau,
                        dispatch_cp_time,
                    )
                    dispatch_cp_report, dispatch_cp_schedule, dispatch_cp_snapshots = (
                        self._solve_local_base_cp_candidate(
                            instance=scaled_instance,
                            computational_time=dispatch_cp_time,
                            solver_thread_cnt=solver_thread_cnt,
                            use_lns_only=surrogate_use_lns_only,
                            cp_model_probing_level=cp_model_probing_level,
                            cp_sat_params=cp_sat_params,
                            make_semi_active=make_semi_active,
                            reference_schedule=surrogate_dispatch_schedule,
                            add_reference_precedence=False,
                            stage_2_job_2_p_dict=scaled_instance.stage_2_job_2_p_map,
                            snapshot_solution_limit=surrogate_cp_snapshot_solution_limit,
                        )
                    )
                    if dispatch_cp_schedule is None:
                        logging.warning(
                            "[Tau coarsened CP] Plain dispatch surrogate CP "
                            "tau=%s infeasible/no solution (status=%s).",
                            tau,
                            dispatch_cp_report.status,
                        )
                    else:
                        add_cp_solution_candidates(
                            tau_schedule_candidates,
                            seen_tau_signatures,
                            source="dispatch_hint_cp",
                            report=dispatch_cp_report,
                            final_schedule=dispatch_cp_schedule,
                            snapshots=dispatch_cp_snapshots,
                        )
            if (
                surrogate_neh_enabled
                and surrogate_neh_position_norm == "before_cp"
                and surrogate_dispatch_schedule is not None
                and surrogate_dispatch_before_cp
                and (
                    surrogate_neh_tau_set is None
                    or int(tau) in surrogate_neh_tau_set
                )
                and "dispatch" in surrogate_neh_source_set
            ):
                neh_time_per_add = (
                    None
                    if surrogate_neh_cp_tl_nc_multiplier is None
                    else float(surrogate_neh_cp_tl_nc_multiplier)
                    * float(scaled_instance.job_count)
                    * float(scaled_instance.stage_count)
                )
                best_neh_schedule: HybridFlowshopLiteSchedule | None = None
                best_neh_source_label: str | None = None
                current_neh_ref = surrogate_dispatch_schedule
                chain_label_parts: list[str] = ["dispatch"]
                for neh_batch_size in surrogate_neh_batch_sizes:
                    logging.info(
                        "[Tau coarsened CP] Running pre-CP surrogate NEH tau=%s "
                        "source=dispatch added_batch_size=%s time_per_add=%s.",
                        tau,
                        neh_batch_size,
                        neh_time_per_add,
                    )
                    try:
                        with self._temporary_instance_context(scaled_instance):
                            neh_result = NehCpConstructor(self).run(
                                ref_schedule=(
                                    current_neh_ref
                                    if bool(surrogate_neh_sequential)
                                    else surrogate_dispatch_schedule
                                ),
                                instance=scaled_instance,
                                job_2_stage_2_p_dict=scaled_instance.job_2_stage_2_p_map,
                                stage_2_job_2_p_dict=scaled_instance.stage_2_job_2_p_map,
                                added_batch_size=int(neh_batch_size),
                                max_time_per_add=neh_time_per_add,
                                profile_fix_by_machine=False,
                                minimize_sum_ci_lex=bool(
                                    surrogate_neh_minimize_sum_ci_lex
                                ),
                                cp_tl_nc_multiplier_2nd_obj=surrogate_neh_cp_tl_nc_multiplier_2nd_obj,
                                make_semi_active_every_cp=make_semi_active,
                                solver_thread_cnt=solver_thread_cnt,
                                use_lns_only=bool(surrogate_neh_use_lns_only),
                                error_if_infeasible=False,
                                stop_before_final_reserve=False,
                                skip_if_estimated_neh_exceeds_remaining=False,
                                log_cp_subproblem_bounds=False,
                                log_cp_subproblem_progress=False,
                            )
                    except Exception:
                        logging.exception(
                            "[Tau coarsened CP] Pre-CP surrogate NEH failed "
                            "tau=%s batch_size=%s.",
                            tau,
                            neh_batch_size,
                        )
                        continue
                    if neh_result.schedule is None:
                        continue
                    if bool(surrogate_neh_sequential):
                        chain_label_parts.append(f"neh_b{int(neh_batch_size)}")
                        neh_source_label = "_".join(chain_label_parts)
                        current_neh_ref = neh_result.schedule
                    else:
                        neh_source_label = f"dispatch_neh_b{int(neh_batch_size)}"
                    add_tau_schedule_candidate(
                        tau_schedule_candidates,
                        seen_tau_signatures,
                        source=neh_source_label,
                        schedule=neh_result.schedule,
                        surrogate_obj=float(neh_result.schedule.makespan),
                        surrogate_bound=None,
                    )
                    if (
                        best_neh_schedule is None
                        or neh_result.schedule.makespan
                        < best_neh_schedule.makespan
                    ):
                        best_neh_schedule = neh_result.schedule
                        best_neh_source_label = neh_source_label
                    logging.info(
                        "[Tau coarsened CP] Pre-CP surrogate NEH tau=%s "
                        "batch_size=%s makespan=%s.",
                        tau,
                        neh_batch_size,
                        neh_result.schedule.makespan,
                    )
                if best_neh_schedule is not None and best_neh_source_label:
                    cp_reference_schedule = best_neh_schedule
                    cp_source_label = f"{best_neh_source_label}_hint_cp"
                    logging.info(
                        "[Tau coarsened CP] Selected pre-CP surrogate NEH "
                        "tau=%s source=%s makespan=%s for CP hint.",
                        tau,
                        best_neh_source_label,
                        best_neh_schedule.makespan,
                    )
                    best_pw_schedule, best_pw_source_label = run_surrogate_pw_cp_chain(
                        tau=tau,
                        source_label=best_neh_source_label,
                        reference_schedule=best_neh_schedule,
                        scaled_instance=scaled_instance,
                        tau_candidates=tau_schedule_candidates,
                        seen_signatures=seen_tau_signatures,
                    )
                    if best_pw_schedule is not None and best_pw_source_label:
                        cp_reference_schedule = best_pw_schedule
                        cp_source_label = f"{best_pw_source_label}_hint_cp"
                        logging.info(
                            "[Tau coarsened CP] Selected surrogate PW-CP "
                            "tau=%s source=%s makespan=%s for CP hint.",
                            tau,
                            best_pw_source_label,
                            best_pw_schedule.makespan,
                        )

            surrogate_time = self.get_remaining_time_limit(raw_surrogate_time)
            if surrogate_time is None or surrogate_time <= 0:
                logging.warning(
                    "[Tau coarsened CP] No remaining time for surrogate tau=%s.",
                    tau,
                )
                break

            logging.info(
                "[Tau coarsened CP] Solving surrogate tau=%s with %.3f sec "
                "(dispatch_hint=%s).",
                tau,
                surrogate_time,
                surrogate_dispatch_schedule is not None
                and surrogate_dispatch_before_cp,
            )
            surrogate_report, tau_schedule, surrogate_snapshots = (
                self._solve_local_base_cp_candidate(
                    instance=scaled_instance,
                    computational_time=surrogate_time,
                    solver_thread_cnt=solver_thread_cnt,
                    use_lns_only=surrogate_use_lns_only,
                    cp_model_probing_level=cp_model_probing_level,
                    cp_sat_params=cp_sat_params,
                    make_semi_active=make_semi_active,
                    reference_schedule=(
                        cp_reference_schedule
                        if surrogate_dispatch_before_cp
                        else None
                    ),
                    add_reference_precedence=False,
                    stage_2_job_2_p_dict=scaled_instance.stage_2_job_2_p_map,
                    snapshot_solution_limit=surrogate_cp_snapshot_solution_limit,
                )
            )
            if tau_schedule is None:
                logging.warning(
                    "[Tau coarsened CP] Surrogate tau=%s infeasible/no solution "
                    "(status=%s).",
                    tau,
                    surrogate_report.status,
                )
            else:
                add_cp_solution_candidates(
                    tau_schedule_candidates,
                    seen_tau_signatures,
                    source=cp_source_label,
                    report=surrogate_report,
                    final_schedule=tau_schedule,
                    snapshots=surrogate_snapshots,
                )

            if (
                surrogate_neh_enabled
                and surrogate_neh_position_norm == "after_cp"
                and tau_schedule_candidates
                and (
                    surrogate_neh_tau_set is None
                    or int(tau) in surrogate_neh_tau_set
                )
            ):
                for tau_candidate in list(tau_schedule_candidates):
                    if tau_candidate.source not in surrogate_neh_source_set:
                        continue
                    neh_time_per_add = (
                        None
                        if surrogate_neh_cp_tl_nc_multiplier is None
                        else float(surrogate_neh_cp_tl_nc_multiplier)
                        * float(scaled_instance.job_count)
                        * float(scaled_instance.stage_count)
                    )
                    logging.info(
                        "[Tau coarsened CP] Running surrogate NEH tau=%s "
                        "source=%s added_batch_size=%s time_per_add=%s.",
                        tau,
                        tau_candidate.source,
                        surrogate_neh_added_batch_size,
                        neh_time_per_add,
                    )
                    try:
                        with self._temporary_instance_context(scaled_instance):
                            neh_result = NehCpConstructor(self).run(
                                ref_schedule=tau_candidate.schedule,
                                instance=scaled_instance,
                                job_2_stage_2_p_dict=scaled_instance.job_2_stage_2_p_map,
                                stage_2_job_2_p_dict=scaled_instance.stage_2_job_2_p_map,
                                added_batch_size=int(
                                    surrogate_neh_added_batch_size
                                ),
                                max_time_per_add=neh_time_per_add,
                                profile_fix_by_machine=False,
                                minimize_sum_ci_lex=bool(
                                    surrogate_neh_minimize_sum_ci_lex
                                ),
                                cp_tl_nc_multiplier_2nd_obj=surrogate_neh_cp_tl_nc_multiplier_2nd_obj,
                                make_semi_active_every_cp=make_semi_active,
                                solver_thread_cnt=solver_thread_cnt,
                                use_lns_only=bool(surrogate_neh_use_lns_only),
                                error_if_infeasible=False,
                                stop_before_final_reserve=False,
                                skip_if_estimated_neh_exceeds_remaining=False,
                                log_cp_subproblem_bounds=False,
                                log_cp_subproblem_progress=False,
                            )
                    except Exception:
                        logging.exception(
                            "[Tau coarsened CP] Surrogate NEH failed tau=%s "
                            "source=%s.",
                            tau,
                            tau_candidate.source,
                        )
                        continue
                    if neh_result.schedule is None:
                        continue
                    add_tau_schedule_candidate(
                        tau_schedule_candidates,
                        seen_tau_signatures,
                        source=f"{tau_candidate.source}_neh",
                        schedule=neh_result.schedule,
                        surrogate_obj=float(neh_result.schedule.makespan),
                        surrogate_bound=None,
                    )
                    logging.info(
                        "[Tau coarsened CP] Surrogate NEH tau=%s source=%s "
                        "makespan=%s.",
                        tau,
                        tau_candidate.source,
                        neh_result.schedule.makespan,
                    )

            if not tau_schedule_candidates:
                continue

            logging.info(
                "[Tau coarsened CP] tau=%s restoring %s unique surrogate "
                "schedule candidates.",
                tau,
                len(tau_schedule_candidates),
            )
            for tau_candidate in tau_schedule_candidates:
                surrogate_obj = (
                    tau_candidate.surrogate_obj
                    if tau_candidate.surrogate_obj is not None
                    else float(tau_candidate.schedule.makespan)
                )
                surrogate_bound = tau_candidate.surrogate_bound
                expanded_surrogate_ub = float(tau) * float(
                    tau_candidate.schedule.makespan
                )
                logging.info(
                    "[Tau coarsened CP] tau=%s source=%s surrogate obj=%s "
                    "bound=%s expanded_ub=%s.",
                    tau,
                    tau_candidate.source,
                    surrogate_obj,
                    surrogate_bound,
                    expanded_surrogate_ub,
                )

                for restore_mode in restore_modes:
                    restored_schedule = self._restore_original_schedule_from_tau_schedule(
                        tau_candidate.schedule,
                        restore_mode=restore_mode,
                        make_semi_active=make_semi_active,
                    )
                    maybe_record_candidate(
                        tau=tau,
                        surrogate_source=tau_candidate.source,
                        mode=restore_mode,
                        phase="restored",
                        schedule=restored_schedule,
                        surrogate_obj=surrogate_obj,
                        surrogate_bound=surrogate_bound,
                        expanded_surrogate_ub=expanded_surrogate_ub,
                        elapsed_sec=sub_timer.elapsed_sec,
                    )

                    if raw_polish_time is None and polish_computational_time is None:
                        continue

                    for polish_profile_mode in polish_profile_modes:
                        polish_time = self.get_remaining_time_limit(raw_polish_time)
                        if polish_time is None or polish_time <= 0:
                            logging.warning(
                                "[Tau coarsened CP] No remaining time for polish tau=%s "
                                "mode=%s profile_mode=%s.",
                                tau,
                                restore_mode,
                                polish_profile_mode,
                            )
                            continue

                        restore_mode_norm = restore_mode.replace("-", "_").lower()
                        polish_mode_norm = polish_profile_mode.replace(
                            "-", "_"
                        ).lower()
                        if polish_mode_norm == "restore":
                            add_precedence = True
                            profile_fix_by_machine = restore_mode_norm in {
                                "machine",
                                "machine_sequence",
                                "machine_profile",
                            }
                        elif polish_mode_norm in {"none", "hint_only", "free"}:
                            add_precedence = False
                            profile_fix_by_machine = False
                        elif polish_mode_norm in {
                            "machine",
                            "machine_sequence",
                            "machine_profile",
                        }:
                            add_precedence = True
                            profile_fix_by_machine = True
                        elif polish_mode_norm in {
                            "stage",
                            "stage_sequence",
                            "stage_profile",
                        }:
                            add_precedence = True
                            profile_fix_by_machine = False
                        else:
                            raise ValueError(
                                "polish_profile_modes entries must be one of "
                                "restore, none, machine_sequence, or stage_sequence"
                            )

                        logging.info(
                            "[Tau coarsened CP] Polishing tau=%s mode=%s "
                            "profile_mode=%s obj=%s with %.3f sec.",
                            tau,
                            restore_mode,
                            polish_profile_mode,
                            restored_schedule.makespan,
                            polish_time,
                        )
                        polish_report, polished_schedule, _polish_snapshots = (
                            self._solve_local_base_cp_candidate(
                                instance=self.instance,
                                computational_time=polish_time,
                                solver_thread_cnt=solver_thread_cnt,
                                use_lns_only=polish_use_lns_only,
                                cp_model_probing_level=cp_model_probing_level,
                                cp_sat_params=cp_sat_params,
                                make_semi_active=make_semi_active,
                                reference_schedule=restored_schedule,
                                add_reference_precedence=add_precedence,
                                reference_profile_fix_by_machine=profile_fix_by_machine,
                                stage_2_job_2_p_dict=self.stage_2_job_2_p_dict,
                            )
                        )
                        if polished_schedule is None:
                            logging.warning(
                                "[Tau coarsened CP] Polish failed tau=%s mode=%s "
                                "profile_mode=%s status=%s.",
                                tau,
                                restore_mode,
                                polish_profile_mode,
                                polish_report.status,
                            )
                            continue
                        maybe_record_candidate(
                            tau=tau,
                            surrogate_source=tau_candidate.source,
                            mode=restore_mode,
                            phase=f"polished_{polish_mode_norm}",
                            schedule=polished_schedule,
                            surrogate_obj=surrogate_obj,
                            surrogate_bound=surrogate_bound,
                            expanded_surrogate_ub=expanded_surrogate_ub,
                            elapsed_sec=sub_timer.elapsed_sec,
                        )

        if save_candidate_artifacts and candidate_rows:
            artifact_path = self.get_file_path_for_subroutine(
                "_tau_coarsened_candidates.csv"
            )
            with artifact_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "tau",
                        "surrogate_source",
                        "restore_mode",
                        "phase",
                        "obj",
                        "surrogate_obj",
                        "surrogate_bound",
                        "expanded_surrogate_ub",
                        "elapsed_sec",
                    ],
                )
                writer.writeheader()
                writer.writerows(candidate_rows)

        if best_schedule is None or best_obj is None:
            msg = "[Tau coarsened CP] No feasible tau-coarsened candidate found."
            if error_if_infeasible:
                raise RuntimeError(msg)
            logging.warning(msg)
            return

        if error_if_infeasible:
            validate_schedule(best_schedule, self.stage_2_job_2_p_dict)

        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=self.solution_manager.get_incumbent() is None,
            subroutine_name="initialize_by_tau_coarsened_cp",
            progress_obj_value_records=progress_records
            or [(sub_timer.elapsed_sec, best_obj)],
        )
        was_updated = self.solution_manager.register(report, best_schedule)
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, best_obj, is_maximize=False)
        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
        )
        logging.info(
            "[Tau coarsened CP] Selected %s with makespan=%s updated_incumbent=%s.",
            best_label,
            best_obj,
            was_updated,
        )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _fix_operations_profile(
        self,
        schedule_profile_fixed_only: HybridFlowshopLiteSchedule,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
    ) -> None:
        BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
            self.cp_model,
            self.params,
            self.vars,
            schedule_profile_fixed_only,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
        )

    def _fix_operation_set_profile(
        self,
        schedule: HybridFlowshopLiteSchedule,
        operation_set: set[OperationType],
        *,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
    ) -> None:
        if not operation_set:
            return
        profile_schedule = schedule.deepcopy()
        removable_ops = set(profile_schedule.get_jik_2_start_time_map()) - operation_set
        profile_schedule.remove_operations(removable_ops)
        self._fix_operations_profile(
            profile_schedule,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
        )

    def _fix_operations_profile_except_selected(
        self,
        rescheduled_ops: set[tuple[str, str, str]],
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        fix_start_times: bool = False,
    ) -> None:
        """
        Helper to deep-copy incumbent solution and remove operations to be rescheduled,
        and add the CP model constraints that enforce precedences/machine assignments for
        the profile-fixed operations.

        Args:
            rescheduled_ops (set[tuple[str, str, str]]): set of (job, stage, machine) tuples
                that are not part of the block (i.e., they will be re-optimized).
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                adjacency; otherwise apply stage-level time-based selection.
                Defaults to False.

        Raises:
            ValueError: If the incumbent solution is not a valid HybridFlowshopLiteSchedule instance.
            TypeError: If CP model does not support precedences/machine assignment enforcement constraints.
        """
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )
        out_of_block_ops_sch = incumbent_solution.deepcopy()
        out_of_block_ops_sch.remove_operations(rescheduled_ops)
        self._fix_operations_profile(
            out_of_block_ops_sch,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )
        if fix_start_times:
            BaseModelBuilder.add_start_time_freezed_operation_constraints(
                self.cp_model,
                self.vars,
                out_of_block_ops_sch.get_jik_2_start_time_map(),
            )

    @staticmethod
    def _resolve_stage_count_from_ratio(
        *,
        explicit_count: int | None,
        ratio: float | None,
        total_count: int,
        min_count: int,
        max_count: int | None,
    ) -> int:
        if explicit_count is not None:
            resolved = int(explicit_count)
        else:
            resolved_ratio = 0.0 if ratio is None else float(ratio)
            if resolved_ratio < 0:
                raise ValueError("stage count ratio must be non-negative.")
            resolved = math.ceil(total_count * resolved_ratio)
        resolved = max(int(min_count), resolved)
        if max_count is not None:
            resolved = min(int(max_count), resolved)
        return max(0, min(total_count, resolved))

    def head_tail_free_middle_precedence_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        head_stage_count: int | None = None,
        head_stage_ratio: float | None = 0.20,
        min_head_stage_count: int = 1,
        max_head_stage_count: int | None = None,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = 0.20,
        min_tail_stage_count: int = 1,
        max_tail_stage_count: int | None = None,
        middle_stage_count: int | None = None,
        middle_stage_ratio: float | None = None,
        min_middle_stage_count: int = 1,
        max_middle_stage_count: int | None = None,
        middle_profile_fix_by_machine: bool = True,
        machine_precedence_stride: int = 1,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Free head/tail stage profiles while keeping middle-stage order only.

        No start times are frozen. The base model still keeps job precedence and
        stage cumulative capacity. This operator only adds incumbent-derived
        within-stage precedence constraints for the middle stages.
        """
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_head_tail_free_middle_precedence_operator(
                head_stage_count=head_stage_count,
                head_stage_ratio=head_stage_ratio,
                min_head_stage_count=min_head_stage_count,
                max_head_stage_count=max_head_stage_count,
                tail_stage_count=tail_stage_count,
                tail_stage_ratio=tail_stage_ratio,
                min_tail_stage_count=min_tail_stage_count,
                max_tail_stage_count=max_tail_stage_count,
                middle_stage_count=middle_stage_count,
                middle_stage_ratio=middle_stage_ratio,
                min_middle_stage_count=min_middle_stage_count,
                max_middle_stage_count=max_middle_stage_count,
                middle_profile_fix_by_machine=middle_profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_head_tail_free_middle_precedence_operator(
        self,
        *,
        head_stage_count: int | None = None,
        head_stage_ratio: float | None = 0.20,
        min_head_stage_count: int = 1,
        max_head_stage_count: int | None = None,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = 0.20,
        min_tail_stage_count: int = 1,
        max_tail_stage_count: int | None = None,
        middle_stage_count: int | None = None,
        middle_stage_ratio: float | None = None,
        min_middle_stage_count: int = 1,
        max_middle_stage_count: int | None = None,
        middle_profile_fix_by_machine: bool = True,
        machine_precedence_stride: int = 1,
    ) -> dict[str, object]:
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )

        stage_order = [str(stage_id) for stage_id in self.instance.stage_id_list]
        stage_count = len(stage_order)
        if stage_count <= 0:
            raise ValueError("Cannot run head/tail-free LNS without stages.")

        if middle_stage_count is not None or middle_stage_ratio is not None:
            middle_count = self._resolve_stage_count_from_ratio(
                explicit_count=middle_stage_count,
                ratio=middle_stage_ratio,
                total_count=stage_count,
                min_count=min_middle_stage_count,
                max_count=max_middle_stage_count,
            )
            free_count = max(0, stage_count - middle_count)
            head_count = free_count // 2
            tail_count = free_count - head_count
        else:
            head_count = self._resolve_stage_count_from_ratio(
                explicit_count=head_stage_count,
                ratio=head_stage_ratio,
                total_count=stage_count,
                min_count=min_head_stage_count,
                max_count=max_head_stage_count,
            )
            tail_count = self._resolve_stage_count_from_ratio(
                explicit_count=tail_stage_count,
                ratio=tail_stage_ratio,
                total_count=stage_count,
                min_count=min_tail_stage_count,
                max_count=max_tail_stage_count,
            )
            if head_count + tail_count > stage_count:
                overflow = head_count + tail_count - stage_count
                tail_reduce = min(tail_count, overflow)
                tail_count -= tail_reduce
                overflow -= tail_reduce
                if overflow > 0:
                    head_count = max(0, head_count - overflow)

        head_stage_ids = stage_order[:head_count]
        tail_stage_ids = stage_order[stage_count - tail_count :] if tail_count else []
        free_stage_set = set(head_stage_ids) | set(tail_stage_ids)
        middle_stage_ids = [
            stage_id for stage_id in stage_order if stage_id not in free_stage_set
        ]

        middle_profile_schedule = incumbent_solution.deepcopy()
        middle_stage_set = set(middle_stage_ids)
        outside_middle_ops = {
            op
            for op in middle_profile_schedule.get_jik_2_start_time_map()
            if str(op[1]) not in middle_stage_set
        }
        middle_profile_schedule.remove_operations(outside_middle_ops)
        if middle_stage_ids:
            self._fix_operations_profile(
                middle_profile_schedule,
                profile_fix_by_machine=middle_profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            )

        middle_op_count = len(middle_profile_schedule.get_jik_2_start_time_map())
        logging.info(
            "Head/tail-free middle-precedence LNS: head=%s tail=%s middle=%s "
            "middle_ops=%d middle_profile_fix_by_machine=%s stride=%d.",
            head_stage_ids,
            tail_stage_ids,
            middle_stage_ids,
            middle_op_count,
            middle_profile_fix_by_machine,
            machine_precedence_stride,
        )
        return {
            "head_stages": head_stage_ids,
            "tail_stages": tail_stage_ids,
            "middle_stages": middle_stage_ids,
            "middle_op_count": middle_op_count,
            "middle_profile_fix_by_machine": middle_profile_fix_by_machine,
        }

    # Subroutine: Operation-block neighbor search (Block operator in 2025 EJOR paper)

    def operation_block_ns(
        self,
        rho: float,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        seed_op_from_critical_block: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_operation_block_operator(
                rho,
                seed_op_from_critical_block=seed_op_from_critical_block,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_operation_block_operator(
        self,
        rho: float,
        seed_op_from_critical_block: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> None:
        """Apply the operation-block operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            seed_op_from_critical_block (bool, optional): If True, the seed operation is
                chosen from critical blocks of the incumbent solution. Defaults to False.
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                adjacency; otherwise apply stage-level time-based selection.
                Defaults to False.
            machine_precedence_stride (int, optional): The stride for selecting machine precedences.
                Defaults to 1.

        Raises:
            ValueError: If rho is not strictly positive.
            ValueError: If no incumbent solution is available.
            ValueError: If the incumbent solution is not a valid HybridFlowshopLiteSchedule instance.
            ValueError: If no start or end times are available.
        """
        if rho <= 0:
            raise ValueError(f"Invalid value for rho {rho}; it must be positive.")
        _rho = min(rho, 1)
        logging.info(f"Applying ops block operator with rho={_rho}")
        if not self.solution_manager.has_incumbent():
            raise ValueError("No incumbent solution available for ops block operator.")
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError("Incumbent solution is not a HybridFlowshopLiteSchedule.")
        start_time_map = incumbent_solution.get_jik_2_start_time_map()
        end_time_map = incumbent_solution.get_jik_2_end_time_map()

        if not start_time_map or not end_time_map:
            raise ValueError("No solution available for ops block operator.")

        all_ops = list(start_time_map.keys())  # TODO: 순서 유지되는지 확인
        total_ops = len(all_ops)
        num_to_select = max(1, int(rho * total_ops))

        # Choose an operation
        if seed_op_from_critical_block:
            # Find critical blocks in the incumbent solution
            incumbent_solution.make_semi_active(self.stage_2_job_2_p_dict)
            critical_blocks = incumbent_solution.find_critical_blocks(
                self.stage_2_job_2_p_dict, include_singletons=True
            )
            if not critical_blocks:
                # If no critical blocks, fall back to random selection
                seed_op = random.choice(all_ops)
            else:
                # Select a random operation from a random critical block
                selected_block = random.choice(critical_blocks)
                seed_op = random.choice(selected_block)
        else:
            seed_op = random.choice(all_ops)
        selected_ops = set([seed_op])
        queue = [seed_op]

        # Expand to overlapping operations
        while queue and len(selected_ops) < num_to_select:
            current_op = queue.pop(0)
            cs, ce = start_time_map[current_op], end_time_map[current_op]
            for op in all_ops:
                if op in selected_ops:
                    continue
                os, oe = start_time_map[op], end_time_map[op]
                if self.closed_intervals_overlap(cs, ce, os, oe):
                    selected_ops.add(op)
                    queue.append(op)
                if len(selected_ops) >= num_to_select:
                    break

        logging.info(
            f"Ops block operator selected {len(selected_ops)} overlapping ops"
            f" (target={num_to_select}; total={total_ops})"
        )

        # Fix out-of-block operations' profile
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )

    @staticmethod
    def closed_intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
        """Check if two closed time intervals overlap."""
        return not (e1 <= s2 or e2 <= s1)

    @staticmethod
    def open_intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
        """Check if two open time intervals overlap."""
        return not (e1 < s2 or e2 < s1)

    def _resolve_time_window_size(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        window_size: int | None,
        window_size_ratio: float | None,
    ) -> int:
        horizon = max(1, int(schedule.makespan))
        if window_size is not None:
            if window_size <= 0:
                raise ValueError("window_size must be positive.")
            if window_size_ratio is not None:
                logging.info(
                    "Ignoring window_size_ratio=%s because explicit window_size=%s was provided.",
                    window_size_ratio,
                    window_size,
                )
            return min(int(window_size), horizon)
        if window_size_ratio is not None:
            if window_size_ratio <= 0:
                raise ValueError("window_size_ratio must be positive.")
            return min(max(1, int(math.ceil(horizon * window_size_ratio))), horizon)
        return min(max(1, int(math.ceil(horizon * 0.10))), horizon)

    def _resolve_time_window_step_size(
        self,
        *,
        horizon: int,
        window_size: int,
        step_size: int | None,
        step_size_ratio: float | None,
    ) -> int:
        if step_size is not None:
            if step_size <= 0:
                raise ValueError("step_size must be positive.")
            if step_size_ratio is not None:
                logging.info(
                    "Ignoring step_size_ratio=%s because explicit step_size=%s was provided.",
                    step_size_ratio,
                    step_size,
                )
            return int(step_size)
        if step_size_ratio is not None:
            if step_size_ratio <= 0:
                raise ValueError("step_size_ratio must be positive.")
            return max(1, int(math.ceil(max(1, horizon) * step_size_ratio)))
        return max(1, window_size // 2)

    @staticmethod
    def _subsample_evenly(
        items: Sequence[tuple[int, int]],
        max_count: int | None,
    ) -> list[tuple[int, int]]:
        item_list = list(items)
        if max_count is None or max_count >= len(item_list):
            return item_list
        if max_count <= 0:
            raise ValueError("max_window_count must be positive when provided.")
        if max_count == 1:
            return [item_list[len(item_list) // 2]]
        last_idx = len(item_list) - 1
        selected_indices = {
            int(round(idx * last_idx / (max_count - 1))) for idx in range(max_count)
        }
        return [item_list[idx] for idx in sorted(selected_indices)]

    def _build_time_window_sweep_windows(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        window_size: int | None,
        window_size_ratio: float | None,
        step_size: int | None,
        step_size_ratio: float | None,
        max_window_count: int | None,
    ) -> list[tuple[int, int]]:
        horizon = max(1, int(schedule.makespan))
        resolved_window_size = self._resolve_time_window_size(
            schedule,
            window_size=window_size,
            window_size_ratio=window_size_ratio,
        )
        if resolved_window_size >= horizon:
            return [(0, horizon)]

        resolved_step_size = self._resolve_time_window_step_size(
            horizon=horizon,
            window_size=resolved_window_size,
            step_size=step_size,
            step_size_ratio=step_size_ratio,
        )
        last_start = horizon - resolved_window_size
        starts = list(range(0, last_start + 1, resolved_step_size))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        windows = [
            (start, min(horizon, start + resolved_window_size)) for start in starts
        ]
        return self._subsample_evenly(windows, max_window_count)

    def _resolve_time_window_stage_shift(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        stage_shift_per_stage: int | None,
        stage_shift_ratio_per_stage: float | None,
    ) -> int:
        if stage_shift_per_stage is not None:
            if stage_shift_ratio_per_stage is not None:
                logging.info(
                    "Ignoring stage_shift_ratio_per_stage=%s because explicit "
                    "stage_shift_per_stage=%s was provided.",
                    stage_shift_ratio_per_stage,
                    stage_shift_per_stage,
                )
            return int(stage_shift_per_stage)
        if stage_shift_ratio_per_stage is None:
            return 0
        horizon = max(1, int(schedule.makespan))
        return int(round(horizon * float(stage_shift_ratio_per_stage)))

    def _resolve_time_window_padding(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        padding: int | None,
        padding_ratio: float | None,
    ) -> int:
        if padding is not None:
            if padding < 0:
                raise ValueError("selection_window_padding must be non-negative.")
            if padding_ratio is not None:
                logging.info(
                    "Ignoring selection_window_padding_ratio=%s because explicit "
                    "selection_window_padding=%s was provided.",
                    padding_ratio,
                    padding,
                )
            return int(padding)
        if padding_ratio is None:
            return 0
        if padding_ratio < 0:
            raise ValueError("selection_window_padding_ratio must be non-negative.")
        horizon = max(1, int(schedule.makespan))
        return int(math.ceil(horizon * float(padding_ratio)))

    def _resolve_time_window_start_tolerance(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        start_time_tolerance: int | None,
        start_time_tolerance_ratio: float | None,
        label: str,
    ) -> int | None:
        if start_time_tolerance is not None:
            if start_time_tolerance < 0:
                raise ValueError(f"{label} must be non-negative.")
            if start_time_tolerance_ratio is not None:
                logging.info(
                    "Ignoring %s_ratio=%s because explicit %s=%s was provided.",
                    label,
                    start_time_tolerance_ratio,
                    label,
                    start_time_tolerance,
                )
            return int(start_time_tolerance)
        if start_time_tolerance_ratio is None:
            return None
        if start_time_tolerance_ratio < 0:
            raise ValueError(f"{label}_ratio must be non-negative.")
        horizon = max(1, int(schedule.makespan))
        return int(math.ceil(horizon * float(start_time_tolerance_ratio)))

    def _build_slanted_time_window_sweep_windows(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        window_size: int | None,
        window_size_ratio: float | None,
        step_size: int | None,
        step_size_ratio: float | None,
        max_window_count: int | None,
        stage_shift_per_stage: int,
    ) -> list[tuple[int, int]]:
        horizon = max(1, int(schedule.makespan))
        resolved_window_size = self._resolve_time_window_size(
            schedule,
            window_size=window_size,
            window_size_ratio=window_size_ratio,
        )
        resolved_step_size = self._resolve_time_window_step_size(
            horizon=horizon,
            window_size=resolved_window_size,
            step_size=step_size,
            step_size_ratio=step_size_ratio,
        )
        stage_count = len(getattr(schedule, "stages", [])) or int(
            getattr(self.instance, "stage_count", 1)
        )
        stage_offsets = [
            stage_idx * int(stage_shift_per_stage) for stage_idx in range(stage_count)
        ]
        min_base_start = -max(stage_offsets)
        max_base_time = horizon - min(stage_offsets)
        last_start = max_base_time - resolved_window_size
        if last_start <= min_base_start:
            return [(min_base_start, min_base_start + resolved_window_size)]

        starts = list(range(min_base_start, last_start + 1, resolved_step_size))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        windows = [(start, start + resolved_window_size) for start in starts]
        return self._subsample_evenly(windows, max_window_count)

    @staticmethod
    def _get_schedule_stage_index_map(
        schedule: HybridFlowshopLiteSchedule,
    ) -> dict[StageIdType, int]:
        return {stage_id: idx for idx, stage_id in enumerate(schedule.stages)}

    def _add_start_time_range_constraints_for_ops(
        self,
        schedule: HybridFlowshopLiteSchedule,
        ops: set[OperationType],
        *,
        tolerance: int,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        start_time_map = schedule.get_jik_2_start_time_map()
        horizon = max(1, int(math.ceil(self.get_horizon())))
        constrained_count = 0
        for op in ops:
            if op not in start_time_map:
                continue
            j, i, _ = op
            incumbent_start = int(start_time_map[op])
            lb = max(0, incumbent_start - tolerance)
            ub = min(horizon, incumbent_start + tolerance)
            self.cp_model.add(self.vars.op_start[j, i] >= lb)
            self.cp_model.add(self.vars.op_start[j, i] <= ub)
            constrained_count += 1
        logging.info(
            "Time-window %s start ranges constrained for %d ops with tolerance=%d.",
            label,
            constrained_count,
            tolerance,
        )

    def _fix_time_window_profile_except_selected(
        self,
        schedule: HybridFlowshopLiteSchedule,
        selected_ops: set[OperationType],
        *,
        profile_fix_by_machine: bool,
        machine_precedence_stride: int,
        fix_outside_start_times: bool,
        selected_start_time_tolerance: int | None,
        outside_start_time_tolerance: int | None,
    ) -> None:
        outside_start_ranges_enabled = outside_start_time_tolerance is not None
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            fix_start_times=fix_outside_start_times
            and not outside_start_ranges_enabled,
        )
        if selected_start_time_tolerance is not None:
            self._add_start_time_range_constraints_for_ops(
                schedule,
                selected_ops,
                tolerance=selected_start_time_tolerance,
                label="selected",
            )
        if outside_start_time_tolerance is not None:
            outside_ops = set(schedule.get_jik_2_start_time_map()) - selected_ops
            self._add_start_time_range_constraints_for_ops(
                schedule,
                outside_ops,
                tolerance=outside_start_time_tolerance,
                label="outside",
            )

    def _select_ops_overlapping_time_window(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        window_start: int,
        window_end: int,
        overlap_mode: str = "intersect",
    ) -> set[OperationType]:
        if window_end <= window_start:
            raise ValueError("window_end must be greater than window_start.")
        if overlap_mode not in {"intersect", "start", "contained"}:
            raise ValueError(
                "overlap_mode must be one of {'intersect', 'start', 'contained'}."
            )

        selected_ops: set[OperationType] = set()
        for op, start_time in schedule.get_jik_2_start_time_map().items():
            end_time = schedule.get_jik_2_end_time_map()[op]
            if overlap_mode == "intersect":
                selected = not (end_time <= window_start or window_end <= start_time)
            elif overlap_mode == "start":
                selected = window_start <= start_time < window_end
            else:
                selected = window_start <= start_time and end_time <= window_end
            if selected:
                selected_ops.add(op)
        return selected_ops

    def _resolve_time_ring_shoulder_size(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        core_window_size: int,
        shoulder_size: int | None,
        shoulder_size_ratio: float | None,
    ) -> int:
        horizon = max(1, int(schedule.makespan))
        if shoulder_size is not None:
            if shoulder_size < 0:
                raise ValueError("shoulder_size must be non-negative.")
            if shoulder_size_ratio is not None:
                logging.info(
                    "Ignoring shoulder_size_ratio=%s because explicit "
                    "shoulder_size=%s was provided.",
                    shoulder_size_ratio,
                    shoulder_size,
                )
            return int(shoulder_size)
        if shoulder_size_ratio is not None:
            if shoulder_size_ratio < 0:
                raise ValueError("shoulder_size_ratio must be non-negative.")
            return int(math.ceil(horizon * float(shoulder_size_ratio)))
        return max(1, int(math.ceil(core_window_size / 2)))

    def _partition_ops_by_time_ring(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        core_start: int,
        core_end: int,
        shoulder_size: int,
        overlap_mode: str,
    ) -> tuple[set[OperationType], set[OperationType], set[OperationType]]:
        horizon = max(1, int(schedule.makespan))
        outer_start = max(0, int(core_start) - int(shoulder_size))
        outer_end = min(horizon, int(core_end) + int(shoulder_size))
        core_ops = self._select_ops_overlapping_time_window(
            schedule,
            window_start=core_start,
            window_end=core_end,
            overlap_mode=overlap_mode,
        )
        outer_ops = self._select_ops_overlapping_time_window(
            schedule,
            window_start=outer_start,
            window_end=outer_end,
            overlap_mode=overlap_mode,
        )
        shoulder_ops = outer_ops - core_ops
        outside_ops = set(schedule.get_jik_2_start_time_map()) - outer_ops
        return core_ops, shoulder_ops, outside_ops

    def apply_time_ring_precedence_operator(
        self,
        core_start: int,
        core_end: int,
        *,
        shoulder_size: int,
        overlap_mode: str = "intersect",
        outside_machine_precedence_stride: int = 1,
        shoulder_machine_precedence_stride: int = 1,
        shoulder_stage_precedence_min_processing_time_diff: int | None = None,
        shoulder_stage_precedence_min_processing_time_diff_ratio: float | None = None,
    ) -> dict[str, int]:
        """Apply a time-ring precedence relaxation around one core window.

        Operations touching the core window receive no incumbent-derived
        within-stage precedence arcs. Operations in the shoulder band keep only
        stage-level precedence. Operations outside the outer band keep only
        incumbent machine-neighbor precedence.
        """
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )
        if core_end <= core_start:
            raise ValueError("core_end must be greater than core_start.")
        if shoulder_size < 0:
            raise ValueError("shoulder_size must be non-negative.")

        core_ops, shoulder_ops, outside_ops = self._partition_ops_by_time_ring(
            incumbent_solution,
            core_start=core_start,
            core_end=core_end,
            shoulder_size=shoulder_size,
            overlap_mode=overlap_mode,
        )
        if not core_ops:
            raise ValueError(
                f"No operations overlap core time-ring window [{core_start}, {core_end})."
            )

        self._fix_operation_set_profile(
            incumbent_solution,
            outside_ops,
            profile_fix_by_machine=True,
            machine_precedence_stride=outside_machine_precedence_stride,
        )
        self._fix_operation_set_profile(
            incumbent_solution,
            shoulder_ops,
            profile_fix_by_machine=False,
            machine_precedence_stride=shoulder_machine_precedence_stride,
            stage_precedence_min_processing_time_diff=shoulder_stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=shoulder_stage_precedence_min_processing_time_diff_ratio,
        )

        outer_start = max(0, int(core_start) - int(shoulder_size))
        outer_end = min(
            int(incumbent_solution.makespan), int(core_end) + int(shoulder_size)
        )
        logging.info(
            "Time-ring precedence operator: core=[%d,%d) shoulder=%d outer=[%d,%d) "
            "core_ops=%d shoulder_ops=%d outside_ops=%d.",
            core_start,
            core_end,
            shoulder_size,
            outer_start,
            outer_end,
            len(core_ops),
            len(shoulder_ops),
            len(outside_ops),
        )
        return {
            "core_ops": len(core_ops),
            "shoulder_ops": len(shoulder_ops),
            "outside_ops": len(outside_ops),
            "outer_start": outer_start,
            "outer_end": outer_end,
        }

    def _select_ops_overlapping_slanted_time_window(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        window_start: int,
        window_end: int,
        stage_shift_per_stage: int,
        overlap_mode: str = "intersect",
    ) -> set[OperationType]:
        if window_end <= window_start:
            raise ValueError("window_end must be greater than window_start.")
        if overlap_mode not in {"intersect", "start", "contained"}:
            raise ValueError(
                "overlap_mode must be one of {'intersect', 'start', 'contained'}."
            )

        stage_index_map = self._get_schedule_stage_index_map(schedule)
        selected_ops: set[OperationType] = set()
        start_time_map = schedule.get_jik_2_start_time_map()
        end_time_map = schedule.get_jik_2_end_time_map()
        for op, start_time in start_time_map.items():
            stage_id = op[1]
            stage_idx = stage_index_map[stage_id]
            stage_shift = stage_idx * int(stage_shift_per_stage)
            shifted_window_start = window_start + stage_shift
            shifted_window_end = window_end + stage_shift
            end_time = end_time_map[op]
            if overlap_mode == "intersect":
                selected = not (
                    end_time <= shifted_window_start or shifted_window_end <= start_time
                )
            elif overlap_mode == "start":
                selected = shifted_window_start <= start_time < shifted_window_end
            else:
                selected = (
                    shifted_window_start <= start_time
                    and end_time <= shifted_window_end
                )
            if selected:
                selected_ops.add(op)
        return selected_ops

    def apply_time_window_operation_operator(
        self,
        window_start: int,
        window_end: int,
        *,
        overlap_mode: str = "intersect",
        stage_shift_per_stage: int = 0,
        selection_window_padding: int | None = None,
        selection_window_padding_ratio: float | None = None,
        profile_fix_by_machine: bool = True,
        machine_precedence_stride: int = 1,
        fix_outside_start_times: bool = True,
        selected_start_time_tolerance: int | None = None,
        selected_start_time_tolerance_ratio: float | None = None,
        outside_start_time_tolerance: int | None = None,
        outside_start_time_tolerance_ratio: float | None = None,
    ) -> set[OperationType]:
        """Free every operation overlapping one rectangular or slanted time window."""
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )
        resolved_padding = self._resolve_time_window_padding(
            incumbent_solution,
            padding=selection_window_padding,
            padding_ratio=selection_window_padding_ratio,
        )
        selection_window_start = window_start - resolved_padding
        selection_window_end = window_end + resolved_padding
        resolved_selected_tolerance = self._resolve_time_window_start_tolerance(
            incumbent_solution,
            start_time_tolerance=selected_start_time_tolerance,
            start_time_tolerance_ratio=selected_start_time_tolerance_ratio,
            label="selected_start_time_tolerance",
        )
        resolved_outside_tolerance = self._resolve_time_window_start_tolerance(
            incumbent_solution,
            start_time_tolerance=outside_start_time_tolerance,
            start_time_tolerance_ratio=outside_start_time_tolerance_ratio,
            label="outside_start_time_tolerance",
        )
        if stage_shift_per_stage:
            selected_ops = self._select_ops_overlapping_slanted_time_window(
                incumbent_solution,
                window_start=selection_window_start,
                window_end=selection_window_end,
                stage_shift_per_stage=stage_shift_per_stage,
                overlap_mode=overlap_mode,
            )
        else:
            selected_ops = self._select_ops_overlapping_time_window(
                incumbent_solution,
                window_start=selection_window_start,
                window_end=selection_window_end,
                overlap_mode=overlap_mode,
            )
        if not selected_ops:
            raise ValueError(
                f"No operations overlap time window [{window_start}, {window_end})."
            )
        logging.info(
            "Time-window operator freeing %d ops in [%d, %d) "
            "(selection=[%d, %d), padding=%d) with overlap_mode=%s "
            "stage_shift_per_stage=%d.",
            len(selected_ops),
            window_start,
            window_end,
            selection_window_start,
            selection_window_end,
            resolved_padding,
            overlap_mode,
            stage_shift_per_stage,
        )
        self._fix_time_window_profile_except_selected(
            incumbent_solution,
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            fix_outside_start_times=fix_outside_start_times,
            selected_start_time_tolerance=resolved_selected_tolerance,
            outside_start_time_tolerance=resolved_outside_tolerance,
        )
        return selected_ops

    def time_window_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        window_start: int | None = None,
        window_end: int | None = None,
        window_start_ratio: float | None = None,
        window_end_ratio: float | None = None,
        window_size: int | None = None,
        window_size_ratio: float | None = None,
        overlap_mode: str = "intersect",
        stage_shift_per_stage: int | None = None,
        stage_shift_ratio_per_stage: float | None = None,
        selection_window_padding: int | None = None,
        selection_window_padding_ratio: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = True,
        machine_precedence_stride: int = 1,
        fix_outside_start_times: bool = True,
        selected_start_time_tolerance: int | None = None,
        selected_start_time_tolerance_ratio: float | None = None,
        outside_start_time_tolerance: int | None = None,
        outside_start_time_tolerance_ratio: float | None = None,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Re-optimize one explicit rectangular or slanted time window."""
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )
        horizon = max(1, int(incumbent_solution.makespan))
        if window_start is not None:
            if window_start_ratio is not None:
                logging.info(
                    "Ignoring window_start_ratio=%s because explicit "
                    "window_start=%s was provided.",
                    window_start_ratio,
                    window_start,
                )
            resolved_window_start = int(window_start)
        elif window_start_ratio is not None:
            if not 0 <= window_start_ratio <= 1:
                raise ValueError("window_start_ratio must be in [0, 1].")
            resolved_window_start = int(math.floor(horizon * window_start_ratio))
        else:
            resolved_window_start = 0

        if window_end is not None:
            if window_end_ratio is not None:
                logging.info(
                    "Ignoring window_end_ratio=%s because explicit window_end=%s "
                    "was provided.",
                    window_end_ratio,
                    window_end,
                )
            resolved_window_end = int(window_end)
        elif window_end_ratio is not None:
            if not 0 <= window_end_ratio <= 1:
                raise ValueError("window_end_ratio must be in [0, 1].")
            resolved_window_end = int(math.ceil(horizon * window_end_ratio))
        else:
            resolved_size = self._resolve_time_window_size(
                incumbent_solution,
                window_size=window_size,
                window_size_ratio=window_size_ratio,
            )
            resolved_window_end = resolved_window_start + resolved_size

        resolved_window_start = max(0, resolved_window_start)
        resolved_window_end = min(horizon, resolved_window_end)
        if resolved_window_end <= resolved_window_start:
            raise ValueError(
                "Resolved time window must have window_end > window_start."
            )

        resolved_stage_shift_per_stage = self._resolve_time_window_stage_shift(
            incumbent_solution,
            stage_shift_per_stage=stage_shift_per_stage,
            stage_shift_ratio_per_stage=stage_shift_ratio_per_stage,
        )
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        logging.info(
            "Single time-window NS starts for [%d, %d) (stage_shift_per_stage=%d).",
            resolved_window_start,
            resolved_window_end,
            resolved_stage_shift_per_stage,
        )

        context_name = f"single_window_{resolved_window_start}_{resolved_window_end}"
        with self.temporarily_extended_context(context_name):
            self._fix_profile_solve_reset(
                lambda: self.apply_time_window_operation_operator(
                    resolved_window_start,
                    resolved_window_end,
                    overlap_mode=overlap_mode,
                    stage_shift_per_stage=resolved_stage_shift_per_stage,
                    selection_window_padding=selection_window_padding,
                    selection_window_padding_ratio=selection_window_padding_ratio,
                    profile_fix_by_machine=profile_fix_by_machine,
                    machine_precedence_stride=machine_precedence_stride,
                    fix_outside_start_times=fix_outside_start_times,
                    selected_start_time_tolerance=selected_start_time_tolerance,
                    selected_start_time_tolerance_ratio=selected_start_time_tolerance_ratio,
                    outside_start_time_tolerance=outside_start_time_tolerance,
                    outside_start_time_tolerance_ratio=outside_start_time_tolerance_ratio,
                ),
                resolved_computational_time,
                solver_thread_cnt,
                no_improvement_timelimit=no_improvement_timelimit,
                swap_before_cp=swap_before_cp,
                make_semi_active_after_cp=make_semi_active_after_cp,
                use_lns_only=use_lns_only,
                obj_value_is_valid=True,
                obj_bound_is_valid=False,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )

    def time_window_sweep_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        window_size: int | None = None,
        window_size_ratio: float | None = 0.10,
        step_size: int | None = None,
        step_size_ratio: float | None = None,
        max_window_count: int | None = None,
        overlap_mode: str = "intersect",
        stage_shift_per_stage: int | None = None,
        stage_shift_ratio_per_stage: float | None = None,
        selection_window_padding: int | None = None,
        selection_window_padding_ratio: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = True,
        machine_precedence_stride: int = 1,
        fix_outside_start_times: bool = True,
        selected_start_time_tolerance: int | None = None,
        selected_start_time_tolerance_ratio: float | None = None,
        outside_start_time_tolerance: int | None = None,
        outside_start_time_tolerance_ratio: float | None = None,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Sweep time windows and re-optimize all operations touched by each window."""
        seed_incumbent = self.solution_manager.get_incumbent()
        if not isinstance(seed_incumbent, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )

        resolved_stage_shift_per_stage = self._resolve_time_window_stage_shift(
            seed_incumbent,
            stage_shift_per_stage=stage_shift_per_stage,
            stage_shift_ratio_per_stage=stage_shift_ratio_per_stage,
        )
        if resolved_stage_shift_per_stage:
            windows = self._build_slanted_time_window_sweep_windows(
                seed_incumbent,
                window_size=window_size,
                window_size_ratio=window_size_ratio,
                step_size=step_size,
                step_size_ratio=step_size_ratio,
                max_window_count=max_window_count,
                stage_shift_per_stage=resolved_stage_shift_per_stage,
            )
        else:
            windows = self._build_time_window_sweep_windows(
                seed_incumbent,
                window_size=window_size,
                window_size_ratio=window_size_ratio,
                step_size=step_size,
                step_size_ratio=step_size_ratio,
                max_window_count=max_window_count,
            )
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        resolved_padding = self._resolve_time_window_padding(
            seed_incumbent,
            padding=selection_window_padding,
            padding_ratio=selection_window_padding_ratio,
        )
        resolved_selected_tolerance = self._resolve_time_window_start_tolerance(
            seed_incumbent,
            start_time_tolerance=selected_start_time_tolerance,
            start_time_tolerance_ratio=selected_start_time_tolerance_ratio,
            label="selected_start_time_tolerance",
        )
        resolved_outside_tolerance = self._resolve_time_window_start_tolerance(
            seed_incumbent,
            start_time_tolerance=outside_start_time_tolerance,
            start_time_tolerance_ratio=outside_start_time_tolerance_ratio,
            label="outside_start_time_tolerance",
        )
        logging.info(
            "Time-window sweep starts with %d windows: %s "
            "(stage_shift_per_stage=%d, selection_padding=%d, "
            "selected_start_tolerance=%s, outside_start_tolerance=%s)",
            len(windows),
            windows,
            resolved_stage_shift_per_stage,
            resolved_padding,
            resolved_selected_tolerance,
            resolved_outside_tolerance,
        )

        for window_idx, (window_start, window_end) in enumerate(windows, start=1):
            incumbent_solution = self.solution_manager.get_incumbent()
            if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
                logging.warning(
                    "Stopping time-window sweep: no incumbent schedule is available."
                )
                break
            selection_window_start = window_start - resolved_padding
            selection_window_end = window_end + resolved_padding
            if resolved_stage_shift_per_stage:
                selected_ops = self._select_ops_overlapping_slanted_time_window(
                    incumbent_solution,
                    window_start=selection_window_start,
                    window_end=selection_window_end,
                    stage_shift_per_stage=resolved_stage_shift_per_stage,
                    overlap_mode=overlap_mode,
                )
            else:
                selected_ops = self._select_ops_overlapping_time_window(
                    incumbent_solution,
                    window_start=selection_window_start,
                    window_end=selection_window_end,
                    overlap_mode=overlap_mode,
                )
            if not selected_ops:
                logging.info(
                    "Skipping empty time-window %d/%d [%d, %d).",
                    window_idx,
                    len(windows),
                    window_start,
                    window_end,
                )
                continue

            context_name = f"window_{window_idx:03d}_{window_start}_{window_end}"
            with self.temporarily_extended_context(context_name):
                self._fix_profile_solve_reset(
                    lambda selected_ops=selected_ops, incumbent_solution=incumbent_solution: (
                        self._fix_time_window_profile_except_selected(
                            incumbent_solution,
                            selected_ops,
                            profile_fix_by_machine=profile_fix_by_machine,
                            machine_precedence_stride=machine_precedence_stride,
                            fix_outside_start_times=fix_outside_start_times,
                            selected_start_time_tolerance=resolved_selected_tolerance,
                            outside_start_time_tolerance=resolved_outside_tolerance,
                        )
                    ),
                    resolved_computational_time,
                    solver_thread_cnt,
                    no_improvement_timelimit=no_improvement_timelimit,
                    swap_before_cp=swap_before_cp,
                    make_semi_active_after_cp=make_semi_active_after_cp,
                    use_lns_only=use_lns_only,
                    obj_value_is_valid=True,
                    obj_bound_is_valid=False,
                    error_if_infeasible=error_if_infeasible,
                    draw_gantt=draw_gantt,
                )

    def time_ring_precedence_sweep_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        window_size: int | None = None,
        window_size_ratio: float | None = 0.20,
        shoulder_size: int | None = None,
        shoulder_size_ratio: float | None = None,
        step_size: int | None = None,
        step_size_ratio: float | None = None,
        max_window_count: int | None = None,
        overlap_mode: str = "intersect",
        outside_machine_precedence_stride: int = 1,
        shoulder_machine_precedence_stride: int = 1,
        shoulder_stage_precedence_min_processing_time_diff: int | None = None,
        shoulder_stage_precedence_min_processing_time_diff_ratio: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Sweep core time windows with ring-shaped precedence relaxation.

        For each core window, the current incumbent is partitioned into:
        core operations with no incumbent-derived profile arcs, shoulder
        operations with stage-level arcs, and outside operations with sparse
        machine-neighbor arcs.
        """
        seed_incumbent = self.solution_manager.get_incumbent()
        if not isinstance(seed_incumbent, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )
        windows = self._build_time_window_sweep_windows(
            seed_incumbent,
            window_size=window_size,
            window_size_ratio=window_size_ratio,
            step_size=step_size,
            step_size_ratio=step_size_ratio,
            max_window_count=max_window_count,
        )
        core_window_size = max(1, int(windows[0][1] - windows[0][0]))
        resolved_shoulder_size = self._resolve_time_ring_shoulder_size(
            seed_incumbent,
            core_window_size=core_window_size,
            shoulder_size=shoulder_size,
            shoulder_size_ratio=shoulder_size_ratio,
        )
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        logging.info(
            "Time-ring precedence sweep starts with %d windows: %s "
            "(shoulder_size=%d, overlap_mode=%s, outside_machine_stride=%d).",
            len(windows),
            windows,
            resolved_shoulder_size,
            overlap_mode,
            outside_machine_precedence_stride,
        )

        for window_idx, (window_start, window_end) in enumerate(windows, start=1):
            incumbent_solution = self.solution_manager.get_incumbent()
            if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
                logging.warning(
                    "Stopping time-ring precedence sweep: no incumbent schedule is available."
                )
                break

            core_ops = self._select_ops_overlapping_time_window(
                incumbent_solution,
                window_start=window_start,
                window_end=window_end,
                overlap_mode=overlap_mode,
            )
            if not core_ops:
                logging.info(
                    "Skipping empty time-ring window %d/%d [%d, %d).",
                    window_idx,
                    len(windows),
                    window_start,
                    window_end,
                )
                continue

            context_name = f"time_ring_{window_idx:03d}_{window_start}_{window_end}"
            with self.temporarily_extended_context(context_name):
                self._fix_profile_solve_reset(
                    lambda window_start=window_start, window_end=window_end: (
                        self.apply_time_ring_precedence_operator(
                            window_start,
                            window_end,
                            shoulder_size=resolved_shoulder_size,
                            overlap_mode=overlap_mode,
                            outside_machine_precedence_stride=outside_machine_precedence_stride,
                            shoulder_machine_precedence_stride=shoulder_machine_precedence_stride,
                            shoulder_stage_precedence_min_processing_time_diff=shoulder_stage_precedence_min_processing_time_diff,
                            shoulder_stage_precedence_min_processing_time_diff_ratio=shoulder_stage_precedence_min_processing_time_diff_ratio,
                        )
                    ),
                    resolved_computational_time,
                    solver_thread_cnt,
                    no_improvement_timelimit=no_improvement_timelimit,
                    swap_before_cp=swap_before_cp,
                    make_semi_active_after_cp=make_semi_active_after_cp,
                    use_lns_only=use_lns_only,
                    obj_value_is_valid=True,
                    obj_bound_is_valid=False,
                    error_if_infeasible=error_if_infeasible,
                    draw_gantt=draw_gantt,
                )

    def _resolve_stage_time_window_center_stage_ids(
        self,
        *,
        stage_center_policy: str,
        max_stage_centers: int | None,
        include_singleton_critical_blocks: bool,
    ) -> list[str]:
        stage_order = list(self.instance.stage_id_list)
        stage_order_set = set(stage_order)
        normalized_policy = stage_center_policy.lower().replace("-", "_").strip()

        if max_stage_centers is not None and max_stage_centers <= 0:
            raise ValueError("max_stage_centers must be positive when provided.")

        def add_unique(target: list[str], stage_id: StageIdType) -> None:
            stage_key = str(stage_id)
            if stage_key in stage_order_set and stage_key not in target:
                target.append(stage_key)

        centers: list[str] = []
        if normalized_policy in {"all", "all_stages"}:
            centers = list(stage_order)
        elif normalized_policy in {"random", "random_stage", "random_stages"}:
            centers = list(stage_order)
            random.shuffle(centers)
        else:
            if "critical" in normalized_policy:
                add_unique(
                    centers,
                    self._resolve_critical_stage_id_for_targeted_ns(
                        include_singleton_critical_blocks=include_singleton_critical_blocks
                    ),
                )
            if "retained" in normalized_policy or "bottleneck" in normalized_policy:
                add_unique(centers, self._resolve_bottleneck_stage_id_for_targeted_ns())
            if "last" in normalized_policy or "tail" in normalized_policy:
                add_unique(centers, stage_order[-1])

        if not centers:
            raise ValueError(
                "stage_center_policy must select at least one stage. "
                "Supported examples: critical, retained_bottleneck, "
                "critical_and_retained_bottleneck, random, all."
            )
        if max_stage_centers is not None:
            centers = centers[: int(max_stage_centers)]
        return centers

    def _build_stage_time_window_sweep_windows(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        window_size: int | None,
        window_size_ratio: float | None,
        step_size: int | None,
        step_size_ratio: float | None,
        max_window_count: int | None,
        tail_window_bias: bool,
        min_tail_start_ratio: float | None,
    ) -> list[tuple[int, int]]:
        windows = self._build_time_window_sweep_windows(
            schedule,
            window_size=window_size,
            window_size_ratio=window_size_ratio,
            step_size=step_size,
            step_size_ratio=step_size_ratio,
            max_window_count=None,
        )
        if tail_window_bias:
            ratio = (
                0.55 if min_tail_start_ratio is None else float(min_tail_start_ratio)
            )
            if ratio < 0 or ratio > 1:
                raise ValueError("min_tail_start_ratio must be in [0, 1].")
            horizon = max(1, int(schedule.makespan))
            tail_cutoff = int(math.floor(horizon * ratio))
            tail_windows = [
                (window_start, window_end)
                for window_start, window_end in windows
                if window_end >= tail_cutoff
            ]
            if tail_windows:
                windows = tail_windows
        return self._subsample_evenly(windows, max_window_count)

    def _build_stage_band_local_time_windows(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        selected_stage_ids: Sequence[StageIdType],
        window_size: int | None,
        window_size_ratio: float | None,
        step_size: int | None,
        step_size_ratio: float | None,
        max_window_count: int | None,
        tail_window_bias: bool,
        min_tail_start_ratio: float | None,
    ) -> list[tuple[int, int]]:
        selected_stage_set = {str(stage_id) for stage_id in selected_stage_ids}
        start_time_map = schedule.get_jik_2_start_time_map()
        end_time_map = schedule.get_jik_2_end_time_map()
        band_ops = [op for op in start_time_map if str(op[1]) in selected_stage_set]
        if not band_ops:
            return []

        band_start = min(int(start_time_map[op]) for op in band_ops)
        band_end = max(int(end_time_map[op]) for op in band_ops)
        band_span = max(1, band_end - band_start)
        if window_size is not None:
            if window_size <= 0:
                raise ValueError("window_size must be positive.")
            resolved_window_size = min(int(window_size), band_span)
        else:
            ratio = 0.10 if window_size_ratio is None else float(window_size_ratio)
            if ratio <= 0:
                raise ValueError("window_size_ratio must be positive.")
            resolved_window_size = min(
                max(1, int(math.ceil(band_span * ratio))),
                band_span,
            )

        if step_size is not None:
            if step_size <= 0:
                raise ValueError("step_size must be positive.")
            resolved_step_size = int(step_size)
        elif step_size_ratio is not None:
            if step_size_ratio <= 0:
                raise ValueError("step_size_ratio must be positive.")
            resolved_step_size = max(1, int(math.ceil(band_span * step_size_ratio)))
        else:
            resolved_step_size = max(1, resolved_window_size // 2)

        last_start = band_end - resolved_window_size
        starts = list(range(band_start, last_start + 1, resolved_step_size))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        windows = [
            (start, min(band_end, start + resolved_window_size)) for start in starts
        ]
        if tail_window_bias:
            ratio = (
                0.55 if min_tail_start_ratio is None else float(min_tail_start_ratio)
            )
            if ratio < 0 or ratio > 1:
                raise ValueError("min_tail_start_ratio must be in [0, 1].")
            tail_cutoff = band_start + int(math.floor(band_span * ratio))
            tail_windows = [
                (window_start, window_end)
                for window_start, window_end in windows
                if window_end >= tail_cutoff
            ]
            if tail_windows:
                windows = tail_windows
        return self._subsample_evenly(windows, max_window_count)

    def _select_ops_in_stage_time_window(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        selected_stage_ids: Sequence[StageIdType],
        window_start: int,
        window_end: int,
        overlap_mode: str,
    ) -> set[OperationType]:
        selected_stage_set = {str(stage_id) for stage_id in selected_stage_ids}
        window_ops = self._select_ops_overlapping_time_window(
            schedule,
            window_start=window_start,
            window_end=window_end,
            overlap_mode=overlap_mode,
        )
        return {op for op in window_ops if str(op[1]) in selected_stage_set}

    def stage_time_window_sweep_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        per_window_tl_nc_multiplier: float | None = None,
        stage_center_policy: str = "critical_and_retained_bottleneck",
        stage_radius: int = 1,
        max_stage_centers: int | None = 3,
        window_time_scope: str = "stage_band",
        window_size: int | None = None,
        window_size_ratio: float | None = 0.04,
        step_size: int | None = None,
        step_size_ratio: float | None = 0.02,
        max_window_count_per_stage_center: int | None = 6,
        overlap_mode: str = "intersect",
        tail_window_bias: bool = True,
        min_tail_start_ratio: float | None = 0.55,
        include_singleton_critical_blocks: bool = True,
        selection_window_padding: int | None = None,
        selection_window_padding_ratio: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = True,
        machine_precedence_stride: int = 1,
        fix_outside_start_times: bool = False,
        selected_start_time_tolerance: int | None = None,
        selected_start_time_tolerance_ratio: float | None = None,
        outside_start_time_tolerance: int | None = None,
        outside_start_time_tolerance_ratio: float | None = None,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """
        Sweep small stage-time neighborhoods in the style of PW-CP.

        A center stage is chosen from the current incumbent's critical stage and/or
        the last retained-CP bottleneck. For each center, this subroutine frees only
        operations inside ``stage_band(center, stage_radius)`` and a small time
        window, while profile-fixing the rest of the incumbent schedule.
        """
        seed_incumbent = self.solution_manager.get_incumbent()
        if not isinstance(seed_incumbent, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a valid HybridFlowshopLiteSchedule instance."
            )
        if per_window_tl_nc_multiplier is not None and tl_nc_multiplier is not None:
            raise ValueError(
                "Use only one of per_window_tl_nc_multiplier and tl_nc_multiplier."
            )

        effective_tl_nc_multiplier = (
            per_window_tl_nc_multiplier
            if per_window_tl_nc_multiplier is not None
            else tl_nc_multiplier
        )
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=effective_tl_nc_multiplier,
        )
        resolved_padding = self._resolve_time_window_padding(
            seed_incumbent,
            padding=selection_window_padding,
            padding_ratio=selection_window_padding_ratio,
        )
        resolved_selected_tolerance = self._resolve_time_window_start_tolerance(
            seed_incumbent,
            start_time_tolerance=selected_start_time_tolerance,
            start_time_tolerance_ratio=selected_start_time_tolerance_ratio,
            label="selected_start_time_tolerance",
        )
        resolved_outside_tolerance = self._resolve_time_window_start_tolerance(
            seed_incumbent,
            start_time_tolerance=outside_start_time_tolerance,
            start_time_tolerance_ratio=outside_start_time_tolerance_ratio,
            label="outside_start_time_tolerance",
        )
        center_stage_ids = self._resolve_stage_time_window_center_stage_ids(
            stage_center_policy=stage_center_policy,
            max_stage_centers=max_stage_centers,
            include_singleton_critical_blocks=include_singleton_critical_blocks,
        )
        normalized_window_time_scope = (
            window_time_scope.lower().replace("-", "_").strip()
        )
        if normalized_window_time_scope not in {"global", "stage_band"}:
            raise ValueError(
                "window_time_scope must be one of {'global', 'stage_band'}."
            )
        global_windows: list[tuple[int, int]] | None = None
        if normalized_window_time_scope == "global":
            global_windows = self._build_stage_time_window_sweep_windows(
                seed_incumbent,
                window_size=window_size,
                window_size_ratio=window_size_ratio,
                step_size=step_size,
                step_size_ratio=step_size_ratio,
                max_window_count=max_window_count_per_stage_center,
                tail_window_bias=tail_window_bias,
                min_tail_start_ratio=min_tail_start_ratio,
            )
        logging.info(
            "Stage-time-window sweep starts: centers=%s stage_radius=%d window_scope=%s windows=%s "
            "per_window_time=%s padding=%d selected_tol=%s outside_tol=%s.",
            center_stage_ids,
            stage_radius,
            normalized_window_time_scope,
            global_windows if global_windows is not None else "<per-stage-band>",
            resolved_computational_time,
            resolved_padding,
            resolved_selected_tolerance,
            resolved_outside_tolerance,
        )

        attempted_count = 0
        improved_count = 0
        skipped_empty_count = 0
        for center_idx, center_stage_id in enumerate(center_stage_ids, start=1):
            selected_stage_ids = self._resolve_stage_band_ids(
                center_stage_id,
                radius=stage_radius,
            )
            if normalized_window_time_scope == "stage_band":
                windows = self._build_stage_band_local_time_windows(
                    seed_incumbent,
                    selected_stage_ids=selected_stage_ids,
                    window_size=window_size,
                    window_size_ratio=window_size_ratio,
                    step_size=step_size,
                    step_size_ratio=step_size_ratio,
                    max_window_count=max_window_count_per_stage_center,
                    tail_window_bias=tail_window_bias,
                    min_tail_start_ratio=min_tail_start_ratio,
                )
                logging.info(
                    "Stage-time-window center=%s stages=%s local windows=%s.",
                    center_stage_id,
                    selected_stage_ids,
                    windows,
                )
            else:
                windows = global_windows or []
            for window_idx, (window_start, window_end) in enumerate(windows, start=1):
                incumbent_solution = self.solution_manager.get_incumbent()
                if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
                    logging.warning(
                        "Stopping stage-time-window sweep: no incumbent schedule is available."
                    )
                    break

                selection_window_start = window_start - resolved_padding
                selection_window_end = window_end + resolved_padding
                selected_ops = self._select_ops_in_stage_time_window(
                    incumbent_solution,
                    selected_stage_ids=selected_stage_ids,
                    window_start=selection_window_start,
                    window_end=selection_window_end,
                    overlap_mode=overlap_mode,
                )
                if not selected_ops:
                    skipped_empty_count += 1
                    logging.info(
                        "Skipping empty stage-time window center=%s stages=%s window=%d/%d [%d, %d).",
                        center_stage_id,
                        selected_stage_ids,
                        window_idx,
                        len(windows),
                        window_start,
                        window_end,
                    )
                    continue

                before_obj = sanitize_optional_float(
                    self.solution_manager.best_obj_value
                )
                attempted_count += 1
                context_name = (
                    f"stage_time_window_{center_idx:02d}_{center_stage_id}_"
                    f"{window_idx:03d}_{window_start}_{window_end}"
                )
                logging.info(
                    "Stage-time-window solve %d: center=%s stages=%s window=[%d, %d) "
                    "selection=[%d, %d) selected_ops=%d incumbent=%s.",
                    attempted_count,
                    center_stage_id,
                    selected_stage_ids,
                    window_start,
                    window_end,
                    selection_window_start,
                    selection_window_end,
                    len(selected_ops),
                    before_obj,
                )
                with self.temporarily_extended_context(context_name):
                    self._fix_profile_solve_reset(
                        lambda selected_ops=selected_ops, incumbent_solution=incumbent_solution: (
                            self._fix_time_window_profile_except_selected(
                                incumbent_solution,
                                selected_ops,
                                profile_fix_by_machine=profile_fix_by_machine,
                                machine_precedence_stride=machine_precedence_stride,
                                fix_outside_start_times=fix_outside_start_times,
                                selected_start_time_tolerance=resolved_selected_tolerance,
                                outside_start_time_tolerance=resolved_outside_tolerance,
                            )
                        ),
                        resolved_computational_time,
                        solver_thread_cnt,
                        no_improvement_timelimit=no_improvement_timelimit,
                        swap_before_cp=swap_before_cp,
                        make_semi_active_after_cp=make_semi_active_after_cp,
                        use_lns_only=use_lns_only,
                        obj_value_is_valid=True,
                        obj_bound_is_valid=False,
                        error_if_infeasible=error_if_infeasible,
                        draw_gantt=draw_gantt,
                    )
                after_obj = sanitize_optional_float(
                    self.solution_manager.best_obj_value
                )
                if (
                    before_obj is not None
                    and after_obj is not None
                    and after_obj < before_obj
                ):
                    improved_count += 1

        logging.info(
            "Stage-time-window sweep finished: attempted=%d improved=%d skipped_empty=%d.",
            attempted_count,
            improved_count,
            skipped_empty_count,
        )

    # Subroutine: Stage neighbor search

    def stage_block_ns(
        self,
        rho: float,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        seed_stage_from_non_singleton_cb: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_stage_operator(
                rho,
                seed_stage_from_non_singleton_cb=seed_stage_from_non_singleton_cb,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_stage_operator(
        self,
        rho: float,
        seed_stage_from_non_singleton_cb: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ):
        """
        Apply the "stage" LNS operator: free a consecutive subset of stages (i.e. allow
        operations on those stages to be rescheduled) and fix the profile of all other
        operations according to the current incumbent schedule.

        Args:
            rho (float): The proportion of stages to free (must be between 0 and 1).
            seed_stage_from_non_singleton_cb (bool, optional): Whether to seed the stage
                selection from non-singleton critical blocks. Defaults to False.
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                adjacency; otherwise apply stage-level time-based selection.
                Defaults to False.
            machine_precedence_stride (int, optional): The stride for selecting machine precedences.
                Defaults to 1.

        Raises:
            ValueError: If rho is not strictly positive.
            ValueError: If no incumbent solution is available.
            ValueError: If the incumbent solution is not a HybridFlowshopLiteSchedule.
        """
        if rho <= 0:
            raise ValueError(f"Invalid value for rho {rho}; it must be positive.")
        _rho = min(rho, 1)
        free_stage_cnt: int = math.ceil(self.instance.stage_count * _rho)
        logging.info(
            f"Applying stage operator with {free_stage_cnt} free stages (rho={rho})"
        )
        if not self.solution_manager.has_incumbent():
            raise ValueError("No incumbent solution available for stage operator.")
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError("Incumbent solution is not a HybridFlowshopLiteSchedule.")

        all_stage_list = self.instance.stage_id_list
        selected_stages: set[str]
        # Choose {free_stage_cnt} consecutive stages
        if free_stage_cnt > len(all_stage_list):
            selected_stages = set(all_stage_list)
        else:
            if seed_stage_from_non_singleton_cb:
                # Find critical blocks in the incumbent solution
                incumbent_solution.make_semi_active(self.stage_2_job_2_p_dict)
                critical_blocks = incumbent_solution.find_critical_blocks(
                    self.stage_2_job_2_p_dict, include_singletons=False
                )
                if not critical_blocks:
                    logging.debug(
                        "No non-singleton critical blocks found, using random consecutive selection"
                    )
                    start_idx = random.randint(0, len(all_stage_list) - free_stage_cnt)
                    selected_stages = set(
                        all_stage_list[start_idx : start_idx + free_stage_cnt]
                    )
                else:
                    selected_block = random.choice(critical_blocks)
                    # stage of the block (all ops in the block are on the same stage)
                    seed_stage = selected_block[0][1]
                    logging.info(
                        f"Seed stage selected from non-singleton critical block: {seed_stage}"
                    )
                    selected_stages = set()
                    # Expand from the seed stage to get consecutive stages until we have enough stages
                    left_idx = all_stage_list.index(seed_stage)
                    right_idx = left_idx
                    selected_stages.add(seed_stage)
                    while len(selected_stages) < free_stage_cnt:
                        expand_left = left_idx > 0
                        expand_right = right_idx < len(all_stage_list) - 1
                        if expand_left and (not expand_right or random.random() < 0.5):
                            left_idx -= 1
                            selected_stages.add(all_stage_list[left_idx])
                        elif expand_right:
                            right_idx += 1
                            selected_stages.add(all_stage_list[right_idx])
                        else:
                            break  # cannot expand further on either side
            else:
                start_idx = random.randint(0, len(all_stage_list) - free_stage_cnt)
                selected_stages = set(
                    all_stage_list[start_idx : start_idx + free_stage_cnt]
                )
        logging.info(f"Selected stages: {sorted(selected_stages)}")

        all_ops = list(incumbent_solution.get_jik_2_start_time_map().keys())
        selected_ops = set([ops for ops in all_ops if ops[1] in selected_stages])

        # Fix out-of-block operations' profile
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )

    def random_stage_band_stage_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        radius: int = 1,
        min_radius: int | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Free a random consecutive stage band and re-optimize it with CP."""

        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_random_stage_band_stage_operator(
                radius=radius,
                min_radius=min_radius,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_random_stage_band_stage_operator(
        self,
        radius: int = 1,
        *,
        min_radius: int | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> list[str]:
        if radius < 0:
            raise ValueError(f"radius must be non-negative. Received {radius}.")
        resolved_min_radius = radius if min_radius is None else int(min_radius)
        if resolved_min_radius < 0:
            raise ValueError(
                f"min_radius must be non-negative. Received {resolved_min_radius}."
            )
        if resolved_min_radius > radius:
            raise ValueError(
                f"min_radius must be <= radius. Received {resolved_min_radius} > {radius}."
            )

        stage_id_list = list(self.instance.stage_id_list)
        if not stage_id_list:
            raise ValueError("instance.stage_id_list cannot be empty.")
        chosen_radius = random.randint(resolved_min_radius, radius)
        center_stage_id = random.choice(stage_id_list)
        selected_stage_ids = self._resolve_stage_band_ids(
            center_stage_id,
            radius=chosen_radius,
        )
        logging.info(
            "Applying random stage-band operator: center_stage=%s radius=%d stages=%s",
            center_stage_id,
            chosen_radius,
            selected_stage_ids,
        )
        return self._apply_stage_selection_operator(
            selected_stage_ids,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )

    def critical_stage_band_stage_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        radius: int = 1,
        include_singleton_critical_blocks: bool = True,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Free the stage band centered on the incumbent's most critical stage."""

        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_critical_stage_band_stage_operator(
                radius=radius,
                include_singleton_critical_blocks=include_singleton_critical_blocks,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def _resolve_critical_stage_id_for_targeted_ns(
        self,
        *,
        include_singleton_critical_blocks: bool = True,
    ) -> str:
        if not self.solution_manager.has_incumbent():
            raise ValueError("No incumbent solution available for critical stage band.")
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError("Incumbent solution is not a HybridFlowshopLiteSchedule.")

        stage_order = list(self.instance.stage_id_list)
        stage_set = set(stage_order)
        incumbent_solution.make_semi_active(self.stage_2_job_2_p_dict)
        critical_blocks = incumbent_solution.find_critical_blocks(
            self.stage_2_job_2_p_dict,
            include_singletons=include_singleton_critical_blocks,
        )
        stage_scores: Counter[str] = Counter()
        for block in critical_blocks:
            block_weight = max(1, len(block))
            for op in block:
                if len(op) < 2:
                    continue
                stage_id = str(op[1])
                if stage_id in stage_set:
                    stage_scores[stage_id] += block_weight

        if not stage_scores:
            fallback_stage = self._resolve_bottleneck_stage_id_for_targeted_ns()
            logging.info(
                "No critical blocks found for critical stage-band operator; "
                "falling back to bottleneck_stage=%s",
                fallback_stage,
            )
            return fallback_stage

        best_stage = max(stage_order, key=lambda stage_id: stage_scores[str(stage_id)])
        logging.info(
            "Critical stage-band center selected: stage=%s score=%s scores=%s",
            best_stage,
            stage_scores[str(best_stage)],
            dict(stage_scores),
        )
        return str(best_stage)

    def apply_critical_stage_band_stage_operator(
        self,
        radius: int = 1,
        *,
        include_singleton_critical_blocks: bool = True,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> list[str]:
        center_stage_id = self._resolve_critical_stage_id_for_targeted_ns(
            include_singleton_critical_blocks=include_singleton_critical_blocks,
        )
        selected_stage_ids = self._resolve_stage_band_ids(
            center_stage_id,
            radius=radius,
        )
        logging.info(
            "Applying critical stage-band operator: center_stage=%s radius=%d stages=%s",
            center_stage_id,
            radius,
            selected_stage_ids,
        )
        return self._apply_stage_selection_operator(
            selected_stage_ids,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )

    def _resolve_rectangle_job_count(
        self,
        *,
        job_count: int | None,
        job_count_ratio: float | None,
        min_job_count: int,
        max_job_count: int | None,
    ) -> int:
        if job_count is not None:
            resolved = int(job_count)
        else:
            ratio = 0.12 if job_count_ratio is None else float(job_count_ratio)
            if ratio <= 0:
                raise ValueError("job_count_ratio must be positive.")
            resolved = math.ceil(float(self.instance.job_count) * ratio)
        resolved = max(int(min_job_count), resolved)
        if max_job_count is not None:
            resolved = min(int(max_job_count), resolved)
        return max(1, min(int(self.instance.job_count), resolved))

    def _get_incumbent_schedule_for_rectangle_lns(
        self,
        operator_name: str,
    ) -> HybridFlowshopLiteSchedule:
        if not self.solution_manager.has_incumbent():
            raise ValueError(f"No incumbent solution available for {operator_name}.")
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                f"Incumbent solution is not a HybridFlowshopLiteSchedule for {operator_name}."
            )
        return incumbent_solution

    @staticmethod
    def _build_job_stage_start_end_maps(
        schedule: HybridFlowshopLiteSchedule,
    ) -> tuple[
        dict[tuple[JobIdType, StageIdType], int],
        dict[tuple[JobIdType, StageIdType], int],
    ]:
        start_by_job_stage: dict[tuple[JobIdType, StageIdType], int] = {}
        end_by_job_stage: dict[tuple[JobIdType, StageIdType], int] = {}
        for (
            stage_id,
            _mc_id,
            start_time,
            end_time,
            job_id,
        ) in schedule._iter_operations():
            start_by_job_stage[(job_id, stage_id)] = int(start_time)
            end_by_job_stage[(job_id, stage_id)] = int(end_time)
        return start_by_job_stage, end_by_job_stage

    @staticmethod
    def _get_last_stage_completion_by_job(
        schedule: HybridFlowshopLiteSchedule,
    ) -> dict[JobIdType, int]:
        if not schedule.stages:
            return {}
        last_stage = schedule.stages[-1]
        return {
            job_id: int(end_time)
            for _mc_id, _start_time, end_time, job_id in schedule.iter_operations_on_stage(
                last_stage
            )
        }

    def _select_tail_critical_jobs_and_stages(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        job_count: int,
        tail_time_ratio: float,
        stage_radius: int,
        include_singleton_critical_blocks: bool,
    ) -> tuple[list[JobIdType], list[str]]:
        if tail_time_ratio <= 0 or tail_time_ratio > 1:
            raise ValueError("tail_time_ratio must satisfy 0 < tail_time_ratio <= 1.")

        schedule.make_semi_active(self.stage_2_job_2_p_dict)
        critical_blocks = schedule.find_critical_blocks(
            self.stage_2_job_2_p_dict,
            include_singletons=include_singleton_critical_blocks,
        )
        end_time_map = schedule.get_jik_2_end_time_map()
        tail_cutoff = int(max(0, math.floor(schedule.makespan * (1 - tail_time_ratio))))
        last_stage_completion = self._get_last_stage_completion_by_job(schedule)

        job_scores: Counter[JobIdType] = Counter()
        all_critical_job_scores: Counter[JobIdType] = Counter()
        stage_scores_for_tail: Counter[str] = Counter()
        stage_scores_all: Counter[str] = Counter()
        for block in critical_blocks:
            block_weight = max(1, len(block))
            for op in block:
                if op not in end_time_map:
                    continue
                job_id, stage_id, _mc_id = op
                all_critical_job_scores[job_id] += block_weight
                stage_scores_all[str(stage_id)] += block_weight
                if end_time_map[op] >= tail_cutoff:
                    job_scores[job_id] += block_weight
                    stage_scores_for_tail[str(stage_id)] += block_weight

        if not job_scores:
            for op, end_time in end_time_map.items():
                if end_time >= tail_cutoff:
                    job_scores[op[0]] += 1
                    stage_scores_for_tail[str(op[1])] += 1
        if not job_scores:
            job_scores.update(all_critical_job_scores)
            stage_scores_for_tail.update(stage_scores_all)
        if not job_scores:
            job_scores.update({job_id: 1 for job_id in schedule.jobs})

        selected_jobs = sorted(
            job_scores,
            key=lambda job_id: (
                -float(job_scores[job_id]),
                -float(last_stage_completion.get(job_id, 0)),
                str(job_id),
            ),
        )[: min(job_count, len(job_scores))]
        selected_job_set = set(selected_jobs)

        selected_stage_scores: Counter[str] = Counter()
        for block in critical_blocks:
            block_weight = max(1, len(block))
            for op in block:
                if op[0] in selected_job_set and op in end_time_map:
                    tail_multiplier = 2 if end_time_map[op] >= tail_cutoff else 1
                    selected_stage_scores[str(op[1])] += block_weight * tail_multiplier
        if not selected_stage_scores:
            for op, end_time in end_time_map.items():
                if op[0] in selected_job_set and end_time >= tail_cutoff:
                    selected_stage_scores[str(op[1])] += 1
        if not selected_stage_scores:
            selected_stage_scores.update(stage_scores_for_tail)
        if not selected_stage_scores:
            selected_stage_scores.update(stage_scores_all)

        stage_order = list(self.instance.stage_id_list)
        if selected_stage_scores:
            center_stage_id = max(
                stage_order,
                key=lambda stage_id: selected_stage_scores[str(stage_id)],
            )
        else:
            center_stage_id = self._resolve_bottleneck_stage_id_for_targeted_ns()
        selected_stage_ids = self._resolve_stage_band_ids(
            center_stage_id,
            radius=stage_radius,
        )

        logging.info(
            "Tail critical rectangle selected jobs=%s stages=%s center_stage=%s "
            "tail_cutoff=%s top_job_scores=%s top_stage_scores=%s",
            selected_jobs,
            selected_stage_ids,
            center_stage_id,
            tail_cutoff,
            self._format_top_score_items(job_scores),
            self._format_top_score_items(selected_stage_scores),
        )
        return selected_jobs, selected_stage_ids

    def critical_tail_rectangle_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        job_count: int | None = None,
        job_count_ratio: float | None = 0.12,
        min_job_count: int = 8,
        max_job_count: int | None = 32,
        tail_time_ratio: float = 0.25,
        stage_radius: int = 1,
        include_singleton_critical_blocks: bool = True,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        resolved_job_count = self._resolve_rectangle_job_count(
            job_count=job_count,
            job_count_ratio=job_count_ratio,
            min_job_count=min_job_count,
            max_job_count=max_job_count,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_critical_tail_rectangle_operator(
                job_count=resolved_job_count,
                tail_time_ratio=tail_time_ratio,
                stage_radius=stage_radius,
                include_singleton_critical_blocks=include_singleton_critical_blocks,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_critical_tail_rectangle_operator(
        self,
        *,
        job_count: int,
        tail_time_ratio: float = 0.25,
        stage_radius: int = 1,
        include_singleton_critical_blocks: bool = True,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> dict[str, object]:
        schedule = self._get_incumbent_schedule_for_rectangle_lns(
            "critical tail rectangle operator"
        )
        selected_jobs, selected_stages = self._select_tail_critical_jobs_and_stages(
            schedule,
            job_count=job_count,
            tail_time_ratio=tail_time_ratio,
            stage_radius=stage_radius,
            include_singleton_critical_blocks=include_singleton_critical_blocks,
        )
        all_ops = list(schedule.get_jik_2_start_time_map().keys())
        selected_job_set = set(selected_jobs)
        selected_stage_set = set(selected_stages)
        selected_ops = {
            op
            for op in all_ops
            if op[0] in selected_job_set and str(op[1]) in selected_stage_set
        }
        if not selected_ops:
            raise ValueError("Critical tail rectangle selected no operations.")
        logging.info(
            "Critical tail rectangle freeing %d jobs, %d stages, %d ops.",
            len(selected_job_set),
            len(selected_stage_set),
            len(selected_ops),
        )
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )
        return {
            "selected_jobs": selected_jobs,
            "selected_stages": selected_stages,
            "selected_op_count": len(selected_ops),
        }

    def _select_handoff_boundary_jobs_and_stages(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        job_count: int,
        stage_radius: int,
        tail_time_ratio: float,
        include_singleton_critical_blocks: bool,
    ) -> tuple[list[JobIdType], list[str], tuple[str, str]]:
        if tail_time_ratio <= 0 or tail_time_ratio > 1:
            raise ValueError("tail_time_ratio must satisfy 0 < tail_time_ratio <= 1.")

        schedule.make_semi_active(self.stage_2_job_2_p_dict)
        critical_blocks = schedule.find_critical_blocks(
            self.stage_2_job_2_p_dict,
            include_singletons=include_singleton_critical_blocks,
        )
        start_by_job_stage, end_by_job_stage = self._build_job_stage_start_end_maps(
            schedule
        )
        last_stage_completion = self._get_last_stage_completion_by_job(schedule)
        tail_cutoff = int(max(0, math.floor(schedule.makespan * (1 - tail_time_ratio))))

        critical_job_scores = self._get_critical_job_weights(critical_blocks)
        critical_stage_scores: Counter[str] = Counter(
            str(op[1]) for block in critical_blocks for op in block
        )
        tail_jobs = {
            job_id
            for job_id, end_time in last_stage_completion.items()
            if end_time >= tail_cutoff
        }

        stage_order = list(self.instance.stage_id_list)
        boundary_scores: dict[tuple[str, str], float] = {}
        boundary_job_waits: dict[tuple[str, str], dict[JobIdType, float]] = {}
        for idx in range(1, len(stage_order)):
            prev_stage = stage_order[idx - 1]
            curr_stage = stage_order[idx]
            boundary = (prev_stage, curr_stage)
            waits: dict[JobIdType, float] = {}
            score = float(
                critical_stage_scores[str(prev_stage)]
                + critical_stage_scores[str(curr_stage)]
            )
            for job_id in schedule.jobs:
                prev_end = end_by_job_stage.get((job_id, prev_stage))
                curr_start = start_by_job_stage.get((job_id, curr_stage))
                curr_end = end_by_job_stage.get((job_id, curr_stage))
                if prev_end is None or curr_start is None:
                    continue
                wait = max(0.0, float(curr_start - prev_end))
                if wait <= 0 and job_id not in critical_job_scores:
                    continue
                tail_multiplier = (
                    2.0
                    if (
                        job_id in tail_jobs
                        or (curr_end is not None and curr_end >= tail_cutoff)
                    )
                    else 1.0
                )
                critical_bonus = float(critical_job_scores.get(job_id, 0))
                waits[job_id] = wait * tail_multiplier + critical_bonus
                score += waits[job_id]
            boundary_scores[boundary] = score
            boundary_job_waits[boundary] = waits

        if not boundary_scores:
            raise ValueError("No stage boundary available for handoff rectangle.")
        best_boundary = max(
            boundary_scores,
            key=lambda boundary: (boundary_scores[boundary], boundary[0], boundary[1]),
        )
        prev_stage, curr_stage = best_boundary
        job_scores = boundary_job_waits.get(best_boundary, {})
        if not job_scores:
            job_scores = {
                job_id: float(score) for job_id, score in critical_job_scores.items()
            }
        if not job_scores:
            job_scores = {job_id: 1.0 for job_id in schedule.jobs}

        selected_jobs = sorted(
            job_scores,
            key=lambda job_id: (
                -float(job_scores[job_id]),
                -float(last_stage_completion.get(job_id, 0)),
                str(job_id),
            ),
        )[: min(job_count, len(job_scores))]

        prev_idx = stage_order.index(prev_stage)
        curr_idx = stage_order.index(curr_stage)
        left_idx = max(0, prev_idx - stage_radius)
        right_idx = min(len(stage_order) - 1, curr_idx + stage_radius)
        selected_stage_ids = stage_order[left_idx : right_idx + 1]

        logging.info(
            "Handoff rectangle selected boundary=%s stages=%s jobs=%s "
            "tail_cutoff=%s top_boundary_scores=%s top_job_scores=%s",
            best_boundary,
            selected_stage_ids,
            selected_jobs,
            tail_cutoff,
            self._format_top_score_items(boundary_scores),
            self._format_top_score_items(job_scores),
        )
        return selected_jobs, selected_stage_ids, best_boundary

    @staticmethod
    def _format_top_score_items(
        scores: Mapping[object, float | int],
        *,
        top_k: int = 8,
    ) -> list[tuple[object, float]]:
        return [
            (key, float(value))
            for key, value in sorted(
                scores.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )[:top_k]
        ]

    def critical_handoff_rectangle_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        job_count: int | None = None,
        job_count_ratio: float | None = 0.12,
        min_job_count: int = 8,
        max_job_count: int | None = 32,
        stage_radius: int = 1,
        tail_time_ratio: float = 0.25,
        include_singleton_critical_blocks: bool = True,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        resolved_job_count = self._resolve_rectangle_job_count(
            job_count=job_count,
            job_count_ratio=job_count_ratio,
            min_job_count=min_job_count,
            max_job_count=max_job_count,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_critical_handoff_rectangle_operator(
                job_count=resolved_job_count,
                stage_radius=stage_radius,
                tail_time_ratio=tail_time_ratio,
                include_singleton_critical_blocks=include_singleton_critical_blocks,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def workload_guarded_critical_handoff_rectangle_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        job_count: int | None = None,
        job_count_ratio: float | None = 0.12,
        min_job_count: int = 8,
        max_job_count: int | None = 32,
        stage_radius: int = 1,
        tail_time_ratio: float = 0.25,
        include_singleton_critical_blocks: bool = True,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
    ) -> None:
        """Run handoff rectangle only inside paper-safe workload/size/gap guards."""
        if not self._bound_gap_guard_allows(
            context_label="Critical Handoff Rectangle NS",
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return

        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        if min_workload_size is not None and workload_size < int(min_workload_size):
            logging.info(
                "[Critical Handoff Rectangle NS] Skipping: workload_size=%d is below min_workload_size=%d.",
                workload_size,
                int(min_workload_size),
            )
            return
        if max_workload_size is not None and workload_size > int(max_workload_size):
            logging.info(
                "[Critical Handoff Rectangle NS] Skipping: workload_size=%d is above max_workload_size=%d.",
                workload_size,
                int(max_workload_size),
            )
            return
        if min_instance_job_count is not None and instance_job_count < int(
            min_instance_job_count
        ):
            logging.info(
                "[Critical Handoff Rectangle NS] Skipping: job_count=%d is below min_instance_job_count=%d.",
                instance_job_count,
                int(min_instance_job_count),
            )
            return
        if max_instance_job_count is not None and instance_job_count > int(
            max_instance_job_count
        ):
            logging.info(
                "[Critical Handoff Rectangle NS] Skipping: job_count=%d is above max_instance_job_count=%d.",
                instance_job_count,
                int(max_instance_job_count),
            )
            return
        if min_instance_stage_count is not None and instance_stage_count < int(
            min_instance_stage_count
        ):
            logging.info(
                "[Critical Handoff Rectangle NS] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                instance_stage_count,
                int(min_instance_stage_count),
            )
            return
        if max_instance_stage_count is not None and instance_stage_count > int(
            max_instance_stage_count
        ):
            logging.info(
                "[Critical Handoff Rectangle NS] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                instance_stage_count,
                int(max_instance_stage_count),
            )
            return

        logging.info(
            "[Critical Handoff Rectangle NS] Running with job_count=%d stage_count=%d "
            "workload_size=%d inside guards workload=[%s, %s] job_count=[%s, %s] "
            "stage_count=[%s, %s].",
            instance_job_count,
            instance_stage_count,
            workload_size,
            min_workload_size,
            max_workload_size,
            min_instance_job_count,
            max_instance_job_count,
            min_instance_stage_count,
            max_instance_stage_count,
        )
        self.critical_handoff_rectangle_ns(
            solver_thread_cnt=solver_thread_cnt,
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
            job_count=job_count,
            job_count_ratio=job_count_ratio,
            min_job_count=min_job_count,
            max_job_count=max_job_count,
            stage_radius=stage_radius,
            tail_time_ratio=tail_time_ratio,
            include_singleton_critical_blocks=include_singleton_critical_blocks,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_critical_handoff_rectangle_operator(
        self,
        *,
        job_count: int,
        stage_radius: int = 1,
        tail_time_ratio: float = 0.25,
        include_singleton_critical_blocks: bool = True,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> dict[str, object]:
        schedule = self._get_incumbent_schedule_for_rectangle_lns(
            "critical handoff rectangle operator"
        )
        selected_jobs, selected_stages, boundary = (
            self._select_handoff_boundary_jobs_and_stages(
                schedule,
                job_count=job_count,
                stage_radius=stage_radius,
                tail_time_ratio=tail_time_ratio,
                include_singleton_critical_blocks=include_singleton_critical_blocks,
            )
        )
        all_ops = list(schedule.get_jik_2_start_time_map().keys())
        selected_job_set = set(selected_jobs)
        selected_stage_set = set(selected_stages)
        selected_ops = {
            op
            for op in all_ops
            if op[0] in selected_job_set and str(op[1]) in selected_stage_set
        }
        if not selected_ops:
            raise ValueError("Critical handoff rectangle selected no operations.")
        logging.info(
            "Critical handoff rectangle freeing %d jobs, %d stages, %d ops around boundary=%s.",
            len(selected_job_set),
            len(selected_stage_set),
            len(selected_ops),
            boundary,
        )
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )
        return {
            "selected_jobs": selected_jobs,
            "selected_stages": selected_stages,
            "selected_op_count": len(selected_ops),
            "boundary": boundary,
        }

    def _select_cmax_tail_jobs(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        job_count: int,
        tail_time_ratio: float,
        include_singleton_critical_blocks: bool,
        job_rank_offset: int = 0,
        job_rank_stride: int = 1,
        anchor_cmax_job_count: int = 2,
    ) -> tuple[list[JobIdType], Counter[JobIdType], int]:
        if tail_time_ratio <= 0 or tail_time_ratio > 1:
            raise ValueError("tail_time_ratio must satisfy 0 < tail_time_ratio <= 1.")

        schedule.make_semi_active(self.stage_2_job_2_p_dict)
        critical_blocks = schedule.find_critical_blocks(
            self.stage_2_job_2_p_dict,
            include_singletons=include_singleton_critical_blocks,
        )
        critical_job_scores: Counter[JobIdType] = Counter(
            op[0] for block in critical_blocks for op in block
        )
        last_stage_completion = self._get_last_stage_completion_by_job(schedule)
        if not last_stage_completion:
            last_stage_completion = {
                job_id: int(schedule.makespan) for job_id in schedule.jobs
            }

        tail_cutoff = int(max(0, math.floor(schedule.makespan * (1 - tail_time_ratio))))
        candidate_jobs = [
            job_id
            for job_id, completion in last_stage_completion.items()
            if completion >= tail_cutoff
        ]
        if not candidate_jobs:
            candidate_jobs = list(last_stage_completion)
        if not candidate_jobs:
            candidate_jobs = list(schedule.jobs)

        ranked_candidate_jobs = sorted(
            candidate_jobs,
            key=lambda job_id: (
                -float(last_stage_completion.get(job_id, 0)),
                -float(critical_job_scores.get(job_id, 0)),
                str(job_id),
            ),
        )
        stride = max(1, int(job_rank_stride))
        offset = max(0, int(job_rank_offset))
        if ranked_candidate_jobs and offset >= len(ranked_candidate_jobs):
            offset = offset % stride
        selected_jobs: list[JobIdType] = []
        for job_id in ranked_candidate_jobs[: max(0, int(anchor_cmax_job_count))]:
            if job_id not in selected_jobs:
                selected_jobs.append(job_id)
            if len(selected_jobs) >= job_count:
                break
        for job_id in ranked_candidate_jobs[offset::stride]:
            if job_id not in selected_jobs:
                selected_jobs.append(job_id)
            if len(selected_jobs) >= job_count:
                break
        for job_id in ranked_candidate_jobs:
            if job_id not in selected_jobs:
                selected_jobs.append(job_id)
            if len(selected_jobs) >= job_count:
                break

        logging.info(
            "Cmax-cone tail jobs selected: jobs=%s tail_cutoff=%s "
            "rank_offset=%s rank_stride=%s top_last_stage_completion=%s "
            "top_critical_scores=%s",
            selected_jobs,
            tail_cutoff,
            offset,
            stride,
            self._format_top_score_items(last_stage_completion),
            self._format_top_score_items(critical_job_scores),
        )
        return selected_jobs, critical_job_scores, tail_cutoff

    def _select_last_stage_suffix_jobs(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        suffix_job_count: int | None,
        suffix_job_count_ratio: float | None,
        min_suffix_jobs_per_machine: int,
        max_suffix_jobs: int | None,
    ) -> list[JobIdType]:
        if not schedule.stages:
            return []
        last_stage = schedule.stages[-1]
        selected_jobs: set[JobIdType] = set()
        completion_by_job = self._get_last_stage_completion_by_job(schedule)
        ops_by_machine: dict[object, list[tuple[int, JobIdType]]] = {}
        for mc_id, _start_time, end_time, job_id in schedule.iter_operations_on_stage(
            last_stage
        ):
            ops_by_machine.setdefault(mc_id, []).append((int(end_time), job_id))

        for ops in ops_by_machine.values():
            ops.sort(key=lambda item: (item[0], str(item[1])))
            if suffix_job_count is not None:
                per_machine_count = int(suffix_job_count)
            else:
                ratio = (
                    0.12
                    if suffix_job_count_ratio is None
                    else float(suffix_job_count_ratio)
                )
                if ratio <= 0:
                    raise ValueError("suffix_job_count_ratio must be positive.")
                per_machine_count = math.ceil(len(ops) * ratio)
            per_machine_count = max(int(min_suffix_jobs_per_machine), per_machine_count)
            for _end_time, job_id in ops[-min(per_machine_count, len(ops)) :]:
                selected_jobs.add(job_id)

        ranked_jobs = sorted(
            selected_jobs,
            key=lambda job_id: (-float(completion_by_job.get(job_id, 0)), str(job_id)),
        )
        if max_suffix_jobs is not None:
            ranked_jobs = ranked_jobs[: max(0, int(max_suffix_jobs))]
        logging.info(
            "Cmax-cone last-stage suffix jobs selected: jobs=%s per_machine_suffix=%s "
            "machines=%d",
            ranked_jobs,
            suffix_job_count
            if suffix_job_count is not None
            else suffix_job_count_ratio,
            len(ops_by_machine),
        )
        return ranked_jobs

    def _rank_cmax_cone_jobs(
        self,
        jobs: Sequence[JobIdType],
        *,
        last_stage_completion: Mapping[JobIdType, int],
        critical_job_scores: Mapping[JobIdType, int | float],
    ) -> list[JobIdType]:
        return sorted(
            dict.fromkeys(jobs),
            key=lambda job_id: (
                -float(last_stage_completion.get(job_id, 0)),
                -float(critical_job_scores.get(job_id, 0)),
                str(job_id),
            ),
        )

    def _resolve_cmax_cone_stage_ids(
        self,
        schedule: HybridFlowshopLiteSchedule,
        *,
        selected_jobs: Sequence[JobIdType],
        tail_stage_count: int | None,
        tail_stage_ratio: float | None,
        include_bottleneck_stage: bool,
        bottleneck_stage_radius: int,
        include_handoff_boundary: bool,
        handoff_stage_radius: int,
        handoff_tail_time_ratio: float,
        include_critical_stage: bool,
        critical_stage_radius: int,
        include_singleton_critical_blocks: bool,
    ) -> tuple[list[str], dict[str, object]]:
        stage_order = list(self.instance.stage_id_list)
        selected_stage_ids: set[str] = set(
            self._resolve_tail_stage_ids(
                tail_stage_count=tail_stage_count,
                tail_stage_ratio=tail_stage_ratio,
            )
        )
        selection_notes: dict[str, object] = {
            "tail_stages": sorted(selected_stage_ids, key=stage_order.index),
        }

        if include_bottleneck_stage:
            bottleneck_stage = self._resolve_bottleneck_stage_id_for_targeted_ns()
            bottleneck_band = self._resolve_stage_band_ids(
                bottleneck_stage,
                radius=bottleneck_stage_radius,
            )
            selected_stage_ids.update(bottleneck_band)
            selection_notes["bottleneck_stage"] = bottleneck_stage
            selection_notes["bottleneck_band"] = bottleneck_band

        if include_handoff_boundary:
            handoff_jobs, handoff_stages, boundary = (
                self._select_handoff_boundary_jobs_and_stages(
                    schedule,
                    job_count=max(1, len(selected_jobs)),
                    stage_radius=handoff_stage_radius,
                    tail_time_ratio=handoff_tail_time_ratio,
                    include_singleton_critical_blocks=include_singleton_critical_blocks,
                )
            )
            selected_stage_ids.update(handoff_stages)
            selection_notes["handoff_boundary"] = boundary
            selection_notes["handoff_stages"] = handoff_stages
            selection_notes["handoff_jobs"] = handoff_jobs

        if include_critical_stage:
            critical_stage = self._resolve_critical_stage_id_for_targeted_ns(
                include_singleton_critical_blocks=include_singleton_critical_blocks,
            )
            critical_band = self._resolve_stage_band_ids(
                critical_stage,
                radius=critical_stage_radius,
            )
            selected_stage_ids.update(critical_band)
            selection_notes["critical_stage"] = critical_stage
            selection_notes["critical_band"] = critical_band

        ordered_stages = [
            stage_id for stage_id in stage_order if str(stage_id) in selected_stage_ids
        ]
        return ordered_stages, selection_notes

    def critical_cmax_cone_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        job_count: int | None = None,
        job_count_ratio: float | None = 0.10,
        min_job_count: int = 8,
        max_job_count: int | None = 36,
        tail_time_ratio: float = 0.20,
        include_last_stage_suffix: bool = True,
        last_stage_suffix_job_count: int | None = None,
        last_stage_suffix_job_count_ratio: float | None = 0.12,
        min_last_stage_suffix_jobs_per_machine: int = 1,
        max_last_stage_suffix_jobs: int | None = 24,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = 0.40,
        include_bottleneck_stage: bool = True,
        bottleneck_stage_radius: int = 1,
        include_handoff_boundary: bool = True,
        handoff_stage_radius: int = 1,
        handoff_tail_time_ratio: float | None = None,
        include_handoff_jobs: bool = True,
        include_critical_stage: bool = True,
        critical_stage_radius: int = 1,
        include_singleton_critical_blocks: bool = True,
        job_rank_offset: int = 0,
        job_rank_stride: int = 1,
        anchor_cmax_job_count: int = 2,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        resolved_job_count = self._resolve_rectangle_job_count(
            job_count=job_count,
            job_count_ratio=job_count_ratio,
            min_job_count=min_job_count,
            max_job_count=max_job_count,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_critical_cmax_cone_operator(
                job_count=resolved_job_count,
                tail_time_ratio=tail_time_ratio,
                include_last_stage_suffix=include_last_stage_suffix,
                last_stage_suffix_job_count=last_stage_suffix_job_count,
                last_stage_suffix_job_count_ratio=last_stage_suffix_job_count_ratio,
                min_last_stage_suffix_jobs_per_machine=(
                    min_last_stage_suffix_jobs_per_machine
                ),
                max_last_stage_suffix_jobs=max_last_stage_suffix_jobs,
                tail_stage_count=tail_stage_count,
                tail_stage_ratio=tail_stage_ratio,
                include_bottleneck_stage=include_bottleneck_stage,
                bottleneck_stage_radius=bottleneck_stage_radius,
                include_handoff_boundary=include_handoff_boundary,
                handoff_stage_radius=handoff_stage_radius,
                handoff_tail_time_ratio=(
                    tail_time_ratio
                    if handoff_tail_time_ratio is None
                    else handoff_tail_time_ratio
                ),
                include_handoff_jobs=include_handoff_jobs,
                include_critical_stage=include_critical_stage,
                critical_stage_radius=critical_stage_radius,
                include_singleton_critical_blocks=include_singleton_critical_blocks,
                job_rank_offset=job_rank_offset,
                job_rank_stride=job_rank_stride,
                anchor_cmax_job_count=anchor_cmax_job_count,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def critical_cmax_cone_sweep_ns(
        self,
        solver_thread_cnt: int,
        attempt_count: int = 24,
        per_attempt_computational_time: float | None = None,
        per_attempt_tl_nc_multiplier: float | None = 0.0015,
        job_count_ratio_values: Sequence[float] | None = None,
        min_job_count: int = 8,
        max_job_count: int | None = 28,
        tail_time_ratio_values: Sequence[float] | None = None,
        include_last_stage_suffix: bool = True,
        last_stage_suffix_job_count_ratio_values: Sequence[float] | None = None,
        min_last_stage_suffix_jobs_per_machine: int = 1,
        max_last_stage_suffix_jobs: int | None = 24,
        tail_stage_ratio_values: Sequence[float] | None = None,
        stage_mode_cycle: Sequence[str] | None = None,
        bottleneck_stage_radius_values: Sequence[int] | None = None,
        handoff_stage_radius_values: Sequence[int] | None = None,
        critical_stage_radius_values: Sequence[int] | None = None,
        job_rank_stride: int = 4,
        anchor_cmax_job_count: int = 2,
        include_singleton_critical_blocks: bool = True,
        no_improvement_timelimit: float | None = None,
        stop_after_no_improvement_attempts: int | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = True,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        if attempt_count <= 0:
            logging.info(
                "[Cmax Cone Sweep] Skipping because attempt_count=%s.", attempt_count
            )
            return

        job_count_ratios = list(job_count_ratio_values or [0.05, 0.07, 0.09])
        tail_time_ratios = list(tail_time_ratio_values or [0.14, 0.20, 0.28])
        suffix_ratios = list(last_stage_suffix_job_count_ratio_values or [0.06, 0.10])
        tail_stage_ratios = list(tail_stage_ratio_values or [0.30, 0.40, 0.50])
        stage_modes = list(
            stage_mode_cycle
            or [
                "tail",
                "tail_bottleneck",
                "tail_handoff",
                "tail_bottleneck_handoff",
                "tail_critical",
            ]
        )
        bottleneck_radii = list(bottleneck_stage_radius_values or [0, 1])
        handoff_radii = list(handoff_stage_radius_values or [0, 1])
        critical_radii = list(critical_stage_radius_values or [0, 1])

        per_attempt_time = self._resolve_tl_nc_computational_time(
            computational_time=per_attempt_computational_time,
            tl_nc_multiplier=per_attempt_tl_nc_multiplier,
        )
        no_improvement_attempts = 0
        improvement_count = 0
        initial_obj = sanitize_optional_float(self.solution_manager.best_obj_value)
        logging.info(
            "[Cmax Cone Sweep] Starting attempts=%d per_attempt_time=%s initial_obj=%s.",
            attempt_count,
            per_attempt_time,
            initial_obj,
        )

        for attempt_idx in range(int(attempt_count)):
            mode = str(stage_modes[attempt_idx % len(stage_modes)])
            include_bottleneck_stage = mode in {
                "full",
                "cone",
                "tail_bottleneck",
                "tail_bottleneck_handoff",
                "tail_bottleneck_critical",
            }
            include_handoff_boundary = mode in {
                "full",
                "cone",
                "tail_handoff",
                "tail_bottleneck_handoff",
                "tail_handoff_critical",
            }
            include_critical_stage = mode in {
                "full",
                "cone",
                "tail_critical",
                "tail_bottleneck_critical",
                "tail_handoff_critical",
            }
            job_count_ratio = float(
                job_count_ratios[attempt_idx % len(job_count_ratios)]
            )
            tail_time_ratio = float(
                tail_time_ratios[
                    (attempt_idx // len(job_count_ratios)) % len(tail_time_ratios)
                ]
            )
            tail_stage_ratio = float(
                tail_stage_ratios[
                    (
                        attempt_idx
                        // max(1, len(job_count_ratios) * len(tail_time_ratios))
                    )
                    % len(tail_stage_ratios)
                ]
            )
            suffix_ratio = float(suffix_ratios[attempt_idx % len(suffix_ratios)])
            bottleneck_radius = int(
                bottleneck_radii[attempt_idx % len(bottleneck_radii)]
            )
            handoff_radius = int(
                handoff_radii[
                    (attempt_idx // max(1, len(bottleneck_radii))) % len(handoff_radii)
                ]
            )
            critical_radius = int(
                critical_radii[
                    (attempt_idx // max(1, len(bottleneck_radii) * len(handoff_radii)))
                    % len(critical_radii)
                ]
            )
            resolved_job_count = self._resolve_rectangle_job_count(
                job_count=None,
                job_count_ratio=job_count_ratio,
                min_job_count=min_job_count,
                max_job_count=max_job_count,
            )
            rank_stride = max(1, int(job_rank_stride))
            job_rank_offset = attempt_idx % rank_stride
            before_obj = sanitize_optional_float(self.solution_manager.best_obj_value)
            logging.info(
                "[Cmax Cone Sweep] Attempt %d/%d mode=%s jobs=%d ratio=%.4f "
                "tail_time=%.3f tail_stage=%.3f suffix=%.3f rank_offset=%d.",
                attempt_idx + 1,
                attempt_count,
                mode,
                resolved_job_count,
                job_count_ratio,
                tail_time_ratio,
                tail_stage_ratio,
                suffix_ratio,
                job_rank_offset,
            )

            self._fix_profile_solve_reset(
                lambda resolved_job_count=resolved_job_count, tail_time_ratio=tail_time_ratio, suffix_ratio=suffix_ratio, tail_stage_ratio=tail_stage_ratio, include_bottleneck_stage=include_bottleneck_stage, bottleneck_radius=bottleneck_radius, include_handoff_boundary=include_handoff_boundary, handoff_radius=handoff_radius, include_critical_stage=include_critical_stage, critical_radius=critical_radius, job_rank_offset=job_rank_offset: (
                    self.apply_critical_cmax_cone_operator(
                        job_count=resolved_job_count,
                        tail_time_ratio=tail_time_ratio,
                        include_last_stage_suffix=include_last_stage_suffix,
                        last_stage_suffix_job_count=None,
                        last_stage_suffix_job_count_ratio=suffix_ratio,
                        min_last_stage_suffix_jobs_per_machine=(
                            min_last_stage_suffix_jobs_per_machine
                        ),
                        max_last_stage_suffix_jobs=max_last_stage_suffix_jobs,
                        tail_stage_count=None,
                        tail_stage_ratio=tail_stage_ratio,
                        include_bottleneck_stage=include_bottleneck_stage,
                        bottleneck_stage_radius=bottleneck_radius,
                        include_handoff_boundary=include_handoff_boundary,
                        handoff_stage_radius=handoff_radius,
                        handoff_tail_time_ratio=tail_time_ratio,
                        include_handoff_jobs=include_handoff_boundary,
                        include_critical_stage=include_critical_stage,
                        critical_stage_radius=critical_radius,
                        include_singleton_critical_blocks=include_singleton_critical_blocks,
                        job_rank_offset=job_rank_offset,
                        job_rank_stride=rank_stride,
                        anchor_cmax_job_count=anchor_cmax_job_count,
                        profile_fix_by_machine=profile_fix_by_machine,
                        machine_precedence_stride=machine_precedence_stride,
                    )
                ),
                per_attempt_time,
                solver_thread_cnt,
                no_improvement_timelimit=no_improvement_timelimit,
                swap_before_cp=swap_before_cp,
                make_semi_active_after_cp=make_semi_active_after_cp,
                use_lns_only=use_lns_only,
                obj_value_is_valid=True,
                obj_bound_is_valid=False,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )
            after_obj = sanitize_optional_float(self.solution_manager.best_obj_value)
            improved = (
                before_obj is not None
                and after_obj is not None
                and after_obj < before_obj
            )
            if improved:
                improvement_count += 1
                no_improvement_attempts = 0
            else:
                no_improvement_attempts += 1
            logging.info(
                "[Cmax Cone Sweep] Attempt %d finished: before=%s after=%s "
                "improved=%s consecutive_no_improve=%d.",
                attempt_idx + 1,
                before_obj,
                after_obj,
                improved,
                no_improvement_attempts,
            )
            if (
                stop_after_no_improvement_attempts is not None
                and no_improvement_attempts >= int(stop_after_no_improvement_attempts)
            ):
                logging.info(
                    "[Cmax Cone Sweep] Stop after %d consecutive non-improving attempts.",
                    no_improvement_attempts,
                )
                break

        logging.info(
            "[Cmax Cone Sweep] Finished: initial_obj=%s final_obj=%s improvements=%d.",
            initial_obj,
            sanitize_optional_float(self.solution_manager.best_obj_value),
            improvement_count,
        )

    def apply_critical_cmax_cone_operator(
        self,
        *,
        job_count: int,
        tail_time_ratio: float = 0.20,
        include_last_stage_suffix: bool = True,
        last_stage_suffix_job_count: int | None = None,
        last_stage_suffix_job_count_ratio: float | None = 0.12,
        min_last_stage_suffix_jobs_per_machine: int = 1,
        max_last_stage_suffix_jobs: int | None = 24,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = 0.40,
        include_bottleneck_stage: bool = True,
        bottleneck_stage_radius: int = 1,
        include_handoff_boundary: bool = True,
        handoff_stage_radius: int = 1,
        handoff_tail_time_ratio: float = 0.20,
        include_handoff_jobs: bool = True,
        include_critical_stage: bool = True,
        critical_stage_radius: int = 1,
        include_singleton_critical_blocks: bool = True,
        job_rank_offset: int = 0,
        job_rank_stride: int = 1,
        anchor_cmax_job_count: int = 2,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> dict[str, object]:
        schedule = self._get_incumbent_schedule_for_rectangle_lns(
            "critical Cmax-cone operator"
        )
        selected_jobs, critical_job_scores, tail_cutoff = self._select_cmax_tail_jobs(
            schedule,
            job_count=job_count,
            tail_time_ratio=tail_time_ratio,
            include_singleton_critical_blocks=include_singleton_critical_blocks,
            job_rank_offset=job_rank_offset,
            job_rank_stride=job_rank_stride,
            anchor_cmax_job_count=anchor_cmax_job_count,
        )
        candidate_jobs = list(selected_jobs)
        if include_last_stage_suffix:
            candidate_jobs.extend(
                self._select_last_stage_suffix_jobs(
                    schedule,
                    suffix_job_count=last_stage_suffix_job_count,
                    suffix_job_count_ratio=last_stage_suffix_job_count_ratio,
                    min_suffix_jobs_per_machine=min_last_stage_suffix_jobs_per_machine,
                    max_suffix_jobs=max_last_stage_suffix_jobs,
                )
            )

        selected_stages, selection_notes = self._resolve_cmax_cone_stage_ids(
            schedule,
            selected_jobs=selected_jobs,
            tail_stage_count=tail_stage_count,
            tail_stage_ratio=tail_stage_ratio,
            include_bottleneck_stage=include_bottleneck_stage,
            bottleneck_stage_radius=bottleneck_stage_radius,
            include_handoff_boundary=include_handoff_boundary,
            handoff_stage_radius=handoff_stage_radius,
            handoff_tail_time_ratio=handoff_tail_time_ratio,
            include_critical_stage=include_critical_stage,
            critical_stage_radius=critical_stage_radius,
            include_singleton_critical_blocks=include_singleton_critical_blocks,
        )
        if include_handoff_jobs:
            candidate_jobs.extend(selection_notes.get("handoff_jobs", []) or [])

        last_stage_completion = self._get_last_stage_completion_by_job(schedule)
        ranked_jobs = self._rank_cmax_cone_jobs(
            candidate_jobs,
            last_stage_completion=last_stage_completion,
            critical_job_scores=critical_job_scores,
        )[: min(job_count, len(set(candidate_jobs)))]
        selected_job_set = set(ranked_jobs)
        selected_stage_set = set(selected_stages)

        all_ops = list(schedule.get_jik_2_start_time_map().keys())
        selected_ops = {
            op
            for op in all_ops
            if op[0] in selected_job_set and str(op[1]) in selected_stage_set
        }
        if not selected_ops:
            raise ValueError("Critical Cmax-cone selected no operations.")

        logging.info(
            "Critical Cmax-cone freeing jobs=%s stages=%s ops=%d tail_cutoff=%s "
            "notes=%s",
            ranked_jobs,
            selected_stages,
            len(selected_ops),
            tail_cutoff,
            selection_notes,
        )
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )
        return {
            "selected_jobs": ranked_jobs,
            "selected_stages": selected_stages,
            "selected_op_count": len(selected_ops),
            "tail_cutoff": tail_cutoff,
            "selection_notes": selection_notes,
        }

    def _apply_stage_selection_operator(
        self,
        selected_stages: Sequence[StageIdType],
        *,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> list[str]:
        if not self.solution_manager.has_incumbent():
            raise ValueError("No incumbent solution available for stage selection.")
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError("Incumbent solution is not a HybridFlowshopLiteSchedule.")

        stage_order = list(self.instance.stage_id_list)
        selected_stage_ids = sorted(
            {
                str(stage_id)
                for stage_id in selected_stages
                if str(stage_id) in stage_order
            },
            key=stage_order.index,
        )
        if not selected_stage_ids:
            raise ValueError(
                "selected_stages must contain at least one valid stage ID."
            )

        all_ops = list(incumbent_solution.get_jik_2_start_time_map().keys())
        selected_stage_set = set(selected_stage_ids)
        selected_ops = {op for op in all_ops if op[1] in selected_stage_set}
        logging.info(
            "Stage-selection operator freeing %d stages and %d ops: %s",
            len(selected_stage_ids),
            len(selected_ops),
            selected_stage_ids,
        )
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )
        return selected_stage_ids

    def _resolve_bottleneck_stage_id_for_targeted_ns(self) -> str:
        retained_result = getattr(self, "last_retained_cp_lb_result", None)
        if retained_result is not None:
            selected_bottleneck_stage_ids = list(
                getattr(retained_result, "selected_bottleneck_stage_ids", ()) or ()
            )
            if selected_bottleneck_stage_ids:
                return str(selected_bottleneck_stage_ids[0])
            bottleneck_stage_id = getattr(retained_result, "bottleneck_stage_id", None)
            if bottleneck_stage_id is not None:
                return str(bottleneck_stage_id)

        params = BaseModelBuilder.make_params(self.instance)
        internal_stage_ids = list(params.i_list[1:-1]) or list(params.i_list)
        return str(
            select_bottleneck_stage_by_average_load(
                params,
                stage_ids=internal_stage_ids,
            )
        )

    def _resolve_stage_band_ids(
        self,
        center_stage_id: StageIdType,
        *,
        radius: int,
    ) -> list[str]:
        if radius < 0:
            raise ValueError(
                f"radius must be non-negative for stage band selection. Received {radius}."
            )
        stage_id_list = list(self.instance.stage_id_list)
        if str(center_stage_id) not in stage_id_list:
            raise ValueError(f"Unknown stage_id={center_stage_id!r} for stage band.")
        center_idx = stage_id_list.index(str(center_stage_id))
        left_idx = max(0, center_idx - radius)
        right_idx = min(len(stage_id_list) - 1, center_idx + radius)
        return stage_id_list[left_idx : right_idx + 1]

    def retained_cp_bottleneck_band_stage_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        radius: int = 2,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_retained_cp_bottleneck_band_stage_operator(
                radius=radius,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def workload_guarded_retained_cp_bottleneck_band_stage_ns(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        radius: int = 2,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
        min_retained_cp_restore_loss_ratio: float | None = None,
        max_retained_cp_restore_loss_ratio: float | None = None,
        run_if_retained_cp_restore_missing: bool = True,
    ) -> None:
        """Run retained-CP bottleneck-band stage repair only inside configured guards."""
        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        if not self._bound_gap_guard_allows(
            context_label="Retained-CP Bottleneck Band NS",
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return
        if not self._retained_cp_restore_loss_guard_allows(
            context_label="Retained-CP Bottleneck Band NS",
            min_retained_cp_restore_loss_ratio=min_retained_cp_restore_loss_ratio,
            max_retained_cp_restore_loss_ratio=max_retained_cp_restore_loss_ratio,
            run_if_retained_cp_restore_missing=run_if_retained_cp_restore_missing,
        ):
            return
        if min_workload_size is not None and workload_size < int(min_workload_size):
            logging.info(
                "[Retained-CP Bottleneck Band NS] Skipping: workload_size=%d is below min_workload_size=%d.",
                workload_size,
                int(min_workload_size),
            )
            return
        if max_workload_size is not None and workload_size > int(max_workload_size):
            logging.info(
                "[Retained-CP Bottleneck Band NS] Skipping: workload_size=%d is above max_workload_size=%d.",
                workload_size,
                int(max_workload_size),
            )
            return
        if min_instance_job_count is not None and instance_job_count < int(
            min_instance_job_count
        ):
            logging.info(
                "[Retained-CP Bottleneck Band NS] Skipping: job_count=%d is below min_instance_job_count=%d.",
                instance_job_count,
                int(min_instance_job_count),
            )
            return
        if max_instance_job_count is not None and instance_job_count > int(
            max_instance_job_count
        ):
            logging.info(
                "[Retained-CP Bottleneck Band NS] Skipping: job_count=%d is above max_instance_job_count=%d.",
                instance_job_count,
                int(max_instance_job_count),
            )
            return
        if min_instance_stage_count is not None and instance_stage_count < int(
            min_instance_stage_count
        ):
            logging.info(
                "[Retained-CP Bottleneck Band NS] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                instance_stage_count,
                int(min_instance_stage_count),
            )
            return
        if max_instance_stage_count is not None and instance_stage_count > int(
            max_instance_stage_count
        ):
            logging.info(
                "[Retained-CP Bottleneck Band NS] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                instance_stage_count,
                int(max_instance_stage_count),
            )
            return
        logging.info(
            "[Retained-CP Bottleneck Band NS] Running with job_count=%d stage_count=%d workload_size=%d "
            "inside guards workload=[%s, %s] job_count=[%s, %s] stage_count=[%s, %s].",
            instance_job_count,
            instance_stage_count,
            workload_size,
            min_workload_size,
            max_workload_size,
            min_instance_job_count,
            max_instance_job_count,
            min_instance_stage_count,
            max_instance_stage_count,
        )
        self.retained_cp_bottleneck_band_stage_ns(
            solver_thread_cnt=solver_thread_cnt,
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
            radius=radius,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_retained_cp_bottleneck_band_stage_operator(
        self,
        radius: int = 2,
        *,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> list[str]:
        bottleneck_stage_id = self._resolve_bottleneck_stage_id_for_targeted_ns()
        selected_stage_ids = self._resolve_stage_band_ids(
            bottleneck_stage_id,
            radius=radius,
        )
        logging.info(
            "Applying retained-CP bottleneck-band stage operator: bottleneck_stage=%s radius=%d stages=%s",
            bottleneck_stage_id,
            radius,
            selected_stage_ids,
        )
        return self._apply_stage_selection_operator(
            selected_stage_ids,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )

    # Subroutine: Job-block neighbor search

    def job_block_ns(
        self,
        rho: float,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        seed_op_from_critical_block: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_job_block_operator(
                rho,
                seed_op_from_critical_block=seed_op_from_critical_block,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_job_block_operator(
        self,
        rho: float,
        seed_op_from_critical_block: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> None:
        """Apply the job-block operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            seed_op_from_critical_block (bool, optional): If True, the seed operation is
                chosen from critical blocks of the incumbent solution. Defaults to False.
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                adjacency; otherwise apply stage-level time-based selection.
                Defaults to False.
            machine_precedence_stride (int, optional): The stride for selecting machine precedences.
                Defaults to 1.

        Raises:
            ValueError: If rho is not strictly positive.
            ValueError: If no incumbent solution is available.
            ValueError: If the incumbent solution is not a valid HybridFlowshopLiteSchedule instance.
            ValueError: If no start or end times are available.
        """
        if rho <= 0:
            raise ValueError(f"Invalid value for rho {rho}; it must be positive.")
        logging.info(f"Applying job-block operator with rho={rho}")
        if not self.solution_manager.has_incumbent():
            raise ValueError("No incumbent solution available for job-block operator.")
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError("Incumbent solution is not a HybridFlowshopLiteSchedule.")
        start_time_map = incumbent_solution.get_jik_2_start_time_map()
        end_time_map = incumbent_solution.get_jik_2_end_time_map()

        if not start_time_map or not end_time_map:
            raise ValueError("No solution available for job-block operator.")

        all_ops = list(start_time_map.keys())
        total_ops = len(all_ops)
        num_to_select = max(1, int(rho * total_ops))

        # Choose an operation
        if seed_op_from_critical_block:
            # Find critical blocks in the incumbent solution
            incumbent_solution.make_semi_active(self.stage_2_job_2_p_dict)
            critical_blocks = incumbent_solution.find_critical_blocks(
                self.stage_2_job_2_p_dict, include_singletons=True
            )
            if not critical_blocks:
                # If no critical blocks, fall back to random selection
                seed_op = random.choice(all_ops)
            else:
                # Select a random operation from a random critical block
                selected_block = random.choice(critical_blocks)
                seed_op = random.choice(selected_block)
        else:
            seed_op = random.choice(all_ops)
        # Select all operations of the selected job
        selected_ops = set(op2 for op2 in all_ops if op2[0] == seed_op[0])
        selected_jobs = set([seed_op[0]])
        queue = [seed_op]

        # Expand to overlapping operations
        while queue and len(selected_ops) < num_to_select:
            current_op = queue.pop(0)
            cs, ce = start_time_map[current_op], end_time_map[current_op]
            for op in all_ops:
                if op in selected_ops:
                    continue
                os, oe = start_time_map[op], end_time_map[op]
                if self.closed_intervals_overlap(cs, ce, os, oe):
                    # Select all operations of each overlapping operation
                    selected_ops.update([op2 for op2 in all_ops if op2[0] == op[0]])
                    selected_jobs.add(op[0])
                    queue.append(op)
                if len(selected_ops) >= num_to_select:
                    break
        logging.info(
            f"Job-block operator selected {len(selected_ops)} overlapping ops"
            f" of {len(selected_jobs)} jobs"
            f" (target={num_to_select}; total={total_ops})"
        )

        # Fix out-of-block operations' profile
        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )

    # Subroutine: Critical-job neighbor search

    def critical_job_ns(
        self,
        job_count: int,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        job_selection_policy: str = "critical_adjacency",
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        selection_state: dict[str, object] = {}

        def profile_fixing_method() -> None:
            selection_state["selected_jobs"] = self.apply_critical_job_operator(
                job_count,
                job_selection_policy=job_selection_policy,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            )

        def pre_solve_visualizer() -> None:
            if not draw_gantt:
                return
            incumbent_solution = self.solution_manager.get_incumbent()
            if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
                logging.warning(
                    "No incumbent solution available to draw pre-solve critical-job Gantt."
                )
                return
            selection_state["before_schedule"] = incumbent_solution.deepcopy()

        def post_solve_visualizer(solution: HybridFlowshopLiteSchedule | None) -> None:
            if not draw_gantt:
                return
            selected_jobs = selection_state.get("selected_jobs")
            before_schedule = selection_state.get("before_schedule")
            if not isinstance(before_schedule, HybridFlowshopLiteSchedule):
                logging.warning(
                    "Missing pre-solve schedule for critical-job Gantt comparison."
                )
                return
            if not isinstance(selected_jobs, list) or not all(
                isinstance(job_id, str) for job_id in selected_jobs
            ):
                logging.warning(
                    "Missing selected critical-job list for Gantt highlighting."
                )
                return
            self._draw_critical_job_comparison_gantts(
                before_schedule,
                solution,
                selected_jobs,
            )

        self._fix_profile_solve_reset(
            profile_fixing_method,
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=False,
            pre_solve_visualizer=pre_solve_visualizer if draw_gantt else None,
            post_solve_visualizer=post_solve_visualizer if draw_gantt else None,
        )

    @staticmethod
    def _normalize_job_count(job_count: int) -> int:
        normalized = int(job_count)
        if normalized != job_count:
            raise ValueError(
                f"Invalid job_count {job_count}; it must be an integer value."
            )
        if normalized <= 0:
            raise ValueError(
                f"Invalid job_count {job_count}; it must be strictly positive."
            )
        return normalized

    def _find_critical_blocks_for_critical_job_ns(
        self, schedule: HybridFlowshopLiteSchedule
    ) -> list[list[OperationType]]:
        schedule.make_semi_active(self.stage_2_job_2_p_dict)
        critical_blocks = schedule.find_critical_blocks(
            self.stage_2_job_2_p_dict, include_singletons=True
        )
        if critical_blocks:
            return critical_blocks

        start_time_map = schedule.get_jik_2_start_time_map()
        end_time_map = schedule.get_jik_2_end_time_map()
        last_stage_ops = 0
        if schedule.stages:
            last_stage_ops = sum(
                1 for _ in schedule.iter_operations_on_stage(schedule.stages[-1])
            )
        logging.warning(
            "No critical blocks found after make_semi_active; using uniform random fallback. "
            "makespan=%s jobs=%d stages=%d start_ops=%d end_ops=%d last_stage_ops=%d",
            schedule.makespan,
            len(schedule.jobs),
            len(schedule.stages),
            len(start_time_map),
            len(end_time_map),
            last_stage_ops,
        )
        return []

    @staticmethod
    def _get_critical_job_weights(
        critical_blocks: Sequence[Sequence[OperationType]],
    ) -> dict[JobIdType, int]:
        return dict(Counter(op[0] for block in critical_blocks for op in block))

    @staticmethod
    def _build_highlight_op_set_for_jobs(
        schedule: HybridFlowshopLiteSchedule,
        selected_jobs: Sequence[JobIdType],
    ) -> set[tuple[str, str]]:
        selected_job_set = set(selected_jobs)
        return {
            (job_id, stage_id)
            for (job_id, stage_id, _mc_id) in schedule.get_jik_2_start_time_map()
            if job_id in selected_job_set
        }

    def _draw_critical_job_comparison_gantts(
        self,
        before_schedule: HybridFlowshopLiteSchedule,
        after_schedule: HybridFlowshopLiteSchedule | None,
        selected_jobs: Sequence[JobIdType],
    ) -> None:
        schedules_to_draw: list[tuple[str, HybridFlowshopLiteSchedule]] = [
            ("before", before_schedule)
        ]
        if isinstance(after_schedule, HybridFlowshopLiteSchedule):
            schedules_to_draw.append(("after", after_schedule))
        else:
            logging.warning(
                "No feasible post-solve schedule available to draw critical-job after Gantt."
            )

        force_end = max(int(schedule.makespan) for _, schedule in schedules_to_draw)
        for suffix, schedule in schedules_to_draw:
            highlight_op_set = self._build_highlight_op_set_for_jobs(
                schedule,
                selected_jobs,
            )
            output_path = self.get_file_path_for_subroutine(
                f"_gantt_critical_job_{suffix}.png"
            )
            self.draw_gantt(
                schedule,
                output_path=output_path,
                force_start=0,
                force_end=force_end,
                highlight_op_set=highlight_op_set,
            )

    @staticmethod
    def _weighted_sample_without_replacement(
        items: Sequence[JobIdType],
        item_2_weight: Mapping[JobIdType, int | float],
        sample_size: int,
    ) -> list[JobIdType]:
        if sample_size <= 0 or not items:
            return []

        remaining_items = list(items)
        selected: list[JobIdType] = []
        while remaining_items and len(selected) < sample_size:
            weights = [
                max(0.0, float(item_2_weight.get(item, 0))) for item in remaining_items
            ]
            if sum(weights) <= 0:
                selected.extend(
                    random.sample(
                        remaining_items,
                        k=min(sample_size - len(selected), len(remaining_items)),
                    )
                )
                break
            chosen = random.choices(remaining_items, weights=weights, k=1)[0]
            selected.append(chosen)
            remaining_items.remove(chosen)
        return selected

    @staticmethod
    def _rank_critical_jobs_by_weight(
        items: Sequence[JobIdType],
        item_2_weight: Mapping[JobIdType, int | float],
    ) -> list[JobIdType]:
        return sorted(
            items,
            key=lambda item: (-float(item_2_weight.get(item, 0)), str(item)),
        )

    @staticmethod
    def _unique_jobs_in_block(
        block: Sequence[OperationType],
    ) -> list[JobIdType]:
        seen: set[JobIdType] = set()
        ordered_jobs: list[JobIdType] = []
        for job_id, _stage_id, _mc_id in block:
            if job_id in seen:
                continue
            seen.add(job_id)
            ordered_jobs.append(job_id)
        return ordered_jobs

    @classmethod
    def _select_jobs_around_block_midpoint(
        cls,
        block_jobs: Sequence[JobIdType],
        *,
        target_count: int,
        midpoint_ratio: float = 0.35,
    ) -> list[JobIdType]:
        if target_count <= 0 or not block_jobs:
            return []

        pivot_idx = int((len(block_jobs) - 1) * midpoint_ratio)
        pivot_idx = max(0, min(len(block_jobs) - 1, pivot_idx))

        selected = [block_jobs[pivot_idx]]
        step = 1
        while len(selected) < target_count and (
            pivot_idx - step >= 0 or pivot_idx + step < len(block_jobs)
        ):
            right_idx = pivot_idx + step
            if right_idx < len(block_jobs):
                selected.append(block_jobs[right_idx])
                if len(selected) >= target_count:
                    break

            left_idx = pivot_idx - step
            if left_idx >= 0:
                selected.append(block_jobs[left_idx])
                if len(selected) >= target_count:
                    break
            step += 1
        return selected

    @staticmethod
    def _build_critical_job_adjacency(
        critical_blocks: Sequence[Sequence[OperationType]],
        all_jobs: Sequence[JobIdType],
    ) -> dict[JobIdType, set[JobIdType]]:
        """Builds an adjacency list of critical jobs based on their operations.

        Args:
            critical_blocks (Sequence[Sequence[OperationType]]): A list of critical blocks,
                where each block is a sequence of operations.
                Each operation is a tuple (job_id, stage_id, machine_id).
            all_jobs (Sequence[JobIdType]): A list of all job IDs.

        Returns:
            dict[JobIdType, set[JobIdType]]: job ID -> a set of its adjacent critical jobs
        """
        adjacency: dict[JobIdType, set[JobIdType]] = {
            job_id: set() for job_id in all_jobs
        }
        for block in critical_blocks:
            for prev_op, next_op in zip(block, block[1:]):
                prev_job = prev_op[0]
                next_job = next_op[0]
                if prev_job == next_job:
                    continue
                adjacency.setdefault(prev_job, set()).add(next_job)
                adjacency.setdefault(next_job, set()).add(prev_job)
        return adjacency

    def _select_critical_jobs(
        self,
        schedule: HybridFlowshopLiteSchedule,
        job_count: int,
        job_selection_policy: str,
    ) -> list[JobIdType]:
        """
        Agent-facing contract for `critical_job_ns` job selection.

        Purpose:
            Choose up to `job_count` jobs that will be unfixed by `critical_job_ns`.
            The returned list is not just a set; its order carries policy-specific
            meaning. In `critical_adjacency`, `selected_jobs[0]` is always the seed job.

        Important side effect:
            This method mutates `schedule` indirectly because it calls
            `_find_critical_blocks_for_critical_job_ns(schedule)`, and that helper
            executes `schedule.make_semi_active(...)` in place before computing
            critical blocks. Any caller that needs the pre-normalized schedule must
            clone it before calling here.

        Candidate generation:
            1. Build critical blocks from the semi-active schedule with
               `include_singletons=True`.
            2. Convert those blocks into per-job weights by counting how many critical
               operations belong to each job. A job that appears more often on
               critical blocks receives a larger sampling weight.
            3. The normal candidate pool is `sorted(job_weights)`, i.e. only jobs that
               appear at least once in a critical block.

        Fallback behavior:
            - If no critical blocks are found after semi-active normalization, this is
              treated as an anomalous state. The helper logs a warning and returns an
              empty critical-block list. This method then falls back to uniform random
              sampling over all jobs in the schedule.
            - If adjacency expansion cannot reach `target_count`, the method logs a
              warning and backfills from the remaining candidate jobs using weighted
              random sampling without replacement.

        Policy semantics:
            - `weighted_random`
              Sample unique jobs without replacement from the critical-job candidate
              pool using the occurrence counts as weights.
            - `weighted_top`
              Select the heaviest critical jobs deterministically. This is intended
              for short repair calls where reproducibility matters more than
              diversification.
            - `critical_adjacency`
              First pick one weighted-random seed job. Then perform frontier
              expansion over the critical-job adjacency graph built from consecutive
              jobs inside each critical block. The queue is BFS-like: pop one
              selected job, shuffle its unseen neighbors, append them in that order,
              and continue until the target is reached or the frontier is exhausted.
            - `critical_adjacency_top`
              Use the heaviest critical job as the seed, then expand adjacency with
              neighbors ordered by critical weight. Backfill also uses the same
              deterministic weight order.
            - `largest_block_midpoint`
              Select a contiguous band around the middle of the longest critical
              block. This targets bottleneck-block order disruptions without letting
              early multi-stage chain jobs dominate a short repair call.

        Ordering guarantees:
            - The return value preserves selection order.
            - For `critical_adjacency`, element 0 is the seed job and later elements
              reflect adjacency expansion order, followed by any weighted-random
              backfill if expansion was insufficient.
            - For `weighted_random`, order is the draw order from the weighted
              sampling routine.

        Size semantics:
            - `job_count` is normalized and must be strictly positive.
            - The returned length is `min(normalized_job_count, available_candidates)`.
              In the no-critical-block fallback, `available_candidates` means all
              jobs in the schedule.
        """
        normalized_job_count = self._normalize_job_count(job_count)
        critical_blocks = self._find_critical_blocks_for_critical_job_ns(schedule)
        all_jobs = sorted(schedule.jobs)
        return self._select_critical_jobs_from_blocks(
            critical_blocks,
            all_jobs=all_jobs,
            normalized_job_count=normalized_job_count,
            job_selection_policy=job_selection_policy,
            log_prefix="Critical-job",
            fallback_log_message=(
                "No critical blocks found for critical-job selection; falling back to uniform random selection. "
                f"schedule makespan={schedule.makespan} jobs={len(schedule.jobs)} stages={len(schedule.stages)}"
            ),
        )

    def _select_critical_jobs_from_blocks(
        self,
        critical_blocks: Sequence[Sequence[OperationType]],
        *,
        all_jobs: Sequence[JobIdType],
        normalized_job_count: int,
        job_selection_policy: str,
        log_prefix: str,
        fallback_log_message: str,
    ) -> list[JobIdType]:
        candidate_pool = list(all_jobs)
        if not critical_blocks:
            logging.warning(fallback_log_message)
            return random.sample(
                candidate_pool,
                k=min(normalized_job_count, len(candidate_pool)),
            )

        job_weights = self._get_critical_job_weights(critical_blocks)
        candidate_jobs = sorted(job_weights)
        target_count = min(normalized_job_count, len(candidate_jobs))

        if job_selection_policy == "weighted_random":
            return self._weighted_sample_without_replacement(
                candidate_jobs,
                job_weights,
                target_count,
            )
        if job_selection_policy == "weighted_top":
            return self._rank_critical_jobs_by_weight(
                candidate_jobs,
                job_weights,
            )[:target_count]
        if job_selection_policy == "largest_block_midpoint":
            largest_block = max(
                critical_blocks,
                key=lambda block: (
                    len(self._unique_jobs_in_block(block)),
                    len(block),
                ),
            )
            block_jobs = [
                job_id
                for job_id in self._unique_jobs_in_block(largest_block)
                if job_id in job_weights
            ]
            selected_jobs = self._select_jobs_around_block_midpoint(
                block_jobs,
                target_count=target_count,
            )
            selected_job_set = set(selected_jobs)
            if len(selected_jobs) < target_count:
                selected_jobs.extend(
                    [
                        job_id
                        for job_id in self._rank_critical_jobs_by_weight(
                            candidate_jobs, job_weights
                        )
                        if job_id not in selected_job_set
                    ][: target_count - len(selected_jobs)]
                )
            logging.info(
                "%s largest-block midpoint selected: seed_job=%s target_count=%d block_jobs=%d candidates=%d",
                log_prefix,
                selected_jobs[0] if selected_jobs else None,
                target_count,
                len(block_jobs),
                len(candidate_jobs),
            )
            return selected_jobs
        if job_selection_policy not in {"critical_adjacency", "critical_adjacency_top"}:
            raise ValueError(
                "Unsupported job_selection_policy for critical-job selection: "
                f"{job_selection_policy}"
            )

        if job_selection_policy == "critical_adjacency_top":
            seed_job = self._rank_critical_jobs_by_weight(candidate_jobs, job_weights)[
                0
            ]
        else:
            seed_job = self._weighted_sample_without_replacement(
                candidate_jobs,
                job_weights,
                1,
            )[0]
        logging.info(
            "%s adjacency seed selected: seed_job=%s target_count=%d candidates=%d",
            log_prefix,
            seed_job,
            target_count,
            len(candidate_jobs),
        )
        adjacency = self._build_critical_job_adjacency(critical_blocks, candidate_jobs)
        selected_jobs: list[JobIdType] = [seed_job]
        selected_job_set = {seed_job}
        queue: deque[JobIdType] = deque([seed_job])

        while queue and len(selected_jobs) < target_count:
            current_job = queue.popleft()
            neighbors = sorted(adjacency.get(current_job, set()) - selected_job_set)
            if job_selection_policy == "critical_adjacency_top":
                neighbors = self._rank_critical_jobs_by_weight(neighbors, job_weights)
            else:
                random.shuffle(neighbors)
            for neighbor in neighbors:
                selected_jobs.append(neighbor)
                selected_job_set.add(neighbor)
                queue.append(neighbor)
                if len(selected_jobs) >= target_count:
                    break

        if len(selected_jobs) < target_count:
            remaining_jobs = [
                job_id for job_id in candidate_jobs if job_id not in selected_job_set
            ]
            if remaining_jobs:
                logging.warning(
                    "%s adjacency expansion exhausted before target; "
                    "backfilling weighted-random jobs. selected=%d target=%d candidates=%d",
                    log_prefix,
                    len(selected_jobs),
                    target_count,
                    len(candidate_jobs),
                )
                selected_jobs.extend(
                    (
                        self._rank_critical_jobs_by_weight(remaining_jobs, job_weights)
                        if job_selection_policy == "critical_adjacency_top"
                        else self._weighted_sample_without_replacement(
                            remaining_jobs,
                            job_weights,
                            target_count - len(selected_jobs),
                        )
                    )[: target_count - len(selected_jobs)]
                )

        return selected_jobs

    def _resolve_tail_stage_ids(
        self,
        *,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
    ) -> list[str]:
        stage_id_list = list(self.instance.stage_id_list)
        if not stage_id_list:
            raise ValueError("instance.stage_id_list cannot be empty.")
        if tail_stage_count is None:
            resolved_ratio = (
                0.3 if tail_stage_ratio is None else float(tail_stage_ratio)
            )
            if not math.isfinite(resolved_ratio) or resolved_ratio <= 0.0:
                raise ValueError(
                    "tail_stage_ratio must be positive when tail_stage_count is not provided."
                )
            tail_stage_count = max(1, math.ceil(len(stage_id_list) * resolved_ratio))
        if tail_stage_count <= 0:
            raise ValueError(
                f"tail_stage_count must be positive. Received {tail_stage_count}."
            )
        count = min(int(tail_stage_count), len(stage_id_list))
        return stage_id_list[-count:]

    def _select_critical_tail_jobs(
        self,
        schedule: HybridFlowshopLiteSchedule,
        job_count: int,
        *,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
        job_selection_policy: str = "critical_adjacency",
    ) -> tuple[list[JobIdType], list[str]]:
        normalized_job_count = self._normalize_job_count(job_count)
        tail_stage_ids = self._resolve_tail_stage_ids(
            tail_stage_count=tail_stage_count,
            tail_stage_ratio=tail_stage_ratio,
        )
        tail_stage_set = set(tail_stage_ids)
        critical_blocks = self._find_critical_blocks_for_critical_job_ns(schedule)
        tail_critical_blocks = [
            [op for op in block if op[1] in tail_stage_set] for block in critical_blocks
        ]
        tail_critical_blocks = [block for block in tail_critical_blocks if block]

        tail_jobs = sorted(
            {
                job_id
                for (job_id, stage_id, _mc_id) in schedule.get_jik_2_start_time_map()
                if stage_id in tail_stage_set
            }
        )
        if not tail_jobs:
            tail_jobs = sorted(schedule.jobs)

        selected_jobs = self._select_critical_jobs_from_blocks(
            tail_critical_blocks,
            all_jobs=tail_jobs,
            normalized_job_count=normalized_job_count,
            job_selection_policy=job_selection_policy,
            log_prefix="Critical-tail-job",
            fallback_log_message=(
                "No tail critical blocks found for critical-tail-job selection; "
                f"falling back to uniform random selection over tail stages {tail_stage_ids}."
            ),
        )
        return selected_jobs, tail_stage_ids

    def critical_tail_job_ns(
        self,
        job_count: int,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        job_selection_policy: str = "critical_adjacency",
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        resolved_computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        self._fix_profile_solve_reset(
            lambda: self.apply_critical_tail_job_operator(
                job_count=job_count,
                tail_stage_count=tail_stage_count,
                tail_stage_ratio=tail_stage_ratio,
                job_selection_policy=job_selection_policy,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            resolved_computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    @staticmethod
    def _resolve_adaptive_job_count(
        *,
        total_job_count: int,
        job_count_ratio: float,
        min_job_count: int,
        max_job_count: int,
    ) -> int:
        if total_job_count <= 0:
            raise ValueError("total_job_count must be positive.")
        if not math.isfinite(job_count_ratio) or job_count_ratio <= 0:
            raise ValueError("job_count_ratio must be positive.")
        if min_job_count <= 0:
            raise ValueError("min_job_count must be positive.")
        if max_job_count <= 0:
            raise ValueError("max_job_count must be positive.")
        if min_job_count > max_job_count:
            raise ValueError(
                "min_job_count must be less than or equal to max_job_count."
            )

        ratio_count = int(math.ceil(float(total_job_count) * float(job_count_ratio)))
        return min(
            int(total_job_count),
            max(int(min_job_count), min(int(max_job_count), ratio_count)),
        )

    def adaptive_critical_tail_job_ns(
        self,
        solver_thread_cnt: int,
        job_count_ratio: float = 0.08,
        min_job_count: int = 6,
        max_job_count: int = 24,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        job_selection_policy: str = "critical_adjacency",
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Run critical-tail-job CP with an instance-size-aware job count.

        The selected jobs are chosen from tail-stage critical blocks, then all
        operations of those jobs are unfixed. Keeping the job count proportional
        to n makes this a small cross-stage neighborhood on large instances
        without over-shrinking it on small ones.
        """
        resolved_job_count = self._resolve_adaptive_job_count(
            total_job_count=int(self.instance.job_count),
            job_count_ratio=job_count_ratio,
            min_job_count=min_job_count,
            max_job_count=max_job_count,
        )
        logging.info(
            "[Adaptive Critical Tail Job NS] job_count=%d resolved from n=%d "
            "ratio=%.4f min=%d max=%d.",
            resolved_job_count,
            int(self.instance.job_count),
            float(job_count_ratio),
            int(min_job_count),
            int(max_job_count),
        )
        self.critical_tail_job_ns(
            job_count=resolved_job_count,
            solver_thread_cnt=solver_thread_cnt,
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
            tail_stage_count=tail_stage_count,
            tail_stage_ratio=tail_stage_ratio,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            job_selection_policy=job_selection_policy,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def workload_guarded_adaptive_critical_tail_job_ns(
        self,
        solver_thread_cnt: int,
        job_count_ratio: float = 0.08,
        min_job_count: int = 6,
        max_job_count: int = 24,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        job_selection_policy: str = "critical_adjacency",
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
    ) -> None:
        """Run adaptive critical-tail repair only for an instance-size band."""
        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        if not self._bound_gap_guard_allows(
            context_label="Adaptive Critical Tail Job NS",
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return
        if min_workload_size is not None and workload_size < int(min_workload_size):
            logging.info(
                "[Adaptive Critical Tail Job NS] Skipping: workload_size=%d is below min_workload_size=%d.",
                workload_size,
                int(min_workload_size),
            )
            return
        if max_workload_size is not None and workload_size > int(max_workload_size):
            logging.info(
                "[Adaptive Critical Tail Job NS] Skipping: workload_size=%d is above max_workload_size=%d.",
                workload_size,
                int(max_workload_size),
            )
            return
        if min_instance_job_count is not None and instance_job_count < int(
            min_instance_job_count
        ):
            logging.info(
                "[Adaptive Critical Tail Job NS] Skipping: job_count=%d is below min_instance_job_count=%d.",
                instance_job_count,
                int(min_instance_job_count),
            )
            return
        if max_instance_job_count is not None and instance_job_count > int(
            max_instance_job_count
        ):
            logging.info(
                "[Adaptive Critical Tail Job NS] Skipping: job_count=%d is above max_instance_job_count=%d.",
                instance_job_count,
                int(max_instance_job_count),
            )
            return
        if min_instance_stage_count is not None and instance_stage_count < int(
            min_instance_stage_count
        ):
            logging.info(
                "[Adaptive Critical Tail Job NS] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                instance_stage_count,
                int(min_instance_stage_count),
            )
            return
        if max_instance_stage_count is not None and instance_stage_count > int(
            max_instance_stage_count
        ):
            logging.info(
                "[Adaptive Critical Tail Job NS] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                instance_stage_count,
                int(max_instance_stage_count),
            )
            return
        logging.info(
            "[Adaptive Critical Tail Job NS] Running with job_count=%d stage_count=%d workload_size=%d "
            "inside guards workload=[%s, %s] job_count=[%s, %s] stage_count=[%s, %s].",
            instance_job_count,
            instance_stage_count,
            workload_size,
            min_workload_size,
            max_workload_size,
            min_instance_job_count,
            max_instance_job_count,
            min_instance_stage_count,
            max_instance_stage_count,
        )
        self.adaptive_critical_tail_job_ns(
            solver_thread_cnt=solver_thread_cnt,
            job_count_ratio=job_count_ratio,
            min_job_count=min_job_count,
            max_job_count=max_job_count,
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
            tail_stage_count=tail_stage_count,
            tail_stage_ratio=tail_stage_ratio,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            job_selection_policy=job_selection_policy,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_critical_tail_job_operator(
        self,
        job_count: int,
        *,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
        job_selection_policy: str = "critical_adjacency",
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> list[JobIdType]:
        normalized_job_count = self._normalize_job_count(job_count)
        logging.info(
            "Applying critical-tail-job operator with job_count=%d policy=%s tail_stage_count=%s tail_stage_ratio=%s",
            normalized_job_count,
            job_selection_policy,
            tail_stage_count,
            tail_stage_ratio,
        )
        if not self.solution_manager.has_incumbent():
            raise ValueError(
                "No incumbent solution available for critical-tail-job operator."
            )
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a HybridFlowshopLiteSchedule"
                f"; is of type {type(incumbent_solution)}."
            )

        selected_jobs, tail_stage_ids = self._select_critical_tail_jobs(
            incumbent_solution,
            normalized_job_count,
            tail_stage_count=tail_stage_count,
            tail_stage_ratio=tail_stage_ratio,
            job_selection_policy=job_selection_policy,
        )
        all_ops = list(incumbent_solution.get_jik_2_start_time_map().keys())
        selected_job_set = set(selected_jobs)
        selected_ops = {op for op in all_ops if op[0] in selected_job_set}
        logging.info(
            "Critical-tail-job operator selected %d jobs and %d ops over tail stages %s (target=%d): %s",
            len(selected_job_set),
            len(selected_ops),
            tail_stage_ids,
            normalized_job_count,
            selected_jobs,
        )

        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )
        return selected_jobs

    def _resolve_target_stage_ids_for_critical_machine_ls(
        self,
        *,
        target_stage_mode: str,
        bottleneck_band_radius: int,
        tail_stage_count: int | None,
        tail_stage_ratio: float | None,
    ) -> list[str] | None:
        if target_stage_mode == "all":
            return None
        if target_stage_mode == "bottleneck":
            return [self._resolve_bottleneck_stage_id_for_targeted_ns()]
        if target_stage_mode == "bottleneck_band":
            bottleneck_stage_id = self._resolve_bottleneck_stage_id_for_targeted_ns()
            return self._resolve_stage_band_ids(
                bottleneck_stage_id,
                radius=bottleneck_band_radius,
            )
        if target_stage_mode == "tail":
            return self._resolve_tail_stage_ids(
                tail_stage_count=tail_stage_count,
                tail_stage_ratio=tail_stage_ratio,
            )
        raise ValueError(
            "target_stage_mode must be one of "
            "{'all', 'bottleneck', 'bottleneck_band', 'tail'}."
        )

    def critical_cross_machine_insertion_ls(
        self,
        max_passes: int = 2,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        target_stage_mode: str = "bottleneck_band",
        bottleneck_band_radius: int = 1,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
        max_machine_candidates_per_op: int | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Apply a critical-path cross-machine local search to the incumbent."""
        sub_timer = ElapsedTimer()
        if max_passes <= 0:
            raise ValueError("max_passes must be positive.")
        resolved_computational_time = self.get_remaining_time_limit(
            self._resolve_tl_nc_computational_time(
                computational_time=computational_time,
                tl_nc_multiplier=tl_nc_multiplier,
            )
        )

        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "No incumbent HybridFlowshopLiteSchedule is available for "
                "critical_cross_machine_insertion_ls."
            )

        target_stage_ids = self._resolve_target_stage_ids_for_critical_machine_ls(
            target_stage_mode=target_stage_mode,
            bottleneck_band_radius=bottleneck_band_radius,
            tail_stage_count=tail_stage_count,
            tail_stage_ratio=tail_stage_ratio,
        )
        logging.info(
            "Running critical_cross_machine_insertion_ls with max_passes=%d computational_time=%s target_stage_mode=%s target_stage_ids=%s",
            max_passes,
            f"{resolved_computational_time:.3f}",
            target_stage_mode,
            target_stage_ids,
        )

        improved_schedule = improve_schedule_by_critical_cross_machine_insertions(
            incumbent_solution,
            self.stage_2_job_2_p_dict,
            target_stage_ids=target_stage_ids,
            max_passes=max_passes,
            max_machine_candidates_per_op=max_machine_candidates_per_op,
            computational_time=resolved_computational_time,
        )
        if error_if_infeasible:
            self.check_feasibility(improved_schedule.get_jik_2_start_time_map())

        obj_value = float(improved_schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
            subroutine_name="critical_cross_machine_insertion_ls",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, improved_schedule)

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def critical_schedule_repair_ls(
        self,
        max_rounds: int = 2,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        target_stage_mode: str = "bottleneck_band",
        bottleneck_band_radius: int = 1,
        tail_stage_count: int | None = None,
        tail_stage_ratio: float | None = None,
        stage_insertion_passes: int = 2,
        stage_insertion_max_shift: int = 4,
        machine_insertion_passes: int = 2,
        max_machine_candidates_per_op: int | None = None,
        adjacent_swap_passes: int = 2,
        max_adjacent_pairs_per_pass: int | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Apply a portfolio of lightweight critical-path repairs to the incumbent."""
        sub_timer = ElapsedTimer()
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive.")

        resolved_computational_time = self.get_remaining_time_limit(
            self._resolve_tl_nc_computational_time(
                computational_time=computational_time,
                tl_nc_multiplier=tl_nc_multiplier,
            )
        )
        deadline = (
            time.perf_counter() + float(resolved_computational_time)
            if resolved_computational_time is not None
            else None
        )

        def time_is_up() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        def remaining_time() -> float | None:
            if deadline is None:
                return None
            return max(0.0, deadline - time.perf_counter())

        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "No incumbent HybridFlowshopLiteSchedule is available for "
                "critical_schedule_repair_ls."
            )

        target_stage_ids = self._resolve_target_stage_ids_for_critical_machine_ls(
            target_stage_mode=target_stage_mode,
            bottleneck_band_radius=bottleneck_band_radius,
            tail_stage_count=tail_stage_count,
            tail_stage_ratio=tail_stage_ratio,
        )
        logging.info(
            "Running critical_schedule_repair_ls with max_rounds=%d computational_time=%s "
            "target_stage_mode=%s target_stage_ids=%s",
            max_rounds,
            f"{resolved_computational_time:.3f}"
            if resolved_computational_time is not None
            else None,
            target_stage_mode,
            target_stage_ids,
        )

        best_schedule = incumbent_solution
        for _round_idx in range(max_rounds):
            if time_is_up():
                break

            candidate_pool = [best_schedule]
            if stage_insertion_passes > 0:
                try:
                    inserted = improve_schedule_by_critical_stage_sequence_insertions(
                        self.create_empty_schedule_from_ins,
                        best_schedule,
                        self.stage_2_job_2_p_dict,
                        target_stage_ids=target_stage_ids,
                        max_passes=stage_insertion_passes,
                        max_shift=max(1, stage_insertion_max_shift),
                    )
                    candidate_pool.append(inserted)
                except Exception:
                    logging.exception(
                        "Stage-sequence insertion failed in critical_schedule_repair_ls."
                    )

            if not time_is_up() and machine_insertion_passes > 0:
                try:
                    reassigned = improve_schedule_by_critical_cross_machine_insertions(
                        best_schedule,
                        self.stage_2_job_2_p_dict,
                        target_stage_ids=target_stage_ids,
                        max_passes=machine_insertion_passes,
                        max_machine_candidates_per_op=max_machine_candidates_per_op,
                        computational_time=remaining_time(),
                    )
                    candidate_pool.append(reassigned)
                except Exception:
                    logging.exception(
                        "Cross-machine insertion failed in critical_schedule_repair_ls."
                    )

            if not time_is_up() and adjacent_swap_passes > 0:
                for base_schedule in tuple(candidate_pool):
                    try:
                        swapped = improve_schedule_by_critical_adjacent_swaps(
                            base_schedule,
                            self.stage_2_job_2_p_dict,
                            max_passes=adjacent_swap_passes,
                            max_adjacent_pairs_per_pass=max_adjacent_pairs_per_pass,
                        )
                        candidate_pool.append(swapped)
                    except Exception:
                        logging.exception(
                            "Adjacent swap failed in critical_schedule_repair_ls."
                        )

            round_best = min(candidate_pool, key=lambda sch: sch.makespan)
            if round_best.makespan >= best_schedule.makespan:
                break
            best_schedule = round_best

        best_schedule.make_semi_active(self.stage_2_job_2_p_dict)
        if error_if_infeasible:
            self.check_feasibility(best_schedule.get_jik_2_start_time_map())

        obj_value = float(best_schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
            subroutine_name="critical_schedule_repair_ls",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, best_schedule)

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
        )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def apply_critical_job_operator(
        self,
        job_count: int,
        job_selection_policy: str = "critical_adjacency",
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> list[JobIdType]:
        normalized_job_count = self._normalize_job_count(job_count)
        logging.info(
            "Applying critical-job operator with job_count=%d policy=%s",
            normalized_job_count,
            job_selection_policy,
        )
        if not self.solution_manager.has_incumbent():
            raise ValueError(
                "No incumbent solution available for critical-job operator."
            )
        incumbent_solution = self.solution_manager.get_incumbent()
        if not isinstance(incumbent_solution, HybridFlowshopLiteSchedule):
            raise ValueError(
                "Incumbent solution is not a HybridFlowshopLiteSchedule"
                f"; is of type {type(incumbent_solution)}."
            )

        selected_jobs = self._select_critical_jobs(
            incumbent_solution, normalized_job_count, job_selection_policy
        )
        all_ops = list(incumbent_solution.get_jik_2_start_time_map().keys())
        selected_job_set = set(selected_jobs)
        selected_ops = {op for op in all_ops if op[0] in selected_job_set}
        logging.info(
            "Critical-job operator selected %d jobs and %d ops (target=%d): %s",
            len(selected_job_set),
            len(selected_ops),
            normalized_job_count,
            selected_jobs,
        )

        self._fix_operations_profile_except_selected(
            selected_ops,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )
        return selected_jobs

    def profile_fixed_ns(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        make_semi_active_after_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self._fix_operations_profile(
                self.solution_manager.get_incumbent(),
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            ),
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            use_lns_only=use_lns_only,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    # Subroutine: Johnson-based Heuristic for initialization

    @staticmethod
    def get_johnsons_rule_sequence(
        job_name_2_p1_map: dict[str, int], job_name_2_p2_map: dict[str, int]
    ) -> list[str]:
        """
        Apply Johnson's rule to determine the job sequence.
        This method takes two dictionaries representing aggregated processing times
        for two stages and returns a job sequence based on the Johnson's rule.

        Args:
            job_name_2_p1_map (dict[str, int]): job ID -> processing time for the 1st stage
            job_name_2_p2_map (dict[str, int]): job ID -> processing time for the 2nd stage

        Returns:
            list[str]: A list of job IDs ordered according to Johnson's rule.
        """
        jobs = list(job_name_2_p1_map.keys())

        l1: list[str] = []
        l2: list[str] = []

        for job in jobs:
            if job_name_2_p1_map[job] <= job_name_2_p2_map[job]:
                l1.append(job)
            else:
                l2.append(job)
            # logging.info(
            #     f"Job {job} with p1={job_name_2_p1_map[job]}, p2={job_name_2_p2_map[job]}"
            #     f"; Min={min(job_name_2_p1_map[job], job_name_2_p2_map[job])}"
            #     f"; p1<=p2: {job_name_2_p1_map[job] <= job_name_2_p2_map[job]}"
            # )

        # Sort by increasing order of p_1j, tie-breaking by job-ID (ascending)
        l1.sort(key=lambda j: (job_name_2_p1_map[j], j))

        # Sort by decreasing order of p_2j, tie-breaking by job-ID (descending)
        l2.sort(key=lambda j: (job_name_2_p2_map[j], j), reverse=True)

        # logging.info(f"Job sequence by Johnson's rule: {l1}+{l2}")

        sequence = l1 + l2
        return sequence

    def get_cds_sequence(self, k: int) -> list[str]:
        """
        Get Campbell-Dudek-Smith (CDS) sequence given $k$.

        The sequence is originally for the permultation flow shop problem
        to minimize the makespan.

        When k=1, result is the same as get_sequence1().

        Args:
            k (int): Index between 1 and m-1, where m is the number of stages.

        Returns:
            list[str]: A list of job IDs ordered according to the CDS rule.
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        m = self.instance.stage_count
        p1_stages = stages[:k]  # Sum over stages 0 ~ k-1 (1~k on paper)
        p2_stages = stages[m - k :]  # Sum over stages m-k ~ m-1 (m-k+1 ~ m on paper)
        p1 = {j: sum(p_dict[j, i] for i in p1_stages) for j in jobs}
        p2 = {j: sum(p_dict[j, i] for i in p2_stages) for j in jobs}

        return self.get_johnsons_rule_sequence(p1, p2)

    def get_gupta_sequence(self) -> list[str]:
        """
        Return the job sequence according to Gupta's functional heuristic algorithm.

        - For each job, calculate:
            f(j) = A / min_{1 <= m <= M-1} (p_{j,m} + p_{j,m+1})
            where
                A = 1 if p_{j,M} <= p_{j,1}
                A = -1 otherwise
        - Sort jobs in ascending order of f(j).
        (Break ties by total processing time, then by job ID)

        Returns:
            list[str]: A list of job IDs in Gupta heuristic order.
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        m = len(stages)
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        gupta_score = {}
        total_p = {}

        for j in jobs:
            # min over all adjacent pairs (m=0 to m-2)
            min_sum = min(
                p_dict[j, stages[m1]] + p_dict[j, stages[m1 + 1]] for m1 in range(m - 1)
            )
            A = 1 if p_dict[j, stages[-1]] <= p_dict[j, stages[0]] else -1
            f_j = A / min_sum if min_sum != 0 else float("inf")  # avoid div by zero
            gupta_score[j] = f_j
            total_p[j] = sum(p_dict[j, s] for s in stages)

        # Sort by ascending order: (f_j, total processing time, job id)
        sorted_jobs = sorted(jobs, key=lambda j: (gupta_score[j], total_p[j], j))
        return sorted_jobs

    def get_palmer_sequence(self) -> list[str]:
        """
        Return the job sequence according to Palmer's slope index heuristic.

        Returns:
            list[str]: Job ID list in Palmer slope order (descending s_i).
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        m = len(stages)
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        palmer_score = {}
        for j in jobs:
            s = sum(
                (m - 2 * (stage_idx + 1) + 1) * p_dict[j, stages[stage_idx]]
                for stage_idx in range(m)
            )
            palmer_score[j] = s

        # Sort by ascending order: (Palmer score, job id)
        sorted_jobs = sorted(jobs, key=lambda j: (palmer_score[j], j))
        return sorted_jobs

    def get_shdlb_for_stage(self, i: str) -> int:
        """
        Compute the stage-specific lower bound for the Hybrid Flow Shop instance
        using the Santos et al. (1995) method.

        Args:
            i (str): The stage index.

        Returns:
            int: The computed lower bound for the stage.
        """
        return santos_stage_lower_bound(self.instance, i)

    def apply_shdlb(self) -> None:
        """
        Compute the global lower bound for the Hybrid Flow Shop instance using the method
        described by Santos et al. (1995) and update the global lower bound.
        """
        sub_timer = ElapsedTimer()
        stages: list[str] = self.instance.stage_id_list

        lb0 = simple_job_lower_bound(self.instance)
        stage_bounds = [self.get_shdlb_for_stage(stage) for stage in stages]
        obj_bound = santos_lower_bound(self.instance)

        logging.info(
            f"[SHD LB] LB(0) = {lb0}, LB(j) = {stage_bounds}, LB_MAX = {obj_bound}"
        )

        # Log
        if self.solution_manager.current_obj_bound_is_worse_than(obj_bound):
            log_time = self.timer.elapsed_sec
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=False)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_bound_is_valid=True
            )

        # Create report and register
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=None,
            obj_bound=obj_bound,
            is_init=True,
            subroutine_name="apply_shdlb",
        )
        self.solution_manager.register(report, None)

    def apply_lb4(self) -> None:
        """
        Compute Chen et al. (2023) LB4 and update the global lower bound.

        LB4 is the modified bin-packing-based lower bound from
        "An Evaluation of Mathematical Programming and Lower-Bound Methods for
        Hybrid Flow Shop Problems With a Makespan Criterion".
        """
        sub_timer = ElapsedTimer()
        stage_bound_map = chen_lb4_stage_lower_bounds(self.instance)
        obj_bound = chen_lb4_lower_bound(self.instance)

        logging.info(
            "[LB4] LB4(k) = %s, LB4_MAX = %s",
            [stage_bound_map[stage] for stage in self.instance.stage_id_list],
            obj_bound,
        )

        # Log
        if self.solution_manager.current_obj_bound_is_worse_than(obj_bound):
            log_time = self.timer.elapsed_sec
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=False)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_bound_is_valid=True
            )

        # Create report and register
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=None,
            obj_bound=obj_bound,
            is_init=True,
            subroutine_name="apply_lb4",
        )
        self.solution_manager.register(report, None)

    def _apply_retained_stage_cp_lb_hints(
        self,
        *,
        build,
        incumbent_schedule: HybridFlowshopLiteSchedule | None,
    ) -> None:
        if incumbent_schedule is None:
            return

        retained_stage_set = set(build.retained_stage_ids)
        start_time_map = {
            key: value
            for key, value in incumbent_schedule.get_jik_2_start_time_map().items()
            if key[1] in retained_stage_set
        }
        end_time_map = {
            key: value
            for key, value in incumbent_schedule.get_jik_2_end_time_map().items()
            if key[1] in retained_stage_set
        }
        if not start_time_map or not end_time_map:
            logging.info(
                "[CP LB] Incumbent exists (makespan=%s), but no retained-stage "
                "start/end hints matched retained stages %s.",
                incumbent_schedule.makespan,
                build.retained_stage_ids,
            )
            return

        BaseModelBuilder.apply_start_hints_from_start_time_map(
            build.model,
            build.params,
            build.variables,
            start_time_map,
            ignore_integrity_check=True,
        )
        BaseModelBuilder.apply_end_hints_from_end_time_map(
            build.model,
            build.params,
            build.variables,
            end_time_map,
            ignore_integrity_check=True,
        )
        build.model.add_hint(
            build.variables.makespan,
            int(math.ceil(float(incumbent_schedule.makespan))),
        )
        logging.info(
            "[CP LB] Applied incumbent retained-stage hints from makespan=%s: "
            "retained_stages=%s start_hints=%d end_hints=%d makespan_hint=%d.",
            incumbent_schedule.makespan,
            build.retained_stage_ids,
            len(start_time_map),
            len(end_time_map),
            int(math.ceil(float(incumbent_schedule.makespan))),
        )

    @staticmethod
    def _retained_stage_cp_covers_all_stages(build) -> bool:
        return list(build.retained_stage_ids) == list(build.params.i_list)

    def _build_full_schedule_from_retained_stage_rows(
        self,
        retained_solution_rows: Sequence[Mapping[str, Any]],
    ) -> HybridFlowshopLiteSchedule:
        """Convert all-stage retained-CP intervals into a concrete machine schedule."""
        jobs = [str(job_id) for job_id in self.instance.job_id_list]
        stages = [str(stage_id) for stage_id in self.instance.stage_id_list]
        machines_per_stage = {
            str(stage_id): [
                str(mc_id) for mc_id in self.instance.stage_2_machines_map[stage_id]
            ]
            for stage_id in self.instance.stage_id_list
        }
        job_set = set(jobs)
        stage_set = set(stages)
        rows_by_stage: dict[str, list[Mapping[str, Any]]] = {
            stage_id: [] for stage_id in stages
        }

        for row in retained_solution_rows:
            stage_id = str(row["stage_id"])
            job_id = str(row["job_id"])
            if stage_id not in stage_set:
                raise ValueError(f"Retained CP row has unknown stage_id={stage_id!r}.")
            if job_id not in job_set:
                raise ValueError(f"Retained CP row has unknown job_id={job_id!r}.")
            rows_by_stage[stage_id].append(row)

        schedule = HybridFlowshopLiteSchedule(
            jobs=jobs,
            stages=stages,
            machines_per_stage=machines_per_stage,
        )
        for stage_id in stages:
            stage_rows = rows_by_stage[stage_id]
            stage_jobs = [str(row["job_id"]) for row in stage_rows]
            if set(stage_jobs) != job_set or len(stage_jobs) != len(job_set):
                raise ValueError(
                    "All-stage retained CP solution must contain exactly one row "
                    f"per job for stage {stage_id!r}."
                )
            machine_ids = machines_per_stage[stage_id]
            machine_available_t = {mc_id: 0 for mc_id in machine_ids}
            for row in sorted(
                stage_rows,
                key=lambda r: (int(r["start"]), int(r["end"]), str(r["job_id"])),
            ):
                job_id = str(row["job_id"])
                start = int(row["start"])
                end = int(row["end"])
                eligible_machines = [
                    mc_id
                    for mc_id in machine_ids
                    if machine_available_t[mc_id] <= start
                ]
                if not eligible_machines:
                    raise ValueError(
                        "Cannot assign retained CP interval to a concrete machine: "
                        f"stage={stage_id!r}, job={job_id!r}, interval=[{start}, {end})."
                    )
                selected_mc_id = min(
                    eligible_machines,
                    key=lambda mc_id: (
                        machine_available_t[mc_id],
                        machine_ids.index(mc_id),
                    ),
                )
                schedule.add_ops_times_2_mc(
                    stage_id=stage_id,
                    mc_id=selected_mc_id,
                    job_id=job_id,
                    start_time=start,
                    end_time=end,
                )
                machine_available_t[selected_mc_id] = end

        validate_schedule(schedule, self.stage_2_job_2_p_dict)
        return schedule

    def apply_retained_stage_cp_lb(
        self,
        threads: int = 24,
        tl_nc_multiplier: float | None = None,
        time_limit_sec: float | None = None,
        time_limit_n_by_c_multiplier: float | None = None,
        retained_stage_mode: str = "first_bottleneck_last",
        bottleneck_stage_id: str | None = None,
        extra_bottleneck_count: int = 1,
        bottleneck_band_radius: int = 1,
        middle_band_radius: int = 1,
        quantile_count: int | None = None,
        retained_stage_ratios: Sequence[float] | None = None,
        save_cp_lb_artifacts: bool = True,
        snapshot_solution_limit: int = 0,
        snapshot_log_progress: bool = True,
        log_cp_lb_ub_progress: bool = True,
    ) -> dict[str, Any] | None:
        """
        Compute a retained-stage CP-SAT lower bound using only a subset of stages exactly.

        Supported retained stage modes:
        - ``first_last``: retain the first and last stages exactly.
        - ``first_bottleneck_last``: retain the first, bottleneck, and last stages.
        - ``first_bottleneck_band_last``: retain the first, last, and the stages
          within ``bottleneck_band_radius`` of a representative bottleneck stage.
        - ``first_topk_bottlenecks_last``: retain the first, top-k bottlenecks, and last stages.
        - ``first_quantiles_topk_bottlenecks_last``: retain the first, quantile
          cut stages, top-k bottleneck stages, and last stages.
        - ``first_bottleneck_midpoints_last``: retain the first, an internal
          bottleneck, the midpoint stages between the first/bottleneck and
          bottleneck/last anchors, and the last stage.
        - ``first_processing_jump_band_last``: retain the first, last, and the
          stages within ``bottleneck_band_radius`` of the internal stage with
          the largest adjacent total-processing jump.
        - ``first_middle_last``: retain the first, middle, and last stages.
        - ``first_middle_band_last``: retain the first, last, and the stages
          within ``middle_band_radius`` of the middle stage.
        - ``middle_band``: retain only the stages within ``middle_band_radius``
          of the middle stage.
        - ``first_n_quantiles_last``: retain the first, n-quantile cut stages, and last stages.
        - ``first_ratio_points_last``: retain the first, user-specified ratio stages, and last stages.
        """
        start_t = self.timer.elapsed_sec
        call_context = self._get_call_context_of_current_method()
        sub_timer = ElapsedTimer()
        self.last_retained_cp_lb_apply_elapsed_sec = None
        self.last_retained_cp_lb_result = None
        self.last_retained_cp_lb_trace_rows = None
        self.last_retained_cp_lb_retained_solution_rows = None
        self.last_retained_cp_lb_solution_snapshots = None

        input_ub = self.solution_manager.best_obj_value
        if input_ub is None:
            logging.warning("[CP LB] No upper bound available, skipping")
            return None

        input_lb = self.solution_manager.best_obj_bound
        instance = self.instance
        incumbent_schedule = self.solution_manager.get_incumbent()

        if tl_nc_multiplier is not None:
            _time_limit_sec = (
                float(tl_nc_multiplier)
                * float(instance.job_count)
                * float(instance.stage_count)
            )
        elif time_limit_sec is None and time_limit_n_by_c_multiplier is not None:
            _time_limit_sec = (
                float(instance.job_count)
                * float(instance.stage_count)
                * float(time_limit_n_by_c_multiplier)
            )
        else:
            _time_limit_sec = time_limit_sec
        _time_limit_sec = self.get_remaining_time_limit(_time_limit_sec)

        logging.info(
            "[CP LB] Starting retained-stage CP-SAT at %.1f with mode=%s bottleneck_stage_id=%s "
            "extra_bottleneck_count=%d bottleneck_band_radius=%d middle_band_radius=%d quantile_count=%s retained_stage_ratios=%s "
            "threads=%d time_limit_sec=%s",
            start_t,
            retained_stage_mode,
            bottleneck_stage_id,
            extra_bottleneck_count,
            bottleneck_band_radius,
            middle_band_radius,
            quantile_count,
            list(retained_stage_ratios or ()),
            threads,
            f"{_time_limit_sec:.1f}" if _time_limit_sec is not None else "None",
        )

        model_build_wall_start = time.perf_counter()
        build = build_retained_stage_cp_model(
            instance,
            input_ub=int(math.ceil(float(input_ub))),
            retained_stage_mode=retained_stage_mode,
            bottleneck_stage_id=bottleneck_stage_id,
            extra_bottleneck_count=extra_bottleneck_count,
            bottleneck_band_radius=bottleneck_band_radius,
            middle_band_radius=middle_band_radius,
            quantile_count=quantile_count,
            retained_stage_ratios=retained_stage_ratios,
        )
        model_build_wall_sec = time.perf_counter() - model_build_wall_start
        self._apply_retained_stage_cp_lb_hints(
            build=build,
            incumbent_schedule=incumbent_schedule,
        )

        snapshot_recorder = None
        obj_value_log_level = logging.INFO if log_cp_lb_ub_progress else None
        if snapshot_solution_limit > 0:
            snapshot_recorder = _RetainedStageSnapshotRecorder(
                build=build,
                snapshot_limit=snapshot_solution_limit,
                e_timer=sub_timer,
                log_level_on_record=(
                    logging.INFO
                    if snapshot_log_progress or log_cp_lb_ub_progress
                    else None
                ),
            )

        solver_report = self.solve_cp_model_2(
            build.model,
            _time_limit_sec,
            threads,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            e_timer=sub_timer,
            log_search_progress=False,
            log_level_obj_value=obj_value_log_level,
            solution_callback=snapshot_recorder,
        )
        self.last_retained_cp_lb_apply_elapsed_sec = sub_timer.elapsed_sec

        objective_ub = None
        if solver_report.status.is_feasible:
            objective_ub = sanitize_optional_float(
                getattr(self.solver, "objective_value", None)
            )
            if objective_ub is None:
                objective_ub = sanitize_optional_float(solver_report.obj_value)

        objective_lb = sanitize_optional_float(
            getattr(self.solver, "best_objective_bound", None)
        )
        if objective_lb is None and solver_report.status.is_feasible:
            objective_lb = sanitize_optional_float(solver_report.obj_bound)
        if objective_lb is None and solver_report.status == CpsatStatus.OPTIMAL:
            objective_lb = objective_ub

        solver_runtime_sec = sanitize_optional_float(
            getattr(self.solver, "wall_time", None)
        )
        if solver_runtime_sec is None:
            solver_runtime_sec = sub_timer.elapsed_sec

        trace_rows = build_trace_rows(
            solver_report.obj_value_records,
            solver_report.obj_bound_records,
            final_runtime_sec=solver_runtime_sec,
            final_objective_ub=objective_ub,
            final_objective_lb=objective_lb,
        )
        retained_solution_rows: list[dict[str, Any]] = []
        if solver_report.status.is_feasible:
            retained_solution_rows = extract_retained_stage_solution_rows(
                self.solver,
                build,
            )
        full_retained_schedule = None
        if retained_solution_rows and self._retained_stage_cp_covers_all_stages(build):
            full_retained_schedule = self._build_full_schedule_from_retained_stage_rows(
                retained_solution_rows
            )
            logging.info(
                "[CP LB] Retained stages cover the full instance; recovered a full "
                "incumbent candidate with makespan=%d from retained CP.",
                full_retained_schedule.makespan,
            )
        snapshot_rows = []
        if snapshot_recorder is not None:
            snapshot_rows = list(snapshot_recorder.snapshots)
            logging.info(
                "[CP LB] Captured %d retained-stage CP incumbent snapshots.",
                len(snapshot_rows),
            )

        result = build_retained_stage_cp_result(
            ins_name=instance.name,
            input_lb=input_lb,
            input_ub=float(input_ub),
            retained_stage_mode=retained_stage_mode,
            retained_stage_ids=build.retained_stage_ids,
            bottleneck_stage_id=build.bottleneck_stage_id,
            selected_bottleneck_stage_ids=build.selected_bottleneck_stage_ids,
            bottleneck_band_radius=build.bottleneck_band_radius,
            middle_band_radius=build.middle_band_radius,
            retained_stage_ratios=build.retained_stage_ratios,
            quantile_count=build.quantile_count,
            job_count=instance.job_count,
            stage_count=instance.stage_count,
            machine_count_per_stage=instance.machine_count_per_stage,
            status=solver_report.status,
            objective_ub=objective_ub,
            objective_lb=objective_lb,
            time_limit_sec_used=_time_limit_sec,
            solver_runtime_sec=solver_runtime_sec,
            wall_runtime_sec=sub_timer.elapsed_sec,
            model_build_wall_sec=model_build_wall_sec,
        )
        self.last_retained_cp_lb_result = result
        self.last_retained_cp_lb_trace_rows = trace_rows
        self.last_retained_cp_lb_retained_solution_rows = retained_solution_rows
        self.last_retained_cp_lb_solution_snapshots = snapshot_rows

        cp_lb_obj_value_records = _extract_retained_cp_trace_records(
            trace_rows,
            "objective_ub",
        )
        cp_lb_obj_bound_records = _extract_retained_cp_trace_records(
            trace_rows,
            "objective_lb",
        )

        current_bound = self.solution_manager.best_obj_bound
        improved_bound_logged = False
        for trace_runtime, trace_lb in cp_lb_obj_bound_records:
            if self.solution_manager._a_is_better_obj_bound(trace_lb, current_bound):
                global_time = start_t + trace_runtime
                self.add_obj_bound_log(global_time, trace_lb, is_maximize=False)
                current_bound = trace_lb
                improved_bound_logged = True

        improved_obj_logged = False
        if full_retained_schedule is not None:
            current_obj = input_ub
            for trace_runtime, trace_ub in cp_lb_obj_value_records:
                if self.solution_manager._a_is_better_obj_value(trace_ub, current_obj):
                    global_time = start_t + trace_runtime
                    self.add_obj_value_log(global_time, trace_ub, is_maximize=False)
                    current_obj = trace_ub
                    improved_obj_logged = True

        report = HfsCpsatSolverReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=(
                float(full_retained_schedule.makespan)
                if full_retained_schedule is not None
                else None
            ),
            obj_bound=result.certified_final_lb,
            is_init=False,
            subroutine_name="apply_retained_stage_cp_lb",
            call_context=call_context,
            progress_obj_value_records=(),
            progress_time_basis="local",
            status=solver_report.status,
            obj_value_records=cp_lb_obj_value_records,
            obj_bound_records=cp_lb_obj_bound_records,
        )
        retained_schedule_updated = self.solution_manager.register(
            report, full_retained_schedule
        )
        if full_retained_schedule is not None:
            logging.info(
                "[CP LB] Full retained-CP incumbent registration updated=%s obj=%d.",
                retained_schedule_updated,
                full_retained_schedule.makespan,
            )
        self._record_retained_cp_lb_summary(
            result,
            call_context=call_context,
            start_sec=start_t,
            apply_elapsed_sec=self.last_retained_cp_lb_apply_elapsed_sec,
        )

        if improved_bound_logged or improved_obj_logged:
            self.obj_store.add_last_timestamp_note(
                call_context,
                obj_value_is_valid=improved_obj_logged,
                obj_bound_is_valid=improved_bound_logged,
            )

        logging.info(
            "[CP LB] Finished retained-stage CP-SAT with status=%s objective_ub=%s objective_lb=%s",
            result.status_name,
            result.objective_ub,
            result.objective_lb,
        )

        if self._working_dir_path is not None and save_cp_lb_artifacts:
            write_retained_stage_cp_artifacts(
                self._working_dir_path / "cp_lb",
                result=result,
                build=build,
                trace_rows=trace_rows,
                retained_solution_rows=retained_solution_rows,
            )
            logging.info(
                "[CP LB] Persisted retained-stage CP artifacts to %s",
                self._working_dir_path / "cp_lb",
            )

        return {
            "result": result,
            "trace_rows": trace_rows,
            "retained_solution_rows": retained_solution_rows,
            "solution_snapshots": snapshot_rows,
            "retained_stage_ids": build.retained_stage_ids,
            "bottleneck_stage_id": build.bottleneck_stage_id,
            "selected_bottleneck_stage_ids": build.selected_bottleneck_stage_ids,
            "retained_stage_ratios": build.retained_stage_ratios,
            "quantile_count": build.quantile_count,
        }

    def _get_instance_workload_size(self) -> int:
        return int(self.instance.job_count) * int(self.instance.stage_count)

    def apply_workload_adaptive_retained_stage_cp_lb(
        self,
        threads: int = 24,
        tl_nc_multiplier: float | None = None,
        time_limit_sec: float | None = None,
        time_limit_n_by_c_multiplier: float | None = None,
        retained_stage_mode: str = "first_topk_bottlenecks_last",
        bottleneck_stage_id: str | None = None,
        small_extra_bottleneck_count: int = 3,
        large_extra_bottleneck_count: int = 4,
        large_workload_threshold: int = 1800,
        adaptive_basis: str = "workload",
        large_stage_count_threshold: int = 15,
        workload_tier_thresholds: Sequence[int] | None = None,
        workload_tier_extra_bottleneck_counts: Sequence[int] | None = None,
        bottleneck_band_radius: int = 1,
        middle_band_radius: int = 1,
        quantile_count: int | None = None,
        retained_stage_ratios: Sequence[float] | None = None,
        save_cp_lb_artifacts: bool = True,
        snapshot_solution_limit: int = 0,
        snapshot_log_progress: bool = True,
        log_cp_lb_ub_progress: bool = True,
    ) -> dict[str, Any] | None:
        workload_size = self._get_instance_workload_size()
        stage_count = int(self.instance.stage_count)
        if adaptive_basis == "workload":
            selected_extra_bottleneck_count = (
                int(large_extra_bottleneck_count)
                if workload_size >= int(large_workload_threshold)
                else int(small_extra_bottleneck_count)
            )
            logging.info(
                "[Adaptive CP LB] basis=workload workload_size=%d threshold=%d "
                "selected extra_bottleneck_count=%d (small=%d large=%d).",
                workload_size,
                int(large_workload_threshold),
                selected_extra_bottleneck_count,
                int(small_extra_bottleneck_count),
                int(large_extra_bottleneck_count),
            )
        elif adaptive_basis == "workload_tier":
            thresholds = [
                int(threshold)
                for threshold in (
                    workload_tier_thresholds or [int(large_workload_threshold)]
                )
            ]
            if any(
                thresholds[idx] >= thresholds[idx + 1]
                for idx in range(len(thresholds) - 1)
            ):
                raise ValueError(
                    "workload_tier_thresholds must be strictly increasing."
                )

            if workload_tier_extra_bottleneck_counts is None:
                tier_counts = [
                    int(small_extra_bottleneck_count),
                    int(large_extra_bottleneck_count),
                ]
            else:
                tier_counts = [
                    int(count)
                    for count in workload_tier_extra_bottleneck_counts
                ]

            if len(tier_counts) != len(thresholds) + 1:
                raise ValueError(
                    "workload_tier_extra_bottleneck_counts must have exactly "
                    "len(workload_tier_thresholds) + 1 entries."
                )

            tier_idx = 0
            for threshold in thresholds:
                if workload_size >= threshold:
                    tier_idx += 1
                else:
                    break
            selected_extra_bottleneck_count = tier_counts[tier_idx]
            logging.info(
                "[Adaptive CP LB] basis=workload_tier workload_size=%d "
                "thresholds=%s counts=%s tier_idx=%d selected "
                "extra_bottleneck_count=%d.",
                workload_size,
                thresholds,
                tier_counts,
                tier_idx,
                selected_extra_bottleneck_count,
            )
        elif adaptive_basis == "stage_count":
            selected_extra_bottleneck_count = (
                int(large_extra_bottleneck_count)
                if stage_count >= int(large_stage_count_threshold)
                else int(small_extra_bottleneck_count)
            )
            logging.info(
                "[Adaptive CP LB] basis=stage_count stage_count=%d threshold=%d "
                "workload_size=%d selected extra_bottleneck_count=%d "
                "(small=%d large=%d).",
                stage_count,
                int(large_stage_count_threshold),
                workload_size,
                selected_extra_bottleneck_count,
                int(small_extra_bottleneck_count),
                int(large_extra_bottleneck_count),
            )
        else:
            raise ValueError(
                "adaptive_basis must be one of "
                "{'workload', 'workload_tier', 'stage_count'}."
            )
        return self.apply_retained_stage_cp_lb(
            threads=threads,
            tl_nc_multiplier=tl_nc_multiplier,
            time_limit_sec=time_limit_sec,
            time_limit_n_by_c_multiplier=time_limit_n_by_c_multiplier,
            retained_stage_mode=retained_stage_mode,
            bottleneck_stage_id=bottleneck_stage_id,
            extra_bottleneck_count=selected_extra_bottleneck_count,
            bottleneck_band_radius=bottleneck_band_radius,
            middle_band_radius=middle_band_radius,
            quantile_count=quantile_count,
            retained_stage_ratios=retained_stage_ratios,
            save_cp_lb_artifacts=save_cp_lb_artifacts,
            snapshot_solution_limit=snapshot_solution_limit,
            snapshot_log_progress=snapshot_log_progress,
            log_cp_lb_ub_progress=log_cp_lb_ub_progress,
        )

    def workload_guarded_retained_stage_cp_lb_dispatch(
        self,
        threads: int = 24,
        tl_nc_multiplier: float | None = None,
        time_limit_sec: float | None = None,
        time_limit_n_by_c_multiplier: float | None = None,
        retained_stage_mode: str = "first_bottleneck_band_last",
        bottleneck_stage_id: str | None = None,
        extra_bottleneck_count: int = 1,
        bottleneck_band_radius: int = 2,
        middle_band_radius: int = 1,
        quantile_count: int | None = None,
        retained_stage_ratios: Sequence[float] | None = None,
        save_cp_lb_artifacts: bool = True,
        snapshot_solution_limit: int = 0,
        snapshot_log_progress: bool = True,
        log_cp_lb_ub_progress: bool = True,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
        min_post_dispatch_improvement: float | None = None,
        max_post_dispatch_improvement: float | None = None,
        min_post_dispatch_improvement_ratio: float | None = None,
        max_post_dispatch_improvement_ratio: float | None = None,
        run_if_post_dispatch_missing: bool = True,
        mixed_schedule_for_former_stages: bool = True,
        mixed_schedule_for_later_stages: bool = True,
        machine_then_job: bool = False,
        respect_anchor_stage_release_lb: bool = True,
        cp_local_repair_max_passes: int = 3,
        cp_local_repair_top_k: int = 1,
        include_consensus_rank: bool = True,
        include_tail_bottleneck_rank: bool = True,
        include_extended_rank_variants: bool = False,
        include_dynamic_priority: bool = False,
        include_piecewise_stage_priority: bool = False,
        prune_unproductive_dispatch_candidates: bool = True,
        randomized_mixed_rank_trials: int = 0,
        mixed_dispatch_methods: Sequence[str] | None = None,
        use_retained_cp_snapshot_portfolio: bool = False,
        retained_cp_snapshot_top_k: int = 0,
        retained_cp_dispatch_selection_strategy: str = "best_makespan",
        retained_cp_dispatch_makespan_slack: float = 0.0,
        save_cp_dispatch_artifacts: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> dict[str, Any] | None:
        """Run a guarded retained-stage CP-LB and restore it immediately by dispatch."""
        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        context_label = "Guarded Retained-Stage CP-LB Dispatch"

        if not self._bound_gap_guard_allows(
            context_label=context_label,
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return None
        if not self._post_dispatch_improvement_guard_allows(
            context_label=context_label,
            min_post_dispatch_improvement=min_post_dispatch_improvement,
            max_post_dispatch_improvement=max_post_dispatch_improvement,
            min_post_dispatch_improvement_ratio=min_post_dispatch_improvement_ratio,
            max_post_dispatch_improvement_ratio=max_post_dispatch_improvement_ratio,
            run_if_post_dispatch_missing=run_if_post_dispatch_missing,
        ):
            return None
        if min_workload_size is not None and workload_size < int(min_workload_size):
            logging.info(
                "[%s] Skipping: workload_size=%d is below min_workload_size=%d.",
                context_label,
                workload_size,
                int(min_workload_size),
            )
            return None
        if max_workload_size is not None and workload_size > int(max_workload_size):
            logging.info(
                "[%s] Skipping: workload_size=%d is above max_workload_size=%d.",
                context_label,
                workload_size,
                int(max_workload_size),
            )
            return None
        if min_instance_job_count is not None and instance_job_count < int(
            min_instance_job_count
        ):
            logging.info(
                "[%s] Skipping: job_count=%d is below min_instance_job_count=%d.",
                context_label,
                instance_job_count,
                int(min_instance_job_count),
            )
            return None
        if max_instance_job_count is not None and instance_job_count > int(
            max_instance_job_count
        ):
            logging.info(
                "[%s] Skipping: job_count=%d is above max_instance_job_count=%d.",
                context_label,
                instance_job_count,
                int(max_instance_job_count),
            )
            return None
        if min_instance_stage_count is not None and instance_stage_count < int(
            min_instance_stage_count
        ):
            logging.info(
                "[%s] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                context_label,
                instance_stage_count,
                int(min_instance_stage_count),
            )
            return None
        if max_instance_stage_count is not None and instance_stage_count > int(
            max_instance_stage_count
        ):
            logging.info(
                "[%s] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                context_label,
                instance_stage_count,
                int(max_instance_stage_count),
            )
            return None

        logging.info(
            "[%s] Running retained-stage CP-LB mode=%s with job_count=%d stage_count=%d "
            "workload_size=%d.",
            context_label,
            retained_stage_mode,
            instance_job_count,
            instance_stage_count,
            workload_size,
        )
        cp_lb_result = self.apply_retained_stage_cp_lb(
            threads=threads,
            tl_nc_multiplier=tl_nc_multiplier,
            time_limit_sec=time_limit_sec,
            time_limit_n_by_c_multiplier=time_limit_n_by_c_multiplier,
            retained_stage_mode=retained_stage_mode,
            bottleneck_stage_id=bottleneck_stage_id,
            extra_bottleneck_count=extra_bottleneck_count,
            bottleneck_band_radius=bottleneck_band_radius,
            middle_band_radius=middle_band_radius,
            quantile_count=quantile_count,
            retained_stage_ratios=retained_stage_ratios,
            save_cp_lb_artifacts=save_cp_lb_artifacts,
            snapshot_solution_limit=snapshot_solution_limit,
            snapshot_log_progress=snapshot_log_progress,
            log_cp_lb_ub_progress=log_cp_lb_ub_progress,
        )
        if cp_lb_result is None:
            logging.info(
                "[%s] CP-LB produced no result; skipping dispatch.", context_label
            )
            return None

        return self.dispatch_from_retained_cp(
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
            respect_anchor_stage_release_lb=respect_anchor_stage_release_lb,
            cp_local_repair_max_passes=cp_local_repair_max_passes,
            cp_local_repair_top_k=cp_local_repair_top_k,
            include_consensus_rank=include_consensus_rank,
            include_tail_bottleneck_rank=include_tail_bottleneck_rank,
            include_extended_rank_variants=include_extended_rank_variants,
            include_dynamic_priority=include_dynamic_priority,
            include_piecewise_stage_priority=include_piecewise_stage_priority,
            prune_unproductive_dispatch_candidates=prune_unproductive_dispatch_candidates,
            randomized_mixed_rank_trials=randomized_mixed_rank_trials,
            mixed_dispatch_methods=mixed_dispatch_methods,
            use_retained_cp_snapshot_portfolio=use_retained_cp_snapshot_portfolio,
            retained_cp_snapshot_top_k=retained_cp_snapshot_top_k,
            retained_cp_dispatch_selection_strategy=retained_cp_dispatch_selection_strategy,
            retained_cp_dispatch_makespan_slack=retained_cp_dispatch_makespan_slack,
            save_cp_dispatch_artifacts=save_cp_dispatch_artifacts,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def _resolve_anchor_stage_ids_from_last_retained_cp_lb(self) -> list[str]:
        result = getattr(self, "last_retained_cp_lb_result", None)
        if result is None:
            return []

        stage_id_list = list(self.instance.stage_id_list)
        if not stage_id_list:
            return []

        if (
            result.retained_stage_mode == "first_bottleneck_band_last"
            and result.bottleneck_stage_id is not None
        ):
            anchor_idx = stage_id_list.index(result.bottleneck_stage_id)
            radius = int(result.bottleneck_band_radius or 0)
            left_idx = max(0, anchor_idx - radius)
            right_idx = min(len(stage_id_list), anchor_idx + radius + 1)
            return stage_id_list[left_idx:right_idx]

        first_stage_id = stage_id_list[0]
        last_stage_id = stage_id_list[-1]
        internal_retained_stage_ids = [
            stage_id
            for stage_id in result.retained_stage_ids
            if stage_id not in {first_stage_id, last_stage_id}
        ]
        if not internal_retained_stage_ids:
            return []

        stage_2_index = {stage_id: idx for idx, stage_id in enumerate(stage_id_list)}
        sorted_internal_stage_ids = sorted(
            internal_retained_stage_ids, key=lambda stage_id: stage_2_index[stage_id]
        )

        contiguous_blocks: list[list[str]] = []
        current_block: list[str] = []
        former_idx: int | None = None
        for stage_id in sorted_internal_stage_ids:
            stage_idx = stage_2_index[stage_id]
            if former_idx is None or stage_idx == former_idx + 1:
                current_block.append(stage_id)
            else:
                contiguous_blocks.append(current_block)
                current_block = [stage_id]
            former_idx = stage_idx
        if current_block:
            contiguous_blocks.append(current_block)

        if not contiguous_blocks:
            return []
        if result.bottleneck_stage_id is not None:
            for block in contiguous_blocks:
                if result.bottleneck_stage_id in block:
                    return block
        return max(
            contiguous_blocks, key=lambda block: (len(block), -stage_2_index[block[0]])
        )

    def _extract_anchor_stage_sequence_from_last_retained_cp_lb(
        self,
        anchor_stage_ids: Sequence[str],
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
        retained_solution_rows = getattr(
            self, "last_retained_cp_lb_retained_solution_rows", None
        )
        if not retained_solution_rows:
            raise ValueError("No retained-stage CP solution rows are available.")

        anchor_stage_id_set = set(anchor_stage_ids)
        stage_2_rows: dict[str, list[dict[str, Any]]] = {
            stage_id: [] for stage_id in anchor_stage_ids
        }
        for row in retained_solution_rows:
            stage_id = str(row["stage_id"])
            if stage_id in anchor_stage_id_set:
                stage_2_rows[stage_id].append(dict(row))

        stage_2_job_sequence: dict[str, list[str]] = {}
        stage_2_job_2_release: dict[str, dict[str, int]] = {}
        for stage_id in anchor_stage_ids:
            rows = stage_2_rows.get(stage_id, [])
            if not rows:
                raise ValueError(
                    f"Missing retained-stage CP rows for anchor stage {stage_id}."
                )
            rows.sort(
                key=lambda row: (
                    int(row["start"]),
                    int(row["end"]),
                    str(row["job_id"]),
                )
            )
            stage_2_job_sequence[stage_id] = [str(row["job_id"]) for row in rows]
            stage_2_job_2_release[stage_id] = {
                str(row["job_id"]): int(row["start"]) for row in rows
            }
        return stage_2_job_sequence, stage_2_job_2_release

    def _get_schedule_by_retained_cp_two_way_stage_band(
        self,
        *,
        anchor_stage_ids: Sequence[str],
        stage_2_job_sequence: Mapping[str, Sequence[str]],
        mixed_schedule_for_former_stages: bool,
        mixed_schedule_for_later_stages: bool,
        machine_then_job: bool,
        stage_2_job_2_release: Mapping[str, Mapping[str, int]] | None,
        anchor_dispatch_mode: str,
    ) -> HybridFlowshopLiteSchedule:
        first_anchor_stage_id = str(anchor_stage_ids[0])
        first_anchor_sequence = list(stage_2_job_sequence[first_anchor_stage_id])
        job_tiebreak_rank = {
            str(job_id): idx for idx, job_id in enumerate(first_anchor_sequence)
        }
        option = BN2DOption(
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
        )
        dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            BN2DDispatcher,
            self.instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        return dispatcher.get_schedule_by_two_way_stage_band(
            anchor_stage_ids,
            stage_2_job_sequence,
            option,
            stage_2_job_2_release=stage_2_job_2_release,
            anchor_dispatch_mode=anchor_dispatch_mode,
        )

    def _get_schedule_by_stage_job_sequences_priority(
        self,
        *,
        stage_2_job_sequence: Mapping[str, Sequence[str]],
        stage_2_job_2_release: Mapping[str, Mapping[str, int]] | None = None,
    ) -> HybridFlowshopLiteSchedule:
        return build_schedule_from_stage_job_sequences_priority_score(
            self.create_empty_schedule_from_ins,
            stage_2_job_sequence,
            self.stage_2_job_2_p_dict,
            stage_2_job_2_release=stage_2_job_2_release,
        )

    def dispatch_from_retained_cp(
        self,
        *,
        mixed_schedule_for_former_stages: bool = True,
        mixed_schedule_for_later_stages: bool = True,
        machine_then_job: bool = False,
        respect_anchor_stage_release_lb: bool = True,
        cp_local_repair_max_passes: int = 3,
        cp_local_repair_top_k: int = 1,
        include_consensus_rank: bool = True,
        include_tail_bottleneck_rank: bool = True,
        include_extended_rank_variants: bool = False,
        include_dynamic_priority: bool = False,
        include_piecewise_stage_priority: bool = False,
        prune_unproductive_dispatch_candidates: bool = True,
        randomized_mixed_rank_trials: int = 0,
        mixed_dispatch_methods: Sequence[str] | None = None,
        use_retained_cp_snapshot_portfolio: bool = False,
        retained_cp_snapshot_top_k: int = 0,
        retained_cp_dispatch_selection_strategy: str = "best_makespan",
        retained_cp_dispatch_makespan_slack: float = 0.0,
        save_cp_dispatch_artifacts: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> dict[str, Any] | None:
        """Dispatch from the last retained-stage CP solution with CP-guided candidate evaluation."""
        sub_timer = ElapsedTimer()
        self.last_retained_cp_dispatch_obj = None
        self.last_retained_cp_post_dispatch_obj = None
        self.last_retained_cp_selected_dispatch_variant = None
        self.last_retained_cp_post_selected_dispatch_variant = None
        self.last_retained_cp_dispatch_anchor_stage_ids = None
        self.last_retained_cp_dispatch_elapsed_sec = None
        self.last_retained_cp_dispatch_was_incumbent_update = None
        self.last_retained_cp_dispatch_kept_incumbent = None
        self.last_retained_cp_dispatch_candidate_schedules = None
        self.last_retained_cp_dispatch_cp_ub = None
        self.last_retained_cp_dispatch_cp_lb = None
        retained_result = getattr(self, "last_retained_cp_lb_result", None)
        retained_solution_rows = getattr(
            self, "last_retained_cp_lb_retained_solution_rows", None
        )
        if retained_result is None or not retained_solution_rows:
            logging.warning(
                "[CP LB] No retained-stage CP incumbent is available for two-way dispatch."
            )
            return None

        def get_best_of_mixed_dispatches_for_retained_cp(
            **kwargs,
        ) -> HybridFlowshopLiteSchedule | None:
            return self._get_schedule_by_best_of_mixed_dispatches(
                mixed_dispatch_methods=mixed_dispatch_methods,
                **kwargs,
            )

        dependencies = PostRetainedCpDispatchDependencies(
            check_feasibility=self.check_feasibility,
            get_selected_dispatch_config=self._get_selected_dispatch_config_for_post_mip,
            get_best_mixed_schedule_from_job_sequence=self._get_best_mixed_schedule_from_job_sequence,
            get_schedule_by_best_of_mixed_dispatches=get_best_of_mixed_dispatches_for_retained_cp,
            get_two_way_schedule_by_stage_band=self._get_schedule_by_retained_cp_two_way_stage_band,
            get_schedule_by_stage_job_sequences_priority=self._get_schedule_by_stage_job_sequences_priority,
            repair_post_retained_cp_dispatch_candidate=self._repair_post_mip_dispatch_candidate,
        )
        dispatch_sources: list[tuple[str, Any, Sequence[Mapping[str, Any]]]] = [
            ("final", retained_result, retained_solution_rows)
        ]
        if use_retained_cp_snapshot_portfolio and retained_cp_snapshot_top_k > 0:
            snapshots = list(
                getattr(self, "last_retained_cp_lb_solution_snapshots", None) or []
            )
            final_ub = sanitize_optional_float(
                getattr(retained_result, "objective_ub", None)
            )
            ranked_snapshots = sorted(
                snapshots,
                key=lambda snapshot: (
                    float("inf")
                    if sanitize_optional_float(snapshot.get("objective_ub")) is None
                    else float(snapshot["objective_ub"]),
                    float("inf")
                    if sanitize_optional_float(snapshot.get("runtime_sec")) is None
                    else float(snapshot["runtime_sec"]),
                ),
            )
            snapshot_count = 0
            seen_snapshot_objectives: set[float] = set()
            for snapshot in ranked_snapshots:
                objective_ub = sanitize_optional_float(snapshot.get("objective_ub"))
                if objective_ub is None:
                    continue
                objective_key = round(objective_ub, 9)
                if final_ub is not None and objective_key == round(final_ub, 9):
                    continue
                if objective_key in seen_snapshot_objectives:
                    continue
                snapshot_rows = snapshot.get("retained_solution_rows")
                if not snapshot_rows:
                    continue
                seen_snapshot_objectives.add(objective_key)
                snapshot_count += 1
                snapshot_result = replace(
                    retained_result,
                    objective_ub=objective_ub,
                    objective_lb=sanitize_optional_float(snapshot.get("objective_lb")),
                    solver_runtime_sec=sanitize_optional_float(
                        snapshot.get("runtime_sec")
                    ),
                )
                dispatch_sources.append(
                    (
                        f"snapshot_{snapshot_count}_ub_{objective_ub:g}",
                        snapshot_result,
                        snapshot_rows,
                    )
                )
                if snapshot_count >= int(retained_cp_snapshot_top_k):
                    break
            logging.info(
                "[CP LB] Retained-CP snapshot dispatch portfolio includes %d snapshot sources.",
                max(0, len(dispatch_sources) - 1),
            )

        dispatch_evaluations: list[
            tuple[str, Any, PostRetainedCpDispatchRunResult]
        ] = []
        for source_label, source_result, source_rows in dispatch_sources:
            dispatch_result_for_source = run_post_retained_cp_dispatch(
                instance=self.instance,
                retained_cp_result=source_result,
                retained_solution_rows=source_rows,
                cp_local_repair_max_passes=cp_local_repair_max_passes,
                cp_local_repair_top_k=cp_local_repair_top_k,
                include_release_anchor_candidates=respect_anchor_stage_release_lb,
                include_consensus_rank=include_consensus_rank,
                include_tail_bottleneck_rank=include_tail_bottleneck_rank,
                include_extended_rank_variants=include_extended_rank_variants,
                include_dynamic_priority=include_dynamic_priority,
                include_piecewise_stage_priority=include_piecewise_stage_priority,
                prune_unproductive_dispatch_candidates=prune_unproductive_dispatch_candidates,
                randomized_mixed_rank_trials=randomized_mixed_rank_trials,
                dependencies=dependencies,
            )
            source_schedule = dispatch_result_for_source.dispatched_schedule
            logging.info(
                "[CP LB] Retained dispatch source %s cp_ub=%s selected %s with makespan=%s",
                source_label,
                getattr(source_result, "objective_ub", None),
                dispatch_result_for_source.selected_dispatch_variant,
                source_schedule.makespan if source_schedule is not None else None,
            )
            dispatch_evaluations.append(
                (source_label, source_result, dispatch_result_for_source)
            )

        feasible_dispatch_tuples = [
            (source_label, source_result, dispatch_result_for_source)
            for (
                source_label,
                source_result,
                dispatch_result_for_source,
            ) in dispatch_evaluations
            if dispatch_result_for_source.dispatched_schedule is not None
        ]
        best_dispatch_tuple = None
        if feasible_dispatch_tuples:
            if retained_cp_dispatch_selection_strategy == "best_makespan":
                best_dispatch_tuple = min(
                    feasible_dispatch_tuples,
                    key=lambda item: item[2].dispatched_schedule.makespan,
                )
            elif (
                retained_cp_dispatch_selection_strategy
                == "earliest_snapshot_within_makespan_slack"
            ):
                best_makespan = min(
                    item[2].dispatched_schedule.makespan
                    for item in feasible_dispatch_tuples
                )
                makespan_slack = max(0.0, float(retained_cp_dispatch_makespan_slack))
                allowed_makespan = best_makespan + makespan_slack
                slack_candidates = [
                    item
                    for item in feasible_dispatch_tuples
                    if item[0] != "final"
                    and item[2].dispatched_schedule.makespan <= allowed_makespan
                ]
                if not slack_candidates:
                    slack_candidates = [
                        item
                        for item in feasible_dispatch_tuples
                        if item[2].dispatched_schedule.makespan <= allowed_makespan
                    ]
                best_dispatch_tuple = (
                    slack_candidates[0]
                    if slack_candidates
                    else min(
                        feasible_dispatch_tuples,
                        key=lambda item: item[2].dispatched_schedule.makespan,
                    )
                )
                logging.info(
                    "[CP LB] Snapshot slack dispatch selection chose %s with "
                    "makespan=%s (best=%s, slack=%s)",
                    best_dispatch_tuple[0],
                    best_dispatch_tuple[2].dispatched_schedule.makespan,
                    best_makespan,
                    makespan_slack,
                )
            else:
                raise ValueError(
                    "Unknown retained_cp_dispatch_selection_strategy: "
                    f"{retained_cp_dispatch_selection_strategy!r}"
                )
        if best_dispatch_tuple is None:
            schedule = None
            dispatch_result = dispatch_evaluations[-1][2]
            dispatch_source_label = "none"
            dispatch_retained_result = retained_result
        else:
            dispatch_source_label, dispatch_retained_result, dispatch_result = (
                best_dispatch_tuple
            )
            schedule = dispatch_result.dispatched_schedule
        if schedule is None:
            logging.warning(
                "[CP LB] No feasible post-retained-CP dispatch schedule was generated."
            )
            return None
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        incumbent_schedule = self.solution_manager.get_incumbent()
        incumbent_before = self.solution_manager.best_obj_value
        post_dispatch_obj = float(schedule.makespan)
        selected_schedule = schedule
        selected_obj = post_dispatch_obj
        post_selected_variant = dispatch_result.selected_dispatch_variant
        if dispatch_source_label == "final" or post_selected_variant is None:
            selected_variant = post_selected_variant
        else:
            selected_variant = f"{dispatch_source_label}:{post_selected_variant}"
        post_selected_variant_with_source = selected_variant
        selected_anchor_stage_ids = list(
            dispatch_result.variant_2_anchor_stage_ids.get(
                str(post_selected_variant),
                (),
            )
        )
        kept_incumbent = False
        is_post_dispatch_improvement = True
        if incumbent_before is not None:
            obj_value_comparator = getattr(
                self.solution_manager, "_a_is_better_obj_value", None
            )
            if callable(obj_value_comparator):
                is_post_dispatch_improvement = bool(
                    obj_value_comparator(post_dispatch_obj, incumbent_before)
                )
            else:
                is_post_dispatch_improvement = post_dispatch_obj < incumbent_before
        if (
            incumbent_schedule is not None
            and incumbent_before is not None
            and not is_post_dispatch_improvement
        ):
            selected_schedule = incumbent_schedule
            selected_obj = float(incumbent_before)
            selected_variant = "incumbent_before_retained_cp"
            selected_anchor_stage_ids = []
            kept_incumbent = True

        is_init = incumbent_schedule is None
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=selected_obj,
            obj_bound=None,
            is_init=is_init,
            subroutine_name="dispatch_from_retained_cp",
            progress_obj_value_records=[(sub_timer.elapsed_sec, selected_obj)],
        )
        was_updated = self.solution_manager.register(report, selected_schedule)
        self.last_retained_cp_dispatch_obj = selected_obj
        self.last_retained_cp_post_dispatch_obj = post_dispatch_obj
        self.last_retained_cp_selected_dispatch_variant = selected_variant
        self.last_retained_cp_post_selected_dispatch_variant = (
            post_selected_variant_with_source
        )
        self.last_retained_cp_selected_dispatch_source = dispatch_source_label
        self.last_retained_cp_dispatch_anchor_stage_ids = selected_anchor_stage_ids
        self.last_retained_cp_dispatch_elapsed_sec = sub_timer.elapsed_sec
        self.last_retained_cp_dispatch_was_incumbent_update = bool(was_updated)
        self.last_retained_cp_dispatch_kept_incumbent = kept_incumbent
        self.last_retained_cp_dispatch_cp_ub = sanitize_optional_float(
            getattr(dispatch_retained_result, "objective_ub", None)
        )
        self.last_retained_cp_dispatch_cp_lb = sanitize_optional_float(
            getattr(dispatch_retained_result, "certified_final_lb", None)
        )

        if kept_incumbent:
            logging.info(
                "[CP LB] Kept incumbent makespan=%s over post-retained-CP dispatch "
                "%s makespan=%s (cp_ub=%s, cp_lb=%s, updated_incumbent=%s)",
                selected_obj,
                post_selected_variant_with_source,
                post_dispatch_obj,
                getattr(dispatch_retained_result, "objective_ub", None),
                getattr(dispatch_retained_result, "certified_final_lb", None),
                was_updated,
            )
        else:
            logging.info(
                "[CP LB] Post-retained-CP dispatch selected %s on anchor stages %s with makespan=%s "
                "(cp_ub=%s, cp_lb=%s, incumbent_before=%s, updated_incumbent=%s)",
                selected_variant,
                self.last_retained_cp_dispatch_anchor_stage_ids,
                selected_obj,
                getattr(dispatch_retained_result, "objective_ub", None),
                getattr(dispatch_retained_result, "certified_final_lb", None),
                incumbent_before,
                was_updated,
            )

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, selected_obj, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        if save_cp_dispatch_artifacts:
            write_post_retained_cp_dispatch_artifacts(
                cp_lb_dir=self._working_dir_path / "cp_lb",
                dispatch_result=dispatch_result,
                retained_cp_result=dispatch_retained_result,
                apply_elapsed_sec=getattr(
                    self, "last_retained_cp_lb_apply_elapsed_sec", None
                ),
                dispatch_elapsed_sec=sub_timer.elapsed_sec,
                draw_visualizations=draw_gantt,
            )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

        dispatch_candidates = {}
        dispatch_candidate_schedules = {}
        for source_label, _, evaluated_dispatch_result in dispatch_evaluations:
            prefix = "" if source_label == "final" else f"{source_label}:"
            for (
                variant,
                candidate_schedule,
            ) in evaluated_dispatch_result.dispatched_schedules.items():
                candidate_label = f"{prefix}{variant}"
                dispatch_candidates[candidate_label] = (
                    candidate_schedule.makespan
                    if candidate_schedule is not None
                    else None
                )
                if candidate_schedule is not None:
                    dispatch_candidate_schedules[candidate_label] = candidate_schedule
        if incumbent_before is not None:
            dispatch_candidates["incumbent_before_retained_cp"] = incumbent_before
        if incumbent_schedule is not None:
            dispatch_candidate_schedules["incumbent_before_retained_cp"] = (
                incumbent_schedule
            )
        self.last_retained_cp_dispatch_candidate_schedules = (
            dispatch_candidate_schedules
        )
        return {
            "schedule": selected_schedule,
            "selected_dispatch_variant": selected_variant,
            "post_cp_selected_dispatch_variant": (post_selected_variant_with_source),
            "post_cp_selected_obj": post_dispatch_obj,
            "kept_incumbent": kept_incumbent,
            "anchor_stage_ids": list(self.last_retained_cp_dispatch_anchor_stage_ids),
            "dispatch_candidates": dispatch_candidates,
            "selected_dispatch_source": dispatch_source_label,
        }

    def improve_retained_cp_dispatch_candidates_with_base_cp(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        max_candidate_count: int = 2,
        max_makespan_slack: float | None = 5.0,
        max_relative_slack: float | None = None,
        include_incumbent_candidate: bool = False,
        use_lns_only: bool | None = True,
        cp_model_probing_level: int | None = 1,
        log_search_progress: bool = False,
        cleanup_added_constraints: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Try short CP-LNS solves from near-miss retained-dispatch candidates.

        ``dispatch_from_retained_cp`` can generate structurally different
        schedules that are slightly worse than the incumbent. Keeping only the
        best incumbent throws that diversity away. This subroutine uses those
        near-miss schedules as full CP hints and registers a result only if the
        CP solve turns one of them into a real incumbent improvement.
        """
        sub_timer = ElapsedTimer()
        candidate_schedules = getattr(
            self, "last_retained_cp_dispatch_candidate_schedules", None
        )
        if not candidate_schedules:
            logging.info(
                "[Retained Candidate CP] No retained-dispatch candidate schedules "
                "are available."
            )
            return

        incumbent_obj = sanitize_optional_float(self.solution_manager.best_obj_value)
        selected_candidates: list[tuple[str, float, HybridFlowshopLiteSchedule]] = []
        for label, schedule in candidate_schedules.items():
            if schedule is None:
                continue
            if (
                label == "incumbent_before_retained_cp"
                and not include_incumbent_candidate
            ):
                continue
            candidate_obj = float(schedule.makespan)
            allowed = incumbent_obj is None or candidate_obj < incumbent_obj
            if not allowed and incumbent_obj is not None:
                if max_makespan_slack is not None:
                    allowed = candidate_obj <= incumbent_obj + float(max_makespan_slack)
                if not allowed and max_relative_slack is not None:
                    allowed = candidate_obj <= incumbent_obj * (
                        1.0 + float(max_relative_slack)
                    )
            if allowed:
                selected_candidates.append((str(label), candidate_obj, schedule))

        selected_candidates.sort(key=lambda item: (item[1], item[0]))
        if max_candidate_count > 0:
            selected_candidates = selected_candidates[: int(max_candidate_count)]
        if not selected_candidates:
            logging.info(
                "[Retained Candidate CP] No candidates passed the near-miss filter "
                "(incumbent=%s max_makespan_slack=%s max_relative_slack=%s).",
                incumbent_obj,
                max_makespan_slack,
                max_relative_slack,
            )
            return

        logging.info(
            "[Retained Candidate CP] Trying %d candidate(s) from retained dispatch: %s",
            len(selected_candidates),
            [(label, obj) for label, obj, _ in selected_candidates],
        )

        try:
            for label, candidate_obj, candidate_schedule in selected_candidates:
                if self.is_stopping_condition(log_reason_if_true=False):
                    logging.info(
                        "[Retained Candidate CP] Stopping before candidate %s because "
                        "the global stopping condition is met.",
                        label,
                    )
                    break

                if self.base_cp_model_is_set:
                    self.cp_model.delete_added_constraints()
                else:
                    self.set_cp_model_as_base_cp_model()
                self.cp_model.clear_hints()
                BaseModelBuilder.apply_start_hints_from_start_time_map(
                    self.cp_model,
                    self.params,
                    self.vars,
                    candidate_schedule.get_jik_2_start_time_map(),
                )
                BaseModelBuilder.apply_end_hints_from_end_time_map(
                    self.cp_model,
                    self.params,
                    self.vars,
                    candidate_schedule.get_jik_2_end_time_map(),
                )
                self.cp_model.add_hint(self.vars.makespan, int(candidate_obj))

                before_obj = self.solution_manager.best_obj_value
                candidate_time = self._resolve_tl_nc_computational_time(
                    computational_time=computational_time,
                    tl_nc_multiplier=tl_nc_multiplier,
                )
                logging.info(
                    "[Retained Candidate CP] Solving from candidate %s obj=%s "
                    "incumbent_before=%s time_limit=%s use_lns_only=%s.",
                    label,
                    candidate_obj,
                    before_obj,
                    candidate_time,
                    use_lns_only,
                )
                report, solution = self.solve_current_cp_remaining_time_limit(
                    candidate_time,
                    solver_thread_cnt,
                    obj_value_is_valid=True,
                    obj_bound_is_valid=False,
                    is_initial_solution=False,
                    use_lns_only=use_lns_only,
                    cp_model_probing_level=cp_model_probing_level,
                    log_search_progress=log_search_progress,
                    error_if_infeasible=error_if_infeasible,
                    draw_gantt=draw_gantt,
                )
                if solution is None:
                    logging.info(
                        "[Retained Candidate CP] Candidate %s produced no feasible "
                        "CP solution (status=%s).",
                        label,
                        report.status,
                    )
                    continue

                was_updated = self.solution_manager.register(report, solution)
                after_obj = self.solution_manager.best_obj_value
                logging.info(
                    "[Retained Candidate CP] Candidate %s finished: cp_obj=%s "
                    "incumbent_before=%s incumbent_after=%s updated=%s.",
                    label,
                    solution.makespan,
                    before_obj,
                    after_obj,
                    was_updated,
                )
                if was_updated and draw_gantt:
                    self.draw_incumbent_gantt()

                log_time = self.timer.elapsed_sec
                if after_obj is not None:
                    self.add_obj_value_log(log_time, after_obj, is_maximize=False)
                    self.obj_store.add_last_timestamp_note(
                        self._get_call_context_of_current_method(),
                        obj_value_is_valid=True,
                    )
        finally:
            cp_model = getattr(self, "cp_model", None)
            if cleanup_added_constraints and getattr(
                self, "base_cp_model_is_set", False
            ):
                cp_model.delete_added_constraints()
            if cleanup_added_constraints and hasattr(cp_model, "clear_hints"):
                cp_model.clear_hints()

        logging.info(
            "[Retained Candidate CP] Finished in %.3fs with incumbent=%s.",
            sub_timer.elapsed_sec,
            self.solution_manager.best_obj_value,
        )

    def dispatch_from_retained_cp_two_way(
        self,
        *,
        mixed_schedule_for_former_stages: bool = True,
        mixed_schedule_for_later_stages: bool = True,
        machine_then_job: bool = False,
        respect_anchor_stage_release_lb: bool = True,
        cp_local_repair_max_passes: int = 3,
        cp_local_repair_top_k: int = 1,
        include_consensus_rank: bool = True,
        include_tail_bottleneck_rank: bool = True,
        include_dynamic_priority: bool = False,
        prune_unproductive_dispatch_candidates: bool = True,
        randomized_mixed_rank_trials: int = 0,
        mixed_dispatch_methods: Sequence[str] | None = None,
        save_cp_dispatch_artifacts: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> dict[str, Any] | None:
        """Backward-compatible alias for dispatch_from_retained_cp()."""
        return self.dispatch_from_retained_cp(
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
            respect_anchor_stage_release_lb=respect_anchor_stage_release_lb,
            cp_local_repair_max_passes=cp_local_repair_max_passes,
            cp_local_repair_top_k=cp_local_repair_top_k,
            include_consensus_rank=include_consensus_rank,
            include_tail_bottleneck_rank=include_tail_bottleneck_rank,
            include_dynamic_priority=include_dynamic_priority,
            prune_unproductive_dispatch_candidates=prune_unproductive_dispatch_candidates,
            randomized_mixed_rank_trials=randomized_mixed_rank_trials,
            mixed_dispatch_methods=mixed_dispatch_methods,
            save_cp_dispatch_artifacts=save_cp_dispatch_artifacts,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def workload_guarded_dispatch_from_retained_cp(
        self,
        *,
        mixed_schedule_for_former_stages: bool = True,
        mixed_schedule_for_later_stages: bool = True,
        machine_then_job: bool = False,
        respect_anchor_stage_release_lb: bool = True,
        cp_local_repair_max_passes: int = 3,
        cp_local_repair_top_k: int = 1,
        include_consensus_rank: bool = True,
        include_tail_bottleneck_rank: bool = True,
        include_extended_rank_variants: bool = False,
        include_dynamic_priority: bool = False,
        include_piecewise_stage_priority: bool = False,
        prune_unproductive_dispatch_candidates: bool = True,
        randomized_mixed_rank_trials: int = 0,
        mixed_dispatch_methods: Sequence[str] | None = None,
        use_retained_cp_snapshot_portfolio: bool = False,
        retained_cp_snapshot_top_k: int = 0,
        retained_cp_dispatch_selection_strategy: str = "best_makespan",
        retained_cp_dispatch_makespan_slack: float = 0.0,
        save_cp_dispatch_artifacts: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
    ) -> dict[str, Any] | None:
        """Run retained-CP dispatch only inside configured instance/gap guards."""
        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        context = "Guarded Retained-CP Dispatch"
        if not self._bound_gap_guard_allows(
            context_label=context,
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return None
        if min_workload_size is not None and workload_size < int(min_workload_size):
            logging.info(
                "[%s] Skipping: workload_size=%d is below min_workload_size=%d.",
                context,
                workload_size,
                int(min_workload_size),
            )
            return None
        if max_workload_size is not None and workload_size > int(max_workload_size):
            logging.info(
                "[%s] Skipping: workload_size=%d is above max_workload_size=%d.",
                context,
                workload_size,
                int(max_workload_size),
            )
            return None
        if min_instance_job_count is not None and instance_job_count < int(
            min_instance_job_count
        ):
            logging.info(
                "[%s] Skipping: job_count=%d is below min_instance_job_count=%d.",
                context,
                instance_job_count,
                int(min_instance_job_count),
            )
            return None
        if max_instance_job_count is not None and instance_job_count > int(
            max_instance_job_count
        ):
            logging.info(
                "[%s] Skipping: job_count=%d is above max_instance_job_count=%d.",
                context,
                instance_job_count,
                int(max_instance_job_count),
            )
            return None
        if min_instance_stage_count is not None and instance_stage_count < int(
            min_instance_stage_count
        ):
            logging.info(
                "[%s] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                context,
                instance_stage_count,
                int(min_instance_stage_count),
            )
            return None
        if max_instance_stage_count is not None and instance_stage_count > int(
            max_instance_stage_count
        ):
            logging.info(
                "[%s] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                context,
                instance_stage_count,
                int(max_instance_stage_count),
            )
            return None

        logging.info(
            "[%s] Running for job_count=%d stage_count=%d workload_size=%d.",
            context,
            instance_job_count,
            instance_stage_count,
            workload_size,
        )
        return self.dispatch_from_retained_cp(
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
            respect_anchor_stage_release_lb=respect_anchor_stage_release_lb,
            cp_local_repair_max_passes=cp_local_repair_max_passes,
            cp_local_repair_top_k=cp_local_repair_top_k,
            include_consensus_rank=include_consensus_rank,
            include_tail_bottleneck_rank=include_tail_bottleneck_rank,
            include_extended_rank_variants=include_extended_rank_variants,
            include_dynamic_priority=include_dynamic_priority,
            include_piecewise_stage_priority=include_piecewise_stage_priority,
            prune_unproductive_dispatch_candidates=prune_unproductive_dispatch_candidates,
            randomized_mixed_rank_trials=randomized_mixed_rank_trials,
            mixed_dispatch_methods=mixed_dispatch_methods,
            use_retained_cp_snapshot_portfolio=use_retained_cp_snapshot_portfolio,
            retained_cp_snapshot_top_k=retained_cp_snapshot_top_k,
            retained_cp_dispatch_selection_strategy=retained_cp_dispatch_selection_strategy,
            retained_cp_dispatch_makespan_slack=retained_cp_dispatch_makespan_slack,
            save_cp_dispatch_artifacts=save_cp_dispatch_artifacts,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def complete_from_retained_cp(
        self,
        solver_thread_cnt: int,
        computational_time: float | None = None,
        tl_nc_multiplier: float | None = None,
        retained_stage_scope: str = "all",
        retained_stage_ids: Sequence[str] | None = None,
        time_slack: int = 0,
        cp_lb_dir: str | None = None,
        cp_lb_source_scenario_dir: str | None = None,
        use_lns_only: bool | None = False,
        cp_model_probing_level: int | None = None,
        log_search_progress: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Complete the last retained-stage CP solution by solving the full CP model.

        Retained-stage CP gives start times for a relaxed subset of stages. This
        subroutine uses those retained start times as hard anchor windows, then
        asks the full HFS CP model to schedule the omitted stages around them.
        """
        sub_timer = ElapsedTimer()
        if cp_lb_dir is not None or cp_lb_source_scenario_dir is not None:
            from lb_bucket.cp.solution_io import read_retained_stage_cp_artifacts

            if cp_lb_dir is not None:
                source_dir = Path(cp_lb_dir)
            else:
                source_dir = (
                    Path(str(cp_lb_source_scenario_dir))
                    / str(self.instance.name)
                    / "cp_lb"
                )

            retained_result, retained_solution_rows = read_retained_stage_cp_artifacts(
                source_dir
            )
            if retained_result is None or not retained_solution_rows:
                logging.warning(
                    "[CP Complete] No saved retained-stage CP artifacts found "
                    "under %s.",
                    source_dir,
                )
                return

            self.last_retained_cp_lb_result = retained_result
            self.last_retained_cp_lb_retained_solution_rows = retained_solution_rows
            self.last_retained_cp_lb_apply_elapsed_sec = None
            logging.info(
                "[CP Complete] Loaded saved retained-stage CP artifacts from %s.",
                source_dir,
            )

        retained_solution_rows = getattr(
            self, "last_retained_cp_lb_retained_solution_rows", None
        )
        if not retained_solution_rows:
            logging.warning(
                "[CP Complete] No retained-stage CP solution rows are available."
            )
            return

        selected_stage_ids = self._resolve_retained_completion_stage_ids(
            retained_stage_scope=retained_stage_scope,
            retained_stage_ids=retained_stage_ids,
        )
        if not selected_stage_ids:
            logging.warning("[CP Complete] No retained stages selected for completion.")
            return

        if self.base_cp_model_is_set:
            self.cp_model.delete_added_constraints()
        else:
            self.set_cp_model_as_base_cp_model()

        selected_stage_id_set = set(selected_stage_ids)
        slack = max(0, int(time_slack))
        added_anchor_count = 0
        for row in retained_solution_rows:
            stage_id = str(row["stage_id"])
            job_id = str(row["job_id"])
            if stage_id not in selected_stage_id_set:
                continue
            if (job_id, stage_id) not in self.vars.op_start:
                continue

            start_time = int(row["start"])
            lower = max(0, start_time - slack)
            upper = start_time + slack
            self.cp_model.add(self.vars.op_start[job_id, stage_id] >= lower)
            self.cp_model.add(self.vars.op_start[job_id, stage_id] <= upper)
            added_anchor_count += 1

        if added_anchor_count == 0:
            logging.warning(
                "[CP Complete] Retained-stage rows produced no valid full-CP anchors."
            )
            return

        logging.info(
            "[CP Complete] Solving full CP with %d retained-stage anchors "
            "(scope=%s stages=%s time_slack=%d).",
            added_anchor_count,
            retained_stage_scope,
            selected_stage_ids,
            slack,
        )

        _computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        if _computational_time is not None:
            _computational_time = max(0.0, _computational_time - sub_timer.elapsed_sec)

        if self.solution_manager.get_incumbent() is None:
            report, solution = self.solve_current_cp_remaining_time_limit(
                _computational_time,
                solver_thread_cnt,
                obj_value_is_valid=True,
                obj_bound_is_valid=False,
                is_initial_solution=True,
                use_lns_only=use_lns_only,
                cp_model_probing_level=cp_model_probing_level,
                log_search_progress=log_search_progress,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )
        else:
            report, solution = self.solve_with_initial_solution(
                _computational_time,
                solver_thread_cnt,
                obj_value_is_valid=True,
                obj_bound_is_valid=False,
                use_lns_only=use_lns_only,
                cp_model_probing_level=cp_model_probing_level,
                log_search_progress=log_search_progress,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )

        if solution is None:
            logging.info(
                "[CP Complete] Full-CP completion did not produce a feasible "
                "schedule (status=%s).",
                report.status,
            )
            return

        report = report.copy(
            elapsed_time=sub_timer.elapsed_sec,
            subroutine_name="complete_from_retained_cp",
            call_context=self._get_call_context_of_current_method(),
        )
        was_updated = self.solution_manager.register(report, solution)

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(solution.makespan), is_maximize=False)
        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
        )
        logging.info(
            "[CP Complete] Completed retained-stage full CP with makespan=%s "
            "updated_incumbent=%s.",
            solution.makespan,
            was_updated,
        )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _resolve_retained_completion_stage_ids(
        self,
        *,
        retained_stage_scope: str,
        retained_stage_ids: Sequence[str] | None,
    ) -> list[str]:
        if retained_stage_ids is not None:
            explicit_ids = [str(stage_id) for stage_id in retained_stage_ids]
            unknown_ids = [
                stage_id
                for stage_id in explicit_ids
                if stage_id not in self.instance.stage_id_list
            ]
            if unknown_ids:
                raise ValueError(
                    f"Unknown retained_stage_ids for completion: {unknown_ids}."
                )
            return explicit_ids

        result = getattr(self, "last_retained_cp_lb_result", None)
        if result is None:
            return []

        stage_id_list = list(self.instance.stage_id_list)
        retained_ids = [str(stage_id) for stage_id in result.retained_stage_ids]
        if retained_stage_scope == "all":
            return retained_ids

        if retained_stage_scope == "preferred_anchor":
            return self._resolve_anchor_stage_ids_from_last_retained_cp_lb()

        if retained_stage_scope == "bottlenecks":
            selected = [
                str(stage_id)
                for stage_id in (
                    getattr(result, "selected_bottleneck_stage_ids", ()) or ()
                )
            ]
            if selected:
                return selected
            bottleneck_stage_id = getattr(result, "bottleneck_stage_id", None)
            return [str(bottleneck_stage_id)] if bottleneck_stage_id is not None else []

        if retained_stage_scope == "first_last":
            if not stage_id_list:
                return []
            return [
                stage_id
                for stage_id in (stage_id_list[0], stage_id_list[-1])
                if stage_id in retained_ids
            ]

        if retained_stage_scope == "first_bottlenecks_last":
            selected = [
                str(stage_id)
                for stage_id in (
                    getattr(result, "selected_bottleneck_stage_ids", ()) or ()
                )
            ]
            candidates = [stage_id_list[0], *selected, stage_id_list[-1]]
            return [stage_id for stage_id in candidates if stage_id in retained_ids]

        raise ValueError(
            "retained_stage_scope must be one of "
            "{'all', 'preferred_anchor', 'bottlenecks', "
            "'first_last', 'first_bottlenecks_last'}."
        )

    def dispatch_from_saved_retained_cp(
        self,
        *,
        cp_lb_dir: str | None = None,
        cp_lb_source_scenario_dir: str | None = None,
        mixed_schedule_for_former_stages: bool = True,
        mixed_schedule_for_later_stages: bool = True,
        machine_then_job: bool = False,
        respect_anchor_stage_release_lb: bool = True,
        cp_local_repair_max_passes: int = 3,
        cp_local_repair_top_k: int = 1,
        include_consensus_rank: bool = True,
        include_tail_bottleneck_rank: bool = True,
        include_dynamic_priority: bool = False,
        prune_unproductive_dispatch_candidates: bool = True,
        randomized_mixed_rank_trials: int = 0,
        mixed_dispatch_methods: Sequence[str] | None = None,
        save_cp_dispatch_artifacts: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> dict[str, Any] | None:
        """Reload a saved retained-CP incumbent and rerun only the post-retained-CP dispatch logic."""
        from lb_bucket.cp.solution_io import read_retained_stage_cp_artifacts

        if cp_lb_dir is not None:
            source_dir = Path(cp_lb_dir)
        elif cp_lb_source_scenario_dir is not None:
            source_dir = (
                Path(cp_lb_source_scenario_dir) / str(self.instance.name) / "cp_lb"
            )
        elif self._working_dir_path is not None:
            source_dir = self._working_dir_path / "cp_lb"
        else:
            logging.warning(
                "[CP LB] No working directory is available to load a saved retained-CP payload."
            )
            return None

        retained_result, retained_solution_rows = read_retained_stage_cp_artifacts(
            source_dir
        )
        if retained_result is None or not retained_solution_rows:
            logging.warning(
                "[CP LB] No saved retained-CP payload was found for instance %s under %s.",
                self.instance.name,
                source_dir,
            )
            return None

        self.last_retained_cp_lb_result = retained_result
        self.last_retained_cp_lb_retained_solution_rows = retained_solution_rows
        self.last_retained_cp_lb_apply_elapsed_sec = None
        logging.info(
            "[CP LB] Reloaded saved retained-CP payload from %s (cp_ub=%s, cp_lb=%s)",
            source_dir,
            retained_result.objective_ub,
            retained_result.certified_final_lb,
        )
        return self.dispatch_from_retained_cp(
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
            respect_anchor_stage_release_lb=respect_anchor_stage_release_lb,
            cp_local_repair_max_passes=cp_local_repair_max_passes,
            cp_local_repair_top_k=cp_local_repair_top_k,
            include_consensus_rank=include_consensus_rank,
            include_tail_bottleneck_rank=include_tail_bottleneck_rank,
            include_dynamic_priority=include_dynamic_priority,
            prune_unproductive_dispatch_candidates=prune_unproductive_dispatch_candidates,
            randomized_mixed_rank_trials=randomized_mixed_rank_trials,
            mixed_dispatch_methods=mixed_dispatch_methods,
            save_cp_dispatch_artifacts=save_cp_dispatch_artifacts,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_mip_lb(
        self,
        # Gurobi parameters
        threads: int = 24,
        tl_nc_multiplier: float | None = None,
        time_limit_sec: float | None = None,
        time_limit_n_by_c_multiplier: float | None = None,
        display_interval_sec: int | None = None,
        log_to_console: bool = False,
        # Delta parameters
        delta: int | None = None,
        delta_pmax_plus_one: bool = False,
        # Formulation parameters
        precedence_formulation: str | None = None,
        base_model_only: bool = False,
        disable_valid_ineq_i: bool = False,
        disable_valid_ineq_ii: bool = False,
        disable_valid_ineq_iii: bool = False,
        disable_valid_ineq_iv: bool = False,
        # Post-MIP dispatch parameters
        es_ls_local_repair_max_passes: int = 3,
        draw_mip_lb_visualizations: bool = True,
    ) -> dict[str, Any] | None:
        """
        Compute lower bound using the bucket-indexed MIP formulation with Gurobi.

        This method follows the same pattern as apply_shdlb, calling the MIP solver
        from lb_bucket/mip/search.py to compute a potentially stronger lower bound.

        Args:
            threads: Gurobi threads (default: 24).
            time_limit_sec: Per-instance Gurobi time limit.
            time_limit_n_by_c_multiplier: If ``time_limit_sec`` is None, uses
                ``job_count * stage_count * time_limit_n_by_c_multiplier`` as the
                MIP time limit before applying the controller's remaining-time cap.
            display_interval_sec: Gurobi DisplayInterval.
            log_to_console: Forward Gurobi logs to console (default: False).
            delta: Fixed bucket size. If None, uses max processing time.
            delta_pmax_plus_one: If True, sets delta = max_p_ij + 1.
            precedence_formulation: Precedence formulation ("bucket", "d", or "e").
            base_model_only: Disable all model strengthening (default: False).
            disable_valid_ineq_i: Disable valid-inequality family (i).
            disable_valid_ineq_ii: Disable valid-inequality family (ii).
            disable_valid_ineq_iii: Disable valid-inequality family (iii).
            disable_valid_ineq_iv: Disable valid-inequality family (iv).
            es_ls_local_repair_max_passes: Number of post-MIP local-repair passes
                applied to the currently best dispatch candidate.
            draw_mip_lb_visualizations: If False, skip bucket-MIP/post-dispatch PNG
                exports while still saving the lightweight YAML/CSV artifacts.
        """
        start_t = self.timer.elapsed_sec
        sub_timer = ElapsedTimer()
        self.last_mip_lb_apply_elapsed_sec = None
        self.last_mip_lb_post_dispatch_elapsed_sec = None
        instance = self.instance

        # Early return: no upper bound
        input_ub = self.solution_manager.best_obj_value
        if input_ub is None:
            logging.warning("[MIP LB] No upper bound available, skipping")
            return

        # Import Gurobi
        gp, grb = import_gurobi()

        # Transform instance to TwoBucketInstance
        processing_times_by_stage = [
            [
                instance.stage_2_job_2_p_map[stage_id][job_id]
                for job_id in instance.job_id_list
            ]
            for stage_id in instance.stage_id_list
        ]
        machine_count_per_stage = [
            len(instance.stage_2_machines_map[stage_id])
            for stage_id in instance.stage_id_list
        ]

        two_bucket_instance = TwoBucketInstance(
            ins_name=instance.name,
            job_count=instance.job_count,
            stage_count=instance.stage_count,
            machine_count_per_stage=machine_count_per_stage,
            processing_times_by_stage=processing_times_by_stage,
            stage_ids=list(instance.stage_id_list),
            job_ids=list(instance.job_id_list),
        )

        # Determine delta
        max_p = max(
            max(times.values()) for times in instance.stage_2_job_2_p_map.values()
        )
        if delta_pmax_plus_one:
            delta = max_p + 1
        elif delta is None:
            delta = max_p

        # Build strengthening options
        if base_model_only:
            strengthening = ModelStrengtheningOptions(
                cumulative_precedence=False,
                valid_ineq_i=False,
                valid_ineq_ii=False,
                valid_ineq_iii=False,
                valid_ineq_iv=False,
            )
        else:
            strengthening = ModelStrengtheningOptions(
                cumulative_precedence=True,
                valid_ineq_i=not disable_valid_ineq_i,
                valid_ineq_ii=not disable_valid_ineq_ii,
                valid_ineq_iii=not disable_valid_ineq_iii,
                valid_ineq_iv=not disable_valid_ineq_iv,
            )

        precedence = PrecedenceOptions(formulation=precedence_formulation or "d")

        # Create summary record
        record = SummaryBoundRecord(
            ins_name=instance.name,
            input_lb=self.solution_manager.best_obj_bound or 1,
            input_ub=input_ub,
            job_count=instance.job_count,
            stage_count=instance.stage_count,
        )

        ub_schedule = None
        last_sch_lite = self.solution_manager.get_incumbent()
        if last_sch_lite is not None:
            ub_schedule = from_start_end_time_maps_create_ub_schedule(
                two_bucket_instance,
                last_sch_lite.get_jik_2_start_time_map(),
                last_sch_lite.get_jik_2_end_time_map(),
            )
        if tl_nc_multiplier is not None:
            _time_limit_sec = (
                float(tl_nc_multiplier)
                * float(instance.job_count)
                * float(instance.stage_count)
            )
        elif time_limit_sec is None and time_limit_n_by_c_multiplier is not None:
            _time_limit_sec = (
                float(instance.job_count)
                * float(instance.stage_count)
                * float(time_limit_n_by_c_multiplier)
            )
        else:
            _time_limit_sec = time_limit_sec
        _time_limit_sec = self.get_remaining_time_limit(_time_limit_sec)
        logging.info(
            "[MIP LB] Starting MIP solver at %.1f with delta=%d strengthening=%s precedence=%s "
            "threads=%d time_limit_sec=%s",
            start_t,
            delta,
            strengthening,
            precedence,
            threads,
            f"{_time_limit_sec:.1f}" if _time_limit_sec is not None else "None",
        )
        result, trace_rows, _, solution_payload = run_bucket_search_for_instance(
            gp,
            grb,
            two_bucket_instance,
            record,
            delta=delta,
            threads=threads,
            time_limit_sec=_time_limit_sec,
            log_to_console=log_to_console,
            display_interval_sec=display_interval_sec or 10,
            log_dir=None,
            search_upper_t=None,
            strengthening=strengthening,
            precedence=precedence,
            time_limit_sec_used=_time_limit_sec,
            ub_schedule=ub_schedule,
        )

        self.last_mip_lb_result = result
        self.last_mip_lb_solution_payload = solution_payload
        self.last_mip_lb_dispatch_window_lookup = (
            build_dispatch_window_lookup(solution_payload["dispatch_windows"])
            if solution_payload is not None
            else None
        )

        global_incumbent_ub_to_beat = self.solution_manager.best_obj_value
        mip_dispatch_cmax = (
            float(solution_payload["metadata"]["dispatch_cmax"])
            if solution_payload is not None
            and solution_payload.get("metadata", {}).get("dispatch_cmax") is not None
            else result.horizon_ub
        )
        logging.info(
            "[MIP LB] Global incumbent schedule UB to beat is %s",
            global_incumbent_ub_to_beat,
        )
        logging.info(
            "[MIP LB] MIP-based dispatch window Cmax is %s",
            mip_dispatch_cmax,
        )

        post_dispatch_timer = ElapsedTimer()
        dispatched_schedule, selected_dispatch_variant = (
            self._run_post_mip_dispatch_from_last_mip_lb_solution(
                es_ls_local_repair_max_passes=es_ls_local_repair_max_passes
            )
        )
        self.last_mip_lb_post_dispatch_elapsed_sec = post_dispatch_timer.elapsed_sec
        if result.status_name == "OPTIMAL":
            report_status = CpsatStatus.OPTIMAL
        elif result.status_name in {"INFEASIBLE", "INF_OR_UNBD"}:
            report_status = CpsatStatus.INFEASIBLE
        elif result.solution_count > 0:
            report_status = CpsatStatus.FEASIBLE
        else:
            report_status = CpsatStatus.UNKNOWN
        obj_bound_records: list[tuple[float, float]] = []
        lb_before = self.solution_manager.best_obj_bound
        for runtime_sec, objective_ub, objective_lb in trace_rows:
            if (
                objective_lb is not None
                and self.solution_manager._a_is_better_obj_bound(
                    objective_lb, lb_before
                )
            ):
                obj_bound_records.append((runtime_sec, objective_lb))
                lb_before = objective_lb
        logging.info("ObjBound trace: %s", obj_bound_records)
        new_lb = result.certified_final_lb
        current_bound = self.solution_manager.best_obj_bound or float("-inf")
        for mip_time, trace_lb in obj_bound_records:
            if trace_lb is not None and trace_lb > current_bound:
                global_time = start_t + mip_time
                logging.info(
                    f"[MIP LB] Recording intermediate bound {trace_lb} at global time {global_time:.2f}s "
                    f"(MIP time: {mip_time:.2f}s)"
                )
                self.add_obj_bound_log(global_time, trace_lb, is_maximize=False)
                current_bound = trace_lb

        if self.solution_manager.current_obj_bound_is_worse_than(new_lb):
            logging.info(
                f"[MIP LB] Final bound {new_lb} improves"
                f" over current bound {self.solution_manager.best_obj_bound}"
            )
        else:
            logging.info(
                f"[MIP LB] Final bound {new_lb} does not improve"
                f" over current bound {self.solution_manager.best_obj_bound}"
            )
        self.add_obj_bound_log(self.timer.elapsed_sec, new_lb, is_maximize=None)
        report = HfsCpsatSolverReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=(
                float(dispatched_schedule.makespan)
                if dispatched_schedule is not None
                else None
            ),
            obj_bound=new_lb,
            is_init=False,
            subroutine_name="apply_mip_lb",
            call_context=self._get_call_context_of_current_method(),
            progress_obj_value_records=(),
            progress_time_basis="local",
            status=report_status,
            obj_value_records=(),
            obj_bound_records=obj_bound_records,
        )
        was_updated = self.solution_manager.register(report, dispatched_schedule)
        if dispatched_schedule is not None:
            self.add_obj_value_log(
                self.timer.elapsed_sec,
                float(dispatched_schedule.makespan),
                is_maximize=False,
            )
            if was_updated:
                logging.info(
                    "[MIP LB] ES/LS-guided dispatched schedule improved incumbent to makespan=%s",
                    dispatched_schedule.makespan,
                )

        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=dispatched_schedule is not None,
            obj_bound_is_valid=True,
        )
        logging.info(
            "[MIP LB] Global incumbent schedule UB after apply_mip_lb is %s",
            self.solution_manager.best_obj_value,
        )
        self.last_mip_lb_apply_elapsed_sec = sub_timer.elapsed_sec
        if self._working_dir_path is not None and solution_payload is not None:
            mip_lb_output_dir = self._working_dir_path / "mip_lb"
            write_solution_payload(mip_lb_output_dir, solution_payload)
            if draw_mip_lb_visualizations:
                write_solution_payload_visualizations(
                    mip_lb_output_dir, solution_payload
                )
            else:
                logging.info(
                    "[MIP LB] Skipping bucket-MIP/post-dispatch plot export because draw_mip_lb_visualizations=false."
                )
            self._write_mip_lb_dispatch_artifacts(
                dispatched_schedule=dispatched_schedule,
                selected_variant=selected_dispatch_variant,
                stage_2_job_sequence=self.last_mip_lb_stage_2_job_sequence,
                stage_2_job_release=self.last_mip_lb_stage_2_job_release,
                mip_result=result,
                draw_visualizations=draw_mip_lb_visualizations,
            )
            logging.info(
                "[MIP LB] Persisted solution payload and dispatch artifacts to %s",
                mip_lb_output_dir,
            )
        return {
            "result": result,
            "solution_payload": solution_payload,
            "dispatch_window_lookup": self.last_mip_lb_dispatch_window_lookup,
            "dispatch_candidates": self.last_mip_lb_dispatched_schedules,
            "dispatched_schedule": dispatched_schedule,
            "selected_dispatch_variant": selected_dispatch_variant,
        }

    def dispatch_from_saved_mip_lb(
        self,
        *,
        mip_lb_dir: str | None = None,
        es_ls_local_repair_max_passes: int = 3,
        draw_mip_lb_visualizations: bool = True,
    ) -> dict[str, Any] | None:
        """Reload a saved MIP-LB payload and rerun only the post-MIP dispatch logic."""
        sub_timer = ElapsedTimer()
        self.last_mip_lb_apply_elapsed_sec = None
        self.last_mip_lb_post_dispatch_elapsed_sec = None
        if mip_lb_dir is not None:
            source_dir = Path(mip_lb_dir)
        elif self._working_dir_path is not None:
            source_dir = self._working_dir_path / "mip_lb"
        else:
            logging.warning(
                "[MIP LB] No working directory is available to load a saved MIP payload."
            )
            return None

        solution_payload = read_solution_payload(source_dir, self.instance.name)
        if solution_payload is None:
            logging.warning(
                "[MIP LB] No saved MIP payload was found for instance %s under %s.",
                self.instance.name,
                source_dir,
            )
            return None

        self.last_mip_lb_result = None
        self.last_mip_lb_solution_payload = solution_payload
        self.last_mip_lb_dispatch_window_lookup = build_dispatch_window_lookup(
            solution_payload["dispatch_windows"]
        )

        global_incumbent_ub_to_beat = self.solution_manager.best_obj_value
        mip_dispatch_cmax = solution_payload.get("metadata", {}).get("dispatch_cmax")
        logging.info(
            "[MIP LB] Global incumbent schedule UB to beat is %s",
            global_incumbent_ub_to_beat,
        )
        logging.info(
            "[MIP LB] MIP-based dispatch window Cmax is %s",
            mip_dispatch_cmax,
        )

        post_dispatch_timer = ElapsedTimer()
        dispatched_schedule, selected_dispatch_variant = (
            self._run_post_mip_dispatch_from_last_mip_lb_solution(
                es_ls_local_repair_max_passes=es_ls_local_repair_max_passes
            )
        )
        self.last_mip_lb_post_dispatch_elapsed_sec = post_dispatch_timer.elapsed_sec

        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=(
                float(dispatched_schedule.makespan)
                if dispatched_schedule is not None
                else None
            ),
            obj_bound=None,
            is_init=False,
        )
        was_updated = self.solution_manager.register(report, dispatched_schedule)
        if dispatched_schedule is not None:
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(
                log_time, float(dispatched_schedule.makespan), is_maximize=False
            )
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
            )
            if was_updated:
                logging.info(
                    "[MIP LB] Saved-payload dispatch improved incumbent to makespan=%s",
                    dispatched_schedule.makespan,
                )
        logging.info(
            "[MIP LB] Global incumbent schedule UB after dispatch_from_saved_mip_lb is %s",
            self.solution_manager.best_obj_value,
        )
        self.last_mip_lb_apply_elapsed_sec = sub_timer.elapsed_sec
        if self._working_dir_path is not None:
            mip_lb_output_dir = self._working_dir_path / "mip_lb"
            if source_dir != mip_lb_output_dir:
                write_solution_payload(mip_lb_output_dir, solution_payload)
                logging.info(
                    "[MIP LB] Copied saved solution payload with dispatch windows to %s",
                    mip_lb_output_dir,
                )
            if draw_mip_lb_visualizations:
                write_solution_payload_visualizations(
                    mip_lb_output_dir, solution_payload
                )
            else:
                logging.info(
                    "[MIP LB] Skipping bucket-MIP/post-dispatch plot export because draw_mip_lb_visualizations=false."
                )
            self._write_mip_lb_dispatch_artifacts(
                dispatched_schedule=dispatched_schedule,
                selected_variant=selected_dispatch_variant,
                stage_2_job_sequence=self.last_mip_lb_stage_2_job_sequence,
                stage_2_job_release=self.last_mip_lb_stage_2_job_release,
                mip_result=None,
                draw_visualizations=draw_mip_lb_visualizations,
            )
        return {
            "solution_payload": solution_payload,
            "dispatch_window_lookup": self.last_mip_lb_dispatch_window_lookup,
            "dispatch_candidates": self.last_mip_lb_dispatched_schedules,
            "dispatched_schedule": dispatched_schedule,
            "selected_dispatch_variant": selected_dispatch_variant,
        }

    def _run_post_mip_dispatch_from_last_mip_lb_solution(
        self,
        *,
        es_ls_local_repair_max_passes: int,
    ) -> tuple[HybridFlowshopLiteSchedule | None, str | None]:
        dispatch_result = run_post_mip_dispatch(
            instance=self.instance,
            stage_2_job_2_p_dict=self.stage_2_job_2_p_dict,
            solution_payload=self.last_mip_lb_solution_payload,
            dispatch_window_lookup=self.last_mip_lb_dispatch_window_lookup,
            es_ls_local_repair_max_passes=es_ls_local_repair_max_passes,
            dependencies=PostMipDispatchDependencies(
                check_feasibility=self.check_feasibility,
                get_selected_dispatch_config=self._get_selected_dispatch_config_for_post_mip,
                get_best_mixed_schedule_from_job_sequence=self._get_best_mixed_schedule_from_job_sequence,
                get_selected_dispatch_candidate_schedules=self._get_selected_dispatch_candidate_schedules,
                get_schedule_by_best_of_mixed_dispatches=self._get_schedule_by_best_of_mixed_dispatches,
                repair_post_mip_dispatch_candidate=self._repair_post_mip_dispatch_candidate,
            ),
        )

        self.last_mip_lb_stage_2_job_sequence = dispatch_result.stage_2_job_sequence
        self.last_mip_lb_stage_2_job_release = dispatch_result.stage_2_job_release
        self.last_mip_lb_selected_dispatch_variant = (
            dispatch_result.selected_dispatch_variant
        )
        self.last_mip_lb_dispatched_schedules = dispatch_result.dispatched_schedules
        self.last_mip_lb_dispatch_candidate_elapsed_sec = (
            dispatch_result.dispatch_candidate_elapsed_sec
        )
        self.last_mip_lb_dispatch_phase_elapsed_sec = (
            dispatch_result.dispatch_phase_elapsed_sec
        )
        self.last_mip_lb_pre_local_repair_selected_dispatch_variant = (
            dispatch_result.pre_local_repair_selected_dispatch_variant
        )
        self.last_mip_lb_pre_local_repair_selected_dispatch_makespan = (
            dispatch_result.pre_local_repair_selected_dispatch_makespan
        )
        return (
            dispatch_result.dispatched_schedule,
            dispatch_result.selected_dispatch_variant,
        )

    def _write_mip_lb_dispatch_artifacts(
        self,
        *,
        dispatched_schedule: HybridFlowshopLiteSchedule | None,
        selected_variant: str | None,
        stage_2_job_sequence: Mapping[str, Sequence[str]] | None,
        stage_2_job_release: Mapping[str, Mapping[str, int]] | None,
        mip_result: Any | None,
        draw_visualizations: bool,
    ) -> None:
        """Write ES/LS-guided dispatch outputs under the instance mip_lb directory."""
        if self._working_dir_path is None:
            return

        write_post_mip_dispatch_artifacts(
            output_dir=self._working_dir_path / "mip_lb",
            dispatch_result=PostMipDispatchRunResult(
                dispatched_schedule=dispatched_schedule,
                selected_dispatch_variant=selected_variant,
                dispatched_schedules=dict(self.last_mip_lb_dispatched_schedules or {}),
                dispatch_candidate_elapsed_sec=dict(
                    self.last_mip_lb_dispatch_candidate_elapsed_sec or {}
                ),
                dispatch_phase_elapsed_sec=dict(
                    self.last_mip_lb_dispatch_phase_elapsed_sec or {}
                ),
                stage_2_job_sequence=stage_2_job_sequence,
                stage_2_job_release=stage_2_job_release,
                pre_local_repair_selected_dispatch_variant=(
                    self.last_mip_lb_pre_local_repair_selected_dispatch_variant
                ),
                pre_local_repair_selected_dispatch_makespan=(
                    self.last_mip_lb_pre_local_repair_selected_dispatch_makespan
                ),
            ),
            solution_payload=self.last_mip_lb_solution_payload,
            mip_result=mip_result,
            apply_mip_lb_elapsed_sec=self.last_mip_lb_apply_elapsed_sec,
            post_mip_dispatch_elapsed_sec=self.last_mip_lb_post_dispatch_elapsed_sec,
            draw_visualizations=draw_visualizations,
        )

    def _repair_post_mip_dispatch_candidate(
        self,
        schedule: HybridFlowshopLiteSchedule | None,
        *,
        target_stage_ids: Sequence[str] | None = None,
        insertion_passes: int = 3,
        max_shift: int = 4,
        swap_passes: int = 3,
        stage_2_job_2_release: Mapping[str, Mapping[str, int]] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        if schedule is None:
            return None

        candidate_pool: list[HybridFlowshopLiteSchedule] = [schedule]

        try:
            inserted = improve_schedule_by_critical_stage_sequence_insertions(
                self.create_empty_schedule_from_ins,
                schedule,
                self.stage_2_job_2_p_dict,
                target_stage_ids=target_stage_ids,
                max_passes=max(1, insertion_passes),
                max_shift=max(1, max_shift),
                stage_2_job_2_release=stage_2_job_2_release,
            )
            candidate_pool.append(inserted)
        except Exception:
            logging.exception(
                "[MIP LB] Stage-sequence insertion repair failed for a post-MIP dispatch candidate."
            )
            inserted = None

        try:
            reassigned = improve_schedule_by_critical_cross_machine_insertions(
                schedule,
                self.stage_2_job_2_p_dict,
                target_stage_ids=target_stage_ids,
                max_passes=max(1, insertion_passes),
                stage_2_job_2_release=stage_2_job_2_release,
            )
            candidate_pool.append(reassigned)
        except Exception:
            logging.exception(
                "[MIP LB] Cross-machine critical-op repair failed for a post-MIP dispatch candidate."
            )
            reassigned = None

        try:
            swapped = improve_schedule_by_critical_adjacent_swaps(
                schedule,
                self.stage_2_job_2_p_dict,
                max_passes=max(1, swap_passes),
                stage_2_job_2_release=stage_2_job_2_release,
            )
            candidate_pool.append(swapped)
        except Exception:
            logging.exception(
                "[MIP LB] Adjacent-swap repair failed for a post-MIP dispatch candidate."
            )

        if inserted is not None:
            try:
                inserted_swapped = improve_schedule_by_critical_adjacent_swaps(
                    inserted,
                    self.stage_2_job_2_p_dict,
                    max_passes=max(1, swap_passes),
                    stage_2_job_2_release=stage_2_job_2_release,
                )
                candidate_pool.append(inserted_swapped)
            except Exception:
                logging.exception(
                    "[MIP LB] Adjacent-swap after insertion repair failed for a post-MIP dispatch candidate."
                )

        if reassigned is not None:
            try:
                reassigned_swapped = improve_schedule_by_critical_adjacent_swaps(
                    reassigned,
                    self.stage_2_job_2_p_dict,
                    max_passes=max(1, swap_passes),
                    stage_2_job_2_release=stage_2_job_2_release,
                )
                candidate_pool.append(reassigned_swapped)
            except Exception:
                logging.exception(
                    "[MIP LB] Adjacent-swap after cross-machine repair failed for a post-MIP dispatch candidate."
                )

        best_schedule = min(candidate_pool, key=lambda sch: sch.makespan)
        return best_schedule

    def initialize_by_dj_cds(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the Campbell-Dudek-Smith (CDS) sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dj_cds()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_dj_cds",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_dj_cds(self) -> HybridFlowshopLiteSchedule:
        dispatcher = JobDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_dj_cds()
        if schedule is None:
            raise ValueError("No schedule found after applying DJ(CDS).")
        logging.info(f"Schedule by DJ(CDS): makespan={schedule.makespan}")
        return schedule

    def initialize_by_dj_gupta(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the gupta sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dj_gupta()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_dj_gupta",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_dj_gupta(self) -> HybridFlowshopLiteSchedule:
        dispatcher = JobDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_dj_gupta()
        if schedule is None:
            raise ValueError("No schedule found after applying DJ(Gupta).")
        logging.info(f"Schedule by DJ(Gupta): makespan={schedule.makespan}")
        return schedule

    def initialize_by_dj_palmer(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the gupta sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dj_palmer()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_dj_palmer",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_dj_palmer(self) -> HybridFlowshopLiteSchedule:
        dispatcher = JobDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_dj_palmer()
        if schedule is None:
            raise ValueError("No schedule found after applying DJ(Palmer).")
        logging.info(f"Schedule by DJ(Palmer): makespan={schedule.makespan}")
        return schedule

    def initialize_by_ds_cds(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the Campbell-Dudek-Smith (CDS) sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_ds_cds()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_ds_cds",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_ds_cds(self) -> HybridFlowshopLiteSchedule:
        dispatcher = StageDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_ds_cds()
        if schedule is None:
            raise ValueError("No schedule found after applying DS(CDS).")
        logging.info(f"Schedule by DS(CDS): makespan={schedule.makespan}")
        return schedule

    def initialize_by_ds_gupta(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the two-partition sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_ds_gupta()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_ds_gupta",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_ds_gupta(self) -> HybridFlowshopLiteSchedule:
        dispatcher = StageDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_ds_gupta()
        if schedule is None:
            raise ValueError("No schedule found after applying DS(Gupta).")
        logging.info(f"Schedule by DS(Gupta): makespan={schedule.makespan}")
        return schedule

    def initialize_by_ds_palmer(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses the two-partition sequence as the job sequence
        & dispatches by job - stage - time priority to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of the solution.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_ds_palmer()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_ds_palmer",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_ds_palmer(self) -> HybridFlowshopLiteSchedule:
        dispatcher = StageDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_ds_palmer()
        if schedule is None:
            raise ValueError("No schedule found after applying DS(Palmer).")
        logging.info(f"Schedule by DS(Palmer): makespan={schedule.makespan}")
        return schedule

    def initialize_by_dm_cds(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses CDS sequence & dispatches by stage - machine - job priority
        to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of
                the solution. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the
                solution. Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dm_cds()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_dm_cds",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_dm_cds(self) -> HybridFlowshopLiteSchedule:
        dispatcher = MachineDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_cds()
        if schedule is not None:
            logging.info(f"Schedule by DM(CDS): makespan={schedule.makespan}")

        reversed_dispatcher = MachineDispatcher(reverse_stages(self.instance))
        reversed_schedule = reversed_dispatcher.get_schedule_by_cds()
        if reversed_schedule is not None:
            logging.info(
                "Schedule by DM(CDS) on reversed instance"
                f": makespan={reversed_schedule.makespan}"
            )

        if reversed_schedule is not None and (
            schedule is None or schedule.makespan > reversed_schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(self.stage_2_job_2_p_dict)
            return converted
        if schedule is None:
            raise ValueError("No schedule found after applying DM(CDS).")
        return schedule

    def initialize_by_dm_gupta(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses Gupta sequence & dispatches by stage - machine - job priority
        to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of
                the solution. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the
                solution. Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dm_gupta()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_dm_gupta",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_dm_gupta(self) -> HybridFlowshopLiteSchedule:
        dispatcher = MachineDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_gupta()
        if schedule is not None:
            logging.info(f"Schedule by DM(Gupta): makespan={schedule.makespan}")

        reversed_dispatcher = MachineDispatcher(reverse_stages(self.instance))
        reversed_schedule = reversed_dispatcher.get_schedule_by_gupta()
        if reversed_schedule is not None:
            logging.info(
                "Schedule by DM(Gupta) on reversed instance"
                f": makespan={reversed_schedule.makespan}"
            )

        if reversed_schedule is not None and (
            schedule is None or schedule.makespan > reversed_schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(self.stage_2_job_2_p_dict)
            return converted
        if schedule is None:
            raise ValueError("No schedule found after applying DM(Gupta).")
        return schedule

    def initialize_by_dm_palmer(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        Uses Palmer sequence & dispatches by stage - machine - job priority
        to initialize a schedule.

        Args:
            error_if_infeasible (bool, optional): If True, checks the feasibility of
                the solution. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the
                solution. Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dm_palmer()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_dm_palmer",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_dm_palmer(self) -> HybridFlowshopLiteSchedule:
        dispatcher = MachineDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_palmer()
        if schedule is not None:
            logging.info(f"Schedule by DM(Palmer): makespan={schedule.makespan}")
            try:
                out_path = self.get_file_path_for_subroutine("_gantt_dm_palmer.png")
                self.draw_gantt(schedule, output_path=out_path)
            except Exception:
                logging.exception("Failed to draw Gantt for DM(Palmer) schedule")

        reversed_dispatcher = MachineDispatcher(reverse_stages(self.instance))
        reversed_schedule = reversed_dispatcher.get_schedule_by_palmer()
        if reversed_schedule is not None:
            logging.info(
                "Schedule by DM(Palmer) on reversed instance"
                f": makespan={reversed_schedule.makespan}"
            )
            try:
                out_path = self.get_file_path_for_subroutine(
                    "_gantt_dm_palmer_reversed.png"
                )
                self.draw_gantt(
                    reversed_schedule,
                    output_path=out_path,
                    stage_list=reversed_schedule.stages,
                )
            except Exception:
                logging.exception(
                    "Failed to draw Gantt for DM(Palmer) reversed schedule"
                )

        if reversed_schedule is not None and (
            schedule is None or schedule.makespan > reversed_schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(self.stage_2_job_2_p_dict)
            return converted
        if schedule is None:
            raise ValueError("No schedule found after applying DM(Palmer).")
        return schedule

    def initialize_by_best_of_dispatches(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        sub_timer = ElapsedTimer()

        schedule = self._get_best_of_dispatches()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_best_of_dispatches",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_best_of_dispatches(self) -> HybridFlowshopLiteSchedule:
        schedule_gen_methods = [
            self._get_schedule_by_dj_cds,
            self._get_schedule_by_dj_gupta,
            self._get_schedule_by_dj_palmer,
            self._get_schedule_by_ds_cds,
            self._get_schedule_by_ds_gupta,
            self._get_schedule_by_ds_palmer,
        ]
        # Subroutine states
        best_makespan = float("inf")
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_method_name = ""

        for method in schedule_gen_methods:
            schedule = method()
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_method_name = method.__name__

        if best_schedule is None:
            raise ValueError("No schedule found after applying dispatching heuristics.")
        logging.info(
            f"Best schedule found by {best_method_name} with makespan {best_makespan}"
        )
        return best_schedule

    def generate_from_dispatches(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        sub_timer = ElapsedTimer()

        schedule = self._get_generate_from_dispatches()
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        # Create report and register the new solution
        obj_value = float(schedule.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="generate_from_dispatches",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_generate_from_dispatches(self) -> HybridFlowshopLiteSchedule:
        import heapq

        # Initial sequences
        job_sequences: set[tuple[str, ...]] = {
            tuple(self.get_gupta_sequence()),
            tuple(self.get_palmer_sequence()),
        }
        for k, stage_id in enumerate(self.instance.stage_id_list):
            job_sequences.add(tuple(self.get_cds_sequence(k + 1)))
            # job_sequences.add(tuple(self.get_bnd_sequence(stage_id)))

        # Subroutine states
        best_makespan = float("inf")
        best_schedule: HybridFlowshopLiteSchedule | None = None

        # seq_2_solution: dict[tuple[str, ...], HybridFlowshopLiteSchedule] = {}
        # seq_2_obj_val: dict[tuple[str, ...], float] = {}
        # Use a dict for sequence lookup and a max-heap (via negative values) to track worst values
        seq_2_obj_val: dict[tuple[str, ...], float] = {}
        # Heap of (-makespan, sequence) to efficiently find and remove worst entries
        obj_seq_heap: list[tuple[float, tuple[str, ...]]] = []

        MAX_SEQUENCES = 500

        def get_current_worst() -> tuple[float, tuple[str, ...]] | None:
            """Return current valid worst entry in seq_2_obj_val (max makespan)."""
            while obj_seq_heap:
                neg_obj, seq = obj_seq_heap[0]
                obj = -neg_obj
                current_obj = seq_2_obj_val.get(seq)
                if current_obj is None or current_obj != obj:
                    heapq.heappop(obj_seq_heap)
                    continue
                return obj, seq
            return None

        def remove_current_worst() -> tuple[float, tuple[str, ...]] | None:
            """Remove and return current valid worst entry from seq_2_obj_val."""
            while obj_seq_heap:
                neg_obj, seq = heapq.heappop(obj_seq_heap)
                obj = -neg_obj
                current_obj = seq_2_obj_val.get(seq)
                if current_obj is None or current_obj != obj:
                    continue
                del seq_2_obj_val[seq]
                return obj, seq
            return None

        # Initial schedules
        for sequence in job_sequences:
            schedule = self._from_job_sequence_get_schedule(sequence)
            # seq_2_solution[sequence] = schedule
            makespan = schedule.makespan
            seq_2_obj_val[sequence] = makespan
            heapq.heappush(obj_seq_heap, (-makespan, sequence))
            if len(seq_2_obj_val) > MAX_SEQUENCES:
                remove_current_worst()
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule

        logging.info(
            f"(Best,worst) initial schedule found by dispatching heuristics with makespan={best_makespan}, {max(seq_2_obj_val.values()) if seq_2_obj_val else float('inf')}"
        )

        trial_cnt = 0
        best_grow_makespan = float("inf")
        while True:
            trial_cnt += 1
            obj_val_avg: float = sum(seq_2_obj_val.values()) / len(seq_2_obj_val)
            new_sequence = self._from_seq_2_obj_val_get_new_sequence(seq_2_obj_val)
            new_seq_tuple = tuple(new_sequence)
            if new_seq_tuple in seq_2_obj_val:
                logging.info(
                    f"Trial {trial_cnt} New sequence already evaluated. Stopping growth from dispatches."
                )
                break
            schedule = self._from_job_sequence_get_schedule(new_sequence)
            makespan = schedule.makespan
            logging.info(
                f"Trial {trial_cnt} Before average {obj_val_avg:.1f} New makespan {makespan}"
            )
            # Keep only top-K sequences with the smallest makespans
            if len(seq_2_obj_val) < MAX_SEQUENCES:
                seq_2_obj_val[new_seq_tuple] = makespan
                heapq.heappush(obj_seq_heap, (-makespan, new_seq_tuple))
            else:
                worst_info = get_current_worst()
                if worst_info is None:
                    seq_2_obj_val[new_seq_tuple] = makespan
                    heapq.heappush(obj_seq_heap, (-makespan, new_seq_tuple))
                else:
                    worst_obj, worst_seq = worst_info
                    if makespan < worst_obj:
                        del seq_2_obj_val[worst_seq]
                        seq_2_obj_val[new_seq_tuple] = makespan
                        heapq.heappush(obj_seq_heap, (-makespan, new_seq_tuple))
                    else:
                        logging.info(
                            f"Trial {trial_cnt} New makespan {makespan} is not better than current worst {worst_obj:.1f}. Stopping growth from dispatches."
                        )
                        break

            # Update best solution
            if best_grow_makespan > makespan:
                best_grow_makespan = makespan
            if best_makespan > makespan:
                logging.info(
                    f"Found better schedule by growing from dispatches (previous best={best_makespan})"
                )
                best_makespan = makespan
                best_schedule = schedule
        logging.info(
            f"Best makespan found by growing from dispatches: {best_grow_makespan}"
        )

        if best_schedule is None:
            raise ValueError("No schedule found after applying dispatching heuristics.")
        logging.info(
            f"Best schedule found by generate_from_dispatches with makespan={best_makespan}"
        )
        return best_schedule

    def _from_seq_2_obj_val_get_new_sequence(
        self, seq_2_obj_val: dict[tuple[str, ...], float]
    ) -> list[str]:
        obj_val_avg: float = sum(seq_2_obj_val.values()) / len(seq_2_obj_val)
        # logging.info(f"Average makespan of current sequences: {obj_val_avg:.2f}")
        seq_2_obj_val_diff: dict[tuple[str, ...], float] = {
            seq: obj_val - obj_val_avg for seq, obj_val in seq_2_obj_val.items()
        }
        # x = x^2 if x>0 else -(-x)^2
        # seq_2_obj_val_diff = {
        #     seq: diff if diff <= 0 else diff**2
        #     for seq, diff in seq_2_obj_val_diff.items()
        # }
        job_2_vop: dict[str, float] = {j: 0.0 for j in self.instance.job_id_list}
        for seq, obj_val_diff in seq_2_obj_val_diff.items():
            for pos, j in enumerate(seq):
                job_2_vop[j] += obj_val_diff * pos

        # Sort jobs by vop in descending order to get a new job sequence
        return sorted(
            self.instance.job_id_list, key=lambda j: job_2_vop[j], reverse=True
        )

    def _from_job_sequence_get_schedule(
        self, job_sequence: Sequence[str]
    ) -> HybridFlowshopLiteSchedule:
        job_dispatched_schedule = self.create_empty_schedule_from_ins()
        for j in job_sequence:
            job_dispatched_schedule.dispatch_job_by_stages(
                j, self.job_2_stage_2_p_dict[j]
            )
        job_dispatched_obj_value = job_dispatched_schedule.makespan
        stage_dispatched_schedule = self.create_empty_schedule_from_ins()
        for i in self.instance.stage_id_list:
            stage_dispatched_schedule.dispatch_stage_by_jobs(
                i, job_sequence, self.stage_2_job_2_p_dict[i]
            )
        stage_dispatched_obj_value = stage_dispatched_schedule.makespan

        if job_dispatched_obj_value < stage_dispatched_obj_value:
            return job_dispatched_schedule
        return stage_dispatched_schedule

    @staticmethod
    def _sequence_position_difference_ratio(
        seq_a: Sequence[str],
        seq_b: Sequence[str],
    ) -> float:
        if not seq_a or not seq_b:
            return 1.0
        pos_b = {job_id: idx for idx, job_id in enumerate(seq_b)}
        common = [job_id for job_id in seq_a if job_id in pos_b]
        if not common:
            return 1.0
        different_position_count = sum(
            1 for idx, job_id in enumerate(common) if pos_b[job_id] != idx
        )
        return different_position_count / max(len(seq_a), len(seq_b))

    @staticmethod
    def _get_neh_reference_sequence(
        schedule: HybridFlowshopLiteSchedule,
        *,
        job_seq_by_1st_stage: bool,
        job_seq_by_bottleneck_stage: bool,
    ) -> list[str]:
        if job_seq_by_1st_stage:
            return get_first_stage_start_sequence(schedule)
        if job_seq_by_bottleneck_stage:
            return get_bottleneck_stage_job_sequence(schedule)
        return get_midpoint_sequence(schedule)

    def _get_neh_beam_source_candidates(
        self,
        *,
        candidate_source: str,
        include_incumbent_candidate: bool,
    ) -> dict[str, HybridFlowshopLiteSchedule]:
        candidate_schedules: dict[str, HybridFlowshopLiteSchedule] = {}
        if candidate_source == "retained_cp_dispatch":
            for label, schedule in (
                getattr(self, "last_retained_cp_dispatch_candidate_schedules", None)
                or {}
            ).items():
                if schedule is not None:
                    candidate_schedules[str(label)] = schedule
        elif candidate_source == "incumbent":
            pass
        else:
            raise ValueError(
                "candidate_source must be one of {'retained_cp_dispatch', 'incumbent'}."
            )

        incumbent = self.solution_manager.get_incumbent()
        if include_incumbent_candidate and incumbent is not None:
            candidate_schedules["current_incumbent"] = incumbent
        return candidate_schedules

    def _select_sequence_diverse_neh_beam_candidates(
        self,
        *,
        candidate_schedules: Mapping[str, HybridFlowshopLiteSchedule],
        candidate_top_k: int,
        candidate_pool_top_k: int | None,
        max_candidate_obj_slack: float | None,
        min_sequence_position_diff_ratio: float,
        allow_sequence_duplicate_fallback: bool,
        job_seq_by_1st_stage: bool,
        job_seq_by_bottleneck_stage: bool,
    ) -> list[tuple[str, HybridFlowshopLiteSchedule, list[str]]]:
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive.")
        if candidate_pool_top_k is not None and candidate_pool_top_k <= 0:
            raise ValueError("candidate_pool_top_k must be positive when provided.")
        if max_candidate_obj_slack is not None and max_candidate_obj_slack < 0:
            raise ValueError("max_candidate_obj_slack must be non-negative.")
        if min_sequence_position_diff_ratio < 0:
            raise ValueError("min_sequence_position_diff_ratio must be non-negative.")

        sorted_candidates = sorted(
            (
                (label, schedule)
                for label, schedule in candidate_schedules.items()
                if schedule is not None
            ),
            key=lambda item: (float(item[1].makespan), item[0]),
        )
        if candidate_pool_top_k is not None:
            sorted_candidates = sorted_candidates[:candidate_pool_top_k]
        if not sorted_candidates:
            return []

        best_candidate_obj = float(sorted_candidates[0][1].makespan)
        if max_candidate_obj_slack is not None:
            allowed_obj = best_candidate_obj + float(max_candidate_obj_slack)
            sorted_candidates = [
                item
                for item in sorted_candidates
                if float(item[1].makespan) <= allowed_obj
            ]

        selected: list[tuple[str, HybridFlowshopLiteSchedule, list[str]]] = []
        duplicate_fallbacks: list[
            tuple[str, HybridFlowshopLiteSchedule, list[str]]
        ] = []
        for label, schedule in sorted_candidates:
            sequence = self._get_neh_reference_sequence(
                schedule,
                job_seq_by_1st_stage=job_seq_by_1st_stage,
                job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
            )
            if not sequence:
                logging.info(
                    "[NEH Beam] Skipping candidate %s because its sequence is empty.",
                    label,
                )
                continue
            if not selected:
                selected.append((label, schedule, sequence))
            else:
                min_diff = min(
                    self._sequence_position_difference_ratio(sequence, selected_seq)
                    for _, _, selected_seq in selected
                )
                if min_diff >= min_sequence_position_diff_ratio:
                    selected.append((label, schedule, sequence))
                else:
                    duplicate_fallbacks.append((label, schedule, sequence))
            if len(selected) >= candidate_top_k:
                break

        if allow_sequence_duplicate_fallback and len(selected) < candidate_top_k:
            selected_labels = {label for label, _, _ in selected}
            for label, schedule, sequence in duplicate_fallbacks:
                if label in selected_labels:
                    continue
                selected.append((label, schedule, sequence))
                selected_labels.add(label)
                if len(selected) >= candidate_top_k:
                    break
        return selected

    def neh_cp_sequence_beam(
        self,
        solver_thread_cnt: int,
        candidate_source: str = "retained_cp_dispatch",
        candidate_top_k: int = 2,
        candidate_pool_top_k: int | None = 16,
        max_candidate_obj_slack: float | None = None,
        min_sequence_position_diff_ratio: float = 0.08,
        allow_sequence_duplicate_fallback: bool = False,
        include_incumbent_candidate: bool = True,
        added_batch_size: int = 1,
        added_batch_count: int | None = None,
        min_added_batch_count: int | None = None,
        max_added_batch_count: int | None = None,
        job_seq_by_1st_stage: bool = False,
        job_seq_by_bottleneck_stage: bool = False,
        preserved_head_job_portion: float = 0.0,
        max_time_per_add: float | None = None,
        cp_tl_nc_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        profile_fix_min_batch_idx: int = 2,
        profile_fix_min_batch_portion: float | None = None,
        profile_fix_max_batch_idx: int | None = None,
        profile_fix_by_machine_from_batch_idx: int | None = None,
        profile_fix_min_job_age_batches: int | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        minimize_sum_ci_lex: bool = False,
        cp_tl_nc_multiplier_2nd_obj: float | None = None,
        cp_tl_c_multiplier_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        make_semi_active_every_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        stop_before_final_reserve: bool = True,
        min_remaining_sec_after_neh: float | None = None,
        min_remaining_nc_after_neh: float | None = None,
        time_guard_estimate_safety_factor: float = 1.15,
        time_guard_min_completed_batches: int = 1,
        skip_if_estimated_neh_exceeds_remaining: bool = True,
        full_neh_estimate_safety_factor: float = 1.0,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
    ) -> None:
        """Run full NEH-CP on sequence-diverse candidate schedules and keep the best."""
        sub_timer = ElapsedTimer()
        incumbent = self.solution_manager.get_incumbent()
        if incumbent is None:
            raise ValueError("No incumbent solution available for NEH-CP beam.")
        input_obj = float(incumbent.makespan)

        candidate_schedules = self._get_neh_beam_source_candidates(
            candidate_source=candidate_source,
            include_incumbent_candidate=include_incumbent_candidate,
        )
        selected_candidates = self._select_sequence_diverse_neh_beam_candidates(
            candidate_schedules=candidate_schedules,
            candidate_top_k=candidate_top_k,
            candidate_pool_top_k=candidate_pool_top_k,
            max_candidate_obj_slack=max_candidate_obj_slack,
            min_sequence_position_diff_ratio=min_sequence_position_diff_ratio,
            allow_sequence_duplicate_fallback=allow_sequence_duplicate_fallback,
            job_seq_by_1st_stage=job_seq_by_1st_stage,
            job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
        )
        if not selected_candidates:
            logging.info(
                "[NEH Beam] No retained dispatch candidates were available; "
                "falling back to the current incumbent."
            )
            selected_candidates = [
                (
                    "current_incumbent",
                    incumbent,
                    self._get_neh_reference_sequence(
                        incumbent,
                        job_seq_by_1st_stage=job_seq_by_1st_stage,
                        job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
                    ),
                )
            ]

        logging.info(
            "[NEH Beam] Selected %d/%d candidate schedules: %s",
            len(selected_candidates),
            candidate_top_k,
            [
                {
                    "label": label,
                    "makespan": schedule.makespan,
                    "sequence_head": sequence[:8],
                }
                for label, schedule, sequence in selected_candidates
            ],
        )

        best_schedule = incumbent
        best_obj = input_obj
        best_label = "current_incumbent"
        best_result: NehCpResult | None = None
        candidate_results: list[dict[str, Any]] = []

        obj_value_comparator = getattr(
            self.solution_manager, "_a_is_better_obj_value", None
        )

        def is_better_obj(candidate_obj: float, incumbent_obj: float | None) -> bool:
            if callable(obj_value_comparator):
                return bool(obj_value_comparator(candidate_obj, incumbent_obj))
            return incumbent_obj is None or candidate_obj < incumbent_obj

        for candidate_idx, (label, schedule, sequence) in enumerate(
            selected_candidates, start=1
        ):
            if self.is_stopping_condition():
                logging.info(
                    "[NEH Beam] Stopping before candidate %s because the global "
                    "stopping condition is met.",
                    label,
                )
                break
            logging.info(
                "[NEH Beam] Running candidate %d/%d label=%s initial_makespan=%s "
                "sequence_head=%s",
                candidate_idx,
                len(selected_candidates),
                label,
                schedule.makespan,
                sequence[:12],
            )
            candidate_timer = ElapsedTimer()
            constructor = NehCpConstructor(self)
            result: NehCpResult = constructor.run(
                schedule,
                self.instance,
                self.job_2_stage_2_p_dict,
                self.stage_2_job_2_p_dict,
                added_batch_size=added_batch_size,
                added_batch_count=added_batch_count,
                min_added_batch_count=min_added_batch_count,
                max_added_batch_count=max_added_batch_count,
                job_seq_by_1st_stage=job_seq_by_1st_stage,
                job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
                preserved_head_job_portion=preserved_head_job_portion,
                max_time_per_add=max_time_per_add,
                cp_tl_nc_multiplier=cp_tl_nc_multiplier,
                cp_tl_c_multiplier=cp_tl_c_multiplier,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
                profile_fix_min_batch_idx=profile_fix_min_batch_idx,
                profile_fix_min_batch_portion=profile_fix_min_batch_portion,
                profile_fix_max_batch_idx=profile_fix_max_batch_idx,
                profile_fix_by_machine_from_batch_idx=profile_fix_by_machine_from_batch_idx,
                profile_fix_min_job_age_batches=profile_fix_min_job_age_batches,
                stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
                stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
                minimize_sum_ci_lex=minimize_sum_ci_lex,
                cp_tl_nc_multiplier_2nd_obj=cp_tl_nc_multiplier_2nd_obj,
                cp_tl_c_multiplier_2nd_obj=cp_tl_c_multiplier_2nd_obj,
                minimize_sum_ci_lin=minimize_sum_ci_lin,
                tighten_ranges=tighten_ranges,
                link_job_completion=link_job_completion,
                make_semi_active_every_cp=make_semi_active_every_cp,
                solver_thread_cnt=solver_thread_cnt,
                use_lns_only=use_lns_only,
                error_if_infeasible=error_if_infeasible,
                stop_before_final_reserve=stop_before_final_reserve,
                min_remaining_sec_after_neh=min_remaining_sec_after_neh,
                min_remaining_nc_after_neh=min_remaining_nc_after_neh,
                time_guard_estimate_safety_factor=time_guard_estimate_safety_factor,
                time_guard_min_completed_batches=time_guard_min_completed_batches,
                skip_if_estimated_neh_exceeds_remaining=skip_if_estimated_neh_exceeds_remaining,
                full_neh_estimate_safety_factor=full_neh_estimate_safety_factor,
                log_cp_subproblem_bounds=log_cp_subproblem_bounds,
                log_cp_subproblem_progress=log_cp_subproblem_progress,
            )
            candidate_obj = float(result.schedule.makespan)
            candidate_results.append(
                {
                    "label": label,
                    "input_makespan": float(schedule.makespan),
                    "output_makespan": candidate_obj,
                    "elapsed_sec": candidate_timer.elapsed_sec,
                    "sequence_head": sequence[:12],
                }
            )
            logging.info(
                "[NEH Beam] Candidate %s finished with makespan=%s elapsed=%.3f sec.",
                label,
                candidate_obj,
                candidate_timer.elapsed_sec,
            )
            if is_better_obj(candidate_obj, best_obj):
                best_schedule = result.schedule
                best_obj = candidate_obj
                best_label = label
                best_result = result

        logging.info(
            "[NEH Beam] Candidate result summary: %s; selected=%s makespan=%s",
            candidate_results,
            best_label,
            best_obj,
        )

        if best_result is not None and best_result.sub_obj_store:
            best_result.sub_obj_store.save_yaml(
                self.get_file_path_for_subroutine("_obj_log.yaml")
            )

        final_report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=False,
            subroutine_name="neh_cp_sequence_beam",
            progress_obj_value_records=(
                best_result.sub_obj_store.obj_value_series.items()
                if best_result is not None and best_result.sub_obj_store
                else [(sub_timer.elapsed_sec, best_obj)]
            ),
        )
        was_updated = self.solution_manager.register(final_report, best_schedule)
        self._record_last_neh_improvement(
            method_name="neh_cp_sequence_beam",
            input_obj=input_obj,
            output_obj=best_obj,
            was_updated=was_updated,
            extra_label=f"selected_candidate={best_label}",
        )

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, best_obj, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        if was_updated:
            self.set_cp_model_as_base_cp_model()
            if draw_gantt:
                self.draw_incumbent_gantt()

    @staticmethod
    def _branch_portfolio_to_plain_obj(value: Any) -> Any:
        if isinstance(value, DynamicDataObject):
            return value.to_obj()
        if isinstance(value, list):
            return [
                HybridFlowShopCpLnsController._branch_portfolio_to_plain_obj(item)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                HybridFlowShopCpLnsController._branch_portfolio_to_plain_obj(item)
                for item in value
            )
        if isinstance(value, Mapping):
            return {
                key: HybridFlowShopCpLnsController._branch_portfolio_to_plain_obj(val)
                for key, val in value.items()
            }
        return value

    def _run_neh_cp_branch_step(
        self,
        ref_schedule: HybridFlowshopLiteSchedule,
        step_config: Mapping[str, Any],
        *,
        solver_thread_cnt: int,
        log_cp_subproblem_bounds: bool,
        log_cp_subproblem_progress: bool,
    ) -> _BranchPortfolioStepResult:
        step = dict(self._branch_portfolio_to_plain_obj(step_config))
        method_name = str(step.get("method", "neh_cp"))
        if method_name in {"solve_base_cp_model", "base_cp"}:
            return self._run_base_cp_branch_step(
                ref_schedule,
                step,
                default_solver_thread_cnt=solver_thread_cnt,
            )
        step.pop("method", None)
        if method_name != "neh_cp":
            raise ValueError(
                "neh_cp_branch_portfolio supports NEH-CP and base-CP lane steps; "
                f"got method={method_name!r}."
            )
        step.pop("name", None)
        step.pop("label", None)

        job_seq_by_last_retained_cp = bool(
            step.pop("job_seq_by_last_retained_cp", False)
        )
        retained_cp_sequence_fallback = str(
            step.pop("retained_cp_sequence_fallback", "midpoint")
        )
        job_sequence_override = None
        if job_seq_by_last_retained_cp:
            try:
                job_sequence_override = (
                    self._get_last_retained_cp_consensus_job_sequence()
                )
            except ValueError:
                fallback = retained_cp_sequence_fallback.lower()
                if fallback == "error":
                    raise
                logging.warning(
                    "[NEH Branch] Last retained-CP job sequence is unavailable; "
                    "falling back to %s sequence.",
                    fallback,
                )
                if fallback == "bottleneck":
                    step["job_seq_by_bottleneck_stage"] = True
                elif fallback == "first_stage":
                    step["job_seq_by_1st_stage"] = True
                elif fallback not in {"midpoint", "default", "none"}:
                    raise ValueError(
                        "Unsupported retained_cp_sequence_fallback: "
                        f"{retained_cp_sequence_fallback!r}."
                    )

        if step.get("solver_thread_cnt") is None:
            step["solver_thread_cnt"] = solver_thread_cnt
        step.setdefault("log_cp_subproblem_bounds", log_cp_subproblem_bounds)
        step.setdefault("log_cp_subproblem_progress", log_cp_subproblem_progress)

        allowed_keys = {
            "added_batch_size",
            "added_batch_count",
            "min_added_batch_count",
            "max_added_batch_count",
            "job_seq_by_1st_stage",
            "job_seq_by_bottleneck_stage",
            "preserved_head_job_portion",
            "max_time_per_add",
            "cp_tl_nc_multiplier",
            "cp_tl_c_multiplier",
            "profile_fix_by_machine",
            "machine_precedence_stride",
            "profile_fix_min_batch_idx",
            "profile_fix_min_batch_portion",
            "profile_fix_max_batch_idx",
            "profile_fix_by_machine_from_batch_idx",
            "profile_fix_min_job_age_batches",
            "stage_precedence_min_processing_time_diff",
            "stage_precedence_min_processing_time_diff_ratio",
            "minimize_sum_ci_lex",
            "cp_tl_nc_multiplier_2nd_obj",
            "cp_tl_c_multiplier_2nd_obj",
            "minimize_sum_ci_lin",
            "tighten_ranges",
            "link_job_completion",
            "make_semi_active_every_cp",
            "solver_thread_cnt",
            "use_lns_only",
            "error_if_infeasible",
            "stop_before_final_reserve",
            "min_remaining_sec_after_neh",
            "min_remaining_nc_after_neh",
            "time_guard_estimate_safety_factor",
            "time_guard_min_completed_batches",
            "skip_if_estimated_neh_exceeds_remaining",
            "full_neh_estimate_safety_factor",
            "log_cp_subproblem_bounds",
            "log_cp_subproblem_progress",
        }
        unsupported_keys = sorted(set(step) - allowed_keys)
        if unsupported_keys:
            raise ValueError(
                "Unsupported NEH-CP branch step keys: " + ", ".join(unsupported_keys)
            )

        constructor = NehCpConstructor(self)
        result = constructor.run(
            ref_schedule,
            self.instance,
            self.job_2_stage_2_p_dict,
            self.stage_2_job_2_p_dict,
            job_sequence_override=job_sequence_override,
            **step,
        )
        return _BranchPortfolioStepResult(
            schedule=result.schedule,
            sub_obj_store=result.sub_obj_store,
            last_obj_value=float(result.last_obj_value),
        )

    def _run_base_cp_branch_step(
        self,
        ref_schedule: HybridFlowshopLiteSchedule,
        step_config: Mapping[str, Any],
        *,
        default_solver_thread_cnt: int,
    ) -> _BranchPortfolioStepResult:
        """Run a base-CP solve from a lane-local schedule without registering it."""
        step = dict(self._branch_portfolio_to_plain_obj(step_config))
        step.pop("method", None)
        step.pop("name", None)
        step.pop("label", None)

        computational_time = step.pop("computational_time", None)
        tl_nc_multiplier = step.pop("tl_nc_multiplier", None)
        small_workload_threshold = step.pop("small_workload_threshold", None)
        small_workload_tl_nc_multiplier = step.pop(
            "small_workload_tl_nc_multiplier", None
        )
        if (
            small_workload_threshold is not None
            and small_workload_tl_nc_multiplier is not None
            and self._get_instance_workload_size() < int(small_workload_threshold)
        ):
            logging.info(
                "[NEH Branch] Base-CP lane uses small-workload budget: "
                "workload_size=%d threshold=%d tl_nc_multiplier=%s -> %s.",
                self._get_instance_workload_size(),
                int(small_workload_threshold),
                tl_nc_multiplier,
                small_workload_tl_nc_multiplier,
            )
            tl_nc_multiplier = float(small_workload_tl_nc_multiplier)

        raw_solver_thread_cnt = step.pop("solver_thread_cnt", default_solver_thread_cnt)
        solver_thread_cnt = (
            default_solver_thread_cnt
            if raw_solver_thread_cnt is None
            else int(raw_solver_thread_cnt)
        )
        use_final_time_reserve = bool(step.pop("use_final_time_reserve", False))
        consume_all_remaining_with_final_reserve = bool(
            step.pop("consume_all_remaining_with_final_reserve", False)
        )
        make_semi_active_after_cp = bool(step.pop("make_semi_active_after_cp", False))
        encode_cumulative_as_reservoir = step.pop(
            "encode_cumulative_as_reservoir", None
        )
        expand_reservoir_constraints = step.pop("expand_reservoir_constraints", None)
        expand_reservoir_using_circuit = step.pop(
            "expand_reservoir_using_circuit", None
        )
        interleave_search = step.pop("interleave_search", None)
        use_lns_only = step.pop("use_lns_only", None)
        tighten_ranges = bool(step.pop("tighten_ranges", False))
        link_job_completion = bool(step.pop("link_job_completion", False))
        cp_model_probing_level = step.pop("cp_model_probing_level", None)
        log_search_progress = bool(step.pop("log_search_progress", False))
        error_if_infeasible = bool(step.pop("error_if_infeasible", False))
        draw_gantt = bool(step.pop("draw_gantt", False))

        unsupported_keys = sorted(step)
        if unsupported_keys:
            raise ValueError(
                "Unsupported base-CP branch step keys: " + ", ".join(unsupported_keys)
            )

        sub_timer = ElapsedTimer()
        if self.base_cp_model_is_set and self._base_cp_model_matches_options(
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        ):
            self.cp_model.delete_added_constraints()
        else:
            self.set_cp_model_as_base_cp_model(
                tighten_ranges=tighten_ranges,
                link_job_completion=link_job_completion,
            )

        self.cp_model.clear_hints()
        BaseModelBuilder.apply_start_hints_from_start_time_map(
            self.cp_model,
            self.params,
            self.vars,
            ref_schedule.get_jik_2_start_time_map(),
        )
        BaseModelBuilder.apply_end_hints_from_end_time_map(
            self.cp_model,
            self.params,
            self.vars,
            ref_schedule.get_jik_2_end_time_map(),
        )
        self.cp_model.add_hint(self.vars.makespan, ref_schedule.makespan)

        resolved_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        if use_final_time_reserve:
            resolved_time = self.consume_reserved_final_time_sec(
                fallback_sec=resolved_time,
                consume_all_remaining=consume_all_remaining_with_final_reserve,
            )
        if resolved_time is not None:
            resolved_time = max(0.0, resolved_time - sub_timer.elapsed_sec)

        logging.info(
            "[NEH Branch] Base-CP lane solving from makespan=%s time_limit=%s "
            "threads=%d use_lns_only=%s tighten_ranges=%s probing=%s.",
            ref_schedule.makespan,
            resolved_time,
            solver_thread_cnt,
            use_lns_only,
            tighten_ranges,
            cp_model_probing_level,
        )
        report, solution = self.solve_current_cp_remaining_time_limit(
            resolved_time,
            solver_thread_cnt,
            make_semi_active_after_cp=make_semi_active_after_cp,
            obj_value_is_valid=True,
            obj_bound_is_valid=True,
            is_initial_solution=False,
            encode_cumulative_as_reservoir=encode_cumulative_as_reservoir,
            expand_reservoir_constraints=expand_reservoir_constraints,
            expand_reservoir_using_circuit=expand_reservoir_using_circuit,
            interleave_search=interleave_search,
            use_lns_only=use_lns_only,
            cp_model_probing_level=cp_model_probing_level,
            log_search_progress=log_search_progress,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )
        if solution is None or not report.is_feasible:
            logging.warning(
                "[NEH Branch] Base-CP lane did not produce a feasible schedule; "
                "keeping lane input makespan=%s.",
                ref_schedule.makespan,
            )
            return _BranchPortfolioStepResult(
                schedule=ref_schedule.deepcopy(),
                sub_obj_store=None,
                last_obj_value=float(ref_schedule.makespan),
            )

        logging.info(
            "[NEH Branch] Base-CP lane finished: input=%s output=%s bound=%s status=%s.",
            ref_schedule.makespan,
            solution.makespan,
            report.obj_bound,
            report.status,
        )
        return _BranchPortfolioStepResult(
            schedule=solution,
            sub_obj_store=None,
            last_obj_value=float(solution.makespan),
        )

    def neh_cp_branch_portfolio(
        self,
        lanes: Sequence[Mapping[str, Any]],
        solver_thread_cnt: int = 16,
        keep_initial_candidate: bool = True,
        accept_equal_obj_from_later_lane: bool = False,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Run several NEH-CP lanes from the same incumbent and keep the best.

        Each lane starts from a deep copy of the incumbent schedule at method entry.
        The global solution manager is updated only once, after all lanes finish, so
        the lane comparison is not biased by the incumbent changes of earlier lanes.
        """
        sub_timer = ElapsedTimer()
        incumbent = self.solution_manager.get_incumbent()
        if incumbent is None:
            raise ValueError(
                "No incumbent solution available for NEH branch portfolio."
            )
        if not lanes:
            raise ValueError("neh_cp_branch_portfolio requires at least one lane.")

        input_obj = float(incumbent.makespan)
        prefix_schedule = incumbent.deepcopy()
        best_schedule = prefix_schedule.deepcopy()
        best_obj = float(best_schedule.makespan)
        best_label = "initial"
        best_progress_records: list[tuple[float, float]] = [
            (0.0, best_obj),
        ]
        lane_summaries: list[dict[str, Any]] = []

        def is_better_or_selected_equal(
            candidate_obj: float, incumbent_obj: float
        ) -> bool:
            if self.solution_manager._a_is_better_obj_value(
                candidate_obj,
                incumbent_obj,
            ):
                return True
            return accept_equal_obj_from_later_lane and math.isclose(
                candidate_obj,
                incumbent_obj,
            )

        if not keep_initial_candidate:
            best_schedule = None
            best_obj = math.inf
            best_label = ""
            best_progress_records = []

        for lane_idx, lane_config in enumerate(lanes, start=1):
            if self.is_stopping_condition(log_reason_if_true=False):
                logging.info(
                    "[NEH Branch] Stopping before lane %d because the global stopping "
                    "condition is met.",
                    lane_idx,
                )
                break

            lane = dict(self._branch_portfolio_to_plain_obj(lane_config))
            lane_name = str(lane.get("name") or lane.get("label") or f"lane_{lane_idx}")
            step_configs = lane.get("steps")
            if not isinstance(step_configs, Sequence) or isinstance(
                step_configs,
                (str, bytes),
            ):
                raise ValueError(f"Lane {lane_name!r} must define a list of steps.")

            lane_schedule = prefix_schedule.deepcopy()
            lane_best_schedule = lane_schedule.deepcopy()
            lane_best_obj = float(lane_best_schedule.makespan)
            lane_progress_records: list[tuple[float, float]] = [(0.0, lane_best_obj)]
            lane_step_summaries: list[dict[str, Any]] = []

            logging.info(
                "[NEH Branch] Starting lane %d/%d label=%s from prefix makespan=%s "
                "with %d steps.",
                lane_idx,
                len(lanes),
                lane_name,
                lane_schedule.makespan,
                len(step_configs),
            )

            for step_idx, step_config in enumerate(step_configs, start=1):
                if self.is_stopping_condition(log_reason_if_true=False):
                    logging.info(
                        "[NEH Branch] Stopping lane %s before step %d because the "
                        "global stopping condition is met.",
                        lane_name,
                        step_idx,
                    )
                    break

                step_label = (
                    dict(self._branch_portfolio_to_plain_obj(step_config)).get("name")
                    or dict(self._branch_portfolio_to_plain_obj(step_config)).get(
                        "label"
                    )
                    or f"step_{step_idx}"
                )
                step_input_obj = float(lane_schedule.makespan)
                step_timer = ElapsedTimer()
                logging.info(
                    "[NEH Branch] Lane %s step %d/%d (%s) starting with makespan=%s.",
                    lane_name,
                    step_idx,
                    len(step_configs),
                    step_label,
                    step_input_obj,
                )
                result = self._run_neh_cp_branch_step(
                    lane_schedule,
                    step_config,
                    solver_thread_cnt=solver_thread_cnt,
                    log_cp_subproblem_bounds=log_cp_subproblem_bounds,
                    log_cp_subproblem_progress=log_cp_subproblem_progress,
                )
                candidate_schedule = result.schedule
                candidate_obj = float(candidate_schedule.makespan)
                accepted = candidate_obj <= step_input_obj or math.isclose(
                    candidate_obj, step_input_obj
                )
                if accepted:
                    lane_schedule = candidate_schedule
                else:
                    logging.info(
                        "[NEH Branch] Lane %s step %s produced worse makespan=%s "
                        "than input=%s; keeping the previous lane schedule.",
                        lane_name,
                        step_label,
                        candidate_obj,
                        step_input_obj,
                    )

                lane_current_obj = float(lane_schedule.makespan)
                if is_better_or_selected_equal(lane_current_obj, lane_best_obj):
                    lane_best_schedule = lane_schedule.deepcopy()
                    lane_best_obj = lane_current_obj

                if result.sub_obj_store:
                    lane_progress_records.extend(
                        (float(t), float(v))
                        for t, v in result.sub_obj_store.obj_value_series.items()
                    )
                else:
                    lane_progress_records.append(
                        (step_timer.elapsed_sec, lane_current_obj)
                    )

                lane_step_summaries.append(
                    {
                        "step": step_idx,
                        "label": step_label,
                        "input_obj": step_input_obj,
                        "candidate_obj": candidate_obj,
                        "accepted": accepted,
                        "lane_current_obj": lane_current_obj,
                        "lane_best_obj": lane_best_obj,
                        "elapsed_sec": step_timer.elapsed_sec,
                    }
                )
                logging.info(
                    "[NEH Branch] Lane %s step %s finished: candidate=%s accepted=%s "
                    "lane_current=%s lane_best=%s elapsed=%.3f sec.",
                    lane_name,
                    step_label,
                    candidate_obj,
                    accepted,
                    lane_current_obj,
                    lane_best_obj,
                    step_timer.elapsed_sec,
                )

            lane_summary = {
                "lane": lane_name,
                "input_obj": input_obj,
                "output_obj": lane_best_obj,
                "improvement": input_obj - lane_best_obj,
                "steps": lane_step_summaries,
            }
            lane_summaries.append(lane_summary)

            if is_better_or_selected_equal(lane_best_obj, best_obj):
                best_schedule = lane_best_schedule.deepcopy()
                best_obj = lane_best_obj
                best_label = lane_name
                best_progress_records = lane_progress_records

            logging.info(
                "[NEH Branch] Lane %s summary: output=%s improvement=%s selected_so_far=%s:%s",
                lane_name,
                lane_best_obj,
                input_obj - lane_best_obj,
                best_label,
                best_obj,
            )

        if best_schedule is None:
            best_schedule = prefix_schedule.deepcopy()
            best_obj = float(best_schedule.makespan)
            best_label = "initial_fallback"
            best_progress_records = [(0.0, best_obj)]

        logging.info(
            "[NEH Branch] Portfolio summary: lanes=%s selected=%s makespan=%s",
            lane_summaries,
            best_label,
            best_obj,
        )

        final_report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=False,
            subroutine_name="neh_cp_branch_portfolio",
            progress_obj_value_records=tuple(best_progress_records),
        )
        was_updated = self.solution_manager.register(final_report, best_schedule)
        self._record_last_neh_improvement(
            method_name="neh_cp_branch_portfolio",
            input_obj=input_obj,
            output_obj=best_obj,
            was_updated=was_updated,
            extra_label=f"selected_lane={best_label}",
        )

        self.add_obj_value_log(self.timer.elapsed_sec, best_obj, is_maximize=False)
        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
        )

        if was_updated:
            self.set_cp_model_as_base_cp_model()
            if draw_gantt:
                self.draw_incumbent_gantt()

    def _run_selected_suffix_portfolio_step(
        self,
        step_config: Mapping[str, Any],
        *,
        lane_name: str,
        step_idx: int,
        step_count: int,
        default_solver_thread_cnt: int,
    ) -> None:
        step = dict(self._branch_portfolio_to_plain_obj(step_config))
        method_name = str(step.pop("method", "neh_cp"))
        if method_name == "base_cp":
            method_name = "solve_base_cp_model"
        step_label = str(step.pop("name", step.pop("label", f"step_{step_idx}")))

        solver_thread_methods = {
            "neh_cp",
            "workload_guarded_neh_cp",
            "workload_scaled_guarded_neh_cp",
            "neh_cp_adaptive_preserved_head",
            "neh_cp_sequence_beam",
            "workload_guarded_neh_cp_sequence_beam",
            "solve_base_cp_model",
            "solve_base_cp_model_with_last_neh_adaptive_time",
            "solve_base_cp_model_if_last_neh_improved",
            "incremental_pw_cp",
            "bound_gap_guarded_incremental_pw_cp",
        }
        allowed_methods = solver_thread_methods | {
            "retained_cp_bottleneck_band_stage_ns",
            "workload_guarded_retained_cp_bottleneck_band_stage_ns",
        }
        if method_name not in allowed_methods:
            raise ValueError(
                "Unsupported suffix probe portfolio suffix step method: "
                f"{method_name!r}."
            )
        if (
            method_name in solver_thread_methods
            and step.get("solver_thread_cnt") is None
        ):
            step["solver_thread_cnt"] = default_solver_thread_cnt

        method = getattr(self, method_name, None)
        if not callable(method):
            raise ValueError(
                f"Suffix probe portfolio step method is not callable: {method_name!r}."
            )

        input_obj = None
        incumbent = self.solution_manager.get_incumbent()
        if incumbent is not None:
            input_obj = float(incumbent.makespan)
        logging.info(
            "[Suffix Probe] Selected lane %s suffix step %d/%d (%s:%s) "
            "starting from incumbent=%s.",
            lane_name,
            step_idx,
            step_count,
            method_name,
            step_label,
            input_obj,
        )
        step_timer = ElapsedTimer()
        method(**step)
        output_obj = None
        incumbent_after = self.solution_manager.get_incumbent()
        if incumbent_after is not None:
            output_obj = float(incumbent_after.makespan)
        logging.info(
            "[Suffix Probe] Selected lane %s suffix step %d/%d (%s:%s) "
            "finished with incumbent=%s elapsed=%.3f sec.",
            lane_name,
            step_idx,
            step_count,
            method_name,
            step_label,
            output_obj,
            step_timer.elapsed_sec,
        )

    def suffix_probe_portfolio(
        self,
        lanes: Sequence[Mapping[str, Any]],
        solver_thread_cnt: int = 16,
        accept_equal_obj_from_later_lane: bool = False,
        commit_equal_obj: bool = True,
        run_selected_suffix: bool = True,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
        draw_gantt: bool = False,
        fallback_steps: Sequence[Mapping[str, Any]] | None = None,
        fallback_label: str = "guard_fallback",
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
        min_retained_cp_restore_loss_ratio: float | None = None,
        max_retained_cp_restore_loss_ratio: float | None = None,
        run_if_retained_cp_restore_missing: bool = True,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
    ) -> None:
        """Probe several suffix lanes from the same incumbent, then run one suffix.

        ``probe_steps`` are evaluated on lane-local schedule copies. Only the selected
        probe schedule is committed to the global solution manager. The selected
        lane's ``suffix_steps`` are then executed normally from that committed
        schedule. This lets a flow cheaply compare 0504-like and 0510-like basins
        before spending the remaining budget on one path.
        """
        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        sub_timer = ElapsedTimer()
        incumbent = self.solution_manager.get_incumbent()
        if incumbent is None:
            raise ValueError(
                "No incumbent solution available for suffix probe portfolio."
            )
        if not lanes:
            raise ValueError("suffix_probe_portfolio requires at least one lane.")

        if fallback_steps is None:
            plain_fallback_steps: Sequence[Mapping[str, Any]] = ()
        else:
            if not isinstance(fallback_steps, Sequence) or isinstance(
                fallback_steps,
                (str, bytes),
            ):
                raise ValueError(
                    "suffix_probe_portfolio fallback_steps must be a list when provided."
                )
            plain_fallback_steps = fallback_steps

        def run_guard_fallback(reason: str) -> None:
            if not plain_fallback_steps:
                logging.info(
                    "[Suffix Probe] Guard skipped portfolio (%s); keeping incumbent.",
                    reason,
                )
                return
            logging.info(
                "[Suffix Probe] Guard skipped portfolio (%s); running %d fallback "
                "steps label=%s.",
                reason,
                len(plain_fallback_steps),
                fallback_label,
            )
            for step_idx, step_config in enumerate(plain_fallback_steps, start=1):
                self._run_selected_suffix_portfolio_step(
                    step_config,
                    lane_name=fallback_label,
                    step_idx=step_idx,
                    step_count=len(plain_fallback_steps),
                    default_solver_thread_cnt=solver_thread_cnt,
                )

        def workload_guard_allows(
            *,
            context_label: str,
            min_workload_size_value: int | None = None,
            max_workload_size_value: int | None = None,
            min_instance_job_count_value: int | None = None,
            max_instance_job_count_value: int | None = None,
            min_instance_stage_count_value: int | None = None,
            max_instance_stage_count_value: int | None = None,
        ) -> bool:
            if (
                min_workload_size_value is not None
                and workload_size < int(min_workload_size_value)
            ):
                logging.info(
                    "[%s] Skipping: workload_size=%d is below min_workload_size=%d.",
                    context_label,
                    workload_size,
                    int(min_workload_size_value),
                )
                return False
            if (
                max_workload_size_value is not None
                and workload_size > int(max_workload_size_value)
            ):
                logging.info(
                    "[%s] Skipping: workload_size=%d is above max_workload_size=%d.",
                    context_label,
                    workload_size,
                    int(max_workload_size_value),
                )
                return False
            if (
                min_instance_job_count_value is not None
                and instance_job_count < int(min_instance_job_count_value)
            ):
                logging.info(
                    "[%s] Skipping: job_count=%d is below min_instance_job_count=%d.",
                    context_label,
                    instance_job_count,
                    int(min_instance_job_count_value),
                )
                return False
            if (
                max_instance_job_count_value is not None
                and instance_job_count > int(max_instance_job_count_value)
            ):
                logging.info(
                    "[%s] Skipping: job_count=%d is above max_instance_job_count=%d.",
                    context_label,
                    instance_job_count,
                    int(max_instance_job_count_value),
                )
                return False
            if (
                min_instance_stage_count_value is not None
                and instance_stage_count < int(min_instance_stage_count_value)
            ):
                logging.info(
                    "[%s] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                    context_label,
                    instance_stage_count,
                    int(min_instance_stage_count_value),
                )
                return False
            if (
                max_instance_stage_count_value is not None
                and instance_stage_count > int(max_instance_stage_count_value)
            ):
                logging.info(
                    "[%s] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                    context_label,
                    instance_stage_count,
                    int(max_instance_stage_count_value),
                )
                return False
            return True

        if not workload_guard_allows(
            context_label="Suffix Probe",
            min_workload_size_value=min_workload_size,
            max_workload_size_value=max_workload_size,
            min_instance_job_count_value=min_instance_job_count,
            max_instance_job_count_value=max_instance_job_count,
            min_instance_stage_count_value=min_instance_stage_count,
            max_instance_stage_count_value=max_instance_stage_count,
        ):
            run_guard_fallback("workload")
            return

        if not self._bound_gap_guard_allows(
            context_label="Suffix Probe",
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            run_guard_fallback("bound_gap")
            return
        if not self._retained_cp_restore_loss_guard_allows(
            context_label="Suffix Probe",
            min_retained_cp_restore_loss_ratio=min_retained_cp_restore_loss_ratio,
            max_retained_cp_restore_loss_ratio=max_retained_cp_restore_loss_ratio,
            run_if_retained_cp_restore_missing=run_if_retained_cp_restore_missing,
        ):
            run_guard_fallback("retained_cp_restore_loss")
            return

        input_obj = float(incumbent.makespan)
        prefix_schedule = incumbent.deepcopy()
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_obj = math.inf
        best_label = ""
        best_suffix_steps: Sequence[Mapping[str, Any]] = ()
        best_progress_records: list[tuple[float, float]] = []
        lane_summaries: list[dict[str, Any]] = []

        def is_better_or_selected_equal(
            candidate_obj: float, incumbent_obj: float
        ) -> bool:
            if self.solution_manager._a_is_better_obj_value(
                candidate_obj,
                incumbent_obj,
            ):
                return True
            return accept_equal_obj_from_later_lane and math.isclose(
                candidate_obj,
                incumbent_obj,
            )

        for lane_idx, lane_config in enumerate(lanes, start=1):
            if self.is_stopping_condition(log_reason_if_true=False):
                logging.info(
                    "[Suffix Probe] Stopping before lane %d because the global "
                    "stopping condition is met.",
                    lane_idx,
                )
                break

            lane = dict(self._branch_portfolio_to_plain_obj(lane_config))
            lane_name = str(lane.get("name") or lane.get("label") or f"lane_{lane_idx}")
            if not workload_guard_allows(
                context_label=f"Suffix Probe lane {lane_name}",
                min_workload_size_value=lane.get("min_workload_size"),
                max_workload_size_value=lane.get("max_workload_size"),
                min_instance_job_count_value=lane.get("min_instance_job_count"),
                max_instance_job_count_value=lane.get("max_instance_job_count"),
                min_instance_stage_count_value=lane.get("min_instance_stage_count"),
                max_instance_stage_count_value=lane.get("max_instance_stage_count"),
            ):
                continue
            probe_steps = lane.get("probe_steps", lane.get("steps", ()))
            suffix_steps = lane.get("suffix_steps", ())
            if not isinstance(probe_steps, Sequence) or isinstance(
                probe_steps,
                (str, bytes),
            ):
                raise ValueError(
                    f"Suffix probe lane {lane_name!r} must define probe_steps."
                )
            if not isinstance(suffix_steps, Sequence) or isinstance(
                suffix_steps,
                (str, bytes),
            ):
                raise ValueError(
                    f"Suffix probe lane {lane_name!r} must define suffix_steps as a list."
                )

            lane_schedule = prefix_schedule.deepcopy()
            lane_best_schedule = lane_schedule.deepcopy()
            lane_best_obj = float(lane_best_schedule.makespan)
            lane_progress_records: list[tuple[float, float]] = [(0.0, lane_best_obj)]
            lane_step_summaries: list[dict[str, Any]] = []

            logging.info(
                "[Suffix Probe] Starting lane %d/%d label=%s from prefix makespan=%s "
                "with %d probe steps and %d suffix steps.",
                lane_idx,
                len(lanes),
                lane_name,
                lane_schedule.makespan,
                len(probe_steps),
                len(suffix_steps),
            )

            for step_idx, step_config in enumerate(probe_steps, start=1):
                if self.is_stopping_condition(log_reason_if_true=False):
                    logging.info(
                        "[Suffix Probe] Stopping lane %s before probe step %d because "
                        "the global stopping condition is met.",
                        lane_name,
                        step_idx,
                    )
                    break

                plain_step = dict(self._branch_portfolio_to_plain_obj(step_config))
                step_label = str(
                    plain_step.get("name")
                    or plain_step.get("label")
                    or f"probe_{step_idx}"
                )
                step_input_obj = float(lane_schedule.makespan)
                step_timer = ElapsedTimer()
                logging.info(
                    "[Suffix Probe] Lane %s probe step %d/%d (%s) starting with "
                    "makespan=%s.",
                    lane_name,
                    step_idx,
                    len(probe_steps),
                    step_label,
                    step_input_obj,
                )
                result = self._run_neh_cp_branch_step(
                    lane_schedule,
                    step_config,
                    solver_thread_cnt=solver_thread_cnt,
                    log_cp_subproblem_bounds=log_cp_subproblem_bounds,
                    log_cp_subproblem_progress=log_cp_subproblem_progress,
                )
                candidate_schedule = result.schedule
                candidate_obj = float(candidate_schedule.makespan)
                accepted = candidate_obj <= step_input_obj or math.isclose(
                    candidate_obj,
                    step_input_obj,
                )
                if accepted:
                    lane_schedule = candidate_schedule
                else:
                    logging.info(
                        "[Suffix Probe] Lane %s probe step %s produced worse "
                        "makespan=%s than input=%s; keeping previous lane schedule.",
                        lane_name,
                        step_label,
                        candidate_obj,
                        step_input_obj,
                    )

                lane_current_obj = float(lane_schedule.makespan)
                if self.solution_manager._a_is_better_obj_value(
                    lane_current_obj,
                    lane_best_obj,
                ) or math.isclose(lane_current_obj, lane_best_obj):
                    lane_best_schedule = lane_schedule.deepcopy()
                    lane_best_obj = lane_current_obj

                if result.sub_obj_store:
                    lane_progress_records.extend(
                        (float(t), float(v))
                        for t, v in result.sub_obj_store.obj_value_series.items()
                    )
                else:
                    lane_progress_records.append(
                        (step_timer.elapsed_sec, lane_current_obj)
                    )

                lane_step_summaries.append(
                    {
                        "step": step_idx,
                        "label": step_label,
                        "input_obj": step_input_obj,
                        "candidate_obj": candidate_obj,
                        "accepted": accepted,
                        "lane_current_obj": lane_current_obj,
                        "lane_best_obj": lane_best_obj,
                        "elapsed_sec": step_timer.elapsed_sec,
                    }
                )
                logging.info(
                    "[Suffix Probe] Lane %s probe step %s finished: candidate=%s "
                    "accepted=%s lane_current=%s lane_best=%s elapsed=%.3f sec.",
                    lane_name,
                    step_label,
                    candidate_obj,
                    accepted,
                    lane_current_obj,
                    lane_best_obj,
                    step_timer.elapsed_sec,
                )

            lane_summary = {
                "lane": lane_name,
                "input_obj": input_obj,
                "probe_output_obj": lane_best_obj,
                "probe_improvement": input_obj - lane_best_obj,
                "probe_steps": lane_step_summaries,
                "suffix_step_count": len(suffix_steps),
            }
            lane_summaries.append(lane_summary)

            if best_schedule is None or is_better_or_selected_equal(
                lane_best_obj,
                best_obj,
            ):
                best_schedule = lane_best_schedule.deepcopy()
                best_obj = lane_best_obj
                best_label = lane_name
                best_suffix_steps = suffix_steps
                best_progress_records = lane_progress_records

            logging.info(
                "[Suffix Probe] Lane %s summary: probe_output=%s improvement=%s "
                "selected_so_far=%s:%s.",
                lane_name,
                lane_best_obj,
                input_obj - lane_best_obj,
                best_label,
                best_obj,
            )

        if best_schedule is None:
            if plain_fallback_steps:
                run_guard_fallback("all_lane_guards")
                return
            best_schedule = prefix_schedule.deepcopy()
            best_obj = float(best_schedule.makespan)
            best_label = "initial_fallback"
            best_suffix_steps = ()
            best_progress_records = [(0.0, best_obj)]

        logging.info(
            "[Suffix Probe] Portfolio summary: lanes=%s selected=%s probe_makespan=%s "
            "suffix_steps=%d.",
            lane_summaries,
            best_label,
            best_obj,
            len(best_suffix_steps),
        )

        final_report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=False,
            subroutine_name="suffix_probe_portfolio",
            progress_obj_value_records=tuple(best_progress_records),
        )
        try:
            was_updated = self.solution_manager.register(
                final_report,
                best_schedule,
                update_if_equal_obj=commit_equal_obj,
            )
        except TypeError:
            was_updated = self.solution_manager.register(final_report, best_schedule)
        self._record_last_neh_improvement(
            method_name="suffix_probe_portfolio",
            input_obj=input_obj,
            output_obj=best_obj,
            was_updated=was_updated,
            extra_label=f"selected_lane={best_label}",
        )

        self.add_obj_value_log(self.timer.elapsed_sec, best_obj, is_maximize=False)
        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
        )

        if was_updated:
            self.set_cp_model_as_base_cp_model()
            if draw_gantt:
                self.draw_incumbent_gantt()

        if run_selected_suffix:
            for step_idx, step_config in enumerate(best_suffix_steps, start=1):
                self._run_selected_suffix_portfolio_step(
                    step_config,
                    lane_name=best_label,
                    step_idx=step_idx,
                    step_count=len(best_suffix_steps),
                    default_solver_thread_cnt=solver_thread_cnt,
                )

    def workload_guarded_neh_cp_sequence_beam(
        self,
        solver_thread_cnt: int,
        candidate_source: str = "retained_cp_dispatch",
        candidate_top_k: int = 2,
        candidate_pool_top_k: int | None = 16,
        max_candidate_obj_slack: float | None = None,
        min_sequence_position_diff_ratio: float = 0.08,
        allow_sequence_duplicate_fallback: bool = False,
        include_incumbent_candidate: bool = True,
        added_batch_size: int = 1,
        added_batch_count: int | None = None,
        min_added_batch_count: int | None = None,
        max_added_batch_count: int | None = None,
        job_seq_by_1st_stage: bool = False,
        job_seq_by_bottleneck_stage: bool = False,
        preserved_head_job_portion: float = 0.0,
        max_time_per_add: float | None = None,
        cp_tl_nc_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        profile_fix_min_batch_idx: int = 2,
        profile_fix_min_batch_portion: float | None = None,
        profile_fix_max_batch_idx: int | None = None,
        profile_fix_by_machine_from_batch_idx: int | None = None,
        profile_fix_min_job_age_batches: int | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        minimize_sum_ci_lex: bool = False,
        cp_tl_nc_multiplier_2nd_obj: float | None = None,
        cp_tl_c_multiplier_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        make_semi_active_every_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        stop_before_final_reserve: bool = True,
        min_remaining_sec_after_neh: float | None = None,
        min_remaining_nc_after_neh: float | None = None,
        time_guard_estimate_safety_factor: float = 1.15,
        time_guard_min_completed_batches: int = 1,
        skip_if_estimated_neh_exceeds_remaining: bool = True,
        full_neh_estimate_safety_factor: float = 1.0,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
        min_retained_cp_restore_loss_ratio: float | None = None,
        max_retained_cp_restore_loss_ratio: float | None = None,
        run_if_retained_cp_restore_missing: bool = True,
    ) -> None:
        """Run sequence-beam NEH only for an instance-size band."""
        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        if not self._bound_gap_guard_allows(
            context_label="NEH Beam",
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return
        if not self._retained_cp_restore_loss_guard_allows(
            context_label="NEH Beam",
            min_retained_cp_restore_loss_ratio=min_retained_cp_restore_loss_ratio,
            max_retained_cp_restore_loss_ratio=max_retained_cp_restore_loss_ratio,
            run_if_retained_cp_restore_missing=run_if_retained_cp_restore_missing,
        ):
            return
        if min_workload_size is not None and workload_size < int(min_workload_size):
            logging.info(
                "[NEH Beam] Skipping: workload_size=%d is below min_workload_size=%d.",
                workload_size,
                int(min_workload_size),
            )
            return
        if max_workload_size is not None and workload_size > int(max_workload_size):
            logging.info(
                "[NEH Beam] Skipping: workload_size=%d is above max_workload_size=%d.",
                workload_size,
                int(max_workload_size),
            )
            return
        if min_instance_job_count is not None and instance_job_count < int(
            min_instance_job_count
        ):
            logging.info(
                "[NEH Beam] Skipping: job_count=%d is below min_instance_job_count=%d.",
                instance_job_count,
                int(min_instance_job_count),
            )
            return
        if max_instance_job_count is not None and instance_job_count > int(
            max_instance_job_count
        ):
            logging.info(
                "[NEH Beam] Skipping: job_count=%d is above max_instance_job_count=%d.",
                instance_job_count,
                int(max_instance_job_count),
            )
            return
        if min_instance_stage_count is not None and instance_stage_count < int(
            min_instance_stage_count
        ):
            logging.info(
                "[NEH Beam] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                instance_stage_count,
                int(min_instance_stage_count),
            )
            return
        if max_instance_stage_count is not None and instance_stage_count > int(
            max_instance_stage_count
        ):
            logging.info(
                "[NEH Beam] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                instance_stage_count,
                int(max_instance_stage_count),
            )
            return
        logging.info(
            "[NEH Beam] Running with job_count=%d stage_count=%d workload_size=%d "
            "inside guards workload=[%s, %s] job_count=[%s, %s] stage_count=[%s, %s].",
            instance_job_count,
            instance_stage_count,
            workload_size,
            min_workload_size,
            max_workload_size,
            min_instance_job_count,
            max_instance_job_count,
            min_instance_stage_count,
            max_instance_stage_count,
        )
        self.neh_cp_sequence_beam(
            solver_thread_cnt=solver_thread_cnt,
            candidate_source=candidate_source,
            candidate_top_k=candidate_top_k,
            candidate_pool_top_k=candidate_pool_top_k,
            max_candidate_obj_slack=max_candidate_obj_slack,
            min_sequence_position_diff_ratio=min_sequence_position_diff_ratio,
            allow_sequence_duplicate_fallback=allow_sequence_duplicate_fallback,
            include_incumbent_candidate=include_incumbent_candidate,
            added_batch_size=added_batch_size,
            added_batch_count=added_batch_count,
            min_added_batch_count=min_added_batch_count,
            max_added_batch_count=max_added_batch_count,
            job_seq_by_1st_stage=job_seq_by_1st_stage,
            job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
            preserved_head_job_portion=preserved_head_job_portion,
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            profile_fix_min_batch_idx=profile_fix_min_batch_idx,
            profile_fix_min_batch_portion=profile_fix_min_batch_portion,
            profile_fix_max_batch_idx=profile_fix_max_batch_idx,
            profile_fix_by_machine_from_batch_idx=profile_fix_by_machine_from_batch_idx,
            profile_fix_min_job_age_batches=profile_fix_min_job_age_batches,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
            minimize_sum_ci_lex=minimize_sum_ci_lex,
            cp_tl_nc_multiplier_2nd_obj=cp_tl_nc_multiplier_2nd_obj,
            cp_tl_c_multiplier_2nd_obj=cp_tl_c_multiplier_2nd_obj,
            minimize_sum_ci_lin=minimize_sum_ci_lin,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
            make_semi_active_every_cp=make_semi_active_every_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
            stop_before_final_reserve=stop_before_final_reserve,
            min_remaining_sec_after_neh=min_remaining_sec_after_neh,
            min_remaining_nc_after_neh=min_remaining_nc_after_neh,
            time_guard_estimate_safety_factor=time_guard_estimate_safety_factor,
            time_guard_min_completed_batches=time_guard_min_completed_batches,
            skip_if_estimated_neh_exceeds_remaining=skip_if_estimated_neh_exceeds_remaining,
            full_neh_estimate_safety_factor=full_neh_estimate_safety_factor,
            log_cp_subproblem_bounds=log_cp_subproblem_bounds,
            log_cp_subproblem_progress=log_cp_subproblem_progress,
        )

    def _get_last_retained_cp_consensus_job_sequence(self) -> list[str]:
        retained_solution_rows = list(
            getattr(self, "last_retained_cp_lb_retained_solution_rows", None) or []
        )
        if not retained_solution_rows:
            raise ValueError("No retained-stage CP solution rows are available.")

        retained_result = getattr(self, "last_retained_cp_lb_result", None)
        stage_order = [str(stage_id) for stage_id in self.instance.stage_id_list]
        job_order = [str(job_id) for job_id in self.instance.job_id_list]
        job_order_index = {job_id: idx for idx, job_id in enumerate(job_order)}
        stage_order_index = {stage_id: idx for idx, stage_id in enumerate(stage_order)}

        stage_2_rows: dict[str, list[dict[str, Any]]] = {}
        for row in retained_solution_rows:
            stage_id = str(row["stage_id"])
            if stage_id in stage_order_index:
                stage_2_rows.setdefault(stage_id, []).append(dict(row))

        retained_stage_ids = [
            stage_id for stage_id in stage_order if stage_2_rows.get(stage_id)
        ]
        if not retained_stage_ids:
            raise ValueError("No retained-stage CP rows match instance stages.")

        bottleneck_stage_ids = {
            str(stage_id)
            for stage_id in (
                getattr(retained_result, "selected_bottleneck_stage_ids", ()) or ()
            )
        }
        bottleneck_stage_id = getattr(retained_result, "bottleneck_stage_id", None)
        if bottleneck_stage_id is not None:
            bottleneck_stage_ids.add(str(bottleneck_stage_id))

        denominator = max(len(stage_order) - 1, 1)
        job_2_weighted_rank = {job_id: 0.0 for job_id in job_order}
        job_2_weighted_start = {job_id: 0.0 for job_id in job_order}
        job_2_weight = {job_id: 0.0 for job_id in job_order}
        for stage_id in retained_stage_ids:
            rows = stage_2_rows[stage_id]
            rows.sort(
                key=lambda row: (
                    int(row["start"]),
                    int(row["end"]),
                    str(row["job_id"]),
                )
            )
            stage_idx = stage_order_index[stage_id]
            stage_weight = 1.0 + float(stage_idx) / float(denominator)
            if stage_id in bottleneck_stage_ids:
                stage_weight += 1.5
            for rank, row in enumerate(rows):
                job_id = str(row["job_id"])
                if job_id not in job_2_weight:
                    continue
                job_2_weighted_rank[job_id] += stage_weight * float(rank)
                job_2_weighted_start[job_id] += stage_weight * float(row["start"])
                job_2_weight[job_id] += stage_weight

        missing_rank = float(len(job_order))
        sortable_rows: list[tuple[tuple[float, float, int], str]] = []
        for job_id in job_order:
            weight = job_2_weight[job_id]
            if weight <= 0.0:
                sortable_rows.append(
                    ((missing_rank, missing_rank, job_order_index[job_id]), job_id)
                )
                continue
            sortable_rows.append(
                (
                    (
                        job_2_weighted_rank[job_id] / weight,
                        job_2_weighted_start[job_id] / weight,
                        job_order_index[job_id],
                    ),
                    job_id,
                )
            )
        sortable_rows.sort(key=lambda row: row[0])
        job_sequence = [job_id for _key, job_id in sortable_rows]
        logging.info(
            "[CP LB -> NEH] Built retained-CP consensus NEH sequence from stages=%s "
            "bottleneck_stages=%s head=%s",
            retained_stage_ids,
            sorted(bottleneck_stage_ids),
            job_sequence[: min(10, len(job_sequence))],
        )
        return job_sequence

    def neh_cp(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        added_batch_count: int | None = None,
        min_added_batch_count: int | None = None,
        max_added_batch_count: int | None = None,
        job_seq_by_1st_stage: bool = False,
        job_seq_by_bottleneck_stage: bool = False,
        job_seq_by_last_retained_cp: bool = False,
        preserved_head_job_portion: float = 0.0,
        max_time_per_add: float | None = None,
        cp_tl_nc_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        profile_fix_min_batch_idx: int = 2,
        profile_fix_min_batch_portion: float | None = None,
        profile_fix_max_batch_idx: int | None = None,
        profile_fix_by_machine_from_batch_idx: int | None = None,
        profile_fix_min_job_age_batches: int | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        minimize_sum_ci_lex: bool = False,
        cp_tl_nc_multiplier_2nd_obj: float | None = None,
        cp_tl_c_multiplier_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        make_semi_active_every_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        stop_before_final_reserve: bool = True,
        min_remaining_sec_after_neh: float | None = None,
        min_remaining_nc_after_neh: float | None = None,
        time_guard_estimate_safety_factor: float = 1.15,
        time_guard_min_completed_batches: int = 1,
        skip_if_estimated_neh_exceeds_remaining: bool = True,
        full_neh_estimate_safety_factor: float = 1.0,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
    ):
        """
        Builds a CP-guided solution using a midpoint sequence from the incumbent solution.

        Args:
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to
                use during search.
            added_batch_size (int, optional): The number of jobs to add in each
                iteration. Defaults to 1.
            added_batch_count (int | None, optional): If set, split reconstructed jobs
                into this many NEH-CP insertion iterations instead of using a fixed
                ``added_batch_size``.
            min_added_batch_count (int | None, optional): Lower bound for the resolved
                number of insertion iterations.
            max_added_batch_count (int | None, optional): Upper bound for the resolved
                number of insertion iterations.
            job_seq_by_1st_stage (bool, optional): If True, defines the job
                sequence according to incumbent schedule's first stage schedule.
                Defaults to False.
            job_seq_by_bottleneck_stage (bool, optional): If True, defines the job
                sequence according to incumbent schedule's bottleneck stage schedule.
                Otherwise, uses the job sequence by increasing order of
                (first stage start time + last stage end time) / 2.
                Defaults to False.
            job_seq_by_last_retained_cp (bool, optional): If True, defines the NEH
                insertion sequence by a weighted rank consensus over the last
                retained-stage CP solution. Defaults to False.
            preserved_head_job_portion (float, optional): The portion of jobs to preserve
                from the head of the incumbent schedule. Defaults to 0.0.
            max_time_per_add (float | None, optional): Time limit (in seconds) for
                solving each incremental subproblem. If None, uses the remaining time
                limit. Defaults to None.
            cp_tl_nc_multiplier (float | None, optional): Multiplier for the time
                limit of each CP subproblem. If None, uses the default value.
                Defaults to None.
            cp_tl_c_multiplier (float | None, optional): Multiplier for the time limit
                of each CP subproblem. If None, uses the default value.
                Defaults to None.
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                adjacency; otherwise apply stage-level time-based selection.
                Defaults to False.
            minimize_sum_ci_lex (bool, optional): If True, minimizes the sum of
                completion times in each CP subproblem after minimizing the makespan.
                Defaults to False.
            cp_tl_nc_multiplier_2nd_obj (float | None, optional): Multiplier for the time
                limit of each CP subproblem for the second objective. If None, uses the default value.
                Defaults to None.
            cp_tl_c_multiplier_2nd_obj (float | None, optional): Multiplier for the time limit
                of each CP subproblem for the second objective. If None, uses the default value.
                Defaults to None.
            minimize_sum_ci_lin (bool, optional): If True, minimizes the sum of
                completion times in each CP subproblem by a linear combination with the
                makespan. Defaults to False.
            make_semi_active_every_cp (bool, optional): If True, makes the solution
                semi-active after solving each CP subproblem. Defaults to False.
            error_if_infeasible (bool, optional): If True, raises an error if the
                solution is infeasible. Defaults to False.
            draw_gantt (bool, optional): If True, draws a Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        ref_schedule = self.solution_manager.get_incumbent()
        if ref_schedule is None:
            raise ValueError("No incumbent solution available for NEH-CP.")
        input_obj = float(ref_schedule.makespan)

        job_sequence_override = (
            self._get_last_retained_cp_consensus_job_sequence()
            if job_seq_by_last_retained_cp
            else None
        )
        constructor = NehCpConstructor(self)
        result: NehCpResult = constructor.run(
            ref_schedule,
            self.instance,
            self.job_2_stage_2_p_dict,
            self.stage_2_job_2_p_dict,
            added_batch_size=added_batch_size,
            added_batch_count=added_batch_count,
            min_added_batch_count=min_added_batch_count,
            max_added_batch_count=max_added_batch_count,
            job_seq_by_1st_stage=job_seq_by_1st_stage,
            job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
            job_sequence_override=job_sequence_override,
            preserved_head_job_portion=preserved_head_job_portion,
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            profile_fix_min_batch_idx=profile_fix_min_batch_idx,
            profile_fix_min_batch_portion=profile_fix_min_batch_portion,
            profile_fix_max_batch_idx=profile_fix_max_batch_idx,
            profile_fix_by_machine_from_batch_idx=profile_fix_by_machine_from_batch_idx,
            profile_fix_min_job_age_batches=profile_fix_min_job_age_batches,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
            minimize_sum_ci_lex=minimize_sum_ci_lex,
            cp_tl_nc_multiplier_2nd_obj=cp_tl_nc_multiplier_2nd_obj,
            cp_tl_c_multiplier_2nd_obj=cp_tl_c_multiplier_2nd_obj,
            minimize_sum_ci_lin=minimize_sum_ci_lin,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
            make_semi_active_every_cp=make_semi_active_every_cp,
            solver_thread_cnt=solver_thread_cnt,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            stop_before_final_reserve=stop_before_final_reserve,
            min_remaining_sec_after_neh=min_remaining_sec_after_neh,
            min_remaining_nc_after_neh=min_remaining_nc_after_neh,
            time_guard_estimate_safety_factor=time_guard_estimate_safety_factor,
            time_guard_min_completed_batches=time_guard_min_completed_batches,
            skip_if_estimated_neh_exceeds_remaining=skip_if_estimated_neh_exceeds_remaining,
            full_neh_estimate_safety_factor=full_neh_estimate_safety_factor,
            log_cp_subproblem_bounds=log_cp_subproblem_bounds,
            log_cp_subproblem_progress=log_cp_subproblem_progress,
        )
        obj_value = float(result.schedule.makespan)
        logging.info(f"NEH-CP done with makespan {obj_value}")
        # Write the objective store to a YAML file
        # TODO: suffix from output_metadata
        if result.sub_obj_store:
            result.sub_obj_store.save_yaml(
                self.get_file_path_for_subroutine("_obj_log.yaml")
            )

        # Create report for the final solution and register it
        final_report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
            subroutine_name="neh_cp",
            progress_obj_value_records=result.sub_obj_store.obj_value_series.items(),
        )
        was_updated: bool = self.solution_manager.register(
            final_report, result.schedule
        )
        self._record_last_neh_improvement(
            method_name="neh_cp",
            input_obj=input_obj,
            output_obj=obj_value,
            was_updated=was_updated,
        )

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        if was_updated:
            # Re-define base CP model with the new makespan
            self.set_cp_model_as_base_cp_model()
            if draw_gantt:
                self.draw_incumbent_gantt()

    def workload_guarded_neh_cp(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        added_batch_count: int | None = None,
        min_added_batch_count: int | None = None,
        max_added_batch_count: int | None = None,
        job_seq_by_1st_stage: bool = False,
        job_seq_by_bottleneck_stage: bool = False,
        job_seq_by_last_retained_cp: bool = False,
        preserved_head_job_portion: float = 0.0,
        max_time_per_add: float | None = None,
        cp_tl_nc_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        profile_fix_min_batch_idx: int = 2,
        profile_fix_min_batch_portion: float | None = None,
        profile_fix_max_batch_idx: int | None = None,
        profile_fix_by_machine_from_batch_idx: int | None = None,
        profile_fix_min_job_age_batches: int | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        minimize_sum_ci_lex: bool = False,
        cp_tl_nc_multiplier_2nd_obj: float | None = None,
        cp_tl_c_multiplier_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        make_semi_active_every_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        stop_before_final_reserve: bool = True,
        min_remaining_sec_after_neh: float | None = None,
        min_remaining_nc_after_neh: float | None = None,
        time_guard_estimate_safety_factor: float = 1.15,
        time_guard_min_completed_batches: int = 1,
        skip_if_estimated_neh_exceeds_remaining: bool = True,
        full_neh_estimate_safety_factor: float = 1.0,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
        min_workload_size: int | None = None,
        max_workload_size: int | None = None,
        min_instance_job_count: int | None = None,
        max_instance_job_count: int | None = None,
        min_instance_stage_count: int | None = None,
        max_instance_stage_count: int | None = None,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
        min_retained_cp_restore_loss_ratio: float | None = None,
        max_retained_cp_restore_loss_ratio: float | None = None,
        run_if_retained_cp_restore_missing: bool = True,
    ) -> None:
        """Run NEH-CP only for an instance-size band."""
        instance_job_count = int(self.instance.job_count)
        instance_stage_count = int(self.instance.stage_count)
        workload_size = self._get_instance_workload_size()
        if not self._bound_gap_guard_allows(
            context_label="Guarded NEH",
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return
        if not self._retained_cp_restore_loss_guard_allows(
            context_label="Guarded NEH",
            min_retained_cp_restore_loss_ratio=min_retained_cp_restore_loss_ratio,
            max_retained_cp_restore_loss_ratio=max_retained_cp_restore_loss_ratio,
            run_if_retained_cp_restore_missing=run_if_retained_cp_restore_missing,
        ):
            return
        if min_workload_size is not None and workload_size < int(min_workload_size):
            logging.info(
                "[Guarded NEH] Skipping: workload_size=%d is below min_workload_size=%d.",
                workload_size,
                int(min_workload_size),
            )
            return
        if max_workload_size is not None and workload_size > int(max_workload_size):
            logging.info(
                "[Guarded NEH] Skipping: workload_size=%d is above max_workload_size=%d.",
                workload_size,
                int(max_workload_size),
            )
            return
        if min_instance_job_count is not None and instance_job_count < int(
            min_instance_job_count
        ):
            logging.info(
                "[Guarded NEH] Skipping: job_count=%d is below min_instance_job_count=%d.",
                instance_job_count,
                int(min_instance_job_count),
            )
            return
        if max_instance_job_count is not None and instance_job_count > int(
            max_instance_job_count
        ):
            logging.info(
                "[Guarded NEH] Skipping: job_count=%d is above max_instance_job_count=%d.",
                instance_job_count,
                int(max_instance_job_count),
            )
            return
        if min_instance_stage_count is not None and instance_stage_count < int(
            min_instance_stage_count
        ):
            logging.info(
                "[Guarded NEH] Skipping: stage_count=%d is below min_instance_stage_count=%d.",
                instance_stage_count,
                int(min_instance_stage_count),
            )
            return
        if max_instance_stage_count is not None and instance_stage_count > int(
            max_instance_stage_count
        ):
            logging.info(
                "[Guarded NEH] Skipping: stage_count=%d is above max_instance_stage_count=%d.",
                instance_stage_count,
                int(max_instance_stage_count),
            )
            return
        logging.info(
            "[Guarded NEH] Running with job_count=%d stage_count=%d workload_size=%d "
            "inside guards workload=[%s, %s] job_count=[%s, %s] stage_count=[%s, %s].",
            instance_job_count,
            instance_stage_count,
            workload_size,
            min_workload_size,
            max_workload_size,
            min_instance_job_count,
            max_instance_job_count,
            min_instance_stage_count,
            max_instance_stage_count,
        )
        self.neh_cp(
            solver_thread_cnt=solver_thread_cnt,
            added_batch_size=added_batch_size,
            added_batch_count=added_batch_count,
            min_added_batch_count=min_added_batch_count,
            max_added_batch_count=max_added_batch_count,
            job_seq_by_1st_stage=job_seq_by_1st_stage,
            job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
            job_seq_by_last_retained_cp=job_seq_by_last_retained_cp,
            preserved_head_job_portion=preserved_head_job_portion,
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            profile_fix_min_batch_idx=profile_fix_min_batch_idx,
            profile_fix_min_batch_portion=profile_fix_min_batch_portion,
            profile_fix_max_batch_idx=profile_fix_max_batch_idx,
            profile_fix_by_machine_from_batch_idx=profile_fix_by_machine_from_batch_idx,
            profile_fix_min_job_age_batches=profile_fix_min_job_age_batches,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
            minimize_sum_ci_lex=minimize_sum_ci_lex,
            cp_tl_nc_multiplier_2nd_obj=cp_tl_nc_multiplier_2nd_obj,
            cp_tl_c_multiplier_2nd_obj=cp_tl_c_multiplier_2nd_obj,
            minimize_sum_ci_lin=minimize_sum_ci_lin,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
            make_semi_active_every_cp=make_semi_active_every_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
            stop_before_final_reserve=stop_before_final_reserve,
            min_remaining_sec_after_neh=min_remaining_sec_after_neh,
            min_remaining_nc_after_neh=min_remaining_nc_after_neh,
            time_guard_estimate_safety_factor=time_guard_estimate_safety_factor,
            time_guard_min_completed_batches=time_guard_min_completed_batches,
            skip_if_estimated_neh_exceeds_remaining=skip_if_estimated_neh_exceeds_remaining,
            full_neh_estimate_safety_factor=full_neh_estimate_safety_factor,
            log_cp_subproblem_bounds=log_cp_subproblem_bounds,
            log_cp_subproblem_progress=log_cp_subproblem_progress,
        )

    def workload_scaled_guarded_neh_cp(
        self,
        solver_thread_cnt: int,
        medium_workload_threshold: int = 1200,
        large_workload_threshold: int = 3000,
        small_added_batch_size: int = 10,
        medium_added_batch_size: int = 10,
        large_added_batch_size: int = 20,
        small_cp_tl_nc_multiplier: float | None = 0.015,
        medium_cp_tl_nc_multiplier: float | None = 0.030,
        large_cp_tl_nc_multiplier: float | None = 0.010,
        small_cp_tl_nc_multiplier_2nd_obj: float | None = 0.0025,
        medium_cp_tl_nc_multiplier_2nd_obj: float | None = 0.004,
        large_cp_tl_nc_multiplier_2nd_obj: float | None = 0.0025,
        small_max_added_batch_count: int | None = 4,
        medium_max_added_batch_count: int | None = 6,
        large_max_added_batch_count: int | None = 4,
        small_min_remaining_nc_after_neh: float | None = 0.65,
        medium_min_remaining_nc_after_neh: float | None = 0.55,
        large_min_remaining_nc_after_neh: float | None = 0.75,
        added_batch_count: int | None = None,
        min_added_batch_count: int | None = None,
        job_seq_by_1st_stage: bool = False,
        job_seq_by_bottleneck_stage: bool = False,
        job_seq_by_last_retained_cp: bool = False,
        preserved_head_job_portion: float = 0.0,
        max_time_per_add: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        profile_fix_min_batch_idx: int = 2,
        profile_fix_min_batch_portion: float | None = None,
        profile_fix_max_batch_idx: int | None = None,
        profile_fix_by_machine_from_batch_idx: int | None = None,
        profile_fix_min_job_age_batches: int | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        minimize_sum_ci_lex: bool = False,
        cp_tl_c_multiplier_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        make_semi_active_every_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        stop_before_final_reserve: bool = True,
        min_remaining_sec_after_neh: float | None = None,
        time_guard_estimate_safety_factor: float = 1.15,
        time_guard_min_completed_batches: int = 1,
        skip_if_estimated_neh_exceeds_remaining: bool = True,
        full_neh_estimate_safety_factor: float = 1.0,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
        min_incumbent_bound_gap_ratio: float | None = None,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
        min_retained_cp_restore_loss_ratio: float | None = None,
        max_retained_cp_restore_loss_ratio: float | None = None,
        run_if_retained_cp_restore_missing: bool = True,
    ) -> None:
        """Run one guarded NEH-CP repair with budget scaled by workload size."""
        workload_size = self._get_instance_workload_size()
        if workload_size >= int(large_workload_threshold):
            workload_band = "large"
            added_batch_size = large_added_batch_size
            cp_tl_nc_multiplier = large_cp_tl_nc_multiplier
            cp_tl_nc_multiplier_2nd_obj = large_cp_tl_nc_multiplier_2nd_obj
            max_added_batch_count = large_max_added_batch_count
            min_remaining_nc_after_neh = large_min_remaining_nc_after_neh
        elif workload_size >= int(medium_workload_threshold):
            workload_band = "medium"
            added_batch_size = medium_added_batch_size
            cp_tl_nc_multiplier = medium_cp_tl_nc_multiplier
            cp_tl_nc_multiplier_2nd_obj = medium_cp_tl_nc_multiplier_2nd_obj
            max_added_batch_count = medium_max_added_batch_count
            min_remaining_nc_after_neh = medium_min_remaining_nc_after_neh
        else:
            workload_band = "small"
            added_batch_size = small_added_batch_size
            cp_tl_nc_multiplier = small_cp_tl_nc_multiplier
            cp_tl_nc_multiplier_2nd_obj = small_cp_tl_nc_multiplier_2nd_obj
            max_added_batch_count = small_max_added_batch_count
            min_remaining_nc_after_neh = small_min_remaining_nc_after_neh

        logging.info(
            "[Workload Scaled NEH] workload_size=%d selected=%s "
            "added_batch_size=%d cp_tl_nc_multiplier=%s "
            "cp_tl_nc_multiplier_2nd_obj=%s max_added_batch_count=%s "
            "min_remaining_nc_after_neh=%s.",
            workload_size,
            workload_band,
            int(added_batch_size),
            cp_tl_nc_multiplier,
            cp_tl_nc_multiplier_2nd_obj,
            max_added_batch_count,
            min_remaining_nc_after_neh,
        )
        self.workload_guarded_neh_cp(
            solver_thread_cnt=solver_thread_cnt,
            added_batch_size=added_batch_size,
            added_batch_count=added_batch_count,
            min_added_batch_count=min_added_batch_count,
            max_added_batch_count=max_added_batch_count,
            job_seq_by_1st_stage=job_seq_by_1st_stage,
            job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
            job_seq_by_last_retained_cp=job_seq_by_last_retained_cp,
            preserved_head_job_portion=preserved_head_job_portion,
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            profile_fix_min_batch_idx=profile_fix_min_batch_idx,
            profile_fix_min_batch_portion=profile_fix_min_batch_portion,
            profile_fix_max_batch_idx=profile_fix_max_batch_idx,
            profile_fix_by_machine_from_batch_idx=profile_fix_by_machine_from_batch_idx,
            profile_fix_min_job_age_batches=profile_fix_min_job_age_batches,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
            minimize_sum_ci_lex=minimize_sum_ci_lex,
            cp_tl_nc_multiplier_2nd_obj=cp_tl_nc_multiplier_2nd_obj,
            cp_tl_c_multiplier_2nd_obj=cp_tl_c_multiplier_2nd_obj,
            minimize_sum_ci_lin=minimize_sum_ci_lin,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
            make_semi_active_every_cp=make_semi_active_every_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
            stop_before_final_reserve=stop_before_final_reserve,
            min_remaining_sec_after_neh=min_remaining_sec_after_neh,
            min_remaining_nc_after_neh=min_remaining_nc_after_neh,
            time_guard_estimate_safety_factor=time_guard_estimate_safety_factor,
            time_guard_min_completed_batches=time_guard_min_completed_batches,
            skip_if_estimated_neh_exceeds_remaining=skip_if_estimated_neh_exceeds_remaining,
            full_neh_estimate_safety_factor=full_neh_estimate_safety_factor,
            log_cp_subproblem_bounds=log_cp_subproblem_bounds,
            log_cp_subproblem_progress=log_cp_subproblem_progress,
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
            min_retained_cp_restore_loss_ratio=min_retained_cp_restore_loss_ratio,
            max_retained_cp_restore_loss_ratio=max_retained_cp_restore_loss_ratio,
            run_if_retained_cp_restore_missing=run_if_retained_cp_restore_missing,
        )

    def neh_cp_adaptive_preserved_head(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        added_batch_count: int | None = None,
        min_added_batch_count: int | None = None,
        max_added_batch_count: int | None = None,
        job_seq_by_1st_stage: bool = False,
        job_seq_by_bottleneck_stage: bool = False,
        small_preserved_head_job_portion: float = 0.25,
        large_preserved_head_job_portion: float = 0.40,
        large_workload_threshold: int = 1800,
        max_time_per_add: float | None = None,
        cp_tl_nc_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        profile_fix_min_batch_idx: int = 2,
        profile_fix_min_batch_portion: float | None = None,
        profile_fix_max_batch_idx: int | None = None,
        profile_fix_by_machine_from_batch_idx: int | None = None,
        profile_fix_min_job_age_batches: int | None = None,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        minimize_sum_ci_lex: bool = False,
        cp_tl_nc_multiplier_2nd_obj: float | None = None,
        cp_tl_c_multiplier_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        make_semi_active_every_cp: bool = False,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        stop_before_final_reserve: bool = True,
        min_remaining_sec_after_neh: float | None = None,
        min_remaining_nc_after_neh: float | None = None,
        time_guard_estimate_safety_factor: float = 1.15,
        time_guard_min_completed_batches: int = 1,
        skip_if_estimated_neh_exceeds_remaining: bool = True,
        full_neh_estimate_safety_factor: float = 1.0,
        log_cp_subproblem_bounds: bool = True,
        log_cp_subproblem_progress: bool = False,
    ) -> None:
        workload_size = self._get_instance_workload_size()
        preserved_head_job_portion = (
            float(large_preserved_head_job_portion)
            if workload_size >= int(large_workload_threshold)
            else float(small_preserved_head_job_portion)
        )
        logging.info(
            "[Adaptive NEH] workload_size=%d threshold=%d selected preserved_head_job_portion=%.3f "
            "(small=%.3f large=%.3f).",
            workload_size,
            int(large_workload_threshold),
            preserved_head_job_portion,
            float(small_preserved_head_job_portion),
            float(large_preserved_head_job_portion),
        )
        self.neh_cp(
            solver_thread_cnt=solver_thread_cnt,
            added_batch_size=added_batch_size,
            added_batch_count=added_batch_count,
            min_added_batch_count=min_added_batch_count,
            max_added_batch_count=max_added_batch_count,
            job_seq_by_1st_stage=job_seq_by_1st_stage,
            job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
            preserved_head_job_portion=preserved_head_job_portion,
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            profile_fix_min_batch_idx=profile_fix_min_batch_idx,
            profile_fix_min_batch_portion=profile_fix_min_batch_portion,
            profile_fix_max_batch_idx=profile_fix_max_batch_idx,
            profile_fix_by_machine_from_batch_idx=profile_fix_by_machine_from_batch_idx,
            profile_fix_min_job_age_batches=profile_fix_min_job_age_batches,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
            minimize_sum_ci_lex=minimize_sum_ci_lex,
            cp_tl_nc_multiplier_2nd_obj=cp_tl_nc_multiplier_2nd_obj,
            cp_tl_c_multiplier_2nd_obj=cp_tl_c_multiplier_2nd_obj,
            minimize_sum_ci_lin=minimize_sum_ci_lin,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
            make_semi_active_every_cp=make_semi_active_every_cp,
            use_lns_only=use_lns_only,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
            stop_before_final_reserve=stop_before_final_reserve,
            min_remaining_sec_after_neh=min_remaining_sec_after_neh,
            min_remaining_nc_after_neh=min_remaining_nc_after_neh,
            time_guard_estimate_safety_factor=time_guard_estimate_safety_factor,
            time_guard_min_completed_batches=time_guard_min_completed_batches,
            skip_if_estimated_neh_exceeds_remaining=skip_if_estimated_neh_exceeds_remaining,
            full_neh_estimate_safety_factor=full_neh_estimate_safety_factor,
            log_cp_subproblem_bounds=log_cp_subproblem_bounds,
            log_cp_subproblem_progress=log_cp_subproblem_progress,
        )

    def _resolve_pw_cp_batch_size(
        self,
        batch_size: int | None = None,
        batch_size_ratio: float | None = None,
    ) -> int:
        resolved_batch_size = 1
        if batch_size is not None:
            resolved_batch_size = max(1, batch_size)
            if batch_size_ratio is not None:
                logging.info(
                    "Ignoring batch_size_ratio=%s because explicit batch_size=%s was provided.",
                    batch_size_ratio,
                    batch_size,
                )
        elif batch_size_ratio is not None:
            resolved_batch_size = max(
                1, int(round(self.instance.job_count * batch_size_ratio))
            )
        return resolved_batch_size

    def _get_pw_cp_batch_count(
        self,
        schedule: HybridFlowshopLiteSchedule,
        batch_size: int,
    ) -> int:
        constructor = PwCpConstructor(self)
        stage_2_batch_list = constructor.build_stage_2_batch_list(
            schedule, batch_size=batch_size
        )
        return constructor.validate_and_get_batch_count(stage_2_batch_list)

    def incremental_pw_cp(
        self,
        solver_thread_cnt: int,
        batch_size: int | None = None,
        batch_size_ratio: float | None = None,
        step_size: int = 1,
        unfixed_batch_count_min: int = 1,
        unfixed_batch_count_max: int = 1,
        increment_unfixed_batch_count_flag: str = "always",
        lr_profile_fixed_batch_count: int = 0,
        left_profile_fixed_batch_count: int = 0,
        right_profile_fixed_batch_count: int = 0,
        enable_promotion_profile_fixed: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        non_time_fixed_op_time_limit_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        max_time_per_batch: float | None = None,
        use_lns_only: bool = False,
        debug_export: bool = False,
        tighten_ranges: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        if unfixed_batch_count_min < 1:
            raise ValueError("unfixed_batch_count_min must be >= 1")
        if unfixed_batch_count_max < unfixed_batch_count_min:
            raise ValueError(
                "unfixed_batch_count_max must be >= unfixed_batch_count_min"
            )
        if increment_unfixed_batch_count_flag not in {"always", "if_no_improvement"}:
            raise ValueError(
                "increment_unfixed_batch_count_flag must be one of "
                "{'always', 'if_no_improvement'}"
            )

        ref_schedule = self.solution_manager.get_incumbent()
        if ref_schedule is None:
            raise ValueError("No incumbent solution available for incremental PW-CP.")

        resolved_batch_size = self._resolve_pw_cp_batch_size(
            batch_size=batch_size,
            batch_size_ratio=batch_size_ratio,
        )
        actual_batch_count = self._get_pw_cp_batch_count(
            ref_schedule,
            batch_size=resolved_batch_size,
        )
        if unfixed_batch_count_min > actual_batch_count:
            raise ValueError(
                "unfixed_batch_count_min exceeds the available PW-CP batch count: "
                f"min={unfixed_batch_count_min}, available={actual_batch_count}"
            )

        effective_unfixed_batch_count_max = min(
            unfixed_batch_count_max, actual_batch_count
        )
        if effective_unfixed_batch_count_max < unfixed_batch_count_max:
            logging.info(
                "Clamping unfixed_batch_count_max from %d to %d based on available PW-CP batches.",
                unfixed_batch_count_max,
                effective_unfixed_batch_count_max,
            )

        base_pw_cp_kwargs = {
            "solver_thread_cnt": solver_thread_cnt,
            "batch_size": batch_size,
            "batch_size_ratio": batch_size_ratio,
            "step_size": step_size,
            "lr_profile_fixed_batch_count": lr_profile_fixed_batch_count,
            "left_profile_fixed_batch_count": left_profile_fixed_batch_count,
            "right_profile_fixed_batch_count": right_profile_fixed_batch_count,
            "enable_promotion_profile_fixed": enable_promotion_profile_fixed,
            "profile_fix_by_machine": profile_fix_by_machine,
            "machine_precedence_stride": machine_precedence_stride,
            "stage_precedence_min_processing_time_diff": stage_precedence_min_processing_time_diff,
            "stage_precedence_min_processing_time_diff_ratio": stage_precedence_min_processing_time_diff_ratio,
            "non_time_fixed_op_time_limit_multiplier": non_time_fixed_op_time_limit_multiplier,
            "cp_tl_c_multiplier": cp_tl_c_multiplier,
            "max_time_per_batch": max_time_per_batch,
            "use_lns_only": use_lns_only,
            "debug_export": debug_export,
            "tighten_ranges": tighten_ranges,
            "error_if_infeasible": error_if_infeasible,
            "draw_gantt": draw_gantt,
        }

        logging.info(
            "Incremental PW-CP starts: policy=%s, unfixed_batch_count=[%d, %d], "
            "resolved_batch_size=%d, actual_batch_count=%d.",
            increment_unfixed_batch_count_flag,
            unfixed_batch_count_min,
            effective_unfixed_batch_count_max,
            resolved_batch_size,
            actual_batch_count,
        )

        for unfixed_batch_count in range(
            unfixed_batch_count_min,
            effective_unfixed_batch_count_max + 1,
        ):
            if self.is_stopping_condition():
                logging.info(
                    "Stopping condition met before incremental PW-CP count=%d.",
                    unfixed_batch_count,
                )
                break

            current_pw_cp_kwargs = {
                **base_pw_cp_kwargs,
                "unfixed_batch_count": unfixed_batch_count,
            }
            context_name = f"batch_{unfixed_batch_count:03d}"

            with self.temporarily_extended_context(context_name):
                if increment_unfixed_batch_count_flag == "if_no_improvement":
                    logging.info(
                        "Incremental PW-CP repeats count=%d until the first non-improving pass.",
                        unfixed_batch_count,
                    )
                    self.repeat_while_improvement(
                        routine_data=DynamicDataObject.from_obj(
                            {
                                "method": "pw_cp",
                                **current_pw_cp_kwargs,
                            }
                        ),
                        n_repeats=None,
                        max_no_improve=0,
                    )
                else:
                    logging.info(
                        "Incremental PW-CP runs one pass at count=%d.",
                        unfixed_batch_count,
                    )
                    self.pw_cp(**current_pw_cp_kwargs)

    def bound_gap_guarded_incremental_pw_cp(
        self,
        solver_thread_cnt: int,
        min_incumbent_bound_gap_ratio: float | None = 0.0,
        max_incumbent_bound_gap_ratio: float | None = None,
        run_if_bound_missing: bool = True,
        batch_size: int | None = None,
        batch_size_ratio: float | None = None,
        step_size: int = 1,
        unfixed_batch_count_min: int = 1,
        unfixed_batch_count_max: int = 1,
        increment_unfixed_batch_count_flag: str = "always",
        lr_profile_fixed_batch_count: int = 0,
        left_profile_fixed_batch_count: int = 0,
        right_profile_fixed_batch_count: int = 0,
        enable_promotion_profile_fixed: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        non_time_fixed_op_time_limit_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        max_time_per_batch: float | None = None,
        use_lns_only: bool = False,
        debug_export: bool = False,
        tighten_ranges: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Run PW-CP only when incumbent-vs-bound gap is inside a configured band."""
        if not self._bound_gap_guard_allows(
            context_label="Bound-gap PW-CP",
            min_incumbent_bound_gap_ratio=min_incumbent_bound_gap_ratio,
            max_incumbent_bound_gap_ratio=max_incumbent_bound_gap_ratio,
            run_if_bound_missing=run_if_bound_missing,
        ):
            return

        self.incremental_pw_cp(
            solver_thread_cnt=solver_thread_cnt,
            batch_size=batch_size,
            batch_size_ratio=batch_size_ratio,
            step_size=step_size,
            unfixed_batch_count_min=unfixed_batch_count_min,
            unfixed_batch_count_max=unfixed_batch_count_max,
            increment_unfixed_batch_count_flag=increment_unfixed_batch_count_flag,
            lr_profile_fixed_batch_count=lr_profile_fixed_batch_count,
            left_profile_fixed_batch_count=left_profile_fixed_batch_count,
            right_profile_fixed_batch_count=right_profile_fixed_batch_count,
            enable_promotion_profile_fixed=enable_promotion_profile_fixed,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
            non_time_fixed_op_time_limit_multiplier=non_time_fixed_op_time_limit_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            max_time_per_batch=max_time_per_batch,
            use_lns_only=use_lns_only,
            debug_export=debug_export,
            tighten_ranges=tighten_ranges,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def pw_cp(
        self,
        solver_thread_cnt: int,
        batch_size: int | None = None,
        batch_size_ratio: float | None = None,
        step_size: int = 1,  # Sliding window: step size for window movement
        unfixed_batch_count: int = 1,  # Sliding window: unfixed batch count
        lr_profile_fixed_batch_count: int = 0,
        left_profile_fixed_batch_count: int = 0,
        right_profile_fixed_batch_count: int = 0,
        enable_promotion_profile_fixed: bool = False,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        stage_precedence_min_processing_time_diff: int | None = None,
        stage_precedence_min_processing_time_diff_ratio: float | None = None,
        non_time_fixed_op_time_limit_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        max_time_per_batch: float | None = None,
        use_lns_only: bool = False,
        debug_export: bool = False,
        tighten_ranges: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        sub_timer = ElapsedTimer()

        ref_schedule = self.solution_manager.get_incumbent()
        if ref_schedule is None:
            raise ValueError("No incumbent solution available for PW-CP.")

        resolved_batch_size = self._resolve_pw_cp_batch_size(
            batch_size=batch_size,
            batch_size_ratio=batch_size_ratio,
        )

        if lr_profile_fixed_batch_count > 0:
            # Override
            _left_profile_fixed_batch_count = lr_profile_fixed_batch_count
            _right_profile_fixed_batch_count = lr_profile_fixed_batch_count
        else:
            if left_profile_fixed_batch_count < 0:
                raise ValueError("left_profile_fixed_batch_count must be >= 0")
            if right_profile_fixed_batch_count < 0:
                raise ValueError("right_profile_fixed_batch_count must be >= 0")
            _left_profile_fixed_batch_count = left_profile_fixed_batch_count
            _right_profile_fixed_batch_count = right_profile_fixed_batch_count

        if (
            non_time_fixed_op_time_limit_multiplier is not None
            and non_time_fixed_op_time_limit_multiplier <= 0
        ):
            raise ValueError("non_time_fixed_op_time_limit_multiplier must be > 0")

        if non_time_fixed_op_time_limit_multiplier is not None:
            _max_time_per_batch = max_time_per_batch
        elif cp_tl_c_multiplier is not None:
            _max_time_per_batch = cp_tl_c_multiplier * self.instance.stage_count
        else:
            _max_time_per_batch = max_time_per_batch

        constructor = PwCpConstructor(self)
        result: PwCpResult = constructor.run(
            ref_schedule,
            self.instance,
            self.stage_2_job_2_p_dict,
            batch_size=resolved_batch_size,
            step_size=step_size,
            unfixed_batch_count=unfixed_batch_count,
            left_profile_fixed_batch_count=_left_profile_fixed_batch_count,
            right_profile_fixed_batch_count=_right_profile_fixed_batch_count,
            enable_promotion_profile_fixed=enable_promotion_profile_fixed,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            stage_precedence_min_processing_time_diff=stage_precedence_min_processing_time_diff,
            stage_precedence_min_processing_time_diff_ratio=stage_precedence_min_processing_time_diff_ratio,
            non_time_fixed_op_time_limit_multiplier=non_time_fixed_op_time_limit_multiplier,
            max_time_per_batch=_max_time_per_batch,
            solver_thread_cnt=solver_thread_cnt,
            use_lns_only=use_lns_only,
            debug_export=debug_export,
            tighten_ranges=tighten_ranges,
            error_if_infeasible=error_if_infeasible,
        )
        obj_value = float(result.schedule.makespan)
        logging.info("PW-CP done with makespan %s", obj_value)
        if debug_export and result.sub_obj_store:
            result.save_yaml(self.get_file_path_for_subroutine("_obj_log.yaml"))

        final_report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
            subroutine_name="pw_cp",
            progress_obj_value_records=result.sub_obj_store.obj_value_series.items(),
        )
        was_updated: bool = self.solution_manager.register(
            final_report, result.schedule
        )

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        if was_updated:
            self.set_cp_model_as_base_cp_model()
            if draw_gantt:
                self.draw_incumbent_gantt()

    def run_reactive_loop(
        self,
        routine_data: list[dict],
        reactive_param_tuner_dict: dict,
        stopping_criteria: dict,
    ):
        looper = ReactiveLooper(
            self,
            routine_data,
            reactive_param_tuner_dict,
            stopping_criteria,
        )
        try:
            looper.run()
        finally:
            # TODO: suffix from output_metadata
            csv_report_path = self.get_file_path_for_subroutine("_report.csv")
            looper.write_report_csv(csv_report_path)
            # yaml_report_path = self.get_file_path_for_subroutine("_report.yaml")
            # looper.write_report_yaml(yaml_report_path)

    # Subroutine: PRTS by Zhou et al. (2024)

    def prts(
        self,
        population_multiplier: int = 2,
        operator_iterations: int = 20,
        similarity_threshold: float = 0.7,
        alpha: int = 5,
        pr_ts_iterations: int = 500,
        tt: int = 2,
        d_1: int = 5,
        d_2: int = 5,
        tabu_list_length_multiplier: int = 1,
        ts_max_iterations_multiplier: int = 100,
        a_hat: float | None = None,
        use_stage_cnt_for_tl: bool = False,
        error_if_infeasible: bool = False,
    ) -> None:
        """
        Run PRTS main loop until controller time limit.

        Output:
        - incumbent schedule is registered into solution_manager
        - obj_store logs updated when incumbent improves
        """
        from .zhou_2024 import PrTs2024Runner

        sub_timer = ElapsedTimer()

        if a_hat is None:
            if hasattr(self.stopping_criteria, "timelimit_n_by_c_multiplier"):
                a_hat = self.stopping_criteria.timelimit_n_by_c_multiplier
            else:
                a_hat = 0.5

        runner = PrTs2024Runner(self.stage_2_job_2_p_dict, self.instance)
        result = runner.run(
            population_multiplier=population_multiplier,
            operator_iterations=operator_iterations,
            similarity_threshold=similarity_threshold,
            alpha=alpha,
            pr_ts_iterations=pr_ts_iterations,
            tt=tt,
            d_1=d_1,
            d_2=d_2,
            tabu_list_length_multiplier=tabu_list_length_multiplier,
            ts_max_iterations_multiplier=ts_max_iterations_multiplier,
            a_hat=a_hat,
            use_stage_cnt_for_tl=use_stage_cnt_for_tl,
        )
        solution = result.schedule
        obj_value: int | float = result.last_obj_value
        obj_value_records = result.sub_obj_store.obj_value_series.items()

        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="prts",
            progress_obj_value_records=obj_value_records,
        )

        # Register report & solution
        self.solution_manager.register(report, solution)

        # Log (time, objective value & bound)
        log_time = self.timer.elapsed_sec
        _last_timestamp_note = self._get_call_context_of_current_method()

        self.extend_obj_value_log(obj_value_records, is_maximize=False)
        last_logged_obj_value = self.obj_store.get_last_obj_value()
        obj_value_is_valid = False
        if last_logged_obj_value is not None:
            self.add_obj_value_log(log_time, last_logged_obj_value, is_maximize=None)
            obj_value_is_valid = True

        obj_bound = self.obj_store.get_last_obj_bound()
        obj_bound_is_valid = False
        if obj_bound is not None:
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=None)
            obj_bound_is_valid = True

        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

    # Subroutine: bottleneck stage centric initialization

    def bn2d_single_stage(
        self,
        left_cap_multiplier: int | None = None,
        right_cap_multiplier: int | None = None,
        left_cap_portion: float | None = None,
        right_cap_portion: float | None = None,
        normalize_by_stage_cnt: bool = False,
        randomize_mid_all: bool = False,
        reverse_mid_all: bool = False,
        reverse_mid_even: bool = False,
        machine_then_job: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Schedule from single bottleneck stage using BN2D option."""
        sub_timer = ElapsedTimer()
        option = BN2DOption(
            left_cap_multiplier=left_cap_multiplier,
            right_cap_multiplier=right_cap_multiplier,
            left_cap_portion=left_cap_portion,
            right_cap_portion=right_cap_portion,
            normalize_by_stage_cnt=normalize_by_stage_cnt,
            reverse_mid_all=reverse_mid_all,
            reverse_mid_even=reverse_mid_even,
            randomize_mid_all=randomize_mid_all,
            machine_then_job=machine_then_job,
        )
        dispatcher = BN2DDispatcher(self.instance)

        schedule = dispatcher.get_schedule_by_bn2d_single_stage(
            option=option,
            gantt_draw_func=self.draw_gantt if draw_gantt else None,
        )

        if schedule is None:
            raise RuntimeError("Failed to create schedule by BN2D")
        complete_makespan = schedule.makespan
        logging.info(f"Bottleneck parallel MC: full_schedule_obj={complete_makespan}")

        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=complete_makespan,
            obj_bound=None,
            is_init=True,
            subroutine_name="bn2d_single_stage",
            progress_obj_value_records=[
                (sub_timer.elapsed_sec, float(complete_makespan))
            ],
        )
        self.solution_manager.register(report, schedule)

    def bn2d_all_stage(
        self,
        left_cap_multiplier: int | None = None,
        right_cap_multiplier: int | None = None,
        left_cap_portion: float | None = None,
        right_cap_portion: float | None = None,
        normalize_by_stage_cnt: bool = False,
        randomize_mid_all: bool = False,
        reverse_mid_all: bool = False,
        reverse_mid_even: bool = False,
        mixed_schedule_for_former_stages: bool = False,
        mixed_schedule_for_later_stages: bool = False,
        machine_then_job: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Schedule from all stages as bottleneck using BN2D option."""
        sub_timer = ElapsedTimer()
        option = BN2DOption(
            left_cap_multiplier=left_cap_multiplier,
            right_cap_multiplier=right_cap_multiplier,
            left_cap_portion=left_cap_portion,
            right_cap_portion=right_cap_portion,
            normalize_by_stage_cnt=normalize_by_stage_cnt,
            reverse_mid_all=reverse_mid_all,
            reverse_mid_even=reverse_mid_even,
            randomize_mid_all=randomize_mid_all,
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
        )
        schedule = self._get_schedule_by_bn2d_all_stages(
            option=option, draw_gantt=draw_gantt
        )

        if schedule is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        best_obj = schedule.makespan
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
            subroutine_name="bn2d_all_stages",
            progress_obj_value_records=[(sub_timer.elapsed_sec, float(best_obj))],
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_obj), is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_bn2d_all_stages(
        self,
        option: BN2DOption,
        draw_gantt: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        gantt_draw_func = self.draw_gantt if draw_gantt else None
        dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            BN2DDispatcher,
            self.instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(
            option=option, gantt_draw_func=gantt_draw_func
        )
        if schedule is not None:
            logging.info(f"BN2D all stages: makespan={schedule.makespan}")

        reversed_dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            BN2DDispatcher,
            reverse_stages(self.instance),
            job_tiebreak_rank=job_tiebreak_rank,
        )
        reversed_schedule = reversed_dispatcher.get_schedule_by_bn2d_all_stages(
            option=option, gantt_draw_func=gantt_draw_func
        )
        if reversed_schedule is not None:
            logging.info(
                "BN2D all stages on reversed instance"
                f": makespan={reversed_schedule.makespan}"
            )

        if reversed_schedule is not None and (
            schedule is None or schedule.makespan > reversed_schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(self.stage_2_job_2_p_dict)
            return converted
        return schedule

    # Dispatch

    @staticmethod
    def _create_dispatcher_with_optional_job_tiebreak(
        dispatcher_cls: type,
        *args: Any,
        job_tiebreak_rank: Mapping[str, int] | None = None,
        **kwargs: Any,
    ) -> Any:
        if job_tiebreak_rank is None:
            return dispatcher_cls(*args, **kwargs)
        return dispatcher_cls(
            *args,
            job_tiebreak_rank=job_tiebreak_rank,
            **kwargs,
        )

    def _from_job_sequence_get_schedule_mixed(
        self,
        job_sequence: Sequence[str],
        stage_2_head: Mapping[str, int],
        prob_instance: HybridFlowshopParameters | None = None,
        job_2_release: dict[str, int] | None = None,
        machine_then_job: bool = False,
        draw_gantt_per_step: bool = False,
    ) -> HybridFlowshopLiteSchedule:
        schedule = self.create_empty_schedule_from_ins(instance=prob_instance)
        from_job_sequence_get_schedule_mixed(
            schedule,
            job_sequence,
            self.stage_2_job_2_p_dict,
            stage_2_head,
            job_2_release=job_2_release,
            machine_then_job=machine_then_job,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=(
                self.get_file_path_for_subroutine if draw_gantt_per_step else None
            ),
        )
        return schedule

    def get_sample_schedule_by_cds(
        self,
        np: int,
        k: int,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt_per_step: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        sub_timer = ElapsedTimer()

        if head_for_all_stages:
            stage_2_head = {stage_id: np for stage_id in self.instance.stage_id_list}
        else:
            stage_2_head = {self.instance.stage_id_list[0]: np}

        stage_id = self.instance.stage_id_list[k]
        job_sequence = self.get_cds_sequence(k)
        dispatched_schedule = self._from_job_sequence_get_schedule_mixed(
            job_sequence,
            stage_2_head,
            machine_then_job=machine_then_job,
            draw_gantt_per_step=draw_gantt_per_step,
        )
        best_obj = dispatched_schedule.makespan

        if best_obj is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(dispatched_schedule.get_jik_2_start_time_map())

        logging.info(
            f"CDS sequence: makespan={best_obj} at k={k}, CDS stage={stage_id}"
        )

        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
            subroutine_name="get_sample_schedule_by_cds",
            progress_obj_value_records=[(sub_timer.elapsed_sec, float(best_obj))],
        )
        was_updated = self.solution_manager.register(report, dispatched_schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_obj), is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_schedule_by_cds(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt_per_step: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        sub_timer = ElapsedTimer()

        best_sch = self._get_schedule_by_cds(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
        )

        if best_sch is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())

        best_obj = best_sch.makespan
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_schedule_by_cds",
            progress_obj_value_records=[(sub_timer.elapsed_sec, float(best_obj))],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_obj), is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_cds(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        draw_gantt_per_step: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        # Dispatch on the original problem
        dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            self.instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        schedule = dispatcher.get_schedule_by_cds(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=(
                self.get_file_path_for_subroutine if draw_gantt_per_step else None
            ),
        )
        if schedule is not None:
            logging.info(
                f"Best of mixed schedule by CDS sequence: objValue={schedule.makespan}"
            )
        # Dispatch on the reversed problem
        reversed_dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            reverse_stages(self.instance),
            job_tiebreak_rank=job_tiebreak_rank,
        )
        reversed_schedule = reversed_dispatcher.get_schedule_by_cds(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=(
                self.get_file_path_for_subroutine if draw_gantt_per_step else None
            ),
        )
        if reversed_schedule is not None:
            logging.info(
                "Best of mixed schedule by CDS sequence on reversed instance"
                f": objValue={reversed_schedule.makespan}"
            )
        if reversed_schedule is not None and (
            schedule is None or schedule.makespan > reversed_schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(self.stage_2_job_2_p_dict)
            return converted
        return schedule

    def initialize_schedule_by_gupta(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt_per_step: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        sub_timer = ElapsedTimer()

        best_sch = self._get_schedule_by_gupta(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
        )

        if best_sch is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())

        best_obj = best_sch.makespan
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_schedule_by_gupta",
            progress_obj_value_records=[(sub_timer.elapsed_sec, float(best_obj))],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_obj), is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_gupta(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        draw_gantt_per_step: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            self.instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        schedule = dispatcher.get_schedule_by_gupta(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=(
                self.get_file_path_for_subroutine if draw_gantt_per_step else None
            ),
        )
        if schedule is not None:
            logging.info(
                f"Best of mixed schedule by Gupta sequence: makespan={schedule.makespan}"
            )

        reversed_dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            reverse_stages(self.instance),
            job_tiebreak_rank=job_tiebreak_rank,
        )
        reversed_schedule = reversed_dispatcher.get_schedule_by_gupta(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=(
                self.get_file_path_for_subroutine if draw_gantt_per_step else None
            ),
        )
        if reversed_schedule is not None:
            logging.info(
                "Best of mixed schedule by Gupta sequence on reversed instance"
                f": makespan={reversed_schedule.makespan}"
            )

        if reversed_schedule is not None and (
            schedule is None or schedule.makespan > reversed_schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(self.stage_2_job_2_p_dict)
            return converted
        return schedule

    def initialize_schedule_by_palmer(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt_per_step: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        sub_timer = ElapsedTimer()

        best_sch = self._get_schedule_by_palmer(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
        )

        if best_sch is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())

        best_obj = best_sch.makespan
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_schedule_by_palmer",
            progress_obj_value_records=[(sub_timer.elapsed_sec, float(best_obj))],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_obj), is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_palmer(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        draw_gantt_per_step: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            self.instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        schedule = dispatcher.get_schedule_by_palmer(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=(
                self.get_file_path_for_subroutine if draw_gantt_per_step else None
            ),
        )
        if schedule is not None:
            logging.info(
                f"Best of mixed schedule by Palmer sequence: makespan={schedule.makespan}"
            )

        reversed_dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            reverse_stages(self.instance),
            job_tiebreak_rank=job_tiebreak_rank,
        )
        reversed_schedule = reversed_dispatcher.get_schedule_by_palmer(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=(
                self.get_file_path_for_subroutine if draw_gantt_per_step else None
            ),
        )
        if reversed_schedule is not None:
            logging.info(
                "Best of mixed schedule by Palmer sequence on reversed instance"
                f": makespan={reversed_schedule.makespan}"
            )

        if reversed_schedule is not None and (
            schedule is None or schedule.makespan > reversed_schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(self.stage_2_job_2_p_dict)
            return converted
        return schedule

    def initialize_by_best_of_mixed_dispatches(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        sub_timer = ElapsedTimer()

        best_sch = self._get_schedule_by_best_of_mixed_dispatches(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
        )

        if best_sch is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())
        best_obj = best_sch.makespan
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_best_of_mixed_dispatches",
            progress_obj_value_records=[(sub_timer.elapsed_sec, float(best_obj))],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_obj), is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_best_of_mixed_dispatches(
        self,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        use_palmer_index: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
        mixed_dispatch_methods: Sequence[str] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        schedule_gen_methods = self._resolve_mixed_dispatch_methods(
            mixed_dispatch_methods
        )

        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None
        best_method_name = ""

        for method in schedule_gen_methods:
            logging.info(f"Generating schedule using {method.__name__}")
            sch = method(
                machine_then_job=machine_then_job,
                head_for_all_stages=head_for_all_stages,
                use_palmer_index=use_palmer_index,
                job_tiebreak_rank=job_tiebreak_rank,
            )
            if sch is not None:
                obj = sch.makespan
                logging.info(f"  -> Schedule makespan: {obj}")
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_sch = sch
                    best_method_name = method.__name__

        if best_sch is not None:
            logging.info(
                "Best mixed-dispatch schedule generated by %s with makespan %s",
                best_method_name,
                best_obj,
            )
        return best_sch

    def _resolve_mixed_dispatch_methods(
        self,
        mixed_dispatch_methods: Sequence[str] | None = None,
    ) -> list[Callable[..., HybridFlowshopLiteSchedule | None]]:
        if mixed_dispatch_methods is None:
            return [
                self._get_schedule_by_cds,
                self._get_schedule_by_gupta,
                self._get_schedule_by_palmer,
            ]

        method_lookup: dict[str, Callable[..., HybridFlowshopLiteSchedule | None]] = {
            "cds": self._get_schedule_by_cds,
            "gupta": self._get_schedule_by_gupta,
            "palmer": self._get_schedule_by_palmer,
        }
        resolved_methods = []
        for method_name in mixed_dispatch_methods:
            normalized_name = str(method_name).strip().lower()
            if normalized_name not in method_lookup:
                raise ValueError(
                    "mixed_dispatch_methods entries must be one of "
                    f"{sorted(method_lookup)}; got {method_name!r}."
                )
            if method_lookup[normalized_name] not in resolved_methods:
                resolved_methods.append(method_lookup[normalized_name])

        if not resolved_methods:
            raise ValueError("mixed_dispatch_methods must not be empty.")
        return resolved_methods

    def _get_best_mixed_schedule_from_job_sequence(
        self,
        job_sequence: Sequence[str],
        *,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
    ) -> HybridFlowshopLiteSchedule | None:
        dispatcher = MixedDispatcher(self.instance)
        schedule = dispatcher.get_best_mixed_schedule_by_sequence(
            job_sequence,
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
        )

        reversed_dispatcher = MixedDispatcher(reverse_stages(self.instance))
        reversed_schedule = reversed_dispatcher.get_best_mixed_schedule_by_sequence(
            job_sequence,
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
        )
        if reversed_schedule is not None:
            reversed_schedule = reversed_schedule.as_reversed()
            reversed_schedule.make_semi_active(self.stage_2_job_2_p_dict)

        if schedule is None:
            return reversed_schedule
        if reversed_schedule is None or schedule.makespan <= reversed_schedule.makespan:
            return schedule
        return reversed_schedule

    @staticmethod
    def _get_default_selected_dispatch_method_list() -> list[str]:
        return ["bn2d_all_stages", "best_of_mixed_dispatches"]

    def _build_selected_dispatch_config(
        self,
        *,
        left_cap_multiplier: int | None = None,
        right_cap_multiplier: int | None = None,
        left_cap_portion: float | None = None,
        right_cap_portion: float | None = None,
        normalize_by_stage_cnt: bool = False,
        randomize_mid_all: bool = False,
        reverse_mid_all: bool = False,
        reverse_mid_even: bool = False,
        mixed_schedule_for_former_stages: bool = False,
        mixed_schedule_for_later_stages: bool = False,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        p_agg_method: str = "sum",
        mi_agg_method: str = "max",
        method_list: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "left_cap_multiplier": left_cap_multiplier,
            "right_cap_multiplier": right_cap_multiplier,
            "left_cap_portion": left_cap_portion,
            "right_cap_portion": right_cap_portion,
            "normalize_by_stage_cnt": normalize_by_stage_cnt,
            "randomize_mid_all": randomize_mid_all,
            "reverse_mid_all": reverse_mid_all,
            "reverse_mid_even": reverse_mid_even,
            "mixed_schedule_for_former_stages": mixed_schedule_for_former_stages,
            "mixed_schedule_for_later_stages": mixed_schedule_for_later_stages,
            "machine_then_job": machine_then_job,
            "head_for_all_stages": head_for_all_stages,
            "p_agg_method": p_agg_method,
            "mi_agg_method": mi_agg_method,
            "method_list": (
                list(method_list)
                if method_list
                else self._get_default_selected_dispatch_method_list()
            ),
        }

    def _get_selected_dispatch_config_for_post_mip(self) -> dict[str, Any]:
        saved_config = getattr(self, "last_selected_dispatch_config", None)
        if saved_config is None:
            return self._build_selected_dispatch_config()
        config = dict(saved_config)
        method_list = (
            list(config["method_list"])
            if config.get("method_list")
            else self._get_default_selected_dispatch_method_list()
        )
        if method_list == ["bn2d_all_stages"]:
            method_list = self._get_default_selected_dispatch_method_list()
        config["method_list"] = method_list
        return config

    def _get_selected_dispatch_candidate_schedules(
        self,
        *,
        left_cap_multiplier: int | None = None,
        right_cap_multiplier: int | None = None,
        left_cap_portion: float | None = None,
        right_cap_portion: float | None = None,
        normalize_by_stage_cnt: bool = False,
        randomize_mid_all: bool = False,
        reverse_mid_all: bool = False,
        reverse_mid_even: bool = False,
        mixed_schedule_for_former_stages: bool = False,
        mixed_schedule_for_later_stages: bool = False,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        p_agg_method: str = "sum",
        mi_agg_method: str = "max",
        method_list: list[str] | None = None,
        draw_gantt: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
        candidate_elapsed_sec_out: dict[str, float] | None = None,
    ) -> dict[str, HybridFlowshopLiteSchedule | None]:
        option = BN2DOption(
            left_cap_multiplier=left_cap_multiplier,
            right_cap_multiplier=right_cap_multiplier,
            left_cap_portion=left_cap_portion,
            right_cap_portion=right_cap_portion,
            normalize_by_stage_cnt=normalize_by_stage_cnt,
            randomize_mid_all=randomize_mid_all,
            reverse_mid_all=reverse_mid_all,
            reverse_mid_even=reverse_mid_even,
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
        )
        candidate_schedules: dict[str, HybridFlowshopLiteSchedule | None] = {}
        selected_methods = (
            list(method_list)
            if method_list
            else self._get_default_selected_dispatch_method_list()
        )
        for method_name in selected_methods:
            candidate_timer = ElapsedTimer()
            sch: HybridFlowshopLiteSchedule | None = None
            if method_name == "bn2d_all_stages":
                sch = self._get_schedule_by_bn2d_all_stages(
                    option=option,
                    draw_gantt=draw_gantt,
                    job_tiebreak_rank=job_tiebreak_rank,
                )
            elif method_name == "best_of_mixed_dispatches":
                sch = self._get_schedule_by_best_of_mixed_dispatches(
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    job_tiebreak_rank=job_tiebreak_rank,
                )
            elif method_name == "stage_agg_2":
                sch = self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
                    stage_agg_count=2,
                    head_stages_to_keep=0,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    draw_gantt_per_step=False,
                    job_tiebreak_rank=job_tiebreak_rank,
                )
            elif method_name == "stage_agg_2_1":
                sch = self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
                    stage_agg_count=2,
                    head_stages_to_keep=1,
                    tail_stages_to_keep=1,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    draw_gantt_per_step=False,
                    job_tiebreak_rank=job_tiebreak_rank,
                )
            elif method_name == "stage_agg_2_2":
                sch = self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
                    stage_agg_count=2,
                    head_stages_to_keep=2,
                    tail_stages_to_keep=2,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    draw_gantt_per_step=False,
                    job_tiebreak_rank=job_tiebreak_rank,
                )
            else:
                logging.warning(
                    "[MIP LB] Unknown selected dispatch method '%s'; skipping.",
                    method_name,
                )
            candidate_schedules[method_name] = sch
            if candidate_elapsed_sec_out is not None:
                candidate_elapsed_sec_out[method_name] = candidate_timer.elapsed_sec
            logging.info(
                "%s: makespan=%s",
                method_name,
                sch.makespan if sch is not None else None,
            )
        return candidate_schedules

    def initialize_by_best_of_selected_dispatches(
        self,
        left_cap_multiplier: int | None = None,
        right_cap_multiplier: int | None = None,
        left_cap_portion: float | None = None,
        right_cap_portion: float | None = None,
        normalize_by_stage_cnt: bool = False,
        randomize_mid_all: bool = False,
        reverse_mid_all: bool = False,
        reverse_mid_even: bool = False,
        mixed_schedule_for_former_stages: bool = False,
        mixed_schedule_for_later_stages: bool = False,
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        p_agg_method: str = "sum",
        mi_agg_method: str = "max",
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
        method_list: list[str] | None = None,
    ) -> None:
        sub_timer = ElapsedTimer()

        selected_dispatch_config = self._build_selected_dispatch_config(
            left_cap_multiplier=left_cap_multiplier,
            right_cap_multiplier=right_cap_multiplier,
            left_cap_portion=left_cap_portion,
            right_cap_portion=right_cap_portion,
            normalize_by_stage_cnt=normalize_by_stage_cnt,
            randomize_mid_all=randomize_mid_all,
            reverse_mid_all=reverse_mid_all,
            reverse_mid_even=reverse_mid_even,
            mixed_schedule_for_former_stages=mixed_schedule_for_former_stages,
            mixed_schedule_for_later_stages=mixed_schedule_for_later_stages,
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            p_agg_method=p_agg_method,
            mi_agg_method=mi_agg_method,
            method_list=method_list,
        )
        self.last_selected_dispatch_config = dict(selected_dispatch_config)

        candidate_schedules = self._get_selected_dispatch_candidate_schedules(
            left_cap_multiplier=selected_dispatch_config["left_cap_multiplier"],
            right_cap_multiplier=selected_dispatch_config["right_cap_multiplier"],
            left_cap_portion=selected_dispatch_config["left_cap_portion"],
            right_cap_portion=selected_dispatch_config["right_cap_portion"],
            normalize_by_stage_cnt=selected_dispatch_config["normalize_by_stage_cnt"],
            randomize_mid_all=selected_dispatch_config["randomize_mid_all"],
            reverse_mid_all=selected_dispatch_config["reverse_mid_all"],
            reverse_mid_even=selected_dispatch_config["reverse_mid_even"],
            mixed_schedule_for_former_stages=selected_dispatch_config[
                "mixed_schedule_for_former_stages"
            ],
            mixed_schedule_for_later_stages=selected_dispatch_config[
                "mixed_schedule_for_later_stages"
            ],
            machine_then_job=selected_dispatch_config["machine_then_job"],
            head_for_all_stages=selected_dispatch_config["head_for_all_stages"],
            p_agg_method=selected_dispatch_config["p_agg_method"],
            mi_agg_method=selected_dispatch_config["mi_agg_method"],
            method_list=selected_dispatch_config["method_list"],
            draw_gantt=draw_gantt,
        )

        best_method_name = ""
        best_sch: HybridFlowshopLiteSchedule | None = None
        best_obj: int | None = None
        for method_name, sch in candidate_schedules.items():
            obj = sch.makespan if sch is not None else None
            if obj is not None and (best_obj is None or obj < best_obj):
                best_sch = sch
                best_obj = obj
                best_method_name = method_name

        if best_sch is None:
            # Failed to find a solution
            return
        logging.info(
            f"Best schedule generated by {best_method_name} with makespan {best_obj}"
        )
        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())

        best_obj = best_sch.makespan
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_best_of_selected_dispatches",
            progress_obj_value_records=[(sub_timer.elapsed_sec, float(best_obj))],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_obj), is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_dispatch_portfolio(
        self,
        portfolio: str = "balanced",
        include_stage_agg: bool = True,
        cap_portions: Sequence[float] | None = None,
        include_mid_order_variants: bool = False,
        randomized_mid_trials: int = 0,
        randomized_tiebreak_trials: int = 0,
        selection_strategy: str = "best",
        selection_obj_slack: float = 0.0,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Initialize from a compact portfolio of dispatch and aggregation variants."""
        sub_timer = ElapsedTimer()
        core_methods = ["bn2d_all_stages", "best_of_mixed_dispatches"]
        bn2d_methods = ["bn2d_all_stages"]
        mixed_methods = ["best_of_mixed_dispatches"]
        stage_agg_methods = [
            "bn2d_all_stages",
            "best_of_mixed_dispatches",
            "stage_agg_2",
            "stage_agg_2_1",
            "stage_agg_2_2",
        ]
        stage_agg_only_methods = [
            "stage_agg_2",
            "stage_agg_2_1",
            "stage_agg_2_2",
        ]
        if portfolio not in {"compact", "balanced", "wide"}:
            raise ValueError(
                "portfolio must be one of {'compact', 'balanced', 'wide'}."
            )
        if selection_strategy not in {"best", "earliest_within_slack"}:
            raise ValueError(
                "selection_strategy must be one of {'best', 'earliest_within_slack'}."
            )
        if selection_obj_slack < 0:
            raise ValueError("selection_obj_slack must be non-negative.")

        base_caps = list(cap_portions) if cap_portions is not None else [0.25]
        extra_caps = [] if cap_portions is not None else [0.2, 0.3]
        wide_caps = [] if cap_portions is not None else [0.15, 0.35]

        config_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()

        def add_config(
            *,
            cap: float,
            machine_then_job: bool,
            head_for_all_stages: bool,
            normalize_by_stage_cnt: bool,
            method_list: list[str],
            p_agg_method: str = "sum",
            mi_agg_method: str = "max",
            randomize_mid_all: bool = False,
            reverse_mid_all: bool = False,
            reverse_mid_even: bool = False,
            random_trial_idx: int | None = None,
            job_tiebreak_rank: Mapping[str, int] | None = None,
            random_tiebreak_trial_idx: int | None = None,
        ) -> None:
            key = (
                round(float(cap), 6),
                machine_then_job,
                head_for_all_stages,
                normalize_by_stage_cnt,
                randomize_mid_all,
                reverse_mid_all,
                reverse_mid_even,
                random_trial_idx,
                random_tiebreak_trial_idx,
                tuple(method_list),
                p_agg_method,
                mi_agg_method,
            )
            if key in seen_keys:
                return
            seen_keys.add(key)
            config_rows.append(
                {
                    "left_cap_portion": float(cap),
                    "right_cap_portion": float(cap),
                    "normalize_by_stage_cnt": normalize_by_stage_cnt,
                    "randomize_mid_all": randomize_mid_all,
                    "reverse_mid_all": reverse_mid_all,
                    "reverse_mid_even": reverse_mid_even,
                    "mixed_schedule_for_former_stages": True,
                    "mixed_schedule_for_later_stages": True,
                    "machine_then_job": machine_then_job,
                    "head_for_all_stages": head_for_all_stages,
                    "p_agg_method": p_agg_method,
                    "mi_agg_method": mi_agg_method,
                    "method_list": method_list,
                    "_random_trial_idx": random_trial_idx,
                    "_job_tiebreak_rank": dict(job_tiebreak_rank or {}),
                    "_random_tiebreak_trial_idx": random_tiebreak_trial_idx,
                }
            )

        def add_mid_order_variants(*, cap: float, method_list: list[str]) -> None:
            if not include_mid_order_variants:
                return
            for reverse_mid_even, reverse_mid_all in ((True, False), (False, True)):
                add_config(
                    cap=cap,
                    machine_then_job=True,
                    head_for_all_stages=True,
                    normalize_by_stage_cnt=False,
                    method_list=bn2d_methods,
                    reverse_mid_even=reverse_mid_even,
                    reverse_mid_all=reverse_mid_all,
                )
            for trial_idx in range(max(0, int(randomized_mid_trials))):
                add_config(
                    cap=cap,
                    machine_then_job=True,
                    head_for_all_stages=True,
                    normalize_by_stage_cnt=False,
                    method_list=bn2d_methods,
                    randomize_mid_all=True,
                    random_trial_idx=trial_idx,
                )

        def add_random_tiebreak_variants(
            *,
            cap: float,
            method_list: list[str],
            p_agg_method: str = "sum",
            mi_agg_method: str = "max",
        ) -> None:
            if randomized_tiebreak_trials <= 0 or not method_list:
                return
            jobs = [str(job_id) for job_id in self.instance.job_id_list]
            for trial_idx in range(max(0, int(randomized_tiebreak_trials))):
                shuffled_jobs = list(jobs)
                random.shuffle(shuffled_jobs)
                job_tiebreak_rank = {
                    job_id: idx for idx, job_id in enumerate(shuffled_jobs)
                }
                add_config(
                    cap=cap,
                    machine_then_job=True,
                    head_for_all_stages=True,
                    normalize_by_stage_cnt=False,
                    method_list=method_list,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    job_tiebreak_rank=job_tiebreak_rank,
                    random_tiebreak_trial_idx=trial_idx,
                )
                add_config(
                    cap=cap,
                    machine_then_job=False,
                    head_for_all_stages=True,
                    normalize_by_stage_cnt=False,
                    method_list=method_list,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    job_tiebreak_rank=job_tiebreak_rank,
                    random_tiebreak_trial_idx=trial_idx,
                )

        for cap in base_caps:
            add_config(
                cap=cap,
                machine_then_job=True,
                head_for_all_stages=False,
                normalize_by_stage_cnt=False,
                method_list=core_methods,
            )
            add_config(
                cap=cap,
                machine_then_job=True,
                head_for_all_stages=True,
                normalize_by_stage_cnt=False,
                method_list=core_methods,
            )
            add_config(
                cap=cap,
                machine_then_job=False,
                head_for_all_stages=True,
                normalize_by_stage_cnt=False,
                method_list=core_methods,
            )
            if portfolio in {"balanced", "wide"}:
                add_config(
                    cap=cap,
                    machine_then_job=True,
                    head_for_all_stages=True,
                    normalize_by_stage_cnt=True,
                    method_list=core_methods,
                )
            add_mid_order_variants(cap=cap, method_list=core_methods)
            add_random_tiebreak_variants(cap=cap, method_list=mixed_methods)

        if portfolio in {"balanced", "wide"}:
            for cap in extra_caps:
                add_config(
                    cap=cap,
                    machine_then_job=True,
                    head_for_all_stages=True,
                    normalize_by_stage_cnt=False,
                    method_list=core_methods,
                )
                add_mid_order_variants(cap=cap, method_list=core_methods)
                add_random_tiebreak_variants(cap=cap, method_list=mixed_methods)

        if portfolio == "wide":
            for cap in wide_caps:
                for machine_then_job in (True, False):
                    add_config(
                        cap=cap,
                        machine_then_job=machine_then_job,
                        head_for_all_stages=True,
                        normalize_by_stage_cnt=False,
                        method_list=core_methods,
                    )
                add_mid_order_variants(cap=cap, method_list=core_methods)
                add_random_tiebreak_variants(cap=cap, method_list=mixed_methods)

        if include_stage_agg:
            agg_pairs = [("sum", "max")]
            if portfolio in {"balanced", "wide"}:
                agg_pairs.append(("sum", "min"))
            if portfolio == "wide":
                agg_pairs.append(("sum", "max"))
            for p_agg_method, mi_agg_method in agg_pairs:
                for cap in base_caps:
                    add_config(
                        cap=cap,
                        machine_then_job=True,
                        head_for_all_stages=True,
                        normalize_by_stage_cnt=False,
                        method_list=stage_agg_methods,
                        p_agg_method=p_agg_method,
                        mi_agg_method=mi_agg_method,
                    )
                    if portfolio in {"balanced", "wide"}:
                        add_config(
                            cap=cap,
                            machine_then_job=False,
                            head_for_all_stages=True,
                            normalize_by_stage_cnt=False,
                            method_list=stage_agg_methods,
                            p_agg_method=p_agg_method,
                            mi_agg_method=mi_agg_method,
                        )
                    add_mid_order_variants(cap=cap, method_list=stage_agg_methods)
                    add_random_tiebreak_variants(
                        cap=cap,
                        method_list=stage_agg_only_methods,
                        p_agg_method=p_agg_method,
                        mi_agg_method=mi_agg_method,
                    )

        best_sch: HybridFlowshopLiteSchedule | None = None
        best_obj: int | None = None
        best_config_idx = -1
        best_method_name = ""
        candidate_records: list[
            tuple[
                int,
                str,
                int,
                HybridFlowshopLiteSchedule,
                dict[str, Any],
                int | None,
                int | None,
            ]
        ] = []
        for config_idx, config in enumerate(config_rows, start=1):
            try:
                dispatch_config = {
                    key: value
                    for key, value in config.items()
                    if not key.startswith("_")
                }
                job_tiebreak_rank = config.get("_job_tiebreak_rank") or None
                candidate_schedules = self._get_selected_dispatch_candidate_schedules(
                    **dispatch_config,
                    job_tiebreak_rank=job_tiebreak_rank,
                    draw_gantt=False,
                )
            except Exception:
                logging.exception(
                    "[Dispatch Portfolio] Candidate config %d/%d failed: %s",
                    config_idx,
                    len(config_rows),
                    config,
                )
                continue
            for method_name, sch in candidate_schedules.items():
                if sch is None:
                    continue
                obj = sch.makespan
                candidate_records.append(
                    (
                        config_idx,
                        method_name,
                        obj,
                        sch,
                        dict(dispatch_config),
                        config.get("_random_trial_idx"),
                        config.get("_random_tiebreak_trial_idx"),
                    )
                )
                if best_obj is None or obj < best_obj:
                    best_sch = sch
                    best_obj = obj
                    best_config_idx = config_idx
                    best_method_name = method_name
                    logging.info(
                        "[Dispatch Portfolio] New best init makespan=%s from "
                        "config %d/%d method=%s random_mid_trial=%s "
                        "random_tiebreak_trial=%s config=%s",
                        best_obj,
                        best_config_idx,
                        len(config_rows),
                        best_method_name,
                        config.get("_random_trial_idx"),
                        config.get("_random_tiebreak_trial_idx"),
                        dispatch_config,
                    )

        if best_sch is None:
            logging.warning(
                "[Dispatch Portfolio] No feasible dispatch candidate found."
            )
            return
        selected_config_idx = best_config_idx
        selected_method_name = best_method_name
        selected_obj = best_obj
        selected_sch = best_sch
        selected_dispatch_config: dict[str, Any] | None = None
        selected_random_trial_idx: int | None = None
        selected_random_tiebreak_trial_idx: int | None = None
        if selection_strategy == "earliest_within_slack":
            threshold = float(best_obj) + float(selection_obj_slack)
            for (
                config_idx,
                method_name,
                obj,
                sch,
                dispatch_config,
                random_trial_idx,
                random_tiebreak_trial_idx,
            ) in candidate_records:
                if float(obj) <= threshold:
                    selected_config_idx = config_idx
                    selected_method_name = method_name
                    selected_obj = obj
                    selected_sch = sch
                    selected_dispatch_config = dispatch_config
                    selected_random_trial_idx = random_trial_idx
                    selected_random_tiebreak_trial_idx = random_tiebreak_trial_idx
                    break
        if selected_dispatch_config is None:
            for (
                config_idx,
                method_name,
                obj,
                _sch,
                dispatch_config,
                random_trial_idx,
                random_tiebreak_trial_idx,
            ) in candidate_records:
                if (
                    config_idx == selected_config_idx
                    and method_name == selected_method_name
                    and obj == selected_obj
                ):
                    selected_dispatch_config = dispatch_config
                    selected_random_trial_idx = random_trial_idx
                    selected_random_tiebreak_trial_idx = random_tiebreak_trial_idx
                    break

        self.last_selected_dispatch_config = dict(selected_dispatch_config or {})
        logging.info(
            "[Dispatch Portfolio] Best raw init makespan=%s from config %d/%d method=%s",
            best_obj,
            best_config_idx,
            len(config_rows),
            best_method_name,
        )
        logging.info(
            "[Dispatch Portfolio] Selected init makespan=%s from config %d/%d "
            "method=%s random_mid_trial=%s random_tiebreak_trial=%s "
            "strategy=%s slack=%s",
            selected_obj,
            selected_config_idx,
            len(config_rows),
            selected_method_name,
            selected_random_trial_idx,
            selected_random_tiebreak_trial_idx,
            selection_strategy,
            selection_obj_slack,
        )

        if error_if_infeasible:
            self.check_feasibility(selected_sch.get_jik_2_start_time_map())

        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(selected_sch.makespan),
            obj_bound=None,
            is_init=self.solution_manager.get_incumbent() is None,
            subroutine_name="initialize_by_dispatch_portfolio",
            progress_obj_value_records=[
                (sub_timer.elapsed_sec, float(selected_sch.makespan))
            ],
        )
        was_updated = self.solution_manager.register(report, selected_sch)

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(
            log_time, float(selected_sch.makespan), is_maximize=False
        )
        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
        )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_sequence_insertion_order(self, order_rule: str) -> list[str]:
        jobs = list(self.instance.job_id_list)
        stages = list(self.instance.stage_id_list)
        p = self.job_2_stage_2_p_dict
        split_idx = max(1, len(stages) // 2)
        first_stages = stages[:split_idx]
        last_stages = stages[split_idx:]
        if not last_stages:
            last_stages = stages[-1:]

        def total_p(job_id: str) -> int:
            return sum(p[job_id][stage_id] for stage_id in stages)

        def first_p(job_id: str) -> int:
            return sum(p[job_id][stage_id] for stage_id in first_stages)

        def last_p(job_id: str) -> int:
            return sum(p[job_id][stage_id] for stage_id in last_stages)

        if order_rule == "total_desc":
            return sorted(jobs, key=lambda j: (-total_p(j), j))
        if order_rule == "total_asc":
            return sorted(jobs, key=lambda j: (total_p(j), j))
        if order_rule == "front_desc":
            return sorted(jobs, key=lambda j: (-first_p(j), -total_p(j), j))
        if order_rule == "tail_desc":
            return sorted(jobs, key=lambda j: (-last_p(j), -total_p(j), j))
        if order_rule == "flow_slope_desc":
            return sorted(jobs, key=lambda j: (first_p(j) - last_p(j), -total_p(j), j))
        if order_rule == "bottleneck_desc":
            bottleneck_stage_id = max(
                stages,
                key=lambda stage_id: (
                    sum(self.stage_2_job_2_p_dict[stage_id].values())
                    / max(1, len(self.instance.stage_2_machines_map[stage_id]))
                ),
            )
            return sorted(
                jobs,
                key=lambda j: (-p[j][bottleneck_stage_id], -total_p(j), j),
            )
        if order_rule == "random":
            shuffled_jobs = list(jobs)
            random.shuffle(shuffled_jobs)
            return shuffled_jobs
        raise ValueError(
            "order_rule must be one of {'total_desc', 'total_asc', 'front_desc', "
            "'tail_desc', 'flow_slope_desc', 'bottleneck_desc', 'random'}. "
            f"Received {order_rule!r}."
        )

    @staticmethod
    def _sample_insertion_positions(
        sequence_len: int,
        max_insert_positions: int | None,
    ) -> list[int]:
        positions = list(range(sequence_len + 1))
        if max_insert_positions is None or max_insert_positions >= len(positions):
            return positions
        if max_insert_positions <= 1:
            return [sequence_len]
        last_idx = len(positions) - 1
        sampled = {
            positions[round(idx * last_idx / (max_insert_positions - 1))]
            for idx in range(max_insert_positions)
        }
        return sorted(sampled)

    def _build_sequence_by_insertion(
        self,
        base_order: Sequence[str],
        *,
        beam_width: int,
        max_insert_positions: int | None,
        order_label: str,
    ) -> tuple[list[str], int]:
        beam: list[tuple[list[str], int]] = [([], 0)]
        resolved_beam_width = max(1, int(beam_width))
        for job_idx, job_id in enumerate(base_order, start=1):
            candidates: list[tuple[list[str], int]] = []
            seen_sequences: set[tuple[str, ...]] = set()
            for sequence, _score in beam:
                for position in self._sample_insertion_positions(
                    len(sequence),
                    max_insert_positions,
                ):
                    candidate_sequence = (
                        sequence[:position] + [job_id] + sequence[position:]
                    )
                    sequence_key = tuple(candidate_sequence)
                    if sequence_key in seen_sequences:
                        continue
                    seen_sequences.add(sequence_key)
                    schedule = self._from_job_sequence_get_schedule(candidate_sequence)
                    candidates.append((candidate_sequence, schedule.makespan))
            candidates.sort(key=lambda item: (item[1], item[0]))
            beam = candidates[:resolved_beam_width]
            if job_idx == len(base_order) or job_idx % 20 == 0:
                logging.info(
                    "[Sequence Init] %s inserted %d/%d jobs; best partial makespan=%s",
                    order_label,
                    job_idx,
                    len(base_order),
                    beam[0][1],
                )
        return beam[0]

    def initialize_by_sequence_insertion_portfolio(
        self,
        order_rules: Sequence[str] | None = None,
        randomized_order_trials: int = 0,
        beam_width: int = 1,
        max_insert_positions: int | None = None,
        machine_then_job_options: Sequence[bool] | None = None,
        head_for_all_stages_options: Sequence[bool] | None = None,
        include_reversed_final_sequences: bool = True,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Initialize from NEH-style insertion sequences evaluated by mixed dispatch."""
        sub_timer = ElapsedTimer()
        resolved_order_rules = list(
            order_rules
            if order_rules is not None
            else [
                "total_desc",
                "tail_desc",
                "front_desc",
                "flow_slope_desc",
                "bottleneck_desc",
            ]
        )
        resolved_machine_then_job_options = list(
            machine_then_job_options
            if machine_then_job_options is not None
            else [True, False]
        )
        resolved_head_for_all_stages_options = list(
            head_for_all_stages_options
            if head_for_all_stages_options is not None
            else [True, False]
        )

        best_sch: HybridFlowshopLiteSchedule | None = None
        best_obj: int | None = None
        best_label = ""
        built_sequences: list[tuple[str, list[str], int]] = []

        for order_rule in resolved_order_rules:
            base_order = self._get_sequence_insertion_order(order_rule)
            sequence, partial_obj = self._build_sequence_by_insertion(
                base_order,
                beam_width=beam_width,
                max_insert_positions=max_insert_positions,
                order_label=order_rule,
            )
            built_sequences.append((order_rule, sequence, partial_obj))

        for trial_idx in range(max(0, int(randomized_order_trials))):
            base_order = self._get_sequence_insertion_order("random")
            order_label = f"random_{trial_idx}"
            sequence, partial_obj = self._build_sequence_by_insertion(
                base_order,
                beam_width=beam_width,
                max_insert_positions=max_insert_positions,
                order_label=order_label,
            )
            built_sequences.append((order_label, sequence, partial_obj))

        for order_label, sequence, partial_obj in built_sequences:
            sequence_variants: list[tuple[str, list[str]]] = [(order_label, sequence)]
            if include_reversed_final_sequences:
                sequence_variants.append(
                    (f"{order_label}_reversed", list(reversed(sequence)))
                )
            for sequence_label, candidate_sequence in sequence_variants:
                for machine_then_job in resolved_machine_then_job_options:
                    for head_for_all_stages in resolved_head_for_all_stages_options:
                        sch = self._get_best_mixed_schedule_from_job_sequence(
                            candidate_sequence,
                            machine_then_job=machine_then_job,
                            head_for_all_stages=head_for_all_stages,
                        )
                        if sch is None:
                            continue
                        obj = sch.makespan
                        if best_obj is None or obj < best_obj:
                            best_sch = sch
                            best_obj = obj
                            best_label = (
                                f"{sequence_label}; partial={partial_obj}; "
                                f"machine_then_job={machine_then_job}; "
                                f"head_for_all_stages={head_for_all_stages}"
                            )
                            logging.info(
                                "[Sequence Init] New best init makespan=%s from %s",
                                best_obj,
                                best_label,
                            )

        if best_sch is None:
            logging.warning("[Sequence Init] No feasible insertion candidate found.")
            return
        logging.info(
            "[Sequence Init] Best init makespan=%s from %s",
            best_obj,
            best_label,
        )

        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())

        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(best_sch.makespan),
            obj_bound=None,
            is_init=self.solution_manager.get_incumbent() is None,
            subroutine_name="initialize_by_sequence_insertion_portfolio",
            progress_obj_value_records=[
                (sub_timer.elapsed_sec, float(best_sch.makespan))
            ],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, float(best_sch.makespan), is_maximize=False)
        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
        )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_job_sequence_from_stage_aggregated_problem(
        self,
        stage_agg_count: int,
        head_stages_to_keep: int = 0,
        tail_stages_to_keep: int = 0,
        p_agg_method: str = "sum",
        mi_agg_method: str = "min",
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt_per_step: bool = False,
        draw_gantt: bool = False,
    ) -> None:

        sub_timer = ElapsedTimer()

        best_sch = self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
            stage_agg_count,
            head_stages_to_keep=head_stages_to_keep,
            tail_stages_to_keep=tail_stages_to_keep,
            p_agg_method=p_agg_method,
            mi_agg_method=mi_agg_method,
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            draw_gantt_per_step=draw_gantt_per_step,
        )

        if best_sch is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())

        best_obj = best_sch.makespan
        logging.info(
            f"Schedule from stage-aggregated problem: makespan={best_obj}"
            f" with stage_agg_count={stage_agg_count}"
        )

        # Create report and register the new solution
        obj_value = float(best_sch.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_job_sequence_from_stage_aggregated_problem",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def _get_schedule_by_job_sequence_from_stage_aggregated_problem(
        self,
        stage_agg_count: int,
        head_stages_to_keep: int = 0,
        tail_stages_to_keep: int = 0,
        p_agg_method: str = "sum",
        mi_agg_method: str = "min",
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        draw_gantt_per_step: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        from hybridflowshop.schedule_lite import (
            get_bottleneck_stage_job_sequence,
            get_midpoint_sequence,
        )

        stage_aggregated_schedule = self._get_schedule_from_stage_aggregated_problem(
            stage_agg_count,
            head_stages_to_keep=head_stages_to_keep,
            tail_stages_to_keep=tail_stages_to_keep,
            p_agg_method=p_agg_method,
            mi_agg_method=mi_agg_method,
            head_for_all_stages=head_for_all_stages,
            draw_gantt_per_step=draw_gantt_per_step,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        if stage_aggregated_schedule is None:
            return None

        job_sequences = [
            get_bottleneck_stage_job_sequence(
                stage_aggregated_schedule,
                job_tiebreak_rank=job_tiebreak_rank,
            ),
            get_midpoint_sequence(
                stage_aggregated_schedule,
                job_tiebreak_rank=job_tiebreak_rank,
            ),
        ]

        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None

        dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            self.instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        for job_sequence in job_sequences:
            dispatched_schedule = dispatcher.get_best_mixed_schedule_by_sequence(
                job_sequence,
                machine_then_job=machine_then_job,
                head_for_all_stages=head_for_all_stages,
            )
            if dispatched_schedule is not None:
                if best_obj is None or dispatched_schedule.makespan < best_obj:
                    best_obj = dispatched_schedule.makespan
                    best_sch = dispatched_schedule

        best_reversed_obj: int | None = None
        best_reversed_sch: HybridFlowshopLiteSchedule | None = None

        reversed_dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            reverse_stages(self.instance),
            job_tiebreak_rank=job_tiebreak_rank,
        )
        for job_sequence in job_sequences:
            dispatched_schedule = (
                reversed_dispatcher.get_best_mixed_schedule_by_sequence(
                    job_sequence,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                )
            )
            if dispatched_schedule is not None:
                if (
                    best_reversed_obj is None
                    or dispatched_schedule.makespan < best_reversed_obj
                ):
                    best_reversed_obj = dispatched_schedule.makespan
                    best_reversed_sch = dispatched_schedule

        if best_reversed_sch is not None:
            if best_sch is None or best_reversed_obj < best_obj:
                converted = best_reversed_sch.as_reversed()
                converted.make_semi_active(self.stage_2_job_2_p_dict)
                return converted
        return best_sch

    def _get_schedule_from_stage_aggregated_problem(
        self,
        stage_agg_count: int,
        head_stages_to_keep: int = 0,
        tail_stages_to_keep: int = 0,
        p_agg_method: str = "sum",
        mi_agg_method: str = "min",
        head_for_all_stages: bool = False,
        draw_gantt_per_step: bool = False,
        draw_gantt: bool = False,
        job_tiebreak_rank: Mapping[str, int] | None = None,
    ) -> HybridFlowshopLiteSchedule | None:
        from schore.parameters_examples.parallel_shop.identical_flow import (
            create_instance_of_aggregated_stages,
        )

        stage_aggregated_instance = create_instance_of_aggregated_stages(
            self.instance,
            stage_agg_count,
            head_stages_to_keep=head_stages_to_keep,
            tail_stages_to_keep=tail_stages_to_keep,
            p_agg_method=p_agg_method,
            mi_agg_method=mi_agg_method,
        )
        dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            stage_aggregated_instance,
            job_tiebreak_rank=job_tiebreak_rank,
        )
        schedule = dispatcher.get_schedule_by_cds(
            head_for_all_stages=head_for_all_stages,
            draw_gantt_per_step=draw_gantt_per_step,
        )

        reversed_dispatcher = self._create_dispatcher_with_optional_job_tiebreak(
            MixedDispatcher,
            reverse_stages(stage_aggregated_instance),
            job_tiebreak_rank=job_tiebreak_rank,
        )
        reversed_schedule = reversed_dispatcher.get_schedule_by_cds(
            head_for_all_stages=head_for_all_stages,
            draw_gantt_per_step=draw_gantt_per_step,
        )

        if reversed_schedule is not None and (
            schedule is None or reversed_schedule.makespan < schedule.makespan
        ):
            converted = reversed_schedule.as_reversed()
            converted.make_semi_active(stage_aggregated_instance.stage_2_job_2_p_map)
            schedule = converted

        if draw_gantt and schedule is not None:
            output_path = self.get_file_path_for_subroutine("_gantt_stage_agg.png")
            self.draw_gantt(schedule, output_path=output_path)
        return schedule

    def init_by_best_of_job_seq_from_stage_agg_problem(
        self,
        stage_agg_count: int,
        p_agg_method: str = "sum",
        mi_agg_method: str = "min",
        machine_then_job: bool = False,
        head_for_all_stages: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt_per_step: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        sub_timer = ElapsedTimer()

        best_sch: HybridFlowshopLiteSchedule | None = None

        for head_stages_to_keep in range(stage_agg_count):
            dispatched_sch = (
                self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
                    stage_agg_count,
                    head_stages_to_keep=head_stages_to_keep,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    draw_gantt_per_step=draw_gantt_per_step,
                )
            )

            if dispatched_sch is not None:
                if best_sch is None or dispatched_sch.makespan < best_sch.makespan:
                    best_sch = dispatched_sch

        if best_sch is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(best_sch.get_jik_2_start_time_map())

        best_obj = best_sch.makespan
        logging.info(
            f"Schedule from stage-aggregated problem: makespan={best_obj}"
            f" with stage_agg_count={stage_agg_count}"
        )

        # Create report and register the new solution
        obj_value = float(best_sch.makespan)
        report = self._make_subroutine_report(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
            subroutine_name="initialize_by_job_sequence_from_stage_aggregated_problem",
            progress_obj_value_records=[(sub_timer.elapsed_sec, obj_value)],
        )
        was_updated = self.solution_manager.register(report, best_sch)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    # End subroutine definition
