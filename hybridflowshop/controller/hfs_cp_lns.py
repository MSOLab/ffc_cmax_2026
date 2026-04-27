import logging
import math
import random
import time
from collections import Counter, deque
from dataclasses import replace
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
from routix import DynamicDataObject, ElapsedTimer
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
from hybridflowshop.report import HfsCpsatSolverReport, HfsSubroutineReport
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    JobIdType,
    OperationType,
    StageIdType,
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
                            self.Value(
                                self._build.variables.op_start[job_id, stage_id]
                            )
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


class HybridFlowShopCpLnsController(HybridFlowShopCpLnsControllerCore):
    """
    Controller for solving Hybrid Flow Shop problems using CP-based algorithms.
    """

    # Override

    def set_cp_model_as_base_cp_model(
        self, tighten_ranges: bool = False, link_job_completion: bool = False
    ) -> None:
        self.cp_model = self.create_base_cp_model(
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        )
        self.cp_model.set_num_base_constraints()
        self.base_cp_model_is_set = True

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
        make_semi_active_after_cp: bool = False,
        is_initial_solution: bool = False,
        encode_cumulative_as_reservoir: bool | None = None,
        expand_reservoir_constraints: bool | None = None,
        expand_reservoir_using_circuit: bool | None = None,
        interleave_search: bool | None = None,
        use_lns_only: bool | None = None,
        cp_model_probing_level: int | None = None,
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
            cp_model_probing_level (int | None, optional): The level of probing for the CP model. Defaults to None.
            log_search_progress (bool, optional): If True, logs the search progress during solving. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution after solving. Defaults to False.
        """
        sub_timer = ElapsedTimer()
        if self.base_cp_model_is_set:
            self.cp_model.delete_added_constraints()
        else:
            self.set_cp_model_as_base_cp_model()

        _should_be_init: bool = self.solution_manager.get_incumbent() is None
        _is_init: bool = _should_be_init or is_initial_solution
        _computational_time = self._resolve_tl_nc_computational_time(
            computational_time=computational_time,
            tl_nc_multiplier=tl_nc_multiplier,
        )
        if use_final_time_reserve:
            _computational_time = self.consume_reserved_final_time_sec(
                fallback_sec=_computational_time,
            )
        if _computational_time is not None:
            # Subtract model handling time from subroutine time limit
            _computational_time = max(
                0.0, _computational_time - sub_timer.elapsed_sec
            )

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

    def clear_final_time_reserve(self) -> None:
        reserve_sec = self.get_reserved_final_time_sec()
        self.clear_reserved_final_time_sec()
        logging.info("[Final Reserve] Cleared %.3f sec final time reserve.", reserve_sec)

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
        cp_model_probing_level: int | None = None,
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
            make_semi_active_after_cp=make_semi_active_after_cp,
            is_initial_solution=is_initial_solution,
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

            retained_result, retained_solution_rows = (
                read_retained_stage_cp_artifacts(source_dir)
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
            for (job_id, stage_id, _), start_time in (
                incumbent_solution.get_jik_2_start_time_map().items()
            ):
                start_hint_by_ji[str(job_id), str(stage_id)] = int(start_time)
            for (job_id, stage_id, _), end_time in (
                incumbent_solution.get_jik_2_end_time_map().items()
            ):
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

    def _fix_operations_profile(
        self,
        schedule_profile_fixed_only: HybridFlowshopLiteSchedule,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> None:
        BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
            self.cp_model,
            self.params,
            self.vars,
            schedule_profile_fixed_only,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
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
            int(round(idx * last_idx / (max_count - 1)))
            for idx in range(max_count)
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
        windows = [
            (start, start + resolved_window_size)
            for start in starts
        ]
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
                    end_time <= shifted_window_start
                    or shifted_window_end <= start_time
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
            "Single time-window NS starts for [%d, %d) "
            "(stage_shift_per_stage=%d).",
            resolved_window_start,
            resolved_window_end,
            resolved_stage_shift_per_stage,
        )

        context_name = (
            f"single_window_{resolved_window_start}_{resolved_window_end}"
        )
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
                    lambda selected_ops=selected_ops, incumbent_solution=incumbent_solution: self._fix_time_window_profile_except_selected(
                        incumbent_solution,
                        selected_ops,
                        profile_fix_by_machine=profile_fix_by_machine,
                        machine_precedence_stride=machine_precedence_stride,
                        fix_outside_start_times=fix_outside_start_times,
                        selected_start_time_tolerance=resolved_selected_tolerance,
                        outside_start_time_tolerance=resolved_outside_tolerance,
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
            {str(stage_id) for stage_id in selected_stages if str(stage_id) in stage_order},
            key=stage_order.index,
        )
        if not selected_stage_ids:
            raise ValueError("selected_stages must contain at least one valid stage ID.")

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
            - `critical_adjacency`
              First pick one weighted-random seed job. Then perform frontier
              expansion over the critical-job adjacency graph built from consecutive
              jobs inside each critical block. The queue is BFS-like: pop one
              selected job, shuffle its unseen neighbors, append them in that order,
              and continue until the target is reached or the frontier is exhausted.

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
        if job_selection_policy != "critical_adjacency":
            raise ValueError(
                "Unsupported job_selection_policy for critical-job selection: "
                f"{job_selection_policy}"
            )

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
                    self._weighted_sample_without_replacement(
                        remaining_jobs,
                        job_weights,
                        target_count - len(selected_jobs),
                    )
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
            resolved_ratio = 0.3 if tail_stage_ratio is None else float(tail_stage_ratio)
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
            [op for op in block if op[1] in tail_stage_set]
            for block in critical_blocks
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
        instance = self.instance
        jobs: list[str] = instance.job_id_list
        stages: list[str] = instance.stage_id_list
        p: dict[tuple[str, str], int] = instance.p_manager.job_stage_2_value_map(
            jobs, stages
        )

        m = len(stages)
        stage_idx = stages.index(i)
        mc_count = len(instance.stage_2_machines_map[i])

        def RS(j: str) -> int:
            return sum(p[j, stages[s]] for s in range(stage_idx + 1, m))

        def LS(j: str) -> int:
            return sum(p[j, stages[s]] for s in range(0, stage_idx))

        LSA = sorted([LS(j) for j in jobs])
        RSA = sorted([RS(j) for j in jobs])
        total_processing = sum(p[j, i] for j in jobs)
        lhs_rum = sum(LSA[y] for y in range(min(mc_count, len(LSA))))
        rhs_rum = sum(RSA[y] for y in range(min(mc_count, len(RSA))))
        return math.ceil((lhs_rum + total_processing + rhs_rum) / mc_count)

    def apply_shdlb(self) -> None:
        """
        Compute the global lower bound for the Hybrid Flow Shop instance using the method
        described by Santos et al. (1995) and update the global lower bound.
        """
        sub_timer = ElapsedTimer()
        instance = self.instance
        jobs: list[str] = instance.job_id_list
        stages: list[str] = instance.stage_id_list
        p: dict[tuple[str, str], int] = instance.p_manager.job_stage_2_value_map(
            jobs, stages
        )

        def LB0() -> int:
            return max(sum(p[j, i] for i in stages) for j in jobs)

        lb0 = LB0()
        stage_bounds = [self.get_shdlb_for_stage(stage) for stage in stages]
        obj_bound = max([lb0] + stage_bounds)

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
    ) -> dict[str, Any] | None:
        """
        Compute a retained-stage CP-SAT lower bound using only a subset of stages exactly.

        Supported retained stage modes:
        - ``first_last``: retain the first and last stages exactly.
        - ``first_bottleneck_last``: retain the first, bottleneck, and last stages.
        - ``first_bottleneck_band_last``: retain the first, last, and the stages
          within ``bottleneck_band_radius`` of a representative bottleneck stage.
        - ``first_topk_bottlenecks_last``: retain the first, top-k bottlenecks, and last stages.
        - ``first_middle_last``: retain the first, middle, and last stages.
        - ``first_middle_band_last``: retain the first, last, and the stages
          within ``middle_band_radius`` of the middle stage.
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
        if snapshot_solution_limit > 0:
            snapshot_recorder = _RetainedStageSnapshotRecorder(
                build=build,
                snapshot_limit=snapshot_solution_limit,
                e_timer=sub_timer,
                log_level_on_record=logging.INFO if snapshot_log_progress else None,
            )

        solver_report = self.solve_cp_model_2(
            build.model,
            _time_limit_sec,
            threads,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            e_timer=sub_timer,
            log_search_progress=False,
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

        current_bound = self.solution_manager.best_obj_bound
        improved_bound_logged = False
        for row in trace_rows:
            trace_runtime = sanitize_optional_float(row.get("runtime_sec"))
            trace_lb = sanitize_optional_float(row.get("objective_lb"))
            if trace_runtime is None or trace_lb is None:
                continue
            if self.solution_manager._a_is_better_obj_bound(trace_lb, current_bound):
                global_time = start_t + trace_runtime
                self.add_obj_bound_log(global_time, trace_lb, is_maximize=False)
                current_bound = trace_lb
                improved_bound_logged = True

        report = HfsCpsatSolverReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=None,
            obj_bound=result.certified_final_lb,
            is_init=False,
            subroutine_name="apply_retained_stage_cp_lb",
            call_context=call_context,
            progress_obj_value_records=(),
            progress_time_basis="local",
            status=solver_report.status,
            obj_value_records=(),
            obj_bound_records=[
                (
                    float(row["runtime_sec"]),
                    float(row["objective_lb"]),
                )
                for row in trace_rows
                if sanitize_optional_float(row.get("runtime_sec")) is not None
                and sanitize_optional_float(row.get("objective_lb")) is not None
            ],
        )
        self.solution_manager.register(report, None)
        self._record_retained_cp_lb_summary(
            result,
            call_context=call_context,
            start_sec=start_t,
            apply_elapsed_sec=self.last_retained_cp_lb_apply_elapsed_sec,
        )

        if improved_bound_logged:
            self.obj_store.add_last_timestamp_note(
                call_context,
                obj_bound_is_valid=True,
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
        return max(contiguous_blocks, key=lambda block: (len(block), -stage_2_index[block[0]]))

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
                    objective_lb=sanitize_optional_float(
                        snapshot.get("objective_lb")
                    ),
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
        for source_label, _, evaluated_dispatch_result in dispatch_evaluations:
            prefix = "" if source_label == "final" else f"{source_label}:"
            for variant, candidate_schedule in (
                evaluated_dispatch_result.dispatched_schedules.items()
            ):
                dispatch_candidates[f"{prefix}{variant}"] = (
                    candidate_schedule.makespan
                    if candidate_schedule is not None
                    else None
                )
        if incumbent_before is not None:
            dispatch_candidates["incumbent_before_retained_cp"] = incumbent_before
        return {
            "schedule": selected_schedule,
            "selected_dispatch_variant": selected_variant,
            "post_cp_selected_dispatch_variant": (
                post_selected_variant_with_source
            ),
            "post_cp_selected_obj": post_dispatch_obj,
            "kept_incumbent": kept_incumbent,
            "anchor_stage_ids": list(self.last_retained_cp_dispatch_anchor_stage_ids),
            "dispatch_candidates": dispatch_candidates,
            "selected_dispatch_source": dispatch_source_label,
        }

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

            retained_result, retained_solution_rows = (
                read_retained_stage_cp_artifacts(source_dir)
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
            logging.warning(
                "[CP Complete] No retained stages selected for completion."
            )
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

    def neh_cp(
        self,
        solver_thread_cnt: int,
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
            preserved_head_job_portion=preserved_head_job_portion,
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
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
            raise ValueError("portfolio must be one of {'compact', 'balanced', 'wide'}.")
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
                    key: value for key, value in config.items() if not key.startswith("_")
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
            logging.warning("[Dispatch Portfolio] No feasible dispatch candidate found.")
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
        self.add_obj_value_log(log_time, float(selected_sch.makespan), is_maximize=False)
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
                sequence_variants.append((f"{order_label}_reversed", list(reversed(sequence))))
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
