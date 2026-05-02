import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mbls.cpsat import CpsatSolverReport, CustomCpModel, ObjValueBoundStore
from ortools.sat.python.cp_model import CpModel
from routix import ElapsedTimer
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
    create_instance_of_job_subset,
)

from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder, CumulativeVars
from hybridflowshop.cpsat_model_2.params import Params
from hybridflowshop.dispatcher.mixed import MixedDispatcher
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
    get_bottleneck_stage_job_sequence,
    get_first_stage_start_sequence,
    get_midpoint_sequence,
)


class NehCpContext(Protocol):
    """
    Minimal dependency interface.
    (FlowshopTardinessControllerCore is effectively designed to satisfy this interface.)
    """

    def get_remaining_time_limit(
        self, subroutine_time_limit: float | None
    ) -> float: ...

    def get_remaining_sec_before_final_reserve(self) -> float: ...

    def final_time_reserve_is_reached(self) -> bool: ...

    # CP solve & decoding
    def solve_cp_model_2(
        self,
        mdl: CpModel,
        computational_time: float,
        solver_thread_cnt: int,
        e_timer: ElapsedTimer | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        use_lns_only: bool | None = None,
        log_level_obj_value: int = logging.INFO,
        log_level_obj_bound: int = logging.INFO,
    ) -> CpsatSolverReport: ...

    def create_empty_schedule_from_ins(self) -> HybridFlowshopLiteSchedule: ...

    def create_schedule(
        self, params: Params, variables: CumulativeVars
    ) -> HybridFlowshopLiteSchedule: ...

    def check_feasibility(
        self, start_time_map: dict[tuple[str, str, str], int]
    ) -> float: ...

    # optional
    def set_obj_lower_bound(
        self, mdl, variables: CumulativeVars, bound: float | None
    ) -> None: ...
    def export_solution_to_yaml(
        self,
        start_time_map: dict[tuple[str, str], int],
        end_time_map: dict[tuple[str, str], int],
        output_path: Path | None = None,
        encoding="utf-8",
    ) -> None: ...
    def get_file_path_for_subroutine(self, suffix: str): ...


@dataclass
class NehCpRunState:
    timer: ElapsedTimer

    last_job_id_list: list[str]
    partial_sol: HybridFlowshopLiteSchedule | None
    current_job_id_list: list[str]
    full_sol: HybridFlowshopLiteSchedule

    @property
    def job_subset_cnt(self) -> int:
        return len(self.current_job_id_list)

    def all_jobs_are_included(self, job_cnt: int) -> bool:
        return self.job_subset_cnt == job_cnt


@dataclass
class NehCpResult:
    schedule: HybridFlowshopLiteSchedule
    sub_obj_store: ObjValueBoundStore[int]
    last_obj_value: int


class NehCpConstructor:
    # Given schedule cache
    ref_schedule: HybridFlowshopLiteSchedule

    # Algorithm parameter cache
    added_batch_size: int
    max_time_per_add: float

    def __init__(self, ctx: NehCpContext):
        self.ctx = ctx

    def _require_state(self) -> NehCpRunState:
        if self._st is None:
            raise RuntimeError("PwCpConstructor.run() is not active; state is missing.")
        return self._st

    def run(
        self,
        ref_schedule: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        job_2_stage_2_p_dict: dict[str, dict[str, int]],
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        added_batch_size: int | None = None,
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
        solver_thread_cnt: int | None = None,
        use_lns_only: bool = False,
        error_if_infeasible: bool = False,
        stop_before_final_reserve: bool = True,
        min_remaining_sec_after_neh: float | None = None,
        min_remaining_nc_after_neh: float | None = None,
        time_guard_estimate_safety_factor: float = 1.15,
        time_guard_min_completed_batches: int = 1,
        skip_if_estimated_neh_exceeds_remaining: bool = True,
        full_neh_estimate_safety_factor: float = 1.0,
    ) -> NehCpResult:
        timer = ElapsedTimer()

        _added_batch_size: int = (
            added_batch_size
            if added_batch_size is not None and added_batch_size > 0
            else 1
        )
        if max_time_per_add is None:
            if cp_tl_nc_multiplier is not None:
                max_time_per_add = (
                    cp_tl_nc_multiplier * instance.job_count * instance.stage_count
                )
                logging.info(
                    f"max_time_per_add is set to {max_time_per_add:.2f} seconds"
                    f" based on cp_tl_nc_multiplier={cp_tl_nc_multiplier}, job & stage count."
                )
            elif cp_tl_c_multiplier is not None:
                max_time_per_add = cp_tl_c_multiplier * instance.stage_count
                logging.info(
                    f"max_time_per_add is set to {max_time_per_add:.2f} seconds"
                    f" based on cp_tl_c_multiplier={cp_tl_c_multiplier} and stage count."
                )
        max_time_per_add_2nd_obj: float | None = None
        if minimize_sum_ci_lex:
            if cp_tl_nc_multiplier_2nd_obj is not None:
                max_time_per_add_2nd_obj = (
                    cp_tl_nc_multiplier_2nd_obj
                    * instance.job_count
                    * instance.stage_count
                )
                logging.info(
                    f"max_time_per_add_2nd_obj is set to {max_time_per_add_2nd_obj:.2f} seconds"
                    f" based on cp_tl_nc_multiplier_2nd_obj={cp_tl_nc_multiplier_2nd_obj}, job & stage count."
                )
            elif cp_tl_c_multiplier_2nd_obj is not None:
                max_time_per_add_2nd_obj = (
                    cp_tl_c_multiplier_2nd_obj * instance.stage_count
                )
                logging.info(
                    f"max_time_per_add for 2nd obj is set to {max_time_per_add_2nd_obj:.2f}"
                    f"  seconds based on cp_tl_c_multiplier_2nd_obj={cp_tl_c_multiplier_2nd_obj}"
                    " and stage count."
                )
            else:
                max_time_per_add_2nd_obj = max_time_per_add
                logging.info(
                    "max_time_per_add for 2nd obj is set to the same as that of the "
                    f"1st obj: {max_time_per_add_2nd_obj:.2f} seconds."
                )

        # Handle out-of-range values with warnings
        if preserved_head_job_portion < 0.0:
            logging.warning(
                f"preserved_head_job_portion ({preserved_head_job_portion}) is negative; "
                "treating as 0.0 (full reconstruction mode)."
            )
            preserved_head_job_portion = 0.0
        elif preserved_head_job_portion >= 1.0:
            if preserved_head_job_portion > 1.0:
                logging.warning(
                    f"preserved_head_job_portion ({preserved_head_job_portion}) exceeds 1.0; "
                    "returning reference schedule unchanged."
                )
            # Early return for >= 1.0
            logging.info(
                "preserved_head_job_portion >= 1.0, returning reference schedule unchanged."
            )
            sub_obj_store = ObjValueBoundStore[int]()
            sub_obj_store.obj_value_series.name = "ObjVal after dispatch"
            sub_obj_store.obj_bound_series.name = "ObjVal before dispatch"
            return NehCpResult(
                schedule=ref_schedule,
                sub_obj_store=sub_obj_store,
                last_obj_value=ref_schedule.makespan,
            )

        if min_remaining_sec_after_neh is not None and min_remaining_sec_after_neh < 0:
            raise ValueError("min_remaining_sec_after_neh must be non-negative.")
        if min_remaining_nc_after_neh is not None and min_remaining_nc_after_neh < 0:
            raise ValueError("min_remaining_nc_after_neh must be non-negative.")
        if time_guard_estimate_safety_factor <= 0:
            raise ValueError("time_guard_estimate_safety_factor must be positive.")
        if full_neh_estimate_safety_factor <= 0:
            raise ValueError("full_neh_estimate_safety_factor must be positive.")
        if time_guard_min_completed_batches < 0:
            raise ValueError("time_guard_min_completed_batches must be non-negative.")
        successor_reserve_sec = float(min_remaining_sec_after_neh or 0.0)
        if min_remaining_nc_after_neh is not None:
            successor_reserve_sec += (
                float(min_remaining_nc_after_neh)
                * float(instance.job_count)
                * float(instance.stage_count)
            )

        sub_obj_store = ObjValueBoundStore[int]()
        """Subroutine-specific objective store"""
        sub_obj_store.obj_value_series.name = "ObjVal after dispatch"
        sub_obj_store.obj_bound_series.name = "ObjVal before dispatch"
        self._st: NehCpRunState = NehCpRunState(
            timer=timer,
            last_job_id_list=[],
            partial_sol=None,
            current_job_id_list=[],
            full_sol=ref_schedule,
        )
        best_full_sol = ref_schedule
        best_full_obj = ref_schedule.makespan

        # Determine job sequence
        # Priority: job_seq_by_1st_stage > job_seq_by_bottleneck_stage > midpoint (default)
        job_sequence: list[str]
        if job_seq_by_1st_stage:
            job_sequence = get_first_stage_start_sequence(ref_schedule)
        elif job_seq_by_bottleneck_stage:
            job_sequence = get_bottleneck_stage_job_sequence(ref_schedule)
        else:
            job_sequence = get_midpoint_sequence(ref_schedule)
        job_cnt = len(job_sequence)

        # Split into head (preserved) and tail (to reconstruct) jobs
        if preserved_head_job_portion > 0.0:
            preserved_cnt = int(job_cnt * preserved_head_job_portion)
            head_jobs = set(job_sequence[:preserved_cnt])
            tail_jobs = job_sequence[preserved_cnt:]
            logging.info(
                f"Partial reconstruction: preserving {preserved_cnt} head jobs ({preserved_head_job_portion * 100:.1f}%), "
                f"reconstructing {len(tail_jobs)} tail jobs."
            )
        else:
            head_jobs = set()
            tail_jobs = job_sequence

        sequence_of_job_sublist = self._split_tail_jobs_into_batches(
            tail_jobs,
            added_batch_size=_added_batch_size,
            added_batch_count=added_batch_count,
            min_added_batch_count=min_added_batch_count,
            max_added_batch_count=max_added_batch_count,
        )

        if self._should_skip_full_neh_before_first_batch(
            remaining_batch_count=len(sequence_of_job_sublist),
            max_time_per_add=max_time_per_add,
            minimize_sum_ci_lex=minimize_sum_ci_lex,
            max_time_per_add_2nd_obj=max_time_per_add_2nd_obj,
            successor_reserve_sec=successor_reserve_sec,
            skip_if_estimated_neh_exceeds_remaining=skip_if_estimated_neh_exceeds_remaining,
            full_neh_estimate_safety_factor=full_neh_estimate_safety_factor,
        ):
            return NehCpResult(
                schedule=best_full_sol,
                sub_obj_store=sub_obj_store,
                last_obj_value=best_full_obj,
            )

        st = self._require_state()
        batch_elapsed_sec_list: list[float] = []

        # Initialize partial solution from head jobs if in partial reconstruction mode
        if head_jobs:
            st.current_job_id_list = [j for j in job_sequence if j in head_jobs]
            st.partial_sol = ref_schedule.deepcopy(job_subsequence=head_jobs)
            st.partial_sol.make_semi_active(stage_2_job_2_p_dict)
            logging.info(
                f"Initialized partial solution from {len(head_jobs)} head jobs, "
                f"makespan = {st.partial_sol.makespan}"
            )

        for batch_idx, job_sublist in enumerate(sequence_of_job_sublist, start=1):
            if self._should_stop_before_next_batch(
                completed_batch_elapsed_sec_list=batch_elapsed_sec_list,
                stop_before_final_reserve=stop_before_final_reserve,
                successor_reserve_sec=successor_reserve_sec,
                time_guard_estimate_safety_factor=time_guard_estimate_safety_factor,
                time_guard_min_completed_batches=time_guard_min_completed_batches,
                next_batch_idx=batch_idx,
                total_batch_count=len(sequence_of_job_sublist),
            ):
                break

            batch_timer = ElapsedTimer()
            st.current_job_id_list.extend(job_sublist)

            # Use MixedDispatcher for simplified dispatch with multiple strategy exploration
            dispatcher = MixedDispatcher(instance)
            base_schedule = (
                self.ctx.create_empty_schedule_from_ins()
                if self._st.partial_sol is None
                else self._st.partial_sol.deepcopy()
            )
            partial_sol_best = dispatcher.get_best_mixed_schedule_by_sequence(
                job_sublist,
                schedule=base_schedule,
                from_stage=instance.stage_id_list[0],
                head_for_all_stages=True,
            )
            if partial_sol_best is None:
                logging.warning(
                    "MixedDispatcher returned None; falling back to dispatch_job_by_stages."
                )
                partial_sol_best = base_schedule
                for j in job_sublist:
                    partial_sol_best.dispatch_job_by_stages(j, job_2_stage_2_p_dict[j])
            logging.info(
                f"After dispatching job sublist, partial solution makespan is {partial_sol_best.makespan}"
            )
            _, new_sol = self._solve_cp_model(
                partial_sol_best,
                instance,
                stage_2_job_2_p_dict,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
                max_time_per_add=max_time_per_add,
                minimize_sum_ci_lex=minimize_sum_ci_lex,
                max_time_per_add_2nd_obj=max_time_per_add_2nd_obj,
                minimize_sum_ci_lin=minimize_sum_ci_lin,
                tighten_ranges=tighten_ranges,
                link_job_completion=link_job_completion,
                do_make_semi_active=make_semi_active_every_cp,
                solver_thread_cnt=solver_thread_cnt,
                use_lns_only=use_lns_only,
            )
            last_timestamp = st.timer.elapsed_sec
            logging.info(
                "After CP adjustment, solution makespan is %d.", new_sol.makespan
            )

            # Update the last partial solution
            st.partial_sol = new_sol

            # Update full solution
            if not st.all_jobs_are_included(job_cnt):
                # Dispatch remaining tail jobs to create a schedule feasible to the original problem
                remaining_jobs = [
                    j for j in tail_jobs if j not in st.current_job_id_list
                ]
                dispatcher = MixedDispatcher(instance)
                _temp_sol = dispatcher.get_best_mixed_schedule_by_sequence(
                    remaining_jobs,
                    schedule=st.partial_sol.deepcopy(),
                    from_stage=instance.stage_id_list[0],
                    head_for_all_stages=True,
                )
                if _temp_sol is None:
                    logging.warning(
                        "MixedDispatcher returned None for remaining jobs; falling back to dispatch_job_by_stages."
                    )
                    st.full_sol = st.partial_sol.deepcopy()
                    for j in remaining_jobs:
                        st.full_sol.dispatch_job_by_stages(j, job_2_stage_2_p_dict[j])
                else:
                    st.full_sol = _temp_sol
            else:
                st.full_sol = st.partial_sol
            if st.full_sol.makespan < best_full_obj:
                best_full_obj = st.full_sol.makespan
                best_full_sol = st.full_sol.deepcopy()
                logging.info(
                    "NEH-CP best feasible full schedule improved to %d.",
                    best_full_obj,
                )

            # Obj. value of dispatched solution as a value
            sub_obj_store.add_obj_value(
                last_timestamp, int(st.full_sol.makespan), is_maximize=None
            )

            # Obj. values of Un-dispatched solution as bounds
            sub_obj_store.add_obj_bound(
                last_timestamp, int(st.partial_sol.makespan), is_maximize=None
            )
            _last_timestamp_note = f"{st.job_subset_cnt}/{job_cnt}"
            sub_obj_store.add_last_timestamp_note(
                _last_timestamp_note,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
            )
            batch_elapsed_sec_list.append(batch_timer.elapsed_sec)

        if error_if_infeasible:
            self.ctx.check_feasibility(best_full_sol.get_jik_2_start_time_map())

        return NehCpResult(
            schedule=best_full_sol,
            sub_obj_store=sub_obj_store,
            last_obj_value=best_full_obj,
        )

    def _should_skip_full_neh_before_first_batch(
        self,
        *,
        remaining_batch_count: int,
        max_time_per_add: float | None,
        minimize_sum_ci_lex: bool,
        max_time_per_add_2nd_obj: float | None,
        successor_reserve_sec: float,
        skip_if_estimated_neh_exceeds_remaining: bool,
        full_neh_estimate_safety_factor: float,
    ) -> bool:
        if not skip_if_estimated_neh_exceeds_remaining:
            return False
        if remaining_batch_count <= 0:
            return False
        if max_time_per_add is None:
            return False

        estimated_batch_sec = float(max_time_per_add)
        if minimize_sum_ci_lex and max_time_per_add_2nd_obj is not None:
            estimated_batch_sec += float(max_time_per_add_2nd_obj)
        estimated_full_neh_sec = (
            remaining_batch_count
            * estimated_batch_sec
            * float(full_neh_estimate_safety_factor)
        )
        remaining_before_final = self.ctx.get_remaining_sec_before_final_reserve()
        available_for_neh = max(0.0, remaining_before_final - successor_reserve_sec)
        if estimated_full_neh_sec > available_for_neh:
            logging.info(
                "NEH-CP time guard: skipping full NEH block before first batch "
                "because estimated full run %.2f sec (%d batches x %.2f sec x %.2f) "
                "exceeds available NEH time %.2f sec "
                "(remaining_before_final=%.2f, successor_reserve=%.2f).",
                estimated_full_neh_sec,
                remaining_batch_count,
                estimated_batch_sec,
                full_neh_estimate_safety_factor,
                available_for_neh,
                remaining_before_final,
                successor_reserve_sec,
            )
            return True

        return False

    def _should_stop_before_next_batch(
        self,
        *,
        completed_batch_elapsed_sec_list: list[float],
        stop_before_final_reserve: bool,
        successor_reserve_sec: float,
        time_guard_estimate_safety_factor: float,
        time_guard_min_completed_batches: int,
        next_batch_idx: int,
        total_batch_count: int,
    ) -> bool:
        if stop_before_final_reserve and self.ctx.final_time_reserve_is_reached():
            logging.info(
                "NEH-CP time guard: stopping before batch %d/%d because final "
                "time reserve has been reached.",
                next_batch_idx,
                total_batch_count,
            )
            return True

        remaining_before_final = self.ctx.get_remaining_sec_before_final_reserve()
        if successor_reserve_sec > 0 and remaining_before_final <= successor_reserve_sec:
            logging.info(
                "NEH-CP time guard: stopping before batch %d/%d because %.2f sec "
                "remaining before final reserve is <= %.2f sec successor reserve.",
                next_batch_idx,
                total_batch_count,
                remaining_before_final,
                successor_reserve_sec,
            )
            return True

        if len(completed_batch_elapsed_sec_list) < time_guard_min_completed_batches:
            return False

        recent_elapsed = completed_batch_elapsed_sec_list[-3:]
        estimated_next_batch_sec = (
            sum(recent_elapsed) / len(recent_elapsed) * time_guard_estimate_safety_factor
        )
        available_for_neh = max(0.0, remaining_before_final - successor_reserve_sec)
        if available_for_neh <= estimated_next_batch_sec:
            logging.info(
                "NEH-CP time guard: stopping before batch %d/%d because estimated "
                "next batch %.2f sec would exceed available NEH time %.2f sec "
                "(remaining_before_final=%.2f, successor_reserve=%.2f).",
                next_batch_idx,
                total_batch_count,
                estimated_next_batch_sec,
                available_for_neh,
                remaining_before_final,
                successor_reserve_sec,
            )
            return True

        return False

    @classmethod
    def _split_tail_jobs_into_batches(
        cls,
        tail_jobs: list[str],
        *,
        added_batch_size: int,
        added_batch_count: int | None,
        min_added_batch_count: int | None,
        max_added_batch_count: int | None,
    ) -> list[list[str]]:
        if not tail_jobs:
            return []
        if added_batch_size <= 0:
            raise ValueError("added_batch_size must be positive.")

        min_count = cls._normalize_optional_positive_int(
            min_added_batch_count,
            "min_added_batch_count",
        )
        max_count = cls._normalize_optional_positive_int(
            max_added_batch_count,
            "max_added_batch_count",
        )
        if min_count is not None and max_count is not None and max_count < min_count:
            raise ValueError(
                "max_added_batch_count must be greater than or equal to "
                "min_added_batch_count."
            )

        batch_count_from_size = max(
            1,
            (len(tail_jobs) + int(added_batch_size) - 1) // int(added_batch_size),
        )
        resolved_batch_count = (
            cls._normalize_optional_positive_int(
                added_batch_count,
                "added_batch_count",
            )
            if added_batch_count is not None
            else batch_count_from_size
        )
        if min_count is not None:
            resolved_batch_count = max(resolved_batch_count, min_count)
        if max_count is not None:
            resolved_batch_count = min(resolved_batch_count, max_count)
        resolved_batch_count = min(resolved_batch_count, len(tail_jobs))

        if added_batch_count is None and resolved_batch_count == batch_count_from_size:
            batches = [
                tail_jobs[i : i + int(added_batch_size)]
                for i in range(0, len(tail_jobs), int(added_batch_size))
            ]
        else:
            batches = cls._split_evenly_by_count(tail_jobs, resolved_batch_count)

        batch_sizes = [len(batch) for batch in batches]
        logging.info(
            "NEH-CP batch partition: tail_job_count=%d added_batch_size=%s "
            "requested_added_batch_count=%s min_added_batch_count=%s "
            "max_added_batch_count=%s resolved_batch_count=%d batch_sizes=%s",
            len(tail_jobs),
            added_batch_size,
            added_batch_count,
            min_added_batch_count,
            max_added_batch_count,
            len(batches),
            batch_sizes,
        )
        return batches

    @staticmethod
    def _split_evenly_by_count(
        jobs: list[str],
        batch_count: int,
    ) -> list[list[str]]:
        if batch_count <= 0:
            raise ValueError("batch_count must be positive.")
        batch_count = min(batch_count, len(jobs))
        base_size, remainder = divmod(len(jobs), batch_count)
        batches: list[list[str]] = []
        start = 0
        for batch_idx in range(batch_count):
            size = base_size + (1 if batch_idx < remainder else 0)
            end = start + size
            batches.append(jobs[start:end])
            start = end
        return batches

    @staticmethod
    def _normalize_optional_positive_int(
        value: int | None,
        name: str,
    ) -> int | None:
        if value is None:
            return None
        normalized = int(value)
        if normalized != value:
            raise ValueError(f"{name} must be an integer.")
        if normalized <= 0:
            raise ValueError(f"{name} must be positive.")
        return normalized

    def _create_sub_cp_model(
        self,
        partial_sol: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        minimize_sum_ci_lex: bool = False,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
    ) -> tuple[CustomCpModel, Params, CumulativeVars]:
        st = self._require_state()
        horizon: int = partial_sol.makespan
        # stage_2_mc_horizon: dict[str, dict[str, int]] = (
        #     partial_sol.get_stage_2_mc_2_last_end_time_map()
        # )
        sub_instance = create_instance_of_job_subset(instance, st.current_job_id_list)
        builder = BaseModelBuilder()
        mdl, params, variables = builder.build(
            sub_instance,
            horizon,
            minimize_sum_ci=minimize_sum_ci_lex,
            minimize_makespan_plus_sum_other_stages=minimize_sum_ci_lin,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        )
        # mdl, params, variables = builder.build_horizon_per_stage(
        #     sub_instance, stage_2_mc_horizon
        # )
        self._add_hints_and_additional_constraints(
            mdl,
            params,
            variables,
            partial_sol,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
        )

        return mdl, params, variables

    def _add_hints_and_additional_constraints(
        self,
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        partial_sol: HybridFlowshopLiteSchedule,
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
    ) -> None:
        st = self._require_state()
        # Apply hint from partial solution
        BaseModelBuilder.apply_start_hints_from_start_time_map(
            mdl,
            params,
            variables,
            partial_sol.get_jik_2_start_time_map(),
            ignore_integrity_check=False,
        )
        BaseModelBuilder.apply_end_hints_from_end_time_map(
            mdl,
            params,
            variables,
            partial_sol.get_jik_2_end_time_map(),
            ignore_integrity_check=False,
        )
        # Fix profile of operations in previous solution
        if st.partial_sol is not None:
            BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
                mdl,
                params,
                variables,
                st.partial_sol,
                profile_fix_by_machine=profile_fix_by_machine,
                machine_precedence_stride=machine_precedence_stride,
            )

    def _solve_cp_model(
        self,
        partial_sol: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        profile_fix_by_machine: bool = False,
        machine_precedence_stride: int = 1,
        max_time_per_add: float | None = None,
        minimize_sum_ci_lex: bool = False,
        max_time_per_add_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        tighten_ranges: bool = False,
        link_job_completion: bool = False,
        do_make_semi_active: bool = False,
        solver_thread_cnt: int | None = None,
        use_lns_only: bool = False,
    ) -> tuple[CpsatSolverReport, HybridFlowshopLiteSchedule]:
        if solver_thread_cnt is None:
            solver_thread_cnt = 1
        ctx = self.ctx
        st = self._require_state()

        # Build CP model with job_subset
        sub_cp_mdl, params, variables = self._create_sub_cp_model(
            partial_sol,
            instance,
            profile_fix_by_machine=profile_fix_by_machine,
            machine_precedence_stride=machine_precedence_stride,
            minimize_sum_ci_lin=minimize_sum_ci_lin,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        )

        _timelimit = self.ctx.get_remaining_time_limit(max_time_per_add)
        report_1: CpsatSolverReport = self.ctx.solve_cp_model_2(
            sub_cp_mdl,
            _timelimit,
            solver_thread_cnt,
            use_lns_only=use_lns_only,
            e_timer=st.timer,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
        )
        if not getattr(report_1, "is_feasible", False):
            logging.info("No solution from sub CP.")
            return report_1, partial_sol

        # If feasible, decode solution
        new_sol: HybridFlowshopLiteSchedule = ctx.create_schedule(params, variables)

        if do_make_semi_active:
            obj_val_before = new_sol.makespan
            new_sol.make_semi_active(stage_2_job_2_p_dict)
            obj_val_after = new_sol.makespan
            if obj_val_after != obj_val_before:
                logging.info(
                    f"NEH-CP: makespan before semi-active adjustment: {obj_val_before},"
                    f" after adjustment: {obj_val_after}."
                )

        # If new_sol is not better than partial_sol, keep partial_sol
        if new_sol.makespan >= partial_sol.makespan:
            logging.info(
                f"Sub CP solution ({new_sol.makespan}) is not better than partial"
                f" solution ({partial_sol.makespan}); keeping the partial solution."
            )
            new_sol = partial_sol

        if not minimize_sum_ci_lex:
            return report_1, new_sol

        # Build secondary CP model to minimize \sum(C_i)
        sub_cp_mdl, params, variables = self._create_sub_cp_model(
            new_sol,
            instance,
            minimize_sum_ci_lex=True,
            tighten_ranges=tighten_ranges,
            link_job_completion=link_job_completion,
        )

        _timelimit = self.ctx.get_remaining_time_limit(max_time_per_add_2nd_obj)
        report_2: CpsatSolverReport = self.ctx.solve_cp_model_2(
            sub_cp_mdl,
            _timelimit,
            solver_thread_cnt,
            use_lns_only=use_lns_only,
            e_timer=st.timer,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
        )
        return_report: CpsatSolverReport = report_1.copy(
            elapsed_time=report_1.elapsed_time + report_2.elapsed_time
        )
        if not getattr(report_2, "is_feasible", False):
            return return_report, new_sol

        # If feasible, decode solution
        new_sol_2 = ctx.create_schedule(params, variables)

        if do_make_semi_active:
            obj_val_before = new_sol_2.makespan
            new_sol_2.make_semi_active(stage_2_job_2_p_dict)
            obj_val_after = new_sol_2.makespan
            if obj_val_after != obj_val_before:
                logging.info(
                    f"NEH-CP: makespan before semi-active adjustment: {obj_val_before},"
                    f" after adjustment: {obj_val_after}."
                )

        return return_report, new_sol_2
