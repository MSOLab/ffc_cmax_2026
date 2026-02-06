import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mbls.cpsat import CpsatSolverReport, CustomCpModel, ObjValueBoundStore
from ortools.sat.python.cp_model import CpModel
from routix import ElapsedTimer
from schore.parameters_examples import HybridFlowshopParameters

from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder, CumulativeVars
from hybridflowshop.cpsat_model_2.params import Params
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class NehCpContext(Protocol):
    """
    Minimal dependency interface.
    (FlowshopTardinessControllerCore is effectively designed to satisfy this interface.)
    """

    def get_remaining_time_limit(
        self, subroutine_time_limit: float | None
    ) -> float: ...

    # CP solve & decoding
    def solve_cp_model_2(
        self,
        mdl: CpModel,
        computational_time: float,
        solver_thread_cnt: int,
        e_timer: ElapsedTimer | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
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
        max_time_per_add: float | None = None,
        solver_thread_cnt: int | None = None,
        error_if_infeasible: bool = False,
    ) -> NehCpResult:
        timer = ElapsedTimer()

        _added_batch_size: int = (
            added_batch_size
            if added_batch_size is not None and added_batch_size > 0
            else 1
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

        job_sequence: list[str] = self.get_midpoint_sequence(instance, ref_schedule)
        job_cnt = len(job_sequence)
        sequence_of_job_sublist = [
            job_sequence[i : i + _added_batch_size]
            for i in range(0, len(job_sequence), _added_batch_size)
        ]

        st = self._require_state()
        for job_sublist in sequence_of_job_sublist:
            st.current_job_id_list.extend(job_sublist)

            # Solution of dispatching job_sublist by jobs to the schedule of last_solution
            partial_sol_dj: HybridFlowshopLiteSchedule = (
                self.ctx.create_empty_schedule_from_ins()
                if self._st.partial_sol is None
                else self._st.partial_sol.deepcopy()
            )

            for j in job_sublist:
                partial_sol_dj.dispatch_job_by_stages(j, job_2_stage_2_p_dict[j])

            # Solution of dispatching job_sublist by stages to the schedule of last_solution
            partial_sol_ds: HybridFlowshopLiteSchedule = (
                self.ctx.create_empty_schedule_from_ins()
                if self._st.partial_sol is None
                else self._st.partial_sol.deepcopy()
            )
            for i in instance.stage_id_list:
                partial_sol_ds.dispatch_stage_by_jobs(
                    i, job_sublist, stage_2_job_2_p_dict[i]
                )

            # Select the best partial solution
            partial_sol_best = (
                partial_sol_dj
                if partial_sol_dj.makespan <= partial_sol_ds.makespan
                else partial_sol_ds
            )

            mdl, params, variables = self._create_sub_cp_model(
                partial_sol_best, instance
            )

            report, new_sol = self._solve_cp_model(
                partial_sol_best,
                instance,
                max_time_per_add,
                solver_thread_cnt,
            )
            last_timestamp = st.timer.elapsed_sec

            # Update the last partial solution
            st.partial_sol = new_sol

            # Update full solution
            if not st.all_jobs_are_included(job_cnt):
                # Dispatch remaining jobs to create a schedule feasible to the original problem
                all_dispatched_sol_dj = st.partial_sol.deepcopy()
                remaining_jobs = [
                    j for j in job_sequence if j not in st.current_job_id_list
                ]
                for j in remaining_jobs:
                    all_dispatched_sol_dj.dispatch_job_by_stages(
                        j, job_2_stage_2_p_dict[j]
                    )

                all_dispatched_sol_ds = st.partial_sol.deepcopy()
                remaining_jobs = [
                    j for j in job_sequence if j not in st.current_job_id_list
                ]
                for i in instance.stage_id_list:
                    all_dispatched_sol_ds.dispatch_stage_by_jobs(
                        i, remaining_jobs, stage_2_job_2_p_dict[i]
                    )

                st.full_sol = (
                    all_dispatched_sol_dj
                    if all_dispatched_sol_dj.makespan <= all_dispatched_sol_ds.makespan
                    else all_dispatched_sol_ds
                )
            else:
                st.full_sol = st.partial_sol

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

        if error_if_infeasible:
            self.ctx.check_feasibility(st.full_sol.get_start_time_map())
        logging.info(f"NEH-CP done with makespan={st.full_sol.makespan}")

        return NehCpResult(
            schedule=st.full_sol,
            sub_obj_store=sub_obj_store,
            last_obj_value=st.full_sol.makespan,
        )

    @staticmethod
    def get_midpoint_sequence(
        instance: HybridFlowshopParameters,
        schedule: HybridFlowshopLiteSchedule,
    ) -> list[str]:
        """Get job sequence based on midpoint criteria.

        Args:
            schedule (HybridFlowshopLiteSchedule): The hybrid flowshop schedule.

        Returns:
            list[str]: A list of job names ordered by midpoint criteria.
        """
        start_map = schedule.get_start_time_map()
        end_map = schedule.get_end_time_map()
        jobs = instance.job_id_list
        idx_map = {j: idx for idx, j in enumerate(jobs)}
        first_stage = instance.stage_id_list[0]
        last_stage = instance.stage_id_list[-1]

        seq_info: list[tuple[float, int, int, str]] = []
        for j in jobs:
            # find any machine k for first and last stage
            s_first = next(
                t
                for (job, stage, _), t in start_map.items()
                if job == j and stage == first_stage
            )
            e_last = next(
                t
                for (job, stage, _), t in end_map.items()
                if job == j and stage == last_stage
            )
            midpoint = (s_first + e_last) / 2
            seq_info.append((midpoint, s_first, idx_map[j], j))

        seq_info.sort(key=lambda x: (x[0], x[1], x[2]))
        return [info[3] for info in seq_info]

    def _create_sub_cp_model(
        self,
        partial_sol: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
    ) -> tuple[CustomCpModel, Params, CumulativeVars]:
        st = self._require_state()
        horizon: int = partial_sol.makespan
        sub_instance = instance.create_instance_of_job_subset(st.current_job_id_list)
        builder = BaseModelBuilder()
        mdl, params, variables = builder.build(sub_instance, horizon)
        self._add_hints_and_additional_constraints(mdl, params, variables, partial_sol)

        return mdl, params, variables

    def _add_hints_and_additional_constraints(
        self,
        mdl: CustomCpModel,
        params: Params,
        variables: CumulativeVars,
        partial_sol: HybridFlowshopLiteSchedule,
    ) -> None:
        st = self._require_state()
        # Apply hint from partial solution
        BaseModelBuilder.apply_start_hints_from_start_time_map(
            mdl, params, variables, partial_sol.get_start_time_map()
        )
        # Fix profile of operations in previous solution
        if st.partial_sol is not None:
            BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
                mdl, params, variables, st.partial_sol
            )

    def _solve_cp_model(
        self,
        partial_sol: HybridFlowshopLiteSchedule,
        instance: HybridFlowshopParameters,
        max_time_per_add: float | None = None,
        solver_thread_cnt: int | None = None,
    ) -> tuple[CpsatSolverReport, HybridFlowshopLiteSchedule]:
        if solver_thread_cnt is None:
            solver_thread_cnt = 1
        ctx = self.ctx
        st = self._require_state()

        # Build CP model with job_subset
        sub_cp_mdl, params, variables = self._create_sub_cp_model(partial_sol, instance)

        _timelimit = self.ctx.get_remaining_time_limit(max_time_per_add)
        report = self.ctx.solve_cp_model_2(
            sub_cp_mdl,
            _timelimit,
            solver_thread_cnt,
            e_timer=st.timer,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
        )
        if not getattr(report, "is_feasible", False):
            logging.info("No solution from sub CP.")
            return report, partial_sol

        # If feasible, decode solution
        new_sol = ctx.create_schedule(params, variables)

        # If new_sol is not better than partial_sol, keep partial_sol
        if new_sol.makespan >= partial_sol.makespan:
            logging.info(
                "Sub CP solution is not better than partial solution;"
                " keeping the partial solution."
            )
            new_sol = partial_sol
        return report, new_sol
