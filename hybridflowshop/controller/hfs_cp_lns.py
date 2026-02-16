import logging
import math
import random
from typing import Any, Callable

from mbls.cpsat import CpsatStatus
from routix import ElapsedTimer
from schore.parameters_examples import HybridFlowshopParameters

from hybridflowshop.controller.neh_cp import NehCpConstructor, NehCpResult
from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder
from hybridflowshop.report import HfsSubroutineReport
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule
from hybridflowshop.select_and_assign import solve_selection_problem
from identical_parallel_machine.cumulative import ParallelMcParams, ParallelMcVars
from identical_parallel_machine.solver import SolveConfig, configure_solver

from .bottleneck_stage_schedule_heuristic_option import (
    BottleneckStageScheduleHeuristicOption,
)
from .controller_core import HybridFlowShopCpLnsControllerCore
from .reactive.reactive_looper import ReactiveLooper


class HybridFlowShopCpLnsController(HybridFlowShopCpLnsControllerCore):
    """
    Controller for solving Hybrid Flow Shop problems using CP-based algorithms.
    """

    # Start subroutine definition

    # Subroutine: solve base CP model

    def solve_base_cp_model(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        make_semi_active_after_cp: bool = False,
        is_initial_solution: bool = False,
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
            computational_time (float): The maximum computational time in seconds for solving the CP model.
            solver_thread_cnt (int): The number of parallel workers (threads) to use during search.
            is_initial_solution (bool, optional): If True, marks this run as producing the initial solution (affects summary/logging). Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution after solving. Defaults to False.
        """
        if self.base_cp_model_is_set:
            self.cp_model.delete_added_constraints()
        else:
            raise RuntimeError(
                "Base CP model is not set. Call set_cp_model_as_base_cp_model() first."
            )

        _should_be_init: bool = self.solution_manager.get_incumbent() is None
        _is_init: bool = _should_be_init or is_initial_solution

        if _is_init:
            report, solution = self.solve_current_cp_remaining_time_limit(
                computational_time,
                solver_thread_cnt,
                make_semi_active_after_cp=make_semi_active_after_cp,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                is_initial_solution=True,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )
        else:
            # If it is not an initial solution, apply the incumbent solution as a hint
            report, solution = self.solve_with_initial_solution(
                computational_time,
                solver_thread_cnt,
                make_semi_active_after_cp=make_semi_active_after_cp,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
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

    # Helper method for LNS-CP

    def _fix_profile_solve_reset(
        self,
        profile_fixing_method: Callable,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        make_semi_active_after_cp: bool = False,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Apply the profile fixing method, solve, and reset the model.

        Args:
            profile_fixing_method (Callable): A callable that applies the profile fixing method to the CP model.
            computational_time (float): The maximum computational time in seconds.
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
                    report = HfsSubroutineReport(
                        elapsed_time=swap_timer.elapsed_sec,
                        obj_value=float(swapped_obj_value),
                        obj_bound=None,
                        is_init=False,
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
        report, solution = self.solve_with_initial_solution(
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            make_semi_active_after_cp=make_semi_active_after_cp,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )
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

    def _fix_operations_profile_except_selected(
        self,
        rescheduled_ops: set[tuple[str, str, str]],
    ) -> None:
        """
        Helper to deep-copy incumbent solution and remove operations to be rescheduled,
        and add the CP model constraints that enforce precedences/machine assignments for
        the profile-fixed operations.

        Args:
            rescheduled_ops (set[tuple[str, str, str]]): set of (job, stage, machine) tuples
                that are not part of the block (i.e., they will be re-optimized).

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
        BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
            self.cp_model, self.params, self.vars, out_of_block_ops_sch
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
        make_semi_active_after_cp: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_operation_block_operator(
                rho, seed_op_from_critical_block=seed_op_from_critical_block
            ),
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_operation_block_operator(
        self, rho: float, seed_op_from_critical_block: bool = False
    ) -> None:
        """Apply the operation-block operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            seed_op_from_critical_block (bool, optional): If True, the seed operation is
                chosen from critical blocks of the incumbent solution. Defaults to False.

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
        self._fix_operations_profile_except_selected(selected_ops)

    @staticmethod
    def closed_intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
        """Check if two closed time intervals overlap."""
        return not (e1 <= s2 or e2 <= s1)

    @staticmethod
    def open_intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
        """Check if two open time intervals overlap."""
        return not (e1 < s2 or e2 < s1)

    # Subroutine: Stage neighbor search

    def stage_block_ns(
        self,
        rho: float,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        seed_stage_from_non_singleton_cb: bool = False,
        make_semi_active_after_cp: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_stage_operator(
                rho, seed_stage_from_non_singleton_cb=seed_stage_from_non_singleton_cb
            ),
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_stage_operator(
        self, rho: float, seed_stage_from_non_singleton_cb: bool = False
    ):
        """
        Apply the "stage" LNS operator: free a consecutive subset of stages (i.e. allow
        operations on those stages to be rescheduled) and fix the profile of all other
        operations according to the current incumbent schedule.

        Args:
            rho (float): The proportion of stages to free (must be between 0 and 1).
            seed_stage_from_non_singleton_cb (bool, optional): Whether to seed the stage
                selection from non-singleton critical blocks. Defaults to False.

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
        self._fix_operations_profile_except_selected(selected_ops)

    # Subroutine: Job-block neighbor search

    def job_block_ns(
        self,
        rho: float,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        swap_before_cp: bool = False,
        seed_op_from_critical_block: bool = False,
        make_semi_active_after_cp: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_job_block_operator(
                rho, seed_op_from_critical_block=seed_op_from_critical_block
            ),
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            swap_before_cp=swap_before_cp,
            make_semi_active_after_cp=make_semi_active_after_cp,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_job_block_operator(
        self, rho: float, seed_op_from_critical_block: bool = False
    ) -> None:
        """Apply the job-block operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            seed_op_from_critical_block (bool, optional): If True, the seed operation is
                chosen from critical blocks of the incumbent solution. Defaults to False.

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
        self._fix_operations_profile_except_selected(selected_ops)

    # Subroutine: Johnson-based Heuristic for initialization

    def _dispatch_by_job_stage_time(
        self,
        job_sequence: list[str],
        schedule: HybridFlowshopLiteSchedule,
        draw_gantt: bool = False,
    ):
        """
        Dispatches jobs according to the given sequence and registers the resulting
        solution with the solution manager.

        This is the classic job-major (job -> stage -> time) dispatching strategy.

        Args:
            job_sequence (list[str]): The sequence of job IDs to be dispatched (dispatch order).
            schedule (HybridFlowshopLiteSchedule): The schedule to which jobs are dispatched.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        for idx, j in enumerate(job_sequence):
            schedule.dispatch_job_by_stages(j, self.job_2_stage_2_p_dict[j])
            # TODO: uncomment only for debug purpose
            # output_path = self.get_file_path_for_subroutine(f"_gantt_{idx}_{j}.png")
            # self.draw_gantt(schedule, output_path=output_path)

        # Create report and register the new solution
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(schedule.makespan),
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log and draw Gantt chart if the solution is an improvement
        if was_updated:
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(
                log_time, float(schedule.makespan), is_maximize=False
            )
            if draw_gantt:
                self.draw_incumbent_gantt()

    def _dispatch_by_stage_time_job(
        self,
        job_sequence: list[str],
        schedule: HybridFlowshopLiteSchedule,
        draw_gantt: bool = False,
    ):
        """
        For each stage (in order), dispatch jobs in the given job_sequence order.
        For each job in the sequence, schedule its operation in the current stage to the earliest available machine.

        This is the stage-major (stage -> job -> time) dispatching strategy.
        In hybrid flowshop, this is equivalent to dispatching jobs by stage, then by job, then by earliest time.

        Args:
            job_sequence (list[str]): The sequence of job IDs to be dispatched (dispatch order).
            schedule (HybridFlowshopLiteSchedule): The schedule to which jobs are dispatched.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        for idx, i in enumerate(self.instance.stage_id_list):
            schedule.dispatch_stage_by_jobs(
                i, job_sequence, self.stage_2_job_2_p_dict[i]
            )
            # TODO: uncomment only for debug purpose
            # output_path = self.get_file_path_for_subroutine(f"_gantt_{idx}_{j}.png")
            # self.draw_gantt(schedule, output_path=output_path)

        # Create report and register the new solution
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(schedule.makespan),
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log and draw Gantt chart if the solution is an improvement
        if was_updated:
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(
                log_time, float(schedule.makespan), is_maximize=False
            )
            if draw_gantt:
                self.draw_incumbent_gantt()

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

    def get_tp_sequence(self, k: int) -> list[str]:
        """
        Get two-partition sequence given $k$.

        - For each job, consider the sum of processing times of stages 0 to k-1
          as the first part (p1),
        - and the sum of processing times of stages k to m-1 as the second part
          (p2).
        - Create a job sequence by applying Johnson's rule for F2||C_max.

        When k = num_stages // 2, result is the same as get_sequence2().

        Args:
            k (int): Index between 1 and m-1, where m is the number of stages.

        Returns:
            list[str]: A list of job IDs ordered according to the TP rule.
        """
        jobs = self.instance.job_id_list
        stages = self.instance.stage_id_list
        p_dict: dict[tuple[str, str], int] = (
            self.instance.p_manager.job_stage_2_value_map(jobs, stages)
        )

        p1_stages = stages[:k]  # Sum over stages 0 ~ k-1
        p2_stages = stages[k:]  # Sum over stages k ~ m-1
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=None,
            obj_bound=obj_bound,
            is_init=True,
        )
        self.solution_manager.register(report, None)

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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
        # Subroutine states
        best_makespan = float("inf")
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_k = -1

        for k in range(1, self.instance.stage_count):
            # Create an empty schedule
            schedule = self.create_empty_schedule_from_ins()
            # Dispatch
            job_sequence = self.get_cds_sequence(k)
            for j in job_sequence:
                schedule.dispatch_job_by_stages(j, self.job_2_stage_2_p_dict[j])
            # Update subroutine states
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        logging.info(f"Schedule by DJ(CDS): makespan={best_makespan} with k={best_k}")
        return best_schedule

    def initialize_by_dj_tp(
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

        # Subroutine states
        best_makespan = float("inf")
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_k = -1

        for k in range(0, self.instance.stage_count):
            # Create an empty schedule
            schedule = self.create_empty_schedule_from_ins()
            # Dispatch
            job_sequence = self.get_tp_sequence(k)
            for j in job_sequence:
                schedule.dispatch_job_by_stages(j, self.job_2_stage_2_p_dict[j])
            # Update subroutine states
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        if error_if_infeasible:
            self.check_feasibility(best_schedule.get_jik_2_start_time_map())
        logging.info(f"Best schedule found with k={best_k}, makespan={best_makespan}")

        # Create report and register the new solution
        obj_value = float(best_schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, best_schedule)

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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
        schedule = self.create_empty_schedule_from_ins()
        # Dispatch
        job_sequence = self.get_gupta_sequence()
        for j in job_sequence:
            schedule.dispatch_job_by_stages(j, self.job_2_stage_2_p_dict[j])

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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
        schedule = self.create_empty_schedule_from_ins()
        # Dispatch
        job_sequence = self.get_palmer_sequence()
        for j in job_sequence:
            schedule.dispatch_job_by_stages(j, self.job_2_stage_2_p_dict[j])
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
        best_makespan = float("inf")
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_k = -1
        for k in range(1, self.instance.stage_count):
            # Create an empty schedule
            schedule = self.create_empty_schedule_from_ins()
            job_sequence = self.get_cds_sequence(k)
            for i in self.instance.stage_id_list:
                schedule.dispatch_stage_by_jobs(
                    i, job_sequence, self.stage_2_job_2_p_dict[i]
                )
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        logging.info(f"Schedule by DS(CDS): makespan={best_makespan} with k={best_k}")
        return best_schedule

    def initialize_by_ds_tp(
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

        best_makespan = float("inf")
        best_schedule: HybridFlowshopLiteSchedule | None = None
        best_k = -1
        for k in range(0, self.instance.stage_count):
            # Create an empty schedule
            schedule = self.create_empty_schedule_from_ins()
            job_sequence = self.get_tp_sequence(k)
            for i in self.instance.stage_id_list:
                schedule.dispatch_stage_by_jobs(
                    i, job_sequence, self.stage_2_job_2_p_dict[i]
                )
            makespan = schedule.makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_k = k

        if best_schedule is None:
            raise ValueError("No schedule found after applying CDS sequence.")
        if error_if_infeasible:
            self.check_feasibility(best_schedule.get_jik_2_start_time_map())
        logging.info(f"Best schedule found with k={best_k}, makespan={best_makespan}")

        # Create report and register the new solution
        obj_value = float(best_schedule.makespan)
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, best_schedule)

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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
        schedule = self.create_empty_schedule_from_ins()
        # Dispatch
        job_sequence = self.get_gupta_sequence()
        for i in self.instance.stage_id_list:
            schedule.dispatch_stage_by_jobs(
                i, job_sequence, self.stage_2_job_2_p_dict[i]
            )

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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
        schedule = self.create_empty_schedule_from_ins()
        # Dispatch
        job_sequence = self.get_palmer_sequence()
        for i in self.instance.stage_id_list:
            schedule.dispatch_stage_by_jobs(
                i, job_sequence, self.stage_2_job_2_p_dict[i]
            )

        logging.info(f"Schedule by DS(Palmer): makespan={schedule.makespan}")
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
            f"Best schedule found by {best_method_name} with makespan={best_makespan}"
        )
        return best_schedule

    def get_incumbent_midpoint_sequence(self) -> list[str]:
        """
        Returns a job sequence based on the incumbent solution, sorted in ascending order by:

        1. midpoint := (start_time at first stage + end_time at last stage) / 2
        2. tie-break by first stage start_time
        3. tie-break by original job index

        Raises:
            ValueError: if no incumbent solution is available.

        Returns:
            list[str]: A list of job IDs representing the midpoint sequence.
        """
        incumbent = self.solution_manager.get_incumbent()
        if incumbent is None:
            raise ValueError(
                "No incumbent solution available to build midpoint sequence."
            )

        start_map = incumbent.get_jik_2_start_time_map()
        end_map = incumbent.get_jik_2_end_time_map()
        jobs = self.instance.job_id_list
        idx_map = {j: idx for idx, j in enumerate(jobs)}
        first_stage = self.instance.stage_id_list[0]
        last_stage = self.instance.stage_id_list[-1]

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

    def neh_cp(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        max_time_per_add: float | None = None,
        cp_tl_nc_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        make_semi_active_every_cp: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Builds a CP-guided solution using a midpoint sequence from the incumbent solution.

        Args:
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            added_batch_size (int, optional): The number of jobs to add in each iteration.
                Defaults to 1.
            max_time_per_add (float | None, optional): Time limit (in seconds) for solving each incremental subproblem.
                If None, uses the remaining time limit. Defaults to None.
            cp_tl_nc_multiplier (float | None, optional): Multiplier for the time limit of each CP subproblem.
                If None, uses the default value. Defaults to None.
            cp_tl_c_multiplier (float | None, optional): Multiplier for the time limit of each CP subproblem.
                If None, uses the default value. Defaults to None.
            error_if_infeasible (bool, optional): If True, raises an error if the solution is infeasible.
                Defaults to False.
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
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            make_semi_active_every_cp=make_semi_active_every_cp,
            solver_thread_cnt=solver_thread_cnt,
            error_if_infeasible=error_if_infeasible,
        )
        obj_value = float(result.last_obj_value)
        logging.info(f"NEH-CP done with makespan {obj_value}")

        # Create report for the final solution and register it
        final_report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
        )
        was_updated: bool = self.solution_manager.register(
            final_report, result.schedule
        )

        if was_updated:
            # Re-define base CP model with the new makespan
            self.set_cp_model_as_base_cp_model()
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(log_time, obj_value, is_maximize=False)
            if draw_gantt:
                self.draw_incumbent_gantt()

        # Write the objective store to a YAML file
        # TODO: suffix from output_metadata
        if result.sub_obj_store:
            result.sub_obj_store.save_yaml(
                self.get_file_path_for_subroutine("_obj_log.yaml")
            )

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

        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )

        # Register report & solution
        self.solution_manager.register(report, solution)

        # Log (time, objective value & bound)
        log_time = self.timer.elapsed_sec
        _last_timestamp_note = self._get_call_context_of_current_method()

        self.extend_obj_value_log(obj_value_records, is_maximize=False)
        obj_value: float | None = self.obj_store.get_last_obj_value()
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

    # Subroutine: bottleneck parallel machine scheduling

    def bottleneck_parallel_mc(
        self,
        computational_time: float | None = None,
        solver_thread_cnt: int | None = None,
    ) -> None:
        # # Identify bottleneck stage
        # stage_id_2_total_p = {}
        # for stage_id in self.instance.stage_id_list:
        #     stage_id_2_total_p[stage_id] = sum(
        #         self.stage_2_job_2_p_dict[stage_id][job_id]
        #         for job_id in self.instance.job_id_list
        #     )
        # # pprint(stage_id_2_total_p)
        # # stage_2_machine_count = {
        # #     stage_id: len(self.instance.stage_2_machines_map[stage_id])
        # #     for stage_id in self.instance.stage_id_list
        # # }
        # # pprint(stage_2_machine_count)
        # stage_id_2_bottleneck_index = {
        #     stage_id: total_p / len(self.instance.stage_2_machines_map[stage_id])
        #     for stage_id, total_p in stage_id_2_total_p.items()
        # }
        # # pprint(stage_id_2_bottleneck_index)

        # # Bottleneck stage is the one with the highest stage_id_2_bottleneck_index
        # bottleneck_stage_id = max(
        #     stage_id_2_bottleneck_index, key=stage_id_2_bottleneck_index.get
        # )

        stage_2_shd_bound = {
            stage: self.get_shdlb_for_stage(stage)
            for stage in self.instance.stage_id_list
        }
        bottleneck_stage_id = max(stage_2_shd_bound, key=stage_2_shd_bound.get)

        logging.info(f"Bottleneck stage: {bottleneck_stage_id}")

        # From hybrid flow shop problem define parallel machine scheduling problem for the bottleneck stage
        bottleneck_stage_index = self.instance.stage_id_list.index(bottleneck_stage_id)
        before_stage_id_list = self.instance.stage_id_list[:bottleneck_stage_index]
        after_stage_id_list = self.instance.stage_id_list[bottleneck_stage_index + 1 :]

        # logging.info("Before stages: %s", before_stage_id_list)
        # logging.info("After stages: %s", after_stage_id_list)

        r_dict = {
            j: sum(self.job_2_stage_2_p_dict[j][s] for s in before_stage_id_list)
            for j in self.instance.job_id_list
        }
        p_dict = {
            j: self.job_2_stage_2_p_dict[j][bottleneck_stage_id]
            for j in self.instance.job_id_list
        }
        tr_dict = {
            j: sum(self.job_2_stage_2_p_dict[j][s] for s in after_stage_id_list)
            for j in self.instance.job_id_list
        }
        # pprint(r_dict)
        # pprint(p_dict)
        # pprint(tr_dict)

        from identical_parallel_machine.cumulative import ParallelMcModelBuilder

        builder = ParallelMcModelBuilder()
        mdl, params, variables = builder.build(
            self.instance.job_id_list,
            self.instance.stage_2_machines_map[bottleneck_stage_id],
            p_dict,
            r_dict,
            tr_dict,
            horizon=self.get_horizon(),
        )

        solve_cfg = SolveConfig(
            log_search_progress=self.log_search_progress,
            time_limit_s=self.get_remaining_time_limit(computational_time),
            num_workers=solver_thread_cnt,
        )
        self.solver = configure_solver(solve_cfg)
        cp_solver_status = self.solver.solve(mdl)
        cpsat_status = CpsatStatus.from_cp_solver_status(cp_solver_status)
        elapsed_time = self.solver.wall_time
        if cpsat_status.is_feasible:
            obj_value = self.solver.objective_value
            if cpsat_status == CpsatStatus.OPTIMAL:
                obj_bound = obj_value
            else:
                obj_bound = self.solver.best_objective_bound
            logging.info(
                f"Bottleneck parallel MC done with obj_value {obj_value} and obj_bound {obj_bound} in {elapsed_time:.2f} seconds."
            )
        else:
            obj_value, obj_bound = CpsatStatus.get_obj_value_and_bound_for_infeasible(
                False
            )
            logging.info(
                f"Bottleneck parallel MC found no feasible solution in {elapsed_time:.2f} seconds."
            )

        if cpsat_status.is_feasible:
            # Make bottleneck-stage-only schedule
            bottleneck_only_schedule = self._extract_schedule_from_parallel_mc_solution(
                params, variables, bottleneck_stage_id
            )
            # Draw Gantt chart; force start time as zero and end time as makespan by CP
            self.draw_gantt(
                bottleneck_only_schedule, force_start=0, force_end=int(obj_value)
            )
            # Create a job sequence from the bottleneck-only schedule
            # Sort by (start time at bottleneck stage - r_dict[j], tie-break by original job index)
            bottleneck_only_start_time_map = self.extract_job_2_start_time_map(
                params, variables
            )
            sorted_j_list = sorted(
                params.j_list,
                key=lambda j: (
                    bottleneck_only_start_time_map[j] - r_dict[j],
                    self.instance.job_id_list.index(j),
                ),
            )
            dispatched_schedule = self.create_empty_schedule_from_ins()
            for j in sorted_j_list:
                dispatched_schedule.dispatch_job_by_stages(
                    j, self.job_2_stage_2_p_dict[j]
                )
            dispatched_obj_value = dispatched_schedule.makespan
            logging.info(
                f"Bottleneck parallel MC: CP_obj={int(obj_value)}, full_schedule_obj={dispatched_obj_value}"
            )

            report = HfsSubroutineReport(
                elapsed_time=elapsed_time,
                obj_value=dispatched_obj_value,
                obj_bound=obj_bound,
                is_init=True,
            )
            self.solution_manager.register(report, dispatched_schedule)

            # solution_dict = {
            #     START_TIME_MAP_KEY: tuple_to_pyyaml_key(schedule.get_jik_2_start_time_map()),
            #     END_TIME_MAP_KEY: tuple_to_pyyaml_key(schedule.get_jik_2_end_time_map()),
            # }
            # object_to_yaml(
            #     solution_dict,
            #     self.get_file_path_for_subroutine("_solution.yaml"),
            #     encoding="utf-8",
            # )

    def _extract_schedule_from_parallel_mc_solution(
        self, params: ParallelMcParams, variables: ParallelMcVars, target_stage_id: str
    ) -> HybridFlowshopLiteSchedule:
        start_time_map = self.extract_job_2_start_time_map(params, variables)

        schedule = self.create_empty_schedule_from_ins()
        sorted_j_list = sorted(
            params.j_list,
            key=lambda j: (start_time_map[j], params.j_list.index(j)),
        )
        for j in sorted_j_list:
            start_time = start_time_map[j]
            schedule.add_operation_2_stage(
                target_stage_id,
                j,
                self.job_2_stage_2_p_dict[j][target_stage_id],
                release_t=start_time,
            )

        return schedule

    def extract_job_2_start_time_map(
        self, params: ParallelMcParams, variables: ParallelMcVars
    ) -> dict[str, int]:
        start_time_map: dict[str, int] = {}
        """job ID -> start time"""
        for j in params.j_list:
            start_value = self.solver.Value(variables.op_start[j])
            start_time_map[j] = start_value
        return start_time_map

    def bottleneck_parallel_mc_2(
        self,
        computational_time: float | None = None,
        solver_thread_cnt: int | None = None,
    ) -> None:
        sub_timer = ElapsedTimer()

        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None
        for bottleneck_stage_id in self.instance.stage_id_list:
            # From hybrid flow shop problem define parallel machine scheduling problem for the bottleneck stage
            bottleneck_stage_index = self.instance.stage_id_list.index(
                bottleneck_stage_id
            )
            before_stage_id_list = self.instance.stage_id_list[:bottleneck_stage_index]
            after_stage_id_list = self.instance.stage_id_list[
                bottleneck_stage_index + 1 :
            ]

            r_dict = {
                j: sum(self.job_2_stage_2_p_dict[j][s] for s in before_stage_id_list)
                for j in self.instance.job_id_list
            }
            tr_dict = {
                j: sum(self.job_2_stage_2_p_dict[j][s] for s in after_stage_id_list)
                for j in self.instance.job_id_list
            }

            # Sort jobs by (r_j - tr_j, tie-break by original job index)
            # Jobs with higher (r_j - tr_j) value are scheduled first
            sorted_j_list = sorted(
                self.instance.job_id_list,
                key=lambda j: (
                    r_dict[j] - tr_dict[j],
                    self.instance.job_id_list.index(j),
                ),
            )

            job_dispatched_schedule = self.create_empty_schedule_from_ins()
            for j in sorted_j_list:
                job_dispatched_schedule.dispatch_job_by_stages(
                    j, self.job_2_stage_2_p_dict[j]
                )
            job_dispatched_obj_value = job_dispatched_schedule.makespan
            stage_dispatched_schedule = self.create_empty_schedule_from_ins()
            for i in self.instance.stage_id_list:
                stage_dispatched_schedule.dispatch_stage_by_jobs(
                    i, sorted_j_list, self.stage_2_job_2_p_dict[i]
                )
            stage_dispatched_obj_value = stage_dispatched_schedule.makespan
            if job_dispatched_obj_value < stage_dispatched_obj_value:
                dispatched_schedule = job_dispatched_schedule
                dispatched_obj_value = job_dispatched_obj_value
            else:
                dispatched_schedule = stage_dispatched_schedule
                dispatched_obj_value = stage_dispatched_obj_value

            if best_obj is None or dispatched_obj_value < best_obj:
                best_obj = dispatched_obj_value
                best_sch = dispatched_schedule

        logging.info(f"Bottleneck parallel MC: full_schedule_obj={best_obj}")

        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
        )
        self.solution_manager.register(report, best_sch)

        # solution_dict = {
        #     START_TIME_MAP_KEY: tuple_to_pyyaml_key(schedule.get_jik_2_start_time_map()),
        #     END_TIME_MAP_KEY: tuple_to_pyyaml_key(schedule.get_jik_2_end_time_map()),
        # }
        # object_to_yaml(
        #     solution_dict,
        #     self.get_file_path_for_subroutine("_solution.yaml"),
        #     encoding="utf-8",
        # )

    def _schedule_from_bottleneck_stage(
        self,
        bottleneck_stage_id: str,
        option: BottleneckStageScheduleHeuristicOption,
        draw_gantt: bool = False,
    ) -> HybridFlowshopLiteSchedule:
        """Schedule the entire hybrid flow shop from a single bottleneck stage.

        Uses two-way dispatching:
        1. Dispatch later stages (after bottleneck) based on bottleneck completion times
        2. Dispatch former stages (before bottleneck) using reversed instance with release times

        Args:
            bottleneck_stage_id: The bottleneck stage ID to schedule from
            option: The bottleneck stage schedule heuristic option
            draw_gantt: Whether to draw Gantt chart for the bottleneck stage only

        Returns:
            Complete schedule for all stages
        """
        bottleneck_schedule, bcmax = self._get_bottleneck_stage_schedule_heuristic(
            bottleneck_stage_id,
            option,
            draw_gantt=draw_gantt,
        )

        # Create a later-dispatched schedule by dispatching from the bottleneck schedule
        later_stage_list = self.instance.stage_id_list[
            self.instance.stage_id_list.index(bottleneck_stage_id) + 1 :
        ]
        logging.debug(f"Later stages: {later_stage_list}")
        if later_stage_list:
            bottleneck_stage_end_time_map = bottleneck_schedule.get_jik_2_end_time_map()
            # Sort jobs by end time at bottleneck stage (ascending)
            job_2_bottleneck_end_time = {}
            for (
                job_id,
                stage_id,
                _,
            ), end_time in bottleneck_stage_end_time_map.items():
                if stage_id == bottleneck_stage_id:
                    job_2_bottleneck_end_time[job_id] = end_time
            sorted_j_list = sorted(
                self.instance.job_id_list,
                key=lambda j: (
                    job_2_bottleneck_end_time[j],
                    self.instance.job_id_list.index(j),
                ),
            )
            later_ds_schedule = bottleneck_schedule.deepcopy()
            # Dispatch later stages
            for stage_id in later_stage_list:
                later_ds_schedule.dispatch_stage_by_jobs(
                    stage_id, sorted_j_list, self.stage_2_job_2_p_dict[stage_id]
                )
            later_ds_obj_value = later_ds_schedule.makespan
            later_dj_schedule = bottleneck_schedule.deepcopy()
            for job_id in sorted_j_list:
                later_dj_schedule.dispatch_job_by_stages(
                    job_id,
                    self.job_2_stage_2_p_dict[job_id],
                    from_stage=later_stage_list[0],
                )
            later_dj_obj_value = later_dj_schedule.makespan
            if later_ds_obj_value < later_dj_obj_value:
                later_schedule = later_ds_schedule
            else:
                later_schedule = later_dj_schedule
        else:
            later_schedule = bottleneck_schedule.deepcopy()

        if draw_gantt:
            self.draw_gantt(later_schedule, force_start=0)

        # Create a former-dispatched schedule
        before_stage_list = self.instance.stage_id_list[
            : self.instance.stage_id_list.index(bottleneck_stage_id)
        ]
        logging.debug(f"Before stages: {before_stage_list}")
        if before_stage_list:
            bottleneck_stage_start_time_map = (
                bottleneck_schedule.get_jik_2_start_time_map()
            )
            job_2_bottleneck_start_time = {}
            for (
                job_id,
                stage_id,
                _,
            ), start_time in bottleneck_stage_start_time_map.items():
                if stage_id == bottleneck_stage_id:
                    job_2_bottleneck_start_time[job_id] = start_time
            # Define a new problem for former stages
            instance_for_former_stages, job_2_release = (
                self._create_reversed_instance_for_former_stages(
                    before_stage_list, job_2_bottleneck_start_time, bcmax
                )
            )
            # Dispatch former stages
            former_schedule = self._dispatch_former_stages(
                instance_for_former_stages, job_2_release
            )
            former_schedule_makespan = former_schedule.makespan
            logging.debug(
                f"Former stages schedule makespan: {former_schedule_makespan}"
            )
            discrepancy = former_schedule_makespan - bcmax
            logging.debug(
                f"Discrepancy between former schedule and bottleneck schedule: {discrepancy}"
            )
            # Right-shift original schedule by discrepancy
            later_schedule.right_shift(discrepancy)

            former_schedule_end_time_map = former_schedule.get_jik_2_end_time_map()
            for op, end_time in former_schedule_end_time_map.items():
                job_id, stage_id, mc_id = op
                start_time = former_schedule_makespan - end_time
                duration = self.job_2_stage_2_p_dict[job_id][stage_id]
                later_schedule.add_ops_times_2_mc(
                    stage_id, mc_id, job_id, start_time, start_time + duration
                )
        later_schedule.make_semi_active(self.stage_2_job_2_p_dict)

        return later_schedule

    def bn2d_single_stage(
        self,
        left_cap_multiplier: int | None = None,
        right_cap_multiplier: int | None = None,
        left_cap_portion: float | None = None,
        right_cap_portion: float | None = None,
        normalize_by_stage_cnt: bool = False,
        reverse_mid_all: bool = False,
        reverse_mid_even: bool = False,
        randomize_mid_all: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Schedule from single bottleneck stage (loading index-based)."""
        sub_timer = ElapsedTimer()
        bottleneck_stage_id = self._get_bottleneck_stage()
        logging.info(f"Bottleneck stage: {bottleneck_stage_id}")

        bottleneck_stage_schedule_heuristic_option = (
            BottleneckStageScheduleHeuristicOption(
                left_cap_multiplier=left_cap_multiplier,
                right_cap_multiplier=right_cap_multiplier,
                left_cap_portion=left_cap_portion,
                right_cap_portion=right_cap_portion,
                normalize_by_stage_cnt=normalize_by_stage_cnt,
                reverse_mid_all=reverse_mid_all,
                reverse_mid_even=reverse_mid_even,
                randomize_mid_all=randomize_mid_all,
            )
        )
        schedule = self._schedule_from_bottleneck_stage(
            bottleneck_stage_id,
            bottleneck_stage_schedule_heuristic_option,
            draw_gantt=False,
        )
        if draw_gantt:
            self.draw_gantt(schedule)
        complete_makespan = schedule.makespan
        logging.info(f"Bottleneck parallel MC: full_schedule_obj={complete_makespan}")

        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=complete_makespan,
            obj_bound=None,
            is_init=True,
        )
        self.solution_manager.register(report, schedule)

    def bn2d_all_stage(
        self,
        left_cap_multiplier: int | None = None,
        right_cap_multiplier: int | None = None,
        left_cap_portion: float | None = None,
        right_cap_portion: float | None = None,
        normalize_by_stage_cnt: bool = False,
        reverse_mid_all: bool = False,
        reverse_mid_even: bool = False,
        randomize_mid_all: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Schedule from all stages as bottleneck and select best solution."""
        sub_timer = ElapsedTimer()

        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None

        for bottleneck_stage_id in self.instance.stage_id_list:
            logging.info(f"Trying bottleneck stage: {bottleneck_stage_id}")
            bottleneck_stage_schedule_heuristic_option = (
                BottleneckStageScheduleHeuristicOption(
                    left_cap_multiplier=left_cap_multiplier,
                    right_cap_multiplier=right_cap_multiplier,
                    left_cap_portion=left_cap_portion,
                    right_cap_portion=right_cap_portion,
                    normalize_by_stage_cnt=normalize_by_stage_cnt,
                    reverse_mid_all=reverse_mid_all,
                    reverse_mid_even=reverse_mid_even,
                    randomize_mid_all=randomize_mid_all,
                )
            )
            schedule = self._schedule_from_bottleneck_stage(
                bottleneck_stage_id,
                bottleneck_stage_schedule_heuristic_option,
                draw_gantt=False,
            )
            makespan = schedule.makespan
            logging.info(f"Bottleneck stage {bottleneck_stage_id}: makespan={makespan}")

            if best_obj is None or makespan < best_obj:
                best_obj = makespan
                best_sch = schedule
                logging.info("  -> New best solution found!")

        if draw_gantt:
            self.draw_gantt(best_sch)
        logging.info(f"Bottleneck MC v4: best_makespan={best_obj}")

        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
        )
        self.solution_manager.register(report, best_sch)

    def _get_bottleneck_stage(self) -> str:
        stage_id_2_total_p = {}
        for stage_id in self.instance.stage_id_list:
            stage_id_2_total_p[stage_id] = sum(
                self.stage_2_job_2_p_dict[stage_id][job_id]
                for job_id in self.instance.job_id_list
            )
        stage_id_2_bottleneck_index = {
            stage_id: total_p / len(self.instance.stage_2_machines_map[stage_id])
            for stage_id, total_p in stage_id_2_total_p.items()
        }
        bottleneck_stage_id = max(
            stage_id_2_bottleneck_index, key=stage_id_2_bottleneck_index.get
        )
        return bottleneck_stage_id

    def _get_bottleneck_stage_schedule_heuristic(
        self,
        bottleneck_stage_id: str,
        option: BottleneckStageScheduleHeuristicOption,
        draw_gantt: bool = False,
    ) -> tuple[HybridFlowshopLiteSchedule, int]:
        # From hybrid flow shop problem define parallel machine scheduling problem for the bottleneck stage
        bottleneck_stage_index = self.instance.stage_id_list.index(bottleneck_stage_id)
        before_stage_id_list = self.instance.stage_id_list[:bottleneck_stage_index]
        after_stage_id_list = self.instance.stage_id_list[bottleneck_stage_index + 1 :]

        # logging.info("Before stages: %s", before_stage_id_list)
        # logging.info("After stages: %s", after_stage_id_list)

        r_dict: dict[str, int] = {
            j: sum(self.job_2_stage_2_p_dict[j][s] for s in before_stage_id_list)
            for j in self.instance.job_id_list
        }
        if option.normalize_by_stage_cnt and len(before_stage_id_list) > 0:
            # Divide r_dict values by the number of before stage IDs
            r_dict = {
                j: 1 + (r // len(before_stage_id_list)) for j, r in r_dict.items()
            }
        p_dict: dict[str, int] = self.stage_2_job_2_p_dict[bottleneck_stage_id]
        tr_dict: dict[str, int] = {
            j: sum(self.job_2_stage_2_p_dict[j][s] for s in after_stage_id_list)
            for j in self.instance.job_id_list
        }
        if option.normalize_by_stage_cnt and len(after_stage_id_list) > 0:
            tr_dict = {
                j: 1 + (tr // len(after_stage_id_list)) for j, tr in tr_dict.items()
            }
        # pprint(r_dict)
        # pprint(p_dict)
        # pprint(tr_dict)
        machine_cnt = len(self.instance.stage_2_machines_map[bottleneck_stage_id])
        job_cnt = self.instance.job_count

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

        left_cap_job_id_list: list[str]
        right_cap_job_id_list: list[str]
        if left_cap_op_cnt > 0 or right_cap_op_cnt > 0:
            # Use CP solver to optimally select head and tail jobs
            result = solve_selection_problem(
                jobs=self.instance.job_id_list,
                r=r_dict,
                t=tr_dict,
                K_L=left_cap_op_cnt,
                K_R=right_cap_op_cnt,
            )
            # from pprint import pformat

            # logging.info(pformat(result))

            if result["status"] in ("OPTIMAL", "FEASIBLE"):
                left_cap_job_id_list = result["L_set"]
                # Sort by r_j in ascending order
                left_cap_job_id_list.sort(key=lambda j: r_dict[j])
                right_cap_job_id_list = result["R_set"]
                # Sort by tr_j in descending order
                right_cap_job_id_list.sort(key=lambda j: tr_dict[j], reverse=True)
            else:
                # Fallback to greedy selection if CP solver does not return a solution
                if left_cap_op_cnt > 0:
                    sorted_by_r = sorted(r_dict.items(), key=lambda x: x[1])
                    left_cap_job_id_list = [j for j, _ in sorted_by_r[:left_cap_op_cnt]]
                else:
                    left_cap_job_id_list = []

                if right_cap_op_cnt > 0:
                    sorted_by_tr = sorted(tr_dict.items(), key=lambda x: x[1])
                    # Exclude those in head_job_id_list
                    sorted_by_tr = [
                        (j, t) for j, t in sorted_by_tr if j not in left_cap_job_id_list
                    ]
                    right_cap_job_id_list = [
                        j for j, _ in sorted_by_tr[:right_cap_op_cnt]
                    ]
                else:
                    right_cap_job_id_list = []
            for j in left_cap_job_id_list:
                logging.debug(
                    f"Left cap job {j}: r={r_dict[j]}, p={p_dict[j]}, tr={tr_dict[j]}"
                )
            for j in right_cap_job_id_list:
                logging.debug(
                    f"Right cap job {j}: r={r_dict[j]}, p={p_dict[j]}, tr={tr_dict[j]}"
                )
        else:
            left_cap_job_id_list = []
            right_cap_job_id_list = []

        # Update mid_job_id_list to only include jobs that are not in head or tail job lists
        mid_job_id_list = [
            j
            for j in self.instance.job_id_list
            if j not in left_cap_job_id_list and j not in right_cap_job_id_list
        ]
        if option.randomize_mid_all:
            random.shuffle(mid_job_id_list)
        else:
            # Sort mid jobs by (r_j - tr_j, tie-break by original job index)
            mid_job_id_list.sort(
                key=lambda j: (
                    r_dict[j] - tr_dict[j],
                    self.instance.job_id_list.index(j),
                )
            )
            if option.reverse_mid_even:
                reverse_even_positions(mid_job_id_list, in_place=True)
            elif option.reverse_mid_all:
                mid_job_id_list.reverse()

        sorted_j_list = left_cap_job_id_list + mid_job_id_list + right_cap_job_id_list

        dispatched_schedule = self.create_empty_schedule_from_ins()
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

        logging.info(f"Bottleneck parallel MC: partial_obj={makespan}")
        # Draw Gantt chart; force start time as zero and end time as makespan by CP
        if draw_gantt:
            self.draw_gantt(dispatched_schedule, force_start=0, force_end=makespan)
        return dispatched_schedule, makespan

    def _create_reversed_instance_for_former_stages(
        self,
        before_stage_list: list[str],
        bottleneck_stage_start_time_map: dict[str, int],
        bcmax: int,
    ) -> tuple[HybridFlowshopParameters, dict[str, int]]:
        """Create a reverse scheduling problem.

        Args:
            before_stage_list (list[str]): List of stage IDs for the former stages (i.e. stages before the bottleneck stage).
            bottleneck_stage_start_time_map (dict[str, int]): job ID -> start time at bottleneck stage
            bcmax (int): The makespan of the bottleneck stage schedule.

        Returns:
            tuple[HybridFlowshopParameters, dict[str, int]]:
                - HybridFlowshopParameters: An instance of HybridFlowshopParameters for the former stages.
                - dict[str, int]: A dictionary mapping job IDs to their release dates.
        """
        # Create a new instance for the former stages
        stage_list = reversed(before_stage_list)
        job_2_release = {}
        for j in self.instance.job_id_list:
            bottleneck_start_time = bottleneck_stage_start_time_map[j]
            job_2_release[j] = bcmax - bottleneck_start_time
        return HybridFlowshopParameters(
            name=self.instance.name,
            job_id_list=self.instance.job_id_list,
            stage_id_list=list(stage_list),
            stage_2_machines_map={
                stage_id: self.instance.stage_2_machines_map[stage_id]
                for stage_id in before_stage_list
            },
            p_manager=self.instance.p_manager,
        ), job_2_release

    def _dispatch_former_stages(
        self,
        instance_for_former_stages: HybridFlowshopParameters,
        job_2_release: dict[str, int],
    ) -> HybridFlowshopLiteSchedule:
        # list of jobs sorted by release time (ascending)
        sorted_j_list = sorted(
            instance_for_former_stages.job_id_list,
            key=lambda j: (
                job_2_release[j],
                instance_for_former_stages.job_id_list.index(j),
            ),
        )

        ds_schedule = self.create_empty_schedule_from_ins(instance_for_former_stages)
        for stage_id in instance_for_former_stages.stage_id_list:
            ds_schedule.dispatch_stage_by_jobs(
                stage_id,
                sorted_j_list,
                {j: self.job_2_stage_2_p_dict[j][stage_id] for j in sorted_j_list},
                job_2_release=job_2_release,
            )

        dj_schedule = self.create_empty_schedule_from_ins(instance_for_former_stages)
        for job_id in sorted_j_list:
            dj_schedule.dispatch_job_by_stages(
                job_id,
                {
                    stage_id: self.job_2_stage_2_p_dict[job_id][stage_id]
                    for stage_id in instance_for_former_stages.stage_id_list
                },
                release_t=job_2_release[job_id],
            )

        if ds_schedule.makespan < dj_schedule.makespan:
            return ds_schedule
        return dj_schedule

    # End subroutine definition


def reverse_even_positions(sequence: list[Any], in_place: bool = False) -> list[Any]:
    """
    Reverse only even positions (1-based), keeping odd positions fixed.
    For example, [A,B,C,D,E,F,G,H] -> [A,H,C,F,E,D,G,B]

    Args:
        sequence (list[Any]): The input sequence to be modified.
        in_place (bool): If True, modify the input sequence in place and return it.
            If False, return a new modified list. Defaults to False.

    Returns:
        list[Any]: The modified sequence with even positions reversed.
    """
    if in_place:
        result = sequence
    else:
        result = sequence.copy()
    even_position_elements = result[1::2]
    even_position_elements.reverse()
    result[1::2] = even_position_elements
    return result
