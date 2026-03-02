import logging
import math
import random
from typing import Callable, Mapping, Sequence

from routix import ElapsedTimer
from schore.parameters_examples import HybridFlowshopParameters

from hybridflowshop.controller.neh_cp import NehCpConstructor, NehCpResult
from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder
from hybridflowshop.dispatcher import (
    BN2DDispatcher,
    BN2DOption,
    JobDispatcher,
    MachineDispatcher,
    MixedDispatcher,
    StageDispatcher,
)
from hybridflowshop.dispatcher.utils import from_job_sequence_get_schedule_mixed
from hybridflowshop.report import HfsSubroutineReport
from hybridflowshop.schedule_lite import (
    HybridFlowshopLiteSchedule,
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
        profile_fix_by_machine: bool = False,
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
        BaseModelBuilder.add_stage_ops_precedence_constraints_after_dispatch_from_schedule(
            self.cp_model,
            self.params,
            self.vars,
            out_of_block_ops_sch,
            profile_fix_by_machine=profile_fix_by_machine,
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
        make_semi_active_after_cp: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_operation_block_operator(
                rho,
                seed_op_from_critical_block=seed_op_from_critical_block,
                profile_fix_by_machine=profile_fix_by_machine,
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
        self,
        rho: float,
        seed_op_from_critical_block: bool = False,
        profile_fix_by_machine: bool = False,
    ) -> None:
        """Apply the operation-block operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            seed_op_from_critical_block (bool, optional): If True, the seed operation is
                chosen from critical blocks of the incumbent solution. Defaults to False.
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                adjacency; otherwise apply stage-level time-based selection.
                Defaults to False.

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
            selected_ops, profile_fix_by_machine=profile_fix_by_machine
        )

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
        profile_fix_by_machine: bool = False,
        make_semi_active_after_cp: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_stage_operator(
                rho,
                seed_stage_from_non_singleton_cb=seed_stage_from_non_singleton_cb,
                profile_fix_by_machine=profile_fix_by_machine,
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
        self,
        rho: float,
        seed_stage_from_non_singleton_cb: bool = False,
        profile_fix_by_machine: bool = False,
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
            selected_ops, profile_fix_by_machine=profile_fix_by_machine
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
        make_semi_active_after_cp: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self._fix_profile_solve_reset(
            lambda: self.apply_job_block_operator(
                rho,
                seed_op_from_critical_block=seed_op_from_critical_block,
                profile_fix_by_machine=profile_fix_by_machine,
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
        self,
        rho: float,
        seed_op_from_critical_block: bool = False,
        profile_fix_by_machine: bool = False,
    ) -> None:
        """Apply the job-block operator to the current CP model.

        Args:
            rho (float): Fraction of total number of operations to include in the block.
            seed_op_from_critical_block (bool, optional): If True, the seed operation is
                chosen from critical blocks of the incumbent solution. Defaults to False.
            profile_fix_by_machine (bool, optional): If True, fix precedence by machine
                adjacency; otherwise apply stage-level time-based selection.
                Defaults to False.

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
            selected_ops, profile_fix_by_machine=profile_fix_by_machine
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
        dispatcher = StageDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_ds_palmer()
        if schedule is None:
            raise ValueError("No schedule found after applying DS(Palmer).")
        logging.info(f"Schedule by DS(Palmer): makespan={schedule.makespan}")
        return schedule

    def initialize_by_dm_cds(
        self,
        spt_on_last_stage: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """
        Uses CDS sequence & dispatches by stage - machine - job priority
        to initialize a schedule.

        Args:
            spt_on_last_stage: If True, use SPT (shortest processing time first) for the last stage.
                If False, use LPT (longest processing time first). Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of
                the solution. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the
                solution. Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dm_cds(spt_on_last_stage=spt_on_last_stage)
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

    def _get_schedule_by_dm_cds(
        self, spt_on_last_stage: bool = False
    ) -> HybridFlowshopLiteSchedule:
        dispatcher = MachineDispatcher(
            self.instance, spt_on_last_stage=spt_on_last_stage
        )
        schedule = dispatcher.get_schedule_by_cds()
        if schedule is None:
            raise ValueError("No schedule found after applying DM(CDS).")
        logging.info(f"Schedule by DM(CDS): makespan={schedule.makespan}")
        return schedule

    def initialize_by_dm_gupta(
        self,
        spt_on_last_stage: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """
        Uses Gupta sequence & dispatches by stage - machine - job priority
        to initialize a schedule.

        Args:
            spt_on_last_stage: If True, use SPT (shortest processing time first) for the last stage.
                If False, use LPT (longest processing time first). Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of
                the solution. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the
                solution. Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dm_gupta(spt_on_last_stage=spt_on_last_stage)
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

    def _get_schedule_by_dm_gupta(
        self, spt_on_last_stage: bool = False
    ) -> HybridFlowshopLiteSchedule:
        dispatcher = MachineDispatcher(
            self.instance, spt_on_last_stage=spt_on_last_stage
        )
        schedule = dispatcher.get_schedule_by_gupta()
        if schedule is None:
            raise ValueError("No schedule found after applying DM(Gupta).")
        logging.info(f"Schedule by DM(Gupta): makespan={schedule.makespan}")
        return schedule

    def initialize_by_dm_palmer(
        self,
        spt_on_last_stage: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """
        Uses Palmer sequence & dispatches by stage - machine - job priority
        to initialize a schedule.

        Args:
            spt_on_last_stage: If True, use SPT (shortest processing time first) for the last stage.
                If False, use LPT (longest processing time first). Defaults to False.
            error_if_infeasible (bool, optional): If True, checks the feasibility of
                the solution. Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the
                solution. Defaults to False.
        """
        sub_timer = ElapsedTimer()

        schedule = self._get_schedule_by_dm_palmer(spt_on_last_stage=spt_on_last_stage)
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

    def _get_schedule_by_dm_palmer(
        self, spt_on_last_stage: bool = False
    ) -> HybridFlowshopLiteSchedule:
        dispatcher = MachineDispatcher(
            self.instance, spt_on_last_stage=spt_on_last_stage
        )
        schedule = dispatcher.get_schedule_by_palmer()
        if schedule is None:
            raise ValueError("No schedule found after applying DM(Palmer).")
        logging.info(f"Schedule by DM(Palmer): makespan={schedule.makespan}")
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

    def _get_generate_from_dispatches(self) -> HybridFlowshopLiteSchedule:
        import heapq

        # Initial sequences
        job_sequences: set[tuple[str, ...]] = {
            tuple(self.get_gupta_sequence()),
            tuple(self.get_palmer_sequence()),
        }
        for k, stage_id in enumerate(self.instance.stage_id_list):
            job_sequences.add(tuple(self.get_cds_sequence(k + 1)))
            job_sequences.add(tuple(self.get_bnd_sequence(stage_id)))

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
        job_seq_by_1st_stage: bool = False,
        job_seq_by_bottleneck_stage: bool = False,
        preserved_head_job_portion: float = 0.0,
        max_time_per_add: float | None = None,
        cp_tl_nc_multiplier: float | None = None,
        cp_tl_c_multiplier: float | None = None,
        profile_fix_by_machine: bool = False,
        minimize_sum_ci_lex: bool = False,
        cp_tl_nc_multiplier_2nd_obj: float | None = None,
        cp_tl_c_multiplier_2nd_obj: float | None = None,
        minimize_sum_ci_lin: bool = False,
        make_semi_active_every_cp: bool = False,
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
            job_seq_by_1st_stage (bool, optional): If True, defines the job
                sequence according to incumbent schedule's first stage schedule.
            job_seq_by_bottleneck_stage (bool, optional): If True, defines the job
                sequence according to incumbent schedule's bottleck stage schedule.
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
            job_seq_by_1st_stage=job_seq_by_1st_stage,
            job_seq_by_bottleneck_stage=job_seq_by_bottleneck_stage,
            preserved_head_job_portion=preserved_head_job_portion,
            max_time_per_add=max_time_per_add,
            cp_tl_nc_multiplier=cp_tl_nc_multiplier,
            cp_tl_c_multiplier=cp_tl_c_multiplier,
            profile_fix_by_machine=profile_fix_by_machine,
            minimize_sum_ci_lex=minimize_sum_ci_lex,
            cp_tl_nc_multiplier_2nd_obj=cp_tl_nc_multiplier_2nd_obj,
            cp_tl_c_multiplier_2nd_obj=cp_tl_c_multiplier_2nd_obj,
            minimize_sum_ci_lin=minimize_sum_ci_lin,
            make_semi_active_every_cp=make_semi_active_every_cp,
            solver_thread_cnt=solver_thread_cnt,
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
        final_report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
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
        dispatcher = BN2DDispatcher(self.instance)

        schedule = dispatcher.get_schedule_by_bn2d_all_stages(
            option=option,
            gantt_draw_func=self.draw_gantt if draw_gantt else None,
        )

        if schedule is None:
            # Failed to find a solution
            return
        if error_if_infeasible:
            self.check_feasibility(schedule.get_jik_2_start_time_map())

        best_obj = schedule.makespan
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
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

    # Dispatch

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
            get_file_path_for_subroutine=self.get_file_path_for_subroutine
            if draw_gantt_per_step
            else None,
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

        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
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
    ) -> HybridFlowshopLiteSchedule | None:
        dispatcher = MixedDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_cds(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=self.get_file_path_for_subroutine
            if draw_gantt_per_step
            else None,
        )
        if schedule is not None:
            logging.info(
                f"Best of mixed schedule by CDS sequence: makespan={schedule.makespan}"
            )
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
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
    ) -> HybridFlowshopLiteSchedule | None:
        dispatcher = MixedDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_gupta(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=self.get_file_path_for_subroutine
            if draw_gantt_per_step
            else None,
        )
        if schedule is not None:
            logging.info(
                f"Best of mixed schedule by Gupta sequence: makespan={schedule.makespan}"
            )
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
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
    ) -> HybridFlowshopLiteSchedule | None:
        dispatcher = MixedDispatcher(self.instance)
        schedule = dispatcher.get_schedule_by_palmer(
            machine_then_job=machine_then_job,
            head_for_all_stages=head_for_all_stages,
            use_palmer_index=use_palmer_index,
            draw_gantt_per_step=draw_gantt_per_step,
            get_file_path_for_subroutine=self.get_file_path_for_subroutine
            if draw_gantt_per_step
            else None,
        )
        if schedule is not None:
            logging.info(
                f"Best of mixed schedule by Palmer sequence: makespan={schedule.makespan}"
            )
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
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
    ) -> HybridFlowshopLiteSchedule | None:
        schedule_gen_methods = [
            self._get_schedule_by_cds,
            self._get_schedule_by_gupta,
            self._get_schedule_by_palmer,
        ]

        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None
        best_method_name = ""

        for method in schedule_gen_methods:
            logging.info(f"Generating schedule using {method.__name__}")
            sch = method(
                machine_then_job=machine_then_job,
                head_for_all_stages=head_for_all_stages,
                use_palmer_index=use_palmer_index,
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
                f"Best schedule generated by {best_method_name} with makespan {best_obj}"
            )
        return best_sch

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

        best_method_name = ""
        best_sch: HybridFlowshopLiteSchedule | None = None
        best_obj: int | None = None

        if not method_list:
            method_list = ["bn2d_all_stages", "best_of_mixed_dispatches"]
        for method_name in method_list:
            if method_name == "bn2d_all_stages":
                dispatcher = BN2DDispatcher(self.instance)
                sch = dispatcher.get_schedule_by_bn2d_all_stages(
                    option=option,
                    gantt_draw_func=self.draw_gantt if draw_gantt else None,
                )
                obj = sch.makespan if sch is not None else None
                logging.info(f"{method_name}: makespan={obj}")
                if obj is not None and (best_obj is None or obj < best_obj):
                    best_sch = sch
                    best_obj = obj
                    best_method_name = method_name
            elif method_name == "best_of_mixed_dispatches":
                sch = self._get_schedule_by_best_of_mixed_dispatches(
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                )
                obj = sch.makespan if sch is not None else None
                logging.info(f"{method_name}: makespan={obj}")
                if obj is not None and (best_obj is None or obj < best_obj):
                    best_sch = sch
                    best_obj = obj
                    best_method_name = method_name
            elif method_name == "stage_agg_2":
                sch = self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
                    stage_agg_count=2,
                    head_stages_to_keep=0,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    draw_gantt_stage_aggregated=False,
                )
                obj = sch.makespan if sch is not None else None
                logging.info(f"{method_name}: makespan={obj}")
                if obj is not None and (best_obj is None or obj < best_obj):
                    best_sch = sch
                    best_obj = obj
                    best_method_name = method_name
            elif method_name == "stage_agg_2_1":
                sch = self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
                    stage_agg_count=2,
                    head_stages_to_keep=1,
                    tail_stages_to_keep=1,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    draw_gantt_stage_aggregated=False,
                )
                obj = sch.makespan if sch is not None else None
                logging.info(f"{method_name}: makespan={obj}")
                if obj is not None and (best_obj is None or obj < best_obj):
                    best_sch = sch
                    best_obj = obj
                    best_method_name = method_name
            elif method_name == "stage_agg_2_2":
                sch = self._get_schedule_by_job_sequence_from_stage_aggregated_problem(
                    stage_agg_count=2,
                    head_stages_to_keep=2,
                    tail_stages_to_keep=2,
                    p_agg_method=p_agg_method,
                    mi_agg_method=mi_agg_method,
                    machine_then_job=machine_then_job,
                    head_for_all_stages=head_for_all_stages,
                    draw_gantt_stage_aggregated=False,
                )
                obj = sch.makespan if sch is not None else None
                logging.info(f"{method_name}: makespan={obj}")
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=best_obj,
            obj_bound=None,
            is_init=True,
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
            draw_gantt_stage_aggregated=draw_gantt_stage_aggregated,
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
        draw_gantt_stage_aggregated: bool = False,
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
            draw_gantt=draw_gantt_stage_aggregated,
        )
        if stage_aggregated_schedule is None:
            return None

        job_sequences = [
            get_bottleneck_stage_job_sequence(stage_aggregated_schedule),
            get_midpoint_sequence(stage_aggregated_schedule),
        ]

        best_obj: int | None = None
        best_sch: HybridFlowshopLiteSchedule | None = None

        dispatcher = MixedDispatcher(self.instance)
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
        dispatcher = MixedDispatcher(stage_aggregated_instance)
        schedule = dispatcher.get_schedule_by_cds(
            head_for_all_stages=head_for_all_stages,
            draw_gantt_per_step=draw_gantt_per_step,
        )
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
        draw_gantt_stage_aggregated: bool = False,
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
                    draw_gantt_stage_aggregated=draw_gantt_stage_aggregated,
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
        report = HfsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
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
