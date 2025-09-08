import logging
from pathlib import Path
from typing import Optional, Sequence

from mbls.cpsat import (
    CpsatStatus,
    CpSubroutineController,
)
from routix import DynamicDataObject, StoppingCriteria
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from .cp_2023_naderi_cumulative import CP2023NaderiCumulative
from .painter.gantt import GanttPlotter
from .scheduling.hybrid_flowshop_schedule import HybridFlowshopSchedule
from .solution_manager import HfsSolutionManager


class HybridFlowShopCpLnsControllerCore(
    CpSubroutineController[
        HybridFlowshopParameters, CP2023NaderiCumulative, StoppingCriteria
    ]
):
    """
    Controller for solving Hybrid Flow Shop problems using CP-based algorithms.
    """

    # Start controller state
    solution_manager: HfsSolutionManager
    """Solution manager for Hybrid Flow Shop scheduling solutions."""
    total_elapsed_time: float
    """Total elapsed time for the controller."""
    # End controller state

    def __init__(
        self,
        instance: HybridFlowshopParameters,
        shared_param_dict: dict,
        subroutine_flow: DynamicDataObject,
        stopping_criteria: StoppingCriteria,
    ):
        super().__init__(
            instance,
            shared_param_dict,
            CP2023NaderiCumulative,
            subroutine_flow,
            stopping_criteria,
        )
        self.solution_manager = HfsSolutionManager()

        # Frequently used parameters
        self.job_2_stage_2_p_dict = self.instance.p_manager.job_2_stage_2_value_map(
            self.instance.job_id_list, self.instance.stage_id_list
        )
        """Job name -> stage name -> processing time map"""
        self.stage_2_job_2_p_dict = self.instance.p_manager.stage_2_job_2_value_map(
            self.instance.stage_id_list, self.instance.job_id_list
        )
        """Stage name -> job name -> processing time map"""

        logging.info(
            f"Start solving {self.instance.name} using CP model class:"
            f" {self.cp_model_class.__module__}.{self.cp_model_class.__name__}",
        )

    # Start abstract getters

    def create_base_cp_model(
        self, impose_all_stage_capacity_constr: bool = True, **kwargs
    ) -> CP2023NaderiCumulative:
        return self.cp_model_class.from_instance(
            self.instance,
            self.get_horizon(),
            impose_all_stage_capacity_constr=impose_all_stage_capacity_constr,
        )

    # End abstract getters

    def get_horizon(self) -> int:
        """Returns the horizon of the scheduling problem."""
        if "horizon" not in self.shared_param_dict:
            raise ValueError("Horizon not found in shared parameters.")
        return self.shared_param_dict["horizon"]

    def set_working_dir(self, dir_path: Path | str):
        super().set_working_dir(dir_path)
        self.log_handlers: list[logging.StreamHandler] = []
        self.add_file_handler()

    def add_file_handler(
        self,
        log_filename: Optional[str] = None,
        level=logging.INFO,
        fmt="%(asctime)s - %(levelname)s - %(message)s",
    ):
        logger = logging.getLogger()
        _log_filename = log_filename or "subroutine_controller.log"
        if self._working_dir_path is not None:
            log_path = self._working_dir_path / _log_filename
            # 이미 같은 파일 핸들러가 등록되어 있는지 확인 (중복 방지)
            for handler in logger.handlers:
                if isinstance(
                    handler, logging.FileHandler
                ) and handler.baseFilename == str(log_path):
                    return  # 이미 등록되어 있으면 추가하지 않음

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(file_handler)
            self.log_handlers = [file_handler]

    def release_log_handlers(self) -> None:
        logger = logging.getLogger()
        for handler in self.log_handlers:
            logger.removeHandler(handler)
            handler.close()

    # Start stopping condition

    def is_stopping_condition(self) -> bool:
        return self.time_is_up()

    def time_is_up(self) -> bool:
        # If total elapsed time exceeds the stopping criteria
        if self.timer.elapsed_sec >= self.stopping_criteria.timelimit:
            logging.info("Stop by timelimit")
            return True
        return False

    def get_remaining_sec(self) -> float:
        return self.timer.get_remaining_sec(self.stopping_criteria.timelimit)

    def get_remaining_time_limit(self, subroutine_time_limit: float | None) -> float:
        """Get the remaining time limit for the subroutine.

        Args:
            subroutine_time_limit (float | None, optional): The time limit for the subroutine in seconds.
                If None, the remaining time limit is used.

        Returns:
            float: The minimum of the subroutine time limit and the remaining time limit.
        """
        if subroutine_time_limit is None:
            return self.get_remaining_sec()
        return min(subroutine_time_limit, self.get_remaining_sec())

    # End stopping condition

    # Start visualization

    def draw_gantt(
        self, schedule: HybridFlowshopSchedule, output_path: Path | None = None
    ):
        """Draws the Gantt chart of the given schedule.

        Args:
            schedule (HybridFlowshopSchedule): The schedule to draw.
            output_path (Path | None, optional): The output path for the Gantt chart image. Defaults to None.
        """
        if output_path is None:
            output_path = self.get_file_path_for_subroutine("_gantt.png")
        if isinstance(schedule, HybridFlowshopSchedule):
            plotter = GanttPlotter()
            plotter.export_hybrid_flowshop_plot(
                output_path,
                schedule.get_start_time_map(),
                schedule.get_end_time_map(),
                self.instance.job_id_list,
            )

    def draw_incumbent_gantt(self, output_path: Path | None = None) -> None:
        """Draws the Gantt chart of the incumbent solution.

        Args:
            output_path (Path | None, optional): The output path for the Gantt chart image. Defaults to None.
        """
        incumbent_solution = self.solution_manager.get_incumbent()
        if isinstance(incumbent_solution, HybridFlowshopSchedule):
            self.draw_gantt(incumbent_solution, output_path=output_path)
        else:
            logging.warning("No incumbent solution available to draw Gantt chart.")

    # End visualization

    def run(self, flow_resume_idx: int = -1) -> None:
        """Overrides the run method to execute the subroutine flow.

        Args:
            flow_resume_idx (int, optional): The index to resume the flow from. Defaults to -1.
        """
        if isinstance(self._subroutine_flow, Sequence) and not isinstance(
            self._subroutine_flow, (str, bytes)
        ):
            for idx, subroutine_data in enumerate(self._subroutine_flow):
                skip_method_call = idx < flow_resume_idx
                self._run_flow(subroutine_data, skip_method_call=skip_method_call)
        self.post_run_process()

    # Start post-run process

    def post_run_process(self) -> None:
        """
        Finalizes the run by checking the feasibility of the incumbent solution
        and releasing log handlers.
        """
        incumbent = self.solution_manager.get_incumbent()
        if incumbent:
            self.check_feasibility(incumbent.get_start_time_map())
        self.release_log_handlers()
        self.total_elapsed_time = self.timer.elapsed_sec

    def check_feasibility(
        self, start_time_map: dict[tuple[str, str, str], int]
    ) -> float:
        """Check the feasibility of the given start times.

        Args:
            start_time_map (dict[tuple[str, str, str], int]): A mapping of (job, stage, machine) to start time.

        Raises:
            ValueError: If any start time is negative or invalid.
            RuntimeError: If the feasibility check fails while solving the model.
            ValueError: If the feasibility check fails with an unexpected status.

        Returns:
            float: The objective value of the solution if feasible.
        """
        logging.info("Feasibility check starts")
        for (j, i, k), start_time in start_time_map.items():
            if start_time < 0:
                raise ValueError(
                    f"Invalid start time for job {j}, stage {i}, machine {k}: {start_time}"
                )
        base_cp = self.create_base_cp_model()

        # Freeze operation start times and machine assignments
        for (j, i, k), start_time in start_time_map.items():
            base_cp.add(self.cp_model.var_op_start[j, i] == start_time)

        # Solve with tight time limit
        timelimit = 2.0
        solver_thread_cnt = 1
        solver_report = self.solve_cp_model(base_cp, timelimit, solver_thread_cnt)
        if solver_report.status not in (CpsatStatus.FEASIBLE, CpsatStatus.OPTIMAL):
            mdl_txt_path = self.get_file_path_for_subroutine(
                "_feasibility_check_failed.txt"
            )
            base_cp.export_to_file(str(mdl_txt_path))
            if solver_report.status == CpsatStatus.INFEASIBLE:
                raise RuntimeError(
                    f"Feasibility check failed: INFEASIBLE. Model saved to {mdl_txt_path}"
                )
            else:
                raise ValueError(
                    f"Feasibility check failed with status {solver_report.status}. "
                    f"Model saved to {mdl_txt_path}"
                )
        logging.info("Feasibility check passed")
        if solver_report.obj_value is None:
            raise ValueError("Feasibility check did not return an objective value.")
        return solver_report.obj_value

    # End post-run process
