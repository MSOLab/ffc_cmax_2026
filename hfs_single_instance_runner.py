from pathlib import Path
from typing import Any

from routix import DynamicDataObject, StoppingCriteria
from routix.io import object_to_yaml
from routix.runner import SingleInstanceRunner
from routix.type_defs import RunMode
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.hfs_input_summary import HfsInputSummary
from hybridflowshop.hfs_summary import HfsSummary
from hybridflowshop.report.hfs_subroutine_report_statistics import (
    HfsSubroutineReportStatistics,
)
from hybridflowshop.utils import tuple_to_pyyaml_key


class HfsSingleInstanceRunner(
    SingleInstanceRunner[HybridFlowshopParameters, HybridFlowShopCpLnsController]
):
    def __init__(
        self,
        instance: HybridFlowshopParameters,
        shared_param_dict: dict,
        subroutine_flow: DynamicDataObject,
        stopping_criteria: StoppingCriteria,
        output_dir: Path,
        output_metadata: dict[str, Any],
        mode: RunMode = RunMode.FULL_RUN,
    ):
        super().__init__(
            instance=instance,
            shared_param_dict=shared_param_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
            output_dir=output_dir,
            output_metadata=output_metadata,
            mode=mode,
        )
        self.name = self.instance.name
        self.encoding = self.output_metadata.get("encoding", "utf-8")
        result_dir_name = self.output_metadata.get("result_dir_name", "results")
        self.result_dir = self.working_dir / result_dir_name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.prepare_saved_file_paths()

    def get_controller(self) -> HybridFlowShopCpLnsController:
        """Initialize the controller with the given instance and parameters."""
        return HybridFlowShopCpLnsController(
            self.instance,
            self.shared_param_dict,
            self.subroutine_flow,
            self.stopping_criteria,
        )

    def post_run_process(self) -> None:
        if self.mode == RunMode.FULL_RUN:
            self.save_files(self.encoding)

        self.from_files_save_analysis(self.encoding)

    def prepare_saved_file_paths(self) -> None:
        self.summary_filename = (
            str(self.output_metadata.get("summary_filename_format", "{}_summary.csv"))
            .strip()
            .format(self.name)
        )
        self.summary_path = self.result_dir / self.summary_filename

        self.solution_filename_format = str(
            self.output_metadata.get("solution_filename_format", "{}_solution.yaml")
        ).strip()
        self.solution_path = self.result_dir / self.solution_filename_format.format(
            self.name
        )

        self.obj_log_filename_format = str(
            self.output_metadata.get("obj_log_filename_format", "{}_obj_log.yaml")
        ).strip()
        self.obj_log_path = self.result_dir / self.obj_log_filename_format.format(
            self.name
        )

    def save_files(self, encoding: str = "utf-8") -> None:
        self.save_summary(encoding=encoding)
        self.save_solution(encoding=encoding)
        self.save_obj_value_bound_store(encoding=encoding)

    def save_summary(self, encoding: str = "utf-8") -> None:
        stats = HfsSubroutineReportStatistics(
            name=self.name,
            reports=[r.report for r in self.ctrlr.solution_manager.history],
            method_call_counts=self.ctrlr.method_call_counts,
        )
        summary = HfsSummary(
            inputs=HfsInputSummary(
                name=self.name,
                job_count=self.instance.job_count,
                stage_count=self.instance.stage_count,
                timelimit=self.stopping_criteria.timelimit,
            ),
            outputs=stats,
        )
        summary.save(self.summary_path, encoding=encoding)

    def save_solution(self, encoding: str = "utf-8") -> None:
        incumbent_solution = self.ctrlr.solution_manager.get_incumbent()
        if incumbent_solution:
            solution_dict = {
                "start_times": tuple_to_pyyaml_key(
                    incumbent_solution.get_start_time_map()
                ),
                "end_times": tuple_to_pyyaml_key(incumbent_solution.get_end_time_map()),
            }
            object_to_yaml(solution_dict, self.solution_path, encoding=encoding)

    def save_obj_value_bound_store(self, encoding: str = "utf-8") -> None:
        self.ctrlr.obj_store.save_yaml(self.obj_log_path, encoding=encoding)

    def from_files_save_analysis(self, encoding: str = "utf-8") -> None:
        if self.output_metadata.get("draw_gantt", False):
            self.from_files_draw_gantt_chart(encoding=encoding)
        if self.output_metadata.get("draw_progress_plot", False):
            self.from_files_draw_progress_plot(encoding=encoding)

    def from_files_draw_gantt_chart(self, encoding: str = "utf-8") -> None:
        """
        Draws Gantt charts from the saved solution files.
        This method looks for files matching the `solution_filename_format` in the working directory
        and generates Gantt charts based on the start and end times stored in the solution files.

        Args:
            encoding (str, optional): The encoding to use when reading files. Defaults to "utf-8".
        """
        result_gantt_filename_format = str(
            self.output_metadata.get("result_gantt_filename_format", "{}_gantt.png")
        ).strip()

        from concurrent_painter import draw_gantt_charts_from_solutions

        draw_gantt_charts_from_solutions(
            working_dir=self.working_dir,
            solution_filename_format=self.solution_filename_format,
            all_job_id_list=self.instance.job_id_list,
            result_gantt_filename_format=result_gantt_filename_format,
            encoding=encoding,
            painter_thread_cnt=self.output_metadata.get("painter_thread_cnt", 4),
        )

    def from_files_draw_progress_plot(self, encoding: str = "utf-8") -> None:
        """
        Draws a progress plot from the saved objective log files.
        This method looks for files matching the `obj_log_filename_format` in the working directory
        and generates a plot based on the objective value records stored in the log.

        Args:
            encoding (str, optional): The encoding to use when reading files. Defaults to "utf-8".
        """
        progress_plot_filename_format = str(
            self.output_metadata.get("progress_plot_filename_format", "{}_progress.png")
        ).strip()

        from concurrent_painter import draw_progress_plots_from_logs

        draw_progress_plots_from_logs(
            working_dir=self.working_dir,
            obj_log_filename_format=self.obj_log_filename_format,
            progress_plot_filename_format=progress_plot_filename_format,
            drop_first_values_percent=self.output_metadata.get(
                "drop_first_values_percent", 0.0
            ),
            encoding=encoding,
            painter_thread_cnt=self.output_metadata.get("painter_thread_cnt", 4),
        )
