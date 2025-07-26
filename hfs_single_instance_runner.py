from pathlib import Path
from typing import Any

import yaml
from mbls.cpsat import ObjValueBoundStore
from mbls.painter import ObjValueBoundPlotter
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
from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.report.hfs_subroutine_report_statistics import (
    HfsSubroutineReportStatistics,
)
from hybridflowshop.utils import pyyaml_key_to_tuple, tuple_to_pyyaml_key


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

        self.solution_filename = (
            str(
                self.output_metadata.get("solution_filename_format", "{}_solution.yaml")
            )
            .strip()
            .format(self.name)
        )
        self.solution_path = self.result_dir / self.solution_filename

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
        result_gantt_filename = (
            str(
                self.output_metadata.get("result_gantt_filename_format", "{}_gantt.png")
            )
            .strip()
            .format(self.name)
        )
        output_path = self.result_dir / result_gantt_filename

        with open(self.solution_path, "r", encoding=encoding) as f:
            solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
            start_time_map = pyyaml_key_to_tuple(solution_dict["start_times"])
            end_time_map = pyyaml_key_to_tuple(solution_dict["end_times"])
            GanttPlotter().export_hybrid_flowshop_plot(
                output_path, start_time_map, end_time_map
            )

    def from_files_draw_progress_plot(self, encoding: str = "utf-8") -> None:
        """
        Draws a progress plot from the saved solution files.
        This method looks for files matching the `obj_log_filename_format` in the working directory
        and generates a plot based on the objective value records stored in the log.

        Args:
            encoding (str, optional): The encoding to use when reading files. Defaults to "utf-8".
        """
        progress_plot_filename_format = str(
            self.output_metadata.get("progress_plot_filename_format", "{}_progress.png")
        ).strip()

        # Find all files that match the progress plot filename format in working_dir
        # Including all subdirectories
        for file in self.working_dir.rglob(self.obj_log_filename_format.format("*")):
            # Define the file's directory, filename prefix, and output path
            file_dir = file.parent
            filename_prefix = extract_brace_key_from_filename(
                self.obj_log_filename_format, file.name
            )
            if filename_prefix is None:
                raise ValueError(
                    f"Could not extract filename prefix from {file.name}"
                    f" using pattern {self.obj_log_filename_format}"
                )
            output_path = file_dir / progress_plot_filename_format.format(
                filename_prefix
            )

            drop_first_values_percent = self.output_metadata.get(
                "drop_first_values_percent", 0.0
            )

            obj_store = ObjValueBoundStore.load_yaml(file, encoding=encoding)
            ObjValueBoundPlotter.plot(
                obj_store,
                output_path,
                drop_first_values_percent=drop_first_values_percent,
                label_y_offset=2.0,
                legend_loc="lower right",
            )


def extract_brace_key_from_filename(pattern: str, filename: str) -> str | None:
    """
    pattern: e.g. '{}_obj_log.yaml'
    filename: e.g. '10-initialize_by_cjims_obj_log.yaml'
    Returns the text that fills the {} in the pattern, or None if not matched.
    """
    import re

    # Escape special regex chars except {}
    regex = re.escape(pattern).replace(r"\{\}", "(.+?)")
    match = re.match(regex, filename)
    if match:
        return match.group(1)
    return None
