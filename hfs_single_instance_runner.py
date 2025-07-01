import logging
from pathlib import Path
from typing import Any

import yaml
from mbls import DynamicDataObject, utils
from mbls.cpsat import ObjValueBoundStore
from mbls.painter import ObjValueBoundPlotter
from routix.runner import SingleInstanceRunner
from schore.examples.hybrid_flowshop import HybridFlowshopParameters

from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.hfs_input_summary import HfsInputSummary
from hybridflowshop.hfs_summary import HfsSummary
from hybridflowshop.painter.gantt import GanttPlotter
from hybridflowshop.stopping_criteria import StoppingCriteria
from hybridflowshop.utils import pyyaml_key_to_tuple


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
    ):
        super().__init__(
            instance=instance,
            shared_param_dict=shared_param_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
            output_dir=output_dir,
            output_metadata=output_metadata,
        )

    def init_controller(self) -> HybridFlowShopCpLnsController:
        """Initialize the controller with the given instance and parameters."""
        return HybridFlowShopCpLnsController(
            self.instance,
            self.shared_param_dict,
            self.subroutine_flow,
            self.stopping_criteria,
        )

    def post_run_process(self) -> None:
        self.name = self.instance.name
        result_dir_name = self.output_metadata.get("result_dir_name", "results")
        self.result_dir = self.working_dir / result_dir_name

        self.prepare_saved_file_paths()
        if not self.output_metadata.get(
            "single_instance_from_files_save_analysis_only", False
        ):
            self.save_files(self.encoding)
        self.from_files_save_analysis(self.encoding)

    def prepare_saved_file_paths(self) -> None:
        self.encoding = self.output_metadata.get("encoding", "utf-8")

        self.summary_filename = self.name + "_summary.csv"
        if "summary_filename_format" in self.output_metadata:
            summary_filename_format = self.output_metadata["summary_filename_format"]
            if isinstance(summary_filename_format, str):
                summary_filename_format = summary_filename_format.strip()
                self.summary_filename = summary_filename_format.format(self.name)
        self.summary_path = self.result_dir / self.summary_filename

        self.solution_filename = self.name + "_solution.yaml"
        if "solution_filename_format" in self.output_metadata:
            solution_filename_format = self.output_metadata["solution_filename_format"]
            if isinstance(solution_filename_format, str):
                solution_filename_format = solution_filename_format.strip()
                self.solution_filename = solution_filename_format.format(self.name)
        self.solution_path = self.result_dir / self.solution_filename

        self.obj_log_filename = self.name + "_obj_log.yaml"
        if "obj_log_filename_format" in self.output_metadata:
            obj_log_filename_format = self.output_metadata["obj_log_filename_format"]
            if isinstance(obj_log_filename_format, str):
                obj_log_filename_format = obj_log_filename_format.strip()
                self.obj_log_filename = obj_log_filename_format.format(self.name)
        self.obj_log_path = self.result_dir / self.obj_log_filename

    def save_files(self, encoding: str = "utf-8") -> None:
        """
        Save the files generated during the run.
        This method is called after the run is complete.

        Args:
            encoding (str, optional): The encoding to use when saving files. Defaults to "utf-8".
        """
        self.save_summary(encoding=encoding)
        self.save_solution(encoding=encoding)
        self.save_obj_value_bound_store(encoding=encoding)

    def save_summary(self, encoding: str = "utf-8") -> None:
        input_summary = HfsInputSummary(
            name=self.name,
            job_count=self.instance.job_count,
            stage_count=self.instance.stage_count,
            timelimit=self.stopping_criteria.timelimit,
        )
        expr_summary = self.ctrlr.experiment_summary
        summary = HfsSummary(inputs=input_summary, outputs=expr_summary)
        summary.save(self.summary_path, encoding=encoding)

    def save_solution(self, encoding: str = "utf-8") -> None:
        solution = self.ctrlr.get_incumbent_solution_dict(for_pyyaml=True)
        utils.object_to_yaml(solution, self.solution_path, encoding=encoding)

    def save_obj_value_bound_store(self, encoding: str = "utf-8") -> None:
        self.ctrlr.obj_store.save_yaml(self.obj_log_path, encoding=encoding)

    def from_files_save_analysis(self, encoding: str = "utf-8") -> None:
        self.from_files_draw_gantt_chart(encoding=encoding)
        self.from_files_draw_progress_plot(encoding=encoding)

    def from_files_draw_gantt_chart(self, encoding: str = "utf-8") -> None:
        # Prepare the Gantt chart file path
        result_gantt_filename_format = "{}_result_gantt.png"
        if "result_gantt_filename_format" in self.output_metadata:
            result_gantt_filename_format = self.output_metadata[
                "result_gantt_filename_format"
            ]
            if isinstance(result_gantt_filename_format, str):
                result_gantt_filename_format = result_gantt_filename_format.strip()
        result_gantt_filename = result_gantt_filename_format.format(self.name)
        output_path = self.result_dir / result_gantt_filename

        # Read saved solution file to create dictionary of start and end times
        with open(self.solution_path, "r", encoding=encoding) as f:
            solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
            # TODO: backward compatibility; change to "start_time_map" in future versions
            start_time_map = pyyaml_key_to_tuple(solution_dict["start_times"])
            # TODO: backward compatibility; change to "end_time_map" in future versions
            end_time_map = pyyaml_key_to_tuple(solution_dict["end_times"])

            GanttPlotter().export_hybrid_flowshop_plot(
                output_path, start_time_map, end_time_map
            )

    def from_files_draw_progress_plot(self, encoding: str = "utf-8") -> None:
        """
        Read the saved obj_log file and draw the progress plot.
        """

        progress_plot_filename_format = "{}_progress_plot.png"
        if "progress_plot_filename_format" in self.output_metadata:
            progress_plot_filename_format = self.output_metadata[
                "progress_plot_filename_format"
            ]
            if isinstance(progress_plot_filename_format, str):
                progress_plot_filename_format = progress_plot_filename_format.strip()
            else:
                logging.warning(
                    "Invalid type for 'progress_plot_filename_format': "
                    f"{type(progress_plot_filename_format)}. Using default format."
                )
                progress_plot_filename_format = "{}_progress_plot.png"

        drop_first_values_percent = 0.0
        if "drop_first_values_percent" in self.output_metadata:
            drop_first_values_percent = self.output_metadata[
                "drop_first_values_percent"
            ]
            if isinstance(drop_first_values_percent, (int, float)):
                drop_first_values_percent = float(drop_first_values_percent)
            else:
                logging.warning(
                    "Invalid type for 'drop_first_values_percent': "
                    f"{type(drop_first_values_percent)}. Using default value of 0.0."
                )
                drop_first_values_percent = 0.0

        # Prepare the progress plot file path
        progress_plot_filename = progress_plot_filename_format.format(self.name)
        output_path = self.result_dir / progress_plot_filename

        # Read the saved obj_log file
        obj_store = ObjValueBoundStore.load_yaml(self.obj_log_path, encoding=encoding)
        # Plot the objective progress
        ObjValueBoundPlotter.plot(
            obj_store, output_path, drop_first_values_percent=drop_first_values_percent
        )
