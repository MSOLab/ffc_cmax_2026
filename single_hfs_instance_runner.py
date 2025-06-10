import csv
from pathlib import Path
from typing import Any

import yaml
from mbls import DynamicDataObject, utils
from routix.runner import SingleInstanceRunner
from schore.hybridflowshop import HybridFlowShopProblem

from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.hfs_input_summary import HFSInputSummary
from hybridflowshop.hfs_summary import HFSSummary
from hybridflowshop.plotter.gantt import GanttPlotter
from hybridflowshop.stopping_criteria import StoppingCriteria
from hybridflowshop.utils import pyyaml_key_to_tuple


# class SingleHFSInstanceRunner(SingleInstanceRunner):
#     instance: HybridFlowShopProblem
#     ctrlr: HybridFlowShopCpLnsController
class SingleHFSInstanceRunner(
    SingleInstanceRunner[HybridFlowShopProblem, HybridFlowShopCpLnsController]
):
    def __init__(
        self,
        instance: HybridFlowShopProblem,
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
            self.save_files()
        self.from_files_save_analysis()

    def prepare_saved_file_paths(self) -> None:
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

        self.obj_log_filename = self.name + "_obj_log.csv"
        if "obj_log_filename_format" in self.output_metadata:
            obj_log_filename_format = self.output_metadata["obj_log_filename_format"]
            if isinstance(obj_log_filename_format, str):
                obj_log_filename_format = obj_log_filename_format.strip()
                self.obj_log_filename = obj_log_filename_format.format(self.name)
        self.obj_log_path = self.result_dir / self.obj_log_filename

    def save_files(self) -> None:
        """
        Save the files generated during the run.
        This method is called after the run is complete.
        """
        self.save_summary()
        self.save_solution()
        self.save_obj_log()

    def save_summary(self) -> None:
        input_summary = HFSInputSummary(
            name=self.name,
            job_count=self.instance.job_count,
            stage_count=self.instance.stage_count,
            timelimit=self.stopping_criteria.timelimit,
        )
        expr_summary = self.ctrlr.experiment_summary
        summary = HFSSummary(inputs=input_summary, outputs=expr_summary)
        summary.save(self.summary_path)

    def save_solution(self) -> None:
        solution = self.ctrlr.get_incumbent_solution_dict(for_pyyaml=True)
        utils.object_to_yaml(solution, self.solution_path)

    def save_obj_log(self) -> None:
        """
        Save the objective log to a CSV file.
        Each row: time,obj_value,obj_bound
        """
        obj_value_log = getattr(self.ctrlr, "obj_value_log", [])
        obj_bound_log = getattr(self.ctrlr, "obj_bound_log", [])

        unique_times = sorted(set(t for t, _ in obj_value_log + obj_bound_log))
        obj_value_dict = {t: v for t, v in obj_value_log}
        obj_bound_dict = {t: v for t, v in obj_bound_log}

        with open(self.obj_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "obj_value", "obj_bound"])
            for t in unique_times:
                value = obj_value_dict.get(t, None)
                bound = obj_bound_dict.get(t, None)
                writer.writerow([t, value, bound])

    def from_files_save_analysis(self) -> None:
        self.from_files_draw_gantt_chart()
        self.from_files_draw_progress_plot()

    def from_files_draw_gantt_chart(self) -> None:
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
        with open(self.solution_path, "r", encoding="utf-8") as f:
            solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
            start_times = pyyaml_key_to_tuple(solution_dict["start_times"])
            end_times = pyyaml_key_to_tuple(solution_dict["end_times"])

            GanttPlotter().export_hybrid_flowshop_plot(
                output_path, start_times, end_times
            )

    def from_files_draw_progress_plot(self) -> None:
        """
        Read the saved obj_log file and draw the progress plot.
        """
        from hybridflowshop.plotter.objective_progress import ObjectiveProgressPlotter

        # Prepare the progress plot file path
        progress_plot_filename_format = "{}_progress_plot.png"
        if "progress_plot_filename_format" in self.output_metadata:
            progress_plot_filename_format = self.output_metadata[
                "progress_plot_filename_format"
            ]
            if isinstance(progress_plot_filename_format, str):
                progress_plot_filename_format = progress_plot_filename_format.strip()

        progress_plot_filename = progress_plot_filename_format.format(self.name)
        output_path = self.result_dir / progress_plot_filename

        # Read the saved obj_log file
        obj_value_log: list[tuple[float, float]] = []
        obj_bound_log: list[tuple[float, float]] = []
        with open(self.obj_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row["time"])
                obj_val = row["obj_value"]
                obj_bound = row["obj_bound"]
                # None 값이 아닌 경우만 추가
                if obj_val not in (None, "", "None"):
                    obj_value_log.append((t, float(obj_val)))
                if obj_bound not in (None, "", "None"):
                    obj_bound_log.append((t, float(obj_bound)))

        # Plot the objective progress
        ObjectiveProgressPlotter.plot_lists_of_time_and_obj(
            [obj_value_log, obj_bound_log],
            output_path,
            labels=["ObjVal", "LBound"],
        )
