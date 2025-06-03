from pathlib import Path
from typing import Any

from mbls import DynamicDataObject, utils
from schore.hybridflowshop import HybridFlowShopProblem

from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.hfs_input_summary import HFSInputSummary
from hybridflowshop.hfs_summary import HFSSummary
from hybridflowshop.stopping_criteria import StoppingCriteria
from single_instance_runner import SingleInstanceRunner


# class SingleHFSInstanceSolver(SingleInstanceSolver):
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

        # controller_init_kwargs
        horizon = self.shared_param_dict["horizon"]

        return HybridFlowShopCpLnsController(
            self.instance,
            self.subroutine_flow,
            self.stopping_criteria,
            horizon,
        )

    def post_run_process(self) -> None:
        self.name = self.instance.name
        result_dir_name = self.output_metadata.get("result_dir_name", "results")
        self.result_dir = self.working_dir / result_dir_name

        self.save_summary()
        self.save_solution()
        if "draw_gantt" in self.output_metadata and self.output_metadata["draw_gantt"]:
            self.save_gantt_chart()
        if (
            "draw_progress_plot" in self.output_metadata
            and self.output_metadata["draw_progress_plot"]
        ):
            self.save_progress_plot()

    def save_summary(self) -> None:
        input_summary = HFSInputSummary(
            name=self.name,
            job_count=self.instance.job_count,
            stage_count=self.instance.stage_count,
            timelimit=self.stopping_criteria.timelimit,
        )
        expr_summary = self.ctrlr.get_experiment_summary()
        summary = HFSSummary(inputs=input_summary, outputs=expr_summary)

        summary_filename = self.name + "_summary.csv"
        if "summary_filename_format" in self.output_metadata:
            summary_filename_format = self.output_metadata["summary_filename_format"]
            if isinstance(summary_filename_format, str):
                summary_filename_format = summary_filename_format.strip()
                summary_filename = summary_filename_format.format(self.name)

        summary.save(self.result_dir / summary_filename)

    def save_solution(self) -> None:
        solution = self.ctrlr.get_incumbent_solution_dict(for_pyyaml=True)

        solution_filename = self.name + "_solution.yaml"
        if "solution_filename_format" in self.output_metadata:
            solution_filename_format = self.output_metadata["solution_filename_format"]
            if isinstance(solution_filename_format, str):
                solution_filename_format = solution_filename_format.strip()
                solution_filename = solution_filename_format.format(self.name)

        utils.object_to_yaml(solution, self.result_dir / solution_filename)

    def save_gantt_chart(self) -> None:
        result_gantt_filename_format = "{}_result_gantt.png"
        if "result_gantt_filename_format" in self.output_metadata:
            result_gantt_filename_format = self.output_metadata[
                "result_gantt_filename_format"
            ]
            if isinstance(result_gantt_filename_format, str):
                result_gantt_filename_format = result_gantt_filename_format.strip()

        result_gantt_filename = result_gantt_filename_format.format(self.name)
        self.ctrlr.draw_incumbent_gantt(self.result_dir / result_gantt_filename)

    def save_progress_plot(self) -> None:
        from hybridflowshop.plotter.objective_progress import ObjectiveProgressPlotter

        progress_plot_filename_format = "{}_progress_plot.png"
        if "progress_plot_filename_format" in self.output_metadata:
            progress_plot_filename_format = self.output_metadata[
                "progress_plot_filename_format"
            ]
            if isinstance(progress_plot_filename_format, str):
                progress_plot_filename_format = progress_plot_filename_format.strip()

        progress_plot_filename = progress_plot_filename_format.format(self.name)
        ObjectiveProgressPlotter.plot_solution_progress(
            self.ctrlr.get_log(), self.result_dir / progress_plot_filename
        )
