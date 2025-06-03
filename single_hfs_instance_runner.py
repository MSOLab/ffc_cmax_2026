from pathlib import Path
from typing import Any

from mbls import DynamicDataObject
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
        hfs_instance: HybridFlowShopProblem,
        pra_shared_params_dict: dict,
        subroutine_flow: DynamicDataObject,
        stopping_criteria: StoppingCriteria,
        output_dir_path: Path,
        output_metadata: dict[str, Any],
    ):
        super().__init__(
            instance=hfs_instance,
            shared_params=pra_shared_params_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
            output_dir=output_dir_path,
            output_metadata=output_metadata,
        )

    def init_controller(self) -> HybridFlowShopCpLnsController:
        """Initialize the controller with the given instance and parameters."""

        # controller_init_kwargs
        horizon = self.shared_params["horizon"]

        return HybridFlowShopCpLnsController(
            self.instance,
            self.subroutine_flow,
            self.stopping_criteria,
            horizon,
        )

    def post_run_process(self) -> None:
        ins_name = self.instance.name

        # save

        input_summary = HFSInputSummary(
            name=ins_name,
            job_count=self.instance.job_count,
            stage_count=self.instance.stage_count,
            timelimit=self.stopping_criteria.timelimit,
        )
        expr_summary = self.ctrlr.get_experiment_summary()
        report = HFSSummary(inputs=input_summary, outputs=expr_summary)

        report_filename = ins_name + "_summary.csv"
        if "report_filename_format" in self.output_metadata:
            report_filename_format = self.output_metadata["report_filename_format"]
            if isinstance(report_filename_format, str):
                report_filename_format = report_filename_format.strip()
                report_filename = report_filename_format.format(ins_name=ins_name)

        report.save(self.working_dir / report_filename)

        if "result_gantt_filename_format" in self.output_metadata:
            result_gantt_filename_format = self.output_metadata[
                "result_gantt_filename_format"
            ]
            if isinstance(result_gantt_filename_format, str):
                result_gantt_filename_format = result_gantt_filename_format.strip()
                result_gantt_filename = result_gantt_filename_format.format(
                    ins_name=ins_name
                )
                self.ctrlr.draw_incumbent_gantt(
                    self.working_dir / result_gantt_filename
                )
