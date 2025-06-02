from pathlib import Path
from typing import Any

from mbls import DynamicDataObject
from schore.hybridflowshop import HybridFlowShopProblem

from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.hfs_input_summary import HFSInputSummary
from hybridflowshop.hfs_summary import HFSSummary
from hybridflowshop.stopping_criteria import StoppingCriteria
from single_instance_solver import SingleInstanceSolver


class SingleHFSInstanceSolver(SingleInstanceSolver):
    instance: HybridFlowShopProblem
    ctrlr: HybridFlowShopCpLnsController

    def __init__(
        self,
        hfs_instance: HybridFlowShopProblem,
        pra_common_params_dict: dict,
        subroutine_flow: DynamicDataObject,
        stopping_criteria: StoppingCriteria,
        output_dir_path: Path,
        output_metadata: dict[str, Any],
    ):
        super().__init__(
            instance=hfs_instance,
            controller_init_kwargs=pra_common_params_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
            output_dir=output_dir_path,
            output_metadata=output_metadata,
        )

    def init_controller(self) -> HybridFlowShopCpLnsController:
        """Initialize the controller with the given instance and parameters."""
        return HybridFlowShopCpLnsController(
            hfs_instance=self.instance,
            subroutine_flow=self.subroutine_flow,
            stopping_criteria=self.stopping_criteria,
            **self.controller_init_kwargs,
        )

    def post_run_process(self):
        hfs_instance = self.instance
        ins_name = hfs_instance.name
        working_dir = self.output_dir_instance
        hfs_cp_lns_ctrlr = self.ctrlr

        # save

        input_summary = HFSInputSummary(
            name=ins_name,
            job_count=hfs_instance.job_count,
            stage_count=hfs_instance.stage_count,
            timelimit=self.stopping_criteria.timelimit,
        )
        expr_summary = hfs_cp_lns_ctrlr.get_experiment_summary()
        report = HFSSummary(inputs=input_summary, outputs=expr_summary)
        report_filename = ins_name + ".csv"
        report.save(working_dir / report_filename)

        if "result_gantt_filename_format" in self.output_metadata:
            result_gantt_filename_format = self.output_metadata[
                "result_gantt_filename_format"
            ]
            result_gantt_filename = result_gantt_filename_format.format(
                ins_name=ins_name
            )
            hfs_cp_lns_ctrlr.draw_incumbent_gantt(working_dir / result_gantt_filename)
