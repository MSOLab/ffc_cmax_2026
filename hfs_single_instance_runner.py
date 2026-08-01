import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from mbls.cpsat import ObjValueBoundStore
from routix import DynamicDataObject, StoppingCriteria
from routix.constants import SubroutineReportStatisticsKeys
from routix.io.yaml import dump_yaml
from routix.runner import SingleInstanceRunner
from routix.type_defs import RunMode
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.constants import (
    INPUT_JOBCOUNT_COLUMN,
    INPUT_MACHINESPERSTAGE_COLUMN,
    INPUT_STAGECOUNT_COLUMN,
    INPUT_TIMELIMIT_COLUMN,
)
from hybridflowshop.controller import HybridFlowShopCpLnsController
from hybridflowshop.hfs_input_summary import HfsInputSummary
from hybridflowshop.hfs_summary import HfsSummary
from hybridflowshop.io_solution import END_TIME_MAP_KEY, START_TIME_MAP_KEY
from hybridflowshop.report.hfs_subroutine_report import HfsSubroutineReport
from hybridflowshop.report.hfs_subroutine_report_statistics import (
    HfsSubroutineReportStatistics,
)
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class HfsSingleInstanceRunner(
    SingleInstanceRunner[HybridFlowshopParameters, HybridFlowShopCpLnsController]
):
    MIP_LB_SUMMARY_COLUMNS = (
        "mipLbBound",
        "mipLbStatus",
        "mipLbDelta",
        "mipLbSolverRuntimeSec",
        "mipLbApplyElapsedSec",
        "mipLbDispatchCmax",
        "mipLbSelectedDispatchVariant",
    )
    RETAINED_CP_LB_SUMMARY_COLUMNS = (
        "retainedCpBound",
        "retainedCpCallCount",
        "retainedCpBounds",
        "retainedCpModes",
        "retainedCpBestBound",
        "retainedCpBestStatus",
        "retainedCpBestMode",
        "retainedCpBestStages",
        "retainedCpBestBottleneckStage",
        "retainedCpBestSolverRuntimeSec",
        "retainedCpBestApplyElapsedSec",
        "retainedCpBestBoundObj",
        "retainedCpStatus",
        "retainedCpMode",
        "retainedCpStages",
        "retainedCpBottleneckStage",
        "retainedCpSolverRuntimeSec",
        "retainedCpApplyElapsedSec",
        "retainedCpBestObj",
        "retainedCpDispatchObj",
        "retainedCpPostDispatchObj",
        "retainedCpDispatchCpUb",
        "retainedCpDispatchCpLb",
        "retainedCpSelectedDispatchVariant",
        "retainedCpPostSelectedDispatchVariant",
        "retainedCpDispatchAnchorStages",
        "retainedCpDispatchElapsedSec",
        "retainedCpDispatchUpdatedIncumbent",
        "retainedCpDispatchKeptIncumbent",
    )
    SUMMARY_ERROR_COLUMN = "error"
    SUMMARY_COLUMNS = (
        SubroutineReportStatisticsKeys.INSTANCE_NAME,
        INPUT_JOBCOUNT_COLUMN,
        INPUT_STAGECOUNT_COLUMN,
        INPUT_MACHINESPERSTAGE_COLUMN,
        INPUT_TIMELIMIT_COLUMN,
        SubroutineReportStatisticsKeys.FOUND_FEASIBLE_SOL,
        SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME,
        SubroutineReportStatisticsKeys.FIRST_OBJ,
        SubroutineReportStatisticsKeys.FIRST_BOUND,
        SubroutineReportStatisticsKeys.BEST_OBJ,
        SubroutineReportStatisticsKeys.BEST_BOUND,
        SubroutineReportStatisticsKeys.IMPROVEMENT_RATIO,
        SubroutineReportStatisticsKeys.METHOD_CALL_COUNTS,
        SubroutineReportStatisticsKeys.REPORT_COUNT,
        *MIP_LB_SUMMARY_COLUMNS,
        *RETAINED_CP_LB_SUMMARY_COLUMNS,
        SUMMARY_ERROR_COLUMN,
    )

    @classmethod
    def normalize_summary_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        """Return a summary row that follows the aggregate CSV contract."""
        return {column: row.get(column) for column in cls.SUMMARY_COLUMNS}

    # Optional member variables for RunMode.RESUME
    resume_start_time_map: dict | None = None
    """Start time map loaded from a resume solution file, if applicable."""
    resume_end_time_map: dict | None = None
    """End time map loaded from a resume solution file, if applicable."""
    resume_obj_store: ObjValueBoundStore | None = None
    """Objective value bound store loaded from a resume solution file, if applicable."""
    resume_summary_dict: dict[str, Any] | None = None
    """Summary dictionary loaded from a resume summary file, if applicable."""

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
        _stopping_criteria = StoppingCriteria.from_dict(stopping_criteria.to_obj())
        super().__init__(
            instance=instance,
            shared_param_dict=shared_param_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=_stopping_criteria,
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

        # Apply instance-wise timelimit if specified
        if (
            isinstance(self.stopping_criteria, StoppingCriteria)
            and hasattr(self.stopping_criteria, "timelimit_n_by_c_multiplier")
            and self.stopping_criteria.timelimit_n_by_c_multiplier is not None
            and self.stopping_criteria.timelimit_n_by_c_multiplier > 0
        ):
            n = self.instance.job_count
            c = self.instance.stage_count
            adjusted_timelimit = self.stopping_criteria.timelimit_n_by_c_multiplier * (
                n * c
            )
            # Override the timelimit
            logging.info(
                f"Adjusting timelimit for instance '{self.name}' with n={n}, m={c}: "
                f"new timelimit = {adjusted_timelimit} seconds."
            )
            self.stopping_criteria.timelimit = adjusted_timelimit

    def get_controller(self) -> HybridFlowShopCpLnsController:
        """Initialize the controller with the given instance and parameters."""
        return HybridFlowShopCpLnsController(
            self.instance,
            self.shared_param_dict,
            self.subroutine_flow,
            self.stopping_criteria,
        )

    def _try_apply_resume(self) -> None:
        # Use resume data injected by the multi-instance runner
        if (
            self.resume_start_time_map is not None
            and self.resume_end_time_map is not None
            and self.resume_obj_store is not None
            and self.resume_summary_dict is not None
        ):
            logging.info(f"Applying injected resume data for instance '{self.name}'")
            self.ctrlr.obj_store = self.resume_obj_store

            def _resume_number_or_none(value: Any) -> float | None:
                if value is None:
                    return None
                if pd.isna(value):
                    return None
                return float(value)

            init_report = HfsSubroutineReport(
                elapsed_time=0.0,
                obj_value=_resume_number_or_none(
                    self.resume_summary_dict.get("initObj", None)
                ),
                obj_bound=_resume_number_or_none(
                    self.resume_summary_dict.get("initBound", None)
                ),
                is_init=True,
            )
            self.ctrlr.solution_manager.register(init_report, None)

            last_report = HfsSubroutineReport(
                elapsed_time=float(
                    self.resume_summary_dict.get("totalElapsedTime", 0.0)
                ),
                obj_value=_resume_number_or_none(
                    self.resume_summary_dict.get("bestObj", None)
                ),
                obj_bound=_resume_number_or_none(
                    self.resume_summary_dict.get("bestBound", None)
                ),
                is_init=False,
            )
            last_solution = HybridFlowshopLiteSchedule(
                jobs=self.ctrlr.instance.job_id_list,
                stages=self.ctrlr.instance.stage_id_list,
                machines_per_stage=self.ctrlr.instance.stage_2_machines_map,
            )
            for key, start_time in self.resume_start_time_map.items():
                end_time = self.resume_end_time_map[key]
                j, i, k = key
                last_solution.add_ops_times_2_mc(
                    stage_id=i,
                    mc_id=k,
                    job_id=j,
                    start_time=start_time,
                    end_time=end_time,
                )
            self.ctrlr.solution_manager.register(last_report, last_solution)

            resume_controller_attr_by_summary_key = {
                "retainedCpDispatchObj": "last_retained_cp_dispatch_obj",
                "retainedCpPostDispatchObj": "last_retained_cp_post_dispatch_obj",
                "retainedCpDispatchCpUb": "last_retained_cp_dispatch_cp_ub",
                "retainedCpDispatchCpLb": "last_retained_cp_dispatch_cp_lb",
                "retainedCpDispatchElapsedSec": "last_retained_cp_dispatch_elapsed_sec",
                "retainedCpDispatchUpdatedIncumbent": (
                    "last_retained_cp_dispatch_was_incumbent_update"
                ),
                "retainedCpDispatchKeptIncumbent": (
                    "last_retained_cp_dispatch_kept_incumbent"
                ),
                "retainedCpSelectedDispatchVariant": (
                    "last_retained_cp_selected_dispatch_variant"
                ),
                "retainedCpPostSelectedDispatchVariant": (
                    "last_retained_cp_post_selected_dispatch_variant"
                ),
            }
            for summary_key, attr_name in resume_controller_attr_by_summary_key.items():
                value = self.resume_summary_dict.get(summary_key)
                if value is None or pd.isna(value):
                    continue
                setattr(self.ctrlr, attr_name, value)

            # current datetime - last_report.elapsed_time
            virtual_dt = datetime.datetime.now() - datetime.timedelta(
                seconds=last_report.elapsed_time
            )
            self.ctrlr.timer.set_start_time(virtual_dt)

    def run(self):
        """
        Run the subroutine controller for the instance.

        - This method initializes the controller and runs it if the mode is FULL_RUN.
        - If the mode is POST_PROCESS_ONLY, it skips the controller run and directly
        calls the post_run_process method.
        """
        try:
            if self.mode == RunMode.RESUME:
                self.ctrlr = self.get_controller()
                self.ctrlr.set_working_dir(self.working_dir)
                self._try_apply_resume()
                self.ctrlr.run(flow_resume_idx=self.flow_resume_idx)
            elif self.mode == RunMode.FULL_RUN:
                self.ctrlr = self.get_controller()
                self.ctrlr.set_working_dir(self.working_dir)
                self.ctrlr.run()
        except Exception:
            logging.error("An error occurred during the run", exc_info=True)
            raise

        return self.post_run_process()

    def post_run_process(self) -> dict[str, Any] | None:
        """Process results after running the instance.

        Returns:
            dict[str, Any] | None: Summary row as a dictionary, or None if no summary available.
        """
        if self.mode in {RunMode.FULL_RUN, RunMode.RESUME}:
            self.save_files(self.encoding)

        self.from_files_save_analysis(self.encoding)

        # Return summary row for multi-instance aggregation
        return self._create_summary_row()

    def _create_summary_row(self) -> dict[str, Any] | None:
        """Create a summary row dictionary from the instance result.

        Returns:
            dict[str, Any] | None: Summary row with instance metadata and results,
                                   or None if summary file not found.
        """
        try:
            if not self.summary_path.exists():
                return None

            df = pd.read_csv(self.summary_path)
            if df.empty:
                return None

            # Get the last row (best result)
            last_row = df.iloc[-1].to_dict()

            machine_count_per_stage = getattr(
                self.instance, "machine_count_per_stage", None
            )
            if (
                isinstance(machine_count_per_stage, list)
                and len(machine_count_per_stage) > 0
            ):
                machines_per_stage = machine_count_per_stage[0]
            else:
                machines_per_stage = None

            # Build summary row with instance info using routix constants
            summary_row = {
                SubroutineReportStatisticsKeys.INSTANCE_NAME: getattr(
                    self.instance, "name", None
                ),
                INPUT_JOBCOUNT_COLUMN: getattr(self.instance, "job_count", None),
                INPUT_STAGECOUNT_COLUMN: getattr(self.instance, "stage_count", None),
                INPUT_MACHINESPERSTAGE_COLUMN: machines_per_stage,
                INPUT_TIMELIMIT_COLUMN: getattr(
                    self.stopping_criteria, "timelimit", None
                ),
                SubroutineReportStatisticsKeys.FOUND_FEASIBLE_SOL: last_row.get(
                    "foundFeasibleSol"
                ),
                SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME: last_row.get(
                    "totalElapsedTime"
                ),
                SubroutineReportStatisticsKeys.FIRST_OBJ: last_row.get("firstObj"),
                SubroutineReportStatisticsKeys.FIRST_BOUND: last_row.get("firstBound"),
                SubroutineReportStatisticsKeys.BEST_OBJ: last_row.get("bestObj"),
                SubroutineReportStatisticsKeys.BEST_BOUND: last_row.get("bestBound"),
                SubroutineReportStatisticsKeys.IMPROVEMENT_RATIO: last_row.get(
                    "improvementRatio"
                ),
                SubroutineReportStatisticsKeys.METHOD_CALL_COUNTS: last_row.get(
                    "methodCallCounts"
                ),
                SubroutineReportStatisticsKeys.REPORT_COUNT: last_row.get(
                    "reportCount"
                ),
            }
            for column_name in (
                *self.MIP_LB_SUMMARY_COLUMNS,
                *self.RETAINED_CP_LB_SUMMARY_COLUMNS,
            ):
                summary_row[column_name] = last_row.get(column_name)

            return self.normalize_summary_row(summary_row)
        except Exception as e:
            logging.error(f"Error creating summary row for instance '{self.name}': {e}")
            return None

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
        self.save_subroutine_progression_json()

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
                # TODO: consider non-identical machines per stage
                machines_per_stage=self.instance.machine_count_per_stage[0],
                timelimit=self.stopping_criteria.timelimit,
            ),
            outputs=stats,
            extra_outputs=self._build_extra_summary_fields(),
        )
        summary.save(self.summary_path, encoding=encoding)

    @staticmethod
    def _format_retained_cp_lb_records(
        records: list[dict[str, Any]],
        key: str,
    ) -> str | None:
        values = []
        for record in records:
            value = record.get(key)
            if value is None:
                continue
            values.append(str(value))
        return ";".join(values) if values else None

    @staticmethod
    def _format_retained_cp_stage_ids(record: dict[str, Any] | None) -> str | None:
        if not record:
            return None
        stage_ids = list(record.get("stage_ids") or [])
        return " ".join(str(stage_id) for stage_id in stage_ids) if stage_ids else None

    def _build_extra_summary_fields(self) -> dict[str, Any]:
        mip_result = getattr(self.ctrlr, "last_mip_lb_result", None)
        solution_payload = getattr(self.ctrlr, "last_mip_lb_solution_payload", None)
        payload_metadata = (
            dict(solution_payload.get("metadata", {}) or {})
            if isinstance(solution_payload, dict)
            else {}
        )
        retained_cp_result = getattr(self.ctrlr, "last_retained_cp_lb_result", None)
        retained_cp_records = list(
            getattr(self.ctrlr, "retained_cp_lb_records", ()) or []
        )
        best_retained_cp_record = getattr(
            self.ctrlr,
            "best_retained_cp_lb_record",
            None,
        )
        retained_cp_stage_ids = (
            list(getattr(retained_cp_result, "retained_stage_ids", ()) or [])
            if retained_cp_result is not None
            else []
        )
        return {
            "mipLbBound": (
                getattr(mip_result, "certified_final_lb", None)
                if mip_result is not None
                else None
            ),
            "mipLbStatus": (
                getattr(mip_result, "status_name", None)
                if mip_result is not None
                else None
            ),
            "mipLbDelta": (
                getattr(mip_result, "delta", None) if mip_result is not None else None
            ),
            "mipLbSolverRuntimeSec": (
                getattr(mip_result, "total_runtime_sec", None)
                if mip_result is not None
                else None
            ),
            "mipLbApplyElapsedSec": getattr(
                self.ctrlr, "last_mip_lb_apply_elapsed_sec", None
            ),
            "mipLbDispatchCmax": payload_metadata.get("dispatch_cmax"),
            "mipLbSelectedDispatchVariant": getattr(
                self.ctrlr, "last_mip_lb_selected_dispatch_variant", None
            ),
            "retainedCpBound": (
                getattr(retained_cp_result, "certified_final_lb", None)
                if retained_cp_result is not None
                else None
            ),
            "retainedCpCallCount": (
                len(retained_cp_records) if retained_cp_records else None
            ),
            "retainedCpBounds": self._format_retained_cp_lb_records(
                retained_cp_records,
                "bound",
            ),
            "retainedCpModes": self._format_retained_cp_lb_records(
                retained_cp_records,
                "mode",
            ),
            "retainedCpBestBound": (
                best_retained_cp_record.get("bound")
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpBestStatus": (
                best_retained_cp_record.get("status")
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpBestMode": (
                best_retained_cp_record.get("mode")
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpBestStages": self._format_retained_cp_stage_ids(
                best_retained_cp_record
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpBestBottleneckStage": (
                best_retained_cp_record.get("bottleneck_stage_id")
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpBestSolverRuntimeSec": (
                best_retained_cp_record.get("solver_runtime_sec")
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpBestApplyElapsedSec": (
                best_retained_cp_record.get("apply_elapsed_sec")
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpBestBoundObj": (
                best_retained_cp_record.get("objective_ub")
                if isinstance(best_retained_cp_record, dict)
                else None
            ),
            "retainedCpStatus": (
                getattr(retained_cp_result, "status_name", None)
                if retained_cp_result is not None
                else None
            ),
            "retainedCpMode": (
                getattr(retained_cp_result, "retained_stage_mode", None)
                if retained_cp_result is not None
                else None
            ),
            "retainedCpStages": (
                " ".join(str(stage_id) for stage_id in retained_cp_stage_ids)
                if retained_cp_stage_ids
                else None
            ),
            "retainedCpBottleneckStage": (
                getattr(retained_cp_result, "bottleneck_stage_id", None)
                if retained_cp_result is not None
                else None
            ),
            "retainedCpSolverRuntimeSec": (
                getattr(retained_cp_result, "solver_runtime_sec", None)
                if retained_cp_result is not None
                else None
            ),
            "retainedCpApplyElapsedSec": getattr(
                self.ctrlr, "last_retained_cp_lb_apply_elapsed_sec", None
            ),
            "retainedCpBestObj": (
                getattr(retained_cp_result, "objective_ub", None)
                if retained_cp_result is not None
                else None
            ),
            "retainedCpDispatchObj": getattr(
                self.ctrlr, "last_retained_cp_dispatch_obj", None
            ),
            "retainedCpPostDispatchObj": getattr(
                self.ctrlr, "last_retained_cp_post_dispatch_obj", None
            ),
            "retainedCpDispatchCpUb": getattr(
                self.ctrlr, "last_retained_cp_dispatch_cp_ub", None
            ),
            "retainedCpDispatchCpLb": getattr(
                self.ctrlr, "last_retained_cp_dispatch_cp_lb", None
            ),
            "retainedCpSelectedDispatchVariant": getattr(
                self.ctrlr, "last_retained_cp_selected_dispatch_variant", None
            ),
            "retainedCpPostSelectedDispatchVariant": getattr(
                self.ctrlr, "last_retained_cp_post_selected_dispatch_variant", None
            ),
            "retainedCpDispatchAnchorStages": (
                " ".join(
                    str(stage_id)
                    for stage_id in getattr(
                        self.ctrlr, "last_retained_cp_dispatch_anchor_stage_ids", ()
                    )
                )
                or None
            ),
            "retainedCpDispatchElapsedSec": getattr(
                self.ctrlr, "last_retained_cp_dispatch_elapsed_sec", None
            ),
            "retainedCpDispatchUpdatedIncumbent": getattr(
                self.ctrlr, "last_retained_cp_dispatch_was_incumbent_update", None
            ),
            "retainedCpDispatchKeptIncumbent": getattr(
                self.ctrlr, "last_retained_cp_dispatch_kept_incumbent", None
            ),
        }

    def save_solution(self, encoding: str = "utf-8") -> None:
        incumbent_solution = self.ctrlr.solution_manager.get_incumbent()
        if incumbent_solution:
            solution_dict = {
                START_TIME_MAP_KEY: incumbent_solution.get_jik_2_start_time_map(),
                END_TIME_MAP_KEY: incumbent_solution.get_jik_2_end_time_map(),
            }
            dump_yaml(solution_dict, self.solution_path, encoding=encoding)

    def save_obj_value_bound_store(self, encoding: str = "utf-8") -> None:
        self.ctrlr.obj_store.save_yaml(self.obj_log_path, encoding=encoding)

    def save_subroutine_progression_json(self) -> None:
        progression_data = self.ctrlr.get_progression_data()
        output_path = self.result_dir / "subroutine_progression.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(progression_data, f, indent=2, default=str)

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
