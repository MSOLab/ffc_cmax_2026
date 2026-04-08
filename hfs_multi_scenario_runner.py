import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from routix import DynamicDataObject, StoppingCriteria
from routix.constants import SubroutineReportStatisticsKeys
from routix.runner import MultiScenarioRunner
from routix.type_defs import RunMode
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)
from xlsxwriter import Workbook
from xlsxwriter.worksheet import Worksheet

from hfs_config import BaselineColumnMapping
from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.report import (
    aggregate_scenario_endpoint_metrics_from_json,
    aggregate_scenario_progression,
    export_multi_scenario_method_rpdf_comparison_html,
)
from hybridflowshop.report.method_progression_report import (
    aggregate_scenario_endpoint_metrics_from_json,
)
from output_filenames import OutputFilenames

RPDF_PREFIX = "gap_"


class HfsMultiScenarioRunner(
    MultiScenarioRunner[
        HybridFlowshopParameters, HfsSingleInstanceRunner, HfsMultiInstanceRunner
    ]
):
    """
    Runner for executing multiple scenarios for Hybrid Flow Shop problems.
    This class specializes in post-processing the results to generate a
    comprehensive Excel report.
    """

    stat_name_func_pairs = [("Average", "mean"), ("Max", "max"), ("Min", "min")]

    def __init__(
        self,
        m_i_runner_class: type[HfsMultiInstanceRunner],
        s_i_runner_class: type[HfsSingleInstanceRunner],
        instances: Sequence[HybridFlowshopParameters],
        shared_param_dict: dict,
        scenario_configs: Sequence[dict[str, Any]],
        output_dir: Path,
        base_output_metadata: dict[str, Any],
        mode: RunMode = RunMode.FULL_RUN,
        instance_worker_cnt: int = 1,
    ):
        super().__init__(
            m_i_runner_class,
            s_i_runner_class,
            instances,
            shared_param_dict,
            scenario_configs,
            output_dir,
            base_output_metadata,
            mode=mode,
            instance_worker_cnt=instance_worker_cnt,
        )
        self.baseline_df: pd.DataFrame | None = None
        """DataFrame containing baseline results for comparison in the report."""

        if self.mode in {RunMode.FULL_RUN, RunMode.RESUME}:
            # --- Save scenario-specific config files for reproducibility ---
            for i, scenario_config in enumerate(self.scenario_configs):
                subroutine_flow: DynamicDataObject | None = scenario_config.get(
                    "subroutine_flow"
                )
                stopping_criteria: StoppingCriteria | None = scenario_config.get(
                    "stopping_criteria"
                )
                if subroutine_flow is None or stopping_criteria is None:
                    continue
                # Use a specific output subdir from config, or create a default one
                scenario_output_dir = self.output_dir / f"scenario_{i + 1}"
                if "output_subdir" in scenario_config:
                    scenario_output_dir = self.output_dir / str(
                        scenario_config["output_subdir"]
                    )
                scenario_output_dir.mkdir(parents=True, exist_ok=True)
                DynamicDataObject.safe_save_yaml(
                    subroutine_flow,
                    scenario_output_dir / OutputFilenames.SUBROUTINE_FLOW_CACHE_FN,
                )
                DynamicDataObject.safe_save_yaml(
                    stopping_criteria,
                    scenario_output_dir / OutputFilenames.STOPPING_CRITERIA_CACHE_FN,
                )

    def set_baseline_df(
        self, baseline_csv_path: Path, column_mapping: BaselineColumnMapping
    ):
        """
        Sets the baseline DataFrame for comparison in the report.
        This DataFrame should contain the baseline results for the scenarios.
        """
        if baseline_csv_path.exists():
            self.baseline_df = pd.read_csv(baseline_csv_path)
            logging.info(f"Baseline DataFrame loaded from {baseline_csv_path}")
            self.baseline_instance_col = column_mapping.instance
            self.baseline_obj_val_col = column_mapping.obj_val
            self.baseline_obj_bound_col = column_mapping.obj_bound
            for m_i_runner in self.runners:
                m_i_runner.set_baseline_df(self.baseline_df, column_mapping)
        else:
            logging.warning(f"Baseline CSV file not found at {baseline_csv_path}")
            self.baseline_df = pd.DataFrame()

    def post_run_process(self):
        """
        Aggregates results from all scenarios and generates a comprehensive Excel report
        that includes a comparative dashboard, raw data, and scenario information.
        """
        raw_summary_df = self._aggregate_scenario_summary(
            input_filename="multi_instance_summary.csv",
            output_filename="all_scenarios_summary.csv",
        )
        if raw_summary_df is None:
            return

        # Main report (full-run summary)
        dashboard_df = self.create_dashboard(raw_summary_df)
        info_df = self.create_info_sheet()
        excel_report_path = self.output_dir / "multi_scenario_report.xlsx"
        self.write_excel_report(
            excel_report_path,
            dashboard_df=dashboard_df,
            raw_summary_df=raw_summary_df,
            info_df=info_df,
            baseline_df=self.baseline_df,
        )
        self._create_method_rpdf_comparison_html()

        # Additional timepoint reports (labels follow configured timepoint_summaries)
        for label in self._resolve_timepoint_labels():
            raw_summary_timepoint_df = self._aggregate_scenario_summary(
                input_filename=f"multi_instance_summary_{label}.csv",
                output_filename=f"all_scenarios_summary_{label}.csv",
            )
            if raw_summary_timepoint_df is None:
                continue
            dashboard_timepoint_df = self.create_dashboard(raw_summary_timepoint_df)
            excel_report_timepoint_path = (
                self.output_dir / f"multi_scenario_report_{label}.xlsx"
            )
            self.write_excel_report(
                excel_report_timepoint_path,
                dashboard_df=dashboard_timepoint_df,
                raw_summary_df=raw_summary_timepoint_df,
                info_df=info_df,
                baseline_df=self.baseline_df,
            )

    def _create_method_rpdf_comparison_html(self) -> None:
        scenario_metrics: list[dict[str, object]] = []

        for i, runner in enumerate(self.runners):
            scenario_payload = self._load_scenario_metrics_from_json_or_csv(
                runner.working_dir, i
            )
            if scenario_payload is None:
                continue

            scenario_name = self.scenario_configs[i].get(
                "output_subdir", f"scenario_{i + 1}"
            )
            scenario_label = Path(str(scenario_name)).name
            endpoint_df = scenario_payload["endpoint_df"]
            if endpoint_df is None or endpoint_df.empty:
                continue
            scenario_metrics.append(
                {
                    "label": scenario_label,
                    "endpoint_df": endpoint_df,
                    "raw_progression_df": scenario_payload["raw_progression_df"],
                }
            )

        if not scenario_metrics:
            logging.warning(
                "No valid scenario method RPD summaries found for top-level HTML comparison."
            )
            return

        output_path = self.output_dir / "multi_scenario_subroutine_flow_comparison.html"
        try:
            created = export_multi_scenario_method_rpdf_comparison_html(
                scenario_metrics=scenario_metrics,
                output_path=output_path,
            )
            if not created:
                logging.warning(
                    "Skipped top-level multi-scenario method comparison HTML generation: "
                    "no valid aggregated traces."
                )
        except Exception as e:
            logging.error(
                f"Failed to export top-level multi-scenario method comparison HTML to {output_path}: {e}",
                exc_info=True,
            )

    def _load_scenario_metrics_from_json_or_csv(
        self, working_dir: Path, scenario_index: int
    ) -> dict[str, pd.DataFrame | None] | None:
        metrics_long_df = aggregate_scenario_endpoint_metrics_from_json(
            working_dir,
            baseline_df=getattr(self, "baseline_df", None),
            baseline_instance_col=getattr(self, "baseline_instance_col", "Instance"),
            baseline_obj_val_col=getattr(self, "baseline_obj_val_col", "UB"),
            record_all_subroutines=True,
            omitted_subroutines={
                "set_random_seed",
                "set_cp_model_as_base_cp_model",
            },
        )
        has_valid_json_metrics = (
            not metrics_long_df.empty
            and {"norm_time", "rpd_f"}.issubset(metrics_long_df.columns)
            and not metrics_long_df.dropna(subset=["norm_time", "rpd_f"]).empty
        )
        if has_valid_json_metrics:
            progression_data = aggregate_scenario_progression(
                working_dir,
                baseline_df=getattr(self, "baseline_df", None),
                baseline_instance_col=getattr(self, "baseline_instance_col", "Instance"),
                baseline_obj_val_col=getattr(self, "baseline_obj_val_col", "UB"),
                omitted_subroutines={
                    "set_random_seed",
                    "set_cp_model_as_base_cp_model",
                },
            )
            return {
                "endpoint_df": metrics_long_df,
                "raw_progression_df": progression_data.get("progression_df"),
            }

        summary_path = working_dir / "summary_method_rpdf_and_norm_time_long.csv"
        if not summary_path.exists():
            logging.warning(
                "Method RPD summary file not found for scenario %d at %s",
                scenario_index + 1,
                summary_path,
            )
            return None

        try:
            logging.warning(
                "Falling back to endpoint CSV for top-level method comparison in scenario %d at %s",
                scenario_index + 1,
                working_dir,
            )
            return {
                "endpoint_df": pd.read_csv(summary_path),
                "raw_progression_df": None,
            }
        except Exception as e:
            logging.error(
                f"Failed to load method RPD summary from {summary_path}: {e}",
                exc_info=True,
            )
            return None

    def _resolve_timepoint_labels(self) -> list[str]:
        output_metadata: dict[str, Any] = {}

        if hasattr(self, "base_output_metadata") and isinstance(
            self.base_output_metadata, dict
        ):
            output_metadata = self.base_output_metadata
        elif self.runners:
            first_runner = self.runners[0]
            if hasattr(first_runner, "output_metadata") and isinstance(
                first_runner.output_metadata, dict
            ):
                output_metadata = first_runner.output_metadata

        configured = output_metadata.get("timepoint_summaries")
        if not configured:
            return []

        labels: list[str] = []
        seen: set[str] = set()
        for cfg in configured:
            if not isinstance(cfg, dict):
                logging.warning(f"Timepoint summary config entry is not a dict: {cfg}")
                continue

            label = cfg.get("label")
            if not label:
                logging.warning("Timepoint summary config entry missing 'label' field")
                continue

            label_str = str(label).strip()
            if not label_str:
                logging.warning(
                    f"Timepoint summary label is blank after conversion: {label}"
                )
                continue
            if label_str in seen:
                logging.warning(
                    f"Duplicate timepoint summary label '{label_str}' skipped"
                )
                continue
            seen.add(label_str)
            labels.append(label_str)
        return labels

    def _aggregate_scenario_summary(
        self,
        input_filename: str,
        output_filename: str,
    ) -> pd.DataFrame | None:
        all_summary_dfs = []
        for i, runner in enumerate(self.runners):
            summary_path = runner.working_dir / input_filename
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                scenario_name = self.scenario_configs[i].get(
                    "output_subdir", f"scenario_{i + 1}"
                )
                scenario_col_name = Path(scenario_name).name
                df["scenario"] = str(scenario_col_name)
                all_summary_dfs.append(df)
            else:
                logging.warning(
                    f"Summary file not found for scenario {i + 1} at {summary_path}"
                )

        if not all_summary_dfs:
            logging.warning(
                f"No scenario summaries found to aggregate for {input_filename}."
            )
            return None

        raw_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
        output_path = self.output_dir / output_filename
        raw_summary_df.to_csv(output_path, index=False)
        logging.info(f"Aggregated summary saved to {output_path}")
        return raw_summary_df

    def create_dashboard(self, raw_summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a pivoted and styled dashboard for performance comparison,
        with a specific column order.
        """
        try:
            # 1. Pivot the raw data to get scenarios as columns
            best_obj_value_df = raw_summary_df.pivot_table(
                index=SubroutineReportStatisticsKeys.INSTANCE_NAME,
                columns="scenario",
                values=SubroutineReportStatisticsKeys.BEST_OBJ,
            ).reset_index()

            # 1b. Pivot totalElapsedTime for running time columns
            elapsed_time_df = raw_summary_df.pivot_table(
                index=SubroutineReportStatisticsKeys.INSTANCE_NAME,
                columns="scenario",
                values=SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME,
            ).reset_index()

            # Get clean scenario names (use last part of path)
            # Rename elapsed time columns to totalElapsedTime_<clean_name> BEFORE merge
            # to avoid suffix conflicts (_x, _y)
            elapsed_time_df_renamed = elapsed_time_df.copy()
            for scenario in elapsed_time_df.columns:
                if scenario != SubroutineReportStatisticsKeys.INSTANCE_NAME:
                    clean_name = Path(scenario).name
                    elapsed_time_df_renamed.rename(
                        columns={scenario: f"totalElapsedTime_{clean_name}"},
                        inplace=True,
                    )

            # 1c. Merge elapsed time into best_obj_value_df to create dashboard_df
            dashboard_df = pd.merge(
                best_obj_value_df,
                elapsed_time_df_renamed,
                on=SubroutineReportStatisticsKeys.INSTANCE_NAME,
                how="left",
            )

            # Rename all scenario columns to use clean names (without path)
            rename_map = {}

            original_scenario_cols = [
                col
                for col in best_obj_value_df.columns
                if col != SubroutineReportStatisticsKeys.INSTANCE_NAME
            ]
            for col in original_scenario_cols:
                clean_name = Path(col).name
                if col != clean_name:
                    rename_map[col] = clean_name

            if rename_map:
                dashboard_df.rename(columns=rename_map, inplace=True)

            # Now extract clean scenario names after renaming
            scenarios = [Path(col).name for col in original_scenario_cols]

            # 2. Merge with baseline data if available
            if self.baseline_df is not None and not self.baseline_df.empty:
                rename_map = {
                    self.baseline_instance_col: SubroutineReportStatisticsKeys.INSTANCE_NAME,
                    self.baseline_obj_val_col: "baselineObjVal",
                    self.baseline_obj_bound_col: "baselineBound",
                }
                cols = [
                    self.baseline_instance_col,
                    self.baseline_obj_val_col,
                    self.baseline_obj_bound_col,
                ]
                baseline_subset = self.baseline_df.loc[:, cols].copy()
                baseline_subset.rename(columns=rename_map, inplace=True)
                if baseline_subset.columns.duplicated().any():
                    dup = baseline_subset.columns[baseline_subset.columns.duplicated()]
                    logging.warning(f"Dropping duplicate baseline columns: {list(dup)}")
                    baseline_subset = baseline_subset.loc[
                        :, ~baseline_subset.columns.duplicated()
                    ]

                dashboard_df = pd.merge(
                    dashboard_df,
                    baseline_subset,
                    on=SubroutineReportStatisticsKeys.INSTANCE_NAME,
                    how="left",
                )
            else:
                logging.warning("Baseline data not available. Skipping merge.")
                dashboard_df["baselineObjVal"] = None

            # 3. Calculate RPDf for each scenario
            scenarios = [
                col
                for col in best_obj_value_df.columns
                if col != SubroutineReportStatisticsKeys.INSTANCE_NAME
            ]
            if (
                "baselineObjVal" in dashboard_df.columns
                and dashboard_df["baselineObjVal"].notna().any()
            ):
                for scenario in scenarios:
                    rpdf_col_name = f"{RPDF_PREFIX}{scenario}"
                    obj = dashboard_df[scenario]
                    ref = dashboard_df["baselineObjVal"]

                    # RPDf = (obj - ref) / ((obj + ref) / 2)
                    # If both obj and ref are 0, define as 0
                    numerator = obj - ref
                    denominator = (obj + ref) / 2

                    # Compute RPDf, handling 0/0 case
                    result = numerator / denominator
                    # Set Inf and NaN to 0 where both values were 0
                    result = result.replace([float("inf"), -float("inf")], 0)
                    result = result.fillna(0)

                    dashboard_df[rpdf_col_name] = result

            # 4. Define the desired column order
            ordered_columns = [SubroutineReportStatisticsKeys.INSTANCE_NAME]
            obj_val_cols = [col for col in scenarios]
            baseline_obj_val_col = (
                ["baselineObjVal"] if "baselineObjVal" in dashboard_df.columns else []
            )
            rel_diff_cols = [
                f"{RPDF_PREFIX}{scenario}"
                for scenario in scenarios
                if f"{RPDF_PREFIX}{scenario}" in dashboard_df.columns
            ]

            # Combine lists in the desired order
            # Order: instanceName, ObjVal cols, runningTime cols, baselineObjVal, RPDf cols
            final_column_order = (
                ordered_columns
                + obj_val_cols
                + [f"totalElapsedTime_{scenario}" for scenario in scenarios]
                + baseline_obj_val_col
                + rel_diff_cols
            )

            # Reorder the DataFrame
            final_dashboard = dashboard_df[final_column_order]

            # 5. Add summary statistics at the bottom

            summary_rows: list[dict[str, Any]] = []
            for stat_name, stat_func in self.stat_name_func_pairs:
                row: dict[str, Any] = {
                    SubroutineReportStatisticsKeys.INSTANCE_NAME: stat_name
                }
                for col in final_dashboard.columns:
                    if col != SubroutineReportStatisticsKeys.INSTANCE_NAME:
                        if pd.api.types.is_numeric_dtype(final_dashboard[col]):
                            row[col] = getattr(final_dashboard[col], stat_func)()
                summary_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)
            final_dashboard = pd.concat(
                [final_dashboard, summary_df], ignore_index=True
            )

            return final_dashboard

        except Exception as e:
            logging.error(f"Failed to create dashboard: {e}", exc_info=True)
            return pd.DataFrame()

    def create_info_sheet(self) -> pd.DataFrame:
        """Creates a DataFrame with detailed information about each scenario."""
        info_data = []
        for i, config in enumerate(self.scenario_configs):
            scenario_name = config.get("output_subdir", f"scenario_{i + 1}")
            info_data.append(
                {
                    "Scenario": str(scenario_name),
                    "Subroutine Flow": str(config.get("subroutine_flow")),
                    "Stopping Criteria": str(config.get("stopping_criteria")),
                    "Description": config.get("description", ""),
                }
            )
        return pd.DataFrame(info_data)

    def write_excel_report(
        self,
        path: Path,
        dashboard_df: pd.DataFrame,
        raw_summary_df: pd.DataFrame,
        info_df: pd.DataFrame,
        baseline_df: pd.DataFrame | None,
    ):
        """
        Writes the DataFrames to a styled Excel file using the xlsxwriter engine
        for robust formatting and auto-adjusted column widths.

        Args:
            path (Path): Path to save the Excel report.
            dashboard_df (pd.DataFrame): DataFrame containing the dashboard data.
            raw_summary_df (pd.DataFrame): DataFrame containing the raw summary data.
            info_df (pd.DataFrame): DataFrame containing scenario information.
            baseline_df (pd.DataFrame | None, optional): DataFrame containing baseline data, if available.
        """
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                # --- Write sheets in the desired order ---
                workbook: Workbook = writer.book
                # --- Create formats ---
                percent_format = workbook.add_format({"num_format": "0.00%"})

                # 1. Best objective Dashboard
                sheet_name = "BestObjDashboard"
                if not dashboard_df.empty:
                    # Create the multi-level header
                    # Categories: ObjVal, runningTime, baselineObjVal, RPDf between baseline
                    header = []
                    for col in dashboard_df.columns:
                        if RPDF_PREFIX in col:
                            header.append(
                                (
                                    "RPDf between baseline",
                                    col.replace(RPDF_PREFIX, ""),
                                )
                            )
                        elif col == SubroutineReportStatisticsKeys.INSTANCE_NAME:
                            header.append(("", "insId"))
                        elif col == "baselineObjVal":
                            header.append(("", "baselineObjVal"))
                        elif col.startswith("totalElapsedTime_"):
                            header.append(
                                ("runningTime", col.replace("totalElapsedTime_", ""))
                            )
                        else:
                            header.append(("ObjVal", col))
                    dashboard_df.columns = pd.MultiIndex.from_tuples(header)

                    dashboard_df.to_excel(writer, sheet_name=sheet_name, index=True)

                    worksheet: Worksheet = writer.sheets[sheet_name]
                    data_start_row = 3
                    data_end_row = len(dashboard_df) + data_start_row - 1

                    # --- Apply formatting and set column widths ---

                    # RPDf first_col and last_col
                    rel_diff_first_col = float("inf")  # Placeholder for first column
                    rel_diff_last_col = 0

                    # runningTime first_col and last_col
                    running_time_first_col = float("inf")
                    running_time_last_col = 0

                    # +1 for the index column
                    for col_idx, col_name in enumerate(dashboard_df.columns, 1):
                        # Calculate max width
                        header_l1 = str(col_name[0])
                        header_l2 = str(col_name[1])
                        data_len = _safe_max_str_len(dashboard_df[col_name])
                        max_len = max(len(header_l1), len(header_l2), data_len) + 2

                        worksheet.set_column(col_idx, col_idx, width=max_len)

                        if col_name[0] == "RPDf between baseline":
                            if rel_diff_first_col == float("inf"):
                                rel_diff_first_col = col_idx
                            if rel_diff_last_col < col_idx:
                                rel_diff_last_col = col_idx
                            worksheet.set_column(
                                col_idx, col_idx, max_len, percent_format
                            )

                        if col_name[0] == "runningTime":
                            if running_time_first_col == float("inf"):
                                running_time_first_col = col_idx
                            if running_time_last_col < col_idx:
                                running_time_last_col = col_idx
                            # Format running time as number with 4 decimal places
                            time_format = workbook.add_format({"num_format": "0.0000"})
                            worksheet.set_column(col_idx, col_idx, max_len, time_format)

                    if rel_diff_first_col != float("inf"):
                        worksheet.conditional_format(
                            data_start_row,
                            rel_diff_first_col,
                            data_end_row,
                            rel_diff_last_col,
                            {
                                "type": "data_bar",
                                "bar_color": "#638EC6",
                                "bar_negative_color": "#F8696B",
                                "bar_axis_position": "middle",
                            },
                        )

                # 2. Scenario_Info
                info_df.to_excel(writer, sheet_name="Scenario_Info", index=False)
                worksheet = writer.sheets["Scenario_Info"]
                for col_idx, col_name in enumerate(info_df.columns):
                    data_len = _safe_max_str_len(info_df[col_name])
                    max_len = max(len(str(col_name)), data_len) + 2

                    if col_name in {"Subroutine Flow", "Stopping Criteria"}:
                        worksheet.set_column(col_idx, col_idx, options={"hidden": True})
                    else:
                        worksheet.set_column(col_idx, col_idx, width=max_len)

                # 3. Raw_Summary
                raw_summary_df.to_excel(writer, sheet_name="Raw_Summary", index=False)
                worksheet = writer.sheets["Raw_Summary"]
                for col_idx, col_name in enumerate(raw_summary_df.columns):
                    data_len = _safe_max_str_len(raw_summary_df[col_name])
                    max_len = max(len(str(col_name)), data_len) + 2

                    if col_name == "methodCallCounts":
                        worksheet.set_column(col_idx, col_idx, options={"hidden": True})
                    else:
                        worksheet.set_column(col_idx, col_idx, width=max_len)
                    if col_name == "improvementRatio":
                        worksheet.set_column(
                            col_idx, col_idx, width=max_len, cell_format=percent_format
                        )

                # 4. Baseline_Data
                if baseline_df is not None and not baseline_df.empty:
                    baseline_df.to_excel(
                        writer, sheet_name="Baseline_Data", index=False
                    )
                    worksheet = writer.sheets["Baseline_Data"]
                    for col_idx, col_name in enumerate(baseline_df.columns):
                        data_len = _safe_max_str_len(baseline_df[col_name])
                        max_len = max(len(str(col_name)), data_len) + 2

                        worksheet.set_column(col_idx, col_idx, width=max_len)
                        if col_name in {"Gap", "RPD"}:
                            worksheet.set_column(
                                col_idx,
                                col_idx,
                                width=max_len,
                                cell_format=percent_format,
                            )

            logging.info(f"Successfully generated Excel report at: {path}")
        except Exception as e:
            logging.error(f"Failed to write Excel report: {e}", exc_info=True)


def _safe_max_str_len(s: pd.Series) -> int:
    # Robust against floats/NaN/None and mixed dtypes.
    lens = s.astype("string").fillna("").str.len()
    m = lens.max()
    return 0 if pd.isna(m) else int(m)
