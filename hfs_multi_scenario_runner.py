import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from routix import DynamicDataObject, StoppingCriteria
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
from output_filenames import OutputFilenames


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
        else:
            logging.warning(f"Baseline CSV file not found at {baseline_csv_path}")
            self.baseline_df = pd.DataFrame()

    def post_run_process(self):
        """
        Aggregates results from all scenarios and generates a comprehensive Excel report
        that includes a comparative dashboard, raw data, and scenario information.
        """
        # 1. Aggregate all scenario summaries
        all_summary_dfs = []
        for i, runner in enumerate(self.runners):
            summary_path = runner.working_dir / "multi_instance_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                scenario_name = self.scenario_configs[i].get(
                    "output_subdir", f"scenario_{i + 1}"
                )
                df["scenario"] = str(scenario_name)
                all_summary_dfs.append(df)
            else:
                logging.warning(
                    f"Summary file not found for scenario {i + 1} at {summary_path}"
                )

        if not all_summary_dfs:
            logging.warning("No scenario summaries found to aggregate.")
            return

        raw_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
        # Save the aggregated raw summary
        raw_summary_df.to_csv(
            self.output_dir / "all_scenarios_summary.csv", index=False
        )
        logging.info(f"Aggregated summary saved to {self.output_dir}")

        # 2. Create the comparison dashboard
        dashboard_df = self.create_dashboard(raw_summary_df)

        # 3. Create the scenario info sheet
        info_df = self.create_info_sheet()

        # 4. Write all DataFrames to a styled Excel report
        excel_report_path = self.output_dir / "multi_scenario_report.xlsx"
        self.write_excel_report(
            excel_report_path,
            dashboard_df=dashboard_df,
            raw_summary_df=raw_summary_df,
            info_df=info_df,
            baseline_df=self.baseline_df,
        )

    def create_dashboard(self, raw_summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a pivoted and styled dashboard for performance comparison,
        with a specific column order.
        """
        try:
            # 1. Pivot the raw data to get scenarios as columns
            best_obj_value_df = raw_summary_df.pivot_table(
                index="instanceName", columns="scenario", values="bestObj"
            ).reset_index()

            # 2. Merge with baseline data if available
            if self.baseline_df is not None and not self.baseline_df.empty:
                rename_map = {
                    self.baseline_instance_col: "instanceName",
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
                    best_obj_value_df,
                    baseline_subset,
                    on="instanceName",
                    how="left",
                )
            else:
                logging.warning("Baseline data not available. Skipping merge.")
                dashboard_df = best_obj_value_df
                dashboard_df["baselineObjVal"] = None

            # 3. Calculate gaps for each scenario
            scenarios = [
                col for col in best_obj_value_df.columns if col != "instanceName"
            ]
            if (
                "baselineObjVal" in dashboard_df.columns
                and dashboard_df["baselineObjVal"].notna().any()
            ):
                for scenario in scenarios:
                    gap_col_name = f"gap_{scenario}"
                    dashboard_df[gap_col_name] = (
                        dashboard_df[scenario] - dashboard_df["baselineObjVal"]
                    ) / dashboard_df["baselineObjVal"]

            # 4. Define the desired column order
            ordered_columns = ["instanceName"]
            obj_val_cols = [col for col in scenarios]
            baseline_obj_val_col = (
                ["baselineObjVal"] if "baselineObjVal" in dashboard_df.columns else []
            )
            rel_diff_cols = [
                f"gap_{scenario}"
                for scenario in scenarios
                if f"gap_{scenario}" in dashboard_df
            ]

            # Combine lists in the desired order
            final_column_order = (
                ordered_columns + obj_val_cols + baseline_obj_val_col + rel_diff_cols
            )

            # Reorder the DataFrame
            final_dashboard = dashboard_df[final_column_order]

            # 5. Add summary statistics at the bottom

            summary_rows: list[dict[str, Any]] = []
            for stat_name, stat_func in self.stat_name_func_pairs:
                row: dict[str, Any] = {"instanceName": stat_name}
                for col in final_dashboard.columns:
                    if col != "instanceName":
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
                    header = []
                    for col in dashboard_df.columns:
                        if "gap_" in col:
                            header.append(
                                ("relDiff between baseline", col.replace("gap_", ""))
                            )
                        elif col == "instanceName":
                            header.append(("", "insId"))
                        elif col == "baselineObjVal":
                            header.append(("", "baselineObjVal"))
                        else:
                            header.append(("ObjVal", col))
                    dashboard_df.columns = pd.MultiIndex.from_tuples(header)

                    dashboard_df.to_excel(writer, sheet_name=sheet_name, index=True)

                    worksheet: Worksheet = writer.sheets[sheet_name]
                    data_start_row = 3
                    data_end_row = len(dashboard_df) + data_start_row - 1

                    # --- Apply formatting and set column widths ---

                    # relDiff first_col and last_col
                    rel_diff_first_col = float("inf")  # Placeholder for first column
                    rel_diff_last_col = 0

                    # +1 for the index column
                    for col_idx, col_name in enumerate(dashboard_df.columns, 1):
                        # Calculate max width
                        header_l1 = str(col_name[0])
                        header_l2 = str(col_name[1])
                        data_len = _safe_max_str_len(dashboard_df[col_name])
                        max_len = max(len(header_l1), len(header_l2), data_len) + 2

                        worksheet.set_column(col_idx, col_idx, width=max_len)

                        if col_name[0] == "relDiff between baseline":
                            if rel_diff_first_col == float("inf"):
                                rel_diff_first_col = col_idx
                            if rel_diff_last_col < col_idx:
                                rel_diff_last_col = col_idx
                            worksheet.set_column(
                                col_idx, col_idx, max_len, percent_format
                            )

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
