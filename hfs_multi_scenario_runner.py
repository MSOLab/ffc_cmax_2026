import logging
from pathlib import Path

import pandas as pd
from routix.runner import MultiScenarioRunner
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hfs_single_instance_runner import HfsSingleInstanceRunner


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

    def set_baseline_df(self, baseline_csv_path: Path):
        """
        Sets the baseline DataFrame for comparison in the report.
        This DataFrame should contain the baseline results for the scenarios.
        """
        if baseline_csv_path.exists():
            self.baseline_df = pd.read_csv(baseline_csv_path)
            logging.info(f"Baseline DataFrame loaded from {baseline_csv_path}")
        else:
            logging.warning(f"Baseline CSV file not found at {baseline_csv_path}")
            self.baseline_df = pd.DataFrame()

    def post_run_process(self):
        """
        Aggregates results from all scenarios and generates a comprehensive Excel report
        that includes a comparative dashboard.
        """
        all_summary_dfs = []
        for i, runner in enumerate(self.runners):
            summary_path = runner.working_dir / "multi_instance_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                scenario_name = self.scenario_configs[i].get(
                    "output_subdir", f"scenario_{i + 1}"
                )
                df["scenario"] = scenario_name
                all_summary_dfs.append(df)
            else:
                logging.warning(
                    f"Summary file not found for scenario {i + 1} at {summary_path}"
                )

        if not all_summary_dfs:
            logging.warning("No scenario summaries found to aggregate.")
            return

        # 1. Create the raw summary DataFrame
        raw_summary_df = pd.concat(all_summary_dfs, ignore_index=True)

        # 2. Create the comparison dashboard
        dashboard_df = self.create_dashboard(raw_summary_df)

        # 3. Create the info DataFrame
        info_df = self.create_info_sheet()

        # 4. Write all DataFrames to an Excel file with styling
        excel_report_path = self.output_dir / "multi_scenario_report.xlsx"
        self.write_excel_report(
            excel_report_path, dashboard_df, raw_summary_df, info_df
        )

    def create_dashboard(
        self,
        raw_summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Creates a pivoted dashboard DataFrame for performance comparison."""
        try:
            # Pivot the raw data to have scenarios as columns
            pivot_df = raw_summary_df.pivot_table(
                index="instanceName", columns="scenario", values="bestObj"
            )

            # Load baseline data for comparison
            if self.baseline_df is not None and not self.baseline_df.empty:
                baseline_df = self.baseline_df.rename(
                    columns={"name": "instanceName", "ObjVal": "baseline"}
                )[["instanceName", "baseline"]]
                # Merge baseline data into the dashboard
                dashboard_df = pd.merge(
                    pivot_df, baseline_df, on="instanceName", how="left"
                )
                # Set instanceName as index again after merge
                dashboard_df.set_index("instanceName", inplace=True)
            else:
                logging.warning(
                    "Baseline data file not available. Skipping gap calculation."
                )
                dashboard_df = pivot_df.copy()
                dashboard_df["baseline"] = None

            # Calculate best overall and gaps
            scenarios_to_compare = [
                col for col in dashboard_df.columns if col != "baseline"
            ]
            dashboard_df["best_overall"] = dashboard_df[scenarios_to_compare].min(
                axis=1
            )

            if (
                "baseline" in dashboard_df.columns
                and dashboard_df["baseline"].notna().any()
            ):
                for col in scenarios_to_compare:
                    dashboard_df[f"gap_vs_baseline_{col}"] = (
                        dashboard_df[col] - dashboard_df["baseline"]
                    ) / dashboard_df["baseline"]

            return (
                dashboard_df.reset_index()
            )  # Reset index to make 'instanceName' a column

        except Exception as e:
            logging.error(
                f"Failed to create pivot table for dashboard: {e}", exc_info=True
            )
            return pd.DataFrame()

    def create_info_sheet(self) -> pd.DataFrame:
        """Creates a DataFrame with information about each scenario."""
        info_data = []
        for i, config in enumerate(self.scenario_configs):
            scenario_name = config.get("output_subdir", f"scenario_{i + 1}")
            info_data.append(
                {
                    "Scenario": scenario_name,
                    "Subroutine Flow": str(config.get("subroutine_flow")),
                    "Stopping Criteria": str(config.get("stopping_criteria")),
                }
            )
        return pd.DataFrame(info_data)

    def write_excel_report(
        self,
        path: Path,
        dashboard_df: pd.DataFrame,
        raw_summary_df: pd.DataFrame,
        info_df: pd.DataFrame,
    ):
        """Writes the DataFrames to a styled Excel file."""
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                dashboard_df.to_excel(writer, sheet_name="Dashboard", index=False)
                raw_summary_df.to_excel(writer, sheet_name="Raw_Summary", index=False)
                info_df.to_excel(writer, sheet_name="Scenario_Info", index=False)

                # Auto-adjust column widths for readability
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = max_length + 2
                        worksheet.column_dimensions[
                            column_letter
                        ].width = adjusted_width

            logging.info(f"Successfully generated Excel report at: {path}")
        except Exception as e:
            logging.error(f"Failed to write Excel report: {e}", exc_info=True)
