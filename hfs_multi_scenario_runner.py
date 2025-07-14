import logging

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

    def post_run_process(self):
        """
        Aggregates results from all scenarios and generates a comprehensive Excel report.
        """
        all_scenario_dfs = []
        for i, df in enumerate(self.results):
            if df is not None and not df.empty:
                # Add a scenario identifier to each DataFrame
                scenario_name = self.scenario_configs[i].get(
                    "output_subdir", f"scenario_{i + 1}"
                )
                df["scenario"] = scenario_name
                all_scenario_dfs.append(df)
            else:
                logging.warning(f"No summary DataFrame found for scenario {i + 1}.")

        if not all_scenario_dfs:
            logging.warning(
                "No scenario summaries found to aggregate into an Excel report."
            )
            return

        # 1. Create the raw summary DataFrame
        raw_summary_df = pd.concat(all_scenario_dfs, ignore_index=True)

        # 2. Create the comparison dashboard (pivot table)
        try:
            dashboard_df = raw_summary_df.pivot_table(
                index="instanceName", columns="scenario", values="bestObj"
            )

            # Add a 'best_overall' column
            dashboard_df["best_overall"] = dashboard_df.min(axis=1)

            # Add relative gap columns
            for col in dashboard_df.columns:
                if col != "best_overall":
                    dashboard_df[f"gap_{col}"] = (
                        dashboard_df[col] - dashboard_df["best_overall"]
                    ) / dashboard_df["best_overall"]

        except Exception as e:
            logging.error(f"Failed to create pivot table for dashboard: {e}")
            dashboard_df = pd.DataFrame()  # Create an empty df on error

        # 3. Create the info DataFrame
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
        info_df = pd.DataFrame(info_data)

        # 4. Write all DataFrames to an Excel file
        excel_report_path = self.output_dir / "multi_scenario_report.xlsx"
        try:
            with pd.ExcelWriter(excel_report_path, engine="openpyxl") as writer:
                dashboard_df.to_excel(writer, sheet_name="Dashboard", index=True)
                raw_summary_df.to_excel(writer, sheet_name="Raw_Summary", index=False)
                info_df.to_excel(writer, sheet_name="Scenario_Info", index=False)

            logging.info(f"Successfully generated Excel report at: {excel_report_path}")
        except Exception as e:
            logging.error(f"Failed to write Excel report: {e}")
