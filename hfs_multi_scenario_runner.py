import logging
from pathlib import Path
from typing import Any, Sequence
import pandas as pd

from routix.runner import MultiScenarioRunner
from routix.type_defs import RunMode
from schore.parameters_examples.parallel_shop.identical_flow import HybridFlowshopParameters

from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hfs_single_instance_runner import HfsSingleInstanceRunner

class HfsMultiScenarioRunner(
    MultiScenarioRunner[
        HybridFlowshopParameters, HfsSingleInstanceRunner, HfsMultiInstanceRunner
    ]
):
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
    ):
        super().__init__(
            m_i_runner_class=m_i_runner_class,
            s_i_runner_class=s_i_runner_class,
            instances=instances,
            shared_param_dict=shared_param_dict,
            scenario_configs=scenario_configs,
            output_dir=output_dir,
            base_output_metadata=base_output_metadata,
        )
        self.mode = mode

    def run(self):
        """
        Executes each scenario sequentially, passing the run mode.
        """
        self.runners.clear()
        self.results.clear()

        for i, scenario_config in enumerate(self.scenario_configs):
            logging.info(f"--- Starting Scenario {i+1}/{len(self.scenario_configs)} ---")
            logging.info(f"Scenario Config: {scenario_config}")

            subroutine_flow = scenario_config.get("subroutine_flow")
            stopping_criteria = scenario_config.get("stopping_criteria")
            
            if subroutine_flow is None or stopping_criteria is None:
                logging.warning(f"Skipping scenario {i+1} due to missing 'subroutine_flow' or 'stopping_criteria'.")
                continue

            scenario_output_dir = self.output_dir / scenario_config.get("output_subdir", f"scenario_{i+1}")
            scenario_output_dir.mkdir(parents=True, exist_ok=True)

            multi_instance_runner = self.m_i_runner_class(
                s_i_runner_class=self.s_i_runner_class,
                instances=self.instances,
                shared_param_dict=self.shared_param_dict,
                subroutine_flow=subroutine_flow,
                stopping_criteria=stopping_criteria,
                output_dir=scenario_output_dir,
                output_metadata=self.base_output_metadata.copy(),
                mode=self.mode,
            )
            
            self.runners.append(multi_instance_runner)
            try:
                result = multi_instance_runner.run()
                self.results.append(result)
            except Exception as e:
                logging.error(f"Error in scenario {i+1}: {e}", exc_info=True)
                self.results.append(None)

            logging.info(f"--- Finished Scenario {i+1}/{len(self.scenario_configs)} ---")

        return self.post_run_process()

    def post_run_process(self):
        """
        Aggregates results from all scenarios and generates a comprehensive Excel report.
        """
        super().post_run_process()
        
        all_scenario_dfs = []
        for i, df in enumerate(self.results):
            if df is not None and not df.empty:
                # Add a scenario identifier to each DataFrame
                scenario_name = self.scenario_configs[i].get("output_subdir", f"scenario_{i+1}")
                df['scenario'] = scenario_name
                all_scenario_dfs.append(df)
            else:
                logging.warning(f"No summary DataFrame found for scenario {i+1}.")

        if not all_scenario_dfs:
            logging.warning("No scenario summaries found to aggregate into an Excel report.")
            return

        # 1. Create the raw summary DataFrame
        raw_summary_df = pd.concat(all_scenario_dfs, ignore_index=True)
        
        # 2. Create the comparison dashboard (pivot table)
        try:
            dashboard_df = raw_summary_df.pivot_table(
                index='instanceName', 
                columns='scenario', 
                values='bestObj'
            )
            
            # Add a 'best_overall' column
            dashboard_df['best_overall'] = dashboard_df.min(axis=1)
            
            # Add relative gap columns
            for col in dashboard_df.columns:
                if col != 'best_overall':
                    dashboard_df[f'gap_{col}'] = (dashboard_df[col] - dashboard_df['best_overall']) / dashboard_df['best_overall']
            
        except Exception as e:
            logging.error(f"Failed to create pivot table for dashboard: {e}")
            dashboard_df = pd.DataFrame() # Create an empty df on error

        # 3. Create the info DataFrame
        info_data = []
        for i, config in enumerate(self.scenario_configs):
            scenario_name = config.get("output_subdir", f"scenario_{i+1}")
            info_data.append({
                "Scenario": scenario_name,
                "Subroutine Flow": str(config.get("subroutine_flow")),
                "Stopping Criteria": str(config.get("stopping_criteria")),
            })
        info_df = pd.DataFrame(info_data)

        # 4. Write all DataFrames to an Excel file
        excel_report_path = self.output_dir / "multi_scenario_report.xlsx"
        try:
            with pd.ExcelWriter(excel_report_path, engine='openpyxl') as writer:
                dashboard_df.to_excel(writer, sheet_name='Dashboard', index=True)
                raw_summary_df.to_excel(writer, sheet_name='Raw_Summary', index=False)
                info_df.to_excel(writer, sheet_name='Scenario_Info', index=False)
            
            logging.info(f"Successfully generated Excel report at: {excel_report_path}")
        except Exception as e:
            logging.error(f"Failed to write Excel report: {e}")
