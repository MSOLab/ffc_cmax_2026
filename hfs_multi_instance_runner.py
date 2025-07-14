import logging

import pandas as pd
from routix.runner import MultiInstanceConcurrentRunner
from routix.type_defs import RunMode
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.report.hfs_subroutine_report_statistics import (
    HfsSubroutineReportStatistics,
)


class HfsMultiInstanceRunner(
    MultiInstanceConcurrentRunner[HybridFlowshopParameters, HfsSingleInstanceRunner]
):
    def post_run_process(self) -> pd.DataFrame:
        """
        Aggregates results from all single instance runs into a summary DataFrame.
        In FULL_RUN mode, it uses the results from memory.
        In POST_PROCESS_ONLY mode, it reads the individual summary files from disk.
        """
        summary_dfs = []

        if self.mode == RunMode.FULL_RUN:
            logging.info("Aggregating results from in-memory objects.")
            all_stats: list[HfsSubroutineReportStatistics] = [
                res for res in self.results if res is not None
            ]
            if all_stats:
                summary_dfs = [stats.to_dataframe() for stats in all_stats]
        else:  # POST_PROCESS_ONLY
            logging.info(f"Reading summary files from disk in: {self.working_dir}")
            result_dir_name = self.output_metadata.get("result_dir_name", "results")
            summary_filename_format = self.output_metadata.get(
                "summary_filename_format", "{}_summary.csv"
            )

            for instance in self.instances:
                summary_filename = summary_filename_format.format(instance.name)
                # Directly construct the path based on the known directory structure
                # The working_dir of HfsMultiInstanceRunner is the timestamped scenario directory
                summary_path = (
                    self.working_dir
                    / instance.name
                    / result_dir_name
                    / summary_filename
                )

                if summary_path.exists():
                    logging.info(f"Found and reading summary file: {summary_path}")
                    summary_dfs.append(pd.read_csv(summary_path))
                else:
                    # Log the exact path that was checked for easier debugging
                    logging.warning(
                        f"Summary file not found for instance '{instance.name}' at expected path: {summary_path.resolve()}"
                    )

        if not summary_dfs:
            logging.warning("No data available to generate a multi-instance summary.")
            return pd.DataFrame()

        combined_df = pd.concat(summary_dfs, ignore_index=True)

        output_filename = "multi_instance_summary.csv"
        summary_path = self.working_dir / output_filename
        combined_df.to_csv(summary_path, index=False)
        logging.info(f"Multi-instance summary saved to {summary_path}")

        return combined_df
