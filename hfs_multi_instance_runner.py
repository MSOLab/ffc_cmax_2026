import logging

import pandas as pd
from routix.runner import MultiInstanceConcurrentRunner
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hfs_single_instance_runner import HfsSingleInstanceRunner


class HfsMultiInstanceRunner(
    MultiInstanceConcurrentRunner[HybridFlowshopParameters, HfsSingleInstanceRunner]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_run_process(self) -> pd.DataFrame:
        """
        Aggregates results from all single instance runs into a summary DataFrame
        by reading the individual summary CSV files from disk.
        """
        summary_dfs = []
        logging.info(f"Aggregating instance summaries in: {self.working_dir}")

        result_dir_name = self.output_metadata.get("result_dir_name", "results")
        summary_filename_format = self.output_metadata.get(
            "summary_filename_format", "{}_summary.csv"
        )

        for instance in self.instances:
            summary_filename = summary_filename_format.format(instance.name)
            # Path construction based on the structure created by SingleInstanceRunner
            summary_path = (
                self.working_dir / instance.name / result_dir_name / summary_filename
            )

            if summary_path.exists():
                summary_dfs.append(pd.read_csv(summary_path))
            else:
                logging.warning(
                    f"Summary file not found for instance '{instance.name}' at: {summary_path.resolve()}"
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
