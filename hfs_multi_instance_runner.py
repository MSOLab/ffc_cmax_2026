import logging
from pathlib import Path
from typing import Any

import pandas as pd
from mbls.cpsat import ObjValueBoundStore
from routix.runner import MultiInstanceConcurrentRunner
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.io_solution import get_end_time_dict, get_start_time_dict


class HfsMultiInstanceRunner(
    MultiInstanceConcurrentRunner[HybridFlowshopParameters, HfsSingleInstanceRunner]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_resume_data(self) -> None:
        self._check_file_existence()
        self._load_resume_solution_check_feasibility()
        self._load_obj_store_check_resume_solution_obj_value()
        self._load_summary_check_obj_values()
        # All resume data loaded & checks passed
        self._inject_resume_data_into_runners()

    def post_run_process(self) -> pd.DataFrame:
        """
        Aggregates results from all single instance runs into a summary DataFrame
        by reading the individual summary CSV files from disk.
        """
        summary_dfs = []
        logging.info(f"Aggregating instance summaries in: {self.working_dir}")

        result_dir_name = self.output_metadata.get("result_dir_name", "results")
        summary_filename_format: str = self.output_metadata.get(
            "summary_filename_format", "{}_summary.csv"
        )

        for instance in self.instances:
            summary_filename = summary_filename_format.format(instance.name)
            # Path construction based on the structure created by SingleInstanceRunner
            summary_path = (
                self.working_dir / instance.name / result_dir_name / summary_filename
            )

            if summary_path.exists():
                try:
                    df = pd.read_csv(summary_path)
                except pd.errors.EmptyDataError:
                    raise ValueError(
                        f"Summary file for instance '{instance.name}' is empty: {summary_path.resolve()}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Error reading summary file for instance '{instance.name}' at: {summary_path.resolve()}: {e}"
                    ) from e
                summary_dfs.append(df)
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

    def _load_resume_solution_check_feasibility(self) -> None:
        # Build filename formats with sensible defaults (can be overridden by output_metadata)
        solution_fn_format: str = self.output_metadata.get(
            "solution_fn_format", "{}_solution.yaml"
        )
        # Resume directory
        if "resume_root" not in self.output_metadata:
            raise ValueError("Missing 'resume_root' in output_metadata")
        resume_dir = Path(self.output_metadata["resume_root"])

        self.ins_name_to_start_time_map_map: dict[str, dict] = {}
        self.ins_name_to_end_time_map_map: dict[str, dict] = {}
        self.ins_name_to_obj_value_map: dict[str, float] = {}
        infeasible_instances = []

        for ins in self.instances:
            ins_name = (
                getattr(ins, "name", None)
                or getattr(ins, "instance_name", None)
                or str(ins)
            )
            inst_dir = resume_dir / str(ins_name) / "results"
            if not inst_dir.exists():
                inst_dir = resume_dir / str(ins_name)
            # solution
            sol_files = (
                list(inst_dir.glob(solution_fn_format.format(ins_name)))
                if inst_dir.exists()
                else []
            )
            if sol_files:
                sol_path = sol_files[0]
                temp_runner = HfsSingleInstanceRunner(
                    instance=ins,
                    shared_param_dict=self.shared_param_dict,
                    subroutine_flow=self.subroutine_flow,
                    stopping_criteria=self.stopping_criteria,
                    output_dir=self.output_dir,
                    output_metadata=self.output_metadata,
                    mode=self.mode,
                )
                temp_controller = temp_runner.get_controller()
                temp_controller.set_cp_model_as_base_cp_model()
                try:
                    start_time_map = get_start_time_dict(sol_path)
                    obj_val = temp_controller.check_feasibility(start_time_map)
                    end_time_map = get_end_time_dict(sol_path)
                    self.ins_name_to_start_time_map_map[ins_name] = start_time_map
                    self.ins_name_to_end_time_map_map[ins_name] = end_time_map
                    self.ins_name_to_obj_value_map[ins_name] = obj_val
                except RuntimeError:
                    infeasible_instances.append(ins_name)
                except Exception as e:
                    raise RuntimeError(
                        f"Error checking feasibility for instance '{ins_name}' with solution file '{sol_path}': {e}"
                    ) from e
            else:
                raise ValueError(
                    f"Solution file not found for instance '{ins_name}' at expected location: {inst_dir / solution_fn_format.format(ins_name)}"
                )
        if infeasible_instances:
            raise ValueError(
                f"The following instances have infeasible resume solutions: {infeasible_instances}"
            )

    def _load_obj_store_check_resume_solution_obj_value(self) -> None:
        if not hasattr(self, "ins_name_to_obj_value_map"):
            raise RuntimeError(
                "Resume solution feasibility has not been checked. Call _check_resume_solution_feasibility() first."
            )

        # Build filename formats with sensible defaults (can be overridden by output_metadata)
        obj_log_fn_format: str = self.output_metadata.get(
            "obj_log_fn_format", "{}_obj_log.yaml"
        )
        # Resume directory
        if "resume_root" not in self.output_metadata:
            raise ValueError("Missing 'resume_root' in output_metadata")
        resume_dir = Path(self.output_metadata["resume_root"])

        self.ins_name_to_obj_store_map: dict[str, ObjValueBoundStore[float]] = {}

        for ins in self.instances:
            ins_name = (
                getattr(ins, "name", None)
                or getattr(ins, "instance_name", None)
                or str(ins)
            )
            if ins_name not in self.ins_name_to_obj_value_map:
                raise ValueError(
                    f"Objective value for instance '{ins_name}' not found in resume data."
                )
            inst_dir = resume_dir / str(ins_name) / "results"
            if not inst_dir.exists():
                inst_dir = resume_dir / str(ins_name)
            # solution
            obj_log_files = (
                list(inst_dir.glob(obj_log_fn_format.format(ins_name)))
                if inst_dir.exists()
                else []
            )
            if obj_log_files:
                obj_log_path = obj_log_files[0]
                try:
                    resume_obj_store = ObjValueBoundStore.load_yaml(obj_log_path)
                    resume_obj_value = resume_obj_store.get_last_obj_value()
                    if resume_obj_value is None:
                        raise ValueError(
                            f"No objective value found in obj log for instance '{ins_name}'"
                        )
                    sol_obj_value = self.ins_name_to_obj_value_map[ins_name]
                    if abs(resume_obj_value - sol_obj_value) > 1e-6:
                        raise ValueError(
                            f"Objective value mismatch for instance '{ins_name}': "
                            f"resume obj log value {resume_obj_value} vs "
                            f"recorded solution obj value {sol_obj_value}"
                        )
                    self.ins_name_to_obj_store_map[ins_name] = resume_obj_store
                except Exception as e:
                    raise RuntimeError(
                        f"Error checking objective value for instance '{ins_name}' with obj log file '{obj_log_path}': {e}"
                    ) from e
            else:
                raise ValueError(
                    f"Objective log file not found for instance '{ins_name}' at expected location: {inst_dir / obj_log_fn_format.format(ins_name)}"
                )

    def _load_summary_check_obj_values(self) -> None:
        if not hasattr(self, "ins_name_to_obj_value_map"):
            raise RuntimeError(
                "Resume solution feasibility has not been checked. Call _check_resume_solution_feasibility() first."
            )

        # Build filename formats with sensible defaults (can be overridden by output_metadata)
        summary_fn_format: str = self.output_metadata.get(
            "summary_fn_format", "{}_summary.csv"
        )
        # Resume directory
        if "resume_root" not in self.output_metadata:
            raise ValueError("Missing 'resume_root' in output_metadata")
        resume_dir = Path(self.output_metadata["resume_root"])

        self.ins_name_to_summary_map: dict[str, dict[str, Any]] = {}

        for ins in self.instances:
            ins_name = (
                getattr(ins, "name", None)
                or getattr(ins, "instance_name", None)
                or str(ins)
            )
            if ins_name not in self.ins_name_to_obj_value_map:
                raise ValueError(
                    f"Objective value for instance '{ins_name}' not found in resume data."
                )
            inst_dir = resume_dir / str(ins_name) / "results"
            if not inst_dir.exists():
                inst_dir = resume_dir / str(ins_name)
            # summary
            sum_files = (
                list(inst_dir.glob(summary_fn_format.format(ins_name)))
                if inst_dir.exists()
                else []
            )
            if sum_files:
                sum_path = sum_files[0]
                try:
                    df = pd.read_csv(sum_path)
                    if "bestObj" not in df.columns:
                        raise ValueError(
                            f"'bestObj' column not found in summary file for instance '{ins_name}'"
                        )
                    if df.empty:
                        raise ValueError(
                            f"Summary file for instance '{ins_name}' is empty"
                        )
                    summary_dict = df.iloc[-1].to_dict()
                    summary_obj_value = summary_dict["bestObj"]
                    sol_obj_value = self.ins_name_to_obj_value_map[ins_name]
                    if abs(summary_obj_value - sol_obj_value) > 1e-6:
                        raise ValueError(
                            f"Objective value mismatch for instance '{ins_name}': "
                            f"summary bestObj value {summary_obj_value} vs "
                            f"recorded solution obj value {sol_obj_value}"
                        )
                    self.ins_name_to_summary_map[ins_name] = summary_dict
                except Exception as e:
                    raise RuntimeError(
                        f"Error checking objective value for instance '{ins_name}' with summary file '{sum_path}': {e}"
                    ) from e
            else:
                raise ValueError(
                    f"Summary file not found for instance '{ins_name}' at expected location: {inst_dir / summary_fn_format.format(ins_name)}"
                )

    def _inject_resume_data_into_runners(self) -> None:
        if not hasattr(self, "ins_name_to_start_time_map_map") or not hasattr(
            self, "ins_name_to_obj_store_map"
        ):
            raise RuntimeError(
                "Resume solution feasibility and objective store have not been loaded. Call the respective methods first."
            )

        for i, ins in enumerate(self.instances):
            ins_name = (
                getattr(ins, "name", None)
                or getattr(ins, "instance_name", None)
                or str(ins)
            )
            runner: HfsSingleInstanceRunner = self.runners[i]
            runner.resume_start_time_map = self.ins_name_to_start_time_map_map[ins_name]
            runner.resume_end_time_map = self.ins_name_to_end_time_map_map[ins_name]
            runner.resume_obj_store = self.ins_name_to_obj_store_map[ins_name]
            runner.resume_summary_dict = self.ins_name_to_summary_map[ins_name]
