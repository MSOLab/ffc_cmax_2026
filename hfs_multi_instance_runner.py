import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from mbls.cpsat import ObjValueBoundStore
from routix.constants import SubroutineReportStatisticsKeys
from routix.runner import MultiInstanceConcurrentRunner
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from exp_compare.metrics import compute_rpdf
from hfs_config import BaselineColumnMapping
from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.constants import INPUT_TIMELIMIT_COLUMN
from hybridflowshop.io_solution import get_end_time_dict, get_start_time_dict
from scripts.process_logs import create_method_end_time_and_obj_value_summary


class HfsMultiInstanceRunner(
    MultiInstanceConcurrentRunner[HybridFlowshopParameters, HfsSingleInstanceRunner]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self) -> Any:
        """Override run to append results to CSV immediately after each instance completes."""
        instance_worker_cnt = min(self.get_instance_worker_cnt(), len(self.instances))
        if instance_worker_cnt == 1:
            # Sequential execution
            return self._run_sequential()
        else:
            # Concurrent execution
            return self._run_concurrent(instance_worker_cnt)

    def _run_sequential(self) -> Any:
        """Run instances sequentially, appending results to CSV after each completion."""
        import traceback

        for idx, runner in enumerate(self.runners):
            try:
                result = runner.run()
                if result is not None and isinstance(result, dict):
                    self.append_result(result)
            except Exception as e:
                logging.error(f"Error in instance {runner.ins_name}: {e}")
                traceback.print_exc()
                result = None
                self.results.append(result)

        return self.post_run_process()

    def _run_concurrent(self, instance_worker_cnt: int) -> Any:
        """Run instances concurrently, appending results to CSV as they complete."""
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=instance_worker_cnt
        ) as executor:
            # Submit the run method of each pre-created runner instance
            futures = {executor.submit(runner.run): runner for runner in self.runners}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None and isinstance(result, dict):
                        self.append_result(result)
                except Exception as e:
                    logging.error(f"Error in concurrent instance: {e}")
                    self.results.append(None)

        return self.post_run_process()

    def _load_resume_data(self) -> None:
        self._check_file_existence()
        self._load_resume_solution_check_feasibility()
        self._load_obj_store_check_resume_solution_obj_value()
        self._load_summary_check_obj_values()
        # All resume data loaded & checks passed
        self._inject_resume_data_into_runners()

    def set_baseline_df(
        self, baseline_df: pd.DataFrame, column_mapping: BaselineColumnMapping
    ) -> None:
        instance_col = column_mapping.instance
        obj_val_col = column_mapping.obj_val
        job_cnt_col = column_mapping.job_cnt
        stage_cnt_col = column_mapping.stage_cnt
        if instance_col not in baseline_df.columns:
            raise ValueError(
                f"Instance column '{instance_col}' not found in baseline DataFrame."
            )
        if obj_val_col not in baseline_df.columns:
            raise ValueError(
                f"Objective value column '{obj_val_col}' not found in baseline DataFrame."
            )
        if job_cnt_col not in baseline_df.columns:
            raise ValueError(
                f"Job count column '{job_cnt_col}' not found in baseline DataFrame."
            )
        if stage_cnt_col not in baseline_df.columns:
            raise ValueError(
                f"Stage count column '{stage_cnt_col}' not found in baseline DataFrame."
            )
        self.baseline_df = baseline_df
        self.baseline_instance_col = instance_col
        self.baseline_job_cnt_col = job_cnt_col
        self.baseline_stage_cnt_col = stage_cnt_col
        self.baseline_obj_val_col = obj_val_col
        logging.info(
            f"Baseline DataFrame set with {len(baseline_df)} rows, "
            f"instance column: '{instance_col}', objective value column: '{obj_val_col}', "
            f"job count column: '{job_cnt_col}', stage count column: '{stage_cnt_col}'."
        )

    def post_run_process(self) -> pd.DataFrame:
        """
        Aggregates results from all single instance runs into a summary DataFrame.
        Each instance's summary row is appended to multi_instance_summary.csv as soon
        as the instance completes.

        Returns:
            pd.DataFrame: Combined summary DataFrame for all instances.
        """
        # Process Logs for this scenario
        logging.info(f"Starting Log Processing for scenario in: {self.working_dir}")
        try:
            method_end_time_obj_val_df = create_method_end_time_and_obj_value_summary(
                self.working_dir,
                baseline_df=self.baseline_df,
                baseline_instance_col=getattr(
                    self, "baseline_instance_col", "Instance"
                ),
                baseline_job_cnt_col=getattr(self, "baseline_job_cnt_col", "n"),
                baseline_stage_cnt_col=getattr(self, "baseline_stage_cnt_col", "s"),
                baseline_obj_val_col=getattr(self, "baseline_obj_val_col", "UB"),
                record_all_subroutines=True,
                omitted_subroutines={
                    "set_random_seed",
                    "set_cp_model_as_base_cp_model",
                },
            )
            if method_end_time_obj_val_df is not None:
                # Create (end time / timelimit, rpd) summary
                self._create_rpd_summary()
        except Exception as e:
            logging.error(f"Error processing logs for {self.working_dir}: {e}")
        logging.info("Log Processing Complete.")

        # Aggregate results from self.results (populated by append_result() during run())
        logging.info(f"Aggregating instance summaries in: {self.working_dir}")

        summary_rows = [r for r in self.results if isinstance(r, dict)]

        if not summary_rows:
            logging.warning("No data available to generate a multi-instance summary.")
            return pd.DataFrame()

        combined_df = pd.DataFrame(summary_rows)

        output_filename = "multi_instance_summary.csv"
        summary_path = self.working_dir / output_filename

        # Re-write the CSV with the final combined data
        combined_df.to_csv(summary_path, index=False)
        logging.info(f"Multi-instance summary saved to {summary_path}")
        self._create_timepoint_summaries(combined_df)

        return combined_df

    def _create_timepoint_summaries(self, base_df: pd.DataFrame) -> None:
        obj_value_points_map, obj_bound_points_map = (
            self._load_obj_store_maps_for_instances()
        )

        for cfg in self._resolve_timepoint_summaries():
            summary_df = self._build_timepoint_summary(
                base_df=base_df,
                label=str(cfg["label"]),
                mode=str(cfg["mode"]),
                value=float(cfg["value"]),
                exclude_if_timelimit_lt=cfg.get("exclude_if_timelimit_lt"),
                obj_value_points_map=obj_value_points_map,
                obj_bound_points_map=obj_bound_points_map,
            )
            output_path = (
                self.working_dir / f"multi_instance_summary_{cfg['label']}.csv"
            )
            summary_df.to_csv(output_path, index=False)
            logging.info(f"Timepoint summary saved to {output_path}")

    def _resolve_timepoint_summaries(self) -> list[dict[str, Any]]:
        configured = self.output_metadata.get("timepoint_summaries")
        if not configured:
            return []

        valid_modes = {"timelimit_ratio", "absolute_sec"}
        summaries: list[dict[str, Any]] = []
        for cfg in configured:
            if not isinstance(cfg, dict):
                continue
            label = cfg.get("label")
            mode = cfg.get("mode")
            value = cfg.get("value")
            if not label or mode not in valid_modes or value is None:
                continue
            summaries.append(
                {
                    "label": str(label),
                    "mode": str(mode),
                    "value": float(value),
                    "exclude_if_timelimit_lt": (
                        float(cfg["exclude_if_timelimit_lt"])
                        if cfg.get("exclude_if_timelimit_lt") is not None
                        else None
                    ),
                }
            )

        if not summaries:
            logging.warning(
                "No valid timepoint_summaries found in output_metadata. Timepoint summaries will be skipped."
            )
            return []
        return summaries

    def _load_obj_store_maps_for_instances(
        self,
    ) -> tuple[
        dict[str, list[tuple[float, float]]], dict[str, list[tuple[float, float]]]
    ]:
        obj_value_points_map: dict[str, list[tuple[float, float]]] = {}
        obj_bound_points_map: dict[str, list[tuple[float, float]]] = {}

        for runner in self.runners:
            ins_name = str(getattr(runner, "name", ""))
            if not ins_name:
                continue

            obj_log_path: Path | None = getattr(runner, "obj_log_path", None)
            if obj_log_path is None:
                obj_log_path = (
                    Path(getattr(runner, "working_dir"))
                    / "results"
                    / f"{ins_name}_obj_log.yaml"
                )

            if not obj_log_path.exists():
                logging.warning(
                    f"Obj log file not found for {ins_name}: {obj_log_path}"
                )
                obj_value_points_map[ins_name] = []
                obj_bound_points_map[ins_name] = []
                continue

            try:
                with open(obj_log_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
            except Exception as e:
                logging.warning(f"Failed to load obj log {obj_log_path}: {e}")
                obj_value_points_map[ins_name] = []
                obj_bound_points_map[ins_name] = []
                continue

            obj_value_data = (
                content.get("obj_value", {}).get("data", {})
                if isinstance(content, dict)
                else {}
            )
            obj_bound_data = (
                content.get("obj_bound", {}).get("data", {})
                if isinstance(content, dict)
                else {}
            )

            obj_value_points_map[ins_name] = self._to_sorted_time_points(obj_value_data)
            obj_bound_points_map[ins_name] = self._to_sorted_time_points(obj_bound_data)

        return obj_value_points_map, obj_bound_points_map

    @staticmethod
    def _to_sorted_time_points(data: dict[Any, Any]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for time_key, value in data.items():
            try:
                sec = float(time_key)
                val = float(value)
            except (TypeError, ValueError):
                continue
            points.append((sec, val))
        points.sort(key=lambda x: x[0])
        return points

    @staticmethod
    def _sample_last_known_value(
        sorted_points: list[tuple[float, float]],
        target_sec: float,
    ) -> float | None:
        last_val: float | None = None
        for sec, val in sorted_points:
            if sec <= target_sec:
                last_val = val
            else:
                break
        return last_val

    @staticmethod
    def _get_instance_points(
        points_map: dict[str, list[tuple[float, float]]], instance_name: Any
    ) -> list[tuple[float, float]]:
        candidate_keys: list[str] = []

        if isinstance(instance_name, str):
            stripped = instance_name.strip()
            candidate_keys.extend([instance_name, stripped])
            try:
                num_val = float(stripped)
                if num_val.is_integer():
                    candidate_keys.append(str(int(num_val)))
            except ValueError:
                pass
        else:
            key = str(instance_name)
            candidate_keys.append(key)
            if isinstance(instance_name, (int, float)):
                num_val = float(instance_name)
                if num_val.is_integer():
                    candidate_keys.append(str(int(num_val)))

        for key in candidate_keys:
            if key in points_map:
                return points_map[key]
        return []

    @staticmethod
    def _build_timepoint_summary(
        base_df: pd.DataFrame,
        label: str,
        mode: str,
        value: float,
        exclude_if_timelimit_lt: float | None,
        obj_value_points_map: dict[str, list[tuple[float, float]]],
        obj_bound_points_map: dict[str, list[tuple[float, float]]],
    ) -> pd.DataFrame:
        if base_df.empty:
            empty_df = base_df.copy()
            empty_df["targetSec"] = pd.Series(dtype=float)
            empty_df["timepointLabel"] = pd.Series(dtype=str)
            return empty_df

        summary_df = base_df.copy()
        if exclude_if_timelimit_lt is not None:
            summary_df = summary_df[
                summary_df[INPUT_TIMELIMIT_COLUMN].astype(float)
                >= float(exclude_if_timelimit_lt)
            ].copy()

        for idx, row in summary_df.iterrows():
            ins_name = row.get(SubroutineReportStatisticsKeys.INSTANCE_NAME)
            timelimit = row.get(INPUT_TIMELIMIT_COLUMN)
            if pd.isna(timelimit):
                continue

            if mode == "timelimit_ratio":
                target_sec = float(timelimit) * value
            else:
                target_sec = value

            obj_points = HfsMultiInstanceRunner._get_instance_points(
                obj_value_points_map, ins_name
            )
            bound_points = HfsMultiInstanceRunner._get_instance_points(
                obj_bound_points_map, ins_name
            )
            sampled_obj = HfsMultiInstanceRunner._sample_last_known_value(
                obj_points, target_sec
            )
            sampled_bound = HfsMultiInstanceRunner._sample_last_known_value(
                bound_points, target_sec
            )

            if sampled_obj is not None:
                summary_df.at[idx, SubroutineReportStatisticsKeys.BEST_OBJ] = (
                    sampled_obj
                )
            if sampled_bound is not None:
                summary_df.at[idx, SubroutineReportStatisticsKeys.BEST_BOUND] = (
                    sampled_bound
                )

            original_elapsed = row.get(
                SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME
            )
            if pd.notna(original_elapsed):
                summary_df.at[
                    idx, SubroutineReportStatisticsKeys.TOTAL_ELAPSED_TIME
                ] = min(float(original_elapsed), target_sec)

            summary_df.at[idx, "targetSec"] = target_sec
            summary_df.at[idx, "timepointLabel"] = label

        return summary_df

    def _get_summary_csv_path(self) -> Path:
        """Get the path to multi_instance_summary.csv."""
        return self.working_dir / "multi_instance_summary.csv"

    def append_result(self, result: dict) -> None:
        """Append a result to self.results and write to multi_instance_summary.csv.

        Args:
            result: Summary dict from a completed instance run.
        """
        # Add to results list
        self.results.append(result)

        # Write to CSV file (append mode)
        csv_path = self._get_summary_csv_path()

        # Convert result to DataFrame and append
        df = pd.DataFrame([result])

        if not csv_path.exists():
            # First result - write with header
            df.to_csv(csv_path, index=False)
            logging.info(f"Created multi-instance summary at {csv_path}")
        else:
            # Append without header
            df.to_csv(csv_path, mode="a", index=False, header=False)

        logging.info(
            f"Appended result for instance '{result.get(SubroutineReportStatisticsKeys.INSTANCE_NAME, 'unknown')}' "
            f"to multi_instance_summary.csv ({len(self.results)} total)"
        )

    def _create_rpd_summary(self) -> pd.DataFrame | None:
        """
        Creates a summary with normalized time and RPD metrics.

        This method:
        1. Reads the baseline CSV for reference values (assumes baseline_df is already loaded)
        2. Reads the method-level summary CSV (summary_method_end_time_and_obj_value_long.csv)
        3. Computes RPDf and RPDv metrics for each method's objective value
        4. Computes normalized time (end_time / timelimit) from runners
        5. Outputs Long format (summary_method_rpdf_and_norm_time_long.csv)
        6. Outputs Wide format (summary_method_rpdf_and_norm_time_wide.csv)

        Returns:
            pd.DataFrame | None: Combined summary with RPD metrics and normalized time in Long format,
                                or None if baseline_df is not available.
        """
        # 1. Load baseline reference values
        if not hasattr(self, "baseline_df") or self.baseline_df.empty:
            logging.warning(
                "Baseline DataFrame not available. Skipping metrics calculation."
            )
            return None

        # Get baseline column names
        instance_col = getattr(self, "baseline_instance_col", "Instance")
        job_cnt_col = getattr(self, "baseline_job_cnt_col", "n")
        stage_cnt_col = getattr(self, "baseline_stage_cnt_col", "s")
        obj_val_col = getattr(self, "baseline_obj_val_col", "UB")

        # Build reference dict: instance_name -> reference_value
        ref_job_cnt_dict = {}
        ref_stage_cnt_dict = {}
        ref_obj_val_dict = {}
        for _, row in self.baseline_df.iterrows():
            ref_name = str(row[instance_col])
            ref_job_cnt_dict[ref_name] = row[job_cnt_col]
            ref_stage_cnt_dict[ref_name] = row[stage_cnt_col]
            ref_obj_val_dict[ref_name] = row[obj_val_col]

        logging.info(f"Loaded {len(ref_obj_val_dict)} reference values from baseline.")

        # 2. Read the method-level summary CSV (Long format)
        method_summary_path = (
            self.working_dir / "summary_method_end_time_and_obj_value_long.csv"
        )
        if not method_summary_path.exists():
            logging.warning(
                f"Method summary file not found: {method_summary_path}. Skipping metrics calculation."
            )
            return None

        method_df = pd.read_csv(method_summary_path)
        logging.info(f"Loaded method summary with {len(method_df)} rows.")

        # 3. Process each row (Long format: one row per method-instance)
        result_rows = []
        for _, row in method_df.iterrows():
            instance_id = str(row["instance_id"])
            subroutine_name = row["subroutine_name"]

            # Get timelimit from corresponding runner
            timelimit = None
            for runner in self.runners:
                if hasattr(runner, "name") and str(runner.name) == instance_id:
                    timelimit = runner.stopping_criteria.timelimit
                    break

            if timelimit is None:
                logging.warning(
                    f"Timelimit not found for instance {instance_id}, skipping."
                )
                continue

            end_time = row.get("end_time")
            if pd.isna(end_time):
                continue

            # Normalized time
            norm_time = end_time / timelimit if timelimit > 0 else None

            # Compute RPDf and RPDv
            obj_val = row.get("obj_value")
            ref_val = ref_obj_val_dict.get(instance_id)

            rpd_f = None
            rpd_v = None

            if ref_val is not None and pd.notna(obj_val):
                # RPDf = (obj - ref) / ((obj + ref) / 2)
                rpd_f = compute_rpdf(obj_val, ref_val)
                # RPDv = (obj - ref) / ref
                rpd_v = (obj_val - ref_val) / ref_val if ref_val != 0 else None

            result_row = {
                "instance_id": instance_id,
                "subroutine_name": subroutine_name,
                "norm_time": norm_time,
                "rpd_f": rpd_f,
                "rpd_v": rpd_v,
            }
            result_rows.append(result_row)

        # 4. Create output Long format DataFrame
        if not result_rows:
            logging.warning("No results to write for metrics summary.")
            return None

        metrics_long_df = pd.DataFrame(result_rows)

        # 5. Save Long format to file
        output_long_path = (
            self.working_dir / "summary_method_rpdf_and_norm_time_long.csv"
        )
        metrics_long_df.to_csv(output_long_path, index=False)
        logging.info(f"Metrics summary (long) saved to {output_long_path}")

        # 6. Create Wide format DataFrame
        wide_rows = []
        instance_ids = sorted(set(metrics_long_df["instance_id"]))
        for instance_id in instance_ids:
            row = {"instance_id": instance_id}
            instance_df = metrics_long_df[metrics_long_df["instance_id"] == instance_id]

            for _, r in instance_df.iterrows():
                m_name = r["subroutine_name"]
                row[f"{m_name}_norm_time"] = r["norm_time"]
                row[f"{m_name}_rpd_f"] = r["rpd_f"]
                row[f"{m_name}_rpd_v"] = r["rpd_v"]

            wide_rows.append(row)

        metrics_wide_df = pd.DataFrame(wide_rows)

        # Order columns: instance_id, then method metrics
        wide_cols = ["instance_id"]
        methods_in_order = list(method_df["subroutine_name"].unique())
        for m_name in methods_in_order:
            wide_cols.append(f"{m_name}_norm_time")
            wide_cols.append(f"{m_name}_rpd_f")
            wide_cols.append(f"{m_name}_rpd_v")

        existing_wide_cols = [c for c in wide_cols if c in metrics_wide_df.columns]
        metrics_wide_df = metrics_wide_df.reindex(columns=existing_wide_cols)

        # 7. Save Wide format to file
        output_wide_path = (
            self.working_dir / "summary_method_rpdf_and_norm_time_wide.csv"
        )
        metrics_wide_df.to_csv(output_wide_path, index=False)
        logging.info(f"Metrics summary (wide) saved to {output_wide_path}")

        return metrics_long_df

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
