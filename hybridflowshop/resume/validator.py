from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from mbls.cpsat import ObjValueBoundStore
from routix.type_defs import RunMode
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.io_solution import get_end_time_dict, get_start_time_dict


@dataclass(slots=True)
class ResumeValidationData:
    start_time_map_by_instance: dict[str, dict[Any, Any]] = field(default_factory=dict)
    end_time_map_by_instance: dict[str, dict[Any, Any]] = field(default_factory=dict)
    obj_value_by_instance: dict[str, float] = field(default_factory=dict)
    obj_store_by_instance: dict[str, ObjValueBoundStore[float]] = field(
        default_factory=dict
    )
    summary_by_instance: dict[str, dict[str, Any]] = field(default_factory=dict)


class ResumeValidator:
    def __init__(
        self,
        instances: Sequence[HybridFlowshopParameters],
        shared_param_dict: dict[str, Any],
        subroutine_flow: Any,
        stopping_criteria: Any,
        output_dir: Path,
        output_metadata: dict[str, Any],
        mode: RunMode,
        runner_class: type[HfsSingleInstanceRunner] = HfsSingleInstanceRunner,
    ):
        self.instances = instances
        self.shared_param_dict = shared_param_dict
        self.subroutine_flow = subroutine_flow
        self.stopping_criteria = stopping_criteria
        self.output_dir = output_dir
        self.output_metadata = output_metadata
        self.mode = mode
        self.runner_class = runner_class
        self.data = ResumeValidationData()
        self._has_feasibility_data = False
        self._has_obj_store_data = False
        self._has_summary_data = False

    def load_resume_data(self) -> ResumeValidationData:
        self.load_resume_solution_check_feasibility()
        self.load_obj_store_check_resume_solution_obj_value()
        self.load_summary_check_obj_values()
        return self.data

    def load_resume_solution_check_feasibility(self) -> None:
        solution_fn_format = self.output_metadata.get(
            "solution_fn_format", "{}_solution.yaml"
        )
        encoding = str(self.output_metadata.get("encoding", "utf-8"))

        self.data.start_time_map_by_instance = {}
        self.data.end_time_map_by_instance = {}
        self.data.obj_value_by_instance = {}
        self.data.obj_store_by_instance = {}
        self.data.summary_by_instance = {}
        self._has_feasibility_data = False
        self._has_obj_store_data = False
        self._has_summary_data = False
        infeasible_instances: list[str] = []

        for ins in self.instances:
            ins_name = self._get_instance_name(ins)
            inst_dir = self._get_instance_resume_dir(ins_name)
            sol_path = self._find_expected_file(
                inst_dir,
                solution_fn_format,
                ins_name,
                file_label="Solution file",
            )

            temp_runner = self.runner_class(
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
                start_time_map = get_start_time_dict(sol_path, encoding=encoding)
                obj_val = temp_controller.check_feasibility(start_time_map)
                end_time_map = get_end_time_dict(sol_path, encoding=encoding)
                self.data.start_time_map_by_instance[ins_name] = start_time_map
                self.data.end_time_map_by_instance[ins_name] = end_time_map
                self.data.obj_value_by_instance[ins_name] = obj_val
            except RuntimeError:
                infeasible_instances.append(ins_name)
            except Exception as e:
                raise RuntimeError(
                    f"Error checking feasibility for instance '{ins_name}' with solution file '{sol_path}': {e}"
                ) from e

        if infeasible_instances:
            raise ValueError(
                "The following instances have infeasible resume solutions: "
                f"{infeasible_instances}"
            )
        self._has_feasibility_data = True

    def load_obj_store_check_resume_solution_obj_value(self) -> None:
        if not self._has_feasibility_data:
            raise RuntimeError(
                "Resume solution feasibility has not been checked. Call _check_resume_solution_feasibility() first."
            )

        obj_log_fn_format = self.output_metadata.get(
            "obj_log_fn_format", "{}_obj_log.yaml"
        )
        self.data.obj_store_by_instance = {}
        self._has_obj_store_data = False
        self._has_summary_data = False

        for ins in self.instances:
            ins_name = self._get_instance_name(ins)
            if ins_name not in self.data.obj_value_by_instance:
                raise ValueError(
                    f"Objective value for instance '{ins_name}' not found in resume data."
                )

            inst_dir = self._get_instance_resume_dir(ins_name)
            obj_log_path = self._find_expected_file(
                inst_dir,
                obj_log_fn_format,
                ins_name,
                file_label="Objective log file",
            )

            try:
                resume_obj_store = ObjValueBoundStore.load_yaml(obj_log_path)
                resume_obj_value = resume_obj_store.get_last_obj_value()
                if resume_obj_value is None:
                    raise ValueError(
                        f"No objective value found in obj log for instance '{ins_name}'"
                    )

                sol_obj_value = self.data.obj_value_by_instance[ins_name]
                if abs(float(resume_obj_value) - float(sol_obj_value)) > 1e-6:
                    raise ValueError(
                        f"Objective value mismatch for instance '{ins_name}': "
                        f"resume obj log value {resume_obj_value} vs "
                        f"recorded solution obj value {sol_obj_value}"
                    )

                self.data.obj_store_by_instance[ins_name] = resume_obj_store
            except Exception as e:
                raise RuntimeError(
                    f"Error checking objective value for instance '{ins_name}' with obj log file '{obj_log_path}': {e}"
                ) from e
        self._has_obj_store_data = True

    def load_summary_check_obj_values(self) -> None:
        if not self._has_feasibility_data:
            raise RuntimeError(
                "Resume solution feasibility has not been checked. Call _check_resume_solution_feasibility() first."
            )
        if not self._has_obj_store_data:
            raise RuntimeError(
                "Resume solution objective log has not been checked. Call _load_obj_store_check_resume_solution_obj_value() first."
            )

        summary_fn_format = self.output_metadata.get(
            "summary_fn_format", "{}_summary.csv"
        )
        self.data.summary_by_instance = {}
        self._has_summary_data = False

        for ins in self.instances:
            ins_name = self._get_instance_name(ins)
            if ins_name not in self.data.obj_value_by_instance:
                raise ValueError(
                    f"Objective value for instance '{ins_name}' not found in resume data."
                )

            inst_dir = self._get_instance_resume_dir(ins_name)
            summary_path = self._find_expected_file(
                inst_dir,
                summary_fn_format,
                ins_name,
                file_label="Summary file",
            )

            try:
                df = pd.read_csv(summary_path)
                if "bestObj" not in df.columns:
                    raise ValueError(
                        f"'bestObj' column not found in summary file for instance '{ins_name}'"
                    )
                if df.empty:
                    raise ValueError(f"Summary file for instance '{ins_name}' is empty")

                summary_dict = {
                    str(key): value for key, value in df.iloc[-1].to_dict().items()
                }
                summary_obj_value = summary_dict["bestObj"]
                sol_obj_value = self.data.obj_value_by_instance[ins_name]
                if abs(float(summary_obj_value) - float(sol_obj_value)) > 1e-6:
                    raise ValueError(
                        f"Objective value mismatch for instance '{ins_name}': "
                        f"summary bestObj value {summary_obj_value} vs "
                        f"recorded solution obj value {sol_obj_value}"
                    )

                self.data.summary_by_instance[ins_name] = summary_dict
            except Exception as e:
                raise RuntimeError(
                    f"Error checking objective value for instance '{ins_name}' with summary file '{summary_path}': {e}"
                ) from e
        self._has_summary_data = True

    def inject_resume_data_into_runners(
        self, runners: Sequence[HfsSingleInstanceRunner]
    ) -> None:
        if not self._has_feasibility_data or not self._has_obj_store_data:
            raise RuntimeError(
                "Resume solution feasibility and objective store have not been loaded. Call the respective methods first."
            )
        if not self._has_summary_data:
            raise RuntimeError(
                "Resume solution summary has not been loaded. Call _load_summary_check_obj_values() first."
            )

        for index, ins in enumerate(self.instances):
            ins_name = self._get_instance_name(ins)
            runner = runners[index]
            runner.resume_start_time_map = self.data.start_time_map_by_instance[
                ins_name
            ]
            runner.resume_end_time_map = self.data.end_time_map_by_instance[ins_name]
            runner.resume_obj_store = self.data.obj_store_by_instance[ins_name]
            runner.resume_summary_dict = self.data.summary_by_instance[ins_name]

    def _get_resume_root(self) -> Path:
        if "resume_root" not in self.output_metadata:
            raise ValueError("Missing 'resume_root' in output_metadata")
        return Path(self.output_metadata["resume_root"])

    def _get_instance_resume_dir(self, ins_name: str) -> Path:
        resume_dir = self._get_resume_root()
        result_dir_name = str(self.output_metadata.get("result_dir_name", "results"))
        inst_dir = resume_dir / ins_name / result_dir_name
        if not inst_dir.exists():
            inst_dir = resume_dir / ins_name
        return inst_dir

    @staticmethod
    def _get_instance_name(ins: HybridFlowshopParameters) -> str:
        return (
            getattr(ins, "name", None)
            or getattr(ins, "instance_name", None)
            or str(ins)
        )

    @staticmethod
    def _find_expected_file(
        inst_dir: Path,
        file_name_format: str,
        ins_name: str,
        file_label: str,
    ) -> Path:
        file_pattern = file_name_format.format(ins_name)
        candidate_paths = list(inst_dir.glob(file_pattern)) if inst_dir.exists() else []
        if candidate_paths:
            return candidate_paths[0]
        raise ValueError(
            f"{file_label} not found for instance '{ins_name}' at expected location: {inst_dir / file_pattern}"
        )
