import ast
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

from output_filenames import OutputFilenames

DEFAULT_CONTROLLER_LOG_NAME = "subroutine_controller.log"
DEFAULT_RESULTS_DIR = "results"
OBJ_LOG_FN_FORMAT = OutputFilenames.OBJ_LOG_FN_FORMAT


def parse_controller_log(log_path: Path) -> list[dict]:
    records = []
    pattern = re.compile(r"INFO - ({'method':.*})")

    if not log_path.exists():
        logging.warning(f"Log file not found: {log_path}")
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    data = ast.literal_eval(match.group(1))
                    if (
                        isinstance(data, dict)
                        and "method" in data
                        and "start_sec" in data
                        and "elapsed_sec" in data
                    ):
                        data["start_sec"] = float(data["start_sec"])
                        data["elapsed_sec"] = float(data["elapsed_sec"])
                        data["end_sec"] = data["start_sec"] + data["elapsed_sec"]
                        records.append(data)
                except (ValueError, SyntaxError, TypeError):
                    continue
    return records


def _find_progression_json_path(instance_dir: Path) -> Path | None:
    results_json_path = instance_dir / DEFAULT_RESULTS_DIR / "subroutine_progression.json"
    if results_json_path.exists():
        return results_json_path

    root_json_path = instance_dir / "subroutine_progression.json"
    if root_json_path.exists():
        return root_json_path

    return None


def _coerce_progression_time(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_progression_time_records(instance_dir: Path) -> list[dict]:
    json_path = _find_progression_json_path(instance_dir)
    if json_path is None:
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            progression_data = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read progression JSON {json_path}: {e}")
        return []

    subroutine_calls = progression_data.get("subroutine_calls", [])
    if not isinstance(subroutine_calls, list):
        return []

    records: list[dict] = []
    sorted_calls = sorted(
        subroutine_calls,
        key=lambda call: int(call.get("call_index", -1)),
    )
    for call in sorted_calls:
        if not isinstance(call, dict):
            continue

        method_name = call.get("subroutine_name")
        call_context = call.get("prefixed_subroutine_name")
        if not method_name or not call_context:
            continue

        start_sec = _coerce_progression_time(call.get("global_start_sec"))
        elapsed_sec = _coerce_progression_time(call.get("elapsed_sec"))
        end_sec = _coerce_progression_time(call.get("global_end_sec"))

        if start_sec is None and end_sec is not None and elapsed_sec is not None:
            start_sec = end_sec - elapsed_sec
        if end_sec is None and start_sec is not None and elapsed_sec is not None:
            end_sec = start_sec + elapsed_sec
        if elapsed_sec is None and start_sec is not None and end_sec is not None:
            elapsed_sec = end_sec - start_sec

        if start_sec is None or elapsed_sec is None or end_sec is None:
            continue

        records.append(
            {
                "method": str(method_name),
                "call_context": str(call_context),
                "start_sec": start_sec,
                "elapsed_sec": elapsed_sec,
                "end_sec": end_sec,
            }
        )

    return records


def parse_obj_log(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        logging.warning(f"Obj log file not found: {yaml_path}")
        return {"data": {}, "notes": {}}

    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            content = yaml.safe_load(f)
            if "obj_value" in content:
                return content["obj_value"]
            return {"data": {}, "notes": {}}
        except Exception as e:
            logging.warning(f"Failed to read yaml {yaml_path}: {e}")
            return {"data": {}, "notes": {}}


def get_obj_value_for_method(
    method_prefix: str,
    method_name: str,
    obj_data: dict,
    obj_notes: dict,
    prev_obj_value,
):
    relevant_times = []
    for time_str, note in obj_notes.items():
        if note.startswith(method_prefix):
            relevant_times.append(time_str)

    if not relevant_times:
        return None

    try:
        max_time_str = max(relevant_times, key=float)
    except ValueError:
        max_time_str = max(relevant_times)

    val = obj_data.get(max_time_str)

    if _is_missing_obj_value(val):
        return prev_obj_value

    return val


def _is_missing_obj_value(val):
    if val is None:
        return True
    if isinstance(val, float) and (pd.isna(val) or val == float("nan")):
        return True
    s_val = str(val).strip()
    if s_val.lower() == "nan" or s_val == "":
        return True
    return False


def get_methods_from_flow(scenario_dir: Path) -> list[tuple[str, str]]:
    flow_path = scenario_dir / "subroutine_flow.yaml"
    if not flow_path.exists():
        logging.warning(f"subroutine_flow.yaml not found in {scenario_dir}.")
        return []

    try:
        with open(flow_path, "r", encoding="utf-8") as f:
            flow = yaml.safe_load(f)
            if isinstance(flow, list):
                methods = []
                for i, item in enumerate(flow):
                    prefix = f"{i + 1}-"
                    m_name = item.get("method", "unknown")
                    methods.append((prefix, m_name))
                return methods
    except Exception as e:
        logging.error(f"Error parsing flow {flow_path}: {e}")
        return []
    return []


def _build_instance_method_rows(
    methods_list: list[tuple[str, str]],
    method_end_times: dict[str, float],
    obj_data: dict,
    obj_notes: dict,
    record_all_subroutines: bool = False,
) -> list[dict]:
    note_values = list(obj_notes.values())
    method_has_note = {
        method_prefix: any(note.startswith(method_prefix) for note in note_values)
        for method_prefix, _ in methods_list
    }

    rows = []
    current_obj_value = None

    for i, (method_prefix, method_name) in enumerate(methods_list):
        end_sec = method_end_times.get(method_prefix)
        in_notes = method_has_note[method_prefix]

        if in_notes:
            obj_val = get_obj_value_for_method(
                method_prefix, method_name, obj_data, obj_notes, current_obj_value
            )
            final_end_sec = end_sec
            final_obj_val = obj_val
            effective_obj_value = obj_val
        else:
            successor_in_notes = any(
                method_has_note[succ_prefix] for succ_prefix, _ in methods_list[i + 1 :]
            )

            final_end_sec = end_sec
            if not successor_in_notes:
                final_obj_val = None
            else:
                final_obj_val = current_obj_value
            effective_obj_value = current_obj_value

        if not _is_missing_obj_value(effective_obj_value):
            current_obj_value = effective_obj_value

        rows.append(
            {
                "method_prefix": method_prefix,
                "method_name": method_name,
                "method_end_sec": final_end_sec,
                "objective_value": final_obj_val,
                "executed": end_sec is not None,
                "effective_obj_value": current_obj_value,
            }
        )

    if record_all_subroutines:
        executed_indices = [idx for idx, row in enumerate(rows) if row["executed"]]
        if executed_indices:
            last_executed_idx = executed_indices[-1]
            fill_end_sec = rows[last_executed_idx]["method_end_sec"]
            fill_obj_val = rows[last_executed_idx]["effective_obj_value"]

            for idx in range(last_executed_idx + 1, len(rows)):
                if rows[idx]["executed"]:
                    continue
                rows[idx]["method_end_sec"] = fill_end_sec
                rows[idx]["objective_value"] = fill_obj_val

    return rows


def process_instance(
    instance_dir: Path,
    methods_list: list[tuple[str, str]],
    record_all_subroutines: bool = False,
    omitted_subroutines: set[str] | None = None,
):
    instance_id = instance_dir.name

    log_path = instance_dir / DEFAULT_CONTROLLER_LOG_NAME
    time_records = parse_controller_log(log_path)
    if not time_records:
        time_records = parse_progression_time_records(instance_dir)
        if time_records:
            logging.info(
                "Recovered %d time records for %s from subroutine_progression.json.",
                len(time_records),
                instance_id,
            )

    with open(instance_dir / "method_time_log.json", "w", encoding="utf-8") as f:
        json.dump(time_records, f, indent=2)

    method_end_times = {}
    for r in time_records:
        if "call_context" in r:
            ctx = r["call_context"]
            for prefix, _ in methods_list:
                if ctx.startswith(prefix):
                    method_end_times[prefix] = r["end_sec"]
                    break

    obj_log_path = (
        instance_dir / DEFAULT_RESULTS_DIR / OBJ_LOG_FN_FORMAT.format(instance_id)
    )
    if not obj_log_path.exists():
        obj_log_path = instance_dir / OBJ_LOG_FN_FORMAT.format(instance_id)

    obj_content = parse_obj_log(obj_log_path)
    obj_data = obj_content.get("data", {})
    obj_notes = obj_content.get("notes", {})

    omitted = omitted_subroutines or set()
    rows = _build_instance_method_rows(
        methods_list=methods_list,
        method_end_times=method_end_times,
        obj_data=obj_data,
        obj_notes=obj_notes,
        record_all_subroutines=record_all_subroutines,
    )
    csv_rows = [
        {
            "method_name": row["method_name"],
            "method_end_sec": row["method_end_sec"],
            "objective_value": row["objective_value"],
        }
        for row in rows
        if row["method_name"] not in omitted
    ]

    df = pd.DataFrame(
        csv_rows,
        columns=["method_name", "method_end_sec", "objective_value"],
    )
    df.to_csv(instance_dir / "method_end_time_and_obj_value.csv", index=False)
    return df


class LogProcessor:
    def __init__(
        self,
        scenario_dir: Path,
        baseline_df: pd.DataFrame | None = None,
        baseline_instance_col: str = "Instance",
        baseline_job_cnt_col: str = "n",
        baseline_stage_cnt_col: str = "s",
        baseline_obj_val_col: str = "UB",
        record_all_subroutines: bool = False,
        omitted_subroutines: set[str] | None = None,
    ):
        self._scenario_dir = scenario_dir
        self._baseline_df = baseline_df
        self._baseline_instance_col = baseline_instance_col
        self._baseline_job_cnt_col = baseline_job_cnt_col
        self._baseline_stage_cnt_col = baseline_stage_cnt_col
        self._baseline_obj_val_col = baseline_obj_val_col
        self._record_all_subroutines = record_all_subroutines
        self._omitted_subroutines = omitted_subroutines or set()

    def create_method_end_time_and_obj_value_summary(
        self,
    ) -> pd.DataFrame | None:
        scenario_dir = self._scenario_dir
        if not scenario_dir.exists():
            logging.warning(f"Scenario directory {scenario_dir} does not exist.")
            return None

        logging.info(f"Processing logs for scenario: {scenario_dir.name}")

        methods_list = get_methods_from_flow(scenario_dir)
        if not methods_list:
            logging.warning("No methods found in flow. Skipping summary generation.")
            return None

        output_methods_list = [
            (method_prefix, method_name)
            for method_prefix, method_name in methods_list
            if method_name not in self._omitted_subroutines
        ]

        ref_job_cnt_dict = {}
        ref_stage_cnt_dict = {}
        ref_obj_val_dict = {}
        if self._baseline_df is not None and not self._baseline_df.empty:
            for _, row in self._baseline_df.iterrows():
                ref_name = str(row[self._baseline_instance_col])
                ref_job_cnt_dict[ref_name] = row[self._baseline_job_cnt_col]
                ref_stage_cnt_dict[ref_name] = row[self._baseline_stage_cnt_col]
                ref_obj_val_dict[ref_name] = row[self._baseline_obj_val_col]
            logging.info(
                f"Loaded {len(ref_job_cnt_dict)} instance metadata from baseline."
            )

        summary_rows = []

        instance_dirs = [
            p for p in scenario_dir.iterdir() if p.is_dir() and p.name.isdigit()
        ]
        instance_dirs.sort(key=lambda p: int(p.name))

        for instance_dir in instance_dirs:
            try:
                df = process_instance(
                    instance_dir,
                    methods_list,
                    record_all_subroutines=self._record_all_subroutines,
                    omitted_subroutines=self._omitted_subroutines,
                )

                instance_id = int(instance_dir.name)
                instance_name = str(instance_id)
                job_cnt = ref_job_cnt_dict.get(instance_name)
                stage_cnt = ref_stage_cnt_dict.get(instance_name)
                obj_val = ref_obj_val_dict.get(instance_name)

                for _, r in df.iterrows():
                    row = {
                        "instance_id": instance_id,
                        "subroutine_name": r["method_name"],
                        "end_time": r["method_end_sec"],
                        "obj_value": r["objective_value"],
                    }
                    if job_cnt is not None:
                        row["job_cnt"] = job_cnt
                    if stage_cnt is not None:
                        row["stage_cnt"] = stage_cnt
                    if obj_val is not None:
                        row["ref_obj_value"] = obj_val

                    summary_rows.append(row)
            except Exception as e:
                logging.error(f"Failed to process {instance_dir.name}: {e}")

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            cols = [
                "instance_id",
                "job_cnt",
                "stage_cnt",
                "ref_obj_value",
                "subroutine_name",
                "end_time",
                "obj_value",
            ]
            existing_cols = [c for c in cols if c in summary_df.columns]
            summary_df = summary_df.reindex(columns=existing_cols)

            long_path = scenario_dir / "summary_method_end_time_and_obj_value_long.csv"
            summary_df.to_csv(long_path, index=False)
            logging.info(
                f"Method end time and obj value summary (long) saved to: {long_path}"
            )

            wide_rows = []
            for instance_dir in instance_dirs:
                try:
                    instance_id = int(instance_dir.name)
                    instance_name = str(instance_id)
                    job_cnt = ref_job_cnt_dict.get(instance_name)
                    stage_cnt = ref_stage_cnt_dict.get(instance_name)
                    obj_val = ref_obj_val_dict.get(instance_name)

                    row = {
                        "instance_id": instance_id,
                    }
                    if job_cnt is not None:
                        row["job_cnt"] = job_cnt
                    if stage_cnt is not None:
                        row["stage_cnt"] = stage_cnt
                    if obj_val is not None:
                        row["ref_obj_value"] = obj_val

                    instance_df = summary_df[summary_df["instance_id"] == instance_id]
                    for _, r in instance_df.iterrows():
                        m_name = r["subroutine_name"]
                        row[f"{m_name}_end_time"] = r["end_time"]
                        row[f"{m_name}_obj_value"] = r["obj_value"]

                    wide_rows.append(row)
                except Exception as e:
                    logging.error(
                        f"Failed to create wide format for {instance_dir.name}: {e}"
                    )

            wide_df = pd.DataFrame(wide_rows)
            wide_cols = ["instance_id", "job_cnt", "stage_cnt", "ref_obj_value"]
            for _, m_name in output_methods_list:
                wide_cols.append(f"{m_name}_end_time")
                wide_cols.append(f"{m_name}_obj_value")
            existing_wide_cols = [c for c in wide_cols if c in wide_df.columns]
            wide_df = wide_df.reindex(columns=existing_wide_cols)

            wide_path = scenario_dir / "summary_method_end_time_and_obj_value_wide.csv"
            wide_df.to_csv(wide_path, index=False)
            logging.info(
                f"Method end time and obj value summary (wide) saved to: {wide_path}"
            )

            return summary_df
        return None


def create_method_end_time_and_obj_value_summary(
    scenario_dir: Path,
    baseline_df: pd.DataFrame | None = None,
    baseline_instance_col: str = "Instance",
    baseline_job_cnt_col: str = "n",
    baseline_stage_cnt_col: str = "s",
    baseline_obj_val_col: str = "UB",
    record_all_subroutines: bool = False,
    omitted_subroutines: set[str] | None = None,
) -> pd.DataFrame | None:
    processor = LogProcessor(
        scenario_dir=scenario_dir,
        baseline_df=baseline_df,
        baseline_instance_col=baseline_instance_col,
        baseline_job_cnt_col=baseline_job_cnt_col,
        baseline_stage_cnt_col=baseline_stage_cnt_col,
        baseline_obj_val_col=baseline_obj_val_col,
        record_all_subroutines=record_all_subroutines,
        omitted_subroutines=omitted_subroutines,
    )
    return processor.create_method_end_time_and_obj_value_summary()


def process_scenario(scenario_dir: Path):
    summary_df = create_method_end_time_and_obj_value_summary(scenario_dir)
    if summary_df is not None:
        out_path = scenario_dir / "summary_method_end_time_and_obj_value.csv"
        summary_df.to_csv(out_path, index=False)
        logging.info(f"Summary saved to: {out_path}")
