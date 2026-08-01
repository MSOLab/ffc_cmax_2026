from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from exp_compare.metrics import compute_rpdf

from .log_processor import get_methods_from_flow


def _find_progression_json_path(instance_dir: Path) -> Path | None:
    json_path = instance_dir / "results" / "subroutine_progression.json"
    if json_path.exists():
        return json_path

    json_path = instance_dir / "subroutine_progression.json"
    if json_path.exists():
        return json_path

    return None


def _iter_instance_progression_json_paths(
    working_dir: Path,
) -> list[tuple[Path, Path]]:
    instance_json_paths: list[tuple[Path, Path]] = []

    for item in sorted(working_dir.iterdir(), key=lambda p: p.name):
        if not item.is_dir():
            continue
        json_path = _find_progression_json_path(item)
        if json_path is not None:
            instance_json_paths.append((item, json_path))

    return instance_json_paths


def _coerce_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing_obj_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _get_call_endpoint_obj_value(
    call_data: dict[str, Any],
    previous_effective_obj_value: float | None,
) -> float | None:
    local_progress_list = call_data.get("local_progress_list", [])
    if not local_progress_list:
        return previous_effective_obj_value

    endpoint_obj_value = local_progress_list[-1].get("obj_value")
    if _is_missing_obj_value(endpoint_obj_value):
        return previous_effective_obj_value

    return _coerce_float(endpoint_obj_value)


def build_instance_endpoint_rows_from_progression(
    progression_data: dict[str, Any],
    methods_list: list[tuple[str, str]],
    record_all_subroutines: bool = False,
    omitted_subroutines: set[str] | None = None,
) -> list[dict[str, Any]] | None:
    subroutine_calls = sorted(
        progression_data.get("subroutine_calls", []),
        key=lambda call: int(call.get("call_index", -1)),
    )
    call_idx = 0
    current_obj_value: float | None = None
    last_executed_idx: int | None = None
    rows: list[dict[str, Any]] = []

    for flow_idx, (_, method_name) in enumerate(methods_list):
        matched_call = None
        if call_idx < len(subroutine_calls):
            candidate_call = subroutine_calls[call_idx]
            if candidate_call.get("subroutine_name") == method_name:
                matched_call = candidate_call
                call_idx += 1

        if matched_call is None:
            rows.append(
                {
                    "subroutine_name": method_name,
                    "end_time": None,
                    "obj_value": None,
                    "executed": False,
                    "effective_obj_value": current_obj_value,
                }
            )
            continue

        end_time = _coerce_float(matched_call.get("global_end_sec"))
        endpoint_obj_value = _get_call_endpoint_obj_value(
            matched_call, current_obj_value
        )
        if not _is_missing_obj_value(endpoint_obj_value):
            current_obj_value = endpoint_obj_value

        rows.append(
            {
                "subroutine_name": method_name,
                "end_time": end_time,
                "obj_value": endpoint_obj_value,
                "executed": end_time is not None,
                "effective_obj_value": current_obj_value,
            }
        )
        if end_time is not None:
            last_executed_idx = flow_idx

    if call_idx != len(subroutine_calls):
        instance_id = progression_data.get("instance_id", "")
        unmatched_calls = [
            str(call.get("subroutine_name", "")) for call in subroutine_calls[call_idx:]
        ]
        logging.warning(
            "Failed to align progression JSON to subroutine_flow for instance %s; "
            "unmatched calls remain: %s",
            instance_id,
            unmatched_calls,
        )
        return None

    if record_all_subroutines and last_executed_idx is not None:
        fill_end_time = rows[last_executed_idx]["end_time"]
        fill_obj_value = rows[last_executed_idx]["effective_obj_value"]
        for idx in range(last_executed_idx + 1, len(rows)):
            if rows[idx]["executed"]:
                continue
            rows[idx]["end_time"] = fill_end_time
            rows[idx]["obj_value"] = fill_obj_value

    omitted = omitted_subroutines or set()
    return [row for row in rows if row["subroutine_name"] not in omitted]


def aggregate_scenario_endpoint_metrics_from_json(
    working_dir: Path,
    baseline_df: pd.DataFrame | None = None,
    baseline_instance_col: str = "Instance",
    baseline_obj_val_col: str = "UB",
    record_all_subroutines: bool = False,
    omitted_subroutines: set[str] | None = None,
) -> pd.DataFrame:
    methods_list = get_methods_from_flow(working_dir)
    if not methods_list:
        return pd.DataFrame()

    instance_json_paths = _iter_instance_progression_json_paths(working_dir)
    if not instance_json_paths:
        logging.warning(f"No subroutine_progression.json found in {working_dir}")
        return pd.DataFrame()

    ref_obj_map: dict[str, float] = {}
    if baseline_df is not None and not baseline_df.empty:
        for _, row in baseline_df.iterrows():
            ref_obj_map[str(row[baseline_instance_col])] = row[baseline_obj_val_col]

    result_rows: list[dict[str, Any]] = []

    for instance_dir, json_path in instance_json_paths:
        progression_data = load_instance_progression_json(json_path)
        if progression_data is None:
            continue

        endpoint_rows = build_instance_endpoint_rows_from_progression(
            progression_data=progression_data,
            methods_list=methods_list,
            record_all_subroutines=record_all_subroutines,
            omitted_subroutines=omitted_subroutines,
        )
        if endpoint_rows is None:
            continue

        instance_id = str(progression_data.get("instance_id") or instance_dir.name)
        timelimit = _coerce_float(progression_data.get("timelimit_sec"))
        ref_obj_value = _coerce_float(ref_obj_map.get(instance_id))

        for row in endpoint_rows:
            end_time = _coerce_float(row.get("end_time"))
            norm_time = None
            if timelimit is not None and timelimit > 0 and end_time is not None:
                norm_time = end_time / timelimit

            obj_value = _coerce_float(row.get("obj_value"))
            rpd_f = None
            rpd_v = None
            if ref_obj_value is not None and obj_value is not None:
                rpd_f = compute_rpdf(obj_value, ref_obj_value)
                rpd_v = (
                    (obj_value - ref_obj_value) / ref_obj_value
                    if ref_obj_value != 0
                    else None
                )

            result_rows.append(
                {
                    "instance_id": instance_id,
                    "subroutine_name": row["subroutine_name"],
                    "norm_time": norm_time,
                    "rpd_f": rpd_f,
                    "rpd_v": rpd_v,
                }
            )

    if not result_rows:
        return pd.DataFrame()

    return pd.DataFrame(result_rows)


def load_instance_progression_json(json_path: Path) -> dict[str, Any] | None:
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load progression JSON {json_path}: {e}")
        return None


def build_progression_points(
    progression_data: dict[str, Any],
    ref_obj_value: float | None = None,
) -> pd.DataFrame:
    rows = []
    timelimit = progression_data.get("timelimit_sec")
    instance_id = progression_data.get("instance_id", "")

    for call in progression_data.get("subroutine_calls", []):
        subroutine_name = call.get("subroutine_name", "")
        prefixed_name = call.get("prefixed_subroutine_name", "")
        call_index = call.get("call_index", -1)
        global_start = call.get("global_start_sec")
        global_end = call.get("global_end_sec")
        elapsed = call.get("elapsed_sec")

        for point in call.get("local_progress_list", []):
            local_sec = point.get("local_sec")
            global_sec = point.get("global_sec")
            obj_value = point.get("obj_value")

            norm_time = None
            if timelimit is not None and timelimit > 0 and global_sec is not None:
                norm_time = global_sec / timelimit

            rpd_f = None
            if ref_obj_value is not None and obj_value is not None:
                rpd_f = compute_rpdf(obj_value, ref_obj_value)

            local_ratio = None
            if elapsed is not None and elapsed > 0 and local_sec is not None:
                local_ratio = local_sec / elapsed

            rows.append(
                {
                    "instance_id": instance_id,
                    "subroutine_name": subroutine_name,
                    "prefixed_subroutine_name": prefixed_name,
                    "call_index": call_index,
                    "global_sec": global_sec,
                    "local_sec": local_sec,
                    "obj_value": obj_value,
                    "norm_time": norm_time,
                    "rpd_f": rpd_f,
                    "local_ratio": local_ratio,
                    "elapsed_sec": elapsed,
                    "global_start_sec": global_start,
                    "global_end_sec": global_end,
                    "timelimit_sec": timelimit,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compute_subroutine_mean_points(
    progression_df: pd.DataFrame,
) -> pd.DataFrame:
    if progression_df.empty:
        return pd.DataFrame()

    clean_df = progression_df.dropna(subset=["norm_time", "rpd_f"])

    if clean_df.empty:
        return pd.DataFrame()

    return (
        clean_df.groupby("subroutine_name", sort=False)
        .agg(
            mean_norm_time=("norm_time", "mean"),
            mean_rpd_f=("rpd_f", "mean"),
        )
        .reset_index()
    )


def compute_mean_progression_curve(
    progression_df: pd.DataFrame,
    local_ratio_grid_size: int = 20,
) -> pd.DataFrame:
    if progression_df.empty:
        return pd.DataFrame()

    clean_df = progression_df.dropna(subset=["local_ratio", "rpd_f"])

    if clean_df.empty:
        return pd.DataFrame()

    import numpy as np

    grid = np.linspace(0, 1, local_ratio_grid_size)
    rows = []

    for subroutine_name, group in clean_df.groupby("subroutine_name", sort=False):
        for u in grid:
            mask = (group["local_ratio"] >= u - 0.05) & (
                group["local_ratio"] <= u + 0.05
            )
            bucket = group[mask]
            if not bucket.empty:
                rows.append(
                    {
                        "subroutine_name": subroutine_name,
                        "local_ratio": u,
                        "mean_rpd_f": bucket["rpd_f"].mean(),
                        "count": len(bucket),
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compute_improvement_curve(
    progression_df: pd.DataFrame,
    local_ratio_grid_size: int = 20,
) -> pd.DataFrame:
    if progression_df.empty:
        return pd.DataFrame()

    clean_df = progression_df.dropna(subset=["local_ratio", "obj_value"])

    if clean_df.empty:
        return pd.DataFrame()

    import numpy as np

    grid = np.linspace(0, 1, local_ratio_grid_size)
    rows = []

    for subroutine_name, group in clean_df.groupby("subroutine_name", sort=False):
        group = group.sort_values(["call_index", "global_sec"]).reset_index(drop=True)
        for u in grid:
            mask = (group["local_ratio"] >= u - 0.05) & (
                group["local_ratio"] <= u + 0.05
            )
            bucket = group[mask]
            if len(bucket) < 2:
                continue
            initial_obj = bucket.iloc[0]["obj_value"]
            final_obj = bucket.iloc[-1]["obj_value"]
            improvement = (
                (initial_obj - final_obj) / initial_obj if initial_obj != 0 else 0
            )
            rows.append(
                {
                    "subroutine_name": subroutine_name,
                    "local_ratio": u,
                    "mean_improvement": improvement,
                    "count": len(bucket),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def aggregate_scenario_progression(
    working_dir: Path,
    baseline_df: pd.DataFrame | None = None,
    baseline_instance_col: str = "Instance",
    baseline_obj_val_col: str = "UB",
    omitted_subroutines: set[str] | None = None,
) -> dict[str, Any]:
    instance_dirs = _iter_instance_progression_json_paths(working_dir)

    if not instance_dirs:
        logging.warning(f"No subroutine_progression.json found in {working_dir}")
        return {}

    ref_obj_map = {}
    if baseline_df is not None and not baseline_df.empty:
        for _, row in baseline_df.iterrows():
            ref_obj_map[str(row[baseline_instance_col])] = row[baseline_obj_val_col]

    all_dfs = []
    for instance_dir, json_path in instance_dirs:
        data = load_instance_progression_json(json_path)
        if data is None:
            continue

        instance_id = str(data.get("instance_id") or instance_dir.name)
        ref_obj = ref_obj_map.get(instance_id)

        df = build_progression_points(data, ref_obj_value=ref_obj)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return {}

    progression_df = pd.concat(all_dfs, ignore_index=True)

    if omitted_subroutines:
        progression_df = progression_df[
            ~progression_df["subroutine_name"].isin(omitted_subroutines)
        ].copy()

    mean_points = compute_subroutine_mean_points(progression_df)
    mean_curve = compute_mean_progression_curve(progression_df)
    improvement_curve = compute_improvement_curve(progression_df)

    return {
        "progression_df": progression_df,
        "mean_points": mean_points,
        "mean_curve": mean_curve,
        "improvement_curve": improvement_curve,
    }
