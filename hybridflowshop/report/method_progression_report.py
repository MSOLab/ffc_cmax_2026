from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from exp_compare.metrics import compute_rpdf


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
    instance_dirs = []
    for item in working_dir.iterdir():
        if item.is_dir():
            json_path = item / "results" / "subroutine_progression.json"
            if not json_path.exists():
                json_path = item / "subroutine_progression.json"
            if json_path.exists():
                instance_dirs.append((item, json_path))

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

        instance_id = data.get("instance_id", "")
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
