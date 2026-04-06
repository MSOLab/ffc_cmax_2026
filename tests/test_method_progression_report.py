import json
from pathlib import Path

import pandas as pd
import pytest

from hybridflowshop.report.method_progression_report import (
    aggregate_scenario_progression,
    build_progression_points,
    compute_improvement_curve,
    compute_mean_progression_curve,
    compute_subroutine_mean_points,
    load_instance_progression_json,
)


def _make_sample_progression_data() -> dict:
    return {
        "artifact_version": 1,
        "instance_id": "1",
        "timelimit_sec": 50.0,
        "subroutine_calls": [
            {
                "call_index": 1,
                "subroutine_name": "initialize",
                "prefixed_subroutine_name": "1-initialize",
                "global_start_sec": 0.0,
                "global_end_sec": 5.0,
                "elapsed_sec": 5.0,
                "local_progress_list": [
                    {
                        "local_sec": 0.0,
                        "obj_value": 120.0,
                        "global_sec": 0.0,
                        "call_index": 1,
                        "prefixed_subroutine_name": "1-initialize",
                    },
                    {
                        "local_sec": 2.5,
                        "obj_value": 115.0,
                        "global_sec": 2.5,
                        "call_index": 1,
                        "prefixed_subroutine_name": "1-initialize",
                    },
                    {
                        "local_sec": 5.0,
                        "obj_value": 110.0,
                        "global_sec": 5.0,
                        "call_index": 1,
                        "prefixed_subroutine_name": "1-initialize",
                    },
                ],
            },
            {
                "call_index": 2,
                "subroutine_name": "repeat_while_improvement",
                "prefixed_subroutine_name": "2-repeat_while_improvement",
                "global_start_sec": 5.0,
                "global_end_sec": 25.0,
                "elapsed_sec": 20.0,
                "local_progress_list": [
                    {
                        "local_sec": 0.0,
                        "obj_value": 110.0,
                        "global_sec": 5.0,
                        "call_index": 2,
                        "prefixed_subroutine_name": "2-repeat_while_improvement",
                    },
                    {
                        "local_sec": 10.0,
                        "obj_value": 105.0,
                        "global_sec": 15.0,
                        "call_index": 2,
                        "prefixed_subroutine_name": "2-repeat_while_improvement",
                    },
                    {
                        "local_sec": 20.0,
                        "obj_value": 100.0,
                        "global_sec": 25.0,
                        "call_index": 2,
                        "prefixed_subroutine_name": "2-repeat_while_improvement",
                    },
                ],
            },
        ],
        "combined_progress_list": [],
        "subroutine_end_marker_list": [],
    }


def test_load_instance_progression_json_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    result = load_instance_progression_json(tmp_path / "nonexistent.json")
    assert result is None


def test_load_instance_progression_json_loads_valid_file(tmp_path: Path) -> None:
    data = _make_sample_progression_data()
    json_path = tmp_path / "subroutine_progression.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = load_instance_progression_json(json_path)
    assert result is not None
    assert result["instance_id"] == "1"
    assert result["timelimit_sec"] == 50.0


def test_build_progression_points_creates_rows(tmp_path: Path) -> None:
    data = _make_sample_progression_data()
    df = build_progression_points(data, ref_obj_value=100.0)

    assert not df.empty
    assert len(df) == 6
    assert set(df.columns) >= {
        "instance_id",
        "subroutine_name",
        "norm_time",
        "rpd_f",
        "local_ratio",
    }
    assert df["instance_id"].iloc[0] == "1"
    assert df["subroutine_name"].iloc[0] == "initialize"
    assert df["norm_time"].iloc[0] == pytest.approx(0.0)
    assert df["local_ratio"].iloc[0] == pytest.approx(0.0)


def test_build_progression_points_without_ref_obj(tmp_path: Path) -> None:
    data = _make_sample_progression_data()
    df = build_progression_points(data, ref_obj_value=None)

    assert not df.empty
    assert df["rpd_f"].isna().all()


def test_build_progression_points_preserves_duplicate_values() -> None:
    data = _make_sample_progression_data()
    data["subroutine_calls"][1]["local_progress_list"].insert(
        1,
        {
            "local_sec": 5.0,
            "obj_value": 110.0,
            "global_sec": 10.0,
            "call_index": 2,
            "prefixed_subroutine_name": "2-repeat_while_improvement",
        },
    )

    df = build_progression_points(data, ref_obj_value=100.0)

    repeat_df = df[df["prefixed_subroutine_name"] == "2-repeat_while_improvement"]
    assert len(repeat_df) == 4
    assert repeat_df["obj_value"].tolist() == [110.0, 110.0, 105.0, 100.0]
    assert repeat_df["local_sec"].tolist() == [0.0, 5.0, 10.0, 20.0]


def test_build_progression_points_empty_data() -> None:
    data = {
        "artifact_version": 1,
        "instance_id": "2",
        "timelimit_sec": 10.0,
        "subroutine_calls": [],
        "combined_progress_list": [],
        "subroutine_end_marker_list": [],
    }
    df = build_progression_points(data, ref_obj_value=50.0)
    assert df.empty


def test_compute_subroutine_mean_points_aggregates() -> None:
    data = _make_sample_progression_data()
    df = build_progression_points(data, ref_obj_value=100.0)
    mean_df = compute_subroutine_mean_points(df)

    assert not mean_df.empty
    assert "subroutine_name" in mean_df.columns
    assert "mean_norm_time" in mean_df.columns
    assert "mean_rpd_f" in mean_df.columns
    assert len(mean_df) == 2


def test_compute_subroutine_mean_points_empty_df() -> None:
    df = pd.DataFrame()
    result = compute_subroutine_mean_points(df)
    assert result.empty


def test_compute_mean_progression_curve_returns_curve() -> None:
    data = _make_sample_progression_data()
    df = build_progression_points(data, ref_obj_value=100.0)
    curve_df = compute_mean_progression_curve(df, local_ratio_grid_size=5)

    assert not curve_df.empty
    assert "subroutine_name" in curve_df.columns
    assert "local_ratio" in curve_df.columns
    assert "mean_rpd_f" in curve_df.columns


def test_compute_mean_progression_curve_empty_df() -> None:
    df = pd.DataFrame()
    result = compute_mean_progression_curve(df)
    assert result.empty


def test_compute_improvement_curve_returns_curve() -> None:
    rows = []
    for call_idx in range(3):
        for i in range(10):
            local_ratio = i / 9.0
            rows.append(
                {
                    "instance_id": "1",
                    "subroutine_name": "repeat",
                    "prefixed_subroutine_name": f"{call_idx}-repeat",
                    "call_index": call_idx,
                    "global_sec": call_idx * 10.0 + i,
                    "local_sec": i,
                    "obj_value": 120.0 - i * 2.0,
                    "norm_time": (call_idx * 10.0 + i) / 50.0,
                    "rpd_f": 0.1 - i * 0.01,
                    "local_ratio": local_ratio,
                    "elapsed_sec": 9.0,
                    "global_start_sec": call_idx * 10.0,
                    "global_end_sec": (call_idx + 1) * 10.0,
                    "timelimit_sec": 50.0,
                }
            )
    df = pd.DataFrame(rows)
    curve_df = compute_improvement_curve(df, local_ratio_grid_size=5)

    assert not curve_df.empty
    assert "subroutine_name" in curve_df.columns
    assert "local_ratio" in curve_df.columns
    assert "mean_improvement" in curve_df.columns


def test_compute_improvement_curve_empty_df() -> None:
    df = pd.DataFrame()
    result = compute_improvement_curve(df)
    assert result.empty


def test_aggregate_scenario_progression_finds_jsons(tmp_path: Path) -> None:
    instance_dir = tmp_path / "instance_1"
    results_dir = instance_dir / "results"
    results_dir.mkdir(parents=True)

    data = _make_sample_progression_data()
    json_path = results_dir / "subroutine_progression.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    baseline_df = pd.DataFrame([{"Instance": "1", "UB": 100.0}])

    result = aggregate_scenario_progression(
        tmp_path,
        baseline_df=baseline_df,
        baseline_instance_col="Instance",
        baseline_obj_val_col="UB",
    )

    assert "progression_df" in result
    assert "mean_points" in result
    assert "mean_curve" in result
    assert "improvement_curve" in result
    assert not result["progression_df"].empty


def test_aggregate_scenario_progression_omits_subroutines(tmp_path: Path) -> None:
    instance_dir = tmp_path / "instance_1"
    results_dir = instance_dir / "results"
    results_dir.mkdir(parents=True)

    data = _make_sample_progression_data()
    json_path = results_dir / "subroutine_progression.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = aggregate_scenario_progression(
        tmp_path,
        omitted_subroutines={"initialize"},
    )

    assert "progression_df" in result
    if not result["progression_df"].empty:
        assert "initialize" not in result["progression_df"]["subroutine_name"].values
