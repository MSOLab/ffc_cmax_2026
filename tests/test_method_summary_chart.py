from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import hfs_multi_instance_runner as multi_runner_module
from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hybridflowshop.report.method_summary_chart import (
    _build_best_so_far_progression_points,
    _build_html_payload,
    _build_method_rpdf_scatter_df,
    _build_raw_instance_progression,
    _build_raw_plotly_series,
    _load_and_merge_for_html,
    _lookup_rpdf_at_or_before,
    export_method_rpdf_scatter_html,
    export_method_rpdf_scatter_svg,
)


def test_build_method_rpdf_scatter_df_aggregates_and_preserves_order() -> None:
    metrics_long_df = pd.DataFrame(
        [
            {
                "subroutine_name": "repeat_while_improvement",
                "norm_time": 0.30,
                "rpd_f": 0.04,
            },
            {"subroutine_name": "initialize", "norm_time": 0.10, "rpd_f": 0.08},
            {
                "subroutine_name": "repeat_while_improvement",
                "norm_time": 0.50,
                "rpd_f": 0.02,
            },
            {"subroutine_name": "apply_shdlb", "norm_time": 0.01, "rpd_f": None},
        ]
    )

    plot_df = _build_method_rpdf_scatter_df(metrics_long_df)

    assert list(plot_df["subroutine_name"]) == [
        "repeat_while_improvement",
        "initialize",
    ]
    assert plot_df.loc[0, "mean_norm_time"] == pytest.approx(0.40)
    assert plot_df.loc[0, "mean_rpd_f"] == pytest.approx(0.03)
    assert plot_df.loc[1, "mean_norm_time"] == pytest.approx(0.10)
    assert plot_df.loc[1, "mean_rpd_f"] == pytest.approx(0.08)


def test_build_method_rpdf_scatter_df_requires_expected_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        _build_method_rpdf_scatter_df(pd.DataFrame({"subroutine_name": ["a"]}))


def test_export_method_rpdf_scatter_svg_creates_svg_with_labels(
    tmp_path: Path,
) -> None:
    metrics_long_df = pd.DataFrame(
        [
            {"subroutine_name": "initialize", "norm_time": 0.10, "rpd_f": 0.08},
            {"subroutine_name": "repeat", "norm_time": 0.30, "rpd_f": 0.04},
            {"subroutine_name": "initialize", "norm_time": 0.20, "rpd_f": 0.06},
        ]
    )
    output_path = tmp_path / "summary_method_rpdf_and_norm_time_scatter.svg"

    created = export_method_rpdf_scatter_svg(metrics_long_df, output_path)

    assert created is True
    content = output_path.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "initialize" in content
    assert "repeat" in content
    assert "%" in content


def test_export_method_rpdf_scatter_svg_supports_single_point(tmp_path: Path) -> None:
    metrics_long_df = pd.DataFrame(
        [{"subroutine_name": "only_one", "norm_time": 0.10, "rpd_f": 0.02}]
    )
    output_path = tmp_path / "single.svg"

    created = export_method_rpdf_scatter_svg(metrics_long_df, output_path)

    assert created is True
    content = output_path.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "only_one" in content


def test_export_method_rpdf_scatter_svg_skips_when_all_points_invalid(
    tmp_path: Path,
) -> None:
    metrics_long_df = pd.DataFrame(
        [
            {"subroutine_name": "apply_shdlb", "norm_time": 0.01, "rpd_f": None},
            {"subroutine_name": "apply_shdlb", "norm_time": 0.02, "rpd_f": None},
        ]
    )
    output_path = tmp_path / "invalid.svg"

    created = export_method_rpdf_scatter_svg(metrics_long_df, output_path)

    assert created is False
    assert not output_path.exists()


def _make_runner(tmp_path: Path) -> HfsMultiInstanceRunner:
    runner = HfsMultiInstanceRunner.__new__(HfsMultiInstanceRunner)
    runner.working_dir = tmp_path
    runner.baseline_df = pd.DataFrame([{"Instance": "1", "n": 20, "s": 5, "UB": 100.0}])
    runner.baseline_instance_col = "Instance"
    runner.baseline_job_cnt_col = "n"
    runner.baseline_stage_cnt_col = "s"
    runner.baseline_obj_val_col = "UB"
    runner.runners = [
        SimpleNamespace(name="1", stopping_criteria=SimpleNamespace(timelimit=50.0))
    ]
    return runner


def _write_method_summary_csv(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "instance_id": 1,
                "subroutine_name": "initialize",
                "end_time": 5.0,
                "obj_value": 110.0,
            },
            {
                "instance_id": 1,
                "subroutine_name": "repeat",
                "end_time": 25.0,
                "obj_value": 100.0,
            },
        ]
    ).to_csv(tmp_path / "summary_method_end_time_and_obj_value_long.csv", index=False)


def test_create_rpd_summary_invokes_svg_export(tmp_path: Path, monkeypatch) -> None:
    runner = _make_runner(tmp_path)
    _write_method_summary_csv(tmp_path)
    captured: dict[str, object] = {}

    def fake_export(metrics_long_df: pd.DataFrame, output_path: Path) -> bool:
        captured["df"] = metrics_long_df.copy()
        captured["path"] = output_path
        return True

    monkeypatch.setattr(
        multi_runner_module, "export_method_rpdf_scatter_svg", fake_export
    )

    result_df = runner._create_rpd_summary()

    assert result_df is not None
    assert (tmp_path / "summary_method_rpdf_and_norm_time_long.csv").exists()
    assert (tmp_path / "summary_method_rpdf_and_norm_time_wide.csv").exists()
    assert (
        captured["path"] == tmp_path / "summary_method_rpdf_and_norm_time_scatter.svg"
    )
    captured_df = captured["df"]
    assert isinstance(captured_df, pd.DataFrame)
    assert list(captured_df["subroutine_name"]) == ["initialize", "repeat"]


def test_create_rpd_summary_keeps_csv_outputs_when_svg_export_fails(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    runner = _make_runner(tmp_path)
    _write_method_summary_csv(tmp_path)

    def fake_export(metrics_long_df: pd.DataFrame, output_path: Path) -> bool:
        raise RuntimeError("svg failed")

    monkeypatch.setattr(
        multi_runner_module, "export_method_rpdf_scatter_svg", fake_export
    )

    result_df = runner._create_rpd_summary()

    assert result_df is not None
    assert (tmp_path / "summary_method_rpdf_and_norm_time_long.csv").exists()
    assert (tmp_path / "summary_method_rpdf_and_norm_time_wide.csv").exists()
    assert "Failed to export method RPD scatter SVG" in caplog.text


def test_load_and_merge_for_html_uses_baseline_metadata_columns() -> None:
    metrics_df = pd.DataFrame(
        [{"instance_id": 1, "subroutine_name": "a", "norm_time": 0.1, "rpd_f": 0.02}]
    )
    baseline_df = pd.DataFrame(
        [{"Instance": 1, "n": 20, "s": 5, "UB": 100.0, "LB": 90.0}]
    )

    result = _load_and_merge_for_html(
        metrics_long_df=metrics_df,
        baseline_df=baseline_df,
        baseline_instance_col="Instance",
        baseline_job_cnt_col="n",
        baseline_stage_cnt_col="s",
    )

    assert "job_cnt" in result.columns
    assert "stage_cnt" in result.columns
    assert result["job_cnt"].iloc[0] == 20
    assert result["stage_cnt"].iloc[0] == 5


def test_load_and_merge_for_html_raises_missing_metrics_columns() -> None:
    metrics_df = pd.DataFrame(
        [{"instance_id": 1, "subroutine_name": "a", "norm_time": 0.1}]
    )
    baseline_df = pd.DataFrame([{"Instance": 1, "n": 20, "s": 5}])

    with pytest.raises(ValueError, match="Missing columns in metrics DataFrame"):
        _load_and_merge_for_html(
            metrics_long_df=metrics_df,
            baseline_df=baseline_df,
            baseline_instance_col="Instance",
            baseline_job_cnt_col="n",
            baseline_stage_cnt_col="s",
        )


def test_load_and_merge_for_html_raises_missing_baseline_columns() -> None:
    metrics_df = pd.DataFrame(
        [{"instance_id": 1, "subroutine_name": "a", "norm_time": 0.1, "rpd_f": 0.02}]
    )
    baseline_df = pd.DataFrame([{"Instance": 1, "n": 20}])

    with pytest.raises(ValueError, match="Missing columns in baseline DataFrame"):
        _load_and_merge_for_html(
            metrics_long_df=metrics_df,
            baseline_df=baseline_df,
            baseline_instance_col="Instance",
            baseline_job_cnt_col="n",
            baseline_stage_cnt_col="s",
        )


def test_load_and_merge_for_html_drops_unmatched_rows_without_failing(
    caplog,
) -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "init",
                "norm_time": 0.1,
                "rpd_f": 0.02,
            },
            {
                "instance_id": "999",
                "subroutine_name": "repeat",
                "norm_time": 0.2,
                "rpd_f": 0.01,
            },
        ]
    )
    baseline_df = pd.DataFrame([{"Instance": 1, "n": 20, "s": 5}])

    result = _load_and_merge_for_html(
        metrics_long_df=metrics_df,
        baseline_df=baseline_df,
        baseline_instance_col="Instance",
        baseline_job_cnt_col="n",
        baseline_stage_cnt_col="s",
    )

    assert len(result) == 1
    assert result["instance_id"].iloc[0] == "1"
    assert "Excluded 1 metric rows from HTML chart" in caplog.text


def test_build_best_so_far_progression_points_dedupes_times_and_keeps_best_values() -> (
    None
):
    grp = pd.DataFrame(
        [
            {"subroutine_name": "init", "norm_time": 0.10, "rpd_f": 0.08},
            {"subroutine_name": "worse_same_time", "norm_time": 0.10, "rpd_f": 0.09},
            {"subroutine_name": "better", "norm_time": 0.35, "rpd_f": 0.05},
            {"subroutine_name": "same_best", "norm_time": 0.50, "rpd_f": 0.05},
        ]
    )

    points = _build_best_so_far_progression_points(grp)

    assert [point.time for point in points] == pytest.approx([0.10, 0.35, 0.50])
    assert [point.rpd_f for point in points] == pytest.approx([0.08, 0.05, 0.05])


def test_lookup_rpdf_at_or_before_returns_previous_best_point() -> None:
    grp = pd.DataFrame(
        [
            {"subroutine_name": "init", "norm_time": 0.10, "rpd_f": 0.08},
            {"subroutine_name": "repeat", "norm_time": 0.20, "rpd_f": 0.07},
            {"subroutine_name": "repeat", "norm_time": 0.30, "rpd_f": 0.06},
        ]
    )

    points = _build_best_so_far_progression_points(grp)

    assert _lookup_rpdf_at_or_before(points, 0.20) == pytest.approx(0.07)
    assert _lookup_rpdf_at_or_before(points, 0.25) == pytest.approx(0.07)
    assert _lookup_rpdf_at_or_before(points, 0.05) is None


def test_build_raw_instance_progression_and_plotly_series_keeps_markers_on_line() -> (
    None
):
    grp = pd.DataFrame(
        [
            {"subroutine_name": "init", "norm_time": 0.10, "rpd_f": 0.08},
            {"subroutine_name": "worse", "norm_time": 0.20, "rpd_f": 0.12},
            {"subroutine_name": "better", "norm_time": 0.35, "rpd_f": 0.05},
            {"subroutine_name": "same_best", "norm_time": 0.50, "rpd_f": 0.05},
        ]
    )

    model = _build_raw_instance_progression(
        instance_id="1",
        endpoint_grp=grp,
        progression_grp=grp,
        job_cnt=20,
        stage_cnt=5,
    )
    series = _build_raw_plotly_series(model)

    assert series["x"] == pytest.approx([0.10, 0.20, 0.35, 0.50])
    assert series["y"] == pytest.approx([0.08, 0.08, 0.05, 0.05])
    assert series["step_x"] == pytest.approx([0.10, 0.20, 0.35, 0.35, 0.50])
    assert series["step_y"] == pytest.approx([0.08, 0.08, 0.08, 0.05, 0.05])
    assert series["text"] == ["init", "worse", "better", "same_best"]


def test_build_html_payload_uses_best_so_far_for_raw_series() -> None:
    merged_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "init",
                "norm_time": 0.10,
                "rpd_f": 0.08,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "worse",
                "norm_time": 0.20,
                "rpd_f": 0.12,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "better",
                "norm_time": 0.35,
                "rpd_f": 0.05,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
        ]
    )

    payload = _build_html_payload(merged_df)

    assert len(payload["raw_series"]) == 1
    raw_series = payload["raw_series"][0]
    assert raw_series["x"] == pytest.approx([0.10, 0.20, 0.35])
    assert raw_series["y"] == pytest.approx([0.08, 0.08, 0.05])
    assert raw_series["step_x"] == pytest.approx([0.10, 0.20, 0.35, 0.35])
    assert raw_series["step_y"] == pytest.approx([0.08, 0.08, 0.08, 0.05])


def test_build_html_payload_uses_progression_points_for_raw_line() -> None:
    endpoint_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "init",
                "norm_time": 0.10,
                "rpd_f": 0.08,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.35,
                "rpd_f": 0.05,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
        ]
    )
    progression_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "init",
                "norm_time": 0.10,
                "rpd_f": 0.08,
                "global_sec": 10.0,
                "call_index": 1,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.20,
                "rpd_f": 0.07,
                "global_sec": 20.0,
                "call_index": 2,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.30,
                "rpd_f": 0.06,
                "global_sec": 30.0,
                "call_index": 2,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.35,
                "rpd_f": 0.05,
                "global_sec": 35.0,
                "call_index": 2,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
        ]
    )

    payload = _build_html_payload(endpoint_df, raw_progression_df=progression_df)

    raw_series = payload["raw_series"][0]
    assert raw_series["x"] == pytest.approx([0.10, 0.35])
    assert raw_series["y"] == pytest.approx([0.08, 0.05])
    assert raw_series["step_x"] == pytest.approx(
        [0.10, 0.20, 0.20, 0.30, 0.30, 0.35, 0.35]
    )
    assert raw_series["step_y"] == pytest.approx(
        [0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05]
    )


def test_build_html_payload_places_marker_on_previous_progression_point() -> None:
    endpoint_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.35,
                "rpd_f": 0.05,
                "job_cnt": 20,
                "stage_cnt": 5,
            }
        ]
    )
    progression_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.10,
                "rpd_f": 0.08,
                "global_sec": 10.0,
                "call_index": 2,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.30,
                "rpd_f": 0.06,
                "global_sec": 30.0,
                "call_index": 2,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
        ]
    )

    payload = _build_html_payload(endpoint_df, raw_progression_df=progression_df)

    raw_series = payload["raw_series"][0]
    assert raw_series["x"] == pytest.approx([0.35])
    assert raw_series["y"] == pytest.approx([0.06])


def test_build_html_payload_groups_mean_series_by_job_and_stage() -> None:
    merged_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "init",
                "norm_time": 0.10,
                "rpd_f": 0.03,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "2",
                "subroutine_name": "init",
                "norm_time": 0.20,
                "rpd_f": 0.05,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
            {
                "instance_id": "1",
                "subroutine_name": "repeat",
                "norm_time": 0.30,
                "rpd_f": 0.01,
                "job_cnt": 20,
                "stage_cnt": 5,
            },
        ]
    )

    payload = _build_html_payload(merged_df)

    assert payload["job_cnt_values"] == [20]
    assert payload["stage_cnt_values"] == [5]
    assert len(payload["raw_series"]) == 2
    assert len(payload["mean_series"]) == 1
    mean_series = payload["mean_series"][0]
    assert mean_series["job_cnt"] == 20
    assert mean_series["stage_cnt"] == 5
    assert mean_series["text"] == ["init", "repeat"]
    assert mean_series["x"][0] == pytest.approx(0.15)
    assert mean_series["y"][0] == pytest.approx(0.04)


def test_export_method_rpdf_scatter_html_creates_file_with_filters(
    tmp_path: Path,
) -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "instance_id": 1,
                "subroutine_name": "initialize",
                "norm_time": 0.1,
                "rpd_f": 0.08,
            },
            {
                "instance_id": 2,
                "subroutine_name": "repeat",
                "norm_time": 0.3,
                "rpd_f": 0.04,
            },
        ]
    )
    baseline_df = pd.DataFrame(
        [
            {"Instance": 1, "n": 20, "s": 5},
            {"Instance": 2, "n": 40, "s": 10},
        ]
    )
    output_path = tmp_path / "test.html"

    created = export_method_rpdf_scatter_html(
        metrics_long_df=metrics_df,
        baseline_df=baseline_df,
        output_path=output_path,
        baseline_instance_col="Instance",
        baseline_job_cnt_col="n",
        baseline_stage_cnt_col="s",
        raw_progression_df=metrics_df.copy(),
    )

    assert created is True
    content = output_path.read_text(encoding="utf-8")
    assert "<html" in content
    assert "plotly" in content.lower()
    assert "instance progression" in content
    assert "mean progression by (job_cnt, stage_cnt)" in content
    assert 'id="mode-filter"' in content
    assert 'id="job-cnt-filter"' in content
    assert 'id="stage-cnt-filter"' in content
    assert "step_x" in content
    assert 'mode: "lines"' in content
    assert 'mode: "markers"' in content


def test_export_method_rpdf_scatter_html_returns_false_when_no_rows_survive_merge(
    tmp_path: Path,
) -> None:
    metrics_df = pd.DataFrame(
        [{"instance_id": 999, "subroutine_name": "a", "norm_time": 0.1, "rpd_f": 0.02}]
    )
    baseline_df = pd.DataFrame([{"Instance": 1, "n": 20, "s": 5}])
    output_path = tmp_path / "test.html"

    created = export_method_rpdf_scatter_html(
        metrics_long_df=metrics_df,
        baseline_df=baseline_df,
        output_path=output_path,
        baseline_instance_col="Instance",
        baseline_job_cnt_col="n",
        baseline_stage_cnt_col="s",
    )

    assert created is False
    assert not output_path.exists()


def test_create_rpd_summary_invokes_html_export(tmp_path: Path, monkeypatch) -> None:
    runner = _make_runner(tmp_path)
    _write_method_summary_csv(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        multi_runner_module,
        "export_method_rpdf_scatter_svg",
        lambda metrics_long_df, output_path: True,
    )
    json_metrics_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "initialize",
                "norm_time": 0.11,
                "rpd_f": 0.07,
                "rpd_v": 0.08,
            }
        ]
    )
    monkeypatch.setattr(
        multi_runner_module,
        "aggregate_scenario_endpoint_metrics_from_json",
        lambda *args, **kwargs: json_metrics_df.copy(),
    )
    progression_df = pd.DataFrame(
        [
            {
                "instance_id": "1",
                "subroutine_name": "initialize",
                "norm_time": 0.09,
                "rpd_f": 0.08,
                "global_sec": 9.0,
                "call_index": 1,
            },
            {
                "instance_id": "1",
                "subroutine_name": "initialize",
                "norm_time": 0.11,
                "rpd_f": 0.07,
                "global_sec": 11.0,
                "call_index": 1,
            },
        ]
    )
    monkeypatch.setattr(
        multi_runner_module,
        "aggregate_scenario_progression",
        lambda *args, **kwargs: {"progression_df": progression_df.copy()},
    )

    def fake_export_html(
        metrics_long_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        output_path: Path,
        baseline_instance_col: str,
        baseline_job_cnt_col: str,
        baseline_stage_cnt_col: str,
        raw_progression_df: pd.DataFrame | None = None,
    ) -> bool:
        captured["df"] = metrics_long_df.copy()
        captured["raw_progression_df"] = raw_progression_df
        captured["baseline_df"] = baseline_df.copy()
        captured["path"] = output_path
        captured["baseline_instance_col"] = baseline_instance_col
        captured["baseline_job_cnt_col"] = baseline_job_cnt_col
        captured["baseline_stage_cnt_col"] = baseline_stage_cnt_col
        return True

    monkeypatch.setattr(
        multi_runner_module, "export_method_rpdf_scatter_html", fake_export_html
    )

    result_df = runner._create_rpd_summary()

    assert result_df is not None
    assert (
        captured["path"] == tmp_path / "summary_method_rpdf_and_norm_time_scatter.html"
    )
    assert captured["baseline_instance_col"] == "Instance"
    assert captured["baseline_job_cnt_col"] == "n"
    assert captured["baseline_stage_cnt_col"] == "s"
    captured_baseline_df = captured["baseline_df"]
    assert isinstance(captured_baseline_df, pd.DataFrame)
    assert list(captured_baseline_df.columns) == ["Instance", "n", "s", "UB"]
    captured_df = captured["df"]
    assert isinstance(captured_df, pd.DataFrame)
    assert captured_df.equals(json_metrics_df)
    captured_raw_progression_df = captured["raw_progression_df"]
    assert isinstance(captured_raw_progression_df, pd.DataFrame)
    assert captured_raw_progression_df.equals(progression_df)


def test_create_rpd_summary_falls_back_to_csv_metrics_when_json_metrics_empty(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _make_runner(tmp_path)
    _write_method_summary_csv(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        multi_runner_module,
        "export_method_rpdf_scatter_svg",
        lambda metrics_long_df, output_path: True,
    )
    monkeypatch.setattr(
        multi_runner_module,
        "aggregate_scenario_endpoint_metrics_from_json",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    def fake_export_html(
        metrics_long_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        output_path: Path,
        baseline_instance_col: str,
        baseline_job_cnt_col: str,
        baseline_stage_cnt_col: str,
        raw_progression_df: pd.DataFrame | None = None,
    ) -> bool:
        captured["df"] = metrics_long_df.copy()
        captured["raw_progression_df"] = raw_progression_df
        return True

    monkeypatch.setattr(
        multi_runner_module, "export_method_rpdf_scatter_html", fake_export_html
    )

    result_df = runner._create_rpd_summary()

    assert result_df is not None
    captured_df = captured["df"]
    assert isinstance(captured_df, pd.DataFrame)
    assert list(captured_df["subroutine_name"]) == ["initialize", "repeat"]
    assert captured["raw_progression_df"] is None


def test_create_rpd_summary_keeps_outputs_when_html_export_fails(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    runner = _make_runner(tmp_path)
    _write_method_summary_csv(tmp_path)

    monkeypatch.setattr(
        multi_runner_module,
        "export_method_rpdf_scatter_svg",
        lambda metrics_long_df, output_path: True,
    )
    monkeypatch.setattr(
        multi_runner_module,
        "aggregate_scenario_endpoint_metrics_from_json",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        multi_runner_module,
        "export_method_rpdf_scatter_html",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("html failed")),
    )

    result_df = runner._create_rpd_summary()

    assert result_df is not None
    assert (tmp_path / "summary_method_rpdf_and_norm_time_long.csv").exists()
    assert (tmp_path / "summary_method_rpdf_and_norm_time_wide.csv").exists()
    assert "Failed to export method RPD scatter HTML" in caplog.text
