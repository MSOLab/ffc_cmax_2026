from pathlib import Path

from mbls.cpsat import ObjValueBoundStore
from routix.io.yaml import dump_yaml

import concurrent_painter


def test_draw_sw_cp_subproblem_progress_plots_creates_one_plot_per_subproblem(
    monkeypatch, tmp_path: Path
):
    log_path = tmp_path / "4-sw_cp_obj_log.yaml"
    dump_yaml(
        {
            "obj_value": {
                "name": "ObjVal after SW-CP batch",
                "data": {"1.0": 100},
                "notes": {"1.0": "batch=1"},
            },
            "obj_bound": {
                "name": "ObjVal before SW-CP batch",
                "data": {"1.0": 100},
                "notes": {"1.0": "batch=1"},
            },
            "sw_cp_metadata": {
                "cp_sat_subproblems": [
                    {
                        "batch_idx": 0,
                        "subproblem_idx": 1,
                        "objective_name": "right_slack",
                        "status": "OPTIMAL",
                        "obj_value_records": [[0.1, 5.0], [0.3, 2.0]],
                        "obj_bound_records": [[0.1, 8.0], [0.3, 2.0]],
                    },
                    {
                        "batch_idx": 1,
                        "subproblem_idx": 2,
                        "objective_name": "makespan",
                        "status": "FEASIBLE",
                        "obj_value_records": [[1.2, 99.0]],
                        "obj_bound_records": [[1.2, 80.0]],
                    },
                ]
            },
        },
        log_path,
    )

    calls: list[tuple[ObjValueBoundStore, Path, dict]] = []

    def fake_plot(store, save_path, **kwargs):
        calls.append((store, save_path, kwargs))

    monkeypatch.setattr(concurrent_painter.ObjValueBoundPlotter, "plot", fake_plot)

    concurrent_painter._draw_sw_cp_subproblem_progress_plots(
        file_path=log_path,
        base_output_path=tmp_path / "4-sw_cp_progress_plot.png",
        encoding="utf-8",
    )

    assert len(calls) == 2
    assert calls[0][1].name == "4-sw_cp_progress_plot_batch_001_subproblem_001.png"
    assert calls[0][0].obj_value_series.items() == [(0.1, 5.0), (0.3, 2.0)]
    assert calls[0][0].obj_bound_series.items() == [(0.1, 8.0), (0.3, 2.0)]
    assert "right_slack" in calls[0][2]["title"]
    assert calls[1][1].name == "4-sw_cp_progress_plot_batch_002_subproblem_002.png"
    assert calls[1][0].obj_value_series.items() == [(1.2, 99.0)]


def test_draw_sw_cp_subproblem_progress_plots_reads_legacy_pw_cp_metadata_key(
    monkeypatch, tmp_path: Path
):
    """Solutions saved before the pw_cp -> sw_cp rename still plot."""
    log_path = tmp_path / "4-incremental_pw_cp_obj_log.yaml"
    dump_yaml(
        {
            "obj_value": {"name": "obj_value", "data": {"1.0": 100}, "notes": {}},
            "obj_bound": {"name": "obj_bound", "data": {"1.0": 100}, "notes": {}},
            "pw_cp_metadata": {
                "cp_sat_subproblems": [
                    {
                        "batch_idx": 0,
                        "subproblem_idx": 1,
                        "objective_name": "makespan",
                        "status": "OPTIMAL",
                        "obj_value_records": [[0.1, 5.0]],
                        "obj_bound_records": [[0.1, 8.0]],
                    },
                ]
            },
        },
        log_path,
    )

    calls: list[tuple[ObjValueBoundStore, Path, dict]] = []
    monkeypatch.setattr(
        concurrent_painter.ObjValueBoundPlotter,
        "plot",
        lambda store, save_path, **kwargs: calls.append((store, save_path, kwargs)),
    )

    concurrent_painter._draw_sw_cp_subproblem_progress_plots(
        file_path=log_path,
        base_output_path=tmp_path / "4-incremental_pw_cp_progress_plot.png",
        encoding="utf-8",
    )

    assert len(calls) == 1
    assert calls[0][0].obj_value_series.items() == [(0.1, 5.0)]


def test_draw_sw_cp_subproblem_progress_plots_skips_when_metadata_missing(
    monkeypatch, tmp_path: Path
):
    log_path = tmp_path / "plain_obj_log.yaml"
    dump_yaml(
        {
            "obj_value": {"name": "obj_value", "data": {"1.0": 10}, "notes": {}},
            "obj_bound": {"name": "obj_bound", "data": {"1.0": 9}, "notes": {}},
        },
        log_path,
    )

    calls: list[tuple] = []
    monkeypatch.setattr(
        concurrent_painter.ObjValueBoundPlotter,
        "plot",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    concurrent_painter._draw_sw_cp_subproblem_progress_plots(
        file_path=log_path,
        base_output_path=tmp_path / "plain_progress.png",
        encoding="utf-8",
    )

    assert calls == []
