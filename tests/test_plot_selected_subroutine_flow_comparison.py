from pathlib import Path

import pandas as pd

import scripts.plot_selected_subroutine_flow_comparison as script_module


def _write_method_rpdf_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_cli_creates_html_from_explicit_paths(tmp_path: Path, caplog) -> None:
    scenario_1 = tmp_path / "Outputs_scenarios" / "run_a" / "ff2020" / "flow-01"
    scenario_2 = tmp_path / "Outputs_scenarios" / "run_b" / "ff2020" / "flow-02"
    output_path = tmp_path / "comparison.html"

    rows_1 = [
        {
            "instance_id": "1",
            "subroutine_name": "initialize",
            "norm_time": 0.10,
            "rpd_f": 0.08,
        },
        {
            "instance_id": "1",
            "subroutine_name": "repeat",
            "norm_time": 0.30,
            "rpd_f": 0.04,
        },
    ]
    rows_2 = [
        {
            "instance_id": "1",
            "subroutine_name": "initialize",
            "norm_time": 0.12,
            "rpd_f": 0.07,
        },
        {
            "instance_id": "1",
            "subroutine_name": "repeat",
            "norm_time": 0.40,
            "rpd_f": 0.02,
        },
    ]

    _write_method_rpdf_summary(scenario_1 / script_module.SUMMARY_FILENAME, rows_1)
    _write_method_rpdf_summary(scenario_2 / script_module.SUMMARY_FILENAME, rows_2)

    exit_code = script_module.run_cli(
        [
            str(scenario_1),
            str(scenario_2),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "flow-01" in content
    assert "flow-02" in content
    assert "initialize" in content
    assert "repeat" in content
    assert "step_x" in content
    assert "guide_marker_x" in content
    assert "range: [0, payload.x_max]" in content
    assert "range: [0, payload.y_max]" in content
    assert (
        "Falling back to endpoint CSV for selected-scenario comparison" in caplog.text
    )


def test_run_cli_uses_global_scenario_path_list_when_args_are_empty(
    tmp_path: Path, monkeypatch
) -> None:
    scenario_1 = tmp_path / "Outputs_scenarios" / "run_a" / "ff2020" / "flow-01"
    scenario_2 = tmp_path / "Outputs_scenarios" / "run_b" / "ff2020" / "flow-02"
    output_path = tmp_path / "comparison.html"

    _write_method_rpdf_summary(
        scenario_1 / script_module.SUMMARY_FILENAME,
        [
            {
                "instance_id": "1",
                "subroutine_name": "initialize",
                "norm_time": 0.10,
                "rpd_f": 0.08,
            }
        ],
    )
    _write_method_rpdf_summary(
        scenario_2 / script_module.SUMMARY_FILENAME,
        [
            {
                "instance_id": "1",
                "subroutine_name": "initialize",
                "norm_time": 0.12,
                "rpd_f": 0.07,
            }
        ],
    )

    monkeypatch.setattr(
        script_module,
        "SCENARIO_PATH_LIST",
        [str(scenario_1), str(scenario_2)],
    )

    exit_code = script_module.run_cli(["--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()


def test_run_cli_fails_when_no_paths_are_resolved(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(script_module, "SCENARIO_PATH_LIST", [])

    exit_code = script_module.run_cli(["--output", str(tmp_path / "comparison.html")])

    assert exit_code == 1
    assert "No scenario directories resolved" in caplog.text


def test_run_cli_fails_when_summary_csv_is_missing(tmp_path: Path, caplog) -> None:
    scenario_1 = tmp_path / "Outputs_scenarios" / "run_a" / "ff2020" / "flow-01"
    scenario_2 = tmp_path / "Outputs_scenarios" / "run_b" / "ff2020" / "flow-02"

    _write_method_rpdf_summary(
        scenario_1 / script_module.SUMMARY_FILENAME,
        [
            {
                "instance_id": "1",
                "subroutine_name": "initialize",
                "norm_time": 0.10,
                "rpd_f": 0.08,
            }
        ],
    )
    scenario_2.mkdir(parents=True, exist_ok=True)

    exit_code = script_module.run_cli([str(scenario_1), str(scenario_2)])

    assert exit_code == 1
    assert "Missing required summary CSV" in caplog.text


def test_build_scenario_labels_disambiguates_duplicate_basenames() -> None:
    scenario_paths = [
        Path("/tmp/Outputs_scenarios/run_a/ff2020/20260331-03"),
        Path("/tmp/Outputs_scenarios/run_b/ff2020/20260331-03"),
    ]

    labels = script_module.build_scenario_labels(scenario_paths)

    assert labels == [
        "run_a/ff2020/20260331-03",
        "run_b/ff2020/20260331-03",
    ]
