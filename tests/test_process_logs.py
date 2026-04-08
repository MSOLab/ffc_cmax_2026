from pathlib import Path

import pandas as pd
import yaml

import hfs_multi_instance_runner as multi_runner_module
from hfs_multi_instance_runner import HfsMultiInstanceRunner
from scripts.process_logs import create_method_end_time_and_obj_value_summary


def _write_flow(scenario_dir: Path, methods: list[str]) -> None:
    with open(scenario_dir / "subroutine_flow.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump([{"method": method} for method in methods], f, sort_keys=False)


def _write_controller_log(instance_dir: Path, entries: list[dict]) -> None:
    lines = []
    for entry in entries:
        lines.append(f"2026-04-01 00:00:00,000 - INFO - {entry}\n")

    with open(instance_dir / "subroutine_controller.log", "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_obj_log(instance_dir: Path, data: dict, notes: dict) -> None:
    results_dir = instance_dir / "results"
    results_dir.mkdir(exist_ok=True)

    with open(
        results_dir / f"{instance_dir.name}_obj_log.yaml", "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            {"obj_value": {"data": data, "notes": notes}}, f, sort_keys=False
        )


def _create_scenario(
    tmp_path: Path,
    methods: list[str],
    log_entries: list[dict],
    obj_data: dict | None = None,
    obj_notes: dict | None = None,
) -> tuple[Path, Path]:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    _write_flow(scenario_dir, methods)

    instance_dir = scenario_dir / "1"
    instance_dir.mkdir()
    _write_controller_log(instance_dir, log_entries)
    _write_obj_log(instance_dir, obj_data or {}, obj_notes or {})
    return scenario_dir, instance_dir


def test_create_summary_default_keeps_missing_subroutines_blank(tmp_path: Path) -> None:
    scenario_dir, instance_dir = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second", "third"],
        log_entries=[
            {
                "method": "first",
                "call_context": "1-first",
                "start_sec": 0.0,
                "kwargs": {},
                "elapsed_sec": 1.0,
            },
            {
                "method": "second",
                "call_context": "2-second",
                "start_sec": 1.0,
                "kwargs": {},
                "elapsed_sec": 2.0,
            },
        ],
        obj_data={"1.0": 100.0, "3.0": 90.0},
        obj_notes={"1.0": "1-first", "3.0": "2-second"},
    )

    summary_df = create_method_end_time_and_obj_value_summary(scenario_dir)

    assert summary_df is not None
    assert list(summary_df["subroutine_name"]) == ["first", "second", "third"]

    third_row = summary_df[summary_df["subroutine_name"] == "third"].iloc[0]
    assert pd.isna(third_row["end_time"])
    assert pd.isna(third_row["obj_value"])

    instance_df = pd.read_csv(instance_dir / "method_end_time_and_obj_value.csv")
    third_instance_row = instance_df[instance_df["method_name"] == "third"].iloc[0]
    assert pd.isna(third_instance_row["method_end_sec"])
    assert pd.isna(third_instance_row["objective_value"])


def test_create_summary_record_all_subroutines_fills_trailing_missing_rows(
    tmp_path: Path,
) -> None:
    scenario_dir, instance_dir = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second", "third"],
        log_entries=[
            {
                "method": "first",
                "call_context": "1-first",
                "start_sec": 0.0,
                "kwargs": {},
                "elapsed_sec": 1.0,
            },
            {
                "method": "second",
                "call_context": "2-second",
                "start_sec": 1.0,
                "kwargs": {},
                "elapsed_sec": 2.0,
            },
        ],
        obj_data={"1.0": 100.0, "3.0": 90.0},
        obj_notes={"1.0": "1-first", "3.0": "2-second"},
    )

    summary_df = create_method_end_time_and_obj_value_summary(
        scenario_dir, record_all_subroutines=True
    )

    assert summary_df is not None
    third_row = summary_df[summary_df["subroutine_name"] == "third"].iloc[0]
    assert third_row["end_time"] == 3.0
    assert third_row["obj_value"] == 90.0

    instance_df = pd.read_csv(instance_dir / "method_end_time_and_obj_value.csv")
    third_instance_row = instance_df[instance_df["method_name"] == "third"].iloc[0]
    assert third_instance_row["method_end_sec"] == 3.0
    assert third_instance_row["objective_value"] == 90.0


def test_create_summary_record_all_subroutines_leaves_blank_when_nothing_executed(
    tmp_path: Path,
) -> None:
    scenario_dir, instance_dir = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second"],
        log_entries=[],
    )

    summary_df = create_method_end_time_and_obj_value_summary(
        scenario_dir, record_all_subroutines=True
    )

    assert summary_df is not None
    assert len(summary_df) == 2
    assert summary_df["end_time"].isna().all()
    assert summary_df["obj_value"].isna().all()

    instance_df = pd.read_csv(instance_dir / "method_end_time_and_obj_value.csv")
    assert instance_df["method_end_sec"].isna().all()
    assert instance_df["objective_value"].isna().all()


def test_create_summary_omits_requested_subroutines_and_wide_columns(
    tmp_path: Path,
) -> None:
    scenario_dir, instance_dir = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second", "third"],
        log_entries=[
            {
                "method": "first",
                "call_context": "1-first",
                "start_sec": 0.0,
                "kwargs": {},
                "elapsed_sec": 1.0,
            },
            {
                "method": "second",
                "call_context": "2-second",
                "start_sec": 1.0,
                "kwargs": {},
                "elapsed_sec": 2.0,
            },
        ],
        obj_data={"1.0": 100.0, "3.0": 90.0},
        obj_notes={"1.0": "1-first", "3.0": "2-second"},
    )

    summary_df = create_method_end_time_and_obj_value_summary(
        scenario_dir,
        record_all_subroutines=True,
        omitted_subroutines={"second"},
    )

    assert summary_df is not None
    assert list(summary_df["subroutine_name"]) == ["first", "third"]

    third_row = summary_df[summary_df["subroutine_name"] == "third"].iloc[0]
    assert third_row["end_time"] == 3.0
    assert third_row["obj_value"] == 90.0

    instance_df = pd.read_csv(instance_dir / "method_end_time_and_obj_value.csv")
    assert list(instance_df["method_name"]) == ["first", "third"]

    wide_df = pd.read_csv(
        scenario_dir / "summary_method_end_time_and_obj_value_wide.csv"
    )
    assert "second_end_time" not in wide_df.columns
    assert "second_obj_value" not in wide_df.columns
    assert wide_df.loc[0, "third_end_time"] == 3.0
    assert wide_df.loc[0, "third_obj_value"] == 90.0


def test_post_run_process_passes_record_all_subroutines_true(
    tmp_path: Path, monkeypatch
) -> None:
    captured_kwargs = {}

    def fake_create_method_end_time_and_obj_value_summary(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        multi_runner_module,
        "create_method_end_time_and_obj_value_summary",
        fake_create_method_end_time_and_obj_value_summary,
    )

    runner = HfsMultiInstanceRunner.__new__(HfsMultiInstanceRunner)
    runner.working_dir = tmp_path
    runner.baseline_df = pd.DataFrame()
    runner.results = []

    out_df = runner.post_run_process()

    assert captured_kwargs["record_all_subroutines"] is True
    assert out_df.empty
