from pathlib import Path
import json

import pandas as pd
import yaml

from hybridflowshop.report.log_processor import (
    LogProcessor,
    create_method_end_time_and_obj_value_summary,
)


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


def _write_progression_json(instance_dir: Path, progression_data: dict) -> None:
    results_dir = instance_dir / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "subroutine_progression.json", "w", encoding="utf-8") as f:
        json.dump(progression_data, f, indent=2)


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


def test_log_processor_default_keeps_missing_subroutines_blank(tmp_path: Path) -> None:
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

    processor = LogProcessor(scenario_dir)
    summary_df = processor.create_method_end_time_and_obj_value_summary()

    assert summary_df is not None
    assert list(summary_df["subroutine_name"]) == ["first", "second", "third"]

    third_row = summary_df[summary_df["subroutine_name"] == "third"].iloc[0]
    assert pd.isna(third_row["end_time"])
    assert pd.isna(third_row["obj_value"])

    instance_df = pd.read_csv(instance_dir / "method_end_time_and_obj_value.csv")
    third_instance_row = instance_df[instance_df["method_name"] == "third"].iloc[0]
    assert pd.isna(third_instance_row["method_end_sec"])
    assert pd.isna(third_instance_row["objective_value"])


def test_log_processor_record_all_subroutines_fills_trailing(
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

    processor = LogProcessor(scenario_dir, record_all_subroutines=True)
    summary_df = processor.create_method_end_time_and_obj_value_summary()

    assert summary_df is not None
    third_row = summary_df[summary_df["subroutine_name"] == "third"].iloc[0]
    assert third_row["end_time"] == 3.0
    assert third_row["obj_value"] == 90.0

    instance_df = pd.read_csv(instance_dir / "method_end_time_and_obj_value.csv")
    third_instance_row = instance_df[instance_df["method_name"] == "third"].iloc[0]
    assert third_instance_row["method_end_sec"] == 3.0
    assert third_instance_row["objective_value"] == 90.0


def test_log_processor_omitted_subroutines_and_wide_columns(tmp_path: Path) -> None:
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

    processor = LogProcessor(
        scenario_dir,
        record_all_subroutines=True,
        omitted_subroutines={"second"},
    )
    summary_df = processor.create_method_end_time_and_obj_value_summary()

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


def test_log_processor_baseline_metadata_merge(tmp_path: Path) -> None:
    scenario_dir, _ = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second"],
        log_entries=[
            {
                "method": "first",
                "call_context": "1-first",
                "start_sec": 0.0,
                "kwargs": {},
                "elapsed_sec": 1.0,
            },
        ],
        obj_data={"1.0": 100.0},
        obj_notes={"1.0": "1-first"},
    )

    baseline_df = pd.DataFrame({"Instance": ["1"], "n": [10], "s": [3], "UB": [95.0]})

    processor = LogProcessor(
        scenario_dir,
        baseline_df=baseline_df,
        baseline_instance_col="Instance",
        baseline_job_cnt_col="n",
        baseline_stage_cnt_col="s",
        baseline_obj_val_col="UB",
    )
    summary_df = processor.create_method_end_time_and_obj_value_summary()

    assert summary_df is not None
    assert "job_cnt" in summary_df.columns
    assert "stage_cnt" in summary_df.columns
    assert "ref_obj_value" in summary_df.columns

    first_row = summary_df[summary_df["subroutine_name"] == "first"].iloc[0]
    assert first_row["job_cnt"] == 10
    assert first_row["stage_cnt"] == 3
    assert first_row["ref_obj_value"] == 95.0


def test_log_processor_record_all_subroutines_blank_when_nothing_executed(
    tmp_path: Path,
) -> None:
    scenario_dir, instance_dir = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second"],
        log_entries=[],
    )

    processor = LogProcessor(scenario_dir, record_all_subroutines=True)
    summary_df = processor.create_method_end_time_and_obj_value_summary()

    assert summary_df is not None
    assert len(summary_df) == 2
    assert summary_df["end_time"].isna().all()
    assert summary_df["obj_value"].isna().all()

    instance_df = pd.read_csv(instance_dir / "method_end_time_and_obj_value.csv")
    assert instance_df["method_end_sec"].isna().all()
    assert instance_df["objective_value"].isna().all()


def test_log_processor_falls_back_to_progression_json_when_controller_log_is_empty(
    tmp_path: Path,
) -> None:
    scenario_dir, instance_dir = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second", "third"],
        log_entries=[],
        obj_data={"1.0": 100.0, "3.0": 90.0},
        obj_notes={"1.0": "1-first", "3.0": "2-second"},
    )
    (instance_dir / "subroutine_controller.log").write_text("", encoding="utf-8")
    _write_progression_json(
        instance_dir,
        {
            "artifact_version": 1,
            "instance_id": "1",
            "timelimit_sec": 10.0,
            "subroutine_calls": [
                {
                    "call_index": 1,
                    "subroutine_name": "first",
                    "prefixed_subroutine_name": "1-first",
                    "global_start_sec": 0.0,
                    "global_end_sec": 1.0,
                    "elapsed_sec": 1.0,
                    "local_progress_list": [],
                },
                {
                    "call_index": 2,
                    "subroutine_name": "second",
                    "prefixed_subroutine_name": "2-second",
                    "global_start_sec": 1.0,
                    "global_end_sec": 3.0,
                    "elapsed_sec": 2.0,
                    "local_progress_list": [],
                },
            ],
        },
    )

    processor = LogProcessor(scenario_dir, record_all_subroutines=True)
    summary_df = processor.create_method_end_time_and_obj_value_summary()

    assert summary_df is not None
    assert list(summary_df["end_time"]) == [1.0, 3.0, 3.0]
    assert list(summary_df["obj_value"]) == [100.0, 90.0, 90.0]

    method_time_log = pd.read_json(instance_dir / "method_time_log.json")
    assert list(method_time_log["call_context"]) == ["1-first", "2-second"]


def test_module_level_wrapper_matches_class_api(tmp_path: Path) -> None:
    scenario_dir, _ = _create_scenario(
        tmp_path=tmp_path,
        methods=["first", "second"],
        log_entries=[
            {
                "method": "first",
                "call_context": "1-first",
                "start_sec": 0.0,
                "kwargs": {},
                "elapsed_sec": 1.0,
            },
        ],
        obj_data={"1.0": 100.0},
        obj_notes={"1.0": "1-first"},
    )

    class_df = LogProcessor(
        scenario_dir, record_all_subroutines=True
    ).create_method_end_time_and_obj_value_summary()

    wrapper_df = create_method_end_time_and_obj_value_summary(
        scenario_dir, record_all_subroutines=True
    )

    assert class_df is not None
    assert wrapper_df is not None
    pd.testing.assert_frame_equal(class_df, wrapper_df)


def test_wrapper_import_from_scripts_process_logs(tmp_path: Path) -> None:
    from scripts.process_logs import (
        create_method_end_time_and_obj_value_summary as wrapper_func,
    )

    scenario_dir, _ = _create_scenario(
        tmp_path=tmp_path,
        methods=["first"],
        log_entries=[
            {
                "method": "first",
                "call_context": "1-first",
                "start_sec": 0.0,
                "kwargs": {},
                "elapsed_sec": 1.0,
            },
        ],
        obj_data={"1.0": 100.0},
        obj_notes={"1.0": "1-first"},
    )

    df = wrapper_func(scenario_dir)
    assert df is not None
    assert list(df["subroutine_name"]) == ["first"]
