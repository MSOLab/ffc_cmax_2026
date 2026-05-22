import pytest

from hfs_config import MainMetadata
from main import order_instances_for_execution

BASE_CONFIG = {
    "pra_common_params_rel_path": "resources/pra_common_params.yaml",
    "baseline_csv_path": "resources/pra_ref/githubData.csv",
    "input_dir": "resources/pra",
    "benchmark_idx_list": [1, 2],
    "benchmark_filename_format": "{}.txt",
    "dicts_of_i_o_data_path": [
        {
            "subroutine_flow_rel_path": "configs_100s/subroutine_flow_baseCp.yaml",
            "stopping_criteria_rel_path": "configs_100s/stopping_criteria.yaml",
            "output_dir": "pra_100s/baseCp",
            "description": "base",
        }
    ],
}


def test_timepoint_summaries_valid():
    cfg = {
        **BASE_CONFIG,
        "timepoint_summaries": [
            {"label": "10p", "mode": "timelimit_ratio", "value": 0.1},
            {
                "label": "100s",
                "mode": "absolute_sec",
                "value": 100,
                "exclude_if_timelimit_lt": 100,
            },
        ],
    }

    m = MainMetadata.model_validate(cfg)

    assert m.timepoint_summaries is not None
    assert len(m.timepoint_summaries) == 2
    assert m.timepoint_summaries[0].label == "10p"


def test_timepoint_summaries_reject_duplicate_labels():
    cfg = {
        **BASE_CONFIG,
        "timepoint_summaries": [
            {"label": "x", "mode": "timelimit_ratio", "value": 0.1},
            {"label": "x", "mode": "absolute_sec", "value": 50},
        ],
    }

    with pytest.raises(ValueError, match="labels must be unique"):
        MainMetadata.model_validate(cfg)


def test_timepoint_summaries_reject_non_positive_value():
    cfg = {
        **BASE_CONFIG,
        "timepoint_summaries": [
            {"label": "bad", "mode": "absolute_sec", "value": 0},
        ],
    }

    with pytest.raises(ValueError, match="must be > 0"):
        MainMetadata.model_validate(cfg)


def test_timepoint_summaries_reject_ratio_out_of_range():
    cfg = {
        **BASE_CONFIG,
        "timepoint_summaries": [
            {"label": "bad", "mode": "timelimit_ratio", "value": 1.5},
        ],
    }

    with pytest.raises(ValueError, match="0 < value <= 1"):
        MainMetadata.model_validate(cfg)


def test_rejects_non_positive_benchmark_order_wave_size():
    cfg = {
        **BASE_CONFIG,
        "benchmark_order_strategy": "mixed_stratified",
        "benchmark_order_wave_size": 0,
    }

    with pytest.raises(ValueError, match="benchmark_order_wave_size must be positive"):
        MainMetadata.model_validate(cfg)


def test_mixed_stratified_order_interleaves_size_bands():
    class FakeInstance:
        def __init__(self, name: str, job_count: int, stage_count: int):
            self.name = name
            self.job_count = job_count
            self.stage_count = stage_count
            self.machine_count_per_stage = [3] * stage_count

    cfg = MainMetadata.model_validate(
        {
            **BASE_CONFIG,
            "benchmark_order_strategy": "mixed_stratified",
            "benchmark_order_seed": 42,
            "benchmark_order_wave_size": 4,
        }
    )
    instances = [
        FakeInstance("small_a", 40, 5),
        FakeInstance("small_b", 40, 10),
        FakeInstance("medium_a", 80, 15),
        FakeInstance("medium_b", 120, 10),
        FakeInstance("large_a", 200, 15),
        FakeInstance("large_b", 240, 10),
        FakeInstance("huge_a", 240, 20),
        FakeInstance("huge_b", 200, 20),
    ]

    ordered = order_instances_for_execution(instances, cfg)
    first_wave = ordered[:4]
    first_wave_workloads = [
        ins.job_count * sum(ins.machine_count_per_stage) for ins in first_wave
    ]

    assert len(ordered) == len(instances)
    assert {ins.name for ins in ordered} == {ins.name for ins in instances}
    assert max(first_wave_workloads) > min(first_wave_workloads)
    assert len({ins.name.split("_")[0] for ins in first_wave}) == 4
