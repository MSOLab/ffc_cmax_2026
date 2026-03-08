import pytest

from hfs_config import MainMetadata

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
