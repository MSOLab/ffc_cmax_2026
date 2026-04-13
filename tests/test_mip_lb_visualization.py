from pathlib import Path

from lb_bucket.mip.visualization import write_solution_payload_visualizations


def test_write_solution_payload_visualizations_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    payload = {
        "metadata": {
            "ins_name": "1",
            "job_count": 2,
            "stage_count": 2,
            "machine_count_per_stage": [1, 2],
            "delta": 10,
            "t_upper": 4,
            "dispatch_cmax": 40.0,
        },
        "a": [
            {"stage": 1, "job": 1, "bucket": 1, "value": 1.0},
            {"stage": 2, "job": 1, "bucket": 2, "value": 1.0},
            {"stage": 1, "job": 2, "bucket": 2, "value": 1.0},
            {"stage": 2, "job": 2, "bucket": 3, "value": 1.0},
        ],
        "b": [
            {"stage": 1, "job": 1, "bucket": 1, "value": 1.0},
            {"stage": 2, "job": 1, "bucket": 2, "value": 1.0},
            {"stage": 1, "job": 2, "bucket": 2, "value": 1.0},
            {"stage": 2, "job": 2, "bucket": 3, "value": 1.0},
        ],
        "x": [
            {"stage": 1, "job": 1, "bucket": 1, "value": 6.0},
            {"stage": 1, "job": 1, "bucket": 2, "value": 4.0},
            {"stage": 2, "job": 1, "bucket": 2, "value": 8.0},
            {"stage": 1, "job": 2, "bucket": 2, "value": 7.0},
            {"stage": 2, "job": 2, "bucket": 3, "value": 5.0},
        ],
        "dispatch_windows": [
            {
                "stage": 1,
                "job": 1,
                "processing_time": 10,
                "a_bucket": 1,
                "b_bucket": 2,
                "x_bucket_1": 1,
                "x_value_1": 6.0,
                "x_bucket_2": 2,
                "x_value_2": 4.0,
                "x_at_a_bucket": 6.0,
                "x_at_b_bucket": 4.0,
                "spans_two_buckets": True,
                "es_candidate": 0.0,
                "completion_candidate": 14.0,
                "early_start": 0.0,
                "late_start": 4.0,
                "slack": 4.0,
            },
            {
                "stage": 2,
                "job": 1,
                "processing_time": 8,
                "a_bucket": 2,
                "b_bucket": 2,
                "x_bucket_1": 2,
                "x_value_1": 8.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 8.0,
                "x_at_b_bucket": 8.0,
                "spans_two_buckets": False,
                "es_candidate": 12.0,
                "completion_candidate": 20.0,
                "early_start": 12.0,
                "late_start": 12.0,
                "slack": 0.0,
            },
            {
                "stage": 1,
                "job": 2,
                "processing_time": 7,
                "a_bucket": 2,
                "b_bucket": 2,
                "x_bucket_1": 2,
                "x_value_1": 7.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 7.0,
                "x_at_b_bucket": 7.0,
                "spans_two_buckets": False,
                "es_candidate": 10.0,
                "completion_candidate": 20.0,
                "early_start": 10.0,
                "late_start": 13.0,
                "slack": 3.0,
            },
            {
                "stage": 2,
                "job": 2,
                "processing_time": 5,
                "a_bucket": 3,
                "b_bucket": 3,
                "x_bucket_1": 3,
                "x_value_1": 5.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 5.0,
                "x_at_b_bucket": 5.0,
                "spans_two_buckets": False,
                "es_candidate": 20.0,
                "completion_candidate": 30.0,
                "early_start": 20.0,
                "late_start": 25.0,
                "slack": 5.0,
            },
        ],
    }

    write_solution_payload_visualizations(tmp_path, payload)

    plot_root = tmp_path / "solutions" / "1" / "plots"
    assert (plot_root / "stage_bucket_x_utilization.csv").is_file()
    assert (plot_root / "stage_bucket_x_occupied_time.png").is_file()
    assert (plot_root / "stage_bucket_x_utilization.png").is_file()
    assert (plot_root / "bucket_heatmaps" / "x_page01.png").is_file()
    assert (plot_root / "bucket_heatmaps" / "a_page01.png").is_file()
    assert (plot_root / "bucket_heatmaps" / "b_page01.png").is_file()
    assert (plot_root / "dispatch_windows" / "dispatch_windows_page01.png").is_file()
