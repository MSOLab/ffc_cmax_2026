from lb_bucket.mip.dispatch_windows import (
    build_dispatch_window_lookup,
    compute_dispatch_window_payload,
)
from lb_bucket.mip.solution_io import read_solution_payload, write_solution_payload
from lb_bucket.mip.shared import TwoBucketInstance


def test_compute_dispatch_window_payload_builds_es_ls_rows() -> None:
    instance = TwoBucketInstance(
        ins_name="toy",
        job_count=1,
        stage_count=2,
        machine_count_per_stage=[1, 1],
        processing_times_by_stage=[[3], [7]],
    )
    solution_payload = {
        "metadata": {
            "delta": 10,
            "input_ub": 20,
            "value_tolerance": 1e-9,
        },
        "a": [
            {"stage": 1, "job": 1, "bucket": 1, "value": 1.0},
            {"stage": 2, "job": 1, "bucket": 1, "value": 1.0},
        ],
        "b": [
            {"stage": 1, "job": 1, "bucket": 1, "value": 1.0},
            {"stage": 2, "job": 1, "bucket": 2, "value": 1.0},
        ],
        "x": [
            {"stage": 1, "job": 1, "bucket": 1, "value": 3.0},
            {"stage": 2, "job": 1, "bucket": 1, "value": 3.0},
            {"stage": 2, "job": 1, "bucket": 2, "value": 4.0},
        ],
    }

    dispatch_payload = compute_dispatch_window_payload(instance, solution_payload)
    lookup = build_dispatch_window_lookup(dispatch_payload["dispatch_windows"])

    op_11 = lookup[1, 1]
    assert op_11["a_bucket"] == 1
    assert op_11["b_bucket"] == 1
    assert op_11["early_start"] == 0.0
    assert op_11["late_start"] == 4.0

    op_21 = lookup[2, 1]
    assert op_21["a_bucket"] == 1
    assert op_21["b_bucket"] == 2
    assert op_21["x_at_b_bucket"] == 4.0
    assert op_21["early_start"] == 7.0
    assert op_21["late_start"] == 7.0


def test_solution_payload_roundtrip_preserves_dispatch_windows(tmp_path) -> None:
    payload = {
        "metadata": {
            "ins_name": "toy",
            "delta": 10,
            "dispatch_cmax": 20.0,
        },
        "a": [{"stage": 1, "job": 1, "bucket": 1, "value": 1.0}],
        "b": [{"stage": 1, "job": 1, "bucket": 1, "value": 1.0}],
        "c": [],
        "x": [{"stage": 1, "job": 1, "bucket": 1, "value": 3.0}],
        "u": [{"bucket": 2, "value": 1.0}],
        "z": [{"bucket": 2, "value": 3.0}],
        "dispatch_window_inputs": [
            {
                "stage": 1,
                "job": 1,
                "processing_time": 3,
                "a_bucket": 1,
                "b_bucket": 1,
                "x_bucket_1": 1,
                "x_value_1": 3.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 3.0,
                "x_at_b_bucket": 3.0,
                "spans_two_buckets": False,
            }
        ],
        "dispatch_windows": [
            {
                "stage": 1,
                "job": 1,
                "processing_time": 3,
                "a_bucket": 1,
                "b_bucket": 1,
                "x_bucket_1": 1,
                "x_value_1": 3.0,
                "x_bucket_2": None,
                "x_value_2": None,
                "x_at_a_bucket": 3.0,
                "x_at_b_bucket": 3.0,
                "spans_two_buckets": False,
                "es_candidate": 0.0,
                "completion_candidate": 10.0,
                "early_start": 0.0,
                "late_start": 7.0,
                "slack": 7.0,
            }
        ],
    }

    write_solution_payload(tmp_path, payload)
    loaded = read_solution_payload(tmp_path, "toy")

    assert loaded is not None
    assert loaded["metadata"]["ins_name"] == "toy"
    assert loaded["dispatch_windows"][0]["early_start"] == 0.0
    assert loaded["dispatch_windows"][0]["late_start"] == 7.0
    assert loaded["dispatch_windows"][0]["x_bucket_2"] is None
