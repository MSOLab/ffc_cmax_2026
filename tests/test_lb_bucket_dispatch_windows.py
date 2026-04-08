from lb_bucket.mip.dispatch_windows import (
    build_dispatch_window_lookup,
    compute_dispatch_window_payload,
)
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
