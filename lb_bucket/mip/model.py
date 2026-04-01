from __future__ import annotations

import math
from typing import Any

from .shared import ModelStrengtheningOptions, TwoBucketInstance


def compute_upstream_totals(instance: TwoBucketInstance) -> dict[tuple[int, int], int]:
    upstream: dict[tuple[int, int], int] = {}
    for job_idx in range(1, instance.job_count + 1):
        running_total = 0
        for stage_idx in range(1, instance.stage_count + 1):
            upstream[stage_idx, job_idx] = running_total
            running_total += instance.processing_times_by_stage[stage_idx - 1][job_idx - 1]
    return upstream


def compute_downstream_totals(instance: TwoBucketInstance) -> dict[tuple[int, int], int]:
    downstream: dict[tuple[int, int], int] = {}
    for job_idx in range(1, instance.job_count + 1):
        running_total = 0
        for stage_idx in range(instance.stage_count, 0, -1):
            downstream[stage_idx, job_idx] = running_total
            running_total += instance.processing_times_by_stage[stage_idx - 1][job_idx - 1]
    return downstream


def build_two_bucket_model(
    gp: Any,
    grb: Any,
    instance: TwoBucketInstance,
    bucket_count: int,
    delta: int,
    strengthening: ModelStrengtheningOptions,
) -> tuple[Any, Any]:
    model = gp.Model(name=f"bucket_lb_{instance.ins_name}_T{bucket_count}")

    stage_index = range(1, instance.stage_count + 1)
    job_index = range(1, instance.job_count + 1)
    bucket_index = range(1, bucket_count + 1)

    processing_time = {
        (i, j): instance.processing_times_by_stage[i - 1][j - 1]
        for i in stage_index
        for j in job_index
    }
    machine_count = {i: instance.machine_count_per_stage[i - 1] for i in stage_index}
    upstream_total = compute_upstream_totals(instance)
    downstream_total = compute_downstream_totals(instance)
    earliest_bucket = {
        (i, j): 1 + math.floor(upstream_total[i, j] / delta)
        for i in stage_index
        for j in job_index
    }
    earliest_residual = {
        (i, j): earliest_bucket[i, j] * delta - upstream_total[i, j]
        for i in stage_index
        for j in job_index
    }
    latest_bucket = {
        (i, j): bucket_count - math.floor(downstream_total[i, j] / delta)
        for i in stage_index
        for j in job_index
    }
    latest_residual = {
        (i, j): bucket_count * delta
        - downstream_total[i, j]
        - (latest_bucket[i, j] - 1) * delta
        for i in stage_index
        for j in job_index
    }
    total_processing_per_job = {
        j: sum(processing_time[i, j] for i in stage_index) for j in job_index
    }
    stage_cut_rhs = {}
    for i in stage_index:
        head_values = sorted(upstream_total[i, j] for j in job_index)
        tail_values = sorted(downstream_total[i, j] for j in job_index)
        machine_prefix = min(machine_count[i], instance.job_count)
        stage_cut_rhs[i] = (
            sum(head_values[:machine_prefix])
            + sum(processing_time[i, j] for j in job_index)
            + sum(tail_values[:machine_prefix])
        )
    global_lb_s = max(
        max(total_processing_per_job.values(), default=0),
        max(stage_cut_rhs[i] / machine_count[i] for i in stage_index),
    )

    a = model.addVars(stage_index, job_index, bucket_index, vtype=grb.BINARY, name="a")
    b = model.addVars(stage_index, job_index, bucket_index, vtype=grb.BINARY, name="b")
    c = model.addVars(
        stage_index,
        job_index,
        bucket_index,
        lb=0.0,
        ub=1.0,
        vtype=grb.CONTINUOUS,
        name="c",
    )
    x = model.addVars(
        stage_index,
        job_index,
        bucket_index,
        lb=0.0,
        vtype=grb.CONTINUOUS,
        name="x",
    )
    z = model.addVar(lb=0.0, ub=float(delta), vtype=grb.INTEGER, name="z")

    model.setObjective(z, grb.MINIMIZE)

    model.addConstrs(
        (
            gp.quicksum(a[i, j, t] for t in bucket_index) == 1
            for i in stage_index
            for j in job_index
        ),
        name="start_one",
    )
    model.addConstrs(
        (
            gp.quicksum(b[i, j, t] for t in bucket_index) == 1
            for i in stage_index
            for j in job_index
        ),
        name="end_one",
    )

    model.addConstrs(
        (
            a[i, j, t] <= b[i, j, t] + b[i, j, t + 1]
            for i in stage_index
            for j in job_index
            for t in range(1, bucket_count)
        ),
        name="adjacent_1",
    )
    model.addConstrs(
        (
            a[i, j, bucket_count] <= b[i, j, bucket_count]
            for i in stage_index
            for j in job_index
        ),
        name="adjacent_2",
    )
    model.addConstrs(
        (b[i, j, 1] <= a[i, j, 1] for i in stage_index for j in job_index),
        name="sym_adjacent_1",
    )
    model.addConstrs(
        (
            b[i, j, t] <= a[i, j, t] + a[i, j, t - 1]
            for i in stage_index
            for j in job_index
            for t in range(2, bucket_count + 1)
        ),
        name="sym_adjacent_2",
    )

    model.addConstrs(
        (
            gp.quicksum(x[i, j, t] for t in bucket_index) == processing_time[i, j]
            for i in stage_index
            for j in job_index
        ),
        name="processing_conservation",
    )
    model.addConstrs(
        (
            x[i, j, t] <= processing_time[i, j] * (a[i, j, t] + b[i, j, t])
            for i in stage_index
            for j in job_index
            for t in bucket_index
        ),
        name="bucket_activation",
    )
    model.addConstrs(
        (
            x[i, j, t] >= processing_time[i, j] * c[i, j, t]
            for i in stage_index
            for j in job_index
            for t in bucket_index
        ),
        name="same_bucket_activation",
    )
    model.addConstrs(
        (
            c[i, j, t] >= a[i, j, t] + b[i, j, t] - 1
            for i in stage_index
            for j in job_index
            for t in bucket_index
        ),
        name="c_def_1",
    )
    model.addConstrs(
        (
            a[i, j, t] >= c[i, j, t]
            for i in stage_index
            for j in job_index
            for t in bucket_index
        ),
        name="c_def_2",
    )
    model.addConstrs(
        (
            b[i, j, t] >= c[i, j, t]
            for i in stage_index
            for j in job_index
            for t in bucket_index
        ),
        name="c_def_3",
    )
    if bucket_count >= 2:
        model.addConstrs(
            (
                gp.quicksum(a[i, j, t] - c[i, j, t] for j in job_index) <= machine_count[i]
                for i in stage_index
                for t in range(1, bucket_count)
            ),
            name="machine_capacity",
        )
    model.addConstrs(
        (
            x[i, j, t] >= a[i, j, t]
            for i in stage_index
            for j in job_index
            for t in bucket_index
        ),
        name="positive_start",
    )
    model.addConstrs(
        (
            x[i, j, t] >= b[i, j, t]
            for i in stage_index
            for j in job_index
            for t in bucket_index
        ),
        name="positive_end",
    )

    if strengthening.valid_ineq_i:
        model.addConstrs(
            (
                a[i, j, t] == 0
                for i in stage_index
                for j in job_index
                for t in bucket_index
                if t < earliest_bucket[i, j]
            ),
            name="rq_fix_earliest_a",
        )
        model.addConstrs(
            (
                b[i, j, t] == 0
                for i in stage_index
                for j in job_index
                for t in bucket_index
                if t < earliest_bucket[i, j]
            ),
            name="rq_fix_earliest_b",
        )
        model.addConstrs(
            (
                x[i, j, t] == 0
                for i in stage_index
                for j in job_index
                for t in bucket_index
                if t < earliest_bucket[i, j]
            ),
            name="rq_fix_earliest_x",
        )
        model.addConstrs(
            (
                a[i, j, t] == 0
                for i in stage_index
                for j in job_index
                for t in bucket_index
                if t > latest_bucket[i, j]
            ),
            name="rq_fix_latest_a",
        )
        model.addConstrs(
            (
                b[i, j, t] == 0
                for i in stage_index
                for j in job_index
                for t in bucket_index
                if t > latest_bucket[i, j]
            ),
            name="rq_fix_latest_b",
        )
        model.addConstrs(
            (
                x[i, j, t] == 0
                for i in stage_index
                for j in job_index
                for t in bucket_index
                if t > latest_bucket[i, j]
            ),
            name="rq_fix_latest_x",
        )
        model.addConstrs(
            (
                x[i, j, earliest_bucket[i, j]] <= earliest_residual[i, j]
                for i in stage_index
                for j in job_index
                if earliest_bucket[i, j] <= bucket_count
            ),
            name="rq_earliest_residual",
        )
        model.addConstrs(
            (
                x[i, j, latest_bucket[i, j]] <= latest_residual[i, j]
                for i in stage_index
                for j in job_index
                if 1 <= latest_bucket[i, j] <= bucket_count
            ),
            name="rq_latest_residual",
        )

    if bucket_count >= 2:
        model.addConstrs(
            (
                gp.quicksum(x[i, j, t] for j in job_index) <= machine_count[i] * delta
                for i in stage_index
                for t in range(1, bucket_count)
            ),
            name="stage_capacity",
        )
        model.addConstrs(
            (
                gp.quicksum(x[i, j, t] for i in stage_index) <= delta
                for j in job_index
                for t in range(1, bucket_count)
            ),
            name="job_bucket_capacity",
        )

    model.addConstrs(
        (
            b[i, j, t]
            <= gp.quicksum(a[i + 1, j, tau] for tau in range(t, bucket_count + 1))
            for i in range(1, instance.stage_count)
            for j in job_index
            for t in bucket_index
        ),
        name="precedence_bucket",
    )
    if strengthening.cumulative_precedence:
        model.addConstrs(
            (
                gp.quicksum(a[i + 1, j, tau] for tau in range(1, t + 1))
                <= gp.quicksum(b[i, j, tau] for tau in range(1, t + 1))
                for i in range(1, instance.stage_count)
                for j in job_index
                for t in bucket_index
            ),
            name="cumulative_precedence",
        )

    if strengthening.valid_ineq_ii:
        model.addConstrs(
            (
                upstream_total[i, j] * a[i, j, t]
                <= gp.quicksum(
                    x[k, j, tau]
                    for k in range(1, i)
                    for tau in range(1, t + 1)
                )
                for i in range(2, instance.stage_count + 1)
                for j in job_index
                for t in bucket_index
            ),
            name="rq_head_link",
        )
        model.addConstrs(
            (
                downstream_total[i, j] * b[i, j, t]
                <= gp.quicksum(
                    x[k, j, tau]
                    for k in range(i + 1, instance.stage_count + 1)
                    for tau in range(t, bucket_count + 1)
                )
                for i in range(1, instance.stage_count)
                for j in job_index
                for t in bucket_index
            ),
            name="rq_tail_link",
        )

    model.addConstrs(
        (
            z >= gp.quicksum(x[i, j, bucket_count] for i in stage_index)
            for j in job_index
        ),
        name="z_single",
    )
    model.addConstrs(
        (
            machine_count[i] * z
            >= gp.quicksum(x[i, j, bucket_count] for j in job_index)
            for i in stage_index
        ),
        name="z_average",
    )

    if strengthening.valid_ineq_iii:
        model.addConstrs(
            (
                (bucket_count - 1) * delta + z >= total_processing_per_job[j]
                for j in job_index
            ),
            name="rq_job_chain_cut",
        )

    if strengthening.valid_ineq_iv:
        model.addConstrs(
            (
                machine_count[i] * ((bucket_count - 1) * delta + z) >= stage_cut_rhs[i]
                for i in stage_index
            ),
            name="rq_stage_cut",
        )
        model.addConstr(
            (bucket_count - 1) * delta + z >= global_lb_s, name="rq_global_cut"
        )

    return model, z
