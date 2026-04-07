from __future__ import annotations

import math
from typing import Any

from .data import compute_range_bucket_bounds
from .shared import (
    BucketModelVars,
    ModelStrengtheningOptions,
    PrecedenceOptions,
    TwoBucketInstance,
)


def _var_or_zero(var_dict: Any, key: tuple[int, int, int]) -> Any:
    if key in var_dict:
        return var_dict[key]
    return 0.0


def _var4_or_zero(var_dict: Any, key: tuple[int, int, int, int]) -> Any:
    if key in var_dict:
        return var_dict[key]
    return 0.0


def compute_upstream_totals(instance: TwoBucketInstance) -> dict[tuple[int, int], int]:
    upstream: dict[tuple[int, int], int] = {}
    for job_idx in range(1, instance.job_count + 1):
        running_total = 0
        for stage_idx in range(1, instance.stage_count + 1):
            upstream[stage_idx, job_idx] = running_total
            running_total += instance.processing_times_by_stage[stage_idx - 1][
                job_idx - 1
            ]
    return upstream


def compute_downstream_totals(
    instance: TwoBucketInstance,
) -> dict[tuple[int, int], int]:
    downstream: dict[tuple[int, int], int] = {}
    for job_idx in range(1, instance.job_count + 1):
        running_total = 0
        for stage_idx in range(instance.stage_count, 0, -1):
            downstream[stage_idx, job_idx] = running_total
            running_total += instance.processing_times_by_stage[stage_idx - 1][
                job_idx - 1
            ]
    return downstream


def build_two_bucket_model(
    gp: Any,
    grb: Any,
    instance: TwoBucketInstance,
    delta: int,
    input_lb: int,
    input_ub: int,
    strengthening: ModelStrengtheningOptions,
    precedence: PrecedenceOptions,
) -> tuple[Any, BucketModelVars]:
    t_lower, t_upper = compute_range_bucket_bounds(input_lb, input_ub, delta)
    model = gp.Model(name=f"bucket_lb_{instance.ins_name}_TL{t_lower}_TU{t_upper}")

    stage_index = range(1, instance.stage_count + 1)
    job_index = range(1, instance.job_count + 1)
    bucket_index = range(1, t_upper + 1)
    range_bucket_index = range(t_lower + 1, t_upper + 1)
    fixed_bucket_index = range(1, t_lower + 1)

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
        (i, j): math.ceil((input_ub - downstream_total[i, j]) / delta)
        for i in stage_index
        for j in job_index
    }
    latest_residual = {
        (i, j): input_ub - downstream_total[i, j] - (latest_bucket[i, j] - 1) * delta
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

    operation_window = {
        (i, j): list(
            range(
                max(1, earliest_bucket[i, j]),
                min(t_upper, latest_bucket[i, j]) + 1,
            )
        )
        for i in stage_index
        for j in job_index
    }
    feasible_bucket_keys = [
        (i, j, t)
        for i in stage_index
        for j in job_index
        for t in operation_window[i, j]
    ]
    feasible_key_set = set(feasible_bucket_keys)
    bucket_keys_by_stage = {
        (i, t): [(i, j, t) for j in job_index if (i, j, t) in feasible_key_set]
        for i in stage_index
        for t in bucket_index
    }
    bucket_keys_by_job = {
        (j, t): [(i, j, t) for i in stage_index if (i, j, t) in feasible_key_set]
        for j in job_index
        for t in bucket_index
    }

    a = model.addVars(feasible_bucket_keys, vtype=grb.BINARY, name="a")
    b = model.addVars(feasible_bucket_keys, vtype=grb.BINARY, name="b")
    c = model.addVars(
        feasible_bucket_keys,
        lb=0.0,
        ub=1.0,
        vtype=grb.BINARY,
        name="c",
    )
    x = model.addVars(
        feasible_bucket_keys,
        lb=0.0,
        vtype=grb.CONTINUOUS,
        name="x",
    )
    u = model.addVars(range_bucket_index, vtype=grb.BINARY, name="u")
    d = None
    e = None
    if precedence.formulation == "d":
        d = model.addVars(
            [
                (i, j, t)
                for i in range(1, instance.stage_count)
                for j in job_index
                for t in bucket_index
            ],
            lb=-1.0,
            ub=1.0,
            vtype=grb.CONTINUOUS,
            name="d",
        )
    elif precedence.formulation == "e":
        e_keys = [
            (i, j, t, t_prime)
            for i in range(1, instance.stage_count)
            for j in job_index
            for t in operation_window[i, j]
            for t_prime in operation_window[i + 1, j]
            if t <= t_prime
        ]
        e = model.addVars(e_keys, vtype=grb.BINARY, name="e")
    z = {}
    for bucket_idx in range_bucket_index:
        lower_bound = 0.0
        upper_bound = float(delta)
        if bucket_idx == t_lower + 1:
            lower_bound = float(input_lb - t_lower * delta)
        if bucket_idx == t_upper:
            upper_bound = float(input_ub - (t_upper - 1) * delta)
        z[bucket_idx] = model.addVar(
            lb=lower_bound,
            ub=upper_bound,
            vtype=grb.INTEGER,
            name=f"z[{bucket_idx}]",
        )

    w_expr = t_lower * delta + gp.quicksum(z[t] for t in range_bucket_index)
    model.setObjective(gp.quicksum(z[t] for t in range_bucket_index), grb.MINIMIZE)

    model.addConstrs(
        (
            gp.quicksum(a[i, j, t] for t in operation_window[i, j]) == 1
            for i in stage_index
            for j in job_index
        ),
        name="start_one",
    )
    model.addConstrs(
        (
            gp.quicksum(b[i, j, t] for t in operation_window[i, j]) == 1
            for i in stage_index
            for j in job_index
        ),
        name="end_one",
    )
    model.addConstrs(
        (
            a[i, j, t] <= _var_or_zero(b, (i, j, t)) + _var_or_zero(b, (i, j, t + 1))
            for i, j, t in feasible_bucket_keys
            if t < t_upper
        ),
        name="adjacent_1",
    )
    model.addConstrs(
        (
            a[i, j, t_upper] <= b[i, j, t_upper]
            for i in stage_index
            for j in job_index
            if (i, j, t_upper) in feasible_key_set
        ),
        name="adjacent_2",
    )
    model.addConstrs(
        (
            b[i, j, 1] <= a[i, j, 1]
            for i in stage_index
            for j in job_index
            if (i, j, 1) in feasible_key_set
        ),
        name="sym_adjacent_1",
    )
    model.addConstrs(
        (
            b[i, j, t] <= _var_or_zero(a, (i, j, t)) + _var_or_zero(a, (i, j, t - 1))
            for i, j, t in feasible_bucket_keys
            if t > 1
        ),
        name="sym_adjacent_2",
    )
    if precedence.formulation == "bucket":
        model.addConstrs(
            (
                a[i, j, t]
                <= gp.quicksum(
                    _var_or_zero(b, (i - 1, j, tau)) for tau in range(1, t + 1)
                )
                for i in range(2, instance.stage_count + 1)
                for j in job_index
                for t in operation_window[i, j]
            ),
            name="precedence_bucket_prefix",
        )
        model.addConstrs(
            (
                b[i, j, t]
                <= gp.quicksum(
                    a[i + 1, j, tau] for tau in operation_window[i + 1, j] if tau >= t
                )
                for i in range(1, instance.stage_count)
                for j in job_index
                for t in operation_window[i, j]
            ),
            name="precedence_bucket",
        )
    elif precedence.formulation == "d":
        model.addConstrs(
            (
                d[i, j, 1]
                == _var_or_zero(a, (i + 1, j, 1)) - _var_or_zero(b, (i, j, 1))
                for i in range(1, instance.stage_count)
                for j in job_index
            ),
            name="precedence_d_init",
        )
        if t_upper >= 2:
            model.addConstrs(
                (
                    d[i, j, t]
                    == d[i, j, t - 1]
                    + _var_or_zero(a, (i + 1, j, t))
                    - _var_or_zero(b, (i, j, t))
                    for i in range(1, instance.stage_count)
                    for j in job_index
                    for t in range(2, t_upper + 1)
                ),
                name="precedence_d_flow",
            )
        model.addConstrs(
            (
                d[i, j, t] <= 0.0
                for i in range(1, instance.stage_count)
                for j in job_index
                for t in bucket_index
            ),
            name="precedence_d_ub",
        )
    elif precedence.formulation == "e":
        model.addConstrs(
            (
                _var_or_zero(b, (i, j, t))
                == gp.quicksum(
                    _var4_or_zero(e, (i, j, t, t_prime))
                    for t_prime in range(t, t_upper + 1)
                )
                for i in range(1, instance.stage_count)
                for j in job_index
                for t in bucket_index
            ),
            name="precedence_e_from_b",
        )
        model.addConstrs(
            (
                _var_or_zero(a, (i + 1, j, t))
                == gp.quicksum(
                    _var4_or_zero(e, (i, j, t_prime, t))
                    for t_prime in range(1, t + 1)
                )
                for i in range(1, instance.stage_count)
                for j in job_index
                for t in bucket_index
            ),
            name="precedence_e_to_a",
        )
    else:
        raise ValueError(
            f"Unsupported precedence formulation '{precedence.formulation}'."
        )
    model.addConstrs(
        (
            gp.quicksum(x[i, j, t] for t in operation_window[i, j])
            == processing_time[i, j]
            for i in stage_index
            for j in job_index
        ),
        name="processing_conservation",
    )
    model.addConstrs(
        (
            x[i, j, t] <= processing_time[i, j] * (a[i, j, t] + b[i, j, t])
            for i, j, t in feasible_bucket_keys
        ),
        name="bucket_activation",
    )
    model.addConstrs(
        (x[i, j, t] >= a[i, j, t] for i, j, t in feasible_bucket_keys),
        name="positive_start",
    )
    model.addConstrs(
        (x[i, j, t] >= b[i, j, t] for i, j, t in feasible_bucket_keys),
        name="positive_end",
    )
    model.addConstrs(
        (
            x[i, j, t] + _var_or_zero(x, (i, j, t + 1))
            >= processing_time[i, j] * a[i, j, t]
            for i, j, t in feasible_bucket_keys
            if t < t_upper
        ),
        name="consec_x_a_relationship",
    )
    model.addConstrs(
        (
            _var_or_zero(x, (i, j, t - 1)) + x[i, j, t]
            >= processing_time[i, j] * b[i, j, t]
            for i, j, t in feasible_bucket_keys
            if t > 1
        ),
        name="consec_x_b_relationship",
    )
    model.addConstrs(
        (c[i, j, t] >= a[i, j, t] + b[i, j, t] - 1 for i, j, t in feasible_bucket_keys),
        name="c_def_1",
    )
    model.addConstrs(
        (a[i, j, t] >= c[i, j, t] for i, j, t in feasible_bucket_keys),
        name="c_def_2",
    )
    model.addConstrs(
        (b[i, j, t] >= c[i, j, t] for i, j, t in feasible_bucket_keys),
        name="c_def_3",
    )
    model.addConstrs(
        (
            x[i, j, t] >= processing_time[i, j] * c[i, j, t]
            for i, j, t in feasible_bucket_keys
        ),
        name="same_bucket_activation",
    )

    if t_upper >= 2:
        model.addConstrs(
            (
                gp.quicksum(
                    a[i, j, t] - c[i, j, t] for _i, j, _t in bucket_keys_by_stage[i, t]
                )
                <= machine_count[i]
                for i in stage_index
                for t in range(1, t_upper)
            ),
            name="machine_capacity",
        )

    if t_lower >= 1:
        model.addConstrs(
            (
                gp.quicksum(x[i, j, t] for _i, j, _t in bucket_keys_by_stage[i, t])
                <= machine_count[i] * delta
                for i in stage_index
                for t in fixed_bucket_index
            ),
            name="stage_capacity",
        )
        model.addConstrs(
            (
                gp.quicksum(x[i, j, t] for i, _j, _t in bucket_keys_by_job[j, t])
                <= delta
                for j in job_index
                for t in fixed_bucket_index
            ),
            name="job_bucket_capacity",
        )

    model.addConstrs(
        (
            gp.quicksum(x[i, j, t] for _i, j, _t in bucket_keys_by_stage[i, t])
            <= machine_count[i] * z[t]
            for i in stage_index
            for t in range_bucket_index
        ),
        name="z_average",
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j, t] for i, _j, _t in bucket_keys_by_job[j, t]) <= z[t]
            for j in job_index
            for t in range_bucket_index
        ),
        name="z_single",
    )

    if t_upper >= t_lower + 2:
        model.addConstrs(
            (delta * u[t + 1] <= z[t] for t in range(t_lower + 1, t_upper)),
            name="u_z_relationship_lb",
        )
        model.addConstrs(
            (z[t] <= delta * u[t] for t in range(t_lower + 1, t_upper)),
            name="u_z_relationship_ub",
        )
        model.addConstrs(
            (u[t] >= u[t + 1] for t in range(t_lower + 1, t_upper)),
            name="u_left_aligned",
        )
    model.addConstrs(
        (
            u[t] >= b[i, j, t]
            for i, j, t in feasible_bucket_keys
            if t in range_bucket_index
        ),
        name="u_b_relationship",
    )

    if strengthening.valid_ineq_i:
        model.addConstrs(
            (
                x[i, j, earliest_bucket[i, j]] <= earliest_residual[i, j]
                for i in stage_index
                for j in job_index
                if (i, j, earliest_bucket[i, j]) in feasible_key_set
            ),
            name="range_earliest_residual",
        )
        model.addConstrs(
            (
                x[i, j, latest_bucket[i, j]] <= latest_residual[i, j]
                for i in stage_index
                for j in job_index
                if (i, j, latest_bucket[i, j]) in feasible_key_set
            ),
            name="range_latest_residual",
        )

    # if strengthening.valid_ineq_ii:
    #     model.addConstrs(
    #         (
    #             upstream_total[i, j] * a[i, j, t]
    #             <= gp.quicksum(
    #                 x[k, j, tau]
    #                 for k in range(1, i)
    #                 for tau in operation_window[k, j]
    #                 if tau <= t
    #             )
    #             for i, j, t in feasible_bucket_keys
    #             if i >= 2
    #         ),
    #         name="range_head_link",
    #     )
    #     model.addConstrs(
    #         (
    #             downstream_total[i, j] * b[i, j, t]
    #             <= gp.quicksum(
    #                 x[k, j, tau]
    #                 for k in range(i + 1, instance.stage_count + 1)
    #                 for tau in operation_window[k, j]
    #                 if tau >= t
    #             )
    #             for i, j, t in feasible_bucket_keys
    #             if i < instance.stage_count
    #         ),
    #         name="range_tail_link",
    #     )

    if strengthening.valid_ineq_iii:
        model.addConstrs(
            (w_expr >= total_processing_per_job[j] for j in job_index),
            name="range_job_chain_cut",
        )

    if strengthening.valid_ineq_iv:
        model.addConstrs(
            (machine_count[i] * w_expr >= stage_cut_rhs[i] for i in stage_index),
            name="range_stage_cut",
        )
        model.addConstr(w_expr >= global_lb_s, name="range_global_cut")

    return model, BucketModelVars(a=a, b=b, c=c, x=x, u=u, z=z, d=d, e=e)
