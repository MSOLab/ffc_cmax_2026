from __future__ import annotations

import math
from typing import Sequence

from mbls.cpsat import CustomCpModel

from hybridflowshop.cpsat_model_2.cumulative import BaseModelBuilder
from hybridflowshop.cpsat_model_2.params import Params
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from .shared import RetainedStageCpBuild, RetainedStageCpVars, RetainedStageSelection


def select_bottleneck_stage_by_average_load(
    params: Params,
    stage_ids: Sequence[str] | None = None,
) -> str:
    candidate_stage_ids = list(stage_ids or params.i_list)
    if not candidate_stage_ids:
        raise ValueError("candidate stage list cannot be empty")

    def stage_key(stage_id: str) -> tuple[float, int]:
        avg_load = sum(params.p[j, stage_id] for j in params.j_list) / max(
            len(params.M_of[stage_id]), 1
        )
        return avg_load, -params.i_list.index(stage_id)

    return max(candidate_stage_ids, key=stage_key)


def select_bottleneck_stage_ids_by_average_load(
    params: Params,
    stage_ids: Sequence[str] | None = None,
    count: int = 1,
) -> list[str]:
    candidate_stage_ids = list(stage_ids or params.i_list)
    if not candidate_stage_ids:
        raise ValueError("candidate stage list cannot be empty")
    if count <= 0:
        raise ValueError(f"count must be positive. Received {count}.")

    def sort_key(stage_id: str) -> tuple[float, int]:
        avg_load = sum(params.p[j, stage_id] for j in params.j_list) / max(
            len(params.M_of[stage_id]), 1
        )
        return avg_load, -params.i_list.index(stage_id)

    ranked_stage_ids = sorted(candidate_stage_ids, key=sort_key, reverse=True)
    return ranked_stage_ids[:count]


def _normalize_internal_stage_candidates(params: Params) -> list[str]:
    internal_stage_ids = list(params.i_list[1:-1])
    return internal_stage_ids or list(params.i_list)


def _normalize_and_sort_retained_stage_ids(
    params: Params,
    retained_stage_ids: Sequence[str],
) -> list[str]:
    normalized_stage_ids = list(dict.fromkeys(retained_stage_ids))
    normalized_stage_ids.sort(key=params.i_list.index)
    return normalized_stage_ids


def _resolve_ratio_stage_ids(
    params: Params,
    retained_stage_ratios: Sequence[float],
) -> list[str]:
    if len(params.i_list) == 1:
        return [params.i_list[0]]

    resolved_stage_ids: list[str] = []
    max_index = len(params.i_list) - 1
    for ratio in retained_stage_ratios:
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(
                "Each retained stage ratio must satisfy 0 < ratio < 1. "
                f"Received {ratio!r}."
            )
        stage_index = int(round(ratio * max_index))
        stage_index = min(max(stage_index, 0), max_index)
        resolved_stage_ids.append(params.i_list[stage_index])
    return resolved_stage_ids


def _resolve_bottleneck_band_stage_ids(
    params: Params,
    anchor_stage_id: str,
    radius: int,
) -> list[str]:
    if anchor_stage_id not in params.i_list:
        raise ValueError(f"Unknown bottleneck_stage_id={anchor_stage_id!r}.")
    if radius < 0:
        raise ValueError(
            "bottleneck_band_radius must be non-negative. "
            f"Received {radius}."
        )

    anchor_index = params.i_list.index(anchor_stage_id)
    left_index = max(0, anchor_index - radius)
    right_index = min(len(params.i_list) - 1, anchor_index + radius)
    return list(params.i_list[left_index : right_index + 1])


def _resolve_middle_band_stage_ids(
    params: Params,
    radius: int,
) -> list[str]:
    if radius < 0:
        raise ValueError(
            "middle_band_radius must be non-negative. "
            f"Received {radius}."
        )
    if not params.i_list:
        raise ValueError("params.i_list cannot be empty")

    center_stage_id = _resolve_ratio_stage_ids(params, [0.5])[0]
    center_index = params.i_list.index(center_stage_id)
    left_index = max(0, center_index - radius)
    right_index = min(len(params.i_list) - 1, center_index + radius)
    return list(params.i_list[left_index : right_index + 1])


def resolve_retained_stage_ids(
    params: Params,
    retained_stage_mode: str,
    bottleneck_stage_id: str | None = None,
    extra_bottleneck_count: int = 1,
    bottleneck_band_radius: int = 1,
    middle_band_radius: int = 1,
    quantile_count: int | None = None,
    retained_stage_ratios: Sequence[float] | None = None,
) -> RetainedStageSelection:
    if not params.i_list:
        raise ValueError("params.i_list cannot be empty")

    first_stage_id = params.i_list[0]
    last_stage_id = params.i_list[-1]
    internal_stage_ids = _normalize_internal_stage_candidates(params)

    if retained_stage_mode == "first_last":
        return RetainedStageSelection(
            retained_stage_ids=[first_stage_id, last_stage_id],
            bottleneck_stage_id=None,
            selected_bottleneck_stage_ids=[],
            bottleneck_band_radius=None,
            middle_band_radius=None,
            retained_stage_ratios=[],
            quantile_count=None,
        )

    selected_bottleneck_stage_ids: list[str] = []
    resolved_bottleneck_band_radius: int | None = None
    resolved_middle_band_radius: int | None = None
    resolved_ratios: list[float] = []

    if retained_stage_mode == "first_bottleneck_last":
        resolved_bottleneck_stage_id = bottleneck_stage_id
        if resolved_bottleneck_stage_id is None:
            resolved_bottleneck_stage_id = select_bottleneck_stage_by_average_load(
                params,
                stage_ids=internal_stage_ids,
            )
        if resolved_bottleneck_stage_id not in params.i_list:
            raise ValueError(
                f"Unknown bottleneck_stage_id={resolved_bottleneck_stage_id!r}."
            )
        selected_bottleneck_stage_ids = [resolved_bottleneck_stage_id]
        retained = [first_stage_id, resolved_bottleneck_stage_id, last_stage_id]
    elif retained_stage_mode == "first_bottleneck_band_last":
        resolved_bottleneck_stage_id = bottleneck_stage_id
        if resolved_bottleneck_stage_id is None:
            resolved_bottleneck_stage_id = select_bottleneck_stage_by_average_load(
                params,
                stage_ids=internal_stage_ids,
            )
        if resolved_bottleneck_stage_id not in params.i_list:
            raise ValueError(
                f"Unknown bottleneck_stage_id={resolved_bottleneck_stage_id!r}."
            )
        selected_bottleneck_stage_ids = [resolved_bottleneck_stage_id]
        resolved_bottleneck_band_radius = int(bottleneck_band_radius)
        retained = [
            first_stage_id,
            *_resolve_bottleneck_band_stage_ids(
                params,
                resolved_bottleneck_stage_id,
                resolved_bottleneck_band_radius,
            ),
            last_stage_id,
        ]
    elif retained_stage_mode == "first_topk_bottlenecks_last":
        if extra_bottleneck_count <= 0:
            raise ValueError(
                "extra_bottleneck_count must be positive for "
                "'first_topk_bottlenecks_last'. "
                f"Received {extra_bottleneck_count}."
            )
        if bottleneck_stage_id is not None and bottleneck_stage_id not in params.i_list:
            raise ValueError(f"Unknown bottleneck_stage_id={bottleneck_stage_id!r}.")

        selected_bottleneck_stage_ids = []
        if bottleneck_stage_id is not None:
            selected_bottleneck_stage_ids.append(bottleneck_stage_id)

        remaining_candidates = [
            stage_id
            for stage_id in internal_stage_ids
            if stage_id not in selected_bottleneck_stage_ids
        ]
        remaining_count = extra_bottleneck_count - len(selected_bottleneck_stage_ids)
        if remaining_count > 0:
            selected_bottleneck_stage_ids.extend(
                select_bottleneck_stage_ids_by_average_load(
                    params,
                    stage_ids=remaining_candidates,
                    count=min(remaining_count, len(remaining_candidates)),
                )
            )
        retained = [first_stage_id, *selected_bottleneck_stage_ids, last_stage_id]
    elif retained_stage_mode == "first_middle_last":
        resolved_ratios = [0.5]
        retained = [
            first_stage_id,
            *_resolve_ratio_stage_ids(params, resolved_ratios),
            last_stage_id,
        ]
    elif retained_stage_mode == "first_middle_band_last":
        resolved_ratios = [0.5]
        resolved_middle_band_radius = int(middle_band_radius)
        retained = [
            first_stage_id,
            *_resolve_middle_band_stage_ids(params, resolved_middle_band_radius),
            last_stage_id,
        ]
    elif retained_stage_mode == "first_n_quantiles_last":
        if quantile_count is None or quantile_count < 2:
            raise ValueError(
                "quantile_count must be an integer >= 2 for "
                "'first_n_quantiles_last'. "
                f"Received {quantile_count!r}."
            )
        resolved_ratios = [
            quantile_idx / quantile_count for quantile_idx in range(1, quantile_count)
        ]
        retained = [
            first_stage_id,
            *_resolve_ratio_stage_ids(params, resolved_ratios),
            last_stage_id,
        ]
    elif retained_stage_mode == "first_ratio_points_last":
        if not retained_stage_ratios:
            raise ValueError(
                "retained_stage_ratios must be provided for 'first_ratio_points_last'."
            )
        resolved_ratios = [float(ratio) for ratio in retained_stage_ratios]
        retained = [
            first_stage_id,
            *_resolve_ratio_stage_ids(params, resolved_ratios),
            last_stage_id,
        ]
    else:
        raise ValueError(
            "retained_stage_mode must be one of "
            "'first_last', 'first_bottleneck_last', "
            "'first_bottleneck_band_last', 'first_topk_bottlenecks_last', "
            "'first_middle_last', 'first_middle_band_last', "
            "'first_n_quantiles_last', or "
            "'first_ratio_points_last'. "
            f"Received: {retained_stage_mode!r}"
        )

    normalized_retained_stage_ids = _normalize_and_sort_retained_stage_ids(
        params, retained
    )
    normalized_bottleneck_stage_ids = [
        stage_id
        for stage_id in _normalize_and_sort_retained_stage_ids(
            params,
            selected_bottleneck_stage_ids,
        )
        if stage_id not in {first_stage_id, last_stage_id}
    ]
    return RetainedStageSelection(
        retained_stage_ids=normalized_retained_stage_ids,
        bottleneck_stage_id=(
            normalized_bottleneck_stage_ids[0]
            if normalized_bottleneck_stage_ids
            else None
        ),
        selected_bottleneck_stage_ids=normalized_bottleneck_stage_ids,
        bottleneck_band_radius=resolved_bottleneck_band_radius,
        middle_band_radius=resolved_middle_band_radius,
        retained_stage_ratios=resolved_ratios,
        quantile_count=quantile_count,
    )


def _compute_compressed_lag_by_pair_job(
    params: Params,
    retained_stage_ids: Sequence[str],
) -> dict[tuple[str, str, str], int]:
    compressed_lag_by_pair_job: dict[tuple[str, str, str], int] = {}
    for left_stage_id, right_stage_id in zip(
        retained_stage_ids[:-1], retained_stage_ids[1:]
    ):
        left_idx = params.i_list.index(left_stage_id)
        right_idx = params.i_list.index(right_stage_id)
        for job_id in params.j_list:
            compressed_lag_by_pair_job[(left_stage_id, right_stage_id, job_id)] = sum(
                params.p[job_id, params.i_list[stage_idx]]
                for stage_idx in range(left_idx + 1, right_idx)
            )
    return compressed_lag_by_pair_job


def _compute_santos_rhs_by_stage(params: Params) -> dict[str, int]:
    head_by_job_stage = BaseModelBuilder._compute_head(params)
    tail_by_job_stage = BaseModelBuilder._compute_tail(params)
    santos_rhs_by_stage: dict[str, int] = {}
    for stage_id in params.i_list:
        machine_count = max(len(params.M_of[stage_id]), 1)
        prefix_size = min(machine_count, len(params.j_list))
        sorted_heads = sorted(
            head_by_job_stage[job_id, stage_id] for job_id in params.j_list
        )
        sorted_tails = sorted(
            tail_by_job_stage[job_id, stage_id] for job_id in params.j_list
        )
        stage_processing = sum(params.p[job_id, stage_id] for job_id in params.j_list)
        santos_rhs_by_stage[stage_id] = (
            sum(sorted_heads[:prefix_size])
            + stage_processing
            + sum(sorted_tails[:prefix_size])
        )
    return santos_rhs_by_stage


def build_retained_stage_cp_model(
    instance: HybridFlowshopParameters,
    *,
    input_ub: int,
    retained_stage_mode: str,
    bottleneck_stage_id: str | None = None,
    extra_bottleneck_count: int = 1,
    bottleneck_band_radius: int = 1,
    middle_band_radius: int = 1,
    quantile_count: int | None = None,
    retained_stage_ratios: Sequence[float] | None = None,
) -> RetainedStageCpBuild:
    if input_ub <= 0:
        raise ValueError(f"input_ub must be positive. Received {input_ub}.")

    params = BaseModelBuilder.make_params(instance)
    head_by_job_stage = BaseModelBuilder._compute_head(params)
    tail_by_job_stage = BaseModelBuilder._compute_tail(params)
    retained_stage_selection = resolve_retained_stage_ids(
        params,
        retained_stage_mode,
        bottleneck_stage_id=bottleneck_stage_id,
        extra_bottleneck_count=extra_bottleneck_count,
        bottleneck_band_radius=bottleneck_band_radius,
        middle_band_radius=middle_band_radius,
        quantile_count=quantile_count,
        retained_stage_ratios=retained_stage_ratios,
    )
    retained_stage_ids = retained_stage_selection.retained_stage_ids
    retained_stage_indices = [
        params.i_list.index(stage_id) + 1 for stage_id in retained_stage_ids
    ]
    compressed_lag_by_pair_job = _compute_compressed_lag_by_pair_job(
        params,
        retained_stage_ids,
    )
    santos_rhs_by_stage = _compute_santos_rhs_by_stage(params)

    job_chain_lb = max(
        sum(params.p[job_id, stage_id] for stage_id in params.i_list)
        for job_id in params.j_list
    )
    santos_lb = max(
        math.ceil(santos_rhs_by_stage[stage_id] / max(len(params.M_of[stage_id]), 1))
        for stage_id in params.i_list
    )
    makespan_lb = max(job_chain_lb, santos_lb)

    mdl = CustomCpModel()
    op_start = {}
    op_end = {}
    op_intvl = {}

    for job_id in params.j_list:
        for stage_id in retained_stage_ids:
            processing_time = params.p[job_id, stage_id]
            head = head_by_job_stage[job_id, stage_id]
            tail = tail_by_job_stage[job_id, stage_id]
            start_lb = head
            start_ub = input_ub - tail - processing_time
            end_lb = head + processing_time
            end_ub = input_ub - tail
            if start_lb > start_ub or end_lb > end_ub:
                raise ValueError(
                    "Retained-stage CP model has an empty time window for "
                    f"(job={job_id}, stage={stage_id})."
                )

            start_var = mdl.new_int_var(
                start_lb, start_ub, f"retained_start_{job_id}_{stage_id}"
            )
            end_var = mdl.new_int_var(
                end_lb, end_ub, f"retained_end_{job_id}_{stage_id}"
            )
            interval_var = mdl.new_interval_var(
                start_var,
                processing_time,
                end_var,
                f"retained_interval_{job_id}_{stage_id}",
            )
            op_start[(job_id, stage_id)] = start_var
            op_end[(job_id, stage_id)] = end_var
            op_intvl[(job_id, stage_id)] = interval_var

    makespan = mdl.new_int_var(makespan_lb, input_ub, "retained_makespan")
    variables = RetainedStageCpVars(
        op_start=op_start,
        op_end=op_end,
        op_intvl=op_intvl,
        makespan=makespan,
    )

    for job_id in params.j_list:
        for stage_id in retained_stage_ids:
            mdl.add(op_start[job_id, stage_id] >= head_by_job_stage[job_id, stage_id])
            mdl.add(
                op_end[job_id, stage_id] + tail_by_job_stage[job_id, stage_id]
                <= makespan
            )

    for left_stage_id, right_stage_id in zip(
        retained_stage_ids[:-1], retained_stage_ids[1:]
    ):
        for job_id in params.j_list:
            mdl.add(
                op_start[job_id, right_stage_id]
                >= op_end[job_id, left_stage_id]
                + compressed_lag_by_pair_job[(left_stage_id, right_stage_id, job_id)]
            )

    for stage_id in retained_stage_ids:
        mdl.add_cumulative(
            [op_intvl[job_id, stage_id] for job_id in params.j_list],
            [1] * len(params.j_list),
            len(params.M_of[stage_id]),
        )

    for job_id in params.j_list:
        mdl.add(
            makespan >= sum(params.p[job_id, stage_id] for stage_id in params.i_list)
        )

    for stage_id in params.i_list:
        mdl.add(len(params.M_of[stage_id]) * makespan >= santos_rhs_by_stage[stage_id])

    mdl.minimize(makespan)
    mdl.set_num_base_constraints()

    return RetainedStageCpBuild(
        model=mdl,
        params=params,
        variables=variables,
        retained_stage_mode=retained_stage_mode,
        retained_stage_ids=retained_stage_ids,
        retained_stage_indices=retained_stage_indices,
        bottleneck_stage_id=retained_stage_selection.bottleneck_stage_id,
        selected_bottleneck_stage_ids=retained_stage_selection.selected_bottleneck_stage_ids,
        bottleneck_band_radius=retained_stage_selection.bottleneck_band_radius,
        middle_band_radius=retained_stage_selection.middle_band_radius,
        retained_stage_ratios=retained_stage_selection.retained_stage_ratios,
        quantile_count=retained_stage_selection.quantile_count,
        head_by_job_stage=head_by_job_stage,
        tail_by_job_stage=tail_by_job_stage,
        compressed_lag_by_pair_job=compressed_lag_by_pair_job,
        santos_rhs_by_stage=santos_rhs_by_stage,
    )
