from __future__ import annotations

from dataclasses import dataclass

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntervalVar, IntVar

from hybridflowshop.cpsat_model_2.params import Params


@dataclass(frozen=True)
class RetainedStageSelection:
    retained_stage_ids: list[str]
    bottleneck_stage_id: str | None
    selected_bottleneck_stage_ids: list[str]
    bottleneck_band_radius: int | None
    retained_stage_ratios: list[float]
    quantile_count: int | None


@dataclass(frozen=True)
class RetainedStageCpVars:
    op_start: dict[tuple[str, str], IntVar]
    op_end: dict[tuple[str, str], IntVar]
    op_intvl: dict[tuple[str, str], IntervalVar]
    makespan: IntVar


@dataclass(frozen=True)
class RetainedStageCpBuild:
    model: CustomCpModel
    params: Params
    variables: RetainedStageCpVars
    retained_stage_mode: str
    retained_stage_ids: list[str]
    retained_stage_indices: list[int]
    bottleneck_stage_id: str | None
    selected_bottleneck_stage_ids: list[str]
    bottleneck_band_radius: int | None
    retained_stage_ratios: list[float]
    quantile_count: int | None
    head_by_job_stage: dict[tuple[str, str], int]
    tail_by_job_stage: dict[tuple[str, str], int]
    compressed_lag_by_pair_job: dict[tuple[str, str, str], int]
    santos_rhs_by_stage: dict[str, int]
