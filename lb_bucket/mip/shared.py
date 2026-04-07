from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = (
    REPO_ROOT
    / "Outputs_debug"
    / "20260327T171019_584033"
    / "ff2020"
    / "20260327-01"
    / "multi_instance_summary.csv"
)
DEFAULT_INPUT_DIR = REPO_ROOT / "resources" / "ff2020big"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lb_bucket" / "runs" / "mip_manual"
DEFAULT_SOLUTION_ROOT = REPO_ROOT / "Outputs_scenarios" / "20260406T092725_768054"
DEFAULT_DELTA = 100


@dataclass(frozen=True)
class SummaryBoundRecord:
    ins_name: str
    input_lb: int
    input_ub: int | None
    job_count: int | None
    stage_count: int | None


@dataclass(frozen=True)
class TwoBucketInstance:
    ins_name: str
    job_count: int
    stage_count: int
    machine_count_per_stage: list[int]
    processing_times_by_stage: list[list[int]]


@dataclass(frozen=True)
class BucketModelVars:
    a: Any
    b: Any
    c: Any
    x: Any
    u: Any
    z: Any
    d: Any | None = None
    e: Any | None = None


@dataclass(frozen=True)
class SearchTraceRow:
    ins_name: str
    bucket_count: int
    status: str
    runtime_sec: float
    objective_ub: float | None
    objective_lb: float | None
    horizon_ub: float | None
    horizon_lb: float | None
    solution_count: int


@dataclass(frozen=True)
class BucketSearchResult:
    ins_name: str
    input_lb: int
    input_ub: int | None
    delta: int
    job_count: int
    stage_count: int
    machine_count_per_stage: str
    t0: int
    search_upper_t: int | None
    first_feasible_t: int | None
    objective_ub: float | None
    objective_lb: float | None
    horizon_ub: float | None
    horizon_lb: float | None
    z_star: float | None
    z_lower_bound_used: float | None
    bucket_indexed_lb: float | None
    certified_final_lb: float
    search_certified: bool
    searched_bucket_count: int
    time_limit_sec_used: float
    total_runtime_sec: float
    termination_reason: str
    status_name: str
    solution_count: int


@dataclass(frozen=True)
class ProgressTraceRow:
    ins_name: str
    event: str
    runtime_sec: float
    objective_ub: float | None
    objective_lb: float | None
    horizon_ub: float | None
    horizon_lb: float | None
    barrier_primal_obj: float | None
    barrier_dual_obj: float | None
    barrier_horizon_primal: float | None
    barrier_horizon_dual: float | None
    primal_inf: float | None
    dual_inf: float | None
    complementarity: float | None
    node_count: float | None
    solution_count: int | None
    barrier_iter: int | None
    simplex_iter: float | None


@dataclass(frozen=True)
class ModelStrengtheningOptions:
    cumulative_precedence: bool
    valid_ineq_i: bool
    valid_ineq_ii: bool
    valid_ineq_iii: bool
    valid_ineq_iv: bool

    def describe(self) -> str:
        return (
            f"cumulative_precedence={self.cumulative_precedence}, "
            f"valid_i={self.valid_ineq_i}, "
            f"valid_ii={self.valid_ineq_ii}, "
            f"valid_iii={self.valid_ineq_iii}, "
            f"valid_iv={self.valid_ineq_iv}"
        )


@dataclass(frozen=True)
class PrecedenceOptions:
    formulation: str

    def describe(self) -> str:
        return f"precedence_formulation={self.formulation}"


def import_gurobi() -> tuple[Any, Any]:
    try:
        import gurobipy as gp
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "gurobipy is required to run this script. "
            "Install Gurobi and make sure its Python package is available."
        ) from exc
    return gp, gp.GRB


def parse_optional_int(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if stripped == "":
        return None
    return int(float(stripped))


def gurobi_status_name(grb: Any, status_code: int) -> str:
    status_lookup = {
        grb.LOADED: "LOADED",
        grb.OPTIMAL: "OPTIMAL",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.CUTOFF: "CUTOFF",
        grb.ITERATION_LIMIT: "ITERATION_LIMIT",
        grb.NODE_LIMIT: "NODE_LIMIT",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.NUMERIC: "NUMERIC",
        grb.SUBOPTIMAL: "SUBOPTIMAL",
        grb.INPROGRESS: "INPROGRESS",
        grb.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_lookup.get(status_code, f"STATUS_{status_code}")


def log_progress(message: str) -> None:
    logging.info(message)


def ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator)
