from __future__ import annotations

import math
import time
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lb_bucket" / "results"
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
class SearchTraceRow:
    ins_name: str
    bucket_count: int
    status: str
    runtime_sec: float
    objective_value: float | None
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
    z_star: float | None
    bucket_indexed_lb: float | None
    certified_final_lb: float
    search_certified: bool
    searched_bucket_count: int
    total_runtime_sec: float
    termination_reason: str


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
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator)
