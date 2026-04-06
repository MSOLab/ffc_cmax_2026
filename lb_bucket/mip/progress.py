from __future__ import annotations

import math
from typing import Any

from .shared import ProgressTraceRow


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_callback_float(model: Any, what: int) -> float | None:
    try:
        return _safe_float(model.cbGet(what))
    except Exception:
        return None


def _safe_callback_int(model: Any, what: int) -> int | None:
    value = _safe_callback_float(model, what)
    return None if value is None else int(round(value))


class GurobiProgressRecorder:
    def __init__(self, grb: Any, ins_name: str, range_base_time: float) -> None:
        self.grb = grb
        self.ins_name = ins_name
        self.range_base_time = range_base_time
        self.rows: list[ProgressTraceRow] = []
        self._last_mip_signature: tuple[Any, ...] | None = None
        self._last_barrier_iter: int | None = None
        self._last_simplex_iter: float | None = None

    def callback(self, model: Any, where: int) -> None:
        if where == self.grb.Callback.MIP:
            self._record_mip(model)
        elif where == self.grb.Callback.BARRIER:
            self._record_barrier(model)
        elif where == self.grb.Callback.SIMPLEX:
            self._record_simplex(model)

    def append_final_row(
        self,
        *,
        runtime_sec: float,
        status: str,
        objective_ub: float | None,
        objective_lb: float | None,
        solution_count: int,
    ) -> None:
        self.rows.append(
            ProgressTraceRow(
                ins_name=self.ins_name,
                event=f"FINAL_{status}",
                runtime_sec=runtime_sec,
                objective_ub=objective_ub,
                objective_lb=objective_lb,
                horizon_ub=_shift_to_horizon(self.range_base_time, objective_ub),
                horizon_lb=_shift_to_horizon(self.range_base_time, objective_lb),
                barrier_primal_obj=None,
                barrier_dual_obj=None,
                barrier_horizon_primal=None,
                barrier_horizon_dual=None,
                primal_inf=None,
                dual_inf=None,
                complementarity=None,
                node_count=None,
                solution_count=solution_count,
                barrier_iter=None,
                simplex_iter=None,
            )
        )

    def _record_mip(self, model: Any) -> None:
        runtime_sec = _safe_callback_float(model, self.grb.Callback.RUNTIME)
        objective_ub = _safe_callback_float(model, self.grb.Callback.MIP_OBJBST)
        objective_lb = _safe_callback_float(model, self.grb.Callback.MIP_OBJBND)
        node_count = _safe_callback_float(model, self.grb.Callback.MIP_NODCNT)
        solution_count = _safe_callback_int(model, self.grb.Callback.MIP_SOLCNT)

        signature = (
            objective_ub,
            objective_lb,
            node_count,
            solution_count,
        )
        if signature == self._last_mip_signature:
            return
        self._last_mip_signature = signature

        self.rows.append(
            ProgressTraceRow(
                ins_name=self.ins_name,
                event="MIP",
                runtime_sec=runtime_sec or 0.0,
                objective_ub=objective_ub,
                objective_lb=objective_lb,
                horizon_ub=_shift_to_horizon(self.range_base_time, objective_ub),
                horizon_lb=_shift_to_horizon(self.range_base_time, objective_lb),
                barrier_primal_obj=None,
                barrier_dual_obj=None,
                barrier_horizon_primal=None,
                barrier_horizon_dual=None,
                primal_inf=None,
                dual_inf=None,
                complementarity=None,
                node_count=node_count,
                solution_count=solution_count,
                barrier_iter=None,
                simplex_iter=None,
            )
        )

    def _record_barrier(self, model: Any) -> None:
        barrier_iter = _safe_callback_int(model, self.grb.Callback.BARRIER_ITRCNT)
        if barrier_iter is None or barrier_iter == self._last_barrier_iter:
            return
        self._last_barrier_iter = barrier_iter

        runtime_sec = _safe_callback_float(model, self.grb.Callback.RUNTIME)
        primal_obj = _safe_callback_float(model, self.grb.Callback.BARRIER_PRIMOBJ)
        dual_obj = _safe_callback_float(model, self.grb.Callback.BARRIER_DUALOBJ)
        primal_inf = _safe_callback_float(model, self.grb.Callback.BARRIER_PRIMINF)
        dual_inf = _safe_callback_float(model, self.grb.Callback.BARRIER_DUALINF)
        complementarity = _safe_callback_float(model, self.grb.Callback.BARRIER_COMPL)

        self.rows.append(
            ProgressTraceRow(
                ins_name=self.ins_name,
                event="BARRIER",
                runtime_sec=runtime_sec or 0.0,
                objective_ub=None,
                objective_lb=None,
                horizon_ub=None,
                horizon_lb=None,
                barrier_primal_obj=primal_obj,
                barrier_dual_obj=dual_obj,
                barrier_horizon_primal=_shift_to_horizon(self.range_base_time, primal_obj),
                barrier_horizon_dual=_shift_to_horizon(self.range_base_time, dual_obj),
                primal_inf=primal_inf,
                dual_inf=dual_inf,
                complementarity=complementarity,
                node_count=None,
                solution_count=None,
                barrier_iter=barrier_iter,
                simplex_iter=None,
            )
        )

    def _record_simplex(self, model: Any) -> None:
        simplex_iter = _safe_callback_float(model, self.grb.Callback.SPX_ITRCNT)
        if simplex_iter is None or simplex_iter == self._last_simplex_iter:
            return
        self._last_simplex_iter = simplex_iter

        runtime_sec = _safe_callback_float(model, self.grb.Callback.RUNTIME)
        obj_value = _safe_callback_float(model, self.grb.Callback.SPX_OBJVAL)
        primal_inf = _safe_callback_float(model, self.grb.Callback.SPX_PRIMINF)
        dual_inf = _safe_callback_float(model, self.grb.Callback.SPX_DUALINF)

        self.rows.append(
            ProgressTraceRow(
                ins_name=self.ins_name,
                event="SIMPLEX",
                runtime_sec=runtime_sec or 0.0,
                objective_ub=obj_value if primal_inf == 0.0 else None,
                objective_lb=None,
                horizon_ub=_shift_to_horizon(self.range_base_time, obj_value)
                if primal_inf == 0.0
                else None,
                horizon_lb=None,
                barrier_primal_obj=None,
                barrier_dual_obj=None,
                barrier_horizon_primal=None,
                barrier_horizon_dual=None,
                primal_inf=primal_inf,
                dual_inf=dual_inf,
                complementarity=None,
                node_count=None,
                solution_count=None,
                barrier_iter=None,
                simplex_iter=simplex_iter,
            )
        )


def _shift_to_horizon(range_base_time: float, objective_value: float | None) -> float | None:
    if objective_value is None:
        return None
    return range_base_time + objective_value
