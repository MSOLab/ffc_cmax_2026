from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd

from .method_summary_chart import (
    _build_best_so_far_progression_points,
    _build_step_path,
    _lookup_rpdf_at_or_before,
)


def _positive_axis_upper(values: list[float]) -> float:
    if not values:
        return 0.01
    max_value = max(values)
    if max_value <= 0:
        return 0.01
    return max_value * 1.05


def _normalize_scenario_input(
    scenario_input: tuple[str, pd.DataFrame] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(scenario_input, tuple):
        label, endpoint_df = scenario_input
        return {
            "label": str(label),
            "endpoint_df": endpoint_df,
            "raw_progression_df": None,
        }

    if not isinstance(scenario_input, dict):
        raise TypeError("Scenario input must be a tuple or dict.")

    return {
        "label": str(scenario_input["label"]),
        "endpoint_df": scenario_input["endpoint_df"],
        "raw_progression_df": scenario_input.get("raw_progression_df"),
    }


def _prepare_scenario_endpoint_df(endpoint_df: pd.DataFrame) -> pd.DataFrame:
    work_df = endpoint_df.copy()
    order_map = {
        name: idx for idx, name in enumerate(pd.unique(work_df["subroutine_name"]), start=1)
    }
    work_df["subroutine_order"] = work_df["subroutine_name"].map(order_map)
    return work_df


def _prepare_scenario_progression_df(
    raw_progression_df: pd.DataFrame | None, order_map_source_df: pd.DataFrame
) -> pd.DataFrame | None:
    if raw_progression_df is None or raw_progression_df.empty:
        return None

    work_df = raw_progression_df.copy()
    order_map = {
        name: idx
        for idx, name in enumerate(pd.unique(order_map_source_df["subroutine_name"]), start=1)
    }
    work_df["subroutine_order"] = work_df["subroutine_name"].map(order_map)
    return work_df.dropna(subset=["subroutine_order"]).copy()


def _build_scenario_progression_models(
    endpoint_df: pd.DataFrame, raw_progression_df: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    progression_by_instance: dict[str, pd.DataFrame] = {}
    if raw_progression_df is not None and not raw_progression_df.empty:
        progression_sort_cols = [
            col
            for col in ["norm_time", "global_sec", "call_index"]
            if col in raw_progression_df.columns
        ]
        progression_by_instance = {
            str(instance_id): grp.sort_values(progression_sort_cols)
            for instance_id, grp in raw_progression_df.groupby("instance_id", sort=True)
        }

    models: list[dict[str, Any]] = []
    for instance_id, endpoint_grp in endpoint_df.groupby("instance_id", sort=True):
        endpoint_grp = endpoint_grp.sort_values(
            ["norm_time", "subroutine_order", "subroutine_name"]
        )
        progression_grp = progression_by_instance.get(str(instance_id))
        source_grp = endpoint_grp if progression_grp is None or progression_grp.empty else progression_grp
        models.append(
            {
                "instance_id": str(instance_id),
                "progression_points": _build_best_so_far_progression_points(source_grp),
            }
        )

    return models


def _build_scenario_mean_series(
    scenario_label: str,
    endpoint_df: pd.DataFrame,
    raw_progression_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    models = _build_scenario_progression_models(endpoint_df, raw_progression_df)
    models = [model for model in models if model["progression_points"]]
    if not models:
        return None

    first_times = [model["progression_points"][0].time for model in models]
    last_times = [model["progression_points"][-1].time for model in models]
    start_time = max(first_times)
    end_time = max(last_times)
    union_times = sorted(
        {
            point.time
            for model in models
            for point in model["progression_points"]
            if start_time <= point.time <= end_time
        }
    )
    if not union_times:
        union_times = [start_time]
        if end_time > start_time:
            union_times.append(end_time)
    elif union_times[-1] < end_time:
        union_times.append(end_time)

    mean_x: list[float] = []
    mean_y: list[float] = []
    for time_val in union_times:
        values = [
            value
            for model in models
            if (value := _lookup_rpdf_at_or_before(model["progression_points"], time_val))
            is not None
        ]
        if len(values) != len(models):
            continue
        mean_x.append(time_val)
        mean_y.append(sum(values) / len(values))

    if not mean_x:
        return None

    step_x, step_y = _build_step_path(mean_x, mean_y)
    guide_df = (
        endpoint_df.sort_values(["subroutine_order", "subroutine_name", "norm_time"])
        .groupby("subroutine_name", as_index=False, sort=False)
        .agg(avg_norm_time=("norm_time", "mean"))
    )
    guide_marker_x = guide_df["avg_norm_time"].astype(float).tolist()
    guide_marker_text = guide_df["subroutine_name"].astype(str).tolist()
    return {
        "scenario": scenario_label,
        "step_x": step_x,
        "step_y": step_y,
        "step_customdata": [[scenario_label, len(models)] for _ in step_x],
        "vertical_guides": [
            {"subroutine_name": name, "x": time_val}
            for name, time_val in zip(guide_marker_text, guide_marker_x, strict=True)
        ],
        "guide_marker_x": guide_marker_x,
        "guide_marker_text": guide_marker_text,
        "guide_marker_customdata": _build_guide_marker_customdata(
            scenario_label, guide_marker_text
        ),
    }


def _build_guide_marker_customdata(
    scenario_label: str, guide_marker_text: list[str]
) -> list[list[Any]]:
    return [
        [scenario_label, str(subroutine_name)]
        for subroutine_name in guide_marker_text
    ]


def _build_multi_scenario_method_rpdf_payload(
    scenario_metrics: list[tuple[str, pd.DataFrame] | dict[str, Any]],
) -> dict:
    traces = []
    all_x: list[float] = []
    all_y: list[float] = []

    for raw_input in scenario_metrics:
        scenario_input = _normalize_scenario_input(raw_input)
        scenario_label = scenario_input["label"]
        endpoint_df = scenario_input["endpoint_df"]
        raw_progression_df = scenario_input["raw_progression_df"]

        if endpoint_df is None or endpoint_df.empty:
            continue

        endpoint_work_df = _prepare_scenario_endpoint_df(endpoint_df)
        progression_work_df = _prepare_scenario_progression_df(
            raw_progression_df, endpoint_work_df
        )
        mean_series = _build_scenario_mean_series(
            str(scenario_label),
            endpoint_work_df,
            progression_work_df,
        )
        if mean_series is None:
            continue
        traces.append(mean_series)
        all_x.extend(float(x) for x in mean_series["step_x"])
        all_y.extend(float(y) for y in mean_series["step_y"])

    return {
        "traces": traces,
        "x_max": _positive_axis_upper(all_x),
        "y_max": _positive_axis_upper(all_y),
    }


def _build_multi_scenario_method_rpdf_html_page(
    payload: dict,
    x_percent_decimals: int = 1,
    y_percent_decimals: int = 1,
) -> str:
    template = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Subroutine Flow Comparison</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 18px;
      color: #1b1b1b;
    }
    h1 {
      font-size: 20px;
      margin: 0 0 8px 0;
    }
    p {
      margin: 0 0 16px 0;
      color: #444;
    }
  </style>
</head>
<body>
  <h1>Subroutine Flow Comparison</h1>
  <p>Mean over-time RPDf progression by scenario.</p>
  <div id="multi-scenario-method-chart" style="width: 100%; height: 760px;"></div>
  <script>
    const payload = $payload_json;
    const SERIES_COLORS = [
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ];
    const SYMBOL_MAP = {
      "initialize_by_best_of_selected_dispatches": "circle",
      "neh_cp": "diamond",
      "set_cp_model_as_base_cp_model": "square",
      "run_reactive_loop": "x",
      "apply_shdlb": "triangle",
      "repeat_while_improvement": "square-open",
      "finalize": "star"
    };

    const traces = payload.traces.flatMap((trace, idx) => {
      const seriesColor = SERIES_COLORS[idx % SERIES_COLORS.length];
      return [
        {
          type: "scatter",
          mode: "lines",
          name: trace.scenario,
          x: trace.step_x,
          y: trace.step_y,
          customdata: trace.step_customdata,
          line: { width: 2, color: seriesColor },
          hovertemplate:
            "scenario=%{customdata[0]}<br>" +
            "instance_cnt=%{customdata[1]}<br>" +
            "Time%=%{x:.4%}<br>" +
            "Mean RPDf=%{y:.4%}<extra></extra>",
          showlegend: true,
        },
        {
          type: "scatter",
          mode: "markers",
          name: trace.scenario,
          x: trace.guide_marker_x,
          y: trace.guide_marker_x.map(() => 0),
          text: trace.guide_marker_text,
          customdata: trace.guide_marker_customdata,
          marker: {
            size: 8,
            color: seriesColor,
            symbol: trace.guide_marker_text.map((name) => SYMBOL_MAP[name] || "circle")
          },
          hovertemplate:
            "scenario=%{customdata[0]}<br>" +
            "subroutine=%{customdata[1]}<br>" +
            "avg end Time%=%{x:.4%}<extra></extra>",
          showlegend: false,
        }
      ];
    });

    const layout = {
      title: { text: "Subroutine flow mean over-time RPDf by scenario" },
      xaxis: {
        title: { text: "Normalized time" },
        tickformat: ".$x_percent_decimals%",
        range: [0, payload.x_max],
      },
      yaxis: {
        title: { text: "Mean RPDf" },
        tickformat: ".$y_percent_decimals%",
        range: [0, payload.y_max],
      },
      template: "plotly_white",
      hovermode: "closest",
      legend: { orientation: "h" },
      margin: { l: 70, r: 20, t: 70, b: 70 },
      shapes: payload.traces.flatMap((trace, idx) => {
        const seriesColor = SERIES_COLORS[idx % SERIES_COLORS.length];
        return (trace.vertical_guides || []).map((guide) => {
          return {
            type: "line",
            xref: "x",
            yref: "paper",
            x0: guide.x,
            x1: guide.x,
            y0: 0,
            y1: 1,
            line: {
              color: seriesColor,
              width: 1,
              dash: "dot"
            }
          };
        });
      }),
    };

    Plotly.newPlot("multi-scenario-method-chart", traces, layout, {
      responsive: true
    });
  </script>
</body>
</html>
""")
    return template.substitute(
        payload_json=json.dumps(payload, separators=(",", ":")),
        x_percent_decimals=x_percent_decimals,
        y_percent_decimals=y_percent_decimals,
    )


def export_multi_scenario_method_rpdf_comparison_html(
    scenario_metrics: list[tuple[str, pd.DataFrame] | dict[str, Any]],
    output_path: Path,
    x_percent_decimals: int = 1,
    y_percent_decimals: int = 1,
) -> bool:
    payload = _build_multi_scenario_method_rpdf_payload(scenario_metrics)
    if not payload["traces"]:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = _build_multi_scenario_method_rpdf_html_page(
        payload=payload,
        x_percent_decimals=x_percent_decimals,
        y_percent_decimals=y_percent_decimals,
    )
    output_path.write_text(html_content, encoding="utf-8")
    logging.info(f"Multi-scenario method comparison HTML saved to {output_path}")
    return True
