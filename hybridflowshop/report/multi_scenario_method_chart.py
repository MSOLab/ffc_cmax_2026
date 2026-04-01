from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template

import pandas as pd

from .method_summary_chart import _build_method_rpdf_scatter_df


def _positive_axis_upper(values: list[float]) -> float:
    if not values:
        return 0.01
    max_value = max(values)
    if max_value <= 0:
        return 0.01
    return max_value * 1.05


def _build_multi_scenario_method_rpdf_payload(
    scenario_metrics: list[tuple[str, pd.DataFrame]],
) -> dict:
    traces = []
    all_x: list[float] = []
    all_y: list[float] = []

    for scenario_label, metrics_long_df in scenario_metrics:
        plot_df = _build_method_rpdf_scatter_df(metrics_long_df)
        if plot_df.empty:
            continue

        x_values = plot_df["mean_norm_time"].astype(float).tolist()
        y_values = plot_df["mean_rpd_f"].astype(float).tolist()
        subroutine_names = plot_df["subroutine_name"].astype(str).tolist()
        traces.append(
            {
                "scenario": str(scenario_label),
                "x": x_values,
                "y": y_values,
                "text": subroutine_names,
                "customdata": [
                    [str(scenario_label), subroutine_name]
                    for subroutine_name in subroutine_names
                ],
            }
        )
        all_x.extend(x_values)
        all_y.extend(y_values)

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
  <p>Mean norm_time vs mean RPDf by scenario.</p>
  <div id="multi-scenario-method-chart" style="width: 100%; height: 760px;"></div>
  <script>
    const payload = $payload_json;
    const traces = payload.traces.map((trace) => {
      return {
        type: "scatter",
        mode: "lines+markers",
        name: trace.scenario,
        x: trace.x,
        y: trace.y,
        text: trace.text,
        customdata: trace.customdata,
        line: { width: 2 },
        marker: { size: 8 },
        hovertemplate:
          "scenario=%{customdata[0]}<br>" +
          "subroutine=%{customdata[1]}<br>" +
          "mean norm_time=%{x:.4%}<br>" +
          "mean RPDf=%{y:.4%}<extra></extra>",
      };
    });

    const layout = {
      title: { text: "Subroutine flow mean norm_time vs mean RPDf" },
      xaxis: {
        title: { text: "Mean normalized time" },
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
    scenario_metrics: list[tuple[str, pd.DataFrame]],
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
