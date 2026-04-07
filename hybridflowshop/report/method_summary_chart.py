from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import matplotlib
import pandas as pd
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_METHOD_RPDF_COLUMNS = {"subroutine_name", "norm_time", "rpd_f"}


def _build_method_rpdf_scatter_df(metrics_long_df: pd.DataFrame) -> pd.DataFrame:
    """Build plotting rows from complete-case observations only.

    Only rows with non-null `subroutine_name`, `norm_time`, and `rpd_f`
    contribute to the grouped means. This intentionally computes both axes
    from the same valid observation set for each subroutine.
    """
    missing_cols = REQUIRED_METHOD_RPDF_COLUMNS - set(metrics_long_df.columns)
    if missing_cols:
        raise ValueError(
            "Missing required columns for method RPD scatter chart: "
            f"{sorted(missing_cols)}"
        )

    clean_df = metrics_long_df.dropna(subset=["norm_time", "rpd_f"])

    return (
        clean_df.groupby("subroutine_name", sort=False)
        .agg(
            mean_norm_time=("norm_time", "mean"),
            mean_rpd_f=("rpd_f", "mean"),
        )
        .reset_index()
    )


def export_method_rpdf_scatter_svg(
    metrics_long_df: pd.DataFrame, output_path: Path
) -> bool:
    plot_df = _build_method_rpdf_scatter_df(metrics_long_df)
    if plot_df.empty:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            ax.plot(
                plot_df["mean_norm_time"],
                plot_df["mean_rpd_f"],
                marker="o",
                linewidth=1.5,
            )

            for _, row in plot_df.iterrows():
                ax.annotate(
                    str(row["subroutine_name"]),
                    (row["mean_norm_time"], row["mean_rpd_f"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                )

            ax.set_xlabel("Mean normalized time")
            ax.set_ylabel("Mean RPDf")
            ax.set_title("Subroutine mean norm_time vs mean RPDf")
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.grid(True, linestyle="--", alpha=0.4)

            # Set both axes origin to 0
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

            fig.savefig(output_path, format="svg", bbox_inches="tight")
        finally:
            plt.close(fig)

    logging.info(f"Method RPD scatter SVG saved to {output_path}")
    return True


REQUIRED_HTML_COLUMNS = {"instance_id", "subroutine_name", "norm_time", "rpd_f"}
T = TypeVar("T")


@dataclass(frozen=True)
class ProgressionPoint:
    time: float
    rpd_f: float


@dataclass(frozen=True)
class MarkerMeta:
    instance_id: str
    job_cnt: int
    stage_cnt: int
    subroutine_name: str
    count: int
    time: float


@dataclass(frozen=True)
class RawInstanceProgression:
    series_id: str
    instance_id: str
    job_cnt: int
    stage_cnt: int
    progression_points: list[ProgressionPoint]
    marker_meta_by_time: dict[float, MarkerMeta]


@dataclass(frozen=True)
class MeanVerticalGuide:
    subroutine_name: str
    x: float


def _normalize_instance_key(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        as_float = float(text)
    except ValueError:
        return text

    return str(int(as_float)) if as_float.is_integer() else text


def _load_and_merge_for_html(
    metrics_long_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    baseline_instance_col: str,
    baseline_job_cnt_col: str,
    baseline_stage_cnt_col: str,
) -> pd.DataFrame:
    """Merge metrics with already-loaded baseline metadata for HTML export.

    Raises:
        ValueError: If required columns are missing or baseline keys are ambiguous.
    """
    missing_metrics = REQUIRED_HTML_COLUMNS - set(metrics_long_df.columns)
    required_baseline_cols = {
        baseline_instance_col,
        baseline_job_cnt_col,
        baseline_stage_cnt_col,
    }
    missing_baseline = required_baseline_cols - set(baseline_df.columns)

    if missing_metrics:
        raise ValueError(
            f"Missing columns in metrics DataFrame: {sorted(missing_metrics)}"
        )
    if missing_baseline:
        raise ValueError(
            f"Missing columns in baseline DataFrame: {sorted(missing_baseline)}"
        )

    metrics_df = metrics_long_df.copy()
    metrics_df["instance_key"] = metrics_df["instance_id"].map(_normalize_instance_key)
    metrics_df["norm_time"] = pd.to_numeric(metrics_df["norm_time"], errors="coerce")
    metrics_df["rpd_f"] = pd.to_numeric(metrics_df["rpd_f"], errors="coerce")
    metrics_df = metrics_df.dropna(
        subset=["instance_key", "subroutine_name", "norm_time", "rpd_f"]
    ).copy()

    baseline_meta_df = baseline_df[
        [baseline_instance_col, baseline_job_cnt_col, baseline_stage_cnt_col]
    ].copy()
    baseline_meta_df = baseline_meta_df.rename(
        columns={
            baseline_instance_col: "instance_id",
            baseline_job_cnt_col: "job_cnt",
            baseline_stage_cnt_col: "stage_cnt",
        }
    )
    baseline_meta_df["instance_key"] = baseline_meta_df["instance_id"].map(
        _normalize_instance_key
    )
    baseline_meta_df["job_cnt"] = pd.to_numeric(
        baseline_meta_df["job_cnt"], errors="coerce"
    )
    baseline_meta_df["stage_cnt"] = pd.to_numeric(
        baseline_meta_df["stage_cnt"], errors="coerce"
    )
    baseline_meta_df = baseline_meta_df.dropna(
        subset=["instance_key", "job_cnt", "stage_cnt"]
    ).copy()
    baseline_meta_df["job_cnt"] = baseline_meta_df["job_cnt"].astype(int)
    baseline_meta_df["stage_cnt"] = baseline_meta_df["stage_cnt"].astype(int)

    if baseline_meta_df["instance_key"].duplicated().any():
        raise ValueError("Duplicate normalized instance keys in baseline DataFrame.")

    merged = metrics_df.merge(
        baseline_meta_df[["instance_key", "job_cnt", "stage_cnt"]],
        on="instance_key",
        how="left",
        validate="many_to_one",
    )

    unmatched_mask = merged["job_cnt"].isna() | merged["stage_cnt"].isna()
    if unmatched_mask.any():
        logging.warning(
            "Excluded %d metric rows from HTML chart due to missing baseline metadata.",
            int(unmatched_mask.sum()),
        )
        merged = merged.loc[~unmatched_mask].copy()

    return merged[
        ["instance_id", "subroutine_name", "norm_time", "rpd_f", "job_cnt", "stage_cnt"]
    ]


def _compute_best_so_far_y_values(y_values: list[float]) -> list[float]:
    best_y_values: list[float] = []
    current_best: float | None = None
    for y_val in y_values:
        current_best = y_val if current_best is None else min(current_best, y_val)
        best_y_values.append(current_best)
    return best_y_values


def _dedupe_progression_points(
    points: list[ProgressionPoint],
) -> list[ProgressionPoint]:
    deduped_by_time: dict[float, ProgressionPoint] = {}
    for point in points:
        deduped_by_time[point.time] = point
    return [deduped_by_time[time] for time in sorted(deduped_by_time)]


def _build_best_so_far_progression_points(grp: pd.DataFrame) -> list[ProgressionPoint]:
    if grp.empty:
        return []

    x_values = grp["norm_time"].tolist()
    best_y_values = _compute_best_so_far_y_values(grp["rpd_f"].tolist())
    points = [
        ProgressionPoint(time=float(x_val), rpd_f=float(y_val))
        for x_val, y_val in zip(x_values, best_y_values)
    ]
    return _dedupe_progression_points(points)


def _lookup_rpdf_at_or_before(
    progression_points: list[ProgressionPoint], query_time: float
) -> float | None:
    matched_rpd_f: float | None = None
    for point in progression_points:
        if point.time > query_time:
            break
        matched_rpd_f = point.rpd_f
    return matched_rpd_f


def _build_step_path(
    x_values: list[float], y_values: list[float]
) -> tuple[list[float], list[float]]:
    step_x: list[float] = []
    step_y: list[float] = []
    for idx, (x_val, y_val) in enumerate(zip(x_values, y_values)):
        if idx == 0:
            step_x.append(x_val)
            step_y.append(y_val)
            continue

        previous_y = y_values[idx - 1]
        step_x.append(x_val)
        step_y.append(previous_y)
        if y_val < previous_y:
            step_x.append(x_val)
            step_y.append(y_val)

    return step_x, step_y


def _build_step_aligned_values(base_values: list[T], y_values: list[float]) -> list[T]:
    aligned_values: list[T] = []
    for idx, base_value in enumerate(base_values):
        if idx == 0:
            aligned_values.append(base_value)
            continue

        aligned_values.append(base_values[idx - 1])
        if y_values[idx] < y_values[idx - 1]:
            aligned_values.append(base_value)

    return aligned_values


def _build_marker_meta_by_time(
    instance_id: object, grp: pd.DataFrame, job_cnt: int, stage_cnt: int
) -> dict[float, MarkerMeta]:
    marker_meta_by_time: dict[float, MarkerMeta] = {}
    for row in grp.itertuples():
        marker_time = float(row.norm_time)
        if marker_time in marker_meta_by_time:
            raise ValueError(
                f"Duplicate endpoint marker time for instance {instance_id}: {marker_time}"
            )
        marker_meta_by_time[marker_time] = MarkerMeta(
            instance_id=str(instance_id),
            job_cnt=job_cnt,
            stage_cnt=stage_cnt,
            subroutine_name=str(row.subroutine_name),
            count=1,
            time=marker_time,
        )
    return marker_meta_by_time


def _build_raw_instance_progression(
    instance_id: object,
    endpoint_grp: pd.DataFrame,
    progression_grp: pd.DataFrame | None,
    job_cnt: int,
    stage_cnt: int,
) -> RawInstanceProgression:
    series_id = f"instance={instance_id}"
    marker_meta_by_time = _build_marker_meta_by_time(
        instance_id, endpoint_grp, job_cnt, stage_cnt
    )

    if progression_grp is None or progression_grp.empty:
        progression_points = _build_best_so_far_progression_points(endpoint_grp)
    else:
        progression_points = _build_best_so_far_progression_points(progression_grp)

    return RawInstanceProgression(
        series_id=series_id,
        instance_id=str(instance_id),
        job_cnt=job_cnt,
        stage_cnt=stage_cnt,
        progression_points=progression_points,
        marker_meta_by_time=marker_meta_by_time,
    )


def _build_raw_plotly_series(model: RawInstanceProgression) -> dict:
    progression_x = [point.time for point in model.progression_points]
    progression_y = [point.rpd_f for point in model.progression_points]
    step_x, step_y = _build_step_path(progression_x, progression_y)

    marker_x = sorted(model.marker_meta_by_time)
    marker_meta = [model.marker_meta_by_time[time] for time in marker_x]
    marker_y = [
        _lookup_rpdf_at_or_before(model.progression_points, marker.time)
        for marker in marker_meta
    ]

    filtered_markers = [
        (x_val, y_val, meta)
        for x_val, y_val, meta in zip(marker_x, marker_y, marker_meta)
        if y_val is not None
    ]

    return {
        "series_id": model.series_id,
        "instance_id": model.instance_id,
        "job_cnt": model.job_cnt,
        "stage_cnt": model.stage_cnt,
        "x": [x_val for x_val, _, _ in filtered_markers],
        "y": [y_val for _, y_val, _ in filtered_markers],
        "step_x": step_x,
        "step_y": step_y,
        "text": [meta.subroutine_name for _, _, meta in filtered_markers],
        "customdata": [
            [
                meta.instance_id,
                meta.job_cnt,
                meta.stage_cnt,
                meta.subroutine_name,
                meta.count,
            ]
            for _, _, meta in filtered_markers
        ],
    }


def _build_raw_instance_progression_models(
    endpoint_df: pd.DataFrame, raw_progression_df: pd.DataFrame | None = None
) -> list[RawInstanceProgression]:
    raw_models: list[RawInstanceProgression] = []
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

    for instance_id, endpoint_grp in endpoint_df.groupby("instance_id", sort=True):
        endpoint_grp = endpoint_grp.sort_values(
            ["norm_time", "subroutine_order", "subroutine_name"]
        )
        n_val = int(endpoint_grp["job_cnt"].iloc[0])
        c_val = int(endpoint_grp["stage_cnt"].iloc[0])
        progression_grp = progression_by_instance.get(str(instance_id))
        raw_model = _build_raw_instance_progression(
            instance_id=instance_id,
            endpoint_grp=endpoint_grp,
            progression_grp=progression_grp,
            job_cnt=n_val,
            stage_cnt=c_val,
        )
        raw_models.append(raw_model)

    return raw_models


def _build_raw_series_payload(
    endpoint_df: pd.DataFrame, raw_progression_df: pd.DataFrame | None = None
) -> list[dict]:
    raw_models = _build_raw_instance_progression_models(endpoint_df, raw_progression_df)
    return [_build_raw_plotly_series(model) for model in raw_models]


def _build_mean_vertical_guides(
    models: list[RawInstanceProgression],
) -> list[MeanVerticalGuide]:
    subroutine_times: dict[str, list[float]] = {}
    for model in models:
        for marker_time in sorted(model.marker_meta_by_time):
            marker_meta = model.marker_meta_by_time[marker_time]
            subroutine_times.setdefault(marker_meta.subroutine_name, []).append(
                marker_time
            )

    return [
        MeanVerticalGuide(
            subroutine_name=subroutine_name,
            x=sum(times) / len(times),
        )
        for subroutine_name, times in sorted(subroutine_times.items())
        if times
    ]


def _build_mean_series_payload(raw_models: list[RawInstanceProgression]) -> list[dict]:
    models_by_group: dict[tuple[int, int], list[RawInstanceProgression]] = {}
    for model in raw_models:
        if not model.progression_points:
            continue
        models_by_group.setdefault((model.job_cnt, model.stage_cnt), []).append(model)

    mean_series: list[dict] = []
    for (job_cnt, stage_cnt), models in sorted(models_by_group.items()):
        first_times = [model.progression_points[0].time for model in models]
        last_times = [model.progression_points[-1].time for model in models]
        start_time = max(first_times)
        end_time = max(last_times)
        union_times = sorted(
            {
                point.time
                for model in models
                for point in model.progression_points
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
                if (
                    value := _lookup_rpdf_at_or_before(
                        model.progression_points, time_val
                    )
                )
                is not None
            ]
            if len(values) != len(models):
                continue
            mean_x.append(time_val)
            mean_y.append(sum(values) / len(values))

        if not mean_x:
            continue

        series_id = f"mean(job_cnt={job_cnt},stage_cnt={stage_cnt})"
        step_x, step_y = _build_step_path(mean_x, mean_y)
        point_customdata = [
            [series_id, job_cnt, stage_cnt, len(models)] for _ in mean_x
        ]
        step_customdata = _build_step_aligned_values(point_customdata, mean_y)
        guides = _build_mean_vertical_guides(models)
        mean_series.append(
            {
                "series_id": series_id,
                "job_cnt": job_cnt,
                "stage_cnt": stage_cnt,
                "x": mean_x,
                "y": mean_y,
                "step_x": step_x,
                "step_y": step_y,
                "instance_count": len(models),
                "vertical_guides": [
                    {"subroutine_name": guide.subroutine_name, "x": guide.x}
                    for guide in guides
                ],
                "guide_marker_x": [guide.x for guide in guides],
                "guide_marker_text": [guide.subroutine_name for guide in guides],
                "customdata": point_customdata,
                "step_customdata": step_customdata,
            }
        )

    return mean_series


def _build_html_payload(
    df: pd.DataFrame, raw_progression_df: pd.DataFrame | None = None
) -> dict:
    """Build the JSON payload used by the interactive HTML page."""
    if df.empty:
        return {
            "job_cnt_values": [],
            "stage_cnt_values": [],
            "raw_series": [],
            "mean_series": [],
        }

    order_map = {
        name: idx for idx, name in enumerate(pd.unique(df["subroutine_name"]), start=1)
    }
    work_df = df.copy()
    work_df["subroutine_order"] = work_df["subroutine_name"].map(order_map)

    n_values = sorted(work_df["job_cnt"].unique())
    c_values = sorted(df["stage_cnt"].unique())

    progression_work_df = None
    if raw_progression_df is not None and not raw_progression_df.empty:
        progression_work_df = raw_progression_df.copy()
        progression_work_df["subroutine_order"] = progression_work_df[
            "subroutine_name"
        ].map(order_map)
        progression_work_df = progression_work_df.dropna(
            subset=["subroutine_order"]
        ).copy()

    raw_models = _build_raw_instance_progression_models(work_df, progression_work_df)
    raw_series = [_build_raw_plotly_series(model) for model in raw_models]
    mean_series = _build_mean_series_payload(raw_models)

    return {
        "job_cnt_values": [int(x) for x in n_values],
        "stage_cnt_values": [int(x) for x in c_values],
        "raw_series": raw_series,
        "mean_series": mean_series,
    }


def _build_html_page(payload: dict, x_decimals: int, y_decimals: int) -> str:
    """Build complete HTML page with inline JavaScript."""
    n_options = "".join(
        [
            f'<option value="{v}">{v}</option>'
            for v in ["All", *payload["job_cnt_values"]]
        ]
    )
    c_options = "".join(
        [
            f'<option value="{v}">{v}</option>'
            for v in ["All", *payload["stage_cnt_values"]]
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RPDf vs Time% Interactive</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 18px;
      color: #1b1b1b;
    }}
    .toolbar {{
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .control {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    label {{
      font-size: 14px;
      font-weight: 600;
    }}
    select {{
      font-size: 14px;
      padding: 4px 8px;
      min-width: 100px;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="control">
      <label for="mode-filter">mode</label>
      <select id="mode-filter">
        <option value="raw">instance progression</option>
        <option value="mean">mean progression by (n,c)</option>
      </select>
    </div>
    <div class="control">
      <label for="job-cnt-filter">job_cnt</label>
      <select id="job-cnt-filter">{n_options}</select>
    </div>
    <div class="control">
      <label for="stage-cnt-filter">stage_cnt</label>
      <select id="stage-cnt-filter">{c_options}</select>
    </div>
  </div>
  <div id="rpdf-chart" style="width: 100%; height: 720px;"></div>
  <script>
    const DATA = {json.dumps(payload, separators=(",", ":"))};
    const chartId = "rpdf-chart";
    const xTickFormat = ".{x_decimals}%";
    const yTickFormat = ".{y_decimals}%";
    const SERIES_COLORS = [
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ];
    const SYMBOL_MAP = {{
      "initialize_by_best_of_selected_dispatches": "circle",
      "neh_cp": "diamond",
      "set_cp_model_as_base_cp_model": "square",
      "run_reactive_loop": "x",
      "apply_shdlb": "triangle",
      "repeat_while_improvement": "square-open",
      "finalize": "star"
    }};
    const modeFilter = document.getElementById("mode-filter");
    const jobCntFilter = document.getElementById("job-cnt-filter");
    const stageCntFilter = document.getElementById("stage-cnt-filter");

    function buildLayout(titleText) {{
      return {{
        title: {{ text: titleText }},
        xaxis: {{
          title: {{ text: "Time%" }},
          tickformat: xTickFormat,
          rangemode: "tozero"
        }},
        yaxis: {{
          title: {{ text: "RPDf%" }},
          tickformat: yTickFormat,
          rangemode: "tozero"
        }},
        template: "plotly_white",
        margin: {{ l: 60, r: 30, t: 70, b: 60 }},
        showlegend: false,
        hovermode: "closest"
      }};
    }}

    function applyFilters() {{
      const modeVal = modeFilter.value;
      const jobCntVal = jobCntFilter.value;
      const stageCntVal = stageCntFilter.value;
      const source = modeVal === "mean" ? DATA.mean_series : DATA.raw_series;
      const selected = source.filter((s) => {{
        const jobCntMatch = (
          jobCntVal === "All" || String(s.job_cnt) === jobCntVal
        );
        const stageCntMatch = (
          stageCntVal === "All" || String(s.stage_cnt) === stageCntVal
        );
        return jobCntMatch && stageCntMatch;
      }});

      const traces = selected.flatMap((s, idx) => {{
        const seriesColor = SERIES_COLORS[idx % SERIES_COLORS.length];
        const traceName = modeVal === "mean"
          ? `job_cnt=${{s.job_cnt}}, stage_cnt=${{s.stage_cnt}}`
          : `instance=${{s.instance_id}}`;

        if (modeVal === "raw") {{
          const symbols = s.text.map((name) => SYMBOL_MAP[name] || "circle");
          return [
            {{
              type: "scatter",
              mode: "lines",
              x: s.step_x,
              y: s.step_y,
              name: traceName,
              line: {{ width: 1.0, color: seriesColor }},
              hoverinfo: "skip",
              showlegend: false
            }},
            {{
              type: "scatter",
              mode: "markers",
              x: s.x,
              y: s.y,
              customdata: s.customdata,
              name: traceName,
              marker: {{ size: 7, symbol: symbols, color: seriesColor }},
              hovertemplate:
                "series=%{{customdata[0]}}<br>" +
                "job_cnt=%{{customdata[1]}}<br>" +
                "stage_cnt=%{{customdata[2]}}<br>" +
                "subroutine=%{{customdata[3]}}<br>" +
                "count=%{{customdata[4]}}<br>" +
                "Time%%=%{{x:.4%}}<br>" +
                "RPDf=%{{y:.4%}}<extra></extra>",
              showlegend: false
            }}
          ];
        }}

        return [{{
          type: "scatter",
          mode: "lines",
          x: s.step_x,
          y: s.step_y,
          customdata: s.step_customdata,
          name: traceName,
          line: {{ width: 2.0, color: seriesColor }},
          hovertemplate:
            "series=%{{customdata[0]}}<br>" +
            "job_cnt=%{{customdata[1]}}<br>" +
            "stage_cnt=%{{customdata[2]}}<br>" +
            "instance_cnt=%{{customdata[3]}}<br>" +
            "Time%%=%{{x:.4%}}<br>" +
            "RPDf=%{{y:.4%}}<extra></extra>",
          showlegend: false
        }}, {{
          type: "scatter",
          mode: "markers",
          x: s.guide_marker_x || [],
          y: (s.guide_marker_x || []).map(() => 0),
          text: s.guide_marker_text || [],
          name: traceName,
          customdata: (s.guide_marker_text || []).map((name) => [
            traceName,
            s.job_cnt,
            s.stage_cnt,
            name
          ]),
          marker: {{
            size: 8,
            symbol: (s.guide_marker_text || []).map((name) => SYMBOL_MAP[name] || "circle"),
            color: seriesColor
          }},
          hovertemplate:
            "series=%{{customdata[0]}}<br>" +
            "job_cnt=%{{customdata[1]}}<br>" +
            "stage_cnt=%{{customdata[2]}}<br>" +
            "subroutine=%{{customdata[3]}}<br>" +
            "avg end Time%%=%{{x:.4%}}<extra></extra>",
          showlegend: false
        }}];
      }});

      const modeLabel = modeVal === "mean"
        ? "mean progression by (job_cnt, stage_cnt)"
        : "instance progression";
      const layout = buildLayout(
        `RPDf vs Time% - ${{modeLabel}} (${{selected.length}} lines)`
      );
      if (modeVal === "mean") {{
        layout.shapes = selected.flatMap((series, idx) => {{
          const seriesColor = SERIES_COLORS[idx % SERIES_COLORS.length];
          return (series.vertical_guides || []).map((guide) => {{
            return {{
              type: "line",
              xref: "x",
              yref: "paper",
              x0: guide.x,
              x1: guide.x,
              y0: 0,
              y1: 1,
              line: {{
                color: seriesColor,
                width: 1,
                dash: "dot"
              }}
            }};
          }});
        }});
      }}
      Plotly.react(chartId, traces, layout, {{ responsive: true }});
    }}

    modeFilter.addEventListener("change", applyFilters);
    jobCntFilter.addEventListener("change", applyFilters);
    stageCntFilter.addEventListener("change", applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def export_method_rpdf_scatter_html(
    metrics_long_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: Path,
    baseline_instance_col: str,
    baseline_job_cnt_col: str,
    baseline_stage_cnt_col: str,
    raw_progression_df: pd.DataFrame | None = None,
    x_percent_decimals: int = 1,
    y_percent_decimals: int = 1,
) -> bool:
    """Generate interactive HTML chart with job/stage filters.

    Args:
        metrics_long_df: DataFrame with instance_id, subroutine_name, norm_time, rpd_f
        baseline_df: DataFrame containing instance metadata
        output_path: Output HTML file path
        baseline_instance_col: Baseline column holding the instance key
        baseline_job_cnt_col: Baseline column holding job count
        baseline_stage_cnt_col: Baseline column holding stage count
        x_percent_decimals: Decimals for x-axis percent ticks
        y_percent_decimals: Decimals for y-axis percent ticks

    Returns:
        True if successful, False if no valid data
    """
    try:
        merged_df = _load_and_merge_for_html(
            metrics_long_df=metrics_long_df,
            baseline_df=baseline_df,
            baseline_instance_col=baseline_instance_col,
            baseline_job_cnt_col=baseline_job_cnt_col,
            baseline_stage_cnt_col=baseline_stage_cnt_col,
        )
    except ValueError as e:
        logging.error(f"Failed to load/merge data for HTML chart: {e}", exc_info=True)
        return False

    merged_raw_progression_df: pd.DataFrame | None = None
    if raw_progression_df is not None and not raw_progression_df.empty:
        try:
            merged_raw_progression_df = _load_and_merge_for_html(
                metrics_long_df=raw_progression_df,
                baseline_df=baseline_df,
                baseline_instance_col=baseline_instance_col,
                baseline_job_cnt_col=baseline_job_cnt_col,
                baseline_stage_cnt_col=baseline_stage_cnt_col,
            )
        except ValueError as e:
            logging.warning(
                "Failed to load raw progression data for HTML chart; "
                "falling back to endpoint-only raw series: %s",
                e,
            )
            merged_raw_progression_df = None

    payload = _build_html_payload(
        merged_df, raw_progression_df=merged_raw_progression_df
    )

    if not payload["raw_series"] and not payload["mean_series"]:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = _build_html_page(payload, x_percent_decimals, y_percent_decimals)
    output_path.write_text(html_content, encoding="utf-8")

    logging.info(f"Method RPD scatter HTML saved to {output_path}")
    return True
