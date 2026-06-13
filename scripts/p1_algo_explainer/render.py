"""Render helper wrapping GanttPlotter for slide-ready SVG panels (plan §2.3).

A single ``show_labels`` flag controls ALL text labels (bar job ID, bar
processing time, x-axis label/ticks, y-axis label/ticks). Default off yields a
clean color-bar + highlight chart. Instance meta (n, c, machine count, file
name, distribution) and the generic chart title are NEVER shown.

The production module ``hybridflowshop/painter/gantt.py`` is not modified; we
subclass ``GanttPlotter`` instead (plan §2.3.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

from hybridflowshop.painter.gantt import GanttPlotter

OpKey = tuple[str, str, str]


class LabelControlledGanttPlotter(GanttPlotter):
    """GanttPlotter whose text labels are governed by a single flag.

    ``show_labels`` toggles bar job IDs, bar durations, and both axis
    labels/ticks together. The instance-meta title is always cleared.
    """

    def __init__(
        self,
        show_labels: bool = False,
        op_color_map: Mapping[OpKey, tuple[float, float, float, float] | str]
        | None = None,
        vlines: Sequence[float] | None = None,
        show_x_ticks: bool = False,
        x_tick_step: int | None = None,
        lane_boundaries: Mapping[tuple[str, str], tuple[float, float]] | None = None,
        band_region_colors: tuple[str, str, str] | None = None,
        region_legend: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.show_labels = show_labels
        # Optional per-operation facecolor override (e.g. ISW-CP 5-region
        # coloring). When an op key is present, its bar uses this color instead
        # of the by-job color. Backward-compatible: None means by-job coloring.
        self.op_color_map = op_color_map
        # Optional red dashed vertical lines (e.g. ISW-CP window boundaries).
        self.vlines = vlines
        # ISW-CP 5-region partition (ref ffc_dw_wET_2026 visual.py): instead of a
        # single global window, every machine lane gets its OWN left/right
        # time-fixed boundary. ``lane_boundaries`` maps (stage, machine) ->
        # (left_b, right_b): left_b = rightmost LTF end on that machine, right_b
        # = leftmost RTF start. Behind the bars we lay three faint background
        # bands -- [x0, left_b] LTF zone, [left_b, right_b] active (LPF/UNFIXED/
        # RPF) zone, [right_b, x1] RTF zone -- and draw a short dashed segment at
        # each boundary, spanning only that lane. This gives every lane its own
        # fixed/unfixed context, which the global red window could not express.
        self.lane_boundaries = lane_boundaries
        # (LTF-zone, active-zone, RTF-zone) hex colors for the background bands.
        self.band_region_colors = band_region_colors
        # Optional [(label, hex_color), ...] for a region color key (legend).
        self.region_legend = region_legend
        # QSR compression visualization (plan §9): numeric x-axis ticks even
        # when show_labels=False. The tick NUMBERS (time coordinates) carry the
        # tau-compression story, not bar width. Independent of show_labels and
        # of the instance-meta ban (tick numbers are time coordinates, allowed).
        self.show_x_ticks = show_x_ticks
        # When set, place x-axis ticks at multiples of this step via
        # MultipleLocator; labels render as integer time coordinates.
        self.x_tick_step = x_tick_step
        # Stage separators: a thick grey dash-dot ("-.-.-") horizontal line drawn
        # in the whitespace gap between the last machine lane of one stage and the
        # first of the next. Not a text label -> independent of show_labels (like
        # vlines). Always on; tweak via these attributes.
        self.stage_separator_color = "#6b6b6b"
        self.stage_separator_linewidth = 2.2
        self.stage_separator_linestyle = "-."

    def draw_operation_bars(
        self,
        start_time_map: Mapping[OpKey, int],
        end_time_map: Mapping[OpKey, int],
        job_to_color: Mapping[str, tuple[float, float, float, float]],
        machine_to_y: Mapping[tuple[str, str], float],
        job_list: Sequence[str],
        highlight_op_set: set[tuple[str, str]] | None = None,
    ) -> None:
        # Same iteration as the parent, but pass show_label/show_duration
        # driven by the single flag.
        for (job, stage, machine), s_time in start_time_map.items():
            if job_list and job not in job_list:
                continue
            if (stage, machine) not in machine_to_y:
                continue

            e_time = end_time_map[(job, stage, machine)]
            y = machine_to_y[(stage, machine)]
            color = job_to_color[job]
            if (
                self.op_color_map is not None
                and (job, stage, machine) in self.op_color_map
            ):
                color = self.op_color_map[(job, stage, machine)]
            is_highlight = (
                highlight_op_set is not None and (job, stage) in highlight_op_set
            )
            self.draw_operation_bar(
                job=job,
                stage=stage,
                machine=machine,
                s_time=s_time,
                e_time=e_time,
                color=color,
                y=y,
                show_label=self.show_labels,
                show_duration=self.show_labels,
                highlight=is_highlight,
            )

    def draw_operation_bar(self, *args, **kwargs):  # type: ignore[override]
        # When region coloring is active, render solid bars so the 5-region
        # palette stays salient (the by-job path uses alpha=0.5).
        if self.op_color_map is not None and not kwargs.get("highlight", False):
            self.bar_alpha = 1.0
        super().draw_operation_bar(*args, **kwargs)
        # Highlighted bars (e.g. MD mixed-dispatch head jobs) get a red border
        # instead of the parent's hardcoded black edge. Overlay an unfilled
        # red-edged rectangle over the same geometry; same linewidth fully
        # covers the black edge. Keeps gantt.py unmodified (module docstring).
        if kwargs.get("highlight", False):
            assert self.ax is not None
            s_time = kwargs["s_time"]
            e_time = kwargs["e_time"]
            y = kwargs["y"]
            self.ax.add_patch(
                mpatches.Rectangle(
                    (s_time, y),
                    e_time - s_time,
                    self.bar_height,
                    facecolor="none",
                    edgecolor="#FF0000",
                    linewidth=3.0,
                    zorder=3,
                )
            )

    def plot_hybrid_flowshop(self, *args, **kwargs):  # type: ignore[override]
        super().plot_hybrid_flowshop(*args, **kwargs)
        assert self.ax is not None
        # Instance meta / generic title: never shown.
        self.ax.set_title("")
        # Per-lane partition bands + boundaries (ISW-CP 5-region viz).
        self._draw_lane_partition_bands(args, kwargs)
        # Stage separators (thick grey dash-dot lines between stages).
        self._draw_stage_separators(args, kwargs)
        # Red dashed vertical window-boundary lines (drawn after the bars so
        # they stay on top). Not a text label -> independent of show_labels.
        if self.vlines:
            for x in self.vlines:
                self.ax.axvline(
                    x=x, color="red", linestyle="--", linewidth=2.0, zorder=5
                )
        if self.show_labels:
            # Restore a neutral x-axis label; keep y machine-lane labels.
            self.ax.set_xlabel("Time")
        else:
            # Clean chart: no axis labels or job/y tick text. The x tick text is
            # only wiped when show_x_ticks is off (QSR keeps numeric x ticks).
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")
            self.ax.set_yticklabels([])
            if not self.show_x_ticks:
                self.ax.set_xticklabels([])
        # Numeric x-axis ticks for the QSR compression story (plan §9). Applied
        # AFTER super() set the xlim so the locator spans this panel's own axis.
        if self.show_x_ticks and self.x_tick_step is not None:
            self.ax.xaxis.set_major_locator(MultipleLocator(self.x_tick_step))
        # Region color key (ISW-CP). A legend is a color key, not instance meta,
        # so it is allowed even with show_labels off.
        if self.region_legend:
            handles = [
                mpatches.Patch(facecolor=color, edgecolor="black", label=label)
                for label, color in self.region_legend
            ]
            self.ax.legend(
                handles=handles,
                loc="upper right",
                fontsize=8,
                ncol=len(handles),
                framealpha=0.85,
                handlelength=1.2,
                columnspacing=1.0,
            )

    def _lane_order(self, args, kwargs) -> list[tuple[str, str]]:
        """Resolve (stage, machine) lanes in painter render order.

        Mirrors ``GanttPlotter.create_machine_lanes``: lane idx i -> y = i.
        """
        start_time_map = args[0] if args else kwargs.get("start_time_map")
        if not start_time_map:
            return []
        stage_list = kwargs.get("stage_list")
        if not stage_list:
            stage_list = sorted({stage for (_, stage, _) in start_time_map})
        machine_list_per_stage = kwargs.get("machine_list_per_stage")
        lanes: list[tuple[str, str]] = []
        for stage in stage_list:
            machines = (
                machine_list_per_stage.get(stage) if machine_list_per_stage else None
            )
            if not machines:
                machines = sorted(
                    {mc for (_, stg, mc) in start_time_map if stg == stage}
                )
            for mc in machines:
                lanes.append((stage, mc))
        return lanes

    def _draw_lane_partition_bands(self, args, kwargs) -> None:
        """Lay faint per-lane LTF/active/RTF bands + dashed boundary segments.

        Each lane ``i`` (the bars span ``[i, i + bar_height]``) gets, behind the
        bars, three background bands split at its ``(left_b, right_b)`` boundary,
        plus a short dashed vertical segment at each boundary confined to that
        lane. Boundaries that coincide with the panel's x-extent are omitted (no
        LTF / no RTF on that machine -> nothing to mark).
        """
        if not self.lane_boundaries:
            return
        assert self.ax is not None
        ltf_color, active_color, rtf_color = self.band_region_colors or (
            "#90A4AE",
            "#81C784",
            "#607D8B",
        )
        x0, x1 = self.ax.get_xlim()
        for idx, lane in enumerate(self._lane_order(args, kwargs)):
            bounds = self.lane_boundaries.get(lane)
            if bounds is None:
                continue
            left_b, right_b = bounds
            left_b = max(x0, min(left_b, x1))
            right_b = max(x0, min(right_b, x1))
            y0 = float(idx)
            h = self.bar_height
            if left_b > x0:
                self.ax.add_patch(
                    mpatches.Rectangle(
                        (x0, y0),
                        left_b - x0,
                        h,
                        facecolor=ltf_color,
                        alpha=0.10,
                        edgecolor="none",
                        zorder=0,
                    )
                )
            if right_b > left_b:
                self.ax.add_patch(
                    mpatches.Rectangle(
                        (left_b, y0),
                        right_b - left_b,
                        h,
                        facecolor=active_color,
                        alpha=0.07,
                        edgecolor="none",
                        zorder=0,
                    )
                )
            if right_b < x1:
                self.ax.add_patch(
                    mpatches.Rectangle(
                        (right_b, y0),
                        x1 - right_b,
                        h,
                        facecolor=rtf_color,
                        alpha=0.10,
                        edgecolor="none",
                        zorder=0,
                    )
                )
            for boundary in (left_b, right_b):
                if x0 < boundary < x1:
                    self.ax.plot(
                        [boundary, boundary],
                        [y0, y0 + h],
                        color="#37474F",
                        linestyle=(0, (5, 3)),
                        linewidth=1.3,
                        zorder=3,
                    )

    def _draw_stage_separators(self, args, kwargs) -> None:
        """Draw a grey dash-dot horizontal line between consecutive stages.

        Machine lanes are stacked stage-by-stage at integer y (lane idx i spans
        ``[i, i + bar_height]``; ``machine_height`` is the lane pitch). A stage
        boundary after a cumulative ``c`` lanes sits in the whitespace gap
        between lane ``c-1`` (bottom ``c-1+bar_height``) and lane ``c`` (top
        ``c``); we centre the line in that gap. The lane order here mirrors
        ``GanttPlotter.create_machine_lanes`` exactly so boundaries align.
        """
        assert self.ax is not None
        # start_time_map is the first positional arg of plot_hybrid_flowshop.
        start_time_map = args[0] if args else kwargs.get("start_time_map")
        if not start_time_map:
            return
        stage_list = kwargs.get("stage_list")
        if not stage_list:
            stage_list = sorted({stage for (_, stage, _) in start_time_map})
        machine_list_per_stage = kwargs.get("machine_list_per_stage")

        # Lanes per stage, in render order (same resolution as the painter).
        lanes_per_stage: list[int] = []
        for stage in stage_list:
            machines = (
                machine_list_per_stage.get(stage) if machine_list_per_stage else None
            )
            if not machines:
                machines = sorted(
                    {mc for (_, stg, mc) in start_time_map if stg == stage}
                )
            lanes_per_stage.append(len(machines))

        gap = self.machine_height - self.bar_height  # whitespace between lanes
        cumulative = 0
        for lane_count in lanes_per_stage[:-1]:  # no line after the last stage
            cumulative += lane_count
            # Centre of the gap below the last lane of this stage.
            separator_y = cumulative * self.machine_height - gap / 2
            self.ax.axhline(
                y=separator_y,
                color=self.stage_separator_color,
                linestyle=self.stage_separator_linestyle,
                linewidth=self.stage_separator_linewidth,
                zorder=4,
            )


def render_panel(
    out_path: str | Path,
    start_map: Mapping[OpKey, int],
    end_map: Mapping[OpKey, int],
    *,
    all_job_list: Sequence[str],
    highlight_op_set: set[tuple[str, str]] | None = None,
    force_start: int | None = None,
    force_end: int | None = None,
    show_labels: bool = False,
    stage_list: Sequence[str] | None = None,
    machine_list_per_stage: Mapping[str, Sequence[str]] | None = None,
    op_color_map: Mapping[OpKey, tuple[float, float, float, float] | str] | None = None,
    vlines: Sequence[float] | None = None,
    show_x_ticks: bool = False,
    x_tick_step: int | None = None,
    lane_boundaries: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    band_region_colors: tuple[str, str, str] | None = None,
    region_legend: Sequence[tuple[str, str]] | None = None,
) -> Path:
    """Instantiate the label-controlled plotter and export one SVG panel.

    Color consistency: pass the demo's full job ID list as ``all_job_list`` so
    every panel colors jobs identically. Axis consistency: pass shared
    ``force_start`` / ``force_end``.

    ISW-CP extensions (both default ``None`` -> existing callers unchanged):

    * ``op_color_map``: per-(job, stage, machine) facecolor override. Used for
      5-region partition coloring (LTF/LPF/UNFIXED/RPF/RTF) that by-job color
      cannot express. Ops absent from the map keep their by-job color.
    * ``vlines``: x positions for red dashed vertical lines (window boundaries).

    QSR extensions (plan §9; both default off -> existing callers unchanged):

    * ``show_x_ticks``: when True, show numeric x-axis tick labels even with
      ``show_labels=False`` (xlabel "Time" stays off; y/job labels stay off).
    * ``x_tick_step``: place x-axis ticks at multiples of this step. The tick
      NUMBER spacing ratio (real 100 : surrogate 4 = tau) carries the
      compression story; bar width is the same on both axes.

    ISW-CP 5-region extensions (all default off -> existing callers unchanged):

    * ``lane_boundaries``: (stage, machine) -> (left_b, right_b) per-lane
      time-fixed boundaries; the renderer paints faint LTF/active/RTF bands and
      dashed boundary segments per lane (sister-repo pw_cp/visual.py style).
    * ``band_region_colors``: (LTF-zone, active-zone, RTF-zone) band colors.
    * ``region_legend``: [(label, hex_color), ...] color key drawn upper-right.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = LabelControlledGanttPlotter(
        show_labels=show_labels,
        op_color_map=op_color_map,
        vlines=vlines,
        show_x_ticks=show_x_ticks,
        x_tick_step=x_tick_step,
        lane_boundaries=lane_boundaries,
        band_region_colors=band_region_colors,
        region_legend=region_legend,
    )
    plotter.export_hybrid_flowshop_plot(
        file_path=out_path,
        start_time_map=start_map,
        end_time_map=end_map,
        stage_list=stage_list,
        machine_list_per_stage=machine_list_per_stage,
        all_job_list=all_job_list,
        highlight_op_set=highlight_op_set,
        force_start=force_start,
        force_end=force_end,
    )
    return out_path
