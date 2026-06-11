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
    ) -> None:
        super().__init__()
        self.show_labels = show_labels
        # Optional per-operation facecolor override (e.g. ISW-CP 5-region
        # coloring). When an op key is present, its bar uses this color instead
        # of the by-job color. Backward-compatible: None means by-job coloring.
        self.op_color_map = op_color_map
        # Optional red dashed vertical lines (e.g. ISW-CP window boundaries).
        self.vlines = vlines
        # QSR compression visualization (plan §9): numeric x-axis ticks even
        # when show_labels=False. The tick NUMBERS (time coordinates) carry the
        # tau-compression story, not bar width. Independent of show_labels and
        # of the instance-meta ban (tick numbers are time coordinates, allowed).
        self.show_x_ticks = show_x_ticks
        # When set, place x-axis ticks at multiples of this step via
        # MultipleLocator; labels render as integer time coordinates.
        self.x_tick_step = x_tick_step

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
            if self.op_color_map is not None and (job, stage, machine) in self.op_color_map:
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

    def plot_hybrid_flowshop(self, *args, **kwargs):  # type: ignore[override]
        super().plot_hybrid_flowshop(*args, **kwargs)
        assert self.ax is not None
        # Instance meta / generic title: never shown.
        self.ax.set_title("")
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
    op_color_map: Mapping[OpKey, tuple[float, float, float, float] | str]
    | None = None,
    vlines: Sequence[float] | None = None,
    show_x_ticks: bool = False,
    x_tick_step: int | None = None,
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
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = LabelControlledGanttPlotter(
        show_labels=show_labels,
        op_color_map=op_color_map,
        vlines=vlines,
        show_x_ticks=show_x_ticks,
        x_tick_step=x_tick_step,
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
