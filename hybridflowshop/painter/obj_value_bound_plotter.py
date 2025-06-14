from pathlib import Path
from typing import Optional

from mbls.cpsat import ObjValueBoundStore

from .time_series_plotter import TimeSeriesPlotter


class ObjValueBoundPainter:
    """
    Painter class to visualize the objective value and bound stored in ObjValueBoundStore.
    It utilizes TimeSeriesPlotter to render time series plots.
    """

    @staticmethod
    def plot(
        store: ObjValueBoundStore,
        save_path: Path,
        show_markers: bool = True,
        drop_first_values_percent: float = 0.01,
        title: str = "Objective Value and Bound Over Time",
        xlabel: str = "Elapsed Time (seconds)",
        ylabel: str = "Objective",
        legend_loc: str = "upper right",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        show: bool = False,
        dpi: int = 300,
    ):
        """
        Plot the objective value and bound from the given store.

        Args:
            store (ObjValueBoundStore): The store containing the time series.
            save_path (Path): File path to save the plot image.
            show_markers (bool): Whether to show dots on step points.
            drop_first_values_percent (float): Drop early fraction of values (e.g. 0.01 for 1%).
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            legend_loc (str): Legend location.
            xlim (tuple, optional): X-axis limits.
            ylim (tuple, optional): Y-axis limits.
            show (bool): Whether to display the plot interactively.
            dpi (int): DPI for saved figure.
        """
        # Get time series
        obj_value_log = store.obj_value_series.items()
        obj_bound_log = store.obj_bound_series.items()

        lists_of_time_and_val = [obj_value_log, obj_bound_log]
        labels = ["ObjValue", "ObjBound"]

        note_map_val = store.obj_value_series.timestamp_note_map
        note_map_bound = store.obj_bound_series.timestamp_note_map
        maps_of_time_to_note = [note_map_val, note_map_bound]

        linestyles = ["-", "--"]  # Solid for value, dashed for bound

        TimeSeriesPlotter.plot_lists_of_time_and_val(
            lists_of_time_and_val=lists_of_time_and_val,
            save_path=save_path,
            maps_of_time_to_note=maps_of_time_to_note,
            linestyles=linestyles,
            show_markers=show_markers,
            labels=labels,
            drop_first_values_percent=drop_first_values_percent,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_loc=legend_loc,
            xlim=xlim,
            ylim=ylim,
            show=show,
            dpi=dpi,
        )
