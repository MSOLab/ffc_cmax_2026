import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt


class TimeSeriesPlotter:
    @staticmethod
    def plot_lists_of_time_and_val(
        lists_of_time_and_val: list[list[tuple[float, float]]],
        save_path: Path,
        maps_of_time_to_note: Optional[list[dict[float, Any]]] = None,
        linestyles: Optional[list[str]] = None,
        show_markers: bool = True,
        labels: Optional[list[str]] = None,
        drop_first_values_percent: float = 0.0,
        title: str = "Objective Progress",
        xlabel: str = "Elapsed Time (seconds)",
        ylabel: str = "Objective Value",
        legend_loc: str = "upper right",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        grid: bool = True,
        grid_style: str = "--",
        grid_alpha: float = 0.6,
        figsize: tuple[int, int] = (10, 6),
        dpi: int = 300,
        save_format: str = "png",
        show: bool = False,
    ):
        """Plot multiple lists of (time, value) pairs.

        Args:
            lists_of_time_and_val (list[list[tuple[float, float]]]): Multiple lists containing (time, value) tuples.
            save_path (Path): Path to save the plot.
            maps_of_time_to_note (Optional[list[dict[float, Any]]], optional): Optional mapping of time to notes for each list.
                Defaults to None.
            linestyles (Optional[list[str]], optional): Optional list of line styles for each list.
                If provided, should match the number of lists.
                Defaults to None.
            show_markers (bool, optional): Whether to show markers at each step.
                Defaults to True.
            labels (list[str], optional): Labels for each list.
                Defaults to None.
            drop_first_values_percent (float, optional): Percentage of the first values to drop from each list.
                This is useful to remove initial noise or irrelevant data.
                0.01 means dropping the first 1% of values from each list.
                If the list is too short, it will not drop any values.
                Defaults to 0.0.
            title (str, optional): Title of the plot. Defaults to "Objective Progress".
            xlabel (str, optional): X-axis label. Defaults to "Elapsed Time (seconds)".
            ylabel (str, optional): Y-axis label. Defaults to "Objective Value".
            legend_loc (str, optional): Location of the legend. Defaults to "upper right".
            xlim (Optional[tuple[float, float]], optional): X-axis limits. Defaults to None.
            ylim (Optional[tuple[float, float]], optional): Y-axis limits. Defaults to None.
            grid (bool, optional): Whether to show grid lines. Defaults to True.
            grid_style (str, optional): Style of the grid lines. Defaults to "--".
            grid_alpha (float, optional): Transparency of the grid lines. Defaults to 0.6.
            figsize (tuple[int, int], optional): Size of the figure. Defaults to (10, 6).
            dpi (int, optional): Dots per inch for the figure. Defaults to 300.
            save_format (str, optional): Format to save the plot. Defaults to "png".
            show (bool, optional): Whether to display the plot interactively. Defaults to False.
        """

        plt.figure(figsize=figsize, dpi=dpi)

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, time_and_obj_list in enumerate(lists_of_time_and_val):
            edge_color = color_cycle[i % len(color_cycle)]
            if not time_and_obj_list:
                logging.warning(f"No data available for list {i + 1}. Skipping.")
                continue

            # Drop the first percentage of values
            n = len(time_and_obj_list)
            skip = int(n * drop_first_values_percent)
            time_and_obj_list = (
                time_and_obj_list[skip:] if skip < n else time_and_obj_list
            )

            times, objectives = zip(*time_and_obj_list)
            linestyle = linestyles[i] if linestyles else "-"

            plt.step(
                times,
                objectives,
                where="post",
                color=edge_color,
                linestyle=linestyle,
                label=labels[i] if labels else f"List {i + 1}",
            )
            if show_markers:
                plt.scatter(
                    times,
                    objectives,
                    facecolor=edge_color,
                    edgecolor="black",
                    marker="o",
                    s=40,
                    zorder=3,
                )
            y_offset = 0
            if maps_of_time_to_note and i < len(maps_of_time_to_note):
                time_to_label_map = maps_of_time_to_note[i]
                reversed_time_list = sorted(time_to_label_map.keys(), reverse=True)
                for time in reversed_time_list:
                    if time not in times:
                        continue  # Skip if time not in the current list
                    label_str = str(time_to_label_map[time])

                    idx = times.index(time)
                    y_pos = objectives[idx]

                    text_offset = (20, 20 + y_offset)
                    y_offset += 10

                    plt.annotate(
                        label_str,
                        xy=(time, y_pos),
                        xytext=text_offset,
                        textcoords="offset points",
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="gray", lw=1),
                        arrowprops=dict(arrowstyle="->", color=edge_color),
                    )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=legend_loc)
        plt.grid(grid, linestyle=grid_style, alpha=grid_alpha)

        if xlim:
            plt.xlim(xlim)
        else:
            plt.xlim(left=0)
        if ylim:
            plt.ylim(ylim)

        plt.tight_layout()

        plt.savefig(save_path, format=save_format)
        logging.info(f"{title} plot saved to {save_path}")

        if show:
            plt.show()

        plt.close()
