import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


class ObjectiveProgressPlotter:
    @staticmethod
    def plot_solution_progress(
        progress_log: list[tuple[float, float, float]],
        save_path: Path,
        show_markers: bool = True,
    ):
        """Plot the solution progress over time with optional markers.

        Args:
            progress_log (list[tuple[float, float, float]]): A list of (elapsed time, objective value, best bound).
            save_path (Path): Path to save the plot.
            show_markers (bool, optional): Whether to show markers at each step. Defaults to True.
        """  # noqa: E501
        if not progress_log:
            logging.warning("No progress data available to plot.")
            return

        elapsed_times, objectives, best_bounds = zip(*progress_log)

        plt.figure(figsize=(10, 6))

        # Objective Value
        plt.step(
            elapsed_times,
            objectives,
            where="post",
            label="Objective Value",
            color="tab:blue",
        )
        if show_markers:
            plt.scatter(
                elapsed_times,
                objectives,
                color="tab:blue",
                facecolor="black",
                marker="o",
                s=40,
                zorder=3,
                label="Obj Update Points",
            )

        # Best Bound
        plt.step(
            elapsed_times,
            best_bounds,
            where="post",
            linestyle="--",
            label="Best Bound",
            color="tab:orange",
        )
        if show_markers:
            plt.scatter(
                elapsed_times,
                best_bounds,
                color="tab:orange",
                facecolor="black",
                marker="x",
                s=40,
                zorder=3,
                label="Bound Update Points",
            )

        plt.xlabel("Elapsed Time (seconds)")
        plt.ylabel("Objective Value")
        plt.title("Solution Progress Over Time")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.xlim(left=0)

        plt.tight_layout()

        plt.savefig(save_path)
        logging.info(f"Solution progress plot saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_lists_of_time_and_obj(
        lists_of_time_and_obj: list[list[tuple[float, float]]],
        save_path: Path,
        show_markers: bool = True,
        labels: Optional[list[str]] = None,
        drop_first_values_percent: float = 0.05,
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
        """Plot multiple lists of (time, objective) pairs.

        Args:
            lists_of_time_and_obj (list[list[tuple[float, float]]]): Multiple lists containing (time, objective) tuples.
            save_path (Path): Path to save the plot.
            show_markers (bool, optional): Whether to show markers at each step. Defaults to True.
            labels (Optional[list[str]] , optional): Labels for each list. Defaults to None.
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

        for i, time_and_obj in enumerate(lists_of_time_and_obj):
            if not time_and_obj:
                logging.warning(f"No data available for list {i + 1}. Skipping.")
                continue

            # 각 list에서 앞의 값 버리기
            n = len(time_and_obj)
            skip = int(n * drop_first_values_percent)
            time_and_obj = time_and_obj[skip:] if skip < n else time_and_obj

            times, objectives = zip(*time_and_obj)

            plt.step(
                times,
                objectives,
                where="post",
                label=labels[i] if labels else f"List {i + 1}",
            )
            if show_markers:
                plt.scatter(
                    times,
                    objectives,
                    facecolor="black",
                    marker="o",
                    s=40,
                    zorder=3,
                )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=legend_loc)
        plt.grid(grid, linestyle=grid_style, alpha=grid_alpha)

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        plt.tight_layout()

        plt.savefig(save_path, format=save_format)
        logging.info(f"{title} plot saved to {save_path}")

        if show:
            plt.show()

        plt.close()
