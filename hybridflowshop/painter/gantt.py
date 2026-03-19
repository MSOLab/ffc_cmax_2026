import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class GanttPlotter:
    # matplotlib.pyplot variables
    fig: Figure
    ax: Axes

    # constants
    cmap_name = "tab20"
    machine_height = 1.0
    bar_height = 0.8
    bar_alpha = 0.5
    grid_alpha = 0.3
    figsize = (12, 8)

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def display_hybrid_flowshop_plot(
        self,
        start_time_map: Mapping[tuple[str, str, str], int],
        end_time_map: Mapping[tuple[str, str, str], int],
        job_list: Sequence[str] | None = None,
        stage_list: Sequence[str] | None = None,
        machine_list_per_stage: Mapping[str, Sequence[str]] | None = None,
        all_job_list: Sequence[str] | None = None,
        highlight_op_set: set[tuple[str, str]] | None = None,
        force_start: int | None = None,
        force_end: int | None = None,
    ):
        self.plot_hybrid_flowshop(
            start_time_map,
            end_time_map,
            job_list=job_list,
            stage_list=stage_list,
            machine_list_per_stage=machine_list_per_stage,
            all_job_list=all_job_list,
            highlight_op_set=highlight_op_set,
            force_start=force_start,
            force_end=force_end,
        )
        plt.show()

    def export_hybrid_flowshop_plot(
        self,
        file_path: Path,
        start_time_map: Mapping[tuple[str, str, str], int],
        end_time_map: Mapping[tuple[str, str, str], int],
        job_list: Sequence[str] | None = None,
        stage_list: Sequence[str] | None = None,
        machine_list_per_stage: Mapping[str, Sequence[str]] | None = None,
        all_job_list: Sequence[str] | None = None,
        highlight_op_set: set[tuple[str, str]] | None = None,
        force_start: int | None = None,
        force_end: int | None = None,
    ):
        self.ax.clear()

        self.plot_hybrid_flowshop(
            start_time_map,
            end_time_map,
            job_list=job_list,
            stage_list=stage_list,
            machine_list_per_stage=machine_list_per_stage,
            all_job_list=all_job_list,
            highlight_op_set=highlight_op_set,
            force_start=force_start,
            force_end=force_end,
        )

        self.fig.savefig(file_path, bbox_inches="tight", dpi=300)
        logging.info(f"Gantt chart saved to {file_path}")

        plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def plot_hybrid_flowshop(
        self,
        start_time_map: Mapping[tuple[str, str, str], int],
        end_time_map: Mapping[tuple[str, str, str], int],
        job_list: Sequence[str] | None = None,
        stage_list: Sequence[str] | None = None,
        machine_list_per_stage: Mapping[str, Sequence[str]] | None = None,
        all_job_list: Sequence[str] | None = None,
        highlight_op_set: set[tuple[str, str]] | None = None,
        force_start: int | None = None,
        force_end: int | None = None,
    ):
        """
        Plot a Gantt chart for a Hybrid Flow Shop solution.

        Args:
            start_time_map (Mapping): (job, stage, machine) -> start time
            end_time_map (Mapping): (job, stage, machine) -> end time
            job_list (Sequence[str], optional): List of jobs to include
            stage_list (Sequence[str], optional): List of stages to include
            machine_list_per_stage (Mapping[str, Sequence[str]], optional): stage -> list of machines
            all_job_list (Sequence[str], optional): List of all jobs for color mapping
            highlight_op_set (set[tuple[str, str]], optional): Set of (job, stage) to highlight
            force_start (int, optional): If provided, forces the x-axis to start at this time
            force_end (int, optional): If provided, forces the x-axis to end at this time
        """
        self.set_x_horizon(
            start_time_map, end_time_map, force_start=force_start, force_end=force_end
        )

        # list of jobs, stages, & machines

        if job_list is None or len(job_list) == 0:
            _job_list = sorted({j for (j, _, _) in start_time_map.keys()})
        else:
            _job_list = job_list
        if stage_list is None or len(stage_list) == 0:
            _stage_list = sorted({i for (_, i, _) in start_time_map.keys()})
        else:
            _stage_list = stage_list

        _machine_list_per_stage: dict[str, Sequence[str]] = {
            stage: [] for stage in _stage_list
        }
        for stage in _stage_list:
            if machine_list_per_stage is None or not machine_list_per_stage.get(stage):
                _machine_list_per_stage[stage] = sorted(
                    {mc for (_, stg, mc) in start_time_map.keys() if stg == stage}
                )
            else:
                _machine_list_per_stage[stage] = machine_list_per_stage[stage]

        # Color map
        if all_job_list:
            job_to_color = self.create_job_to_color_map(all_job_list)
        else:
            job_to_color = self.create_job_to_color_map(_job_list)

        # Prepare machine lanes & labels
        machine_lanes, machine_labels = GanttPlotter.create_machine_lanes(
            start_time_map, _stage_list, _machine_list_per_stage
        )

        # Mapping machine to y-axis
        machine_to_y = {
            mc: self.machine_height * idx for idx, mc in enumerate(machine_lanes)
        }
        self.draw_operation_bars(
            start_time_map=start_time_map,
            end_time_map=end_time_map,
            job_to_color=job_to_color,
            machine_to_y=machine_to_y,
            job_list=_job_list,
            highlight_op_set=highlight_op_set,
        )

        # Axis formatting
        self.ax.set_yticks([y + 0.4 for y in range(len(machine_lanes))])
        self.ax.set_yticklabels(machine_labels)
        self.ax.set_ylim(
            -self.machine_height / 2,
            len(machine_lanes) + (self.bar_height - self.machine_height / 2),
        )
        self.ax.set_xlabel("Time")
        self.ax.set_title("Hybrid Flow Shop Schedule Gantt Chart")
        self.ax.grid(True, axis="x", linestyle="--", alpha=self.grid_alpha)
        self.ax.invert_yaxis()
        plt.tight_layout()

    @staticmethod
    def compute_horizon(
        start_time_map: Mapping[tuple[str, str, str], int],
        end_time_map: Mapping[tuple[str, str, str], int],
    ) -> tuple[int, int]:
        """
        Computes the (start, end) horizon of the schedule from start_time_map and end_time_map.

        Args:
            start_time_map (Mapping): (job, stage, machine) -> start time
            end_time_map (Mapping): (job, stage, machine) -> end time

        Returns:
            (int, int): (minimum start time, maximum end time)
        """  # noqa: E501
        if not start_time_map or not end_time_map:
            raise ValueError("start_time_map and end_time_map must not be empty.")

        min_start = min(start_time_map.values())
        max_end = max(end_time_map.values())

        return min_start, max_end

    def set_x_horizon(
        self,
        start_time_map: Mapping[tuple[str, str, str], int],
        end_time_map: Mapping[tuple[str, str, str], int],
        force_start: int | None = None,
        force_end: int | None = None,
    ):
        earliest_start, latest_completion = GanttPlotter.compute_horizon(
            start_time_map, end_time_map
        )
        if force_start is not None:
            earliest_start = force_start
        if force_end is not None:
            latest_completion = force_end
        self.ax.set_xlim(earliest_start, latest_completion + 1)

    def create_job_to_color_map(
        self, job_list: Sequence[str]
    ) -> dict[str, tuple[float, float, float, float]]:
        """
        Create a mapping from job name to color.

        Args:
            job_list (Sequence[str]): List of unique job names.
            cmap_name (str, optional): Name of the matplotlib colormap.

        Returns:
            dict[str, tuple]: A dictionary mapping each job to a color (RGBA tuple).
        """
        cmap = plt.get_cmap(self.cmap_name)
        n_jobs = max(len(job_list) - 1, 1)  # avoid division by zero
        return {job: cmap(i / n_jobs) for i, job in enumerate(job_list)}

    @staticmethod
    def create_machine_lanes(
        start_time_map: Mapping[tuple[str, str, str], int],
        stage_list: Sequence[str],
        machine_list_per_stage: Mapping[str, Sequence[str]],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """
        Create a list of (stage, machine) lanes and corresponding machine labels.

        Args:
            start_time_map (Mapping): (job, stage, machine) -> start time mapping.
            stage_list (Sequence[str]): List of stages to include.
            machine_list_per_stage (Mapping[str, Sequence[str]]): Mapping stage -> list of machines.

        Returns:
            tuple:
                - List of (stage, machine) tuples (machine_lanes)
                - List of machine labels (stage-machine)
        """  # noqa: E501
        machine_lanes = []
        machine_labels = []

        for stage in stage_list:
            machines = (
                machine_list_per_stage.get(stage) if machine_list_per_stage else None
            )
            if machines is None or len(machines) == 0:
                machines = sorted(
                    {mc for (_, stg, mc) in start_time_map.keys() if stg == stage}
                )
            for mc in machines:
                machine_lanes.append((stage, mc))
                machine_labels.append(f"{stage}-{mc}")

        return machine_lanes, machine_labels

    def draw_operation_bar(
        self,
        job: str,
        stage: str,
        machine: str,
        s_time: int,
        e_time: int,
        color: tuple[float, float, float, float],
        y: float,
        show_label: bool = True,
        show_duration: bool = True,
        highlight: bool = False,
    ):
        """
        Draw a single operation bar on the Gantt chart.

        Args:
            job (str): Job name.
            stage (str): Stage name.
            machine (str): Machine name.
            s_time (int): Start time.
            e_time (int): End time.
            color (tuple): RGBA color.
            y (float): Y-axis position.
            show_label (bool, optional): Whether to show the job label. Defaults to True.
            show_duration (bool, optional): Whether to show the duration. Defaults to True.
            highlight (bool, optional): Whether to highlight this operation.
                Highlighted operations have thicker edges. Defaults to False.
        """  # noqa: E501
        duration = e_time - s_time

        edgecolor = "black"
        linewidth = 3.0 if highlight else 1.0
        alpha = 1.0 if highlight else self.bar_alpha

        self.ax.add_patch(
            patches.Rectangle(
                (s_time, y),
                duration,
                self.bar_height,
                edgecolor=edgecolor,
                facecolor=color,
                alpha=alpha,
                linewidth=linewidth,
            )
        )
        if show_label:
            self.ax.text(
                (s_time + e_time) / 2,
                y + self.bar_height / 2,
                job,
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
        if show_duration:
            self.ax.text(
                (s_time + e_time) / 2,
                y + self.bar_height - 0.05,
                str(duration),
                ha="center",
                va="bottom",
                color="gray",
                fontsize=7,
            )

    def draw_operation_bars(
        self,
        start_time_map: Mapping[tuple[str, str, str], int],
        end_time_map: Mapping[tuple[str, str, str], int],
        job_to_color: Mapping[str, tuple[float, float, float, float]],
        machine_to_y: Mapping[tuple[str, str], float],
        job_list: Sequence[str],
        highlight_op_set: set[tuple[str, str]] | None = None,
    ):
        """Draw the operation bars and labels on the Gantt chart.

        Args:
            start_time_map (Mapping[tuple[str, str, str], int]): (job, stage, machine) -> start time
            end_time_map (Mapping[tuple[str, str, str], int]): (job, stage, machine) -> end time
            job_to_color (Mapping[str, tuple[float, float, float, float]]): job -> color
            machine_to_y (Mapping[tuple[str, str], float]): (stage, machine) -> y-axis position
            job_list (Sequence[str]): List of jobs to include
            highlight_op_set (set[tuple[str, str]] | None, optional): Set of (job, stage) to highlight.
                Defaults to None.
        """
        for (job, stage, machine), s_time in start_time_map.items():
            if job_list and job not in job_list:
                continue
            if (stage, machine) not in machine_to_y:
                continue

            e_time = end_time_map[(job, stage, machine)]
            y = machine_to_y[(stage, machine)]
            color = job_to_color[job]
            is_highlight = (
                highlight_op_set is not None
                and (job, stage) in highlight_op_set
            )

            self.draw_operation_bar(
                job=job,
                stage=stage,
                machine=machine,
                s_time=s_time,
                e_time=e_time,
                color=color,
                y=y,
                highlight=is_highlight,
            )
