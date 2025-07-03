import logging
from pathlib import Path

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
    bar_alpha = 0.7
    grid_alpha = 0.5
    figsize = (12, 8)

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def display_hybrid_flowshop_plot(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        end_time_map: dict[tuple[str, str, str], int],
        job_list: list[str] = [],
        stage_list: list[str] = [],
        machine_list_per_stage: dict[str, list[str]] = {},
    ):
        self.plot_hybrid_flowshop(
            start_time_map,
            end_time_map,
            job_list=job_list,
            stage_list=stage_list,
            machine_list_per_stage=machine_list_per_stage,
        )
        plt.show()

    def export_hybrid_flowshop_plot(
        self,
        file_path: Path,
        start_time_map: dict[tuple[str, str, str], int],
        end_time_map: dict[tuple[str, str, str], int],
        job_list: list[str] = [],
        stage_list: list[str] = [],
        machine_list_per_stage: dict[str, list[str]] = {},
    ):
        self.plot_hybrid_flowshop(
            start_time_map,
            end_time_map,
            job_list=job_list,
            stage_list=stage_list,
            machine_list_per_stage=machine_list_per_stage,
        )
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        logging.info(f"Gantt chart saved to {file_path}")
        plt.close()

    def plot_hybrid_flowshop(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        end_time_map: dict[tuple[str, str, str], int],
        job_list: list[str] = [],
        stage_list: list[str] = [],
        machine_list_per_stage: dict[str, list[str]] = {},
    ):
        """
        Plot a Gantt chart for a Hybrid Flow Shop solution.

        Args:
            start_time_map (dict): (job, stage, machine) -> start time
            end_time_map (dict): (job, stage, machine) -> end time
            job_list (list, optional): List of jobs to include
            stage_list (list, optional): List of stages to include
            machine_list_per_stage (dict, optional): stage -> list of machines
        """
        self.set_x_horizon(start_time_map, end_time_map)

        # list of jobs, stages, & machines

        if len(job_list) == 0:
            _job_list = sorted({j for (j, _, _) in start_time_map.keys()})
        else:
            _job_list = job_list.copy()
        if len(stage_list) == 0:
            _stage_list = sorted({i for (_, i, _) in start_time_map.keys()})
        else:
            _stage_list = stage_list.copy()

        _machine_list_per_stage: dict[str, list[str]] = {
            stage: [] for stage in _stage_list
        }
        for stage in stage_list:
            if not machine_list_per_stage.get(stage):
                _machine_list_per_stage[stage] = sorted(
                    {mc for (_, stg, mc) in start_time_map.keys() if stg == stage}
                )
            else:
                _machine_list_per_stage[stage] = machine_list_per_stage[stage].copy()

        # Color map
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
        start_time_map: dict[tuple[str, str, str], int],
        end_time_map: dict[tuple[str, str, str], int],
    ) -> tuple[int, int]:
        """
        Computes the (start, end) horizon of the schedule from start_time_map and end_time_map.

        Args:
            start_time_map (dict): (job, stage, machine) -> start time
            end_time_map (dict): (job, stage, machine) -> end time

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
        start_time_map: dict[tuple[str, str, str], int],
        end_time_map: dict[tuple[str, str, str], int],
    ):
        earliest_start, latest_completion = GanttPlotter.compute_horizon(
            start_time_map, end_time_map
        )
        self.ax.set_xlim(earliest_start, latest_completion + 1)

    def create_job_to_color_map(
        self, job_list: list[str]
    ) -> dict[str, tuple[float, float, float, float]]:
        """
        Create a mapping from job name to color.

        Args:
            job_list (list[str]): List of unique job names.
            cmap_name (str, optional): Name of the matplotlib colormap.

        Returns:
            dict[str, tuple]: A dictionary mapping each job to a color (RGBA tuple).
        """
        cmap = plt.get_cmap(self.cmap_name)
        n_jobs = max(len(job_list) - 1, 1)  # avoid division by zero
        return {job: cmap(i / n_jobs) for i, job in enumerate(job_list)}

    @staticmethod
    def create_machine_lanes(
        start_time_map: dict[tuple[str, str, str], int],
        stage_list: list[str],
        machine_list_per_stage: dict[str, list[str]],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """
        Create a list of (stage, machine) lanes and corresponding machine labels.

        Args:
            start_time_map (dict): (job, stage, machine) -> start time dictionary.
            stage_list (list[str]): List of stages to include.
            machine_list_per_stage (dict[str, list[str]]): Mapping stage -> list of machines.

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
            show_label (bool, optional): Whether to show the job label. Default is True.
            show_duration (bool, optional): Whether to show the duration. Default is True.
        """  # noqa: E501
        duration = e_time - s_time

        self.ax.add_patch(
            patches.Rectangle(
                (s_time, y),
                duration,
                self.bar_height,
                edgecolor="black",
                facecolor=color,
                alpha=self.bar_alpha,
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
        start_time_map: dict[tuple[str, str, str], int],
        end_time_map: dict[tuple[str, str, str], int],
        job_to_color: dict[str, tuple[float, float, float, float]],
        machine_to_y: dict[tuple[str, str], float],
        job_list: list[str],
    ):
        """Draw the operation bars and labels on the Gantt chart."""
        for (job, stage, machine), s_time in start_time_map.items():
            if job_list and job not in job_list:
                continue
            if (stage, machine) not in machine_to_y:
                continue

            e_time = end_time_map[(job, stage, machine)]
            y = machine_to_y[(stage, machine)]
            color = job_to_color[job]

            self.draw_operation_bar(
                job=job,
                stage=stage,
                machine=machine,
                s_time=s_time,
                e_time=e_time,
                color=color,
                y=y,
            )
