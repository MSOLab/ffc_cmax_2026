from pathlib import Path
from typing import Any, Optional

from mbls import SolverOutputSummary, SolverStatus

from .painter import GanttPlotter
from .pure_cp_2023_naderi import PureCP2023Naderi
from .utils import tuple_to_pyyaml_key


class SolutionManager:
    """
    Manages the incumbent solution obtained from the CP model,
    including summary reporting and visualization.
    """

    def __init__(
        self,
        start_times: dict[tuple[str, str, str], int],
        end_times: dict[tuple[str, str, str], int],
        summary: SolverOutputSummary,
    ):
        self.start_times = start_times
        self.end_times = end_times
        self.summary = summary
        self.is_feasible = SolverStatus.found_feasible_solution(summary.status)
        """
        Indicates whether the solution is feasible based on the solver status.
        True if the status is FEASIBLE or OPTIMAL, False otherwise.
        """

    def get_result_summary(self) -> SolverOutputSummary:
        """
        Returns the CP model's result summary.

        Returns:
            SolverOutputSummary: The summary object
        """
        return self.summary

    def get_obj_value(self) -> Optional[float]:
        return self.summary.objective_value

    def get_obj_bound(self) -> Optional[float]:
        return self.summary.best_objective_bound

    def apply_start_and_present_hints_to(self, target_model: PureCP2023Naderi) -> None:
        """
        Apply current incumbent solution as initial variable hints to another CP model.

        Args:
            target_model (PureCP2023Naderi): The target CP model to receive hints
        """
        target_model.clear_hints()
        target_model.add_start_and_present_hints_from_start_times(self.start_times)

    @staticmethod
    def get_time_dict_pyyaml(
        time_dict: dict[tuple[str, str, str], int],
    ) -> dict[str, int]:
        """
        Convert a time dictionary to a format suitable for PyYAML serialization.

        Args:
            time_dict (dict[tuple[str, str, str], int]): The time dictionary to convert

        Returns:
            dict[str, int]: !!python/tuple [left, center, right] -> time
        """
        return tuple_to_pyyaml_key(time_dict)

    def get_solution_dict(self, for_pyyaml: bool = False) -> dict[str, Any]:
        """
        Convert the incumbent solution to a dictionary format.

        Args:
            for_pyyaml (bool, optional): If true, create start time and end time dictionary for PyYAML.
                Defaults to False.

        Returns:
            dict[str, Any]: A dictionary representation of the incumbent solution
        """
        start_times: dict[Any, int]
        end_times: dict[Any, int]
        if for_pyyaml:
            start_times = self.get_time_dict_pyyaml(self.start_times)
            end_times = self.get_time_dict_pyyaml(self.end_times)
        else:
            start_times = self.start_times
            end_times = self.end_times
        return {
            "start_times": start_times,
            "end_times": end_times,
        }

    def save_gantt_as_png(self, output_path: Path) -> None:
        """
        Save the current incumbent solution as a Gantt chart image.

        Args:
            filename (str): Filename to save the Gantt chart (relative to output_dir)
            figsize (tuple): Size of the matplotlib figure
        """
        plotter = GanttPlotter()
        plotter.export_hybrid_flowshop_plot(
            output_path, self.start_times, self.end_times
        )
