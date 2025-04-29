from pathlib import Path

from clad.solver_output_summary import SolverOutputSummary
from pure_cp_2023_naderi import PureCP2023Naderi


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

    def get_result_summary(self) -> SolverOutputSummary:
        """
        Returns the CP model's result summary.

        Returns:
            SolverOutputSummary: The summary object
        """
        return self.summary

    def get_objective_value(self) -> float:
        return self.summary.objective_value

    def apply_hint_to(self, target_model: PureCP2023Naderi) -> None:
        """
        Apply current incumbent solution as initial variable hints to another CP model.

        Args:
            target_model (PureCP2023Naderi): The target CP model to receive hints
        """
        target_model.clear_hints()
        for (j, i, k), s_time in self.start_times.items():
            target_model.add_hint(target_model.var_op_start[j][i][k], s_time)
            target_model.add_hint(target_model.var_op_is_present[j][i][k], 1)

    def save_gantt_as_png(self, filename: str, output_dir: Path) -> None:
        """
        Save the current incumbent solution as a Gantt chart image.

        Args:
            filename (str): Filename to save the Gantt chart (relative to output_dir)
            figsize (tuple): Size of the matplotlib figure
        """
        from plotter import GanttPlotter

        plotter = GanttPlotter()
        plotter.export_hybrid_flowshop_plot(
            output_dir / filename, self.start_times, self.end_times
        )
