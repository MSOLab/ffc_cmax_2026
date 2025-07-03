from pathlib import Path
from typing import Any, Generic

from routix.report import SubroutineReportT

from .painter import GanttPlotter
from .pure_cp_2023_naderi import PureCP2023Naderi
from .utils import tuple_to_pyyaml_key


# TODO: remove
class SolutionManager(Generic[SubroutineReportT]):
    """
    Manages the incumbent solution obtained from the CP model,
    including summary reporting and visualization.
    """

    def __init__(
        self,
        start_time_map: dict[tuple[str, str, str], int],
        end_time_map: dict[tuple[str, str, str], int],
        report: SubroutineReportT,
    ):
        self.start_time_map = start_time_map
        self.end_time_map = end_time_map
        self.report = report
        self.is_feasible = report.obj_value is not None
        """Indicates whether the solution is feasible based on the report's objective value."""

    def apply_start_and_present_hints(
        self, target_model: PureCP2023Naderi, ignore_integrity_check: bool = True
    ) -> None:
        # TODO: move to PureCP2023Naderi
        """
        Apply current incumbent solution as initial variable hints to another CP model.

        Args:
            target_model (PureCP2023Naderi): The target CP model to receive hints
            ignore_integrity_check (bool, optional): If true, skip integrity checks.
                Defaults to True.
        """
        target_model.add_start_and_present_hints_from_start_time_map(
            self.start_time_map, ignore_integrity_check=ignore_integrity_check
        )

    def apply_fixed_machine_and_ops_precedence_constraints(
        self, target_model: PureCP2023Naderi, ignore_integrity_check: bool = True
    ) -> None:
        # TODO: move to PureCP2023Naderi
        """
        Add fixed constraints based on the incumbent solution to another CP model.

        Args:
            target_model (PureCP2023Naderi): The target CP model to receive fixed constraints
            ignore_integrity_check (bool, optional): If true, skip integrity checks.
                Defaults to True.
        """
        target_model.add_fixed_machine_and_ops_precedence_constraints_from_start_time_map(
            self.start_time_map, ignore_integrity_check=ignore_integrity_check
        )

    @staticmethod
    def get_time_dict_pyyaml(
        time_dict: dict[tuple[str, str, str], int],
    ) -> dict[str, int]:
        # TODO: move to HybridFlowshopSchedule
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
        # TODO: move to HybridFlowshopSchedule
        start_time_map: dict[Any, int]
        end_time_map: dict[Any, int]
        if for_pyyaml:
            start_time_map = self.get_time_dict_pyyaml(self.start_time_map)
            end_time_map = self.get_time_dict_pyyaml(self.end_time_map)
        else:
            start_time_map = self.start_time_map
            end_time_map = self.end_time_map
        return {
            "start_times": start_time_map,  # TODO: backward compatibility; change to "start_time_map" in future versions
            "end_times": end_time_map,  # TODO: backward compatibility; change to "end_time_map" in future versions
        }

    def save_gantt_as_png(self, output_path: Path) -> None:
        """
        Save the current incumbent solution as a Gantt chart image.

        Args:
            filename (str): Filename to save the Gantt chart (relative to output_dir)
            figsize (tuple): Size of the matplotlib figure
        """
        # TODO: move to HybridFlowshopSchedule
        plotter = GanttPlotter()
        plotter.export_hybrid_flowshop_plot(
            output_path, self.start_time_map, self.end_time_map
        )
