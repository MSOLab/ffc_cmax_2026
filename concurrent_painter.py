import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mbls.cpsat import ObjValueBoundStore
from mbls.painter import ObjValueBoundPlotter
from routix.io.path import extract_prefix_from_filename

from hybridflowshop.io_solution import get_end_time_dict, get_start_time_dict
from hybridflowshop.painter.gantt import GanttPlotter


def draw_gantt_charts_from_solutions(
    working_dir: Path,
    solution_filename_format: str,
    all_job_id_list: list[str] | None = None,
    result_gantt_filename_format: str | None = None,
    encoding: str = "utf-8",
    painter_thread_cnt: int = 4,
):
    """
    Draws Gantt charts from solution files in the working directory (including subdirectories),
    using parallel processing for performance.

    Args:
        working_dir (Path): Root directory to search for solution files.
        solution_filename_format (str): Filename pattern with {} for prefix (e.g., "sol_{}.yaml").
        all_job_id_list (list[str] | None, optional): List of all job IDs in the instance.
            It is used as a reference to use consistent job color map across multiple Gantt charts.
            If unspecified, job IDs in the solution files will be used.
        result_gantt_filename_format (str | None, optional): Output chart filename pattern (e.g., "{}_gantt.png").
            If unspecified, "{}_gantt.png" will be used.
        encoding (str, optional): Encoding to use when reading the YAML solution files.
            Defaults to "utf-8".
        painter_thread_cnt (int, optional): Maximum number of threads to use for processing.
            Defaults to 4.
    """
    working_dir = Path(working_dir)
    files = list(working_dir.rglob(solution_filename_format.format("*")))
    if not files:
        logging.info(
            f"No solution files found in {working_dir} with pattern {solution_filename_format}"
        )
        return
    max_worker_cnt = min(painter_thread_cnt, len(files))

    _result_gantt_filename_format = result_gantt_filename_format or "{}_gantt.png"

    with ProcessPoolExecutor(max_workers=max_worker_cnt) as executor:
        futures = [
            executor.submit(
                _process_solution_file,
                file,
                solution_filename_format,
                all_job_id_list,
                _result_gantt_filename_format,
                encoding,
            )
            for file in files
        ]
        for future in futures:
            future.result()  # Optional: raise exception if any


def _process_solution_file(
    file_path: Path,
    solution_filename_format: str,
    all_job_id_list: list[str] | None,
    result_gantt_filename_format: str,
    encoding: str,
):
    file_dir = file_path.parent
    filename_prefix = extract_prefix_from_filename(
        solution_filename_format, file_path.name
    )
    if filename_prefix is None:
        raise ValueError(
            f"Could not extract filename prefix from {file_path.name}"
            f" using pattern {solution_filename_format}"
        )

    output_path = file_dir / result_gantt_filename_format.format(filename_prefix)

    start_time_map = get_start_time_dict(file_path, encoding=encoding)
    end_time_map = get_end_time_dict(file_path, encoding=encoding)
    GanttPlotter().export_hybrid_flowshop_plot(
        output_path, start_time_map, end_time_map, job_list=all_job_id_list
    )


def draw_progress_plots_from_logs(
    working_dir: Path,
    obj_log_filename_format: str,
    progress_plot_filename_format: str = "{}_progress.png",
    drop_first_values_percent: float = 0.0,
    encoding: str = "utf-8",
    painter_thread_cnt: int = 4,
):
    """
    Draws progress plots from objective log files in the working directory (including subdirectories),
    using parallel processing for performance.

    Args:
        working_dir (str | Path): Root directory to search for objective log files.
        obj_log_filename_format (str): Filename pattern with {} for prefix (e.g., "log_{}.yaml").
        progress_plot_filename_format (str): Output filename pattern (e.g., "{}_progress.png").
        drop_first_values_percent (float): Drop initial portion of the objective log values.
        encoding (str): Encoding to use when reading YAML files.
        painter_thread_cnt (int): Maximum number of threads to use for processing.
    """
    working_dir = Path(working_dir)
    files = list(working_dir.rglob(obj_log_filename_format.format("*")))
    max_worker_cnt = min(painter_thread_cnt, len(files))

    with ProcessPoolExecutor(max_workers=max_worker_cnt) as executor:
        futures = [
            executor.submit(
                _process_progress_log_file,
                file,
                obj_log_filename_format,
                progress_plot_filename_format,
                drop_first_values_percent,
                encoding,
            )
            for file in files
        ]
        for future in futures:
            future.result()


def _process_progress_log_file(
    file_path: Path,
    obj_log_filename_format: str,
    progress_plot_filename_format: str,
    drop_first_values_percent: float,
    encoding: str,
):
    file_dir = file_path.parent
    filename_prefix = extract_prefix_from_filename(
        obj_log_filename_format, file_path.name
    )
    if filename_prefix is None:
        raise ValueError(
            f"Could not extract filename prefix from {file_path.name}"
            f" using pattern {obj_log_filename_format}"
        )

    output_path = file_dir / progress_plot_filename_format.format(filename_prefix)

    obj_store = ObjValueBoundStore.load_yaml(file_path, encoding=encoding)
    ObjValueBoundPlotter.plot(
        obj_store,
        output_path,
        drop_first_values_percent=drop_first_values_percent,
        label_y_offset=2.0,
        legend_loc="lower right",
        show_markers=False,
    )
