import logging
from pathlib import Path
from typing import Any

import yaml
from mbls import DynamicDataObject, ElapsedTimer, SubroutineFlowValidator
from schore.hybridflowshop import HybridFlowShopProblem

from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.stopping_criteria import StoppingCriteria

MAIN_METADATA_FILENAME = "main_metadata.yaml"
WORKER_CNT = 1


def main():
    # Read the main metadata file
    main_metadata = read_yaml(Path(MAIN_METADATA_FILENAME))
    for i_o_data_path_dict in main_metadata.get("dicts_of_i_o_data_path"):
        main_metadata["stopping_criteria_rel_path"] = i_o_data_path_dict[
            "stopping_criteria_rel_path"
        ]
        main_metadata["subroutine_flow_rel_path"] = i_o_data_path_dict[
            "subroutine_flow_rel_path"
        ]
        main_metadata["output_dir"] = i_o_data_path_dict["output_dir"]
        run_hfs_instance_set_runner(main_metadata)


def run_hfs_instance_set_runner(main_metadata_dict: dict[str, Any]) -> None:
    e_timer = ElapsedTimer()

    single_instance_skip_run_do_post_process = main_metadata_dict.get(
        "single_instance_skip_run_do_post_process", False
    )
    single_instance_from_files_save_analysis_only = main_metadata_dict.get(
        "single_instance_from_files_save_analysis_only", False
    )
    if single_instance_skip_run_do_post_process:
        if "analysis_timestamp" in main_metadata_dict:
            e_timer.set_start_dt_from_dir_name(main_metadata_dict["analysis_timestamp"])
        else:
            raise ValueError(
                "single_instance_skip_run_do_post_process is True, "
                "but 'analysis_timestamp' is not provided in main_metadata_dict."
            )
    if single_instance_from_files_save_analysis_only:
        single_instance_skip_run_do_post_process = True
        if "analysis_timestamp" in main_metadata_dict:
            e_timer.set_start_dt_from_dir_name(main_metadata_dict["analysis_timestamp"])
        else:
            raise ValueError(
                "single_instance_from_files_save_analysis_only is True, "
                "but 'analysis_timestamp' is not provided in main_metadata_dict."
            )

    # Read common parameters for PRA benchmarks
    pra_common_params_rel_path = Path(main_metadata_dict["pra_common_params_rel_path"])
    pra_common_params_dict = read_yaml(pra_common_params_rel_path)

    # Read subroutine flow and stopping criteria
    subroutine_flow_rel_path = Path(main_metadata_dict["subroutine_flow_rel_path"])
    subroutine_flow_obj = read_yaml(subroutine_flow_rel_path)
    stopping_criteria_rel_path = Path(main_metadata_dict["stopping_criteria_rel_path"])
    stopping_criteria_dict = read_yaml(stopping_criteria_rel_path)

    # Input parameters
    first = int(main_metadata_dict["first"])
    last = int(main_metadata_dict["last"])
    input_dir_path = Path(main_metadata_dict["input_dir"])
    benchmark_filename_format = str(main_metadata_dict["benchmark_filename_format"])
    benchmark_filenames = [
        benchmark_filename_format.format(i) for i in range(first, last + 1)
    ]

    # Initialize working directory
    output_dir = Path(main_metadata_dict["output_dir"])
    working_dir_path = init_working_dir(output_dir, e_timer)

    log_handlers = add_file_handler(working_dir_path / "hfs_instance_set_runner.log")

    # Subroutine controller arguments
    subroutine_flow = DynamicDataObject.from_obj(subroutine_flow_obj)
    stopping_criteria = StoppingCriteria(stopping_criteria_dict)

    # Validate the subroutine flow
    validator = SubroutineFlowValidator(HybridFlowShopCpLnsController)
    validator.validate(subroutine_flow)

    # Save main metadata, subroutine flow, and stopping criteria
    algorithm_data_dir = "algorithm_data"
    algorithm_data_dir_path = working_dir_path / algorithm_data_dir
    algorithm_data_dir_path.mkdir(parents=True, exist_ok=True)

    main_metadata_filename = algorithm_data_dir_path / MAIN_METADATA_FILENAME
    with open(main_metadata_filename, "w") as f:
        yaml.safe_dump(main_metadata_dict, f, default_flow_style=False)
    DynamicDataObject.safe_save_yaml(
        subroutine_flow, algorithm_data_dir_path / subroutine_flow_rel_path
    )
    stopping_criteria.to_yaml(algorithm_data_dir_path / stopping_criteria_rel_path)

    # Output metadata
    result_dir_name = str(main_metadata_dict["result_dir_name"])
    output_metadata = {
        "start_dt": e_timer.start_dt,
        "result_dir_name": result_dir_name,
        "single_instance_skip_run_do_post_process": single_instance_skip_run_do_post_process,
        "single_instance_from_files_save_analysis_only": single_instance_from_files_save_analysis_only,
    }
    draw_gantt = main_metadata_dict.get("draw_gantt", False)
    output_metadata["draw_gantt"] = draw_gantt
    if draw_gantt:
        result_gantt_filename_format = str(
            main_metadata_dict["result_gantt_filename_format"]
        )
        output_metadata["gantt_filename_format"] = result_gantt_filename_format
    draw_progress_plot = main_metadata_dict.get("draw_progress_plot", False)
    output_metadata["draw_progress_plot"] = draw_progress_plot
    if draw_progress_plot:
        progress_plot_filename_format = str(
            main_metadata_dict["progress_plot_filename_format"]
        )
        output_metadata["progress_plot_filename_format"] = progress_plot_filename_format

    # Load problem instances
    instances = load_list_of_instances(input_dir_path, benchmark_filenames)

    # Create and run the multi instance runner
    hfs_instance_set_runner = HfsMultiInstanceRunner(
        s_i_runner_class=HfsSingleInstanceRunner,
        instances=instances,
        shared_params=pra_common_params_dict,
        subroutine_flow=subroutine_flow,
        stopping_criteria=stopping_criteria,
        output_dir=working_dir_path,
        output_metadata=output_metadata,
    )
    # Default is 2; if set to 1, it will run sequentially
    hfs_instance_set_runner.set_max_workers(WORKER_CNT)
    hfs_instance_set_runner.run()

    # Print elapsed time
    logging.info(f"Elapsed time: {e_timer.get_formatted_elapsed_time()} seconds")
    release_log_handlers(log_handlers)


# Helper methods


def read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Error reading YAML from {path}: {e}")


def load_hfs_instance(file_path: Path) -> HybridFlowShopProblem:
    try:
        ins_name = file_path.stem
        with open(file_path, "r") as f:
            return HybridFlowShopProblem.from_pra_data(ins_name, f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading benchmark file {file_path}: {e}")


def init_working_dir(output_dir: Path, e_timer: ElapsedTimer) -> Path:
    """
    Prepare the output directory for the instance run.
    """
    working_dir = output_dir / e_timer.get_start_dt_for_dir_name()
    working_dir.mkdir(parents=True, exist_ok=True)
    return working_dir


def add_file_handler(
    log_path: Path,
    level=logging.INFO,
    fmt="%(asctime)s - %(levelname)s - %(message)s",
) -> list[logging.Handler]:
    """
    Set the dual handler for logging.
    This function configures the logging to handle both console and file outputs.
    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(
            log_path
        ):
            return []

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return [file_handler]


def release_log_handlers(handlers: list[logging.Handler]) -> None:
    """
    Reset the log handlers to avoid duplicate logs.
    This function clears all existing log handlers.
    """
    logger = logging.getLogger()
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def load_list_of_instances(
    input_dir_path: Path, benchmark_filenames: list[str]
) -> list[HybridFlowShopProblem]:
    """Load a list of hybrid flow shop problem instances from the specified directory.

    Args:
        input_dir_path (Path): Path to the directory containing benchmark files.
        benchmark_filenames (list[str]): List of benchmark filenames to load.

    Returns:
        list[HybridFlowShopProblem]: List of loaded hybrid flow shop problem instances.
    """
    instances = []
    for benchmark_filename in benchmark_filenames:
        input_file_path = input_dir_path / benchmark_filename
        instances.append(load_hfs_instance(input_file_path))
    return instances


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
