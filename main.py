import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from routix import (
    DynamicDataObject,
    ElapsedTimer,
    StoppingCriteria,
    SubroutineFlowValidator,
)
from routix.io import init_timestamped_working_dir, object_to_yaml
from routix.type_defs import RunMode
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hfs_config import MainMetadata
from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hfs_multi_scenario_runner import HfsMultiScenarioRunner
from hfs_single_instance_runner import HfsSingleInstanceRunner
from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController

MAIN_METADATA_FILENAME = "main_metadata.yaml"


def main():
    e_timer = ElapsedTimer()

    # --- Load and validate metadata ---
    try:
        raw_metadata = read_yaml(Path(MAIN_METADATA_FILENAME))
        config = MainMetadata.model_validate(raw_metadata)
    except FileNotFoundError:
        logging.error(f"Metadata file not found at '{MAIN_METADATA_FILENAME}'")
        return
    except ValidationError as e:
        logging.error(f"Metadata validation failed:\n{e}")
        return

    # --- Determine RunMode and base_output_dir_path ---
    run_mode = RunMode.FULL_RUN
    if config.analysis_timestamp:
        potential_path = config.output_dir_scenarios / config.analysis_timestamp
        if potential_path.is_dir():
            run_mode = RunMode.POST_PROCESS_ONLY
            base_output_dir_path = potential_path
            e_timer.set_start_dt_from_dir_name(config.analysis_timestamp)
        else:
            base_output_dir_path = init_timestamped_working_dir(
                base_output_dir=config.output_dir_scenarios, e_timer=e_timer
            )
    else:
        base_output_dir_path = init_timestamped_working_dir(
            base_output_dir=config.output_dir_scenarios, e_timer=e_timer
        )

    # --- Setup logging ---
    log_handlers = add_file_handler(base_output_dir_path / config.scenario_log_filename)

    logging.info(f"Base output directory is: {base_output_dir_path}")
    if run_mode is RunMode.POST_PROCESS_ONLY:
        logging.info(
            "Found valid timestamp. "
            f"Running in POST_PROCESS_ONLY mode for: {config.analysis_timestamp}"
        )
    else:
        if config.analysis_timestamp:
            logging.warning(
                f"Timestamp '{config.analysis_timestamp}' provided, "
                f"but directory not found at '{base_output_dir_path}'. "
                "Proceeding with a new FULL_RUN."
            )
        else:
            logging.info("Running in FULL_RUN mode.")

    # --- Load data common to all scenarios ---
    pra_common_params_dict = read_yaml(config.pra_common_params_rel_path)

    # Main metadata & common parameters handling
    # - If run_mode is full run, dump the metadata and common parameters
    # - If post-processing-only, load from the dumped files
    main_metadata_dump_path = base_output_dir_path / MAIN_METADATA_FILENAME
    pra_common_params_dump_path = (
        base_output_dir_path / config.pra_common_params_rel_path.name
    )
    # if run_mode is RunMode.FULL_RUN:
    if run_mode == RunMode.FULL_RUN:
        object_to_yaml(config.to_dict(), main_metadata_dump_path)
        object_to_yaml(pra_common_params_dict, pra_common_params_dump_path)
    elif run_mode is RunMode.POST_PROCESS_ONLY:
        if not main_metadata_dump_path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found at '{main_metadata_dump_path}'"
            )
        config = MainMetadata.model_validate(read_yaml(main_metadata_dump_path))
        # Set config.analysis_timestamp to the one from metadata
        config.analysis_timestamp = e_timer.get_start_dt_for_dir_name()

        if not pra_common_params_dump_path.is_file():
            raise FileNotFoundError(
                f"Common parameters file not found at '{pra_common_params_dump_path}'"
            )
        pra_common_params_dict = read_yaml(pra_common_params_dump_path)

    benchmark_filenames = config.get_benchmark_filename_list()
    instances = load_list_of_instances(config.input_dir, benchmark_filenames)

    # --- Prepare scenario configurations ---
    scenario_configs = []
    for path_config in config.dicts_of_i_o_data_path:
        subroutine_flow_obj = read_yaml(path_config.subroutine_flow_rel_path)
        stopping_criteria_dict = read_yaml(path_config.stopping_criteria_rel_path)

        if run_mode == RunMode.FULL_RUN:
            validator = SubroutineFlowValidator(HybridFlowShopCpLnsController)
            validator.validate(DynamicDataObject.from_obj(subroutine_flow_obj))

        scenario_configs.append(
            {
                "subroutine_flow": DynamicDataObject.from_obj(subroutine_flow_obj),
                "stopping_criteria": StoppingCriteria(stopping_criteria_dict),
                "output_subdir": path_config.output_dir,
                "description": path_config.description,
            }
        )

    # --- Base output metadata ---
    base_output_metadata = config.model_dump(
        include={
            "result_dir_name",
            "draw_gantt",
            "painter_thread_cnt",
            "result_gantt_filename_format",
            "draw_progress_plot",
            "progress_plot_filename_format",
            "drop_first_values_percent",
        }
    )
    base_output_metadata["start_dt"] = e_timer.start_dt

    # --- Create and run the multi-scenario runner ---
    multi_scenario_runner = HfsMultiScenarioRunner(
        m_i_runner_class=HfsMultiInstanceRunner,
        s_i_runner_class=HfsSingleInstanceRunner,
        instances=instances,
        shared_param_dict=pra_common_params_dict,
        scenario_configs=scenario_configs,
        output_dir=base_output_dir_path,
        base_output_metadata=base_output_metadata,
        mode=run_mode,
        instance_worker_cnt=config.instance_worker_cnt,
    )
    multi_scenario_runner.set_baseline_df(
        config.baseline_csv_path, config.baseline_column_mapping
    )
    logging.info("Starting HFS Multi-Scenario Runner.")
    multi_scenario_runner.run()

    logging.info(
        "Finished HFS Multi-Scenario Runner. "
        f"Total elapsed time: {e_timer.get_formatted_elapsed_time()} seconds."
    )
    release_log_handlers(log_handlers)


# Helper methods


def read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Error reading YAML from {path}: {e}")


def load_hfs_instance(file_path: Path) -> HybridFlowshopParameters:
    try:
        ins_name = file_path.stem
        with open(file_path, "r") as f:
            return HybridFlowshopParameters.from_pra_data(ins_name, f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading benchmark file {file_path}: {e}")


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
) -> list[HybridFlowshopParameters]:
    """Load a list of hybrid flow shop problem instances from the specified directory.

    Args:
        input_dir_path (Path): Path to the directory containing benchmark files.
        benchmark_filenames (list[str]): List of benchmark filenames to load.

    Returns:
        list[HybridFlowshopParameters]: List of loaded hybrid flow shop problem instances.
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
