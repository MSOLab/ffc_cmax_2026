import argparse
import logging
from pathlib import Path
from typing import Any

import yaml
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
from hybridflowshop.controller import HybridFlowShopCpLnsController
from output_filenames import OutputFilenames

MAIN_METADATA_FILENAME = "main_metadata.yaml"
REVERSE_INSTANCE_ORDER = True


def run_experiment(
    e_timer: ElapsedTimer,
    config: MainMetadata,
    run_mode: RunMode,
    base_output_dir_path: Path,
    prev_flow: DynamicDataObject | None = None,
    resume_dir: Path | None = None,
) -> None:
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
            logging.info(f"Running in {run_mode.name} mode.")

    try:
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
        if run_mode in {RunMode.FULL_RUN, RunMode.RESUME}:
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

        benchmark_filenames = config.get_benchmark_filename_list(
            reversed=REVERSE_INSTANCE_ORDER
        )
        instances = load_list_of_instances(config.input_dir, benchmark_filenames)

        # --- Prepare scenario configurations ---
        scenario_configs = []
        for path_config in config.dicts_of_i_o_data_path:
            subroutine_flow_obj = read_yaml(path_config.subroutine_flow_rel_path)
            stopping_criteria_dict = read_yaml(path_config.stopping_criteria_rel_path)

            # Validate the flow if FULL_RUN or RESUME
            validator = SubroutineFlowValidator(HybridFlowShopCpLnsController)
            if run_mode in {RunMode.FULL_RUN, RunMode.RESUME}:
                try:
                    validator.validate(DynamicDataObject.from_obj(subroutine_flow_obj))
                    logging.info(
                        f"Subroutine flow validated for scenario {path_config.output_dir}."
                    )
                except Exception as e:
                    logging.error(
                        f"Subroutine flow validation failed for scenario {path_config.output_dir}: {e}",
                        exc_info=True,
                    )
                    return
            # Validate prefix against previous flow in RESUME mode
            flow_resume_idx = -1
            if run_mode == RunMode.RESUME:
                try:
                    flow_resume_idx = validator.validate_subroutine_flow_prefix(
                        DynamicDataObject.from_obj(prev_flow),
                        DynamicDataObject.from_obj(subroutine_flow_obj),
                    )
                    logging.info(
                        f"Resume prefix validated for scenario {path_config.output_dir}; flow resume index={flow_resume_idx}"
                    )
                except Exception as e:
                    logging.error(
                        f"Resume validation failed for scenario {path_config.output_dir}: {e}",
                        exc_info=True,
                    )
                    return
            scenario_config_dict = {
                "subroutine_flow": DynamicDataObject.from_obj(subroutine_flow_obj),
                "stopping_criteria": StoppingCriteria(stopping_criteria_dict),
                "output_subdir": path_config.output_dir,
                "description": path_config.description,
            }
            if run_mode == RunMode.RESUME:
                # In RESUME mode, also provide flow_resume_idx
                scenario_config_dict["flow_resume_idx"] = flow_resume_idx
            scenario_configs.append(scenario_config_dict)

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
        # Provide filename formats
        base_output_metadata["summary_fn_format"] = OutputFilenames.SUMMARY_FN_FORMAT
        base_output_metadata["solution_fn_format"] = OutputFilenames.SOLUTION_FN_FORMAT
        base_output_metadata["obj_log_fn_format"] = OutputFilenames.OBJ_LOG_FN_FORMAT
        # If running in RESUME mode, include resume info for runners to locate previous artifacts
        if run_mode == RunMode.RESUME:
            # Provide resume_root and resume_timestamp so runners can find per-instance files
            base_output_metadata["resume_root"] = str(resume_dir)
            # Try to extract timestamp from resume_dir name if possible
            if resume_dir is not None:
                base_output_metadata["resume_timestamp"] = resume_dir.name
            else:
                base_output_metadata["resume_timestamp"] = None
        # ask runners to be strict by default when resuming
        base_output_metadata["resume_strict"] = True

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

    finally:
        logging.info(
            "Finished HFS Multi-Scenario Runner. "
            f"Total elapsed time: {e_timer.get_formatted_elapsed_time()} seconds."
        )


# Helper methods


def read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Error reading YAML from {path}: {e}")


def load_hfs_instance(
    file_path: Path, is_ff2020_format: bool = False
) -> HybridFlowshopParameters:
    try:
        ins_name = file_path.stem
        with open(file_path, "r") as f:
            if is_ff2020_format:
                return HybridFlowshopParameters.from_ff2020_data(ins_name, f)
            else:
                return HybridFlowshopParameters.from_pra_data(ins_name, f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading benchmark file {file_path}: {e}")


def determine_run_mode_and_base_dir(
    config: MainMetadata, e_timer: ElapsedTimer
) -> tuple[RunMode, Path, DynamicDataObject | None, Path | None]:
    """Determine run mode and base output directory.

    Args:
        config (MainMetadata): The main metadata configuration.
        e_timer (ElapsedTimer): The elapsed timer for tracking execution time.

    Raises:
        FileNotFoundError: If the specified resume or analysis directory does not exist.

    Returns:
        tuple[RunMode, Path, DynamicDataObject | None, Path | None]: prev_flow and resume_dir are None
            unless RESUME mode is selected.
    """
    run_mode = RunMode.FULL_RUN
    prev_flow = None
    _target_path = None

    def new_ts_dir():
        return init_timestamped_working_dir(
            base_output_dir=config.output_dir_scenarios, e_timer=e_timer
        )

    _target_path = config.get_analysis_dir_path()
    _timestamp = config.analysis_timestamp
    if _target_path:
        _timestamp = _target_path.name
        if not _target_path.exists() or not _target_path.is_dir():
            raise FileNotFoundError(f"Analysis directory not found: {_target_path}")
        run_mode = RunMode.POST_PROCESS_ONLY
        base_output = _target_path
        e_timer.set_start_dt_from_dir_name(_timestamp)
        config.analysis_timestamp = _timestamp
    elif _timestamp:
        _target_path = config.output_dir_scenarios / _timestamp
        if not _target_path.exists() or not _target_path.is_dir():
            raise FileNotFoundError(f"Analysis directory not found: {_target_path}")
        run_mode = RunMode.POST_PROCESS_ONLY
        base_output = _target_path
        e_timer.set_start_dt_from_dir_name(_timestamp)
    elif config.resume_dir_path:
        _target_path = Path(config.resume_dir_path)
        if not _target_path.exists() or not _target_path.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {_target_path}")
        # attempt to load prev_flow; propagate errors to caller
        prev_flow = DynamicDataObject.from_yaml(
            _target_path / OutputFilenames.SUBROUTINE_FLOW_CACHE_FN
        )
        run_mode = RunMode.RESUME
        logging.info(f"Running in RESUME mode using resume dir: {_target_path}")
        base_output = new_ts_dir()
    else:
        base_output = new_ts_dir()

    return run_mode, base_output, prev_flow, _target_path


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
    is_ff2020_format = False
    if "ff2020" in input_dir_path.name.lower():
        is_ff2020_format = True
        logging.info("Detected FF2020 format based on input directory name.")

    instances = []
    for benchmark_filename in benchmark_filenames:
        input_file_path = input_dir_path / benchmark_filename
        instances.append(
            load_hfs_instance(input_file_path, is_ff2020_format=is_ff2020_format)
        )
    return instances


def _setup_logging(log_path: Path, quiet: bool) -> None:
    """Configure logging.

    - Always writes INFO‑level logs to ``log_path``.
    - If ``quiet`` is ``True`` no console output is emitted (only the file).
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler – always present
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler – optional
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)


def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner.

    Currently only ``--quiet`` is added; existing arguments can be appended here.
    """
    parser = argparse.ArgumentParser(description="Hybrid Flowshop experiment runner")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output (log is still written to file)",
    )
    # Add other arguments here if the project already defines them elsewhere.
    return parser.parse_args()


def main():
    from pathlib import Path

    e_timer = ElapsedTimer()
    args = _parse_cli()

    # --- Load and validate metadata ---
    raw_metadata = read_yaml(Path(MAIN_METADATA_FILENAME))
    config = MainMetadata.model_validate(raw_metadata)
    run_mode, base_output_dir_path, prev_flow, resume_dir = (
        determine_run_mode_and_base_dir(config, e_timer)
    )
    # --- Setup logging ---
    _setup_logging(
        base_output_dir_path / config.scenario_log_filename, quiet=args.quiet
    )

    run_experiment(
        e_timer,
        config,
        run_mode,
        base_output_dir_path,
        prev_flow,
        resume_dir,
    )


if __name__ == "__main__":
    main()
