from pathlib import Path
from typing import Any

import yaml
from mbls import DynamicDataObject, ElapsedTimer, SubroutineFlowValidator
from schore.hybridflowshop import HybridFlowShopProblem

from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.stopping_criteria import StoppingCriteria
from single_hfs_instance_runner import SingleHFSInstanceRunner

MAIN_METADATA_FILENAME = "main_metadata.yaml"


def main():
    e_timer = ElapsedTimer()

    # Read the main metadata file
    main_metadata_dict = read_yaml(Path(MAIN_METADATA_FILENAME))

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

    # Output parameters
    benchmark_filename_format = str(main_metadata_dict["benchmark_filename_format"])
    benchmark_filenames = [
        benchmark_filename_format.format(i) for i in range(first, last + 1)
    ]
    output_dir_path = Path(main_metadata_dict["output_dir"])
    result_gantt_filename_format = str(
        main_metadata_dict["result_gantt_filename_format"]
    )
    progress_plot_filename_format = str(
        main_metadata_dict["progress_plot_filename_format"]
    )

    # Initialize working directory
    working_dir_path = init_working_dir(output_dir_path, e_timer)

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
    output_metadata = {
        "start_dt": e_timer.start_dt,
        "result_gantt_filename_format": result_gantt_filename_format,
        "progress_plot_filename_format": progress_plot_filename_format,
    }

    for benchmark_filename in benchmark_filenames:
        # Read the problem instance
        input_file_path = input_dir_path / benchmark_filename
        hfs_instance = load_hfs_instance(input_file_path)

        single_hfs_ins_solver = SingleHFSInstanceRunner(
            hfs_instance,
            pra_common_params_dict,
            subroutine_flow,
            stopping_criteria,
            output_dir_path,
            output_metadata,
        )
        single_hfs_ins_solver.solve()

    # Print elapsed time
    print(f"Elapsed time: {e_timer.get_formatted_elapsed_time()} seconds")


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
    working_dir = output_dir / e_timer.get_formatted_start_dt()
    working_dir.mkdir(parents=True, exist_ok=True)
    return working_dir


if __name__ == "__main__":
    main()
