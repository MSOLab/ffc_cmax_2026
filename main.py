from pathlib import Path
from typing import Any

import yaml
from cplnx import DynamicDataObject, SubroutineFlowValidator
from schore.hybridflowshop import HybridFlowShopProblem

from hybridflowshop.hfs_cp_lns import HybridFlowShopCpLnsController
from hybridflowshop.hfs_input_summary import HFSInputSummary
from hybridflowshop.hfs_summary import HFSSummary
from hybridflowshop.stopping_criteria import StoppingCriteria

MAIN_METADATA_FILENAME = "main_metadata.yaml"


def main():
    # Read the main metadata file
    main_metadata_dict = read_yaml(Path(MAIN_METADATA_FILENAME))

    # Read common parameters for PRA benchmarks
    pra_common_params_rel_path = Path(main_metadata_dict["pra_common_params_rel_path"])
    pra_common_params_dict = read_yaml(pra_common_params_rel_path)

    # Read the stopping criteria and subroutine flow
    stopping_criteria_rel_path = Path(main_metadata_dict["stopping_criteria_rel_path"])
    stopping_criteria_dict = read_yaml(stopping_criteria_rel_path)
    subroutine_flow_rel_path = Path(main_metadata_dict["subroutine_flow_rel_path"])
    subroutine_flow_obj = read_yaml(subroutine_flow_rel_path)

    # I/O parameters
    first: int = main_metadata_dict["first"]
    last: int = main_metadata_dict["last"]
    benchmark_filename_format: str = main_metadata_dict["benchmark_filename_format"]
    benchmark_filenames = [
        benchmark_filename_format.format(i) for i in range(first, last + 1)
    ]
    input_dir_path = Path(main_metadata_dict["input_dir"])
    output_dir_path = Path(main_metadata_dict["output_dir"])
    result_gantt_filename_format: str = main_metadata_dict[
        "result_gantt_filename_format"
    ]

    # Initialize output directory
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Subroutine controller arguments
    stopping_criteria = StoppingCriteria(stopping_criteria_dict)
    subroutine_flow = DynamicDataObject.from_obj(subroutine_flow_obj)

    # Validate the subroutine flow
    validator = SubroutineFlowValidator(HybridFlowShopCpLnsController)
    validator.validate(subroutine_flow)

    for benchmark_filename in benchmark_filenames:
        # Read the problem instance
        input_file_path = input_dir_path / benchmark_filename
        hfs_instance = load_hfs_instance(input_dir_path / benchmark_filename)
        ins_name = input_file_path.stem

        input_summary = HFSInputSummary(
            name=ins_name,
            num_jobs=hfs_instance.num_jobs,
            num_stages=hfs_instance.num_stages,
            timelimit=stopping_criteria.timelimit,
        )

        cp_lns_ctrlr = create_controller(
            hfs_instance, stopping_criteria, subroutine_flow, pra_common_params_dict
        )
        cp_lns_ctrlr.set_working_dir(output_dir_path / ins_name)
        cp_lns_ctrlr.run()

        # Save the result
        result_gantt_filename = result_gantt_filename_format.format(ins_name=ins_name)
        cp_lns_ctrlr.draw_incumbent_gantt(output_dir_path / result_gantt_filename)
        expr_summary = cp_lns_ctrlr.get_experiment_summary()

        summary = HFSSummary(inputs=input_summary, outputs=expr_summary)
        summary_filename = input_file_path.stem + ".csv"
        summary.save(output_dir_path / summary_filename)


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


def create_controller(
    hfs_instance, stopping_criteria, subroutine_flow, controller_init_kwargs
):
    return HybridFlowShopCpLnsController(
        hfs_instance, stopping_criteria, subroutine_flow, **controller_init_kwargs
    )


if __name__ == "__main__":
    main()
