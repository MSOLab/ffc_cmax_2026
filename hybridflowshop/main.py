from pathlib import Path
from typing import Any

from hfs_input_summary import HFSInputSummary
from hfs_summary import HFSSummary
from schore.hybridflowshop import HybridFlowShopProblem


def main():
    # I/O parameters
    first = 0
    last = 2
    benchmark_filenames = [str(i) + ".txt" for i in range(first, last + 1)]
    input_dir = "resources/pra/"
    output_dir = "../Outputs/pra/"

    # Problem parameter
    horizon = 100000

    # Solver parameters
    computational_time = 5
    n_threads = 8

    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    for benchmark_filename in benchmark_filenames:
        # Read the problem instance
        try:
            with open(input_dir_path / benchmark_filename, "r") as f:
                hfs_instance = HybridFlowShopProblem.from_pra_data(f)
        except FileNotFoundError:
            print(f"File {benchmark_filename} not found in {input_dir}.")
            continue
        except Exception as e:
            print(f"Error reading file {benchmark_filename}: {e}")
            continue

        input_summary = HFSInputSummary(
            name=benchmark_filename,
            num_jobs=hfs_instance.num_jobs,
            num_stages=hfs_instance.num_stages,
            computational_time=computational_time,
            n_threads=n_threads,
        )

        from hfs_cp_lns import HybridFlowShopCpLnsController

        kwargs_dict_for_init: dict[str, Any] = {"horizon": horizon}
        kwargs_list: list[dict[str, Any]] = [
            {
                "method_name": "solve_cp",
                "computational_time": computational_time,
                "n_threads": n_threads,
            },
            {
                "method_name": "apply_time_window_search",
                "rho": 0.2,
                "computational_time": computational_time,
                "n_threads": n_threads,
            },
        ]
        cp_lns_ctrlr = HybridFlowShopCpLnsController(
            hfs_instance, **kwargs_dict_for_init
        )
        cp_lns_ctrlr.run(kwargs_list)
        output_summary = cp_lns_ctrlr.get_result_summary()
        output_summary.report_status()
        summary = HFSSummary(inputs=input_summary, outputs=output_summary)
        summary.save(output_dir_path / benchmark_filename)

        # from pure_cp_2023_naderi import PureCP2023Naderi

        # solver_ins = PureCP2023Naderi(hfs_instance)
        # solver_ins.solve(computational_time=computational_time, n_threads=n_threads)
        # summary = HFSSummary(inputs=input_summary, outputs=solver_ins.summary)
        # summary.outputs.report_status()
        # summary.save(output_dir_path / benchmark_filename)


if __name__ == "__main__":
    main()
