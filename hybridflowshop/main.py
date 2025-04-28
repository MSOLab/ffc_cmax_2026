from pathlib import Path

from hfs_summary import HFSInputSummary, HFSSummary
from schore.hybridflowshop import HybridFlowShopProblem


def main():
    first = 0
    last = 2

    benchmark_filenames = [str(i) + ".txt" for i in range(first, last + 1)]
    input_dir = "resources/pra/"
    output_dir = "../Outputs/pra/"
    computational_time = 10
    n_threads = 12

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

        from pure_cp_2023_naderi import PureCP2023Naderi

        solver_ins = PureCP2023Naderi(hfs_instance)
        solver_ins.solve(computational_time=computational_time, n_threads=n_threads)
        summary = HFSSummary(inputs=input_summary, outputs=solver_ins.summary)
        summary.outputs.report_status()
        summary.save(output_dir_path / benchmark_filename)


if __name__ == "__main__":
    main()
