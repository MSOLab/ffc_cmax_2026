from typing import Any

from pure_cp_2023_naderi import PureCP2023Naderi
from schore.hybridflowshop.problem import HybridFlowShopProblem


class HybridFlowShopCpLnsController:
    base_cp_model: PureCP2023Naderi

    def __init__(self, hfs_instance: HybridFlowShopProblem):
        self.base_cp_model = PureCP2023Naderi(hfs_instance)

    def run(self, kwargs_dict_by_subroutine: dict[str, dict[str, Any]]):
        self.execute_subroutine_flow(kwargs_dict_by_subroutine)

    def execute_subroutine_flow(
        self, args_dict_by_subroutine: dict[str, dict[str, Any]]
    ):
        self.solve_base_cp(args_dict_by_subroutine["solve_pure_cp"])

    def solve_base_cp(self, kwargs_dict: dict[str, Any]):
        try:
            computational_time = float(kwargs_dict["computational_time"])
            n_threads = int(kwargs_dict["n_threads"])
        except KeyError as e:
            raise ValueError(f"Missing required argument: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid argument value: {e}")
        # Call the solve method to execute the solver
        print(
            f"Solving base CP with computational_time={computational_time}, n_threads={n_threads}"
        )
        self.base_cp_model.solve(computational_time, n_threads)

    def get_result_summary(self):
        return self.base_cp_model.summary
