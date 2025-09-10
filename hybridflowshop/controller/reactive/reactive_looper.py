import logging

from mbls.cpsat import CpsatStatus

from ...report import HfsCpsatSolverReport
from ..controller_core import HybridFlowShopCpLnsControllerCore
from .local_stopping_criteria import LocalStoppingCriteria
from .reactive_param_tuner import ReactiveParamTuner, TunerParams


class ReactiveLooper:
    ctrlr: HybridFlowShopCpLnsControllerCore
    """Controller managing subroutines."""

    # Initialized attributes
    subroutine_names: list[str]
    reactive_param_tuner_dict: dict[str, ReactiveParamTuner]
    """Parameter tuner for the reactive looper."""
    stopping_criteria: LocalStoppingCriteria
    """(Local) Stopping criteria for the reactive looper."""

    # State
    obj_value_before_step: float | None
    loop_count: int
    no_improvement_step_count: int

    def __init__(
        self,
        ctrlr: HybridFlowShopCpLnsControllerCore,
        subroutine_names: list[str],
        opening_kwargs: dict,
        reactive_param_tuner_dict: dict,
        stopping_criteria: dict,
    ):
        # Initialized attributes

        self.ctrlr = ctrlr

        self.subroutine_names = subroutine_names.copy()
        for subroutine_name in self.subroutine_names:
            if not hasattr(ctrlr, subroutine_name):
                raise ValueError(
                    f"Controller does not have subroutine '{subroutine_name}'."
                )

        self.reactive_param_tuner_dict = {}
        for subroutine_name in self.subroutine_names:
            tuner_param_dict = {}
            tuner_param_dict_keys = {"rho", "computational_time"}
            for key in tuner_param_dict_keys:
                if key not in reactive_param_tuner_dict:
                    raise ValueError(
                        f"Reactive parameter tuner dict must contain '{key}'."
                    )
                tuner_param_dict[key] = TunerParams(**reactive_param_tuner_dict[key])
            self.reactive_param_tuner_dict[subroutine_name] = ReactiveParamTuner(
                method=getattr(ctrlr, subroutine_name),
                opening_kwargs=opening_kwargs,
                tuner_param_dict=tuner_param_dict,
            )

        self.stopping_criteria = LocalStoppingCriteria(stopping_criteria)

        # State

        self.obj_value_before_step = ctrlr.solution_manager.best_obj_value
        self.loop_count = 0
        self.no_improvement_step_count = 0

    # @classmethod
    # def from_param_dict(
    #     cls, ctrlr: HybridFlowShopCpLnsControllerCore, param_dict: dict
    # ):
    #     subroutine_names_key = "subroutine_names"
    #     opening_kwargs_key = "opening_kwargs"
    #     reactive_param_tuner_dict_key = "reactive_param_tuner_dict"
    #     stopping_criteria_key = "stopping_criteria"
    #     missing_param_dict_keys = [
    #         key
    #         for key in {
    #             subroutine_names_key,
    #             opening_kwargs_key,
    #             reactive_param_tuner_dict_key,
    #             stopping_criteria_key,
    #         }
    #         if key not in param_dict
    #     ]
    #     if missing_param_dict_keys:
    #         raise ValueError(
    #             f"param_dict must contain {', '.join(missing_param_dict_keys)}."
    #         )
    #     return cls(
    #         ctrlr=ctrlr,
    #         subroutine_names=param_dict[subroutine_names_key],
    #         opening_kwargs=param_dict[opening_kwargs_key],
    #         reactive_param_tuner_dict=param_dict[reactive_param_tuner_dict_key],
    #         stopping_criteria=param_dict[stopping_criteria_key],
    #     )

    def is_local_stopping_condition(self) -> bool:
        min_rho = min(
            tuner.get_current_value("rho")
            for tuner in self.reactive_param_tuner_dict.values()
        )
        return self.stopping_criteria.is_stopping_condition(
            self.loop_count,
            self.no_improvement_step_count,
            min_rho,
            self.ctrlr.timer.get_remaining_sec(self.ctrlr.stopping_criteria.timelimit),
            self.ctrlr.obj_store.get_last_gap(),
        )

    def call_subroutine(self, subroutine_name: str) -> None:
        if subroutine_name not in self.reactive_param_tuner_dict:
            raise ValueError(f"Subroutine {subroutine_name} is not recognized.")
        tuner = self.reactive_param_tuner_dict[subroutine_name]
        tuner.call_method()
        report = self.ctrlr.solution_manager.get_last_report()
        if isinstance(report, HfsCpsatSolverReport):
            self.update_reactive_params(report)
        else:
            raise ValueError(
                "The last report is not an instance of HfsCpsatSolverReport."
            )

    def update_reactive_params(
        self, report_by_last_subroutine: HfsCpsatSolverReport
    ) -> None:
        if report_by_last_subroutine.status == CpsatStatus.OPTIMAL:
            if report_by_last_subroutine.obj_value is None:
                raise ValueError("Optimal solution must have an objective value.")
            last_obj_value = report_by_last_subroutine.obj_value
            if self.ctrlr.solution_manager._a_is_better_obj_bound(
                last_obj_value, self.obj_value_before_step
            ):
                logging.info("Last solution was optimal & improved.")
                self.no_improvement_step_count = 0
                self.obj_value_before_step = last_obj_value
                for tuner in self.reactive_param_tuner_dict.values():
                    tuner.increment("rho")
                    tuner.decrement("computational_time")
            else:
                logging.info("Last solution was optimal & not improved.")
                self.no_improvement_step_count += 1
                for tuner in self.reactive_param_tuner_dict.values():
                    tuner.increment("rho")
        else:
            if report_by_last_subroutine.is_feasible:
                if report_by_last_subroutine.obj_value is None:
                    raise ValueError("Optimal solution must have an objective value.")
                last_obj_value = report_by_last_subroutine.obj_value
                if self.ctrlr.solution_manager._a_is_better_obj_bound(
                    last_obj_value, self.obj_value_before_step
                ):
                    logging.info("Last solution was timeout & improved.")
                    self.no_improvement_step_count = 0
                    self.obj_value_before_step = last_obj_value
                    for tuner in self.reactive_param_tuner_dict.values():
                        tuner.increment("computational_time")
                else:
                    logging.info("Last solution was timeout & not improved.")
                    self.no_improvement_step_count += 1
                    for tuner in self.reactive_param_tuner_dict.values():
                        tuner.decrement("rho")
                        tuner.increment("computational_time")
            else:
                logging.info("Last solution was timeout & not improved.")
                self.no_improvement_step_count += 1
                for tuner in self.reactive_param_tuner_dict.values():
                    tuner.decrement("rho")
                    tuner.increment("computational_time")

    def run(self) -> None:
        while (
            not self.ctrlr.is_stopping_condition()
            and not self.is_local_stopping_condition()
        ):
            self.loop_count += 1
            logging.info(f"Reactive looper iteration {self.loop_count} starts.")

            for subroutine_name in self.subroutine_names:
                logging.info(f"Calling subroutine '{subroutine_name}'.")
                self.call_subroutine(subroutine_name)

                if (
                    self.ctrlr.is_stopping_condition()
                    or self.is_local_stopping_condition()
                ):
                    break

        logging.info("Reactive looper ends.")
