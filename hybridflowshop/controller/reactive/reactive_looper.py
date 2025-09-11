import logging
from pathlib import Path

from mbls.cpsat import CpsatStatus
from routix.util.concurrent import batch_write_data_to_csv, batch_write_data_to_yaml

from ...report import HfsCpsatSolverReport
from ..controller_core import HybridFlowShopCpLnsControllerCore
from .local_stopping_criteria import LocalStoppingCriteria
from .reactive_loop_report import ReactiveLoopReportEntry
from .reactive_param_tuner import ReactiveParamTuner, TunerParams


class ReactiveLooper:
    ctrlr: HybridFlowShopCpLnsControllerCore
    """Controller managing subroutines."""

    # Initialized attributes
    subroutine_names: list[str]
    """List of subroutine names to be called in each iteration."""
    reactive_param_tuner_dict: dict[str, ReactiveParamTuner]
    """Parameter tuner for the reactive looper."""
    stopping_criteria: LocalStoppingCriteria
    """(Local) Stopping criteria for the reactive looper."""

    # State
    obj_value_before_step: float | None
    """Objective value before the last subroutine call."""
    loop_count: int
    """Current iteration count."""
    no_improvement_step_count: int
    """Number of consecutive iterations without improvement."""
    report_entries: list[ReactiveLoopReportEntry]
    """History of report entries collected during run."""

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
        self.report_entries = []

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

        # capture snapshot of parameters and objective before call
        kwargs_snapshot = tuner.current_kwargs.copy()
        prev_obj = self.obj_value_before_step
        time_start = self.ctrlr.timer.get_elapsed_sec()

        # timelimit by global - offset
        timelimit_by_global = self.ctrlr.timer.get_remaining_sec(
            self.ctrlr.stopping_criteria.timelimit
        )
        if self.stopping_criteria.stop_at_global_timelimit_minus is not None:
            timelimit_by_global -= self.stopping_criteria.stop_at_global_timelimit_minus
        if timelimit_by_global <= kwargs_snapshot.get(
            "computational_time", float("inf")
        ):
            kwargs_snapshot["computational_time"] = timelimit_by_global
        tuner.call_method(timelimit_by_global)

        report = self.ctrlr.solution_manager.get_last_report()
        if isinstance(report, HfsCpsatSolverReport):
            # compute report entry fields
            obj_value = (
                report.obj_value
                if report.obj_value is not None
                else self.ctrlr.solution_manager.best_obj_value
            )
            is_optimal = report.status == CpsatStatus.OPTIMAL
            is_improved = False
            if report.obj_value is not None and prev_obj is not None:
                try:
                    is_improved = self.ctrlr.solution_manager._a_is_better_obj_value(
                        report.obj_value, prev_obj
                    )
                except Exception:
                    is_improved = False

            elapsed = (
                report.elapsed_time
                if hasattr(report, "elapsed_time") and report.elapsed_time is not None
                else (self.ctrlr.timer.get_elapsed_sec() - time_start)
            )

            # create and store report entry
            entry = ReactiveLoopReportEntry(
                iter_count=self.loop_count,
                subroutine_name=subroutine_name,
                kwargs=kwargs_snapshot,
                time_start=time_start,
                time_elapsed=elapsed,
                obj_value=obj_value if obj_value is not None else float("nan"),
                is_optimal=is_optimal,
                is_improved=is_improved,
            )
            self.report_entries.append(entry)

            # update tuner logic/state
            self.update_reactive_params(subroutine_name, report)
        else:
            raise ValueError(
                "The last report is not an instance of HfsCpsatSolverReport."
            )

    def write_report_yaml(self, report_path: Path) -> None:
        """Append collected report entries to a YAML file (multi-doc).

        Args:
            report_path (Path): The path to the YAML report file.
        """
        report_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [e.get_row_dict() for e in self.report_entries]
        if rows:
            batch_write_data_to_yaml(report_path, rows)

    def write_report_csv(self, report_path: Path) -> None:
        """Append collected report entries to a CSV file.

        Args:
            report_path (Path): The path to the CSV report file.
        """
        report_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [e.get_row_dict() for e in self.report_entries]
        if rows:
            header = ReactiveLoopReportEntry.get_header()
            batch_write_data_to_csv(report_path, rows, header)

    def update_reactive_params(
        self, subroutine_name: str, report_by_last_subroutine: HfsCpsatSolverReport
    ) -> None:
        tuner = self.reactive_param_tuner_dict[subroutine_name]

        if report_by_last_subroutine.status == CpsatStatus.OPTIMAL:
            if report_by_last_subroutine.obj_value is None:
                raise ValueError("Optimal solution must have an objective value.")
            last_obj_value = report_by_last_subroutine.obj_value
            if self.ctrlr.solution_manager._a_is_better_obj_value(
                last_obj_value, self.obj_value_before_step
            ):
                logging.info("Last solution was optimal & improved.")
                self.no_improvement_step_count = 0
                self.obj_value_before_step = last_obj_value

                tuner.increment("rho")
                tuner.decrement("computational_time")
            else:
                logging.info("Last solution was optimal & not improved.")
                self.no_improvement_step_count += 1

                tuner.increment("rho")
        else:
            if report_by_last_subroutine.is_feasible:
                if report_by_last_subroutine.obj_value is None:
                    raise ValueError("Optimal solution must have an objective value.")
                last_obj_value = report_by_last_subroutine.obj_value
                if self.ctrlr.solution_manager._a_is_better_obj_value(
                    last_obj_value, self.obj_value_before_step
                ):
                    logging.info("Last solution was timeout & improved.")
                    self.no_improvement_step_count = 0
                    self.obj_value_before_step = last_obj_value

                    tuner.increment("computational_time")
                else:
                    logging.info("Last solution was timeout & not improved.")
                    self.no_improvement_step_count += 1
                    if tuner.current_value_hits_ub("computational_time"):
                        tuner.increment("rho")
                    else:
                        tuner.decrement("rho")
                        tuner.increment("computational_time")
            else:
                logging.info("Last solution was timeout & not improved.")
                self.no_improvement_step_count += 1
                if tuner.current_value_hits_ub("computational_time"):
                    tuner.increment("rho")
                else:
                    tuner.decrement("rho")
                    tuner.increment("computational_time")

    def run(self) -> None:
        def call_and_true_if_stop(
            name_for_context_manager: str, subroutine_name: str
        ) -> bool:
            logging.info(f"Calling subroutine for {name_for_context_manager}.")

            with self.ctrlr.temporarily_extended_context(name_for_context_manager):
                self.call_subroutine(subroutine_name)

            return (
                self.ctrlr.is_stopping_condition() or self.is_local_stopping_condition()
            )

        while (
            not self.ctrlr.is_stopping_condition()
            and not self.is_local_stopping_condition()
        ):
            self.loop_count += 1
            logging.info(f"Reactive loop #{self.loop_count} starts.")

            if len(self.subroutine_names) == 0:
                logging.info("No subroutines to call. Reactive looper ends.")
                break
            elif len(self.subroutine_names) == 1:
                subroutine_name = self.subroutine_names[0]
                name_for_context_manager = f"{self.loop_count}_{subroutine_name}"
                if call_and_true_if_stop(name_for_context_manager, subroutine_name):
                    break
            else:
                for idx, subroutine_name in enumerate(self.subroutine_names):
                    name_for_context_manager = (
                        f"{self.loop_count}_{idx + 1}-{subroutine_name}"
                    )
                    if call_and_true_if_stop(name_for_context_manager, subroutine_name):
                        break


        logging.info("Reactive looper ends.")
