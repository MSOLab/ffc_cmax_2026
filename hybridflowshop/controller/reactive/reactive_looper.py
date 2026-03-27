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
    size_param_name_by_subroutine: dict[str, str | None]
    """Primary neighborhood-size parameter name for each subroutine."""
    stopping_criteria: LocalStoppingCriteria
    """(Local) Stopping criteria for the reactive looper."""

    # State
    obj_value_before_step: float | None
    """Objective value before the last subroutine call."""
    loop_count: int
    """Current iteration count."""
    no_improvement_step_series_lth: int
    """Number of consecutive iterations without improvement."""
    report_entries: list[ReactiveLoopReportEntry]
    """History of report entries collected during run."""

    def __init__(
        self,
        ctrlr: HybridFlowShopCpLnsControllerCore,
        routine_data: list[dict],
        reactive_param_tuner_dict: dict,
        stopping_criteria: dict,
    ):
        # Initialized attributes

        self.ctrlr = ctrlr

        self.subroutine_names = []
        self.size_param_name_by_subroutine = {}
        opening_kwargs_list = []
        for subroutine_data in routine_data:
            if "method" not in subroutine_data:
                continue
            subroutine_name = subroutine_data["method"]
            if not hasattr(ctrlr, subroutine_name):
                raise ValueError(
                    f"Controller does not have subroutine '{subroutine_name}'."
                )
            self.subroutine_names.append(subroutine_name)
            opening_kwargs_list.append(
                {k: v for k, v in subroutine_data.items() if k != "method"}
            )

        self.reactive_param_tuner_dict = {}
        for subroutine_name, subroutine_opening_kwargs in zip(
            self.subroutine_names, opening_kwargs_list
        ):
            tuner_param_dict = {}
            size_param_name = None
            for candidate_key in ("job_count", "rho"):
                if candidate_key in subroutine_opening_kwargs:
                    size_param_name = candidate_key
                    break
            tuner_param_dict_keys = ["computational_time"]
            if size_param_name is not None:
                tuner_param_dict_keys.append(size_param_name)
            for key in tuner_param_dict_keys:
                if key not in reactive_param_tuner_dict:
                    raise ValueError(
                        f"Reactive parameter tuner dict must contain '{key}'."
                    )
                tuner_param_dict[key] = TunerParams(**reactive_param_tuner_dict[key])
            self.size_param_name_by_subroutine[subroutine_name] = size_param_name
            self.reactive_param_tuner_dict[subroutine_name] = ReactiveParamTuner(
                method=getattr(ctrlr, subroutine_name),
                opening_kwargs=subroutine_opening_kwargs,
                tuner_param_dict=tuner_param_dict,
            )
            logging.info(
                f"Reactive parameter tuner initialized for subroutine '{subroutine_name}'"
                f" with opening kwargs {subroutine_opening_kwargs}."
            )

        self.stopping_criteria = LocalStoppingCriteria(stopping_criteria)

    def _get_size_param_name(self, subroutine_name: str) -> str | None:
        return self.size_param_name_by_subroutine.get(subroutine_name)

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

    def _is_loop_stopping_condition(self, log_reason_if_true: bool = True) -> bool:
        global_timelimit = self.ctrlr.stopping_criteria.timelimit
        return self.stopping_criteria.is_loop_stopping_condition(
            self.loop_count,
            self.no_improvement_step_series_lth,
            self.ctrlr.timer.get_remaining_sec(global_timelimit),
            self.ctrlr.obj_store.get_last_gap(),
            global_timelimit,
            log_reason_if_true=log_reason_if_true,
        )

    def _call_subroutine(self, subroutine_name: str) -> None:
        if subroutine_name not in self.reactive_param_tuner_dict:
            raise ValueError(f"Subroutine {subroutine_name} is not recognized.")
        tuner = self.reactive_param_tuner_dict[subroutine_name]

        # capture snapshot of parameters and objective before call
        kwargs_snapshot = tuner.current_kwargs.copy()
        prev_obj = self.obj_value_before_step
        time_start = self.ctrlr.timer.get_elapsed_sec()

        # timelimit by global - offset
        global_remaining = self.ctrlr.timer.get_remaining_sec(
            self.ctrlr.stopping_criteria.timelimit
        )
        timelimit_by_global = self.stopping_criteria.get_subroutine_timelimit(
            global_remaining, global_timelimit=self.ctrlr.stopping_criteria.timelimit
        )
        if timelimit_by_global <= kwargs_snapshot.get(
            "computational_time", float("inf")
        ):
            kwargs_snapshot["computational_time"] = timelimit_by_global
        logging.info(
            f"Calling subroutine {subroutine_name} with kwargs {kwargs_snapshot}."
        )
        tuner.call_method(timelimit_by_global)

        report = self.ctrlr.solution_manager.get_last_report()
        if isinstance(report, HfsCpsatSolverReport):
            # compute report entry fields
            obj_value = (
                report.obj_value
                if report.obj_value is not None
                else self.ctrlr.solution_manager.best_obj_value
            )
            if tuner.current_kwargs["computational_time"] - report.elapsed_time <= 1e-6:
                timelimit_reached = True
            else:
                timelimit_reached = False
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
                prev_obj_value=prev_obj if prev_obj is not None else float("nan"),
                obj_value=obj_value if obj_value is not None else float("nan"),
                timelimit_reached=timelimit_reached,
                is_optimal=is_optimal,
                is_improved=is_improved,
            )
            self.report_entries.append(entry)

            # update tuner logic/state
            self._update_reactive_params(subroutine_name, report)
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

    def _update_reactive_params(
        self, subroutine_name: str, report_by_last_subroutine: HfsCpsatSolverReport
    ) -> None:
        tuner = self.reactive_param_tuner_dict[subroutine_name]
        size_param_name = self._get_size_param_name(subroutine_name)

        if report_by_last_subroutine.status == CpsatStatus.OPTIMAL:
            if report_by_last_subroutine.obj_value is None:
                raise ValueError("Optimal solution must have an objective value.")
            last_obj_value = report_by_last_subroutine.obj_value
            if self.ctrlr.solution_manager._a_is_better_obj_value(
                last_obj_value, self.obj_value_before_step
            ):
                logging.info("Last solution was optimal & improved.")
                self.no_improvement_step_series_lth = 0
                self.obj_value_before_step = last_obj_value
                # If optimal & improved, do nothing
            else:
                logging.info("Last solution was optimal & not improved.")
                self.no_improvement_step_series_lth += 1
                if size_param_name is not None:
                    tuner.increment(size_param_name)
        else:
            if report_by_last_subroutine.is_feasible:
                if report_by_last_subroutine.obj_value is None:
                    raise ValueError("Feasible solution must have an objective value.")
                last_obj_value = report_by_last_subroutine.obj_value
                if self.ctrlr.solution_manager._a_is_better_obj_value(
                    last_obj_value, self.obj_value_before_step
                ):
                    logging.info("Last solution was timeout & improved.")
                    self.no_improvement_step_series_lth = 0
                    self.obj_value_before_step = last_obj_value
                    # If feasible & improved, do nothing
                else:
                    logging.info("Last solution was timeout & not improved.")
                    self.no_improvement_step_series_lth += 1
                    if not tuner.current_value_hits_ub("computational_time"):
                        # If not improved but not enough time, increase time limit
                        # If tl_hits_ub in stopping condition, run method will exclude the subroutine
                        tuner.increment("computational_time")
                    elif size_param_name is not None:
                        # If not improved despite maximum time, increase neighborhood size
                        tuner.increment(size_param_name)

            else:
                logging.info("Last solution was timeout & not improved.")
                self.no_improvement_step_series_lth += 1
                if not tuner.current_value_hits_ub("computational_time"):
                    # If no solution but not enough time, increase time limit
                    # If tl_hits_ub in stopping condition, run method will exclude the subroutine
                    tuner.increment("computational_time")
                elif size_param_name is not None:
                    # If not improved despite maximum time, increase neighborhood size
                    tuner.increment(size_param_name)

    def initialize_states(self) -> None:
        self.obj_value_before_step = self.ctrlr.solution_manager.best_obj_value
        self.loop_count = 0
        self.no_improvement_step_series_lth = 0
        self.report_entries = []

    def run(self) -> None:
        self.initialize_states()
        excluded_subroutines = set()

        def call_and_true_if_stop(
            name_for_context_manager: str, subroutine_name: str
        ) -> bool:
            with self.ctrlr.temporarily_extended_context(name_for_context_manager):
                self._call_subroutine(subroutine_name)

            tuner = self.reactive_param_tuner_dict[subroutine_name]
            size_param_name = self._get_size_param_name(subroutine_name)
            if (
                size_param_name is not None
                and self.stopping_criteria.rho_hits_ub
                and tuner.current_value_hits_ub(size_param_name)
            ):
                size_param_value = tuner.get_current_value(size_param_name)
                size_param_ub = tuner._tuner_param_dict[size_param_name].max
                logging.info(
                    f"Subroutine '{subroutine_name}' is excluded in the next loop: "
                    f"{size_param_name}_hits_ub "
                    f"(value={size_param_value} >= {size_param_ub}=criteria)"
                )
                excluded_subroutines.add(subroutine_name)
            if self.stopping_criteria.tl_hits_ub and tuner.current_value_hits_ub(
                "computational_time"
            ):
                tl = tuner.get_current_value("computational_time")
                tl_ub = tuner._tuner_param_dict["computational_time"].max
                logging.info(
                    f"Subroutine '{subroutine_name}' is excluded in the next loop: "
                    f"tl_hits_ub (value={tl} >= {tl_ub}=criteria)"
                )
                excluded_subroutines.add(subroutine_name)

            return self.ctrlr.is_stopping_condition(
                log_reason_if_true=False
            ) or self._is_loop_stopping_condition(log_reason_if_true=False)

        while (
            not self.ctrlr.is_stopping_condition()
            and not self._is_loop_stopping_condition()
        ):
            if len(excluded_subroutines) == len(self.subroutine_names):
                logging.info(
                    "All subroutines are excluded at the end of loop %d.",
                    self.loop_count,
                )
                break
            if len(self.subroutine_names) == 0:
                logging.info(
                    "No subroutines to call at the end of loop %d", self.loop_count
                )
                break

            self.loop_count += 1
            logging.info("Reactive loop %d starts.", self.loop_count)

            if len(self.subroutine_names) == 1:
                subroutine_name = self.subroutine_names[0]
                if subroutine_name in excluded_subroutines:
                    break
                name_for_context_manager = f"{self.loop_count}_{subroutine_name}"
                if call_and_true_if_stop(name_for_context_manager, subroutine_name):
                    break
            else:
                for idx, subroutine_name in enumerate(self.subroutine_names):
                    if subroutine_name in excluded_subroutines:
                        continue
                    name_for_context_manager = (
                        f"{self.loop_count}_{idx + 1}-{subroutine_name}"
                    )
                    if call_and_true_if_stop(name_for_context_manager, subroutine_name):
                        break

        logging.info(
            "Reactive looper done with objValue %d",
            self.ctrlr.solution_manager.best_obj_value,
        )
