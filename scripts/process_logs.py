from pathlib import Path

from hybridflowshop.report.log_processor import (
    DEFAULT_CONTROLLER_LOG_NAME,
    DEFAULT_RESULTS_DIR,
    OBJ_LOG_FN_FORMAT,
    LogProcessor,
    create_method_end_time_and_obj_value_summary,
    get_methods_from_flow,
    parse_controller_log,
    parse_obj_log,
    process_instance,
    process_scenario,
)

__all__ = [
    "DEFAULT_CONTROLLER_LOG_NAME",
    "DEFAULT_RESULTS_DIR",
    "OBJ_LOG_FN_FORMAT",
    "LogProcessor",
    "create_method_end_time_and_obj_value_summary",
    "get_methods_from_flow",
    "parse_controller_log",
    "parse_obj_log",
    "process_instance",
    "process_scenario",
]


def main():
    cwd = Path.cwd()
    if (cwd / "subroutine_flow.yaml").exists():
        process_scenario(cwd)
    else:
        print("Run this script from a scenario output directory, or import it.")


if __name__ == "__main__":
    main()
