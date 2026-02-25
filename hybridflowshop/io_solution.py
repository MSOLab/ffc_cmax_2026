from pathlib import Path

import yaml
from routix.io import pyyaml_key_to_tuple

START_TIME_MAP_KEY = "start_time_map"
END_TIME_MAP_KEY = "end_time_map"


def get_start_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    with open(sol_path, "r", encoding=encoding) as f:
        solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        if START_TIME_MAP_KEY in solution_dict:
            data = solution_dict[START_TIME_MAP_KEY]
        elif "start_times" in solution_dict:
            data = solution_dict["start_times"]  # backward compatibility
        else:
            raise ValueError(
                f"Neither 'start_times' nor 'start_time_map' found in solution file: {sol_path}"
            )
    return pyyaml_key_to_tuple(data)


def get_end_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    with open(sol_path, "r", encoding=encoding) as f:
        solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        if END_TIME_MAP_KEY in solution_dict:
            data = solution_dict[END_TIME_MAP_KEY]
        elif "end_times" in solution_dict:
            data = solution_dict["end_times"]  # backward compatibility
        else:
            raise ValueError(
                f"Neither 'end_times' nor 'end_time_map' found in solution file: {sol_path}"
            )
    return pyyaml_key_to_tuple(data)
