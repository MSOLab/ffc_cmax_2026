from pathlib import Path

from routix.io.yaml import load_yaml

START_TIME_MAP_KEY = "start_time_map"
END_TIME_MAP_KEY = "end_time_map"


def get_start_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    solution_dict = load_yaml(sol_path, encoding=encoding)
    if START_TIME_MAP_KEY in solution_dict:
        data = solution_dict[START_TIME_MAP_KEY]
    elif "start_times" in solution_dict:
        data = solution_dict["start_times"]  # backward compatibility
    else:
        raise ValueError(
            f"Neither '{START_TIME_MAP_KEY}' nor 'start_times' found in solution file: {sol_path}"
        )
    return data


def get_end_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    solution_dict = load_yaml(sol_path, encoding=encoding)
    if END_TIME_MAP_KEY in solution_dict:
        data = solution_dict[END_TIME_MAP_KEY]
    elif "end_times" in solution_dict:
        data = solution_dict["end_times"]  # backward compatibility
    else:
        raise ValueError(
            f"Neither '{END_TIME_MAP_KEY}' nor 'end_times' found in solution file: {sol_path}"
        )
    return data
