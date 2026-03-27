from pathlib import Path

from routix.io.yaml import load_yaml

START_TIME_MAP_KEY = "start_time_map"
END_TIME_MAP_KEY = "end_time_map"
HIGHLIGHT_OPS_KEY = "highlight_ops"


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


def get_highlight_op_set(
    sol_path: Path, encoding: str = "utf-8"
) -> set[tuple[str, str]]:
    data = None
    solution_dict = load_yaml(sol_path, encoding=encoding)
    if HIGHLIGHT_OPS_KEY in solution_dict:
        data = {
            _validate_and_normalize_highlight_op(op)
            for op in solution_dict[HIGHLIGHT_OPS_KEY]
        }
    else:
        data = set()  # No highlight ops specified, return empty set
    return data


def _validate_and_normalize_highlight_op(op: object) -> tuple[str, str]:
    if not isinstance(op, (list, tuple)) or len(op) != 2:
        raise ValueError(
            "highlight_ops entries must be [job, stage] pairs in solution file."
        )
    job_id, stage_id = op
    return str(job_id), str(stage_id)
