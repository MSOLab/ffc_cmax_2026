from pathlib import Path

import pytest
from routix.io.yaml import dump_yaml

from hybridflowshop.io_solution import get_highlight_op_set


def test_get_highlight_op_set_reads_job_stage_pairs(tmp_path: Path) -> None:
    solution_path = tmp_path / "solution.yaml"
    dump_yaml(
        {
            "start_time_map": {},
            "end_time_map": {},
            "highlight_ops": [["j1", "s1"], ["j2", "s2"]],
        },
        solution_path,
    )

    assert get_highlight_op_set(solution_path) == {("j1", "s1"), ("j2", "s2")}


def test_get_highlight_op_set_rejects_non_pair_entries(tmp_path: Path) -> None:
    solution_path = tmp_path / "solution.yaml"
    dump_yaml(
        {
            "start_time_map": {},
            "end_time_map": {},
            "highlight_ops": [["j1", "s1", "m1"]],
        },
        solution_path,
    )

    with pytest.raises(ValueError, match="highlight_ops entries must be"):
        get_highlight_op_set(solution_path)
