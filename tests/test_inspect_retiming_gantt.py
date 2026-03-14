from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


def test_inspect_retiming_gantt_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "inspect_retiming_gantt.py"
    python_path = repo_root / ".venv" / "bin" / "python"
    out_dir = tmp_path / "retiming_pra_0"

    result = subprocess.run(
        [
            str(python_path),
            str(script_path),
            "--instance",
            "resources/pra/0.txt",
            "--out-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_files = {
        "0_base.png",
        "1_right_justified_partial.png",
        "2_right_justified_full.png",
        "3_semi_active_after_rj_partial.png",
        "4_semi_active_after_rj_full.png",
    }
    assert {path.name for path in out_dir.iterdir()} == expected_files

    stdout = result.stdout
    selected_jobs_match = re.search(r"selected_partial_jobs: (\[.*\])", stdout)
    assert selected_jobs_match is not None
    selected_jobs = ast.literal_eval(selected_jobs_match.group(1))
    assert len(selected_jobs) == 2
    assert len(set(selected_jobs)) == 2

    selected_last_stage_jobs_match = re.search(
        r"selected_last_stage_jobs: (\[.*\])",
        stdout,
    )
    assert selected_last_stage_jobs_match is not None
    selected_last_stage_jobs = ast.literal_eval(selected_last_stage_jobs_match.group(1))
    assert len(selected_last_stage_jobs) == 2
    assert [
        job_id for job_id, _mc_id, _end_time in selected_last_stage_jobs
    ] == selected_jobs

    total_shift_match = re.search(r"selected_total_shift: (\d+)", stdout)
    assert total_shift_match is not None
    assert int(total_shift_match.group(1)) >= 0

    partial_rj_match = re.search(
        r"partial_right_justified_operation_set: (\[.*\])",
        stdout,
    )
    assert partial_rj_match is not None
    partial_rj_operation_set = ast.literal_eval(partial_rj_match.group(1))
    assert len(partial_rj_operation_set) == 6
    assert {job_id for job_id, _stage_id, _mc_id in partial_rj_operation_set} == set(
        selected_jobs
    )

    partial_sa_match = re.search(
        r"partial_semi_active_operation_set: (\[.*\])",
        stdout,
    )
    assert partial_sa_match is not None
    partial_sa_operation_set = ast.literal_eval(partial_sa_match.group(1))

    partial_sa_jobs_match = re.search(r"selected_partial_sa_jobs: (\[.*\])", stdout)
    assert partial_sa_jobs_match is not None
    partial_sa_jobs = ast.literal_eval(partial_sa_jobs_match.group(1))
    assert len(partial_sa_jobs) == 3
    assert len(set(partial_sa_jobs)) == 3

    selected_first_stage_jobs_match = re.search(
        r"selected_first_stage_jobs: (\[.*\])",
        stdout,
    )
    assert selected_first_stage_jobs_match is not None
    selected_first_stage_jobs = ast.literal_eval(
        selected_first_stage_jobs_match.group(1)
    )
    assert len(selected_first_stage_jobs) == 3
    assert [
        job_id for job_id, _mc_id, _start_time in selected_first_stage_jobs
    ] == partial_sa_jobs

    assert len(partial_sa_operation_set) == 9
    assert {job_id for job_id, _stage_id, _mc_id in partial_sa_operation_set} == set(
        partial_sa_jobs
    )

    makespan_matches = dict(
        re.findall(r"makespan\[(.+?)\]=(\d+)", stdout),
    )
    assert set(makespan_matches) == {
        "base",
        "right_justified_partial",
        "right_justified_full",
        "semi_active_after_rj_partial",
        "semi_active_after_rj_full",
    }
    assert makespan_matches["right_justified_full"] == makespan_matches["base"]

    for file_name in expected_files:
        assert (out_dir / file_name).stat().st_size > 0
