from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from routix.io.yaml import dump_yaml

from .search import RetainedStageCpResult
from .shared import RetainedStageCpBuild


def extract_retained_stage_solution_rows(
    solver: Any,
    build: RetainedStageCpBuild,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_id in build.retained_stage_ids:
        for job_id in build.params.j_list:
            rows.append(
                {
                    "stage_id": stage_id,
                    "job_id": job_id,
                    "start": int(
                        solver.Value(build.variables.op_start[job_id, stage_id])
                    ),
                    "end": int(solver.Value(build.variables.op_end[job_id, stage_id])),
                    "processing_time": build.params.p[job_id, stage_id],
                    "head": build.head_by_job_stage[job_id, stage_id],
                    "tail": build.tail_by_job_stage[job_id, stage_id],
                }
            )
    return rows


def write_retained_stage_cp_artifacts(
    output_dir: Path,
    *,
    result: RetainedStageCpResult,
    build: RetainedStageCpBuild,
    trace_rows: Sequence[Mapping[str, Any]],
    retained_solution_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = result.to_dict()
    summary_payload["retained_stage_indices"] = list(build.retained_stage_indices)
    dump_yaml(summary_payload, output_dir / "summary.yaml", encoding="utf-8")

    metadata_payload = {
        **result.to_dict(),
        "job_ids": list(build.params.j_list),
        "stage_ids": list(build.params.i_list),
        "retained_stage_indices": list(build.retained_stage_indices),
        "compressed_lags": [
            {
                "from_stage_id": left_stage_id,
                "to_stage_id": right_stage_id,
                "job_id": job_id,
                "value": value,
            }
            for (left_stage_id, right_stage_id, job_id), value in sorted(
                build.compressed_lag_by_pair_job.items()
            )
        ],
        "santos_rhs_by_stage": dict(build.santos_rhs_by_stage),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    _write_csv_rows(
        output_dir / "trace.csv",
        ("runtime_sec", "objective_ub", "objective_lb"),
        trace_rows,
    )
    _write_csv_rows(
        output_dir / "retained_solution.csv",
        ("stage_id", "job_id", "start", "end", "processing_time", "head", "tail"),
        retained_solution_rows,
    )


def _write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
