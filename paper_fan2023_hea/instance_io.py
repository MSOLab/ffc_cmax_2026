from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)


@dataclass(frozen=True)
class BaselineRecord:
    instance: int
    ub: float
    lb: float | None = None


def load_ff2020_instance(file_path: Path) -> HybridFlowshopParameters:
    """Load an FF2020-format instance using the same parser as the main repo flow."""

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return HybridFlowshopParameters.from_ff2020_data(file_path.stem, f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark file not found: {file_path}") from None
    except Exception as exc:
        raise RuntimeError(f"Error reading benchmark file {file_path}: {exc}") from exc


def load_instances(
    input_dir: Path,
    benchmark_filenames: list[str],
) -> list[HybridFlowshopParameters]:
    return [load_ff2020_instance(input_dir / name) for name in benchmark_filenames]


def load_baseline_records(path: Path) -> dict[int, BaselineRecord]:
    df = pd.read_csv(path)
    required = {"Instance", "UB"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline CSV is missing required columns: {sorted(missing)}")

    records: dict[int, BaselineRecord] = {}
    for row in df.to_dict("records"):
        instance_id = int(row["Instance"])
        lb = float(row["LB"]) if "LB" in row and not pd.isna(row["LB"]) else None
        records[instance_id] = BaselineRecord(
            instance=instance_id,
            ub=float(row["UB"]),
            lb=lb,
        )
    return records
