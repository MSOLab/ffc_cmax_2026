from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PaperRunConfig(BaseModel):
    """Run configuration for the standalone Fan 2023 HEA track."""

    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    baseline_csv_path: Path
    benchmark_idx_list: list[int] | None = None
    seed_count: int = 10
    output_dir: Path = Path("Outputs_paper_fan2023")
    draw_convergence: bool = False
    random_seed_base: int = 1

    scenario: str = "fan2023_hea"
    benchmark_filename_format: str = "{}.txt"
    benchmark_idx_start: int | None = None
    benchmark_idx_end: int | None = None
    time_limit_scale: float = 1.0
    max_iterations: int | None = None
    plot_instance_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_benchmark_selection(self) -> PaperRunConfig:
        has_list = bool(self.benchmark_idx_list)
        has_range = (
            self.benchmark_idx_start is not None or self.benchmark_idx_end is not None
        )
        if not has_list and not has_range:
            raise ValueError(
                "Provide benchmark_idx_list or benchmark_idx_start/benchmark_idx_end."
            )
        if has_range and (
            self.benchmark_idx_start is None or self.benchmark_idx_end is None
        ):
            raise ValueError(
                "benchmark_idx_start and benchmark_idx_end must be provided together."
            )
        if self.seed_count <= 0:
            raise ValueError("seed_count must be positive.")
        if self.time_limit_scale <= 0:
            raise ValueError("time_limit_scale must be positive.")
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive when provided.")
        return self

    @property
    def benchmark_indices(self) -> list[int]:
        if self.benchmark_idx_list:
            return list(self.benchmark_idx_list)
        assert self.benchmark_idx_start is not None
        assert self.benchmark_idx_end is not None
        return list(range(self.benchmark_idx_start, self.benchmark_idx_end + 1))

    def benchmark_filenames(self) -> list[str]:
        return [
            self.benchmark_filename_format.format(idx) for idx in self.benchmark_indices
        ]


def load_config(path: Path) -> PaperRunConfig:
    with path.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f) or {}
    return PaperRunConfig.model_validate(raw)
