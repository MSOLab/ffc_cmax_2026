from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SeedRunRecord:
    name: str
    insName: str
    scenario: str
    seed: int
    bestOriginTag: str
    bestSeed: int
    bestObj: int
    refUb: float | None
    rpdv: float | None
    rpdf: float | None
    runtimeSeconds: float


@dataclass(frozen=True)
class SummaryRecord:
    name: str
    insName: str
    scenario: str
    bestOriginTag: str
    bestSeed: int
    bestObj: int
    refUb: float | None
    rpdv: float | None
    rpdf: float | None
    seedCount: int


def rpdv(obj: float, ref: float | None) -> float | None:
    if ref is None or ref == 0:
        return None
    return (obj - ref) / ref


def rpdf(obj: float, ref: float | None) -> float | None:
    if ref is None:
        return None
    denominator = (obj + ref) / 2
    if denominator == 0:
        return 0.0 if obj == 0 and ref == 0 else None
    return (obj - ref) / denominator


def write_seed_runs(path: Path, records: Iterable[SeedRunRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(record) for record in records])
    df.to_csv(path, index=False)
    return df


def write_summary(path: Path, records: Iterable[SummaryRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(record) for record in records])
    df.to_csv(path, index=False)
    return df


def write_baseline_comparison(path: Path, summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col
        for col in [
            "name",
            "insName",
            "scenario",
            "bestObj",
            "refUb",
            "rpdv",
            "rpdf",
            "bestOriginTag",
            "bestSeed",
        ]
        if col in summary_df.columns
    ]
    comparison_df = summary_df.loc[:, cols]
    comparison_df.to_csv(path, index=False)
    return comparison_df


def write_convergence_plot(
    output_path: Path,
    history_rows: list[dict[str, object]],
) -> None:
    if not history_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        csv_fallback = output_path.with_suffix(".csv")
        pd.DataFrame(history_rows).to_csv(csv_fallback, index=False)
        return

    df = pd.DataFrame(history_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed, group in df.groupby("seed"):
        ax.step(
            group["elapsedSeconds"],
            group["bestObj"],
            where="post",
            label=f"seed {seed}",
        )
    ax.set_xlabel("elapsed seconds")
    ax.set_ylabel("best makespan")
    ax.set_title("Fan 2023 HEA convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
