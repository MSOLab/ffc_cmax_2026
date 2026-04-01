from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_METHOD_RPDF_COLUMNS = {"subroutine_name", "norm_time", "rpd_f"}


def _build_method_rpdf_scatter_df(metrics_long_df: pd.DataFrame) -> pd.DataFrame:
    """Build plotting rows from complete-case observations only.

    Only rows with non-null `subroutine_name`, `norm_time`, and `rpd_f`
    contribute to the grouped means. This intentionally computes both axes
    from the same valid observation set for each subroutine.
    """
    missing_cols = REQUIRED_METHOD_RPDF_COLUMNS - set(metrics_long_df.columns)
    if missing_cols:
        raise ValueError(
            "Missing required columns for method RPD scatter chart: "
            f"{sorted(missing_cols)}"
        )

    clean_df = metrics_long_df.dropna(subset=["norm_time", "rpd_f"])

    return (
        clean_df.groupby("subroutine_name", sort=False)
        .agg(
            mean_norm_time=("norm_time", "mean"),
            mean_rpd_f=("rpd_f", "mean"),
        )
        .reset_index()
    )


def export_method_rpdf_scatter_svg(
    metrics_long_df: pd.DataFrame, output_path: Path
) -> bool:
    plot_df = _build_method_rpdf_scatter_df(metrics_long_df)
    if plot_df.empty:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            ax.plot(
                plot_df["mean_norm_time"],
                plot_df["mean_rpd_f"],
                marker="o",
                linewidth=1.5,
            )

            for _, row in plot_df.iterrows():
                ax.annotate(
                    str(row["subroutine_name"]),
                    (row["mean_norm_time"], row["mean_rpd_f"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                )

            ax.set_xlabel("Mean normalized time")
            ax.set_ylabel("Mean RPDf")
            ax.set_title("Subroutine mean norm_time vs mean RPDf")
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.grid(True, linestyle="--", alpha=0.4)

            # Set both axes origin to 0
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

            fig.savefig(output_path, format="svg", bbox_inches="tight")
        finally:
            plt.close(fig)

    logging.info(f"Method RPD scatter SVG saved to {output_path}")
    return True
