from pathlib import Path

import matplotlib.pyplot as plt


class ObjectiveProgressPlotter:
    @staticmethod
    def plot_solution_progress(
        progress_log: list[tuple[float, float, float]],
        save_path: Path,
        show_markers: bool = True,
    ):
        """Plot the solution progress over time with optional markers.

        Args:
            progress_log (list[tuple[float, float, float]]): A list of (elapsed time, objective value, best bound).
            save_path (Path): Path to save the plot.
            show_markers (bool, optional): Whether to show markers at each step. Defaults to True.
        """  # noqa: E501
        if not progress_log:
            print("No progress data available to plot.")
            return

        elapsed_times, objectives, best_bounds = zip(*progress_log)

        plt.figure(figsize=(10, 6))

        # Objective Value
        plt.step(
            elapsed_times,
            objectives,
            where="post",
            label="Objective Value",
            color="tab:blue",
        )
        if show_markers:
            plt.scatter(
                elapsed_times,
                objectives,
                color="tab:blue",
                facecolor="black",
                marker="o",
                s=40,
                zorder=3,
                label="Obj Update Points",
            )

        # Best Bound
        plt.step(
            elapsed_times,
            best_bounds,
            where="post",
            linestyle="--",
            label="Best Bound",
            color="tab:orange",
        )
        if show_markers:
            plt.scatter(
                elapsed_times,
                best_bounds,
                color="tab:orange",
                facecolor="black",
                marker="x",
                s=40,
                zorder=3,
                label="Bound Update Points",
            )

        plt.xlabel("Elapsed Time (seconds)")
        plt.ylabel("Objective Value")
        plt.title("Solution Progress Over Time")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.xlim(left=0)

        plt.tight_layout()

        plt.savefig(save_path)
        print(f"Solution progress plot saved to {save_path}")
