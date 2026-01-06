import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis_metadata import AnalysisMetadata


class ReactiveLoopAnalyzer:
    METADATA = AnalysisMetadata(
        name="Calop1 PRA 600s - 20251119-6",
        result_dir_path_str="$HFS_RESULT_DIR/Calop1/20251121T020724_678708/pra_600s/20251119-6/",
    )

    def __init__(self, metadata: AnalysisMetadata | None = None):
        self.metadata = metadata or self.METADATA
        self.analysis_root = self.get_analysis_dir_path(
            self.metadata.result_dir_path_str
        )

    # -------- Path helper --------
    @classmethod
    def get_analysis_dir_path(cls, result_dir_path_str: str) -> Path:
        expanded = os.path.expandvars(result_dir_path_str)
        return Path(expanded).expanduser()

    # -------- Core loaders --------
    def collect_rho_sequence_for_instance(
        self, instance_dir: Path
    ) -> pd.DataFrame | None:
        """
        instance_dir: e.g., .../20251120T020842_570616/1
        returns: pandas.DataFrame with columns:
            iterCount, rho, timelimit, subroutineName, isImproved
        or None if the file does not exist.
        """
        report_path = instance_dir / self.metadata.reactive_loop_report_rel_path
        if not report_path.exists():
            print(f"[INFO] No reactive loop report for {instance_dir}")
            return None

        try:
            df = pd.read_csv(report_path)
        except Exception as e:
            print(f"[WARN] Failed to read {report_path}: {e}")
            return None

        self.metadata.assert_reactive_loop_report_columns(set(df.columns))

        # iteration 순서대로 정렬
        df = df.sort_values("iterCount").reset_index(drop=True)
        return df

    def collect_all_instances_rho_sequences(
        self, root_dir: Path
    ) -> dict[int, pd.DataFrame]:
        """
        Traverses instance folders under root_dir, returns a dict:
            { instance_id (int): DataFrame }
        """
        rho_data: dict[int, pd.DataFrame] = {}

        for child in sorted(root_dir.iterdir()):
            if not child.is_dir():
                continue
            # Folder name must be an integer for benchmark instance ID
            if not child.name.isdigit():
                continue

            instance_id = int(child.name)
            df = self.collect_rho_sequence_for_instance(child)
            if df is not None:
                rho_data[instance_id] = df

        return rho_data

    # -------- A1: per-operator rho trajectories (builder) --------
    @staticmethod
    def build_per_operator_rho_trajectories_per_iter(
        df: pd.DataFrame, call_index_per_op: bool = True
    ) -> pd.DataFrame:
        """
        df: reactive loop report for a single instance
        반환: long-format DataFrame, columns:
            subroutineName, op_call_index/global_index, iterCount, rho, timelimit, isImproved
        """
        cols = ["iterCount", "rho", "timelimit", "subroutineName", "isImproved"]
        df = df[cols].copy()

        if not call_index_per_op:
            df["global_index"] = df.index
            return df[
                [
                    "subroutineName",
                    "global_index",
                    "iterCount",
                    "rho",
                    "timelimit",
                    "isImproved",
                ]
            ]

        pieces: list[pd.DataFrame] = []
        for op_name, g in df.groupby("subroutineName", sort=False):
            g = g.sort_values("iterCount").reset_index(drop=True)
            g["op_call_index"] = range(1, len(g) + 1)
            g["subroutineName"] = op_name  # group key
            pieces.append(g)

        if not pieces:
            return pd.DataFrame(
                columns=[
                    "subroutineName",
                    "op_call_index",
                    "iterCount",
                    "rho",
                    "timelimit",
                    "isImproved",
                ]
            )

        long_df = pd.concat(pieces, ignore_index=True)
        return long_df[
            [
                "subroutineName",
                "op_call_index",
                "iterCount",
                "rho",
                "timelimit",
                "isImproved",
            ]
        ]

    # -------- A1: per-operator rho over time (builder) --------
    @staticmethod
    def build_per_operator_parameter_over_time(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "iterCount",
            "rho",
            "timelimit",
            "subroutineName",
            "isImproved",
            "timeStart",
            "timeElapsed",
        ]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in reactive loop report: {missing}")

        d = df[required_cols].copy()
        d["timeEnd"] = d["timeStart"] + d["timeElapsed"]
        d = d.sort_values("iterCount").reset_index(drop=True)

        pieces: list[pd.DataFrame] = []
        for op_name, g in d.groupby("subroutineName", sort=False):
            g = g.sort_values("timeEnd").reset_index(drop=True)
            g["subroutineName"] = op_name
            pieces.append(g)

        if not pieces:
            return pd.DataFrame(
                columns=[
                    "subroutineName",
                    "timeEnd",
                    "iterCount",
                    "rho",
                    "timelimit",
                    "isImproved",
                ]
            )

        long_df = pd.concat(pieces, ignore_index=True)
        return long_df[
            [
                "subroutineName",
                "timeEnd",
                "iterCount",
                "rho",
                "timelimit",
                "isImproved",
            ]
        ]

    # -------- A1 / 공통 plot helper --------
    @classmethod
    def _plot_grouped_step(
        cls,
        long_df: pd.DataFrame,
        group_col: str,
        x_col: str,
        y_col: str,
        instance_id: int,
        xlabel: str,
        ylabel: str,
        title: str,
        save_path: Path | None = None,
        step_where: str = "post",
    ) -> None:
        """
        operator별 step-plot을 공통 처리하는 helper.
        """
        if long_df.empty:
            print(f"[INFO] Empty dataframe for {title}, nothing to plot.")
            return

        fig, ax = plt.subplots()

        for name, g in long_df.groupby(group_col):
            g = g.sort_values(x_col)
            ax.step(
                g[x_col],
                g[y_col],
                where=step_where,
                label=name,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (instance {instance_id})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[INFO] Saved plot to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    # -------- A1: plotting (중복 제거된 wrapper들) --------
    @classmethod
    def plot_rho_evolution_over_iter_for_instance(
        cls,
        long_df: pd.DataFrame,
        instance_id: int,
        call_index_colname: str = "op_call_index",
        save_path: Path | None = None,
    ):
        cls._plot_grouped_step(
            long_df=long_df,
            group_col="subroutineName",
            x_col=call_index_colname,
            y_col="rho",
            instance_id=instance_id,
            xlabel="Operator call index",
            ylabel="Neighborhood size parameter $\\rho$",
            title="Evolution of $\\rho$ per operator",
            save_path=save_path,
        )

    @classmethod
    def plot_rho_evolution_over_time_for_instance(
        cls,
        long_df: pd.DataFrame,
        instance_id: int,
        save_path: Path | None = None,
    ):
        cls._plot_grouped_step(
            long_df=long_df,
            group_col="subroutineName",
            x_col="timeEnd",
            y_col="rho",
            instance_id=instance_id,
            xlabel="Time since start (s)",
            ylabel="Neighborhood size parameter $\\rho$",
            title="Evolution of $\\rho$ over time",
            save_path=save_path,
        )

    @classmethod
    def plot_timelimit_evolution_over_time_for_instance(
        cls,
        long_df: pd.DataFrame,
        instance_id: int,
        save_path: Path | None = None,
    ):
        cls._plot_grouped_step(
            long_df=long_df,
            group_col="subroutineName",
            x_col="timeEnd",
            y_col="timelimit",
            instance_id=instance_id,
            xlabel="Time since start (s)",
            ylabel="Timelimit",
            title="Evolution of Timelimit over time",
            save_path=save_path,
        )

    @classmethod
    def plot_rho_evolution_over_timelimit_for_instance(
        cls,
        long_df: pd.DataFrame,
        instance_id: int,
        save_path: Path | None = None,
    ):
        cls._plot_grouped_step(
            long_df=long_df,
            group_col="subroutineName",
            x_col="timelimit",
            y_col="rho",
            instance_id=instance_id,
            xlabel="Subproblem time limit (s)",
            ylabel="Neighborhood size parameter $\\rho$",
            title="Evolution of $\\rho$ vs time limit",
            save_path=save_path,
            step_where="post",  # 필요 시 변경 가능
        )

    # -------- Objective trajectory builders --------
    @staticmethod
    def build_objective_trajectory_over_iterations(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["iterCount", "prevObjValue", "objValue", "isImproved"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in reactive loop report: {missing}")

        d = df[required_cols].copy()
        d = d.sort_values("iterCount").reset_index(drop=True)

        best = d.loc[0, "prevObjValue"]
        best_list: list[float] = []

        for _, row in d.iterrows():
            curr_obj = row["objValue"]
            best = min(best, curr_obj)
            best_list.append(best)

        d["bestObj"] = best_list
        return d[["iterCount", "objValue", "bestObj", "isImproved"]]

    @staticmethod
    def build_objective_trajectory_over_time(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "iterCount",
            "prevObjValue",
            "objValue",
            "timeStart",
            "timeElapsed",
            "isImproved",
        ]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in reactive loop report: {missing}"
            )

        d = df[required_cols].copy()
        d["timeEnd"] = d["timeStart"] + d["timeElapsed"]
        d = d.sort_values("timeEnd").reset_index(drop=True)

        best = d.loc[0, "prevObjValue"]
        best_list: list[float] = []

        for _, row in d.iterrows():
            curr_obj = row["objValue"]
            best = min(best, curr_obj)
            best_list.append(best)

        d["bestObj"] = best_list
        return d[["timeEnd", "objValue", "bestObj", "iterCount", "isImproved"]]

    # -------- Objective plot 공통 helper --------
    @classmethod
    def _plot_best_objective_trajectory(
        cls,
        traj_df: pd.DataFrame,
        instance_id: int,
        x_col: str,
        xlabel: str,
        title_prefix: str,
        save_path: Path | None = None,
    ) -> None:
        if traj_df.empty:
            print("[INFO] Empty objective trajectory dataframe, nothing to plot.")
            return

        fig, ax = plt.subplots()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.step(
            traj_df[x_col],
            traj_df["bestObj"],
            where="post",
            label="Best-so-far objective",
        )

        improved = traj_df[traj_df["isImproved"] == True]
        if not improved.empty:
            ax.scatter(
                improved[x_col],
                improved["bestObj"],
                marker="o",
                s=30,
                zorder=3,
                label="Improving iterations",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Best-so-far objective")
        ax.set_title(f"{title_prefix} (instance {instance_id})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[INFO] Saved objective trajectory plot to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    @classmethod
    def plot_objective_trajectory_over_iterations_for_instance(
        cls,
        traj_df: pd.DataFrame,
        instance_id: int,
        save_path: Path | None = None,
    ):
        cls._plot_best_objective_trajectory(
            traj_df=traj_df,
            instance_id=instance_id,
            x_col="iterCount",
            xlabel="ALNS iteration (iterCount)",
            title_prefix="Objective improvement trajectory",
            save_path=save_path,
        )

    @classmethod
    def plot_objective_trajectory_over_time_for_instance(
        cls,
        traj_df: pd.DataFrame,
        instance_id: int,
        save_path: Path | None = None,
    ):
        cls._plot_best_objective_trajectory(
            traj_df=traj_df,
            instance_id=instance_id,
            x_col="timeEnd",
            xlabel="Time since start (s)",
            title_prefix="Objective improvement over time",
            save_path=save_path,
        )


def main():
    analyzer = ReactiveLoopAnalyzer()

    analysis_root = analyzer.analysis_root
    print(f"Target result directory: {analysis_root}")

    rho_by_instance = analyzer.collect_all_instances_rho_sequences(analysis_root)
    print(f"Loaded reactive-loop rho logs for {len(rho_by_instance)} instances.")

    target_instance_id = 1440
    if target_instance_id not in rho_by_instance:
        print(
            f"Instance {target_instance_id} not found or has no reactive loop report."
        )
        return

    df = rho_by_instance[target_instance_id]

    # 시간별 per-operator parameter dataframe
    parameter_df = analyzer.build_per_operator_parameter_over_time(df)

    # timelimit over time plot
    timelimit_evolution_over_time_fig_path = (
        analysis_root
        / str(target_instance_id)
        / f"{target_instance_id:04d}_timelimit_evolution_over_time.png"
    )
    ReactiveLoopAnalyzer.plot_timelimit_evolution_over_time_for_instance(
        parameter_df,
        target_instance_id,
        save_path=timelimit_evolution_over_time_fig_path,
    )

    # objective over time trajectory + plot
    objective_df = analyzer.build_objective_trajectory_over_time(df)
    out_path = (
        analysis_root
        / str(target_instance_id)
        / f"{target_instance_id:04d}_objective_over_time.png"
    )
    ReactiveLoopAnalyzer.plot_objective_trajectory_over_time_for_instance(
        objective_df, target_instance_id, save_path=out_path
    )


if __name__ == "__main__":
    main()
