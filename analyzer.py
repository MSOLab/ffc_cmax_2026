from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis_metadata import AnalysisMetadata

METADATA = AnalysisMetadata(
    name="Calop1 PRA 600s - 20251119-6",
    result_dir_path_str="$HFS_RESULT_DIR/Calop1/20251121T020724_678708/pra_600s/20251119-6/",
)


def main():
    analysis_root = METADATA.get_analysis_dir_path()
    print(f"Target result directory: {analysis_root}")

    rho_by_instance = collect_all_instances_rho_sequences(analysis_root)
    print(f"Loaded reactive-loop rho logs for {len(rho_by_instance)} instances.")

    target_instance_id = 1
    if target_instance_id not in rho_by_instance:
        print(
            f"Instance {target_instance_id} not found or has no reactive loop report."
        )
        return

    df = rho_by_instance[target_instance_id]

    # # iteration별 per-operator rho trajectory 생성 및 플로팅
    # use_global_index = True
    # long_df = build_per_operator_rho_trajectories_per_iter(
    #     df, call_index_per_op=not use_global_index
    # )
    # print("Per-operator rho trajectories (head):")
    # print(long_df.head(10))
    # rho_evolution_over_iter_fig_path = (
    #     analysis_root
    #     / str(target_instance_id)
    #     / f"{target_instance_id:04d}_rho_evolution_over_iter.png"
    # )
    # if use_global_index:
    #     call_index_colname = "global_index"
    # else:
    #     call_index_colname = "op_call_index"
    # plot_rho_evolution_over_iter_for_instance(
    #     long_df,
    #     target_instance_id,
    #     call_index_colname=call_index_colname,
    #     save_path=rho_evolution_over_iter_fig_path,
    # )

    # # 시간별 per-operator rho trajectory 생성 및 플로팅
    time_long_df = build_per_operator_rho_over_time(df)
    # rho_evolution_over_time_fig_path = (
    #     analysis_root
    #     / str(target_instance_id)
    #     / f"{target_instance_id:04d}_rho_evolution_over_time.png"
    # )
    # plot_rho_evolution_over_time_for_instance(
    #     time_long_df,
    #     target_instance_id,
    #     save_path=rho_evolution_over_time_fig_path,
    # )

    # 시간별 per-operator timelimit trajectory 생성 및 플로팅
    timelimit_evolution_over_time_fig_path = (
        analysis_root
        / str(target_instance_id)
        / f"{target_instance_id:04d}_timelimit_evolution_over_time.png"
    )
    plot_timelimit_evolution_over_time_for_instance(
        time_long_df,
        target_instance_id,
        save_path=timelimit_evolution_over_time_fig_path,
    )

    # # timelimit-rho trajectory 생성 및 플로팅
    # tl_long_df = build_per_operator_rho_over_timelimit(df)
    # # 그림 저장
    # out_path = (
    #     analysis_root
    #     / str(target_instance_id)
    #     / f"{target_instance_id:04d}_rho_over_timelimit.png"
    # )
    # plot_rho_evolution_over_timelimit_for_instance(
    #     tl_long_df, target_instance_id, save_path=out_path
    # )

    # # iteration별 objective trajectory 생성 및 플로팅
    # traj_df = build_objective_trajectory_over_iterations(df)
    # out_path = (
    #     analysis_root
    #     / str(target_instance_id)
    #     / f"{target_instance_id:04d}_objective_trajectory.png"
    # )
    # plot_objective_trajectory_over_iterations_for_instance(
    #     traj_df, target_instance_id, save_path=out_path
    # )

    # 시간별 objective trajectory 생성 및 플로팅
    traj_time_df = build_objective_trajectory_over_time(df)

    out_path = (
        analysis_root
        / str(target_instance_id)
        / f"{target_instance_id:04d}_objective_over_time.png"
    )
    plot_objective_trajectory_over_time_for_instance(
        traj_time_df, target_instance_id, save_path=out_path
    )


# Core loaders


def collect_rho_sequence_for_instance(instance_dir: Path) -> pd.DataFrame | None:
    """
    instance_dir: e.g., .../20251120T020842_570616/1
    returns: pandas.DataFrame with columns:
        iterCount, rho, timelimit, subroutineName, isImproved
    or None if the file does not exist.
    """
    report_path = instance_dir / METADATA.reactive_loop_report_rel_path
    if not report_path.exists():
        print(f"[INFO] No reactive loop report for {instance_dir}")
        return None

    try:
        df = pd.read_csv(report_path)
    except Exception as e:
        print(f"[WARN] Failed to read {report_path}: {e}")
        return None

    METADATA.assert_reactive_loop_report_columns(set(df.columns))

    # iteration 순서대로 정렬
    df = df.sort_values("iterCount").reset_index(drop=True)
    return df


def collect_all_instances_rho_sequences(root_dir: Path):
    """
    Traverses instance folders under root_dir, returns a dict:
        { instance_id (int): DataFrame }
    """
    rho_data = {}

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        # Folder name must be an integer for benchmark instance ID
        if not child.name.isdigit():
            continue

        instance_id = int(child.name)
        df = collect_rho_sequence_for_instance(child)
        if df is not None:
            rho_data[instance_id] = df

    return rho_data


# ---- A1: per-operator rho trajectories ----


def build_per_operator_rho_trajectories_per_iter(
    df: pd.DataFrame, call_index_per_op=True
) -> pd.DataFrame:
    """
    df: reactive loop report for a single instance
    반환: long-format DataFrame, columns:
        subroutineName, op_call_index, iterCount, rho, timelimit, isImproved

    - op_call_index: 해당 operator가 몇 번째 호출인지 (1, 2, 3, ...)
    - iterCount: 전역 reactive-loop iteration index (그룹핑용으로만 사용)
    """
    # 안전하게 필요한 컬럼만 사용
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

    # operator별로 나눠서, 각 operator에 대해 "호출 순서" 인덱스 생성
    pieces: list[pd.DataFrame] = []
    for op_name, g in df.groupby("subroutineName", sort=False):
        g = g.sort_values("iterCount").reset_index(drop=True)

        g["op_call_index"] = range(1, len(g) + 1)
        g["subroutineName"] = op_name  # group key 붙이기
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


# ---- A1: plotting helper (대표 인스턴스용) ----


def plot_rho_evolution_over_iter_for_instance(
    long_df: pd.DataFrame,
    instance_id: int,
    call_index_colname: str = "op_call_index",
    save_path: Path | None = None,
):
    """
    한 인스턴스에 대해 operator별 rho 진화 곡선을 그림.
    x축: op_call_index (해당 operator 입장에서 몇 번째 호출인지)
    y축: rho
    """
    if long_df.empty:
        print("[INFO] Empty rho trajectory dataframe, nothing to plot.")
        return

    fig, ax = plt.subplots()

    for op_name, g in long_df.groupby("subroutineName"):
        g = g.sort_values(call_index_colname)
        ax.step(
            g[call_index_colname],
            g["rho"],
            where="post",
            label=op_name,
        )

    ax.set_xlabel("Operator call index")
    ax.set_ylabel("Neighborhood size parameter $\\rho$")
    ax.set_title(f"Evolution of $\\rho$ per operator (instance {instance_id})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved rho evolution plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def build_per_operator_rho_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reactive-loop report(df)에서 각 subroutineName별로
    종료시각(timeStart + timeElapsed) 기준 rho 시계열을 만든다.

    반환 컬럼:
        subroutineName, timeEnd, iterCount, rho, timelimit, isImproved
    """
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

    # 종료시각 = timeStart + timeElapsed
    d["timeEnd"] = d["timeStart"] + d["timeElapsed"]

    # iterCount 기준으로 먼저 정렬
    d = d.sort_values("iterCount").reset_index(drop=True)

    # operator별로 timeEnd 기준 정렬 (사실상 iterCount와 동일 순서일 것)
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


def plot_rho_evolution_over_time_for_instance(
    long_df: pd.DataFrame,
    instance_id: int,
    save_path: Path | None = None,
):
    """
    build_per_operator_rho_over_time() 결과(long_df)를 받아
    x축 = timeEnd, y축 = rho 로 operator별 step plot을 그림.
    """
    if long_df.empty:
        print("[INFO] Empty rho-over-time dataframe, nothing to plot.")
        return

    fig, ax = plt.subplots()

    for op_name, g in long_df.groupby("subroutineName"):
        g = g.sort_values("timeEnd")
        ax.step(
            g["timeEnd"],
            g["rho"],
            where="post",
            label=op_name,
        )

    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Neighborhood size parameter $\\rho$")
    ax.set_title(f"Evolution of $\\rho$ over time (instance {instance_id})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved rho-over-time plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_timelimit_evolution_over_time_for_instance(
    long_df: pd.DataFrame,
    instance_id: int,
    save_path: Path | None = None,
):
    """
    build_per_operator_rho_over_time() 결과(long_df)를 받아
    x축 = timeEnd, y축 = timelimit 로 operator별 step plot을 그림.
    """
    if long_df.empty:
        print("[INFO] Empty timelimit-over-time dataframe, nothing to plot.")
        return

    fig, ax = plt.subplots()

    for op_name, g in long_df.groupby("subroutineName"):
        g = g.sort_values("timeEnd")
        ax.step(
            g["timeEnd"],
            g["timelimit"],
            where="post",
            label=op_name,
        )

    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Timelimit")
    ax.set_title(f"Evolution of Timelimit over time (instance {instance_id})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved timelimit-over-time plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def build_per_operator_rho_over_timelimit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reactive-loop report(df)에서 각 subroutineName별로
    timelimit 기준 rho 진화 시계열을 생성한다.

    반환 컬럼:
        subroutineName, timelimit, iterCount, rho, timeStart, timeElapsed, isImproved
    """
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

    # 먼저 iterCount 기준 정렬
    d = d.sort_values("iterCount").reset_index(drop=True)

    # operator별로 timelimit 기준 step plotting을 위해 정렬
    pieces: list[pd.DataFrame] = []
    for op_name, g in d.groupby("subroutineName", sort=False):
        g = g.sort_values("iterCount").reset_index(drop=True)
        g["subroutineName"] = op_name
        pieces.append(g)

    if not pieces:
        return pd.DataFrame(
            columns=[
                "subroutineName",
                "timelimit",
                "iterCount",
                "rho",
                "timeStart",
                "timeElapsed",
                "isImproved",
            ]
        )

    long_df = pd.concat(pieces, ignore_index=True)

    return long_df[
        [
            "subroutineName",
            "timelimit",
            "iterCount",
            "rho",
            "timeStart",
            "timeElapsed",
            "isImproved",
        ]
    ]


def plot_rho_evolution_over_timelimit_for_instance(
    long_df: pd.DataFrame,
    instance_id: int,
    save_path: Path | None = None,
):
    """
    x축: timelimit (초)
    y축: rho (neighborhood size)
    operator별 step-curve로 그린다.
    """
    if long_df.empty:
        print("[INFO] Empty timelimit-rho dataframe, nothing to plot.")
        return

    fig, ax = plt.subplots()

    for op_name, g in long_df.groupby("subroutineName"):
        # g = g.sort_values("timelimit")
        ax.step(
            g["timelimit"],
            g["rho"],
            # where="post",
            label=op_name,
        )

    ax.set_xlabel("Subproblem time limit (s)")
    ax.set_ylabel("Neighborhood size parameter $\\rho$")
    ax.set_title(f"Evolution of $\\rho$ vs time limit (instance {instance_id})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved timelimit–rho plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def build_objective_trajectory_over_iterations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reactive-loop report(df)에서 iteration별 best-so-far objective trajectory를 만든다.

    입력 df에는 최소한 다음 컬럼이 필요:
        iterCount, prevObjValue, objValue, isImproved

    반환 DF 컬럼:
        iterCount      : iteration index (정렬됨)
        objValue       : 해당 iteration에서 solver가 보고한 objective
        bestObj        : 그 시점까지의 best-so-far objective (누적 min)
        isImproved     : 해당 iteration 호출에서 개선 발생 여부
    """
    required_cols = ["iterCount", "prevObjValue", "objValue", "isImproved"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in reactive loop report: {missing}")

    d = df[required_cols].copy()
    d = d.sort_values("iterCount").reset_index(drop=True)

    # 시작 시점의 best objective: 첫 iteration의 prevObjValue
    best = d.loc[0, "prevObjValue"]
    best_list: list[float] = []

    for _, row in d.iterrows():
        # iteration 결과 objective
        curr_obj = row["objValue"]
        # best-so-far 갱신
        best = min(best, curr_obj)
        best_list.append(best)

    d["bestObj"] = best_list
    return d[["iterCount", "objValue", "bestObj", "isImproved"]]


def plot_objective_trajectory_over_iterations_for_instance(
    traj_df: pd.DataFrame,
    instance_id: int,
    save_path: Path | None = None,
):
    """
    build_objective_trajectory_over_iterations() 결과(traj_df)를 받아
    iteration별 best-so-far objective trajectory를 플로팅한다.

    - x축: iterCount
    - y축: bestObj (step plot)
    - isImproved=True인 지점에 marker 표시
    """
    if traj_df.empty:
        print("[INFO] Empty objective trajectory dataframe, nothing to plot.")
        return

    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # step plot: best-so-far objective
    ax.step(
        traj_df["iterCount"],
        traj_df["bestObj"],
        where="post",
        label="Best-so-far objective",
    )

    # 개선이 일어난 iteration에 마커 찍기
    improved = traj_df[traj_df["isImproved"] == True]
    if not improved.empty:
        ax.scatter(
            improved["iterCount"],
            improved["bestObj"],
            marker="o",
            s=30,
            zorder=3,
            label="Improving iterations",
        )

    ax.set_xlabel("ALNS iteration (iterCount)")
    ax.set_ylabel("Best-so-far objective")
    ax.set_title(f"Objective improvement trajectory (instance {instance_id})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved objective trajectory plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def build_objective_trajectory_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reactive-loop report(df)에서 timeEnd(=timeStart+timeElapsed) 기준
    best-so-far objective trajectory를 만든다.

    입력 df에는 다음 컬럼 필요:
        iterCount, prevObjValue, objValue, timeStart, timeElapsed, isImproved

    반환 DF 컬럼:
        timeEnd      : 해당 iteration 종료 시간(초)
        objValue     : 그 iteration에서 solver가 보고한 objective
        bestObj      : 그 시점까지의 best-so-far objective
        iterCount
        isImproved
    """
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
        raise ValueError(f"Missing required columns in reactive loop report: {missing}")

    d = df[required_cols].copy()

    # 종료 시각 계산
    d["timeEnd"] = d["timeStart"] + d["timeElapsed"]

    # timeEnd 기준 정렬 (iteration 순서와 유사하지만 더 정확히 시간 순서로)
    d = d.sort_values("timeEnd").reset_index(drop=True)

    # 처음 best objective = 첫 prevObjValue
    best = d.loc[0, "prevObjValue"]
    best_list = []

    for _, row in d.iterrows():
        curr_obj = row["objValue"]
        best = min(best, curr_obj)
        best_list.append(best)

    d["bestObj"] = best_list

    return d[["timeEnd", "objValue", "bestObj", "iterCount", "isImproved"]]


def plot_objective_trajectory_over_time_for_instance(
    traj_df: pd.DataFrame,
    instance_id: int,
    save_path: Path | None = None,
):
    """
    build_objective_trajectory_over_time() 결과(traj_df)를 받아
    x축 = timeEnd, y축 = bestObj 로 step plot을 생성한다.

    개선(iter) 지점은 marker로 표시한다.
    """
    if traj_df.empty:
        print("[INFO] Empty objective-over-time dataframe, nothing to plot.")
        return

    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # step plot: best-so-far objective vs timeEnd
    ax.step(
        traj_df["timeEnd"],
        traj_df["bestObj"],
        where="post",
        label="Best-so-far objective",
    )

    # improvement 이벤트 표시
    improved = traj_df[traj_df["isImproved"] == True]
    if not improved.empty:
        ax.scatter(
            improved["timeEnd"],
            improved["bestObj"],
            marker="o",
            s=30,
            zorder=3,
            label="Improving iterations",
        )

    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Best-so-far objective")
    ax.set_title(f"Objective improvement over time (instance {instance_id})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved objective-over-time plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
