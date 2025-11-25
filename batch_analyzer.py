from pathlib import Path

import pandas as pd

from analysis_metadata import AnalysisMetadata

METADATA = AnalysisMetadata(
    name="Calop1 PRA 600s - 20251119-6",
    result_dir_path_str="$HFS_RESULT_DIR/Calop1/20251121T020724_678708/pra_600s/20251119-6/",
)


def load_reactive_loop_report(instance_dir: Path) -> pd.DataFrame | None:
    report_path = instance_dir / METADATA.reactive_loop_report_rel_path
    if not report_path.exists():
        print(f"[INFO] No reactive loop report for {instance_dir}")
        return None

    try:
        df = pd.read_csv(report_path)
    except Exception as e:
        print(f"[WARN] Failed to read {report_path}: {e}")
        return None

    required = {"subroutineName", "timeElapsed", "isImproved", "timelimit", "rho"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Missing required columns {missing} in file: {report_path}")
        return None

    return df


def build_lns_operator_efficiency_table(root_dir: Path) -> pd.DataFrame:
    """
    instance별 LNS operator efficiency + 7개 지표:

        *_call_count
        *_avg_timelimit
        *_avg_rho
        *_improve_count
        *_avg_rho_improved      (isImproved=True인 row들에서 rho 평균, 없으면 NaN)
        *_time_spent
        *_improve_per_call_count (improve_count / call_count)
        *_improve_per_time_spent (improve_count / time_spent)
    """
    operators = ["operation_block_ns", "job_block_ns", "stage_block_ns"]
    rows = []

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue

        instance_id = int(child.name)
        df = load_reactive_loop_report(child)

        # 기본값 초기화
        stats = {}
        for op in operators:
            p = op.replace("_ns", "")  # e.g., "operation_block"
            stats[f"{p}_call_count"] = 0
            stats[f"{p}_avg_timelimit"] = 0.0
            stats[f"{p}_avg_rho"] = 0.0
            stats[f"{p}_improve_count"] = 0
            stats[f"{p}_avg_rho_improved"] = float("nan")
            stats[f"{p}_time_spent"] = 0.0
            stats[f"{p}_improve_per_call_count"] = 0.0
            stats[f"{p}_improve_per_time_spent"] = 0.0

        if df is not None:
            df["timeElapsed"] = df["timeElapsed"].fillna(0.0)
            df["timelimit"] = df["timelimit"].fillna(0.0)
            df["rho"] = df["rho"].fillna(0.0)

            for op in operators:
                p = op.replace("_ns", "")

                g = df[df["subroutineName"] == op]

                if not g.empty:
                    call_count = len(g)
                    avg_timelimit = g["timelimit"].mean()
                    avg_rho = g["rho"].mean()

                    improved_mask = (g["isImproved"] == True)
                    improve_count = int(improved_mask.sum())

                    if improve_count > 0:
                        avg_rho_improved = g.loc[improved_mask, "rho"].mean()
                    else:
                        avg_rho_improved = float("nan")

                    time_spent = float(g["timeElapsed"].sum())
                    improve_per_call_count = (
                        improve_count / call_count if call_count > 0 else 0.0
                    )
                    improve_per_time_spent = (
                        improve_count / time_spent if time_spent > 0 else 0.0
                    )
                else:
                    call_count = 0
                    avg_timelimit = 0.0
                    avg_rho = 0.0
                    improve_count = 0
                    avg_rho_improved = float("nan")
                    time_spent = 0.0
                    improve_per_call_count = 0.0
                    improve_per_time_spent = 0.0

                stats[f"{p}_call_count"] = call_count
                stats[f"{p}_avg_timelimit"] = avg_timelimit
                stats[f"{p}_avg_rho"] = avg_rho
                stats[f"{p}_improve_count"] = improve_count
                stats[f"{p}_avg_rho_improved"] = avg_rho_improved
                stats[f"{p}_time_spent"] = time_spent
                stats[f"{p}_improve_per_call_count"] = improve_per_call_count
                stats[f"{p}_improve_per_time_spent"] = improve_per_time_spent

        row = {"instance_id": instance_id}
        row.update(stats)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)


def main():
    root = METADATA.get_analysis_dir_path()
    print(f"Analysis root: {root}")

    df = build_lns_operator_efficiency_table(root)
    # sanity check 출력
    print(df.head())

    out_path = root / "lns_operator_efficiency_summary.csv"
    df.to_csv(out_path, index=False)

    print(f"[INFO] Saved LNS operator summary to {out_path}")


if __name__ == "__main__":
    main()
