from pathlib import Path
from typing import List, Union

import pandas as pd
import yaml


def create_obj_value_dataframe(
    base_path: Path, timestamps: List[Union[int, float]], encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    주어진 경로와 시간 목록을 기반으로 각 인스턴스의 목적 함수 값 데이터프레임을 생성합니다.

    Args:
        base_path (Path): 인스턴스 디렉터리(예: '1', '2', ...)를 포함하는 기본 경로.
        timestamps (List[Union[int, float]]): 목적 함수 값을 확인할 시간(초) 목록.
        encoding (str, optional): 파일 읽기 시 사용할 인코딩. 기본값은 'utf-8'입니다.

    Returns:
        pd.DataFrame: 행은 인스턴스 ID, 열은 타임스탬프이며, 각 셀은 해당 시점의
                      목적 함수 값을 가집니다.
    """
    results_data = {}

    # 타임스탬프 목록을 오름차순으로 정렬
    timestamps.sort()

    # base_path 아래의 모든 디렉터리를 순회
    for instance_dir in sorted(base_path.iterdir()):
        # 디렉터리 이름이 숫자인 경우에만 처리 (인스턴스 디렉터리로 간주)
        if instance_dir.is_dir() and instance_dir.name.isdigit():
            instance_id = int(instance_dir.name)
            log_file_name = f"{instance_id}_obj_log.yaml"
            log_file_path = instance_dir / "results" / log_file_name

            if not log_file_path.exists():
                print(f"경고: {log_file_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue

            with open(log_file_path, "r", encoding=encoding) as f:
                log_data = yaml.safe_load(f)

            # 'obj_value' 및 'data' 키가 있는지 확인하고, 없으면 빈 딕셔너리 사용
            obj_value_dict = log_data.get("obj_value", {}).get("data", {})

            if not obj_value_dict:
                # 로그 데이터가 없는 경우, 모든 타임스탬프에 대해 None으로 채움
                results_data[instance_id] = [None] * len(timestamps)
                continue

            # 로그의 타임스탬프(str)를 float으로 변환하고 시간순으로 정렬
            # (시간, 값) 튜플의 리스트 생성
            sorted_log = sorted([(float(t), v) for t, v in obj_value_dict.items()])

            row_values = []
            log_idx = 0

            # 각 요청된 타임스탬프에 대해 값을 찾음
            for target_time in timestamps:
                last_known_value = None
                # 정렬된 로그를 순회하며 target_time 이하의 마지막 값을 찾음
                for log_time, log_value in sorted_log:
                    if log_time <= target_time:
                        last_known_value = log_value
                    else:
                        # 로그가 시간순으로 정렬되어 있으므로,
                        # 현재 로그 시간이 목표 시간을 초과하면 더 이상 볼 필요 없음
                        break
                row_values.append(last_known_value)

            results_data[instance_id] = row_values

    if not results_data:
        print("경고: 처리할 인스턴스 데이터를 찾지 못했습니다.")
        return pd.DataFrame(columns=timestamps)

    # 결과 딕셔너리를 데이터프레임으로 변환
    df = pd.DataFrame.from_dict(results_data, orient="index", columns=timestamps)
    df.index.name = "instance_id"

    return df


if __name__ == "__main__":
    # --- 사용 예시 ---
    # 1. 분석할 기본 경로 설정
    # Path 객체를 사용하여 OS에 맞는 경로 구분자 처리
    target_base_path = Path("Outputs_scenarios/20250716T103729_601420/pra_600s/baseCp")
    target_base_path = Path(
        "C:/Users/hjt/data/results_hybridflowshop/20250722/20250723T005742_123412/pra_100s/dispatch_cjims55_baseCp"
    )

    # 2. 값을 확인할 시간(초) 목록 정의
    timestamps_to_check = [20, 100, 600]

    # 3. 함수 호출
    print(f"'{target_base_path}' 경로의 데이터를 분석합니다...")
    obj_value_df = create_obj_value_dataframe(
        base_path=target_base_path, timestamps=timestamps_to_check
    )

    # 4. 결과 출력
    print("\n--- 목적 함수 값 변화 보고서 ---")
    print(obj_value_df)

    # 5. 결과를 CSV 파일로 저장 (선택 사항)
    output_filename = f"obj_report_{target_base_path.name}.csv"
    obj_value_df.to_csv(output_filename)
    print(f"\n결과를 '{output_filename}' 파일로 저장했습니다.")
