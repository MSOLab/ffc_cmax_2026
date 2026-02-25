import csv
from pathlib import Path

# CSV 파일과 인스턴스 파일들이 있는 폴더 경로
folder_path = Path(__file__).parent
csv_file = folder_path / 'instance_ids_ff2020small.csv'

# CSV 파일 읽기
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 파일 이름 변경
renamed_count = 0
error_count = 0

for row in rows:
    old_name = row['as_is_filename']
    new_name = row['to_be_filename']

    old_path = folder_path / old_name
    new_path = folder_path / new_name

    # 파일이 존재하는지 확인
    if old_path.exists():
        try:
            # 새 파일명이 이미 존재하는지 확인
            if new_path.exists():
                print(f"경고: {new_name}이 이미 존재합니다. 건너뜁니다.")
                continue

            # 파일 이름 변경
            old_path.rename(new_path)
            renamed_count += 1
            print(f"변경 완료: {old_name} -> {new_name}")
        except Exception as e:
            error_count += 1
            print(f"오류 발생: {old_name} 변경 실패 - {e}")
    else:
        error_count += 1
        print(f"파일 없음: {old_name}")

print(f"\n총 {renamed_count}개 파일 이름 변경 완료")
print(f"오류: {error_count}개")
