# 변경할 경로로 수정
BASE="20250906T212736_116407"
cd "$BASE" || exit 1

# dry-run: 삭제할 항목만 화면에 출력
# for sub in */ ; do
#   [ -d "$sub" ] || continue
#   echo "----- DRY-RUN: $sub -----"
#   (cd "$sub" && find . -mindepth 1 ! -path './results' ! -path './results/*' -print)
# done

# 검토 후 실제 삭제:
for sub in */ ; do
  [ -d "$sub" ] || continue
  echo "----- DELETING IN: $sub -----"
  (cd "$sub" && find . -mindepth 1 ! -path './results' ! -path './results/*' -exec rm -rf -- {} +)
done

# 확인:
