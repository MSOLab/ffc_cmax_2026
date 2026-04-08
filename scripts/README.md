# Scripts

## `plot_selected_subroutine_flow_comparison.py`

이 스크립트는 이미 `post_run_process`가 완료된 scenario 디렉터리들을 직접 골라서,
각 scenario의 `summary_method_rpdf_and_norm_time_long.csv`를 읽고
하나의 HTML 차트에 `subroutine mean norm_time vs mean RPDf`를 겹쳐 그립니다.

생성되는 차트의 특징:

- scenario별로 1개 trace
- subroutine flow 순서대로 line + marker 연결
- x축, y축 모두 `%` 형식
- x축, y축 모두 `0`을 원점으로 시작

### 입력 조건

각 scenario 디렉터리에는 아래 파일이 이미 있어야 합니다.

- `summary_method_rpdf_and_norm_time_long.csv`

예를 들면 이런 경로입니다.

- `Outputs_scenarios/20260324T205559_877608/ff2020/20260324-03`
- `Outputs_scenarios/20260331T235223_209924/ff2020/20260331-03`

### 실행 방법 1: 경로를 argument로 직접 전달

```bash
uv run python scripts/plot_selected_subroutine_flow_comparison.py \
  Outputs_scenarios/20260324T205559_877608/ff2020/20260324-03 \
  Outputs_scenarios/20260331T235223_209924/ff2020/20260331-03
```

기본 출력 파일:

- `selected_subroutine_flow_comparison.html`

현재 작업 디렉터리에 생성됩니다.

### 실행 방법 2: 스크립트 내부 `SCENARIO_PATH_LIST` 사용

`scripts/plot_selected_subroutine_flow_comparison.py` 상단의
`SCENARIO_PATH_LIST`를 원하는 scenario 경로들로 채운 뒤,
인자 없이 실행할 수 있습니다.

```python
SCENARIO_PATH_LIST = [
    "Outputs_scenarios/20260324T205559_877608/ff2020/20260324-03",
    "Outputs_scenarios/20260331T235223_209924/ff2020/20260331-03",
]
```

그 다음 실행:

```bash
uv run python scripts/plot_selected_subroutine_flow_comparison.py
```

CLI positional argument가 하나라도 들어오면, `SCENARIO_PATH_LIST` 대신 그 인자를 사용합니다.

### 옵션

#### `--output`

출력 HTML 경로를 직접 지정합니다.

```bash
uv run python scripts/plot_selected_subroutine_flow_comparison.py \
  Outputs_scenarios/20260324T205559_877608/ff2020/20260324-03 \
  Outputs_scenarios/20260331T235223_209924/ff2020/20260331-03 \
  --output Outputs_scenarios/custom_selected_flow_comparison.html
```

#### `--label-mode`

trace label 방식을 정합니다.

- `basename`:
  scenario 디렉터리 이름만 사용
  예: `20260331-03`
- `path3`:
  마지막 3개 path component를 사용
  예: `20260331T235223_209924/ff2020/20260331-03`

기본값은 `basename`입니다.

`basename` 사용 시 이름이 충돌하면, 충돌하는 항목만 자동으로 마지막 3개 path component로 확장해 구분합니다.

예시:

```bash
uv run python scripts/plot_selected_subroutine_flow_comparison.py \
  Outputs_scenarios/20260324T205559_877608/ff2020/20260324-03 \
  Outputs_scenarios/20260331T235223_209924/ff2020/20260331-03 \
  --label-mode path3
```

### 실패 조건

다음 경우에는 HTML을 만들지 않고 종료합니다.

- 선택된 scenario 경로가 2개 미만
- 선택한 디렉터리에 `summary_method_rpdf_and_norm_time_long.csv`가 없는 경우
- 집계 후 plotting 가능한 trace가 하나도 남지 않는 경우
