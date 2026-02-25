# Experiment Comparison Tool (`exp_compare`)

여러 실험(run directory) 결과를 읽어 **공통 인스턴스(name) 교집합**에 대해 알고리즘별 성능 비교 CSV를 생성합니다.

## Overview

핵심 산출물은 엑셀에서 pivot chart를 바로 만들 수 있는 wide CSV이며, **RPDf를 우선**, **RPDv를 보조**로 생성합니다.

## Installation

`exp_compare`는 프로젝트 내에 포함되어 있으므로 별도 설치 없이 사용 가능합니다.

```bash
uv sync  # dependencies 설치 (필요시)
```

## Usage

### CLI 방식

```bash
# 기본 config 사용 (exp_compare/config.yaml)
uv run python -m exp_compare

# 커스텀 config 사용
uv run python -m exp_compare --config path/to/config.yaml

# 콘솔 출력 억제
uv run python -m exp_compare --quiet
```

### 옵션

- `--config`, `-c`: YAML 설정 파일 경로
  - 미지정시 `exp_compare/config.yaml` 사용
- `--quiet`, `-q`: 콘솔 출력 억제 (로그는 `exp_compare.log` 파일로 기록)

## Configuration

### YAML 스키마

```yaml
runs:
  - path: "Outputs_scenarios/20260218T012813_090868/"
    summary_csv: "all_scenarios_summary.csv"
  - path: "Outputs_scenarios/20260217T153612_195650/"
    summary_csv: "all_scenarios_summary.csv"

reference:
  mode: "best_among_compared"   # or "fixed_dataset"
  sense: "min"                  # currently only min is expected

  # only when mode=fixed_dataset
  ref_path: null                # e.g., "refs/best_known.csv"
  ref_format: "csv"             # only csv required
  instance_key_column_in_ref: "name"
  reference_value_column: null  # e.g., "UB"

output:
  out_dir: "Outputs_analysis/compare_YYYYMMDD/"
  basename: "rpd_compare"
```

### Configuration 설명

#### `runs` (필수)
- 비교할 run directory 목록
- `path`: run directory 경로
- `summary_csv`: 각 run 내부의 summary CSV 파일명 (기본: `"all_scenarios_summary.csv"`)

#### `reference` (필수)
- Reference 값 결정 방식

| 필드 | 설명 | 기본값 |
|------|------|--------|
| `mode` | `best_among_compared` (비교 대상 중 최소값) 또는 `fixed_dataset` (외부 CSV) | `"best_among_compared"` |
| `sense` | 최적화 방향 (`min` 또는 `max`) | `"min"` |
| `ref_path` | Reference CSV 파일 경로 (mode=fixed_dataset 시 필요) | `null` |
| `ref_format` | Reference 파일 포맷 | `"csv"` |
| `instance_key_column_in_ref` | Reference CSV의 인스턴스 키 컬럼명 | `"name"` |
| `reference_value_column` | Reference CSV의 값 컬럼명 | `null` |

#### `output` (필수)
- 출력 설정

| 필드 | 설명 | 기본값 |
|------|------|--------|
| `out_dir` | 출력 디렉토리 경로 | `"Outputs_analysis/compare_YYYYMMDD/"` |
| `basename` | 출력 파일명 prefix | `"rpd_compare"` |

## Input Format

각 run directory 내부에 있는 summary CSV 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼명 | 설명 | 타입 |
|--------|------|------|
| `name` 또는 `instanceName` | instance id | string |
| `scenario` | algorithm id | string |
| `bestObj` 또는 `objValue` | objective value | float |

> **Note:** `instanceName` 컬럼은 자동으로 `name`으로 rename됩니다.

## Output Files

모든 출력 파일은 `output.out_dir` 하위에 생성됩니다. 컬럼명은 camelCase를 사용합니다.

### 1. `{basename}_long.csv`

행 단위: (name, algoUid)

| 컬럼명 | 설명 |
|--------|------|
| `name` | instance id |
| `runId` | run 식별자 (디렉토리명) |
| `scenario` | scenario 이름 |
| `algoUid` | `{runId}::{scenario}` |
| `objValue` | objective value |
| `refValue` | reference value |
| `RPDf` | Relative Percentage Difference |
| `RPDv` | Relative Percentage Deviation |
| `rank` | within-instance rank (dense rank) |

### 2. `{basename}_rpdf_wide.csv`

행 단위: `name`
컬럼: `name`, `[algoUid_1, algoUid_2, ...]`
값: `RPDf`의 float 값 (퍼센트 기호 없음)

### 3. `{basename}_rpdv_wide.csv`

행 단위: `name`
컬럼: `name`, `[algoUid_1, algoUid_2, ...]`
값: `RPDv`의 float 값

### 4. `{basename}_summary_rpdf.csv`

groupby: `algoUid`

| 컬럼명 | 설명 |
|--------|------|
| `algoUid` | algorithm identifier |
| `count` | 유효 값 개수 (NaN 제외) |
| `min` | minimum RPDf |
| `max` | maximum RPDf |
| `mean` | mean RPDf |
| `median` | median RPDf |
| `std` | standard deviation |

### 5. `{basename}_summary_rpdv.csv`

동일한 구조, 값은 `RPDv`

## Metrics Definitions

### RPDf (Relative Percentage Difference)

```
RPDf = (obj - ref) / ((obj + ref) / 2)
```

Special cases:
- `obj == 0 AND ref == 0` → `RPDf = 0`
- denominator `((obj+ref)/2) == 0` → `RPDf = NaN`

### RPDv (Relative Percentage Deviation)

```
RPDv = (obj - ref) / ref
```

Special cases:
- `ref == 0` → `RPDv = NaN`

### Rank

- instance(name) 내부에서 objValue 기준
- `sense="min"`: 작은 값이 좋음 (기본)
- `sense="max"`: 큰 값이 좋음
- 결측(obj NaN)인 경우 rank는 NaN
- 동점 처리는 dense rank (1, 2, 2, 3)

## Exit Codes

| 코드 | 의미 |
|------|------|
| 0 | 정상 완료 (intersection empty 포함) |
| 1 | 기타 예외 |
| 2 | 설정/입력 파일 없거나 파싱 불가 |

## Examples

### Example 1: 기본 사용 (best_among_compared)

```yaml
runs:
  - path: "Outputs_scenarios/20260218T012813_090868/"
    summary_csv: "all_scenarios_summary.csv"
  - path: "Outputs_scenarios/20260217T153612_195650/"
    summary_csv: "all_scenarios_summary.csv"

reference:
  mode: "best_among_compared"
  sense: "min"

output:
  out_dir: "Outputs_analysis/compare_20260219/"
  basename: "rpd_compare"
```

### Example 2: Fixed Reference 사용 (fan2023.csv)

```yaml
runs:
  - path: "Outputs_scenarios/20260218T012813_090868/"
    summary_csv: "all_scenarios_summary.csv"
  - path: "Outputs_scenarios/20260217T153612_195650/"
    summary_csv: "all_scenarios_summary.csv"

reference:
  mode: "fixed_dataset"
  sense: "min"
  ref_path: "resources/ff2020big_ref/fan2023.csv"
  instance_key_column_in_ref: "Instance"
  reference_value_column: "UB"

output:
  out_dir: "Outputs_analysis/compare_20260219/"
  basename: "rpd_compare"
```

`fan2023.csv` 파일 구조 예시:

```csv
Instance,n,s,rep,UB,LB
1,40,5,1,976,976
2,40,10,1,1292,1292
...
```

### Example 3: maximization 문제

```yaml
reference:
  mode: "best_among_compared"
  sense: "max"  # 큰 값이 더 좋음

output:
  out_dir: "Outputs_analysis/compare_20260219/"
  basename: "rpd_compare_max"
```

## Notes

- Intersection가 비어 있으면 warning 로그 1줄 출력 후 exit 0 (파일 미생성)
- 경로는 상대/절대 모두 허용
- 숫자 저장 시 `%` 기호 붙이지 않음
- NaN은 CSV에 빈 칸으로 저장
- `bestObj` 컬럼은 내부적으로 `objValue`로 rename되어 처리됩니다
