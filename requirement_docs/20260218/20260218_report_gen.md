# 구현 지시 프롬프트: Experiment Comparison (RPDf/RPDv) Post Analysis

## 0. 목표

여러 실험(run directory) 결과를 읽어 **공통 인스턴스(name) 교집합**에 대해 알고리즘별 성능 비교 CSV를 생성한다.
핵심 산출물은 엑셀에서 pivot chart를 바로 만들 수 있는 wide CSV이며, **RPDf를 우선**, **RPDv를 보조**로 생성한다.

---

## 1. 입력: 실행 argument YAML

### 1.1 YAML 스키마

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
  reference_value_column: null  # e.g., "best_known"

output:
  out_dir: "Outputs_analysis/compare_YYYYMMDD/"
  basename: "rpd_compare"
```

### 1.2 입력 CSV(각 run directory 내부)

각 run directory 안에 `summary_csv` 파일이 존재한다. (기본 `"all_scenarios_summary.csv"`)

필수 컬럼:

* `name` : instance id (instance_key)
* `scenario` : algorithm id
* `bestObj` : objective value (float)

추가 컬럼은 무시해도 된다.

---

## 2. 식별자 규칙

### 2.1 instance_key

* `instance_key := name` 단일 컬럼

### 2.2 run_id

* `run_id := basename(normalized_run_path_without_trailing_slash)`

예: `"Outputs_scenarios/20260218T012813_090868/"` → `run_id="20260218T012813_090868"`

### 2.3 algo_uid (wide 컬럼명)

* `algo_uid := f"{run_id}::{scenario}"`

예: `20260218T012813_090868::20260216-24`

---

## 3. 교집합(intersection) 규칙

* 각 run에서 등장하는 `name` set을 만든다.
* `intersection_names = ∩ (모든 run의 name set)`
* 만약 `intersection_names`가 비어 있으면:

  * warning 로그 1줄 출력
  * **아무 파일도 생성하지 말고 정상 종료(Exit code 0)**

---

## 4. Reference 값 규칙

### 4.1 mode = best_among_compared (기본)

* intersection 내에서, 각 instance(name)별로 비교 대상 모든 algo_uid의 objective 중 **최소값**을 reference로 사용 (sense=min)

### 4.2 mode = fixed_dataset

* `ref_path` CSV를 읽는다.
* join key는:

  * left: main `name`
  * right: ref CSV의 `instance_key_column_in_ref` (기본 `"name"`)
* reference value는 ref CSV의 `reference_value_column`을 사용한다.
* ref에서 해당 name이 없으면 reference_value는 NaN 처리.

---

## 5. 지표 정의 (중요)

### 5.1 RPDf (Relative Percentage Difference) — 우선 지표

정의:

* `rpdf = (obj - ref) / ((obj + ref)/2)`

예외 규칙:

* `obj == 0` AND `ref == 0` → `rpdf = 0`
* 그 외 분모 `((obj+ref)/2) == 0` → `rpdf = NaN`

### 5.2 RPDv (Relative Percentage Deviation) — 보조 지표

정의:

* `rpdv = (obj - ref) / ref`

예외 규칙:

* `ref == 0` → `rpdv = NaN`

### 5.3 Rank

* instance(name) 내부에서 objective_value 기준 (sense=min, 작은 값이 좋음)
* 결측(obj NaN)인 경우 rank는 NaN
* 동점 처리는 dense rank (1,2,2,3)

---

## 6. 산출물 (CSV만)

모든 출력 파일은 `output.out_dir` 하위에 생성한다. 디렉토리 없으면 생성한다.
파일명은 `output.basename`을 prefix로 사용한다.

### 6.1 Long 포맷: `{basename}_long.csv`

행 단위: (name, algo_uid)

필수 컬럼(정확히 이 이름으로):

* `name`
* `run_id`
* `scenario`
* `algo_uid`
* `objective_value`
* `reference_value`
* `rpdf`
* `rpdv`
* `rank`

### 6.2 Wide 포맷: `{basename}_rpdf_wide.csv`

행 단위: name
컬럼:

* 첫 컬럼: `name`
* 이후 컬럼들: 각 `algo_uid` (문자열 그대로)
  값:
* `rpdf`의 float 값만 (퍼센트 기호, 문자열 포맷 금지)

### 6.3 Wide 포맷: `{basename}_rpdv_wide.csv`

구조 동일, 값은 `rpdv`

### 6.4 Summary 통계: `{basename}_summary_rpdf.csv`, `{basename}_summary_rpdv.csv`

groupby: algo_uid

각 파일 컬럼:

* `algo_uid`
* `count` (유효 값 개수; NaN 제외)
* `min`
* `max`
* `mean`
* `median`
* `std`

(필요 시 rank/objective summary도 만들 수 있으나 최소 요구는 rpdf/rpdv 2개)

---

## 7. 구현 제약/권장

* Python 3.10+
* pandas 사용 가능하면 사용 (권장)
* 경로는 상대/절대 모두 허용, OS path 안전하게 처리
* `scenario` 컬럼명은 고정 `"scenario"`
* `bestObj` 컬럼명은 고정 `"bestObj"`
* 숫자 저장 시 `%` 기호 붙이지 말 것
* NaN은 CSV에 빈 칸으로 자연스럽게 저장되도록 해도 됨

---

## 8. CLI 형태 (권장)

아래 형태로 실행 가능하게 만든다.

```bash
python -m exp_compare --config path/to/config.yaml
```

* exit code:

  * 정상 완료: 0
  * 설정/입력 파일 없거나 파싱 불가: 2
  * 기타 예외: 1
* intersection empty는 0으로 종료

---

## 9. 최소 테스트

* run 2개 이상에서 intersection 존재 → 4개 CSV 생성 확인
* intersection empty → 파일 미생성 + warning 1줄 + exit 0
* fixed reference 모드에서 ref missing 인스턴스 → rpdf/rpdv NaN 확인
* obj=0, ref=0 케이스 → rpdf=0 확인
