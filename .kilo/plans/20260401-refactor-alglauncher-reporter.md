# 리팩토링 계획: AlgLauncher / Reporter 분리

> 날짜: 2026-04-01
> 상태: 계획 확정, 구현 준비

## 목표
1. 알고리즘 실행과 사후 처리(리포트 생성)를 명확히 분리
2. Reporter만 독립 실행 가능하도록 하여, 기존 결과에 대한 리포트 재생성이 자연스럽게 되도록 함
3. HfsMultiInstanceRunner (822줄)를 여러 모듈로 분해

## 명명 변경
- AlgorithmRunner -> AlgLauncher (인스턴스 병렬 실행 전용)
- PostRunProcessor -> Reporter (모든 사후 처리 통합)

---

## Phase 1: 핵심 클래스 분리

### 1A. ResumeValidator 분리

신규 파일: hybridflowshop/resume/validator.py

이동 대상 (현재 hfs_multi_instance_runner.py의 4개 메서드, ~230줄):
- _load_resume_solution_check_feasibility()
- _load_obj_store_check_resume_solution_obj_value()
- _load_summary_check_obj_values()
- _inject_resume_data_into_runners()

### 1B. LogProcessor 분리

신규 파일: hybridflowshop/report/log_processor.py

현재 scripts/process_logs.py의 핵심 로직을 이동. CLI 스크립트는 thin wrapper로 남김.

### 1C. MetricsCalculator 분리

신규 파일: hybridflowshop/report/metrics_calculator.py

현재 _create_rpd_summary() (~170줄) 이동. RPDf/RPDv 계산, long/wide CSV 저장, SVG 차트 export 포함.

### 1D. TimepointSampler 분리

신규 파일: hybridflowshop/report/timepoint_sampler.py

현재 _create_timepoint_summaries(), _build_timepoint_summary(), _load_obj_store_maps_for_instances() (~170줄) 이동.

### 1E. ExcelReportWriter 분리

신규 파일: hybridflowshop/report/excel_writer.py

현재 HfsMultiScenarioRunner.write_excel_report() (~140줄) + create_dashboard() + create_info_sheet() 이동.

---

## Phase 2: AlgLauncher / Reporter 재구성

### 2A. AlgLauncher (신규: hfs_alg_launcher.py)

HfsMultiInstanceRunner에서 실행 관련만 추출:
- 인스턴스 병렬 실행 (sequential/concurrent)
- Resume 데이터 로드 및 검증 (ResumeValidator 사용)
- raw 결과 리스트 반환

### 2B. Reporter (신규: hybridflowshop/report/reporter.py)

모든 사후 처리를 통합:
- process_all(): LogProcessor + MetricsCalculator + TimepointSampler
- build_scenario_report(): per-scenario 리포트
- build_multi_scenario_report(): cross-scenario Excel 대시보드

### 2C. HfsMultiInstanceRunner 리팩토링

AlgLauncher + Reporter에 위임하는 thin wrapper로 축소 (~150줄)

### 2D. HfsMultiScenarioRunner 리팩토링

Reporter에 위임하는 thin wrapper로 축소 (~200줄)

---

## Phase 3: 독립 Post-Run 스크립트

신규 파일: post_run.py

사용 예:
  uv run python post_run.py Outputs_scenarios/20260401T143000_123456 --baseline resources/ff2020big_ref/fan2023.csv

---

## Phase 4: 설정 로드 구조 정리

YAML 파일 분리는 유지. main.py가 여러 YAML을 읽어 단일 config dict로 병합 후 전달.

---

## 파일 구조 (최종)

hybridflowshop/
  resume/
    validator.py              # Phase 1A
  report/
    log_processor.py           # Phase 1B
    metrics_calculator.py      # Phase 1C
    timepoint_sampler.py       # Phase 1D
    excel_writer.py            # Phase 1E
    reporter.py                # Phase 2B

hfs_alg_launcher.py            # Phase 2A
hfs_multi_instance_runner.py   # Phase 2C (축소)
hfs_multi_scenario_runner.py   # Phase 2D (축소)
post_run.py                    # Phase 3

---

## 예상 효과

| 파일 | 현재 | 리팩토링 후 |
|------|------|-------------|
| hfs_multi_instance_runner.py | 822줄 | ~150줄 |
| hfs_multi_scenario_runner.py | 579줄 | ~200줄 |
| scripts/process_logs.py | 364줄 | ~30줄 |
| 신규 모듈 | - | 7개 파일, 각각 80-200줄 |
