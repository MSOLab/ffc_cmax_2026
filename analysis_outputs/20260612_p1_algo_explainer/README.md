# P1 (`FFc||Cmax`) 알고리즘 설명용 Gantt 패널

박사 방어 슬라이드의 빈 페이지(MD / BN2D / QSR / CP-LB / ISW-CP)를 Gantt
다단계 빌드로 채우기 위한 **슬라이드 전용** 그림 모음. 학위논문 본문 `fig/`로
승격하지 않는다. (계획서: `plans/20260612/p1_algo_explainer_gantt.md`)

## 재현

```bash
uv run python scripts/p1_algo_explainer/build_all.py   # 데모 인스턴스 + 5개 알고리즘 패널 일괄 재생성
```

설명용 인스턴스: `resources/demo_p1_10x4/` (FF instance 1의 first 10 jobs × first 4
stages, machines 2/stage, p_ij 원본 유지). 실험 결과와 무관한 설명 전용 축소
인스턴스 — `resources/demo_p1_10x4/PROVENANCE.md` 참조.

공통 규칙(계획서 §2.3): job별 색 일관, 빌드 시퀀스 내 축 고정, `show_labels=False`
무라벨 클린 차트, 인스턴스 메타 비표출, **SVG 전용**.

## 알고리즘별 패널 (각 폴더의 `panels.md`에 패널↔캡션↔슬라이드 매핑)

| 폴더 | 알고리즘 | tex 근거 | 핵심 makespan 흐름 |
|------|----------|----------|--------------------|
| [`md/`](md/panels.md) | Mixed Dispatch | §`sec:p1-md` | MD 전용 8-job subset: 순수 A(385) / 순수 B(378) / **혼합 n_p=4(367)←최적** |
| [`bn2d/`](bn2d/panels.md) | Bottleneck two-way dispatch | §`sec:p1-bn2d` | 비종단 bottleneck i2 기준 최종 465 |
| [`cp_lb/`](cp_lb/panels.md) | retained-stage relaxation LB | §`sec:p1-cp-lb-*` | LB=419(certified) vs incumbent=426, gap=7 |
| [`isw_cp/`](isw_cp/panels.md) | incremental sliding-window CP | §`sec:p1-isw-cp` | 5-region partition + sliding + enlargement |
| [`qsr/`](qsr/panels.md) | Quantize–Schedule–Reconstruct | §`sec:p1-qsr` | τ=25(설명용 과장, 논문 5): 426 → surrogate 21 → restore 443 → repair 426→420 |

## 구현 메모

- **MD만 8-job subset 사용**: 알고리즘별 job sampling은 허용(색·축 일관성은 알고리즘
  내부에서만 필요). MD는 데모 10 job 중 `[0,1,3,4,5,7,8,9]`를 써서 중간 n_p(=4)가 양쪽
  pure를 strictly 이기는(367 < pure B 378 < pure A 385) 인스턴스를 고름.
  `run_md.py --search-subset-size 8`로 탐색. 나머지 알고리즘은 10-job 데모 유지.
- 모든 패널은 **비침습**으로 생성: 프로덕션 모듈(`hybridflowshop/**`)을 수정하지
  않고, 컨트롤러/디스패처 헬퍼를 standalone 호출해 중간 스케줄을 캡처한다.
  (QSR은 `_solve_local_base_cp_candidate`의 기존 `snapshot_solution_limit` 활용,
  ISW-CP는 `_build_operation_partition` 등 헬퍼 직접 호출.)
- 렌더 공통 인프라: `scripts/p1_algo_explainer/{recorder,render}.py`.
  ISW-CP용으로 `render.py`에 후방호환 옵션(`op_color_map` 5-region 색,
  `vlines` window 경계 빨강 점선) 추가, QSR §9용으로 x눈금 옵션
  (`show_x_ticks`, `x_tick_step`) 추가.
- **QSR §9 개정**: 압축 가시화를 위해 설명 그림에 한해 τ=25로 과장(논문/프로덕션 5).
  모든 QSR 패널은 자체축 auto-fit, x눈금 간격(real 100 vs surrogate 4 = 비율 25 = τ)
  으로 1/τ 다운샘플↔복원 서사를 전달.
