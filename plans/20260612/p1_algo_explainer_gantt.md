# P1(`FFc||Cmax`) 알고리즘 설명용 Gantt 빌드 패널 생성 계획

작성일: 2026-06-12
대상 문제: `FFc||Cmax` (학위논문 Part 1)
산출 목적: 박사 방어 슬라이드의 빈 페이지(MD / BN2D / QSR / CP-LB / ISW-CP)를,
기존 SW-CP[1/5]·FMM[1/4] 슬라이드처럼 **Gantt 다단계 빌드**로 채우기 위한 그림 생성.

알고리즘 정의 출처(단일 진실): `Juntaek-PhD-Thesis/contents/ffc_cmax.tex`
(MD §`sec:p1-md` / BN2D §`sec:p1-bn2d` / QSR §`sec:p1-qsr` /
CP-LB §`sec:p1-cp-lb-quantile`,`sec:p1-cp-lb-adaptive` / ISW-CP §`sec:p1-isw-cp` /
전체 흐름 `alg:full-flow`).

---

## 0. 결정사항 (확정)

- **설명용 인스턴스**: 10 jobs × 4 stages, **2 machines/stage** (총 80 ops). 가독성 우선.
- **스케줄 생성**: 실제 알고리즘 코드 실행 + 컴포넌트 경계마다 스냅샷 훅으로 incumbent 캡처.
- **QSR 설명 프레임**: "저해상도 프록시(low-res proxy)" 메타포
  — Quantize=시간축 τ배 압축 → Schedule=작은 문제 빠르게 풂 → Reconstruct=원해상도로 stretch → Repair.
- **패널 구성**: 알고리즘마다 다단계 빌드(기존 슬라이드 스타일).

---

## 1. 설명용 인스턴스 (`make_demo_instance.py`)

FF 첫 인스턴스(`resources/ff2020big/1.txt`, 40×5, machines `[3,3,2,3,3]`)에서
**처음 10개 job 열 × 처음 4개 stage 행**을 잘라내고 machine 라인을 `2 2 2 2`로 강제.

- **processing time은 원본 그대로 유지**: FF 인스턴스 1의 `p_ij`를 그대로 사용 →
  분포는 원본과 동일한 **uniform[1,99]** (실제값 46,6,48,27,… ). job/stage 부분집합과
  machine 수(→2/stage)만 바꾸고 p 값은 손대지 않음.
- 출력: `resources/demo_p1_10x4/1.txt` (FF 포맷 그대로: `n c` / machine-per-stage / stage별 p 행).
- 메타 주석 파일 `resources/demo_p1_10x4/PROVENANCE.md`:
  "FF instance 1 의 first 10 jobs × first 4 stages, machines forced to 2/stage, **p_ij 원본 유지**.
  학위논문 실험 결과(`data/ffc_cmax/references.md`)와 무관한 **설명 전용** 축소 인스턴스"임을 명시.
- 크기가 작아 모든 컴포넌트가 <1s 내 종료 → 패널 반복 생성/튜닝 빠름. seed=42 고정으로 재현.

> **τ 가독성(해결됨)**: p∈[1,99] 에 τ=5 → surrogate width = ⌈p/5⌉ ∈ **[1,20]**.
> horizon이 ~5배 압축되어 Quantize 효과가 또렷하고, width 1로 뭉개지는 건 p≤5인 소수 op뿐.
> **10×4 데모 + τ=5 그대로로 QSR 압축 스토리 성립** → 별도 τ 조정·전용 데모 불필요.

---

## 2. 스냅샷/렌더 공통 인프라

### 2.1 비침습 원칙 (중요)
이 레포는 학위논문 **authoritative 결과의 출처**(`CLAUDE.md`, `data/ffc_cmax/references.md`).
따라서 설명용 코드는 **프로덕션 경로의 동작·seed·시간예산을 바꾸지 않는다.**

- 가능한 곳은 **컴포넌트를 standalone 호출**하여 입출력만 캡처(라이브러리 무수정).
  - MD: `dispatcher/mixed.py::MixedDispatcher`
  - BN2D: `dispatcher/bn2d_option.py` (`BN2DDispatcher`,`BN2DOption`)
  - CP-LB: `lower_bounds.py` / `lb_enum.py`의 retained-stage CP 함수
  - NEH-CP: `controller/neh_cp.py`
  - ISW-CP: `controller/hfs_cp_lns.py`
- 내부 하위 스텝(QSR 내부 chain, ISW-CP per-window, CP-LB retained relaxation 해)은
  standalone로 안 나오므로, **선택적 `snapshot_sink` kwarg(기본 None=no-op)** 를 해당 함수에
  추가. 기본값이 no-op이라 프로덕션 호출은 100% 불변(KISS, 행위 변화 0).

### 2.2 `recorder.py`
```
class GanttSnapshotRecorder:
    record(label, start_map, end_map, *, highlight=None, axis=None, note=None)
        # (job,stage,machine)->time 두 맵 + 강조 op 집합 + 축 힌트를 디스크에 직렬화
```
- 직렬화는 `io_solution.get_start_time_dict/get_end_time_dict` 와 호환되는 형식 사용.

### 2.3 렌더 헬퍼 (`render.py`) — `painter/gantt.py::GanttPlotter.plot_hybrid_flowshop` 래핑
공통 규칙으로 패널 간 일관성 확보:
- **색 일관성**: `all_job_list = 데모 10 job 고정` → 모든 패널에서 job별 동일 색.
- **축 일관성**: 한 알고리즘 빌드 시퀀스 내에서 `force_start/force_end` 고정 → before/after 비교 가능.
  - **예외(QSR)**: 압축 표현 방식은 **§9로 개정**됨(공유 절대축 폐기 → 자체축 + τ-배율 x눈금).
- **강조**: `highlight_op_set` 로 bottleneck stage / batch·window 멤버십 / critical path / cap 강조.
- **출력 형식: SVG 단일** (`analysis_outputs/20260612_p1_algo_explainer/<algo>/step_NN.svg`).
  본 figure는 **슬라이드 전용**(학위논문 본문 `fig/`로 승격하지 않음).

### 2.3.1 라벨/주석 표시 규칙 (확정)
- **인스턴스 정보 출력 금지(항상 off)**: Gantt 어디에도 인스턴스 메타(n·c·machine 수, 파일명,
  분포 등) 텍스트/제목/footnote를 출력하지 않는다. 비협상 기본값.
- **나머지 텍스트 라벨은 단일 flag로 일괄 제어, 기본값 off**:
  `show_labels: bool = False` 하나가 아래 전부를 켜고 끈다.
  - bar 안 **job ID**
  - bar 안 **processing time**
  - **x축 label**(시간축 제목/눈금 라벨)
  - **y축 label**(stage/machine 제목/라벨)
  - 기본(off) = 색 막대 + 강조만 있는 **무라벨 클린 차트**(슬라이드용). flag on 시 위 4종 동시 표출.
- 이 규칙은 `render.py` 래퍼에서 `GanttPlotter.plot_hybrid_flowshop` 호출 전후로 적용
  (job/p 라벨은 bar 그리기 옵션으로, 축 라벨은 `ax.set_xlabel/ylabel('')`·tick 제거로).
  단계 강조선(window 경계, head/tail 경계, lag 화살표 등)은 라벨이 아니므로 flag와 무관하게 유지.

### 2.4 가드 무력화 (설명 모드)
QSR·ISW-CP·NEH-CP의 time-budget pre-call/per-batch 가드가 작은 인스턴스에서 호출을 skip할 수
있음 → explainer 드라이버에서 **넉넉한 budget 주입 또는 가드 우회 플래그**로 항상 실행되게 함
(프로덕션 기본값은 불변).

---

## 3. 스크립트 구조

```
scripts/p1_algo_explainer/
  make_demo_instance.py     # §1
  recorder.py / render.py   # §2.2, §2.3
  hooks.py                  # snapshot_sink 주입 지점 목록·패치 (선택 kwarg)
  run_md.py                 # §4.1
  run_bn2d.py               # §4.2
  run_qsr.py                # §4.3  (가장 공들일 것)
  run_cp_lb.py              # §4.4
  run_isw_cp.py             # §4.5
  build_all.py              # 위 전부 순차 실행 → 패널 일괄 생성
```

---

## 4. 알고리즘별 패널 스토리보드 (tex 정의 근거)

각 패널 = Gantt 1장(또는 좌우 비교 2장) + 한 줄 캡션. 슬라이드 1장에 1패널 대응.

### 4.1 MD — Mixed Dispatch  (근거 §`sec:p1-md`, `alg:md`)
1. **우선순위 시퀀스**: Palmer P / Gupta G / CDS {Q_k} 인덱스를 표로 제시 → 시퀀스 π 선택(표, Gantt 아님).
2. **순수 모드 A — job-then-stage** (n_p=n): Gantt. 한 job이 전 stage를 먼저 채우는 "계단" 강조.
3. **순수 모드 B — stage-then-job** (n_p=0): Gantt. stage 1→c 순으로 채움 강조.
4. **혼합** (n_p=⌈n/2⌉): head=job-then-stage, tail=stage-then-job. **head/tail 경계 수직선** 강조.
5. **최종 선택**: (sequence, n_p) ladder 중 최소 makespan → MD 결과 Gantt + makespan 주석.
시각 훅: head/tail 경계선, 두 dispatch 방향.

### 4.2 BN2D — Bottleneck-centered two-way dispatching  (근거 §`sec:p1-bn2d`, `alg:bn2d`)
1. **bottleneck stage i\*** 강조 + job별 r_j(앞 workload)/tr_j(뒤 workload) 막대.
2. **cap 선택**: L(작은 r_j)=앞, R(작은 tr_j)=뒤; 선택 job 색칠, 순서 `L ‖ mid ‖ R` 표시.
3. **bottleneck dispatch**: i\* 를 단일 stage PMS로 스케줄 → i\* 한 행만 Gantt(L/mid/R 순).
4. **왼쪽 전파**: i\* 이전 stage들을 reversed-time subinstance로 (← 화살표).
5. **오른쪽 전파**: i\* 이후 stage들을 MD로 forward (→ 화살표). 전체 Gantt + makespan.
시각 훅: bottleneck stage 밴드 강조, 바깥 양방향 화살표(← reversed / → forward), L/R cap 색.

### 4.3 QSR — Quantize–Schedule–Reconstruct  ★ 저해상도 프록시  (근거 §`sec:p1-qsr`, `alg:qsr`)
핵심 메시지: **"문제의 저해상도 복사본을 싸게 풀고, 해(순서)를 원해상도로 업스케일."**

1. **Quantize**: 좌=원본 Gantt(실제 p_ij), 우=surrogate Gantt(width=⌈p/τ⌉).
   캡션: "원본을 1/τ 해상도로 다운샘플".
   **압축 가시화 방식·τ 값은 §9로 개정**(τ=25 과장 + x눈금 비율로 표현; 막대는 자체축 auto-fit).
2. **Schedule (surrogate)**: 압축된 작은 문제에서 MD/BN2D→NEH-CP→ISW-CP 실행.
   surrogate before→after: makespan이 surrogate 축에서 줄어듦. 스냅샷 2~3개 저장.
   캡션: "작은 탐색공간 → CP가 좋은 operation 순서를 싸게 학습".
3. **Reconstruct**: surrogate operation 순서를 **원본 p_ij로 replay** → 시간축이 다시 stretch.
   **2가지 복원 좌우 비교**:
   - stage-sequence restore (stage별 시간순 보존)
   - machine-sequence restore (machine별 op 순서 보존)
   둘 중 better 채택(각 makespan 주석, 어느 쪽이 이겼는지 표시).
   캡션: "학습한 순서를 원해상도로 업스케일 (2가지 복원, better 채택)".
4. **Repair**: best restored에 critical-schedule local search → before→after, 틈 메움/critical op 이동,
   makespan ↓. 캡션: "임계경로 국소수선으로 마무리".
시각 훅(=막막함 해소 포인트): **Quantize(압축) → Reconstruct(stretch)** 의 τ배 축 스케일 대비가 펀치라인.

### 4.4 CP-LB — retained-stage relaxation  (근거 §`sec:p1-cp-lb-quantile`,`adaptive`, `alg:cp-lb-quantile`)
1. **retained subset 선택**: 전 4 stage 중 비유지 stage를 회색 처리, 유지 R={1, last, +quantile/bottleneck}.
   드롭된 stage는 compressed lag ℓ로 붕괴 → 유지 stage 사이 **점선 lag 화살표**.
2. **relaxation 해**: 유지 stage만 Gantt; 이 relaxation의 makespan = **유효 하한(LB)**.
   비유지 stage는 용량 무시(순간 multi-machine)임을 캡션에.
3. **hint-based dispatch**: 유지 stage start anchor를 hint로 → 원문제 feasible 전체 스케줄 복원(다시 4 stage).
   축에 **LB(점선) vs incumbent** gap 표시.
- 2-pass 비교 패널: quantile R^quant vs workload-adaptive R^adap 의 retained-stage 선택 차이 1장씩.
시각 훅: 회색 드롭 stage + lag 화살표, 시간축 위 LB 마커 vs incumbent.

### 4.5 ISW-CP — incremental sliding-window CP  (근거 §`sec:p1-isw-cp`, `alg:isw-cp`)
기존 `fig/isw_cp_5_partition_p1.pdf` 스타일 재사용.
1. **operation 5-region partition**: stage별 ops를 시간순 batch화, window 위치 w에서
   LTF / LPF / UNFIXED / RPF / RTF 5색 + **빨강 점선 window 경계**.
2. **right-justify**: backward ALAP pass로 unfixed center 왼쪽 slack 확보(ops 오른쪽 이동, before/after 점선).
3. **window subproblem 해**: UNFIXED 재배열/재스케줄, LPF/RPF within-stage 순서 유지·shift,
   LTF/RTF 고정. window before→after.
4. **slide**: window가 Δ만큼 전진 → 다음 위치 2~3컷.
5. **incremental enlargement**: pass가 더 개선 못하면 U를 U0=2→U_max=8로 +1 확대. narrow vs wide window 대비.
시각 훅: 5색 partition + sliding window + 확대. `highlight_op_set`로 region 색 구동.

---

## 5. 산출물 정리

- `analysis_outputs/20260612_p1_algo_explainer/{md,bn2d,qsr,cp_lb,isw_cp}/step_NN.svg` (**SVG 전용**)
- 각 폴더에 `panels.md` (패널 번호 ↔ 캡션 ↔ 대응 슬라이드 매핑) → 슬라이드 제작자가 그대로 사용.
- **슬라이드 전용** 산출물 — 학위논문 본문 `fig/`로 승격하지 않음.

---

## 6. 작업 순서 (구현 단계)

1. `make_demo_instance.py` + PROVENANCE → 데모 인스턴스 확정, 크기/τ 가독성 sanity check.
2. `recorder.py` / `render.py` 공통 인프라 + MD 1개로 end-to-end 파이프 검증(색/축/강조/출력).
3. `hooks.py`: snapshot_sink 주입 지점 확정(QSR 내부, ISW-CP per-window, CP-LB relaxation). no-op 기본 검증.
4. run_md → run_bn2d → run_cp_lb → run_isw_cp → **run_qsr(마지막, 가장 공들임)**.
5. `build_all.py` 일괄 생성 → panels.md 작성.

## 7. 미해결/검증 항목

- [ ] BN2D reversed-time subinstance 복원이 4-stage 데모에서 시각적으로 또렷한지(i\* 위치 선택 의존).
- [ ] CP-LB가 4 stage에서 retained subset이 충분히 줄지(|R|≥2, {1,last} 포함). q·bottleneck 파라미터 조정.
- [ ] snapshot_sink kwarg 추가가 프로덕션 결과 불변임을 회귀 테스트(작은 인스턴스 1개 결과 동일성)로 확인.

## 8. 확정된 스코프 (구현 전 합의)

- processing time: 원본 uniform[1,99] 유지(p 미수정).
- 출력: **SVG 전용**. 산출물은 **슬라이드 전용**(본문 `fig/` 승격 없음).
- **구현은 `ffc_cmax_2026` 레포 대화에서 진행** (본 계획서는 그 레포 `plans/20260612/`에 위치).

---

## 9. 개정 — QSR 압축 가시화 방식 (2026-06-12, 1차 구현 후)

**문제.** 1차 구현은 quantize surrogate(step_02)를 자체 축 auto-fit으로 그렸고,
⌈p/τ⌉가 비율을 보존하므로 원본(step_01)과 **거의 동일한 너비·모양**으로 렌더됨.
슬라이드 기본값 `show_labels=False`라 축 숫자도 없어 **τ 압축이 시각적으로 전혀 전달되지 않음**
(저해상도 프록시 펀치라인 소실). "공유 절대축으로 surrogate를 1/τ 너비로 축소" 안은 τ가 크면
surrogate가 너무 작아져 구조가 안 보이므로 폐기.

**개정 결정.**

1. **τ = 25 (설명 전용 과장값).** 논문/프로덕션 값은 τ=5이나, **이 QSR 설명 그림에 한해 τ=25**로
   과장해 저해상도 효과를 또렷이 한다.
   - 슬라이드 전용 figure이며 **thesis 실험 결과·`data/ffc_cmax/references.md`와 무관**.
     "동일 파라미터 원칙"의 **의도적·문서화된 예외**.
   - τ=25, p∈[1,99] → surrogate width = ⌈p/25⌉ ∈ {1,2,3,4} (붕괴 없이 다양). 데모 p로 검증됨.
   - `run_qsr.py`의 `TAU` 상수 변경 + 헤더 주석·`panels.md`·PROVENANCE 성격 메모에 "τ=25 과장(논문 5)" 명시.

2. **자체축 유지 + x눈금 숫자로 압축 표현** ("공유 절대축" 안 폐기, §2.3 예외·§4.3 step1 개정).
   각 패널은 자체축 auto-fit(막대 구조 잘 보임), **x축 눈금 라벨만 켜서** 숫자로 스케일 차이를 드러낸다.
   - 원본·복원·repair 패널(real time 단위): x눈금 step = **100** → 100, 200, 300, …
   - τ-축소(surrogate) 패널(quantize/schedule, surrogate 단위): x눈금 step = **4** → 4, 8, 12, …
   - 눈금 간격비 **100 : 4 = 25 = τ** 가 곧 압축률. (surrogate 1눈금(4)×τ = 100 real → 막대 너비는
     양쪽이 동일, **숫자만 25× 차이**로 "저해상도"가 읽힘.)
   - 흐름: 원본(100,200,…) → surrogate(4,8,…)로 quantize → surrogate축에서 schedule →
     복원 시 다시 100,200,… 로 stretch. **눈금 숫자가 압축↔복원 서사를 전달**.

3. **라벨 규칙 보강(§2.3.1).** `show_labels`는 여전히 기본 off지만, **QSR 패널은 x축 눈금만 예외적으로 on**.
   - job ID·duration·y축 라벨은 계속 off. **x축 눈금/숫자만** 표시.
   - `render.py`에 타깃 옵션 추가: 예) `x_tick_step: int | None = None`,
     `show_x_ticks: bool = False` (show_labels와 독립). QSR 드라이버가 원본축 패널엔 step=100,
     surrogate축 패널엔 step=4 를 전달.
   - **인스턴스 메타 출력 금지 원칙은 불변**(축 눈금 숫자는 메타가 아니라 시간 좌표라 허용).

4. (선택) 패널 사이 **"÷25 / ×25" 화살표 주석**을 슬라이드 제작 단계에서 덧붙이면 압축↔복원이 더 명확.

**구현 변경 요약**: `run_qsr.py` `TAU=5→25`; surrogate 패널 렌더에 `show_x_ticks=True, x_tick_step=4`,
원본/복원/repair 패널에 `show_x_ticks=True, x_tick_step=100`; `render.py`에 x눈금 타깃 옵션 추가;
헤더 주석·`qsr/panels.md`에 τ=25 과장 사유 명시. 다른 알고리즘(MD/BN2D/CP-LB/ISW-CP) 패널은 불변.

---

## 10. 추가 — NEH-CP 패널 (`run_neh_cp.py`, 2026-06-12, 6567d60 이후)

**배경.** §2.1에서 NEH-CP는 QSR 내부 Schedule 단계의 **컴포넌트**로만 언급되고 독립 패널
스토리보드가 없었다. 방어 슬라이드에서 NEH-CP(점진 삽입 + CP 재최적화)를 단독으로 보여줄
Gantt 빌드가 필요하므로, MD/ISW-CP와 동급의 standalone 패널 세트를 추가한다.

**알고리즘(근거 `controller/neh_cp.py::NehCpConstructor.run`).**
NEH 계열의 **점진 구성(incremental construction)**에 CP 재최적화를 결합:

1. 참조 스케줄에서 **삽입 우선순위 시퀀스** 결정(기본 `get_midpoint_sequence`; 1st-stage /
   bottleneck / override 옵션). 이 시퀀스가 곧 job 삽입 순서.
2. tail job들을 `added_batch_size` 단위 **배치로 분할**(데모: size=2 → 5배치).
3. 각 배치마다:
   - (a) **dispatch**: 새 배치 job들을 현재 부분해 위에 `MixedDispatcher`
     (`head_for_all_stages=True`)로 올림 → 부분해가 커짐.
   - (b) **CP 재최적화**: 현재 job 부분집합에 대해 CP 서브모델을 풂. dispatch 결과를 hint로,
     오래된 op들의 상대순서는 profile-fix 제약으로 고정(batch ≥ `profile_fix_min_batch_idx`),
     UNFIXED op만 재배치 → makespan ↓ (dispatch보다 나을 때만 채택).
   - (c) 아직 안 들어간 tail job들을 추가 dispatch해 **원문제 feasible full 스케줄** 생성,
     배치별 best full 추적.
4. 배치 전부 또는 시간예산 소진 시 종료, **best full 스케줄 반환**.

**비침습 캡처 전략(§2.1 zero-edit 유지).** `hybridflowshop/**` 무수정. 드라이버에서
`NehCpConstructor`를 만들되 인스턴스의 `_solve_cp_model` 바운드 메서드를 **스크립트에서 래핑**
(속성 재할당)해 배치마다 (dispatch 직후 partial, CP 직후 partial, 현재 job subset,
job→삽입배치 맵)를 캡처한 뒤 실제 `run()`을 호출. 라이브러리 동작·seed·시간예산은 불변
(ISW-CP 패널이 `_build_batch_spec`/solve helper를 직접 구동한 것과 같은 결).
설명 모드 가드 무력화(§2.4): `skip_if_estimated_neh_exceeds_remaining=False`,
`stop_before_final_reserve=False`, 넉넉한 `max_time_per_add`로 모든 배치가 항상 실행되게 함.

**패널 스토리보드** (`analysis_outputs/20260612_p1_algo_explainer/neh_cp/step_NN.svg`):

1. `step_01_insertion_sequence.md` — 삽입 우선순위 시퀀스 + 배치 분할(표/노트, Gantt 아님).
2. `step_02_batch1_partial.svg` — 배치1 삽입·CP 후 부분해(새 job 강조). "구성 시드(2 jobs)".
3. `step_03_batch2_dispatch.svg` — 배치2 dispatch 직후(새 job 강조, **CP 전**).
4. `step_04_batch2_cp.svg` — 배치2 **CP 후**(같은 강조). dispatch→CP makespan ↓ = **-CP 펀치라인**.
5. `step_05_batch3_partial.svg` … 6. `step_06_batch4_partial.svg` … 7. `step_07_batch5_partial.svg`
   — 배치별 CP 후 부분해가 채워지는 **점진 성장**(매 패널 새 배치 job 강조; 배치5 = 전 job).
8. `step_08_final_full.svg` — 반환된 best full 스케줄 + makespan 주석(배치별 full 중 최소).

**공통 규칙 준수.** 색: 데모 10 job 고정 팔레트. 축: 한 빌드 내 `force_start=0`,
`force_end = max(부분해·full makespan)` 공유 → 부분해 성장이 한 축에서 보임.
강조: `highlight_op_set`로 **현재 배치에 새로 삽입된 job**(= `job_2_inserted_batch_idx==batch`).
라벨: §2.3.1 기본 off(무라벨 클린 차트). `build_all.py`에 `run_neh_cp` 추가.

**데모 수치(seed=42, batch_size=2, midpoint seq).** seed incumbent=426;
배치별 dispatch→CP: 96→96, 244→**234**, 327→**321**, 377→377, 427→427;
best full=426. 배치2·3에서 CP가 dispatch를 각각 −10/−6 개선 → 재최적화 효과가 시각적으로 또렷.

**배치 크기 파라미터화.** `run_neh_cp.py`는 `--batch-size N`(기본 2)을 받아 삽입 입도를 바꾼다.
기본값(2)은 `neh_cp/`에, 그 외는 `neh_cp_bs<N>/`에 출력(기존 패널 불변). `build_all.py`는 기본 2만 생성.
- **batch_size=3 예시**(`neh_cp_bs3/`, 4배치=[3,3,3,1]): 배치별 dispatch→CP
  174→174, 335→**308**, 414→414, 434→434; best full=426. 배치2에서 CP가 dispatch를 **−27** 개선해
  더 큰 입도에서 재최적화 효과가 한층 또렷(step_03/step_04 대비).

---

## 11. 추가 — 스테이지 구분선 (모든 패널 공통, 2026-06-12)

**문제.** 데모는 stage당 2 machine이라 한 패널에 머신 행이 여러 개 쌓이는데(데모: 4 stage × 2 =
8행), 행 사이에 스테이지 경계 표시가 없어 어느 행이 어느 stage인지 슬라이드에서 읽기 불편.

**결정.** 모든 Gantt 패널에 **stage 사이 수평 구분선**을 추가. 스타일은 **굵은 회색 dash-dot
(`-.-.-`)**: `color="#6b6b6b"`, `linestyle="-."`, `linewidth=2.2`, `zorder=4`.

**구현(비침습).** `render.py::LabelControlledGanttPlotter`에만 추가(`painter/gantt.py` 무수정).
`plot_hybrid_flowshop` 오버라이드가 `super()` 후 `_draw_stage_separators`를 호출:
머신 lane은 stage→machine 순으로 정수 y에 쌓이고(lane idx i = `[i, i+bar_height]`,
pitch=`machine_height`), stage 누적 lane 수 c 지점의 경계선은 lane c-1 막대 하단과 lane c 막대
상단 사이 빈 구간 중앙 `y = c·machine_height − (machine_height−bar_height)/2`에 그림. lane 순서는
`GanttPlotter.create_machine_lanes`와 동일하게 재현해 경계가 정확히 정렬됨. 마지막 stage 뒤에는
선 없음.

- 텍스트 라벨이 아니므로 **`show_labels`와 무관하게 항상 표시**(window 경계선·vlines와 같은 결, §2.3.1).
- ISW-CP의 빨강 점선 세로 window 경계, NEH-CP 강조(굵은 검정 테두리), 5영역 색과 충돌 없이 공존 확인.
- 스타일은 `stage_separator_color/_linewidth/_linestyle` 인스턴스 속성으로 조정 가능.
- `build_all.py` 재실행으로 MD/BN2D/CP-LB/NEH-CP/ISW-CP/QSR 전 패널 재생성(+ `neh_cp_bs3` 별도).
