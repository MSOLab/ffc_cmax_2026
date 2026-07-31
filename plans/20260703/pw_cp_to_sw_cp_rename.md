# `pw_cp` → `sw_cp` 이름 통일 계획 (hybridflowshop)

- **날짜**: 2026-07-03 (계획) / 2026-08-01 (실행 완료)
- **상태**: ✅ 완료 — 브랜치 `20260703_pw_to_sw`, 311개 파일 변경. 결과는 §6, 착수 전 계획 대비
  사실관계 정정은 §0의 ⓘ 표시 참조.
- **동기**: 논문에서 이 알고리즘을 **sliding-window CP (`sw_cp`)** 로 표기. 형제 저장소
  `ffc_dw_wET_2026`(선행 완료), `flowshop-tardiness`와 이름을 맞춘다.
- **참고**: `~/code/ffc_dw_wET_2026/plans/20260703/pw_cp_to_sw_cp_rename.md` (선행 완료 사례),
  `~/code/flowshop-tardiness/plans/20260703_pw_cp_to_sw_cp_rename.md`.

> 이 저장소는 세 저장소 중 **리네임 표면이 가장 크고 위험**하다. 아래 §0의 두 특수사항
> (`surrogate_pw_cp_*` config 키, `pw_cp_metadata` 저장 키)을 먼저 결정하지 않고 일괄 치환하면
> 수백 개 config·기존 저장물이 조용히 깨진다.

---

## 0. 이 저장소의 특수사항 (착수 전 필독)

1. **소스 모듈이 2개.** `hybridflowshop/controller/pw_cp.py`,
   `hybridflowshop/cpsat_model_2/pw_cp.py` — 둘 다 `git mv`로 `sw_cp.py`.
2. **`method:` 문자열은 `getattr`로 디스패치** (`hfs_cp_lns.py:13583` `getattr(self, method_name, None)`).
   → `def pw_cp`/`def incremental_pw_cp`를 바꾸면 모든 config의 `method:` 문자열도 함께.
3. **⚠️ `surrogate_pw_cp_*` 생성자 kwargs (30여 개)** — `HybridFlowShopCpLnsController.__init__`
   파라미터명이자 **YAML config 키**다. `configs_mip_lb/`(≈195 파일), `configs_ncs/`(≈71 파일)이
   `surrogate_pw_cp_enabled:`, `surrogate_pw_cp_batch_size_ratio:` 등으로 설정한다.
   **결정 필요**: 이 kwargs까지 `surrogate_sw_cp_*`로 바꿀지(=수백 config 동시 수정), 아니면
   내부 알고리즘 명칭과 무관한 안정 파라미터로 보고 **그대로 둘지**. → 범위 폭발의 핵심 지점.

   > ⓘ **착수 시 정정 (2026-08-01)**: 위 기술은 두 군데가 사실과 달랐다.
   > (a) 소유자는 `__init__`이 아니라 **`initialize_by_tau_coarsened_cp`**
   > (`hfs_cp_lns.py:2948`, 시그니처 kwargs 20개 + 내부 식별자 2개 = 토큰 22종).
   > (b) 실제 설정 파일은 **YAML 21개뿐**(`configs_mip_lb/` 16, `configs_ablation_p1/` 5)이고
   > **`configs_ncs/`에는 0개**. 195/71은 `surrogate_` 키가 아니라 단순히 `pw_cp`(주로 `method:`)를
   > 포함한 파일 수였다. → "수백 config 동시 수정"이라는 범위 폭발 전제는 성립하지 않으며,
   > 이에 따라 치환하기로 결정했다(§5-1).
4. **⚠️ `pw_cp_metadata` 저장 키** — `controller/pw_cp.py:218`가
   `solution_dict["pw_cp_metadata"] = {...}`로 쓰고, `concurrent_painter.py`가
   `content.get("pw_cp_metadata")`로 읽는다. 키를 바꾸면 **기존에 저장된 YAML은 플롯 실패**.
   → 쓰기/읽기 동시 변경 + (필요시) 읽기측에서 구/신 키 모두 fallback.
5. **ruff 미설정** (`pyproject.toml`에 `[tool.ruff]` 없음). isort 최초 도입 → 전역 재정렬 발생.
6. **collision 없음**: `sw_cp`/`SwCp`/`SW-CP`는 현재 저장소에 0회. 안전.

   > ⓘ **착수 시 정정 (2026-08-01)**: 0회가 아니다. **`isw_cp` (= incremental sliding-window CP)가
   > 이미 존재**한다 — `scripts/p1_algo_explainer/run_isw_cp.py`,
   > `configs_ablation_p1/subroutine_flow_c1_no_isw_cp{,_full}.yaml`,
   > `main_metadata_p1_improv_ablation*.yaml`, `analysis_outputs/20260612_p1_algo_explainer/isw_cp/`,
   > `plans/20260612/p1_algo_explainer_gantt.md`(`SW-CP`, `ISW-CP`, `sec:p1-isw-cp`).
   > 이름 충돌 위험은 아니고 오히려 **§0-7의 sliding-window 확정을 기정사실로 만든다.**
   > 다만 §1이 `run_isw_cp.py`를 리네임 대상에서 **누락**했다 — 이 파일이 `controller.pw_cp`(:83)와
   > `cpsat_model_2.pw_cp`(:87)를 직접 import하므로 반드시 포함해야 한다(실행 시 포함함).
7. **명칭 근거**: 코드/주석은 이미 "sliding window"를 내부 모델로 사용
   (`controller/pw_cp.py:274-275,351,377,549`, 커밋 `32490ec0 "sliding window for batch processing"`).
   단 `AGENTS.md:12`는 "Prefix-window", 일부 문서는 "Partial-window"로 P를 혼용 → **최종 명칭 확정 필요**.

## 1. 리네임 대상 (load-bearing)

**소스 (파일명 + 내용):**
- `hybridflowshop/controller/pw_cp.py` → `sw_cp.py` — 클래스 `PwCpContext`,
  `PwCpSubproblemSpec`, `PwCpSubproblemLog`, `PwCpRunState`, `PwCpResult`, `PwCpConstructor` → `SwCp*`;
  로그/독스트링 `PW-CP` → `SW-CP`; `:18` `from hybridflowshop.cpsat_model_2.pw_cp import ...` 경로.
- `hybridflowshop/cpsat_model_2/pw_cp.py` → `sw_cp.py` — `PwCpVars`, `PwCpModelBuilder`,
  `create_pw_cp_schedule` → `SwCp*`/`create_sw_cp_schedule`.
- `hybridflowshop/controller/hfs_cp_lns.py` (파일명 유지, 내용만, ≈129곳):
  - `:47` import `from hybridflowshop.controller.pw_cp import PwCpConstructor, PwCpResult`.
  - defs: `incremental_pw_cp`(15014), `bound_gap_guarded_incremental_pw_cp`(15153), `pw_cp`(15217),
    `_resolve_pw_cp_batch_size`(14983), `_get_pw_cp_batch_count`(15003), 중첩 `run_surrogate_pw_cp_chain`(3258).
  - 호출부: `self.pw_cp(**current_pw_cp_kwargs)`(15151), `PwCpConstructor(self)`(3289,15008,15277,3360).
  - 디스패치 문자열: allow-list set `"incremental_pw_cp"`,
    `"bound_gap_guarded_incremental_pw_cp"`(13565-13566), `"method": "pw_cp"`(15139).

**설정 (내용 — `method:` 디스패치, 반드시):**
- `configs_pw_cp/` 디렉토리(5개 yaml)의 `- method: pw_cp` → `sw_cp`. 디렉토리명도 `configs_sw_cp/`로
  옮길지 결정(참조 경로가 있으면 함께).
- `tests/test_reactive_helpers.py:421,428,429,452` `{"method": "pw_cp", ...}`.

> ⓘ **착수 시 정정 (2026-08-01)**: 위 두 항목은 `method:` 치환 범위를 크게 과소 기술했다.
> 실제 `method:` 문자열은 `incremental_pw_cp` 366회 / `pw_cp` 236회 /
> `bound_gap_guarded_incremental_pw_cp` 2회 = **약 277개 config 파일**에 걸쳐 있다
> (`configs_mip_lb/` 201, `configs_ncs/` 71 포함). §3이 "파일명 보존, 내용 치환"으로 의도는
> 담고 있으나 §1에는 드러나 있지 않았다. → 전체 치환하기로 결정(§5-3).

**리포트/페인터:**
- `concurrent_painter.py`(리포 루트): `_draw_pw_cp_subproblem_progress_plots`,
  `_build_pw_cp_progress_output_path`, `content.get("pw_cp_metadata")` (§0-4 참조).
- `hybridflowshop/report/hfs_subroutine_report.py`: docstring 예시 `"4-pw_cp"`, `PwCpResult` 언급.
- `controller/controller_core.py:825,854,858`: `{step_idx}-{method_name}` 규칙 설명 예시
  (`"4-pw_cp"`, `"2-incremental_pw_cp"`) — 예시 문자열이므로 갱신은 선택.

**테스트 (파일명 + 내용):**
- `tests/controller/test_pw_cp.py` → `test_sw_cp.py`,
  `tests/controller/test_incremental_pw_cp.py` → `test_incremental_sw_cp.py` (`git mv`).
- 내용에 `pw_cp` 참조: `tests/controller/test_hfs_cp_lns_portfolio_flow.py`(11),
  `test_subroutine_progression_recorder.py`(18), `tests/test_concurrent_painter.py`(11).
- `verify/pw_cp_positive_boundary_deviation_demo.py` → `sw_cp_..._demo.py`.

**클래스 식별자 요약** (`PwCp*` → `SwCp*`): `PwCpConstructor`(22), `PwCpSubproblemSpec`(14),
`PwCpContext`(11), `PwCpVars`(8), `PwCpRunState`(6), `PwCpResult`(6), `PwCpSubproblemLog`(4),
`PwCpModelBuilder`(3). `PW-CP`(175, 로그·문서) → 코드 한정 `SW-CP`.

**문서 (선택):** `requirement_docs/algorithm/pw_cp_constructor_run.md`(+`.html`) —
`AGENTS.md:51` 및 `.claude/skills/algorithm-doc-kr/`가 참조하는 표준 예시 문서. 리네임 시 참조도 갱신.
`plans/20260319/PW_CP_SLACK_HANDOFF.md`, `plans/20260320/pw_cp_slack_retiming_summary.md`,
`plans/20260321/pw_cp_model.md`, `plans/20260322/create_pw_cp_schedule.md`.

## 2. isort 규칙 추가

`pyproject.toml`에 신규:
```toml
[tool.ruff.lint]
# Keep ruff's default rules (E4, E7, E9, F) and add isort import sorting.
extend-select = ["I"]
```
`uv run ruff check --fix` → 전역 import 정렬. **ruff 최초 도입이라 F841/E402 등 default 위반도
표면화**될 수 있음 — rename 무관 위반은 이번 커밋에서 손대지 말 것. `AGENTS.md:62-63`가
`uv run ruff check .` / `uv run ruff format .`를 문서화하고 있으니 그 명령으로 검증.

## 3. 절대 건드리지 말 것 (사실관계 보존)

- **`Outputs_scenarios/`, `Outputs_debug/`** — `*pw_cp*` 경로가 **182,284개**. 과거 실행 산출물.
  런타임에 당시 `method:`명으로 생성됨 → 향후 실행분만 `sw_cp`.
- **`.codex/skills/add-subalgorithm/references/pw-cp-precedent.md`, `SKILL.md:67`** — 커밋 해시/
  메시지 인용(`30746ad2 "Add PW-CP"` 등). 역사 기록 → 인용은 그대로(필요시 "(now sw_cp)" 병기).
- **`main_metadata_20260512_resume18_...yaml:30`** `resume_dir_path:
  "...checkpoints/18-incremental_pw_cp"` — 디스크에 실재하는 과거 체크포인트 경로. 문자열을
  `18-incremental_sw_cp`로 바꾸면 존재하지 않는 경로가 됨. **그대로 둔다.**
- **`configs_mip_lb/**`, `configs_ncs/**`의 날짜형 실험 config 파일명** (예:
  `subroutine_flow_20260512-resume18-relaxed-pwcp-r030-2nc.yaml`) — 파일 **내용**(method/kwargs)은
  §1/§0-3 결정에 따라 치환하되, **파일명**은 실험 추적성 때문에 보존 권장. (파일명 ≠ 내용 별개 판단)

## 4. 절차 (권장 순서)

1. **결정 먼저**: §0-3(`surrogate_pw_cp_*` kwargs 치환 여부), §0-4(`pw_cp_metadata` 키 처리),
   §0-7(최종 명칭 sliding-window 확정), `configs_pw_cp/` 디렉토리명 변경 여부.
2. `git switch -c 20260703_pw_to_sw`
3. `git mv` 소스 2개 + 테스트 2개 (+ verify 데모, 문서 선택).
4. 코드 내용 치환: `PwCp`→`SwCp`, `PW-CP`→`SW-CP`, `def pw_cp/incremental_pw_cp/...` 및 helper,
   import 경로 `.pw_cp`→`.sw_cp`, 디스패치 문자열.
5. config `method:` 문자열 치환 (grep으로 `method: *pw_cp` 한정). surrogate kwargs는 §0-3 결정대로.
6. `pw_cp_metadata` 쓰기/읽기 동시 변경(+읽기측 fallback).
7. `pyproject.toml` isort 추가 → `uv run ruff check --fix` → `uv run ruff format .`.
8. `AGENTS.md:58-61`의 `uv run pytest tests/controller/test_pw_cp.py` 등 문서 내 명령 갱신.
9. 검증: `uv run pytest`, import 스모크, 샘플 config로 `method: sw_cp` 디스패치 확인,
   저장→플롯 왕복(`concurrent_painter.py`)으로 `sw_cp_metadata` 경로 확인.

## 5. 확인 필요한 결정 → **확정 (2026-08-01)**

1. `surrogate_pw_cp_*` config 키를 바꿀지(수백 config 파급) vs 안정 파라미터로 보존.
   → ✅ **바꾼다.** 파급이 YAML 21개뿐임이 §0-3 정정으로 드러나 이름 통일을 택함.
2. `pw_cp_metadata` 저장 키 변경 시 기존 저장물 하위호환(fallback) 범위.
   → ✅ **신규 키로 쓰고, 읽기측에서 구 키 fallback.** 과거 산출물 플롯 계속 동작.
3. 최종 명칭을 `sw_cp`(sliding-window)로 확정 — 문서의 Prefix/Partial 표기 정리.
   → ✅ **`sw_cp` / sliding-window 확정.** 문서 산문의 "Prefix-Window"도 전부 "Sliding-Window"로 통일.
4. `configs_pw_cp/` 디렉토리명, 문서 파일명까지 리네임 범위에 포함할지.
   → ✅ **`configs_sw_cp/` + `requirement_docs` 문서명 리네임. `plans/` 과거 문서는 파일명·내용 모두 보존.**

## 6. 실행 결과 (2026-08-01, 브랜치 `20260703_pw_to_sw`)

311개 파일 변경 (`R` 12, `M` 298, `A` 1).

**리네임 (`git mv` 12개 — rename 이력 보존):** `controller/pw_cp.py`·`cpsat_model_2/pw_cp.py` →
`sw_cp.py`; `tests/controller/test_{,incremental_}pw_cp.py` → `test_{,incremental_}sw_cp.py`;
`verify/pw_cp_positive_boundary_deviation_demo.py` → `sw_cp_...`; `configs_pw_cp/` →
`configs_sw_cp/`(5 yaml); `requirement_docs/algorithm/pw_cp_constructor_run.{md,html}` → `sw_cp_*`.

**내용 치환 276개 파일** (`pw_cp`→`sw_cp`, `PwCp`→`SwCp`, `PW-CP`→`SW-CP`): 클래스 8종, 디스패치
메서드 5개, allow-list 문자열, import 경로, `surrogate_sw_cp_*` 22종, `run_isw_cp.py`(§0-6 누락분).
추가로 `main_metadata_debug.yaml`의 `configs_sw_cp/` 경로,
`main_metadata_p1_improv_ablation{,_full}.yaml` 주석, 참조 문서 링크
(`AGENTS.md:51`, `algorithm-doc-convention.md:6`, `initialize_by_tau_coarsened_cp.html:174`) 갱신.

**하위호환:** `controller/sw_cp.py:218`이 `sw_cp_metadata`로 쓰고 `concurrent_painter.py:188`이
`content.get("sw_cp_metadata", content.get("pw_cp_metadata"))`로 fallback. 회귀 테스트
`test_draw_sw_cp_subproblem_progress_plots_reads_legacy_pw_cp_metadata_key` 추가 — fallback 없이
`assert 0 == 1`로 실패함을 먼저 확인 후 구현(TDD).

**isort:** `pyproject.toml`에 `[tool.ruff.lint] extend-select = ["I"]` 추가,
`ruff check --select I --fix`로 I001 35건 해소.

### 검증

- `uv run pytest` → **515 passed**.
- `SubroutineFlowValidator` 전수: `sw_cp` 포함 config **245개 전부 통과**. (전체 608개 중 121개 실패는
  `initialize_by_cjq1`, `apply_apsc_lb` 등 오래전 제거된 메서드 참조로 **리네임 이전부터 존재**;
  실패 메시지에 `sw_cp`/`pw_cp` 언급 0건.)
- import 스모크: `SwCp*` 6종, 모델 export 3종, `getattr` 디스패치 메서드 5개,
  `surrogate_sw_cp_*` 20개 시그니처 kwargs 정상. `requirement_docs/algorithm/*.html` 링크 전수 검사 깨짐 0.
- HEAD worktree 대조로 ruff 린트 잔여(E402 25, F841 5, E712 4, F401 3, F811 1)와 mypy 오류 1건이
  **전부 pre-existing**임을 확인 — §2대로 손대지 않음.

### 계획과 다르게 처리한 것

- **§4-7의 `uv run ruff format .` 미실행.** HEAD 기준 67개 → 현재 66개 파일이 reformat 대상으로,
  ruff format이 이 저장소에 한 번도 돌지 않아 생긴 기존 편차다. 리네임이 새로 만든 편차는 0이므로
  실행하면 리네임과 무관한 churn 66개 파일이 커밋에 섞인다(`AGENTS.md`의 "avoid unrelated
  formatting churn"). **별도 커밋으로 분리 권장 — 미해결 과제로 남김.**

### 남겨둔 것 (의도적 보존)

- **`configs_mip_lb/*/main_metadata_*.yaml` 등 약 40개 파일의 `description:` 산문에 `PW-CP` 표기 잔존**
  (예: `"Resume from PW-CP checkpoint 18; ..."`). 일괄 치환에서 `main_metadata*.yaml`을 제외한 결과인데,
  상당수가 디스크에 실재하는 `18-incremental_pw_cp` 체크포인트를 서술하고 있어 바꾸면 설명과 실제
  경로가 어긋난다. 과거 실행 기록으로 보고 §3 원칙대로 보존. **순수 산문이므로 향후 정리 가능.**
- §3 목록 전부: `Outputs*/`(182,284 경로), `analysis_outputs/`, `plans/`, `.codex/skills/` 커밋 인용,
  `resume_dir_path` 체크포인트 경로, `*pwcp*` 실험 config 파일명.
- `AGENTS.md:109`의 `fix(pw-cp): ...` — 과거 커밋 스타일을 서술하는 문장이라 그대로 둠.
