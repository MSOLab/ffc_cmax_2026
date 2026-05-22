# 알고리즘 문서 problem-mechanism 프레임 cross-repo 통일 (hybridflowshop 작업분)

## 배경

- `agent-skills` 레포의 `algorithm-doc-kr` 스킬 리뷰 과정에서, 본 스킬이 hybridflowshop 의
  컨벤션에 강하게 결합되어 있음이 드러났다 (problem-mechanism 프레임, 한국어, `requirement_docs/algorithm/`
  경로, `pw_cp_constructor_run.md` 등 본보기 의존).
- 같은 종류의 솔버 알고리즘 문서를 `flowshop-tardiness`, `ffc_ddw_sum_et` 두 자매
  레포에서도 작성하므로, 스킬을 레포-specific 으로 두기 전에 먼저 **문서 컨벤션 자체를
  problem-mechanism 프레임으로 통일**한 뒤 각 레포에 스킬 사본을 배치하기로 결정.
- 언어는 통일하지 않음. **프레임만 통일하고 언어는 각 레포 기존대로** 둔다.

## problem-mechanism 프레임 (canonical)

두 Part로 구성된다. Part 1(문제 설명)은 파라미터·변수·목적·제약 4하위 섹션을 가진다.
Part 2(메커니즘)는 핵심 아이디어로 시작해 알고리즘별 상세 → 전체 실행 흐름으로 이어진다.
이외에 개요, 파라미터 요약, 주의사항은 부속 섹션.

```md
# <알고리즘 이름> (<영문 슬러그/약어>)

호출: `def <함수명>` (<파일명>.py)

## 개요                                    ← 부속
## 문제 설명                               ← Part 1
### 파라미터 (Parameters)
### 변수 (Variables)
### 목적 (Objective)
### 제약 (Constraints)
## 핵심 아이디어                            ← Part 2 시작
## <알고리즘별 상세 섹션들>
## 전체 실행 흐름
## 파라미터 요약                            ← 부속
## 주의사항 및 응용 고려사항                 ← 부속
```

본보기: `requirement_docs/algorithm/pw_cp_constructor_run.md` (가장 정제된 예).
보조 본보기: `requirement_docs/algorithm/neh_cp.md`, `get_best_mixed_schedule_by_sequence.md`,
`get_schedule_by_bn2d_all_stages.md`.

## 본 레포 현황

5/5 중 4개가 이미 프레임 준수. 1개만 보강 필요.

| 파일 | 상태 | 비고 |
|------|------|------|
| `get_best_mixed_schedule_by_sequence.md` | ✅ 준수 | – |
| `get_schedule_by_bn2d_all_stages.md` | ✅ 준수 | – |
| `neh_cp.md` | ✅ 준수 | – |
| `pw_cp_constructor_run.md` | ✅ 준수 | canonical 본보기 |
| `initialize_by_tau_coarsened_cp.md` | ❌ 「개요」+「핵심 아이디어」만 | problem-mechanism 보강 필요 |

## 작업 항목

### 1. `initialize_by_tau_coarsened_cp.md` problem-mechanism 보강
대상 메서드를 코드에서 확인하고 「문제 설명」 4 하위 섹션, 「전체 실행 흐름」,
「파라미터 요약」, 「주의사항 및 응용 고려사항」을 추가한다.

위험: 단순 constructor 라 변수·제약이 코드에 명시적 객체로 없을 수 있다. 그 경우에도
「변수: 절차가 추적하는 incumbent / tau 격자 상태 등」, 「제약: 절차가 보장하는
스케줄 가능성 불변식」 식으로 절차 속 형태로 기술하고 하위 섹션을 생략하지 않는다.
정 어색하면 한 줄로 "이 단계에서는 별도 결정변수 없음(절차적 구성만)" 식 명시도 가능.

### 2. AGENTS.md 보강
별도 섹션 `## Algorithm Documentation` 을 추가하고 다음을 명시:
- 알고리즘 문서 경로: `requirement_docs/algorithm/`
- 컨벤션: problem-mechanism 프레임 (위 골격 그대로, Part 1 = 문제 설명 4하위, Part 2 = 메커니즘)
- 본보기 문서: `pw_cp_constructor_run.md`

### 3. 컨벤션 문서 분리 (선택)
형식 규칙을 한 곳에 두려면 `requirement_docs/algorithm-doc-convention.md` 같은
파일을 만들어 골격·톤·표현 규칙을 명문화. AGENTS.md 는 한 줄로 가리키기만.
(agent-skills 의 `algorithm-doc-kr/references/format.md` 내용을 옮겨오면 됨.)

### 4. 스킬 배치 (별도 단계, 위 1–3 이후)
`agent-skills` 의 `algorithm-doc-kr` 를 본 레포 내 프로젝트 스킬 경로로 이동.
- 호스트가 Claude Code 면: `.claude/skills/algorithm-doc-kr/`
- 이동 후 `agent-skills/` 의 사본은 README.md 와 함께 정리.

## 자매 레포 작업 계획 (참조)

같은 통일 작업이 두 레포에서 별도로 진행됨:
- `flowshop-tardiness/plans/20260521_algorithm_doc_5element_unification.md`
- `ffc_ddw_sum_et/plans/20260521/algorithm_doc_5element_unification.md`

세 레포 작업이 모두 끝난 뒤에야 스킬 이동/사본 배치 단계로 넘어간다.

## 체크리스트

- [ ] `initialize_by_tau_coarsened_cp.md` problem-mechanism 보강
- [ ] AGENTS.md 에 알고리즘 문서 경로·컨벤션 명시
- [ ] (선택) `algorithm-doc-convention.md` 분리
- [ ] 스킬 이동 — 자매 레포 통일 완료 후
