# 2026-04-07 기준 value over time plotting 변경사항

이 문서는 5개 commit에서 반영된 그래프 동작을 요약한다.

- `70ca7114` `feat(report): show best-so-far RPD progression`
- `c1b0d6e5` `refactor(chart): fix best-so-far progression logic`
- `425fde39` `fix(report): correct mean RPDf progression by (n,c)`
- `1d2b3728` `fix multi_scenario_subroutine_flow_comparison.html`
- `1b9f57fd` `add hover tooltops in mean regression plot`

핵심 변화는 세 종류의 HTML 그래프가 더 이상 endpoint만 대각선으로 잇는 방식이 아니라, 명시적인 value-over-time progression을 바탕으로 그려진다는 점이다.

## 공통 progression model

이제 plotting 코드는 각 instance를 normalized time 상의 explicit한 best-so-far progression으로 다룬다.

$$P_i = \bigl((t_{i,1}, r_{i,1}), (t_{i,2}, r_{i,2}), \ldots \bigr)$$

여기서:

- $t_{i,j}$ 는 normalized time (`norm_time`)
- $r_{i,j}$ 는 그 시점까지의 best-so-far `RPDf`
- time은 오름차순으로 정렬된다
- 같은 time이 여러 번 나오면 하나의 effective value만 남기도록 dedupe 한다

이 점들은 각 call의 `subroutine_progression.json` 안에 있는
`local_progress_list` 에서 나온다. 따라서 선 자체는 subroutine endpoint만이 아니라, call 내부에서 기록된 전체 progression을 사용한다.

계단형(step-line) 렌더링 규칙은 다음과 같다.

- 다음 시점까지는 수평으로 이동
- 새 best value가 나오면 그 same time에서 수직으로 하강

즉 아래의 모든 progression line은 대각선 보간이 아니라 계단형 곡선이다.

## 1. `summary_method_rpdf_and_norm_time_scatter.html`

이 scenario-level HTML은 두 개의 mode를 유지한다.

- `instance progression`
- `mean progression by (job_cnt, stage_cnt)`

### 1.1 `instance progression`

각 instance의 raw line은 `local_progress_list` 기반의 전체 progression을
best-so-far `RPDf`로 변환해서 그린다.

subroutine endpoint marker는 계속 표시되지만, marker 위치 계산은 line 계산과 분리되었다.

- marker의 `x` 는 실제 endpoint `norm_time`
- marker의 `y` 는 그 시점의 선 위 값, 즉 그 `x` 이하에서 가장 최신 best-so-far value

따라서 endpoint 자체 값이 더 나쁘더라도 marker는 항상 실제로 그려진 stair-step curve 위에 놓인다.

### 1.2 `mean progression by (job_cnt, stage_cnt)`

이 mode는 더 이상 subroutine별 endpoint row를 평균하지 않는다. 대신 선택된
`(job_cnt, stage_cnt)` 그룹 안의 instance progression들을 시간축 위에서 합성한 over-time mean curve를 그린다.

한 그룹에 대해:

- 첫 time point:
  모든 instance가 최소 1개의 값을 가지게 되는 가장 이른 시각, 즉 각 instance의 첫 시각들의 최댓값
- 마지막 time point:
  각 instance의 마지막 시각들의 최댓값
- 중간 time point:
  위 구간 안에 있는 모든 instance progression time의 union을 정렬한 값

시각 $t$ 에서의 mean 값은:

$$\bar r(t)=\frac{1}{|G|}\sum_{I_i \in G} r_i^\ast(t)$$

여기서 $r_i^\ast(t)$ 는 $t$ 이하에서 가장 최신의 best-so-far `RPDf` 이다.
어떤 instance가 다른 것보다 더 빨리 끝나면, 이후 시각에서는 그 instance의 마지막 값을 carry-forward 해서 평균에 포함한다.

### 1.3 Mean subroutine guide marker

`mean progression by (job_cnt, stage_cnt)` 에서는 subroutine endpoint를 mean line 위의 점으로 찍지 않고, average-time guide로 표시한다.

각 subroutine 이름에 대해:

- 그룹 내 instance들의 endpoint `norm_time` 평균 계산
- 그 평균 시간에 수직 점선 표시
- 그 점선이 x축과 만나는 지점에 marker 표시

guide line과 x축 marker는 해당 mean curve와 같은 색을 사용한다.

hover 동작은 다음과 같다.

- mean line hover는 활성화되어 있고 series와 `(job_cnt, stage_cnt)` 정보를 보여준다
- x축 guide marker hover도 활성화되어 있고 subroutine 이름과 평균 종료 시각을 보여준다
- raw line hover는 비활성화되어 있고, raw endpoint marker hover는 활성화되어 있다

## 2. `multi_scenario_subroutine_flow_comparison.html`

이 top-level comparison HTML은 더 이상 scenario별 endpoint scatter line을 그리지 않는다. 대신 scenario별 over-time mean progression curve를 그린다.

각 scenario에 대해:

- `subroutine_progression.json` 에서 instance별 best-so-far progression을 구성
- `mean progression by (job_cnt, stage_cnt)` 와 같은 규칙으로 시간축 위에서 평균
- scenario당 하나의 stair-step mean `RPDf` curve를 렌더링

즉 이 그래프는 scenario level에서의 "평균의 평균 over time" 비교라고 볼 수 있다.

- scenario 내부에서는 instance 평균
- scenario 사이에서는 그렇게 만들어진 scenario-level mean curve 비교

subroutine 평균 종료 시각도 scenario별로 다음 형태로 표시한다.

- 수직 점선 guide
- x축 접점 marker

hover 동작은 다음과 같다.

- scenario mean line hover는 활성화
- x축 guide marker hover도 활성화되어 있고 scenario 이름, subroutine 이름, 평균 종료 시각을 보여준다

## 3. Source selection 과 fallback

세 종류의 그래프는 모두 가능하면 progression JSON을 우선 사용한다.

### Scenario-level chart

`summary_method_rpdf_and_norm_time_scatter.html` 은 다음을 사용한다.

- endpoint marker용으로는 `subroutine_progression.json` 에서 복원한 endpoint row
- raw 및 mean stair-step line용으로는 `subroutine_progression.json` 에서 복원한 raw progression row

유효한 JSON endpoint metric을 만들 수 없으면 기존 endpoint CSV 경로로 fallback 한다.

- `summary_method_rpdf_and_norm_time_long.csv`

### Top-level multi-scenario chart

`multi_scenario_subroutine_flow_comparison.html` 은 scenario별로 가능한 경우 JSON progression을 사용한다. 어떤 scenario에서 JSON progression을 쓸 수 없으면, 그 scenario는 endpoint CSV로 fallback 하고, 어떤 scenario가 fallback 되었는지 식별 가능한 warning log를 남긴다.

selected-scenario comparison script도 같은 exporter shape를 사용하지만, 현재 구조상 endpoint CSV를 직접 읽고 warning log를 남기도록 되어 있다.

## 4. 현재 그래프 해석

이 다섯 개 commit 이후, 세 종류 그래프는 다음처럼 해석해야 한다.

- scenario raw chart:
  full per-instance best-so-far progression이며, endpoint marker는 선 위에 맞춰진다
- scenario grouped-mean chart:
  각 `(job_cnt, stage_cnt)` 그룹에 대해 instance progression들을 over-time 평균한 curve
- top-level multi-scenario chart:
  각 scenario에 대해 instance progression들을 over-time 평균한 curve를 scenario 간 비교

즉 세 경우 모두, 그래프의 선은 endpoint를 단순히 잇는 선이 아니라 best-so-far value-over-time trajectory를 계단형으로 시각화한 것이라고 봐야 한다.
