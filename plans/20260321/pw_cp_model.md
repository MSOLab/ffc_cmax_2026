# 코드 변경 계획

## 1. 목표

PW-CP subproblem에서
**right-time-fixed ops와 그 직전 operation들이 모든 stage, 모든 machine에서 공통 간격 `x` 만큼 이격되는 schedule**을 찾도록 모델을 재구성한다.

핵심 의도는 다음과 같다.

* right-time-fixed ops를 prefix-window의 **절대 경계**처럼 작동하게 만든다.
* 그 이전 operation들도 각 stage에서 동일한 여유량을 갖도록 만들어, 전체 prefix 부분을 **semi-definite**하게 정렬한다.
* 이렇게 얻어진 해를 후처리에서 **semi-active schedule**로 정리했을 때, **makespan을 최소 `x` 만큼 줄일 수 있는 구조적 보장**을 얻는다.

---

## 2. 기존 구현과 한계

### 2.1 기존 구현 요약

현재 구현은 대략 다음 구조이다.

* PW-CP subproblem 모델에 **모든 operation**을 포함한다.
* right-time-fixed ops는 **시간이 고정된 interval variable**로 둔다.
* slack interval은

  * stage별
  * machine 수만큼
    정의되며,
  * machine assignment는 고정되지 않고
  * end time만 고정
  * length는 결정변수인 interval variable이다.
* slack interval의 end time은
  incumbent schedule에서 해당 stage의 각 machine에 배치된 right-time-fixed ops 시작 시각들 중 **가장 빠른 값**을 사용한다.

### 2.2 문제점

#### (1) slack에 machine assignment가 없어서 의도한 schedule 보장이 없음

현재 slack은 machine별 gap을 암시적으로 표현하려는 장치이지만,
정작 어떤 slack이 어떤 machine의 여유를 의미하는지가 명확히 결정되지 않는다.

그 결과:

* CP model이 feasible solution을 내더라도
* 그 해가 실제로는 각 machine의 prefix 구간을 의도한 방식으로 비워주는 구조인지
* 보장할 수 없다.

즉, **모델 내부 표현과 실제 machine-level schedule 의미가 분리**되어 있다.

#### (2) right-time-fixed ops가 경계선 역할을 충분히 못함

prefix-window의 의도는 right-time-fixed ops가 사실상 **오른쪽 경계**가 되는 것이다.
하지만 현재는 단순히 해당 ops의 시간만 고정했을 뿐,

* 그 이전 구간을 실제로 막는 장치가 없고
* fixed op들 사이 공간도 비어 있다.

따라서 non-fixed ops가 time-fixed ops들 사이에 들어갈 수 있다.
이 경우 결과 해를 semi-active하게 만들더라도,
**makespan 감소가 구조적으로 보장되지 않는다.**

#### (3) machine assignment 후처리 과정이 복잡하고 오류 가능성이 큼

현재 설계는 CP 해 이후에 machine assignment나 정렬을 추가로 맞춰야 하는 부분이 많다.
이 방식은 다음 문제를 만든다.

* 해석 로직이 복잡해짐
* incumbent 기반 경계와 실제 배치가 어긋날 수 있음
* 특정 stage/machine에서 예외 케이스가 쉽게 발생함

즉, 모델이 단순하지 않고 **후처리 의존성이 높다.**

---

## 3. 변경 방향

핵심 변경 방향은 다음 한 줄로 요약할 수 있다.

> **left/right-time-fixed ops를 개별 operation으로 다루지 않고, machine별 긴 dummy bar로 단순화한다.**

이렇게 하면 fixed ops를 직접 모델링하면서 생기는 복잡성을 줄이고,
prefix-window의 경계 의미를 더 직접적으로 반영할 수 있다.

변경의 중심 아이디어는 두 가지다.

* 각 machine의 오른쪽 경계를 나타내는 **dummy interval 구조**로 변환
* non-time-fixed job set에 대해서만 필요한 precedence를 추가하여
  flowshop 구조를 유지

---

## 4. 변경 모델링 상세

## 4.1 time-fixed ops의 단순화: machine별 dummy bar

### 기존

* right-time-fixed ops를 실제 operation interval로 모델링

### 변경

* left-time-fixed ops와 right-time-fixed ops를 직접 interval로 두지 않고
* 각 machine에서 이들을 대표하는 **긴 막대(dummy interval/bar)** 로 단순화

이 dummy는 실제 작업을 의미한다기보다 다음 의미를 가진다.

* 특정 machine에서 non-time-fixed ops가 들어갈 수 있는 유효 구간을 제한
* right boundary 이후 구간을 구조적으로 봉쇄
* prefix-window의 경계를 절대적으로 반영

이렇게 하면 다음 효과가 있다.

* time-fixed ops 개별 변수 수 감소
* stage/machine마다 경계를 더 명확하게 표현 가능
* fixed op들 사이에 non-fixed op가 끼어드는 현상 방지 가능

---

## 4.2 slack 정의 변경: machine별이 아니라 공통 길이 변수 기반

### 기존

* stage별 machine 수만큼 slack interval 정의
* end time만 고정되고 machine assignment 없음

### 변경

* **전 machine에 공통으로 적용되는 slack length 변수 `x`** 를 선언
* 각 machine의 right dummy interval 길이 정의에 이 `x`가 포함되도록 구성

즉, `x`는 “모든 machine이 공통으로 확보해야 하는 오른쪽 여유량” 역할을 한다.

구조적으로는 다음 의미가 된다.

* 각 machine의 right boundary 이후 dummy가 `x`만큼 더 길어지도록 만들어
* 실제 유효 작업 가능 영역을 `x`만큼 왼쪽으로 압축하는 효과를 준다.

문서 수준에서는 다음처럼 적으면 충분하다.

* 공통 slack 변수 `x >= 0` 선언
* 각 machine의 right dummy interval length는
  최소한 `(makespan - right_boundary_of_machine)`를 포함해야 하며,
* 여기에 공통 이격량 `x`를 추가 반영한다.

즉, 개념적으로:

* right dummy length
  = baseline tail coverage + common spacing `x`

이렇게 되면 machine마다 별도의 slack assignment를 해석할 필요 없이,
**모든 machine에서 동일한 압축량 `x`를 직접 강제**할 수 있다.

---

## 4.3 non-time-fixed job set에 대한 precedence 추가

time-fixed ops를 dummy bar로 단순화하면,
실제 operation-level flowshop 제약이 일부 약해질 수 있다.

이를 보완하기 위해
**non-time-fixed variable의 job set에 대해 job precedence constraint를 명시적으로 추가**한다.

의도는 다음과 같다.

* 단순화된 fixed 영역이 있어도
* non-fixed 부분의 stage 간 순서는 유지되어야 한다.
* 즉, job별 flowshop 구조가 깨지지 않도록 한다.

추가할 제약은 기본적으로 job 내부 precedence다.

* 같은 job의 연속 stage operation에 대해

  * 이전 stage 종료 <= 다음 stage 시작

필요하다면 더 강하게:

* prefix-window에서 실제로 고려 대상인 non-fixed operation subset에 대해만 적용

이렇게 하면 dummy interval로 fixed 영역을 단순화하면서도
전체 feasible region이 flowshop 의미를 잃지 않게 된다.

---

## 5. 기대 효과

## 5.1 semi-active 후처리 단순화

기존에는 CP 해 이후 machine assignment를 다시 해석하거나 보정해야 하는 여지가 컸다.
변경 후에는 각 machine의 경계가 dummy bar로 명확히 표현되므로,

* CP model이 이미 machine-level 경계 구조를 반영하고
* 후처리는 단순히 non-fixed ops를 왼쪽으로 당겨 **semi-active화**하는 수준으로 정리된다.

즉, 후처리 책임이 줄고 해석이 쉬워진다.

## 5.2 makespan 감소 보장 구조 강화

공통 slack 변수 `x`가 모든 machine에 대해 동일하게 반영되므로,
prefix 부분이 충분히 비워지는 경우,

* semi-active 변환 후
* 전체 makespan이 적어도 `x`만큼 줄어드는 구조를 기대할 수 있다.

기존처럼 “fixed op는 있었지만 실제 경계 역할은 못한 경우”를 줄일 수 있다.

## 5.3 모델 경량화

* left/right-time-fixed ops를 개별 variable로 두지 않음
* slack interval도 machine별로 복잡하게 둘 필요가 없음

따라서

* interval variable 수 감소
* optional/해석용 변수 감소
* constraint graph 단순화

효과적으로 subproblem이 더 가벼워질 가능성이 높다.

---

## 6. 구현 단계 제안

## 6.1 데이터 구조 정리

먼저 현재 로직에서 아래 개념을 분리해 두는 것이 좋다.

* left-time-fixed op 집합
* right-time-fixed op 집합
* non-time-fixed op 집합
* machine별 right boundary
* stage별 active machine 집합
* non-fixed job 집합 / non-fixed op 집합

추천하는 내부 파생 데이터:

* `stage_machine_to_right_boundary`
* `stage_machine_to_left_boundary`
  필요 시
* `non_fixed_ops`
* `non_fixed_jobs`
* `common_spacing_var` 또는 `common_slack_var`

이 단계에서 가장 중요하다.
기존 구현이 “실제 op” 중심이면, 변경 후에는 **machine 경계 중심 데이터**로 옮겨가야 한다.

---

## 6.2 time-fixed op interval 생성 로직 제거 또는 우회

기존 builder에서

* right-time-fixed ops interval 생성
* fixed start/end 설정
* 관련 no-overlap 포함

이 부분을 분리한다.

대신:

* machine별 dummy interval 생성 로직 추가
* 해당 dummy가 경계 역할을 하도록 end/start/size 관련 식 구성

즉, “실제 fixed op를 모델링”하던 곳을
“machine별 blocked bar를 모델링”하는 곳으로 교체한다.

---

## 6.3 공통 slack 변수 `x` 도입

모델에 전역 정수변수 하나를 추가한다.

예:

* `common_spacing`
* `common_tail_gap`
* `global_suffix_slack`

이 변수는 목적상 매우 중요하므로 이름이 직관적이어야 한다.

개인적으로는 아래 둘 중 하나가 좋다.

* `common_spacing`
* `global_right_gap`

가장 무난한 것은 `common_spacing`이다.

이 변수에 대해:

* 하한 0
* 필요 시 상한은 window 크기 또는 `(incumbent_makespan - earliest_right_boundary)` 등으로 설정

그리고 각 machine의 dummy 길이 식에 포함시킨다.

---

## 6.4 machine별 right dummy interval 구성

각 stage의 각 machine에 대해 right dummy interval을 만든다.

이 dummy는 다음을 반영해야 한다.

* right boundary 이후 구간을 차지
* 최소한 incumbent makespan까지 tail을 덮음
* 추가로 `common_spacing` 만큼 더 덮어
  non-fixed ops가 그만큼 더 왼쪽에 위치하도록 강제

설계 방식은 구현 프레임워크에 따라 달라질 수 있다.

가능한 방식:

* fixed end + variable size
* fixed start + variable size
* 또는 start/size/end의 선형 관계 활용

중요한 건 수식 의미다.

> machine별 유효 작업 가능 마지막 시점이 right boundary보다 `common_spacing`만큼 더 왼쪽으로 당겨지게 해야 한다.

---

## 6.5 non-fixed ops에 대한 precedence 제약 강화

non-time-fixed ops만을 대상으로

* job 내부 precedence
* 필요 시 additional stage-ordering simplification

을 명시적으로 추가한다.

권장 구현 원칙:

* fixed/dummy 영역에 의해 암묵적으로 보장될 것이라 기대하지 말고
* non-fixed job flow는 직접 제약으로 유지한다.

이 부분은 추후 디버깅에서도 중요하다.
“왜 이 op가 다음 stage보다 먼저 못 갔는가?”를 설명하기 쉬워진다.

---

## 6.6 후처리 로직 단순화

CP 해를 얻은 뒤에는 다음만 수행하면 되도록 목표를 잡는 것이 좋다.

* non-fixed ops를 각 machine에서 가능한 한 왼쪽으로 당김
* precedence와 machine capacity만 유지
* fixed/dummy 경계는 침범 금지

즉, 별도의 machine reassignment repair나 복잡한 gap 해석을 최소화한다.
