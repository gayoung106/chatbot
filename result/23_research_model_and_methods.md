# 연구모형과 연구방법 정리

## 1. 현재 연구모형

현재 전략은 `Q16_1~Q16_4`를 조직지원 인식의 관측지수로 사용하고, 이를 `motivation`과 병렬로 놓이는 두 번째 독립변수로 해석한 뒤 `Q20_1~Q20_3`를 main item-level outcome으로 분석하는 것이다. `Q20_4`는 supplementary outcome으로 분리한다.

- 독립변수 1: `motivation = mean(Q9_3, Q9_4)`
- 독립변수 2: `support_main = mean(Q16_1, Q16_2, Q16_3, Q16_4)` = 조직지원 인식
- 매개변수: `effect = mean(Q7_1~Q7_5)`
- 종속변수:
  - `Q20_1`: 업무효율 개선 기대
  - `Q20_2`: 의사결정 지원 기대
  - `Q20_3`: 반복업무 자동화 기대
  - `Q20_4`: 일자리 대체 인식 (`supplementary`)

핵심은 더 이상 Q20을 하나의 composite으로 처리하지 않는다는 점과, `motivation`과 `support_main`을 상호작용항 없이 병렬로 투입하는 dual-driver 구조라는 점이다. 생성형 AI 기대는 단일한 평균기대가 아니라 서로 다른 내용영역의 전략적 기대 유형으로 본다.

## 2. 왜 이렇게 바꾸는가

### 2-1. Q16은 `Q16_1~Q16_4`만 유지

`Q16_1~Q16_4`는 조직의 강조, 지원, 구성원 태도, 구성원 사용을 직접 반영한다. 따라서 이 네 문항은 조직 차원의 AI 지원 분위기 또는 organizational AI support climate를 가장 직접적으로 포착하는 조합이다.

현재 해석 원칙은 다음과 같다.

- `support_main`은 강한 latent construct로 주장하지 않는다.
- `support_main`은 조직 맥락을 나타내는 보수적 observed index로 사용한다.
- 따라서 방법 서술에서도 `Q16_1~Q16_4` 평균지수라는 점을 분명히 한다.

### 2-2. Q20은 세 문항을 main item-level outcome으로 처리

이전 단계에서는 `Q20_1~Q20_3` composite, `Q20_1~Q20_4` composite, 또는 dual-DV split을 검토했다. 그러나 현재 전략은 그 어느 것도 메인 분석으로 두지 않는다.

이유는 다음과 같다.

- Q20 문항들은 내용영역이 서로 다르다.
- composite로 묶을수록 해석이 흐려진다.
- main outcome을 Q20_1~Q20_3으로 두면 positive strategic expectancy 내부의 차이를 직접 보여줄 수 있다.
- Q20_4는 일자리 대체 인식이므로 supplementary로 따로 제시하는 편이 이론적으로 더 명확하다.

따라서 현재 방법론적 기준은 `Q20_1`, `Q20_2`, `Q20_3`에 동일한 분석식을 적용하고, `Q20_4`는 supplementary analysis로만 별도 보고하는 것이다.

## 3. 연구모형 도식

```text
motivation ---------> Q20_1
     |                Q20_2
     |                Q20_3
     |                Q20_4
     v
   effect ----------> Q20_1
     ^               Q20_2
     |               Q20_3
     |               Q20_4
     |
support_main ------> Q20_1
                     Q20_2
                     Q20_3
                     Q20_4
```

이 모형의 해석 포인트는 다음과 같다.

- `motivation`은 개인 차원의 자발적 AI 활용동기다.
- `support_main`은 조직 차원의 지원 맥락을 나타내는 두 번째 독립변수다.
- `effect`는 두 변수의 영향을 받아 형성되는 인식된 업무효과다.
- `Q20_1~Q20_3`은 서로 다른 positive strategic expectancy의 결과항목이다.
- `Q20_4`는 job replacement perception을 나타내는 supplementary outcome이다.

## 4. 연구방법

### 4-1. 분석대상

전체 응답자 1,608명 중 생성형 AI 사용 경험이 있다고 응답한 공무원 377명을 분석대상으로 한다.

### 4-2. 변수 구성

- `motivation = mean(Q9_3, Q9_4)`
- `effect = mean(Q7_1, Q7_2, Q7_3, Q7_4, Q7_5)`
- `support_main = mean(Q16_1, Q16_2, Q16_3, Q16_4)`
- outcomes:
  - `Q20_1`
  - `Q20_2`
  - `Q20_3`
  - `Q20_4`

통제변수는 `gender`, `rank_code`, `career_code`다.

### 4-3. 측정 해석

현재 문서에서는 다음 원칙을 따른다.

- `support_main`은 관측합성지수로 보고한다.
- Q20에 대해서는 composite 신뢰도나 1요인 CFA 적합도를 메인 정당화 논리로 쓰지 않는다.
- Q20은 네 개의 결과항목으로 직접 분석한다.

### 4-4. 분석식

각 결과항목에 대해 동일한 구조의 회귀 및 매개분석을 수행한다.

1. `effect ~ motivation + support_main + gender + rank_code + career_code`
2. `Q20_k ~ motivation + support_main + gender + rank_code + career_code`
3. `Q20_k ~ motivation + support_main + effect + gender + rank_code + career_code`

여기서 `k = 1, 2, 3`이다.

추가 원칙은 다음과 같다.

- HC3 robust standard errors 사용
- BCa bootstrap 5,000회 사용
- 결과는 문항별 총효과, 직접효과, 간접효과를 함께 보고

## 5. 결과 해석 방향

현재 데이터에서 보이는 패턴은 다음과 같이 요약된다.

- `Q20_1`: motivation과 effect의 영향이 가장 강하다.
- `Q20_2`: motivation과 effect 중심이지만 support_main의 간접경로도 존재한다.
- `Q20_3`: support_main과 effect가 강하게 작동한다.
- `Q20_4`: supplementary에서 support_main의 직접효과와 총효과가 모두 유의하다.

즉, 연구의 포인트는 `어떤 Q20 문항을 남기고 뺄 것인가`가 아니라, `자발적 활용동기와 조직지원 인식이라는 두 독립변수가 각 전략적 기대 문항에서 어떻게 다르게 작동하는가`에 있다.

## 6. 논문용 핵심 문장

`본 연구는 생성형 AI에 대한 전략적 기대를 단일 composite으로 가정하지 않고, Q20_1~Q20_3을 개별 결과항목으로 분석하였다. 또한 자발적 AI 활용동기와 조직지원 인식을 병렬적인 두 독립변수로 설정하고, 이들이 인식된 업무효과를 매개로 각 전략적 기대 결과항목에 미치는 영향을 HC3 회귀와 BCa bootstrap으로 추정하였다. Q20_4는 job replacement perception을 측정하는 supplementary outcome으로만 제시하였다.`
