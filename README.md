# 공공조직 생성형 AI 활용 연구 분석 프로젝트

## 프로젝트 개요

이 저장소는 공공조직 공무원의 생성형 AI 챗봇 활용에 관한 행정논문 분석을 재현하기 위한 Python 기반 분석 프로젝트이다. 연구의 핵심 관심은 공무원의 자발적 AI 활용동기가 인식된 업무효과를 통해 전략적 활용 기대에 어떤 영향을 미치는지 검증하는 데 있다. 또한 조직지원 인식은 전략적 활용 기대의 독립적 선행요인으로 설정하여 함께 분석한다.

이 프로젝트는 애플리케이션 구조가 아니라, 하나의 정제 데이터셋을 여러 독립 실행형 분석 스크립트가 공통으로 읽는 재현형 분석 구조로 구성되어 있다.

## 연구목적

본 연구의 목적은 다음과 같다.

1. 공무원의 자발적 생성형 AI 활용동기가 인식된 업무효과에 미치는 영향을 검증한다.
2. 인식된 업무효과가 전략적 활용 기대에 미치는 영향을 검증한다.
3. 자발적 AI 활용동기가 전략적 활용 기대에 미치는 직접효과를 검증한다.
4. 인식된 업무효과의 매개효과를 Sobel 검정과 bootstrap으로 검증한다.
5. 조직지원 인식이 전략적 활용 기대에 미치는 독립적 효과를 검증한다.
6. 두 독립변수의 상대적 영향력과 결과의 강건성을 추가 분석으로 확인한다.

## 연구모형

- 독립변수: 자발적 AI 활용동기 (`motivation`)
- 매개변수: 인식된 업무효과 (`effect`)
- 종속변수: 전략적 활용 기대 (`expectation`)
- 추가 독립변수: 조직지원 인식 (`support`)
- 통제변수: 성별 (`gender`), 직급 (`rank_code`), 근속연수 (`career_code`)

가설적 경로는 다음과 같다.

- `motivation -> effect`
- `effect -> expectation`
- `motivation -> expectation`
- `support -> expectation`

## 데이터 개요

- 원자료: `chatbot_input.SAV`
- 중간 변환 파일: `chatbot_output.csv`
- 최종 분석용 데이터: `chatbot_output_selected_preprocessed.csv`
- 전체 응답자 수: 1,608명
- AI 활용자 수: 377명

본 연구의 핵심 분석은 AI 활용 경험이 있는 응답자만을 대상으로 수행한다. 이는 연구질문 자체가 AI 활용 경험이 있는 공무원 내부에서, 활용동기와 업무효과 인식이 전략적 활용 기대 형성에 어떤 방식으로 연결되는지 검토하는 데 있기 때문이다. 따라서 본 연구의 추론 범위는 전체 공무원 집단이 아니라 AI 활용자 하위집단의 인식 메커니즘에 한정된다. 대부분의 분석 스크립트는 전체 전처리 파일을 읽은 후 `Q3 == 1` 조건으로 AI 활용자 집단만 다시 필터링한다.

## 전처리 및 분석 흐름

전체 흐름은 다음과 같다.

1. `01_convert.py`
   SPSS 원자료(`chatbot_input.SAV`)를 Python에서 다룰 수 있는 CSV 형식으로 변환한다.

2. `02_preprocessed.py`
   분석에 필요한 변수만 추출하고, 문자형 응답을 숫자형으로 변환하며, 통제변수를 재코딩한 뒤 분석용 데이터셋을 생성한다.

3. `03_analysis_ai_group.py`
   AI 활용자 집단을 대상으로 신뢰도 분석, 탐색적 요인분석, 기술통계, 통합 EFA를 수행한다.

4. `05_measurement_validity.py`
   AVE, CR, Fornell-Larcker, HTMT 및 2문항 척도의 Spearman-Brown 계수를 통해 수렴타당성과 판별타당성을 점검한다.

5. `04_regression_analysis_ai_users.py`
   상관분석, HC3 robust 위계적 회귀분석, VIF 점검, Sobel 검정, bootstrap 매개효과 분석을 수행한다.

6. `09_cmv_harman.py`
   Harman의 단일요인 검정을 통해 공통방법편의 가능성을 점검한다.

7. `15_compare_ai_users_nonusers.py`
   AI 활용자와 비활용자의 표본 특성을 비교하여 표본 선택편의 우려를 점검하기 위한 기초 비교표를 생성한다.

8. `16_ai_use_selection_model.py`
   AI 활용 여부를 종속변수로 한 이항 회귀모형을 통해 표본 선택 과정과 자기선택 가능성을 보조적으로 점검한다.

9. `11_support_motivation.py`
   조직지원 인식이 자발적 AI 활용동기의 선행요인으로 작동하는지 대안모형을 검토한다.

10. `12_compare_two_ivs.py`
   자발적 AI 활용동기와 조직지원 인식의 상대적 영향력을 표준화계수 기준으로 비교한다.

11. `13_robustness_q16_7.py`
   조직지원 척도에서 `Q16_7` 문항 포함 여부에 따라 결과가 유지되는지 강건성 검정을 수행한다.

12. `14_bootstrap_support.py`
   조직지원 인식의 간접효과(`support -> effect -> expectation`)를 Sobel 및 bootstrap 방식으로 검토한다.

13. `17_measurement_validity_q16_excluded.py`
   조직지원 척도에서 `Q16_7` 포함 여부에 따른 AVE, CR, Fornell-Larcker, HTMT 변화를 비교하여 측정타당성의 강건성을 점검한다.

## 전처리 세부 내용

`02_preprocessed.py`에서는 다음 작업을 수행한다.

- 분석에 필요한 문항만 선택
- AI 활용 여부 문항 `Q3`를 0/1로 변환
- 업무유형 복수응답 `Q4`를 0/1 더미로 변환하고 `ai_task_count` 생성
- 리커트형 문항(`Q7`, `Q9`, `Q16`, `Q20`)을 1~5 숫자형으로 변환
- 성별, 직급, 근속연수 문항을 회귀분석용 코드 변수로 재구성
- 전체 표본 기반의 분석용 CSV 파일을 저장

## 변수 구성 방식

주요 구성개념은 평균척도(mean scale) 방식으로 산출한다.

- `motivation`: `Q9_3`, `Q9_4` 평균
- `effect`: `Q7_1` ~ `Q7_5` 평균
- `support`: `Q16_1` ~ `Q16_6` 평균
- `expectation`: `Q20_2` ~ `Q20_4` 평균

`motivation`은 Q9 전체를 단일척도로 사용하지 않고, 탐색적 요인분석에서 자발적 활용동기로 구분된 `Q9_3`, `Q9_4`만을 사용한다. 이는 Q9 문항이 수동적 동기와 자발적 동기로 분리되는 구조를 보였고, 본 연구가 자기결정성이론에 따라 자발적 활용동기를 핵심 독립변수로 설정하기 때문이다.

본 연구는 자발적 활용동기의 효과가 전략적 활용 기대에 직접 연결되기보다, 우선 AI의 실제 업무효과에 대한 인식을 통해 번역된다고 본다. 따라서 최종모형에서 자발적 활용동기의 직접효과가 약화되거나 비유의적으로 나타나더라도, 이는 연구가설의 실패라기보다 동기의 효과가 인지된 업무성과 경험을 통해 전달되는 완전 또는 준완전 매개 구조의 가능성을 시사하는 결과로 해석한다.

`support`에서 `Q16_7`("I am interested in using generative AI chatbots for my work")은 조직의 지원 환경이 아닌 개인의 관심·흥미를 측정하는 문항으로, 조직지원 구성개념의 내용타당도를 저해하고 독립변수(`motivation`)와 개념적으로 중첩된다. 이에 이론적 근거에 따라 제외하였다(Q16_1~Q16_6, 6문항).

`expectation`에서 `Q20_1`("AI will improve work efficiency")은 매개변수인 인식된 업무효과(`effect`)와 개념적으로 중첩되어 측정 오염의 우려가 있으므로 제외하였다(Q20_2~Q20_4, 3문항).

추가적으로 일부 보조분석에서는 다음 변수가 사용된다.

- `support_full`: `Q16_1` ~ `Q16_7` 평균 (강건성 비교용, `13_robustness_q16_7.py`)
- `z_*`: 표준화계수 비교를 위한 z-score 변수

`support`는 본 연구에서 매개변수나 조절변수가 아니라, 공공조직의 AI 활용 맥락을 반영하는 독립적 조직맥락 변수로 정의한다. 즉 개인의 활용동기와 별도로, 조직이 AI 활용을 지원하고 장려하는 환경이 전략적 활용 기대를 직접 형성할 수 있다는 조직지원 관점에 따라 본모형에 포함한다.

## 연구방법 요약

이 프로젝트에서 실제 논문 본문에 활용되는 분석방법은 아래와 같다.

- 기술통계
- 신뢰도 분석(Cronbach's alpha)
- 탐색적 요인분석(EFA)
- 수렴타당성 및 판별타당성 검토(CR, AVE, 판별타당성)
- 상관분석
- HC3 robust 위계적 회귀분석
- Sobel 검정
- bootstrap 매개효과 분석(BCa 신뢰구간)
- Harman의 단일요인 검정
- 대안모형 및 강건성 검정

본 연구는 활용동기 문항이 수동적 동기와 자발적 동기의 2차원으로 분리되는 특성을 보여, 확인적 접근보다 탐색적 접근을 우선 적용한다. 따라서 측정타당성 평가는 EFA를 중심으로 정리하고, CR, AVE, Fornell-Larcker, HTMT를 보조적 근거로 활용한다. CFA 적합도 결과는 해석의 참고자료로 제한하며, 본문 주장 자체는 탐색적 측정구조와 이론적 정합성에 기반하여 제시하는 것을 원칙으로 한다.

## 부트스트랩 모형 원칙

매개효과 bootstrap은 본 회귀모형과 동일한 통제변수 구조를 유지하도록 맞춰져 있다.

- 핵심 매개효과: `effect ~ motivation + gender + rank_code + career_code`
- 결과식: `expectation ~ motivation + effect + support + gender + rank_code + career_code`
- 조직지원 간접효과: `effect ~ motivation + support + gender + rank_code + career_code`

즉 Sobel 검정, 위계적 회귀, bootstrap 추정식이 동일한 통제변수 체계를 공유한다.

## 주요 스크립트 역할 정리

| 파일                                 | 역할                           | 논문 활용도    |
| ------------------------------------ | ------------------------------ | -------------- |
| `01_convert.py`                      | 원자료 SAV -> CSV 변환         | 방법론 보조    |
| `02_preprocessed.py`                 | 전처리 및 분석용 데이터셋 생성 | 방법론 필수    |
| `03_analysis_ai_group.py`            | 신뢰도, EFA, 기술통계          | 본문 필수      |
| `05_measurement_validity.py`         | AVE, CR, HTMT, Fornell-Larcker | 본문 필수      |
| `04_regression_analysis_ai_users.py` | 상관, 회귀, 매개효과           | 본문 핵심      |
| `09_cmv_harman.py`                   | CMV 점검                       | 본문 또는 부록 |
| `15_compare_ai_users_nonusers.py`    | AI 활용자/비활용자 집단 비교   | 방법론 보조    |
| `16_ai_use_selection_model.py`       | AI 활용 여부 선택모형          | 방법론 보조    |
| `11_support_motivation.py`           | 대안모형 검토                  | 보조분석       |
| `12_compare_two_ivs.py`              | 두 독립변수 영향력 비교        | 본문 또는 부록 |
| `13_robustness_q16_7.py`             | 문항 제외 강건성 검정          | 보조분석       |
| `14_bootstrap_support.py`            | 조직지원 간접효과 검정         | 보조분석       |
| `17_measurement_validity_q16_excluded.py` | Q16_7 제외 측정타당성 비교 | 보조분석       |

## 생성 결과 파일

분석 스크립트 실행 시 다음 결과 파일이 생성될 수 있다.

- `analysis_results.md`
- `robustness_q16_7_results.md`
- `bootstrap_support_indirect.md`
- `sample_group_comparison.md`
- `ai_use_selection_model.md`
- `measurement_validity_q16_excluded.md`

이 파일들은 본문 표 작성, 부록 정리, 결과문장 초안 작성에 참고할 수 있는 산출물이다.

## 활용 패키지

프로젝트에서 사용하는 주요 패키지는 다음과 같다.

- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `pyreadstat`
- `factor_analyzer`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pingouin`
- `semopy`
- `tqdm`

설치 버전은 `requirements.txt`를 따른다.

## 실행 환경

- Python 3.x
- Windows PowerShell 기준 실행 확인
- 가상환경 사용 권장

패키지 설치 예시는 다음과 같다.

```bash
pip install -r requirements.txt
```

## 권장 실행 순서

논문 재현 목적이라면 아래 순서대로 실행하는 것이 가장 안전하다.

```bash
python 01_convert.py
python 02_preprocessed.py
python 03_analysis_ai_group.py
python 05_measurement_validity.py
python 04_regression_analysis_ai_users.py
python 09_cmv_harman.py
python 15_compare_ai_users_nonusers.py
python 16_ai_use_selection_model.py
python 12_compare_two_ivs.py
python 13_robustness_q16_7.py
python 14_bootstrap_support.py
python 17_measurement_validity_q16_excluded.py
```

`11_support_motivation.py`는 대안모형 검토가 필요할 때 선택적으로 실행하면 된다.

본문 중심의 최소 실행 순서는 아래와 같다.

```bash
python 01_convert.py
python 02_preprocessed.py
python 03_analysis_ai_group.py
python 05_measurement_validity.py
python 04_regression_analysis_ai_users.py
python 09_cmv_harman.py
python 15_compare_ai_users_nonusers.py
python 16_ai_use_selection_model.py
```

## 논문 본문 반영 권장 범위

본문 중심으로 사용할 스크립트는 다음과 같이 정리할 수 있다.

- 연구방법: `01`, `02`, `03`, `05`, `04`, `09`
- 표본 선택 보조분석: `15`, `16`
- 연구결과: `03`, `05`, `04`
- 보조분석 또는 부록: `11`, `12`, `13`, `14`, `17`

다만 두 독립변수 비교 결과를 본문에서 강조하려면 `12`를 결과 본문에 포함할 수 있다.

## 해석 및 한계

- 본 연구는 AI 활용자 집단만을 대상으로 하므로, 결과를 AI 비활용자를 포함한 전체 공무원 집단으로 직접 일반화하기 어렵다.
- AI 활용 여부 자체가 자기선택의 결과일 수 있으므로, 표본 선택편의 가능성은 `15_compare_ai_users_nonusers.py`의 집단 비교표와 함께 해석해야 한다.
- Harman의 단일요인 검정은 공통방법편의를 완전히 배제하는 강한 검정이 아니므로, 결과 해석 시 그 한계를 함께 명시해야 한다. 본 연구는 단일 시점 자기보고 설문 데이터를 사용하므로 공통방법편의의 가능성을 완전히 배제하기 어렵다. Harman 단일요인 검정은 보조적 점검으로만 활용하며, 향후 연구에서는 다시점 설계 또는 marker variable 접근을 적용할 필요가 있다.
- 횡단면 자료에 기반한 회귀 및 매개분석은 변수 간 가설적 방향성과 정합성을 보여주지만, 엄밀한 시간적 인과성을 확정하지는 못한다.
- `support`는 본모형에서 전략적 활용 기대의 독립적 조직맥락 요인으로 정의한다. `11`, `14`의 추가 경로는 대안 가능성을 점검하는 보조분석이며, 본 연구의 주된 이론모형으로 해석하지 않는다.

## 유의사항

- 일부 원본 파일은 한글 인코딩 상태에 따라 터미널에서 문자가 깨져 보일 수 있다.
- 분석결과는 표본 필터링과 결측 처리 방식에 따라 달라질 수 있으므로, 스크립트 실행 전 입력 파일명을 확인해야 한다.
- bootstrap 반복 횟수는 계산 시간이 다소 소요될 수 있다.

## 라이선스 및 데이터 주의

원자료는 외부 조사자료에 기반함
