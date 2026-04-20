# 공공조직 생성형 AI 사용 연구 분석 프로젝트

## 개요

이 프로젝트는 공공조직 공무원의 생성형 AI 사용 경험과 기대를 분석하기 위한 Python 기반 분석 저장소다. 현재 기준 전략은 다음과 같다.

- 조직 맥락 독립변수: `support_main = mean(Q16_1~Q16_4)`
- 매개변수: `effect = mean(Q7_1~Q7_5)`
- 개인 동기 변수: `motivation = mean(Q9_3, Q9_4)`
- 주종속변수: `Q20_1`, `Q20_2`, `Q20_3` item-level outcome
- 보조종속변수: `Q20_4` supplementary outcome

즉, 더 이상 Q20을 하나의 composite으로 묶지 않는다. 주분석은 `Q20_1~Q20_3` 각각에 대해 동일한 회귀 및 매개모형을 적용하고, `Q20_4`는 supplementary analysis로 분리한다.

## 연구목적

1. 공무원의 자발적 AI 활용동기(`motivation`)가 인식된 업무효과(`effect`)에 미치는 영향을 검증한다.
2. 조직 차원의 AI 지원 분위기(`support_main`)가 인식된 업무효과(`effect`)에 미치는 영향을 검증한다.
3. `motivation`과 `support_main`을 병렬적인 두 독립변수로 두고, 각 전략적 기대 문항(`Q20_1`~`Q20_3`)에 미치는 총효과와 직접효과를 검증한다.
4. `effect`를 매개로 한 간접효과를 BCa bootstrap으로 검증한다.
5. 생성형 AI 기대가 단일한 척도가 아니라 내용영역별로 다르게 형성되는지를 item-level 결과로 확인한다.

## 데이터

- 원자료: `chatbot_input.SAV`
- 중간 변환 파일: `chatbot_output.csv`
- 최종 분석용 데이터: `chatbot_output_selected_preprocessed.csv`
- 전체 응답자: 1,608명
- AI 사용 경험자: 377명

대부분의 분석은 AI 사용 경험이 있는 응답자만을 대상으로 하며, 스크립트에서는 `Q3 == 1` 조건으로 필터링한다.

## 변수 구성

- `motivation`: `Q9_3`, `Q9_4` 평균
- `effect`: `Q7_1` ~ `Q7_5` 평균
- `support_main`: `Q16_1` ~ `Q16_4` 평균

### Q20 item-level outcomes

- `Q20_1`: AI will improve work efficiency
- `Q20_2`: AI will support decision-making processes
- `Q20_3`: AI will automate routine tasks
- `Q20_4`: AI will replace some human jobs (`supplementary`)

## 핵심 스크립트

1. `01_convert.py`
   SPSS 원자료를 CSV로 변환한다.

2. `02_preprocessed.py`
   분석에 필요한 변수를 정리하고 전처리된 CSV를 생성한다.

3. `03_analysis_ai_group.py`
   AI 사용 집단의 기초통계와 척도 구조를 탐색한다.

4. `09_cmv_harman.py`
   Harman 단일요인 검정을 수행한다.

5. `15_compare_ai_users_nonusers.py`
   AI 사용자와 비사용자 집단을 비교한다.

6. `16_ai_use_selection_model.py`
   AI 사용 여부 선택모형을 추정한다.

7. `19_parallel_mediation_hc3_bca.py`
   현재 메인 결과 보고용 HC3 회귀와 BCa bootstrap 결과를 정리한 스크립트다.

8. `28_item_level_expectancy_models.py`
   현재 메인 전략 스크립트다. `Q20_1~Q20_3`를 main item-level outcome으로, `Q20_4`를 supplementary outcome으로 분석한다.

## 메인 결과 문서

- `result/02_manuscript_revision_draft.md`
- `result/23_research_model_and_methods.md`
- `result/24_model_and_methods_for_manuscript.md`
- `result/15_compare_ai_users_nonusers.md`
- `result/19_parallel_mediation_hc3_bca.md`
- `result/28_item_level_expectancy_models.md`
- `result/29_item_level_thesis_rewrite_ko.md`
- `result/30_research_model_diagram.md`

## 현재 방법론적 입장

현재 원고는 다음 원칙을 따른다.

- `Q16_1~Q16_4`는 조직 AI 지원 분위기를 나타내는 보수적 observed index로 사용한다.
- Q20은 composite 신뢰도나 CFA 적합도로 방어하지 않고, `Q20_1~Q20_3`를 주결과항목으로 다룬다.
- `Q20_4`는 일자리 대체 인식이므로 supplementary outcome으로만 제시한다.
- 모든 결과항목에 동일한 HC3 회귀 및 BCa bootstrap 매개모형을 적용한다.
- 해석의 초점은 "자발적 활용동기와 조직지원 인식이라는 두 독립변수가 어떤 전략적 기대에서 각각 더 강하게 작동하는가"에 둔다.

## 사용 패키지

- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `pyreadstat`
- `factor_analyzer`
- `pingouin`
- `semopy`

## 실행 환경

- Python 3.x
- Windows PowerShell 기준 실행 확인

```bash
pip install -r requirements.txt
```
