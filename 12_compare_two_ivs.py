import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import zscore
import sys

from result_utils import markdown_output

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"] = df[[f"Q7_{i}" for i in range(1,6)]].mean(axis=1)
df["support"] = df[[f"Q16_{i}" for i in range(1,7)]].mean(axis=1)
df["expectation"] = df[[f"Q20_{i}" for i in range(2,5)]].mean(axis=1)

df['z_motivation'] = zscore(df['motivation'], nan_policy='omit')
df['z_support'] = zscore(df['support'], nan_policy='omit')
df['z_effect'] = zscore(df['effect'], nan_policy='omit')
df['z_expectation'] = zscore(df['expectation'], nan_policy='omit')

df = df.dropna(subset=['motivation', 'support', 'effect', 'expectation', 'gender', 'rank_code', 'career_code'])

with markdown_output("12_compare_two_ivs.md") as result_path:
    print("# 12 조직지원인식 vs AI활용동기 영향력 비교\n")
    print(f"- 분석 표본: AI 활용자 {len(df)}명\n")
    print("## 1. 매개변수(effect)에 미치는 영향 (Path A)\n")
    model_a = smf.ols("effect ~ motivation + support + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    print("### 비표준화 계수 (Unstandardized)")
    print("```text")
    print(f"AI활용동기(motivation) -> 매개변수(effect): {model_a.params['motivation']:.4f} (p={model_a.pvalues['motivation']:.4f})")
    print(f"조직지원인식(support)  -> 매개변수(effect): {model_a.params['support']:.4f} (p={model_a.pvalues['support']:.4f})")
    print("```\n")

    model_a_z = smf.ols("z_effect ~ z_motivation + z_support + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    print("### 표준화 계수 (Standardized Beta) - 영향력 크기 비교용")
    print("```text")
    print(f"AI활용동기(motivation) -> 매개변수(effect): ß = {model_a_z.params['z_motivation']:.4f} (p={model_a_z.pvalues['z_motivation']:.4f})")
    print(f"조직지원인식(support)  -> 매개변수(effect): ß = {model_a_z.params['z_support']:.4f} (p={model_a_z.pvalues['z_support']:.4f})")
    print("```\n")

    print("## 2. 종속변수(expectation)에 미치는 영향 (총효과, Total Effect)\n")
    model_total = smf.ols("expectation ~ motivation + support + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    print("### 비표준화 계수 (Unstandardized)")
    print("```text")
    print(f"AI활용동기(motivation) -> 종속변수(expectation): {model_total.params['motivation']:.4f} (p={model_total.pvalues['motivation']:.4f})")
    print(f"조직지원인식(support)  -> 종속변수(expectation): {model_total.params['support']:.4f} (p={model_total.pvalues['support']:.4f})")
    print("```\n")

    model_total_z = smf.ols("z_expectation ~ z_motivation + z_support + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    print("### 표준화 계수 (Standardized Beta) - 영향력 크기 비교용")
    print("```text")
    print(f"AI활용동기(motivation) -> 종속변수(expectation): ß = {model_total_z.params['z_motivation']:.4f} (p={model_total_z.pvalues['z_motivation']:.4f})")
    print(f"조직지원인식(support)  -> 종속변수(expectation): ß = {model_total_z.params['z_support']:.4f} (p={model_total_z.pvalues['z_support']:.4f})")
    print("```\n")

    print("## 3. 종속변수(expectation)에 미치는 영향 (직접효과, Direct Effect - 매개변수 통제)\n")
    model_direct = smf.ols("expectation ~ motivation + support + effect + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    print("### 비표준화 계수 (Unstandardized)")
    print("```text")
    print(f"AI활용동기(motivation) -> 종속변수(expectation): {model_direct.params['motivation']:.4f} (p={model_direct.pvalues['motivation']:.4f})")
    print(f"조직지원인식(support)  -> 종속변수(expectation): {model_direct.params['support']:.4f} (p={model_direct.pvalues['support']:.4f})")
    print("```\n")

    model_direct_z = smf.ols("z_expectation ~ z_motivation + z_support + z_effect + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    print("### 표준화 계수 (Standardized Beta) - 영향력 크기 비교용")
    print("```text")
    print(f"AI활용동기(motivation) -> 종속변수(expectation): ß = {model_direct_z.params['z_motivation']:.4f} (p={model_direct_z.pvalues['z_motivation']:.4f})")
    print(f"조직지원인식(support)  -> 종속변수(expectation): ß = {model_direct_z.params['z_support']:.4f} (p={model_direct_z.pvalues['z_support']:.4f})")
    print("```\n")

    print("## 4. 종합 결과 비교 요약표 (표준화 계수 Beta 기준)\n")
def sig(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    elif p < 0.1: return "†"
    else: return "ns"

    print("| 구분 | AI활용동기(motivation) | 조직지원인식(support) |")
    print("|---|---|---|")
    p_m_a, p_s_a = model_a_z.pvalues['z_motivation'], model_a_z.pvalues['z_support']
    print(f"| 매개변수에 미치는 영향 | ß = {model_a_z.params['z_motivation']:.3f} {sig(p_m_a)} | ß = {model_a_z.params['z_support']:.3f} {sig(p_s_a)} |")

    p_m_t, p_s_t = model_total_z.pvalues['z_motivation'], model_total_z.pvalues['z_support']
    print(f"| 종속변수에 미치는 총효과 | ß = {model_total_z.params['z_motivation']:.3f} {sig(p_m_t)} | ß = {model_total_z.params['z_support']:.3f} {sig(p_s_t)} |")

    p_m_d, p_s_d = model_direct_z.pvalues['z_motivation'], model_direct_z.pvalues['z_support']
    print(f"| 종속변수에 미치는 직접효과 | ß = {model_direct_z.params['z_motivation']:.3f} {sig(p_m_d)} | ß = {model_direct_z.params['z_support']:.3f} {sig(p_s_d)} |")

    print("\n> († p<.10, * p<.05, ** p<.01, *** p<.001)\n")

    print("## 5. 상세 회귀 요약\n")
    print("<details><summary>매개변수 모형 - Path A</summary>\n")
    print("```text")
    print(model_a.summary().tables[1])
    print("```\n</details>\n")

    print("<details><summary>종속변수 직접효과 모형 - Path C'</summary>\n")
    print("```text")
    print(model_direct.summary().tables[1])
    print("```\n</details>\n")

    print("## 주요 해석\n")
    print("- 표준화계수 기준으로 두 독립변수의 상대적 영향력을 직접 비교할 수 있다.")
    print("- `effect`에 대한 영향이 더 큰 변수는 업무효과 인식 형성에 더 핵심적인 선행요인으로 해석된다.")
    print("- `expectation`에 대한 총효과와 직접효과를 함께 보면 특정 변수의 효과가 매개변수 `effect`를 통해 얼마나 전달되는지 가늠할 수 있다.")

print(f"완료: {result_path} 생성")
