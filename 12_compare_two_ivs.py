import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import zscore
import sys

# output redirection to utf8 file
sys.stdout = open('analysis_results.md', 'w', encoding='utf-8')

print("# 조직지원인식 vs AI활용동기 영향력 비교 분석 결과\n")
print("데이터를 로드하고 통계 분석을 수행합니다...\n")
df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

# 1. 변수 생성
df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1) # AI활용동기
df["effect"] = df[[f"Q7_{i}" for i in range(1,6)]].mean(axis=1) # 매개변수
df["support"] = df[[f"Q16_{i}" for i in range(1,7)]].mean(axis=1) # 조직지원인식 (Q16_7 제외: 개인 관심 문항, motivation과 중첩)
df["expectation"] = df[[f"Q20_{i}" for i in range(2,5)]].mean(axis=1) # 종속변수 (Q20_1 제외: 업무효과와 개념 중첩)

# 영향력 크기(magnitude) 비교를 위한 표준화 변수 생성
df['z_motivation'] = zscore(df['motivation'], nan_policy='omit')
df['z_support'] = zscore(df['support'], nan_policy='omit')
df['z_effect'] = zscore(df['effect'], nan_policy='omit')
df['z_expectation'] = zscore(df['expectation'], nan_policy='omit')

df = df.dropna(subset=['motivation', 'support', 'effect', 'expectation', 'gender', 'rank_code', 'career_code'])

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
