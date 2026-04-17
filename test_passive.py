import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import zscore

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

df["motivation_passive"] = df[["Q9_1", "Q9_2"]].mean(axis=1) # 수동적 AI활용동기
df["effect"] = df[[f"Q7_{i}" for i in range(1,6)]].mean(axis=1) # 매개변수
df["support"] = df[[f"Q16_{i}" for i in range(1,7)]].mean(axis=1) # 조직지원인식 (Q16_7 제외: 개인 관심 문항, motivation과 중첩)
df["expectation"] = df[[f"Q20_{i}" for i in range(2,5)]].mean(axis=1) # 종속변수 (Q20_1 제외: 업무효과와 개념 중첩)

# 영향력 크기(magnitude) 비교를 위한 표준화 변수 생성
df['z_motivation_passive'] = zscore(df['motivation_passive'], nan_policy='omit')
df['z_support'] = zscore(df['support'], nan_policy='omit')
df['z_effect'] = zscore(df['effect'], nan_policy='omit')
df['z_expectation'] = zscore(df['expectation'], nan_policy='omit')

df = df.dropna(subset=['motivation_passive', 'support', 'effect', 'expectation', 'gender', 'rank_code', 'career_code'])

with open("test_passive_results.txt", "w", encoding="utf-8") as f:
    model_a = smf.ols("effect ~ motivation_passive + support + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    f.write("## Path A (effect <- motivation_passive)\n")
    f.write(model_a.summary().as_text() + "\n\n")

    model_total = smf.ols("expectation ~ motivation_passive + support + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    f.write("## Path C (expectation <- motivation_passive (Total Effect))\n")
    f.write(model_total.summary().as_text() + "\n\n")

    model_direct = smf.ols("expectation ~ motivation_passive + support + effect + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    f.write("## Path C' (expectation <- motivation_passive (Direct Effect))\n")
    f.write(model_direct.summary().as_text() + "\n\n")
