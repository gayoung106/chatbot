import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2

from result_utils import markdown_output

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")


def llr_pvalue(full_model, null_model):
    lr_stat = -2 * (null_model.llf - full_model.llf)
    df_diff = int(full_model.df_model - null_model.df_model)
    p_value = chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def fit_binomial_glm(formula, data):
    return smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()


analysis_cols = ["Q3", "gender", "rank_code", "career_code", "SQ1", "SQ4"]
data = df[analysis_cols].dropna().copy()

models = {
    "Model 1": "Q3 ~ gender + rank_code + career_code",
    "Model 2": "Q3 ~ gender + rank_code + career_code + C(SQ1)",
    "Model 3": "Q3 ~ gender + rank_code + career_code + C(SQ1) + C(SQ4)",
}

with markdown_output("16_ai_use_selection_model.md") as result_path:
    print("# 16 AI 활용 여부 선택모형 분석\n")
    print("본 분석은 AI 활용자 표본만을 대상으로 한 본 연구의 표본 선택 과정을 보조적으로 점검하기 위해 수행하였다.\n")
    print("주의: `ai_task_count`는 AI 활용 여부 이후에 형성되는 성격이 강한 변수이므로 선택모형에서는 제외하였다.\n")
    print(f"- 분석 표본 수: {len(data)}")
    print(f"- AI 활용자 비율: {data['Q3'].mean():.3f}\n")

    for model_name, formula in models.items():
        print(f"## {model_name}\n")
        print(f"- Formula: `{formula}`\n")

        model = fit_binomial_glm(formula, data)
        null_model = fit_binomial_glm("Q3 ~ 1", data)
        lr_stat, df_diff, lr_p = llr_pvalue(model, null_model)
        pseudo_r2 = 1 - (model.llf / null_model.llf)

        odds_ratios = pd.DataFrame({
            "coef": model.params,
            "odds_ratio": np.exp(model.params),
            "std_err": model.bse,
            "z": model.tvalues,
            "p_value": model.pvalues,
        })

        print("```text")
        print(model.summary2().tables[1])
        print("```\n")

        print("### Odds Ratios")
        print("| 변수 | 계수 | 오즈비 | 표준오차 | z | p-value |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for index, row in odds_ratios.iterrows():
            print(
                f"| {index} | {row['coef']:.3f} | {row['odds_ratio']:.3f} | "
                f"{row['std_err']:.3f} | {row['z']:.3f} | {row['p_value']:.3f} |"
            )
        print()
        print(f"- McFadden pseudo R² = {pseudo_r2:.3f}")
        print(f"- LLR chi-square = {lr_stat:.3f} (df = {df_diff}, p = {lr_p:.3f})\n")

    print("## 주요 해석\n")
    print("- 본 모형은 AI 활용 여부 자체가 무작위가 아니라 일부 개인적 및 조직적 특성과 관련될 수 있음을 보여주기 위한 보조 분석이다.")
    print("- 따라서 본 연구의 본 분석 결과는 AI 활용자 하위집단의 인식 메커니즘으로 해석해야 하며, 전체 공무원 집단으로 직접 일반화하는 데에는 주의가 필요하다.")
    print("- 특정 변수의 오즈비가 유의하게 1보다 크거나 작다면 그 변수는 AI 활용 여부와 체계적으로 관련된 선택요인으로 해석할 수 있다.")

print(f"완료: {result_path} 생성")
