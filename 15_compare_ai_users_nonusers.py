import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
import sys

sys.stdout = open("sample_group_comparison.md", "w", encoding="utf-8")

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")

ai_users = df[df["Q3"] == 1].copy()
non_users = df[df["Q3"] == 0].copy()

print("# AI 활용자와 비활용자 집단 비교\n")
print(f"- 전체 응답자 수: {len(df)}")
print(f"- AI 활용자: {len(ai_users)}")
print(f"- AI 비활용자: {len(non_users)}\n")


def format_p_value(p_value):
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def run_mann_whitney(data_a, data_b, column):
    series_a = data_a[column].dropna()
    series_b = data_b[column].dropna()
    stat, p_value = mannwhitneyu(series_a, series_b, alternative="two-sided")
    return {
        "ai_mean": series_a.mean(),
        "ai_std": series_a.std(),
        "non_mean": series_b.mean(),
        "non_std": series_b.std(),
        "stat": stat,
        "p": p_value,
    }


def run_chi_square(data, column):
    table = pd.crosstab(data["Q3"], data[column])
    chi2, p_value, _, _ = chi2_contingency(table)
    return table, chi2, p_value


print("## 1. 연속 또는 서열형 변수 비교\n")
print("| 변수 | AI 활용자 평균 | AI 활용자 SD | 비활용자 평균 | 비활용자 SD | Mann-Whitney U | p-value |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

for variable in ["gender", "rank_code", "career_code", "ai_task_count"]:
    result = run_mann_whitney(ai_users, non_users, variable)
    print(
        f"| {variable} | {result['ai_mean']:.3f} | {result['ai_std']:.3f} | "
        f"{result['non_mean']:.3f} | {result['non_std']:.3f} | "
        f"{result['stat']:.1f} | {format_p_value(result['p'])} |"
    )

print("\n## 2. 범주형 변수 분포 비교\n")
for variable in ["SQ1", "SQ4"]:
    table, chi2, p_value = run_chi_square(df, variable)
    print(f"### {variable}\n")
    print("```text")
    print(table)
    print("```")
    print(f"- Chi-square = {chi2:.3f}")
    print(f"- p-value = {format_p_value(p_value)}\n")

print("## 3. 해석 메모\n")
print("- 이 표는 AI 활용자만을 분석대상으로 제한할 때 발생할 수 있는 표본 선택편의 우려를 점검하기 위한 기초 비교표이다.")
print("- 집단 간 차이가 확인되더라도 본 연구는 전체 공무원 집단의 일반적 차이를 설명하기보다, AI 활용 경험이 있는 집단 내부의 인식 메커니즘을 분석하는 데 목적이 있다.")
print("- 따라서 본 연구 결과는 AI 비활용자를 포함한 전체 공무원 집단으로 직접 일반화하지 않도록 해석해야 한다.")

sys.stdout.close()
print("완료: sample_group_comparison.md 생성", file=sys.__stdout__)
