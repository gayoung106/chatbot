import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from result_utils import markdown_output


df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
ai_users = df[df["Q3"] == 1].copy()
non_users = df[df["Q3"] == 0].copy()


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def run_mann_whitney(data_a: pd.DataFrame, data_b: pd.DataFrame, column: str):
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


def run_chi_square(data: pd.DataFrame, column: str):
    table = pd.crosstab(data["Q3"], data[column])
    chi2, p_value, _, _ = chi2_contingency(table)
    return table, chi2, p_value


with markdown_output("15_compare_ai_users_nonusers.md") as result_path:
    print("# 15 AI User vs Non-user Comparison\n")
    print(f"- Total respondents: {len(df)}")
    print(f"- AI users: {len(ai_users)}")
    print(f"- AI non-users: {len(non_users)}\n")

    print("## Appendix Table A1. Descriptive comparison of AI users and non-users\n")
    print("| Variable | AI users Mean/Prop. | AI users SD/N | Non-users Mean/Prop. | Non-users SD/N | Test statistic | p-value |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variable in ["gender", "rank_code", "career_code", "ai_task_count"]:
        result = run_mann_whitney(ai_users, non_users, variable)
        print(
            f"| {variable} | {result['ai_mean']:.3f} | {result['ai_std']:.3f} | "
            f"{result['non_mean']:.3f} | {result['non_std']:.3f} | "
            f"{result['stat']:.1f} | {format_p_value(result['p'])} |"
        )
    print()

    print("## Distribution checks for categorical variables\n")
    for variable in ["SQ1", "SQ4"]:
        table, chi2, p_value = run_chi_square(df, variable)
        print(f"### {variable}\n")
        print("```text")
        print(table)
        print("```")
        print(f"- Chi-square = {chi2:.3f}")
        print(f"- p-value = {format_p_value(p_value)}\n")

    print("## Interpretation\n")
    print("- This table is provided as a descriptive appendix check for possible selection differences between AI users and non-users.")
    print("- The main analysis remains restricted to AI users because the focal mechanism concerns expectations among respondents with direct AI use experience.")

print(f"완료: {result_path} 생성")
