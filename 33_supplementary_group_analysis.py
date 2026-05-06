from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


RESULT_DIR = Path("result")
INPUT_PRIORITY = [
    Path("chatbot_output_selected_preprocessed.csv"),
    Path("chatbot_output.csv"),
]
AI_USE_COLUMN_CANDIDATES = [
    "Q3",
    "ai_use",
    "ai_user",
    "ai_users",
    "ai_experience",
    "genai_use",
    "genai_user",
    "chatgpt_use",
    "chatbot_use",
]
AI_USE_POSITIVE_VALUES = {
    "1",
    "yes",
    "y",
    "true",
    "t",
    "있다",
    "예",
    "사용 경험 있음",
    "사용함",
    "경험 있음",
    "활용 경험 있음",
    "활용함",
}
AI_USE_NEGATIVE_MARKERS = ["없", "아니", "미사용", "비사용", "no", "false"]

DERIVED_VARIABLES = {
    "strategic_expectation_123": ["Q20_1", "Q20_2", "Q20_3"],
    "job_replacement_perception": ["Q20_4"],
    "org_support_1234": ["Q16_1", "Q16_2", "Q16_3", "Q16_4"],
    "ai_friendly_org_567": ["Q16_5", "Q16_6", "Q16_7"],
    "voluntary_motivation": ["Q9_3", "Q9_4"],
    "perceived_work_effect": ["Q7_1", "Q7_2", "Q7_3", "Q7_4", "Q7_5"],
}

CSV_COLUMNS = [
    "analysis",
    "grouping_variable",
    "comparison_variable",
    "low_group_name",
    "high_group_name",
    "low_n",
    "high_n",
    "low_mean",
    "high_mean",
    "low_sd",
    "high_sd",
    "mean_diff_high_minus_low",
    "t_value",
    "p_value",
    "cohen_d",
    "interpretation",
]


def select_input_file() -> Path:
    for path in INPUT_PRIORITY:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in INPUT_PRIORITY)
    print(f"Input data file not found. Searched: {searched}")
    sys.exit(1)


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = sorted({column for columns in DERIVED_VARIABLES.values() for column in columns})
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print("Missing required columns:")
        for column in missing_columns:
            print(f"- {column}")
        sys.exit(1)


def create_common_variables(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    source_columns = sorted({column for columns in DERIVED_VARIABLES.values() for column in columns})
    for column in source_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for variable, columns in DERIVED_VARIABLES.items():
        if len(columns) == 1:
            data[variable] = data[columns[0]]
        else:
            data[variable] = data[columns].mean(axis=1)
    return data


def find_ai_use_column(df: pd.DataFrame) -> str | None:
    column_lookup = {column.lower(): column for column in df.columns}
    for candidate in AI_USE_COLUMN_CANDIDATES:
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]
    return None


def is_ai_user_value(value: object) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return bool(value == 1)

    normalized = str(value).strip().lower()
    if normalized in AI_USE_POSITIVE_VALUES:
        return True
    if normalized in {"1.0", "1.00"}:
        return True
    if any(marker in normalized for marker in AI_USE_NEGATIVE_MARKERS):
        return False
    if "사용" in normalized and ("있" in normalized or "함" in normalized):
        return True
    if "활용" in normalized and ("있" in normalized or "함" in normalized):
        return True
    return False


def filter_ai_users(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ai_use_column = find_ai_use_column(df)
    if ai_use_column is not None:
        mask = df[ai_use_column].map(is_ai_user_value)
        return df.loc[mask].copy(), f"AI use column `{ai_use_column}`"

    fallback_columns = ["voluntary_motivation", "perceived_work_effect"]
    missing_fallback_columns = [column for column in fallback_columns if column not in df.columns]
    if missing_fallback_columns:
        print("AI use column not found and fallback variables are unavailable:")
        for column in missing_fallback_columns:
            print(f"- {column}")
        sys.exit(1)

    mask = df[fallback_columns].notna().all(axis=1)
    return df.loc[mask].copy(), "fallback: non-missing voluntary_motivation and perceived_work_effect"


def cohen_d_independent(low_values: pd.Series, high_values: pd.Series) -> float:
    low = low_values.dropna().to_numpy(dtype=float)
    high = high_values.dropna().to_numpy(dtype=float)
    low_n = len(low)
    high_n = len(high)
    if low_n < 2 or high_n < 2:
        return np.nan

    low_var = np.var(low, ddof=1)
    high_var = np.var(high, ddof=1)
    denominator_df = low_n + high_n - 2
    if denominator_df <= 0:
        return np.nan

    pooled_sd = np.sqrt(((low_n - 1) * low_var + (high_n - 1) * high_var) / denominator_df)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return float((np.mean(high) - np.mean(low)) / pooled_sd)


def effect_size_label(cohen_d: float) -> str:
    if pd.isna(cohen_d):
        return "effect size unavailable"

    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def make_interpretation(p_value: float, high_mean: float, low_mean: float, cohen_d: float) -> str:
    if pd.isna(p_value):
        significance = "test not available"
    elif p_value < 0.05 and high_mean > low_mean:
        significance = "high group shows significantly higher mean"
    elif p_value < 0.05 and high_mean < low_mean:
        significance = "high group shows significantly lower mean"
    else:
        significance = "no statistically significant difference"
    return f"{significance}; {effect_size_label(cohen_d)} effect size"


def run_group_analysis(
    df: pd.DataFrame,
    analysis_name: str,
    grouping_variable: str,
    low_group_name: str,
    high_group_name: str,
    comparison_variables: list[str],
) -> tuple[pd.DataFrame, float]:
    median_value = df[grouping_variable].median(skipna=True)
    if pd.isna(median_value):
        print(f"Median could not be calculated for {grouping_variable}.")
        sys.exit(1)

    rows = []
    for comparison_variable in comparison_variables:
        pairwise = df[[grouping_variable, comparison_variable]].dropna().copy()
        low_values = pairwise.loc[pairwise[grouping_variable] <= median_value, comparison_variable]
        high_values = pairwise.loc[pairwise[grouping_variable] > median_value, comparison_variable]

        low_n = int(low_values.count())
        high_n = int(high_values.count())
        low_mean = float(low_values.mean()) if low_n else np.nan
        high_mean = float(high_values.mean()) if high_n else np.nan
        low_sd = float(low_values.std(ddof=1)) if low_n > 1 else np.nan
        high_sd = float(high_values.std(ddof=1)) if high_n > 1 else np.nan

        if low_n > 1 and high_n > 1:
            t_stat, p_value = stats.ttest_ind(high_values, low_values, equal_var=False, nan_policy="omit")
            t_value = float(t_stat)
            p_value = float(p_value)
        else:
            t_value = np.nan
            p_value = np.nan

        mean_diff = high_mean - low_mean if not (pd.isna(high_mean) or pd.isna(low_mean)) else np.nan
        cohen_d = cohen_d_independent(low_values, high_values)

        rows.append(
            {
                "analysis": analysis_name,
                "grouping_variable": grouping_variable,
                "comparison_variable": comparison_variable,
                "low_group_name": low_group_name,
                "high_group_name": high_group_name,
                "low_n": low_n,
                "high_n": high_n,
                "low_mean": low_mean,
                "high_mean": high_mean,
                "low_sd": low_sd,
                "high_sd": high_sd,
                "mean_diff_high_minus_low": mean_diff,
                "t_value": t_value,
                "p_value": p_value,
                "cohen_d": cohen_d,
                "interpretation": make_interpretation(p_value, high_mean, low_mean, cohen_d),
            }
        )

    return pd.DataFrame(rows, columns=CSV_COLUMNS), float(median_value)


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def format_number(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def print_console_summary(result: pd.DataFrame, title: str) -> None:
    display = result[
        [
            "comparison_variable",
            "low_n",
            "high_n",
            "low_mean",
            "high_mean",
            "mean_diff_high_minus_low",
            "t_value",
            "p_value",
            "cohen_d",
        ]
    ].copy()
    for column in ["low_mean", "high_mean", "mean_diff_high_minus_low", "t_value", "cohen_d"]:
        display[column] = display[column].map(format_number)
    display["p_value"] = display["p_value"].map(format_p_value)

    print(f"\n{title}")
    print(display.to_string(index=False))


def markdown_table(result: pd.DataFrame) -> str:
    columns = [
        "comparison_variable",
        "low_n",
        "high_n",
        "low_mean",
        "high_mean",
        "mean_diff_high_minus_low",
        "t_value",
        "p_value",
        "cohen_d",
        "interpretation",
    ]
    lines = [
        "| Variable | Low n | High n | Low mean | High mean | Mean diff | t | p | Cohen's d | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in result[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["comparison_variable"]),
                    str(int(row["low_n"])),
                    str(int(row["high_n"])),
                    format_number(row["low_mean"]),
                    format_number(row["high_mean"]),
                    format_number(row["mean_diff_high_minus_low"]),
                    format_number(row["t_value"]),
                    format_p_value(row["p_value"]),
                    format_number(row["cohen_d"]),
                    str(row["interpretation"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_summary_markdown(
    input_file: Path,
    original_n: int,
    ai_user_n: int,
    filter_basis: str,
    median_a: float,
    median_b: float,
    result_a: pd.DataFrame,
    result_b: pd.DataFrame,
) -> Path:
    RESULT_DIR.mkdir(exist_ok=True)
    summary_path = RESULT_DIR / "33_supplementary_group_analysis_summary_ai_users.md"
    content = [
        "# 33 Supplementary Group Analysis: AI Users Only",
        "",
        f"- Input data file: `{input_file.name}`",
        f"- Original sample size: {original_n}",
        f"- AI user filter basis: {filter_basis}",
        f"- AI user sample size used in analyses: {ai_user_n}",
        f"- Analysis A median (`job_replacement_perception`): {median_a:.3f}",
        f"- Analysis B median (`ai_friendly_org_567`): {median_b:.3f}",
        "- Missing data handling: pairwise deletion by grouping variable and comparison variable.",
        "",
        "## Analysis A: Q20_4-Based Group Comparison",
        "",
        markdown_table(result_a),
        "",
        "## Analysis B: Q16_5-Q16_7-Based Group Comparison",
        "",
        markdown_table(result_b),
        "",
    ]
    summary_path.write_text("\n".join(content), encoding="utf-8")
    return summary_path


def main() -> None:
    input_file = select_input_file()
    df = pd.read_csv(input_file)
    original_n = len(df)
    validate_required_columns(df)
    df = create_common_variables(df)
    df, filter_basis = filter_ai_users(df)

    result_a, median_a = run_group_analysis(
        df=df,
        analysis_name="A_Q20_4_based_group_comparison",
        grouping_variable="job_replacement_perception",
        low_group_name="low_job_replacement",
        high_group_name="high_job_replacement",
        comparison_variables=[
            "strategic_expectation_123",
            "voluntary_motivation",
            "perceived_work_effect",
            "org_support_1234",
            "ai_friendly_org_567",
        ],
    )

    result_b, median_b = run_group_analysis(
        df=df,
        analysis_name="B_Q16_567_based_group_comparison",
        grouping_variable="ai_friendly_org_567",
        low_group_name="low_ai_friendly_org",
        high_group_name="high_ai_friendly_org",
        comparison_variables=[
            "strategic_expectation_123",
            "job_replacement_perception",
            "voluntary_motivation",
            "perceived_work_effect",
            "org_support_1234",
        ],
    )

    RESULT_DIR.mkdir(exist_ok=True)
    path_a = RESULT_DIR / "33_supplementary_group_analysis_A_Q20_4_ai_users.csv"
    path_b = RESULT_DIR / "33_supplementary_group_analysis_B_Q16_567_ai_users.csv"
    result_a.to_csv(path_a, index=False, encoding="utf-8-sig")
    result_b.to_csv(path_b, index=False, encoding="utf-8-sig")
    summary_path = write_summary_markdown(
        input_file=input_file,
        original_n=original_n,
        ai_user_n=len(df),
        filter_basis=filter_basis,
        median_a=median_a,
        median_b=median_b,
        result_a=result_a,
        result_b=result_b,
    )

    print(f"Used data file: {input_file.name}")
    print(f"Original sample size: {original_n}")
    print(f"AI user filter basis: {filter_basis}")
    print(f"AI user sample size after filtering: {len(df)}")
    print(f"Analysis A median (job_replacement_perception): {median_a:.3f}")
    print(f"Analysis B median (ai_friendly_org_567): {median_b:.3f}")
    print_console_summary(result_a, "Analysis A: Q20_4-based group comparison")
    print_console_summary(result_b, "Analysis B: Q16_5-Q16_7-based group comparison")
    print("\nSaved result files:")
    print(f"- {path_a}")
    print(f"- {path_b}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
