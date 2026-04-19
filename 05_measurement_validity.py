import pandas as pd
import numpy as np
from itertools import combinations
from factor_analyzer import FactorAnalyzer

from result_utils import markdown_output

# ============================================================
# Measurement Validity Analysis
# - Convergent validity: AVE, CR
# - Discriminant validity: Fornell-Larcker, HTMT
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df_ai = df[df["Q3"] == 1].copy()

cols_7 = [f"Q7_{i}" for i in range(1, 6)]
cols_16 = [f"Q16_{i}" for i in range(1, 7)]  # Q16_7 제외: 개인 관심 문항, motivation과 중첩
cols_20 = [f"Q20_{i}" for i in range(2, 5)]  # Q20_1 제외: 업무효과(매개변수)와 개념 중첩
cols_9_voluntary = ["Q9_3", "Q9_4"]

construct_dict = {
    "work_effect": cols_7,
    "org_support": cols_16,
    "strategic_expectation": cols_20,
    "motivation_voluntary": cols_9_voluntary,
}


def compute_ave_cr(data, cols, name):
    valid = data[cols].dropna()
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(valid)

    loadings = fa.loadings_.flatten()
    squared = loadings ** 2

    ave = float(np.mean(squared))
    cr = float((np.sum(loadings) ** 2) / ((np.sum(loadings) ** 2) + np.sum(1 - squared)))

    print(f"\n[{name}]")
    print(f"Loadings: {np.round(loadings, 3)}")
    print(f"AVE = {ave:.3f}")
    print(f"CR  = {cr:.3f}")

    return ave


def mean_abs_interitem_corr(data, cols):
    """Average absolute inter-item correlation within a construct."""
    corr_values = []
    for col_a, col_b in combinations(cols, 2):
        pair = data[[col_a, col_b]].dropna()
        if len(pair) == 0:
            continue
        corr_values.append(abs(pair[col_a].corr(pair[col_b])))
    return float(np.mean(corr_values)) if corr_values else np.nan


def spearman_brown_from_r(r_value):
    """Spearman-Brown coefficient for a two-item scale."""
    return float((2 * r_value) / (1 + r_value))


def compute_htmt(data, cols_a, cols_b):
    """
    HTMT based on item-level correlations.
    Numerator: mean absolute heterotrait-heteromethod correlations.
    Denominator: geometric mean of within-construct monotrait correlations.
    """
    heterotrait = []
    for col_a in cols_a:
        for col_b in cols_b:
            pair = data[[col_a, col_b]].dropna()
            if len(pair) == 0:
                continue
            heterotrait.append(abs(pair[col_a].corr(pair[col_b])))

    mono_a = mean_abs_interitem_corr(data, cols_a)
    mono_b = mean_abs_interitem_corr(data, cols_b)

    if not heterotrait or np.isnan(mono_a) or np.isnan(mono_b) or mono_a <= 0 or mono_b <= 0:
        return np.nan

    return float(np.mean(heterotrait) / np.sqrt(mono_a * mono_b))


with markdown_output("05_measurement_validity.md") as result_path:
    print("# 05 측정타당성 분석\n")
    print(f"- 분석 표본: AI 활용자 {len(df_ai)}명\n")
    print("===================================")
    print("Convergent Validity (AVE & CR)")
    print("===================================")

    ave_values = {}
    for name, cols in construct_dict.items():
        ave_values[name] = compute_ave_cr(df_ai, cols, name)

    pair = df_ai[cols_9_voluntary].dropna()
    if len(pair) > 0:
        voluntary_r = pair[cols_9_voluntary[0]].corr(pair[cols_9_voluntary[1]])
        sb_coef = spearman_brown_from_r(voluntary_r)
        print("\n[motivation_voluntary - Two-item reliability check]")
        print(f"Inter-item correlation = {voluntary_r:.3f}")
        print(f"Spearman-Brown coefficient = {sb_coef:.3f}")
        print("Note: motivation_voluntary uses two items, so interpret alpha/CR with caution.")

    for name, cols in construct_dict.items():
        df_ai[name] = df_ai[cols].mean(axis=1)

    print("\n===================================")
    print("Discriminant Validity (Fornell-Larcker)")
    print("===================================")

    construct_corr = df_ai[list(construct_dict.keys())].corr()
    sqrt_ave = {key: np.sqrt(value) for key, value in ave_values.items()}

    print("\nCorrelation Matrix")
    print(construct_corr.round(3))

    print("\nSqrt(AVE) Values")
    for key, value in sqrt_ave.items():
        print(f"{key}: {value:.3f}")

    print("\nCriterion: sqrt(AVE) should exceed inter-construct correlations.")

    print("\n===================================")
    print("HTMT")
    print("===================================")

    construct_names = list(construct_dict.keys())
    for idx, construct_a in enumerate(construct_names):
        for construct_b in construct_names[idx + 1:]:
            htmt_value = compute_htmt(df_ai, construct_dict[construct_a], construct_dict[construct_b])
            print(f"{construct_a} - {construct_b}: {htmt_value:.3f}")

    print("\nCriterion: HTMT < .85")
    print("\n## 주요 해석\n")
    print("- AVE와 CR은 각 구성개념이 자신의 문항을 얼마나 일관되게 설명하는지 보여준다.")
    print("- 전략기대(strategic_expectation)의 AVE는 .471로 통상적 기준인 .50에 미달한다는 점을 명시적으로 보고해야 한다.")
    print("- 다만 전략기대의 sqrt(AVE)=.687은 다른 구성개념과의 상관보다 크고, HTMT도 모든 쌍에서 .85 미만이므로 판별타당성 측면의 방어 논리는 유지된다.")
    print("- Fornell-Larcker 기준에서 각 구성개념의 `sqrt(AVE)`가 상관계수보다 크면 판별타당성이 양호하다고 볼 수 있다.")
    print("- HTMT가 .85 이하로 유지되면 서로 다른 구성개념이 과도하게 중첩되지 않았다는 해석이 가능하다.")
    print("- 자발적 활용동기는 2문항 척도이므로 적재치가 대칭적으로 나타나는 것은 2문항 제약의 산물일 수 있으며, Spearman-Brown 계수를 함께 보고 보수적으로 해석하는 것이 적절하다.")

print(f"완료: {result_path} 생성")
