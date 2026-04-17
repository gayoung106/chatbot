import sys
import pandas as pd
import numpy as np
from itertools import combinations
from factor_analyzer import FactorAnalyzer

sys.stdout = open("measurement_validity_q16_excluded.md", "w", encoding="utf-8")

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df_ai = df[df["Q3"] == 1].copy()

cols_7 = [f"Q7_{i}" for i in range(1, 6)]
cols_16_full = [f"Q16_{i}" for i in range(1, 8)]
cols_16_ex = [f"Q16_{i}" for i in range(1, 7)]
cols_20 = [f"Q20_{i}" for i in range(2, 5)]
cols_9_voluntary = ["Q9_3", "Q9_4"]


def compute_ave_cr(data, cols):
    valid = data[cols].dropna()
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(valid)

    loadings = fa.loadings_.flatten()
    squared = loadings ** 2

    ave = float(np.mean(squared))
    cr = float((np.sum(loadings) ** 2) / ((np.sum(loadings) ** 2) + np.sum(1 - squared)))
    return loadings, ave, cr


def mean_abs_interitem_corr(data, cols):
    corr_values = []
    for col_a, col_b in combinations(cols, 2):
        pair = data[[col_a, col_b]].dropna()
        if len(pair) == 0:
            continue
        corr_values.append(abs(pair[col_a].corr(pair[col_b])))
    return float(np.mean(corr_values)) if corr_values else np.nan


def compute_htmt(data, cols_a, cols_b):
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


def run_validity_block(support_cols, label):
    construct_dict = {
        "work_effect": cols_7,
        "org_support": support_cols,
        "strategic_expectation": cols_20,
        "motivation_voluntary": cols_9_voluntary,
    }

    print(f"## {label}\n")
    ave_values = {}

    print("### AVE / CR")
    print("| 구성개념 | 문항 | AVE | CR |")
    print("| --- | --- | ---: | ---: |")
    for name, cols in construct_dict.items():
        loadings, ave, cr = compute_ave_cr(df_ai, cols)
        ave_values[name] = ave
        print(f"| {name} | {', '.join(cols)} | {ave:.3f} | {cr:.3f} |")
    print()

    block_df = df_ai.copy()
    for name, cols in construct_dict.items():
        block_df[name] = block_df[cols].mean(axis=1)

    corr = block_df[list(construct_dict.keys())].corr()
    sqrt_ave = {key: np.sqrt(value) for key, value in ave_values.items()}

    print("### Fornell-Larcker")
    print("```text")
    print(corr.round(3))
    print("```")
    print()
    print("| 구성개념 | sqrt(AVE) |")
    print("| --- | ---: |")
    for key, value in sqrt_ave.items():
        print(f"| {key} | {value:.3f} |")
    print()

    print("### HTMT")
    print("| 구성개념 쌍 | HTMT |")
    print("| --- | ---: |")
    names = list(construct_dict.keys())
    for idx, a_name in enumerate(names):
        for b_name in names[idx + 1:]:
            htmt_value = compute_htmt(block_df, construct_dict[a_name], construct_dict[b_name])
            print(f"| {a_name} - {b_name} | {htmt_value:.3f} |")
    print("\n---\n")


print("# Q16_7 제외 전후 측정타당성 비교\n")
print("본 분석은 조직지원 인식 척도에서 Q16_7 문항 포함 여부가 측정타당성 결과에 미치는 영향을 점검하기 위한 강건성 검토이다.\n")

run_validity_block(cols_16_full, "원래 척도: Q16_1 ~ Q16_7")
run_validity_block(cols_16_ex, "대안 척도: Q16_1 ~ Q16_6 (Q16_7 제외)")

print("## 해석 메모\n")
print("- Q16_7은 기존 EFA에서 다른 문항들보다 상대적으로 낮은 적재값을 보였으므로, 제외 버전의 측정타당성 결과를 추가 비교하였다.")
print("- 본 비교 결과는 조직지원 척도의 특정 문항 선택이 본 연구의 실증결과를 과도하게 좌우하는지 여부를 점검하기 위한 보조 근거로 사용한다.")

sys.stdout.close()
print("완료: measurement_validity_q16_excluded.md 생성", file=sys.__stdout__)
