# ============================================================
# Measurement Validity Analysis
# AVE, CR, Fornell-Larcker, HTMT
# ============================================================

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer

# ------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df_ai = df[df["Q3"] == 1].copy()

# ------------------------------------------------------------
# 2. 문항 정의
# ------------------------------------------------------------

cols_7 = [f"Q7_{i}" for i in range(1, 6)]
cols_16 = [f"Q16_{i}" for i in range(1, 8)]
cols_20 = [f"Q20_{i}" for i in range(1, 5)]
cols_9_voluntary = ["Q9_3", "Q9_4"]

construct_dict = {
    "work_effect": cols_7,
    "org_support": cols_16,
    "strategic_expectation": cols_20,
    "motivation_voluntary": cols_9_voluntary
}

# ------------------------------------------------------------
# 3. AVE, CR 계산 함수
# ------------------------------------------------------------

def compute_ave_cr(data, cols, name):
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(data[cols].dropna())

    loadings = fa.loadings_.flatten()
    squared = loadings ** 2

    ave = np.mean(squared)

    cr = (np.sum(loadings) ** 2) / (
        (np.sum(loadings) ** 2) + np.sum(1 - squared)
    )

    print(f"\n[{name}]")
    print(f"Loadings: {np.round(loadings,3)}")
    print(f"AVE = {ave:.3f}")
    print(f"CR  = {cr:.3f}")

    return ave

# ------------------------------------------------------------
# 4. 수렴타당도
# ------------------------------------------------------------

print("===================================")
print("Convergent Validity (AVE & CR)")
print("===================================")

ave_values = {}

for name, cols in construct_dict.items():
    ave_values[name] = compute_ave_cr(df_ai, cols, name)

# ------------------------------------------------------------
# 5. 평균 변수 생성
# ------------------------------------------------------------

for name, cols in construct_dict.items():
    df_ai[name] = df_ai[cols].mean(axis=1)

# ------------------------------------------------------------
# 6. Fornell-Larcker 판별타당도
# ------------------------------------------------------------

print("\n===================================")
print("Discriminant Validity (Fornell-Larcker)")
print("===================================")

construct_corr = df_ai[list(construct_dict.keys())].corr()
sqrt_ave = {k: np.sqrt(v) for k, v in ave_values.items()}

print("\nCorrelation Matrix")
print(construct_corr.round(3))

print("\n√AVE Values")
for k, v in sqrt_ave.items():
    print(f"{k}: {v:.3f}")

print("\nCriterion: √AVE > Inter-construct correlations")

# ------------------------------------------------------------
# 7. HTMT (간단 근사)
# ------------------------------------------------------------

print("\n===================================")
print("HTMT (Approximation)")
print("===================================")

for i in construct_dict.keys():
    for j in construct_dict.keys():
        if i < j:
            val = abs(df_ai[i].corr(df_ai[j]))
            print(f"{i} - {j}: {val:.3f}")

print("\nCriterion: HTMT < .85")