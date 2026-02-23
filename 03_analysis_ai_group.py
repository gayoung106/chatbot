import pandas as pd
import numpy as np
from pingouin import cronbach_alpha
from factor_analyzer import FactorAnalyzer
from scipy.stats import skew, kurtosis

# ============================================================
# 1. 데이터 로드
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")

df_ai = df[df["Q3"] == 1].copy()
df_non = df[df["Q3"] == 0].copy()

print(f"전체 응답자 수: {len(df)}")
print(f"AI 활용자: {len(df_ai)}")
print(f"비활용자: {len(df_non)}")

# ============================================================
# 2. 문항 정의
# ============================================================

cols_7 = [f"Q7_{i}" for i in range(1, 6)]
cols_9_passive = ["Q9_1", "Q9_2"]
cols_9_voluntary = ["Q9_3", "Q9_4"]
cols_16 = [f"Q16_{i}" for i in range(1, 8)]
cols_20 = [f"Q20_{i}" for i in range(1, 5)]

all_item_cols = (
    cols_9_passive +
    cols_9_voluntary +
    cols_7 +
    cols_20 +
    cols_16
)

# ============================================================
# 3. 신뢰도 분석
# ============================================================

def reliability(df_sub, cols, label):
    data = df_sub[cols].dropna()
    alpha = cronbach_alpha(data)[0]
    print(f"{label} (N={len(data)}): {alpha:.3f}")
    return alpha

print("\n==============================")
print("신뢰도 분석 (AI 활용자)")
print("==============================")

reliability(df_ai, cols_7, "업무효과(Q7)")
reliability(df_ai, cols_16, "조직지원(Q16)")
reliability(df_ai, cols_9_passive, "수동적 활용동기")
reliability(df_ai, cols_9_voluntary, "자발적 활용동기")
reliability(df_ai, cols_20, "전략기대(Q20)")

# ============================================================
# 4. 평균 변수 생성
# ============================================================

df_ai["work_effect"] = df_ai[cols_7].mean(axis=1)
df_ai["org_support"] = df_ai[cols_16].mean(axis=1)
df_ai["motivation_passive"] = df_ai[cols_9_passive].mean(axis=1)
df_ai["motivation_voluntary"] = df_ai[cols_9_voluntary].mean(axis=1)
df_ai["strategic_expectation"] = df_ai[cols_20].mean(axis=1)

# ============================================================
# 5. EFA
# ============================================================

def run_efa(df_sub, cols, n_factors, title):
    data = df_sub[cols].dropna()
    print("\n====================================")
    print(f"EFA - {title}")
    print(f"유효표본 수 (N) = {len(data)}")
    print("====================================")

    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
    fa.fit(data)

    loadings = pd.DataFrame(
        fa.loadings_,
        index=cols,
        columns=[f"Factor{i+1}" for i in range(n_factors)]
    )

    print(loadings.round(3))
    return loadings

run_efa(df_ai, cols_9_passive + cols_9_voluntary, 2, "활용동기")
run_efa(df_ai, cols_7, 1, "업무효과")
run_efa(df_ai, cols_16, 1, "조직지원")
run_efa(df_ai, cols_20, 1, "전략기대")

# ============================================================
# 6. 문항별 기술통계 (왜도·첨도 포함)
# ============================================================

print("\n====================================")
print("문항별 기술통계 (AI 활용자)")
print("====================================")

item_desc = pd.DataFrame()

for col in all_item_cols:
    item_desc.loc[col, "mean"] = df_ai[col].mean()
    item_desc.loc[col, "std"] = df_ai[col].std()
    item_desc.loc[col, "min"] = df_ai[col].min()
    item_desc.loc[col, "max"] = df_ai[col].max()
    item_desc.loc[col, "skew"] = skew(df_ai[col])
    item_desc.loc[col, "kurtosis"] = kurtosis(df_ai[col])

print(item_desc.round(3))

# ============================================================
# 7. 구성개념 평균 기술통계
# ============================================================

print("\n====================================")
print("구성개념 평균 기술통계 (AI 활용자)")
print("====================================")

construct_desc = (
    df_ai[[
        "work_effect",
        "org_support",
        "strategic_expectation",
        "motivation_passive",
        "motivation_voluntary"
    ]]
    .describe()
    .T[["mean", "std", "min", "max"]]
)

print(construct_desc.round(3))

# ============================================================
# 8. 통제변수 기술통계
# ============================================================

print("\n====================================")
print("통제변수 기술통계 (AI 활용자)")
print("====================================")

print(
    df_ai[["gender", "rank_code", "career_code"]]
    .describe()
    .T[["mean", "std", "min", "max"]]
    .round(3)
)