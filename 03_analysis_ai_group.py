import pandas as pd
import numpy as np
from pingouin import cronbach_alpha
from factor_analyzer import FactorAnalyzer

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

# ============================================================
# 3. Q20 응답 구조 확인 (전체 기준으로 확인)
# ============================================================

print("\n🔎 Q20 각 문항 응답 수 (전체 기준)")
print(df[cols_20].notna().sum())

print("\n🔎 Q20 4문항 모두 응답자 수 (전체 기준)")
print(df[cols_20].dropna().shape[0])

# ============================================================
# 4. 신뢰도 분석 (Listwise)
# ============================================================

def reliability(df_sub, cols, label):
    data = df_sub[cols].dropna()
    alpha = cronbach_alpha(data)[0]
    print(f"{label} (N={len(data)}): {alpha:.3f}")
    return alpha

print("\n==============================")
print("신뢰도 분석")
print("==============================")

# AI 활용자 기준
reliability(df_ai, cols_7, "업무효과(Q7) - AI 활용자")
reliability(df_ai, cols_16, "조직지원(Q16) - AI 활용자")
reliability(df_ai, cols_9_passive, "수동적 활용동기 - AI 활용자")
reliability(df_ai, cols_9_voluntary, "자발적 활용동기 - AI 활용자")

# 전략기대는 전체 기준
reliability(df_ai, cols_20, "전략기대(Q20) - 전체 응답자")

# ============================================================
# 5. 평균 변수 생성 (skipna=True)
# ============================================================

# AI 활용자 변수
df_ai["work_effect"] = df_ai[cols_7].mean(axis=1, skipna=True)
df_ai["org_support"] = df_ai[cols_16].mean(axis=1, skipna=True)
df_ai["motivation_passive"] = df_ai[cols_9_passive].mean(axis=1, skipna=True)
df_ai["motivation_voluntary"] = df_ai[cols_9_voluntary].mean(axis=1, skipna=True)

# 전략기대는 전체 기준
df["strategic_expectation"] = df[cols_20].mean(axis=1, skipna=True)

# ============================================================
# 6. EFA (Raw Data 기반, Listwise)
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


# 활용동기 (AI 활용자)
run_efa(df_ai, cols_9_passive + cols_9_voluntary, 2, "활용동기 (AI 활용자)")

# 업무효과 (AI 활용자)
run_efa(df_ai, cols_7, 1, "업무효과 (AI 활용자)")

# 조직지원 (AI 활용자)
run_efa(df_ai, cols_16, 1, "조직지원 (AI 활용자)")

# 전략기대 (전체 응답자 기준)
run_efa(df_ai, cols_20, 1, "전략기대 (AI 활용자)")

print("각 문항 결측 수:")
print(df[cols_20].isna().sum())

print("\n4문항 모두 응답한 사람 수:")
print(df[cols_20].notna().all(axis=1).sum())

# ============================================================
# 7. 회귀용 데이터 구성
#    전략기대 응답자 + AI 활용자 교집합
# ============================================================

df_reg = df_ai.merge(
    df[["strategic_expectation"]],
    left_index=True,
    right_index=True,
    how="inner"
)

df_reg = df_reg[df_reg["strategic_expectation"].notna()]

print("\n회귀분석용 데이터 크기:", len(df_reg))

# ============================================================
# 8. 통제변수 기술통계 (AI 활용자 기준)
# ============================================================

print("\n통제변수 기술통계 (AI 활용자)")
print(
    df_ai[["gender", "rank_code", "career_code"]]
    .describe()
    .T[["mean", "std", "min", "max"]]
    .round(2)
)