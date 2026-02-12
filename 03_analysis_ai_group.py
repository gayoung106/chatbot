import pandas as pd
import numpy as np
from pingouin import cronbach_alpha
from sklearn.decomposition import FactorAnalysis

# ============================================================
# 1. 데이터 로드 및 AI 활용 여부 분리
# ============================================================
df = pd.read_csv("chatbot_output_selected_preprocessed.csv")

df_ai = df[df["Q3"] == 1].copy()    # AI 활용자
df_non = df[df["Q3"] == 0].copy()   # AI 비활용자

print(f"전체 응답자 수: {len(df)}, AI 활용자: {len(df_ai)}, 비활용자: {len(df_non)}")

# ============================================================
# 2. 분석 대상 문항 정의
# ============================================================
# 인식된 업무효과
cols_7 = [f"Q7_{i}" for i in range(1, 6)]

# 활용동기 (이론적으로 분리)
cols_9_passive = ["Q9_1", "Q9_2"]        # 수동적 활용동기
cols_9_voluntary = ["Q9_3", "Q9_4"]      # 자발적 활용동기

# 조직지원 인식
cols_16 = [f"Q16_{i}" for i in range(1, 8)]

# 전략적 활용 기대
cols_20 = [f"Q20_{i}" for i in range(1, 5)]

# EFA 대상 문항
efa_cols = cols_7 + cols_16 + cols_20

# ============================================================
# 3. Likert 문항의 0값 → 결측치(NaN) 처리
# ============================================================
likert_cols = efa_cols + cols_9_passive + cols_9_voluntary

df_ai[likert_cols] = df_ai[likert_cols].replace(0, np.nan)
df_non[likert_cols] = df_non[likert_cols].replace(0, np.nan)

# ============================================================
# 4. 기술통계
# ============================================================
def descriptive_stats(df_sub, label, cols):
    desc = df_sub[cols].describe().T
    desc["skew"] = df_sub[cols].skew()
    desc["kurt"] = df_sub[cols].kurt()

    print(f"\n[기술통계표 - {label}]")
    print(desc[["mean", "std", "min", "max", "skew", "kurt"]].round(3))


# AI 활용자: Q7 + Q16 + Q20
descriptive_stats(df_ai, "AI 활용자", cols_7 + cols_16 + cols_20)

# AI 비활용자: Q16 + Q20만 (Q7 제외)
descriptive_stats(df_non, "AI 비활용자", cols_16 + cols_20)

# ============================================================
# 5. 신뢰도 분석 (Cronbach’s α)
# ============================================================
def reliability(df_sub, label):
    print("\n" + "=" * 60)
    print(f"[신뢰도 분석 - {label}]")
    print("=" * 60)

    print(f"업무효과(Q7): {cronbach_alpha(df_sub[cols_7].dropna())[0]:.3f}")
    print(f"조직지원(Q16): {cronbach_alpha(df_sub[cols_16].dropna())[0]:.3f}")
    print(f"전략기대(Q20): {cronbach_alpha(df_sub[cols_20].dropna())[0]:.3f}")
    print(f"수동적 활용동기(Q9-1,2): {cronbach_alpha(df_sub[cols_9_passive].dropna())[0]:.3f}")
    print(f"자발적 활용동기(Q9-3,4): {cronbach_alpha(df_sub[cols_9_voluntary].dropna())[0]:.3f}")


reliability(df_ai, "AI 활용자")

# ============================================================
# 6. 활용동기 평균 변수 생성
# ============================================================
df_ai["motivation_passive"] = df_ai[cols_9_passive].mean(axis=1)
df_ai["motivation_voluntary"] = df_ai[cols_9_voluntary].mean(axis=1)

# ============================================================
# 7. 탐색적 요인분석 (EFA) – AI 활용자만
# ============================================================
print("\n" + "=" * 60)
print("[탐색적 요인분석 - AI 활용자]")
print("=" * 60)

efa_data = df_ai[efa_cols].dropna()

fa = FactorAnalysis(n_components=3, random_state=42)
fa.fit(efa_data)

loadings = pd.DataFrame(
    fa.components_.T,
    index=efa_cols,
    columns=["Work_Effect", "Org_Support", "Strategic_Expectation"]
)

print(loadings.round(3))

# ============================================================
# 8. 통제변수 기술통계 (AI 활용자)
# ============================================================
print("\n⚙️ [통제변수 기술통계 - AI 활용자]")
control_desc = df_ai[["gender", "rank_code", "career_code"]].describe().T
print(control_desc[["mean", "std", "min", "max"]].round(2))

# ============================================================
# 9. 요약
# ============================================================
print("\n[요약]")
print("1. Q3 == 1 (AI 활용자) 집단만이 이후 회귀분석 대상.")
print("2. 활용동기는 수동적/자발적으로 이론 분리하여 독립변수로 사용.")
print("3. EFA는 Q7, Q16, Q20에 대해서만 AI 활용자 기준으로 수행.")
print("4. AI 비활용자 집단은 기술통계 및 참고용으로만 사용.")
