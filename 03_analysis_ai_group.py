import pandas as pd
import numpy as np
from pingouin import cronbach_alpha
from sklearn.decomposition import FactorAnalysis

# ============================================================
# 1️ 데이터 로드 및 AI 활용 여부 분리
# ============================================================
df = pd.read_csv("chatbot_output_selected_preprocessed.csv")

df_ai = df[df["Q3"] == 1].copy()   # AI 활용자
df_non = df[df["Q3"] == 0].copy()  # 비활용자

print(f"전체 응답자 수: {len(df)}, AI 활용자: {len(df_ai)}, 비활용자: {len(df_non)}")

# ============================================================
# 2️ 분석 대상 문항
# ============================================================
cols_7 = [f"Q7_{i}" for i in range(1,6)]   # 인식된 업무효과
cols_9 = [f"Q9_{i}" for i in range(1,5)]   # 활용동기
cols_16 = [f"Q16_{i}" for i in range(1,8)] # 조직지원 인식
cols_20 = [f"Q20_{i}" for i in range(1,5)] # 전략적 활용 기대

all_cols = cols_7 + cols_9 + cols_16 + cols_20

# ============================================================
# 3️ (1) 기술통계: AI 활용자 / 비활용자 각각
# ============================================================
def descriptive_stats(df_sub, label):
    desc = df_sub[all_cols].describe().T
    desc["skew"] = df_sub[all_cols].skew()
    desc["kurt"] = df_sub[all_cols].kurt()
    print(f"\n [기술통계표 - {label}]")
    print(desc[["mean","std","min","max","skew","kurt"]].round(3))

descriptive_stats(df_ai, "AI 활용자")
descriptive_stats(df_non, "AI 비활용자")

# ============================================================
# 4️ (2) 신뢰도분석 (Cronbach’s α)
# ============================================================
def reliability(df_sub, label):
    print(f"\n [신뢰도분석 - {label}]")
    alpha_7 = cronbach_alpha(df_sub[cols_7])[0]
    alpha_9 = cronbach_alpha(df_sub[cols_9])[0]
    alpha_16 = cronbach_alpha(df_sub[cols_16])[0]
    alpha_20 = cronbach_alpha(df_sub[cols_20])[0]
    print(f"업무효과(Q7): {alpha_7:.3f}")
    print(f"활용동기(Q9): {alpha_9:.3f}")
    print(f"조직지원(Q16): {alpha_16:.3f}")
    print(f"전략기대(Q20): {alpha_20:.3f}")

reliability(df_ai, "AI 활용자")
reliability(df_non, "AI 비활용자")

# ============================================================
# 5️ (3) 탐색적 요인분석 (EFA)
# ============================================================
def efa_analysis(df_sub, label):
    print(f"\n [탐색적 요인분석 - {label}]")
    fa = FactorAnalysis(n_components=4, random_state=42)
    fa.fit(df_sub[all_cols])
    loadings = pd.DataFrame(fa.components_.T, index=all_cols, columns=[f"Factor{i+1}" for i in range(4)])
    print(loadings.round(3))

efa_analysis(df_ai, "AI 활용자")
efa_analysis(df_non, "AI 비활용자")

# ============================================================
# 6️ (4) 통제변수 기술통계 (AI 활용자 기준)
# ============================================================
print("\n⚙️ [통제변수 기술통계 - AI 활용자]")
control_desc = df_ai[["gender","rank_code","career_code"]].describe().T
print(control_desc[["mean","std","min","max"]].round(2))

# ============================================================
# 7️ (5) 요약
# ============================================================
print("\n [요약]")
print("1. Q3==1 집단만이 이후 회귀분석 대상.")
print("2. 각 변수별 신뢰도 α ≥ 0.7이면 평균 산출로 전환하여 매개/조절 분석 진행.")
print("3. AI 비활용자 집단은 기술통계 및 참고용으로만 사용.")
