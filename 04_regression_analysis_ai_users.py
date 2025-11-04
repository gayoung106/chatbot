import pandas as pd
import statsmodels.formula.api as smf

# ============================================================
# 1️ 데이터 로드 및 필터링 (AI 활용자만)
# ============================================================
df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()
print(f"AI 활용자 수: {len(df)}명")

# ============================================================
# 2️ 평균 변수 생성
# ============================================================
df["motivation_mean"] = df[[f"Q9_{i}" for i in range(1,5)]].mean(axis=1)
df["effect_mean"] = df[[f"Q7_{i}" for i in range(1,6)]].mean(axis=1)
df["support_mean"] = df[[f"Q16_{i}" for i in range(1,8)]].mean(axis=1)
df["expectation_mean"] = df[[f"Q20_{i}" for i in range(1,5)]].mean(axis=1)

print("\n 평균 변수 생성 완료")

# ============================================================
# 3️ 상관분석
# ============================================================
corr_vars = ["motivation_mean", "effect_mean", "expectation_mean", "support_mean"]
corr_matrix = df[corr_vars].corr()
print("\n🔗 [상관분석 결과 - AI 활용자]")
print(corr_matrix.round(3))

# ============================================================
# 4️ 매개효과 분석 (Baron & Kenny 3단계 접근)
# ============================================================

print("\n [회귀분석: 매개효과 검증]")

# Step 1: motivation → expectation
model1 = smf.ols("expectation_mean ~ motivation_mean", data=df).fit()
print("\nStep 1: motivation → expectation")
print(model1.summary())

# Step 2: motivation → effect
model2 = smf.ols("effect_mean ~ motivation_mean", data=df).fit()
print("\nStep 2: motivation → effect")
print(model2.summary())

# Step 3: motivation + effect → expectation
model3 = smf.ols("expectation_mean ~ motivation_mean + effect_mean", data=df).fit()
print("\nStep 3: motivation + effect → expectation")
print(model3.summary())

# ============================================================
# 5️ 조절효과 분석 (조직지원 × 활용동기)
# ============================================================

print("\n [회귀분석: 조절효과 검증]")

df["interaction"] = df["motivation_mean"] * df["support_mean"]

model_mod = smf.ols(
    "expectation_mean ~ motivation_mean * support_mean + gender + rank_code + career_code",
    data=df
).fit()
print(model_mod.summary())

# ============================================================
# 6️ 조절된 매개효과 (Moderated Mediation: PROCESS Model 7 구조)
# ============================================================

print("\n [회귀분석: 매개경로 조절효과 (Model 7)]")

df["interaction2"] = df["motivation_mean"] * df["support_mean"]
model_medmod = smf.ols(
    "effect_mean ~ motivation_mean * support_mean + gender + rank_code + career_code",
    data=df
).fit()
print(model_medmod.summary())

# ============================================================
# 7️ 결과 요약
# ============================================================
print("\n [분석 요약]")
print("1️ 상관분석: motivation, effect, expectation 간 양의 상관 기대")
print("2️ 매개효과: motivation → effect → expectation 경로 유의시 매개효과 존재")
print("3️ 조절효과: motivation_mean:support_mean 항목 p<0.05 → 조절효과 유의")
print("4️ Model7: 조절된 매개효과(Moderated Mediation) 검증")
