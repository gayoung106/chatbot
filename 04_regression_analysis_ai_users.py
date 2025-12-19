import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from tqdm import trange

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

# PROCESS Model 4 (부트스트랩 매개효과)
N_BOOT = 5000
indirect_effects = []

for _ in trange(N_BOOT):
    sample = df.sample(len(df), replace=True)

    # a path: X → M
    a_model = smf.ols("effect_mean ~ motivation_mean", data=sample).fit()
    a = a_model.params["motivation_mean"]

    # b path: M → Y (X 통제)
    b_model = smf.ols(
        "expectation_mean ~ motivation_mean + effect_mean",
        data=sample
    ).fit()
    b = b_model.params["effect_mean"]

    indirect_effects.append(a * b)

indirect_effects = np.array(indirect_effects)

ci_lower, ci_upper = np.percentile(indirect_effects, [2.5, 97.5])

print("\n[PROCESS Model 4: 부트스트랩 매개효과]")
print(f"간접효과 평균: {indirect_effects.mean():.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# PROCESS Model 7 (조절된 매개효과)
conditional_indirect = {
    "low": [],
    "mean": [],
    "high": []
}

support_mean = df["support_mean"].mean()
support_sd = df["support_mean"].std()

levels = {
    "low": support_mean - support_sd,
    "mean": support_mean,
    "high": support_mean + support_sd
}

for _ in trange(N_BOOT):
    sample = df.sample(len(df), replace=True)

    # a path with moderation
    a_model = smf.ols(
        "effect_mean ~ motivation_mean * support_mean",
        data=sample
    ).fit()

    a1 = a_model.params["motivation_mean"]
    a3 = a_model.params["motivation_mean:support_mean"]

    # b path
    b_model = smf.ols(
        "expectation_mean ~ motivation_mean + effect_mean",
        data=sample
    ).fit()
    b = b_model.params["effect_mean"]

    for level, w in levels.items():
        a_cond = a1 + a3 * w
        conditional_indirect[level].append(a_cond * b)

print("\n[PROCESS Model 7: 조건부 간접효과]")
for level in conditional_indirect:
    arr = np.array(conditional_indirect[level])
    ci_l, ci_u = np.percentile(arr, [2.5, 97.5])
    print(f"{level.upper()} → mean={arr.mean():.3f}, 95% CI=[{ci_l:.3f}, {ci_u:.3f}]")


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
