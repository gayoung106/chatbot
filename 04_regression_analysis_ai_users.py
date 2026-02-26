import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from tqdm import trange
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# 1. 데이터 로드
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()
print(f"AI 활용자 수: {len(df)}명")

# ============================================================
# 2. 변수 생성
# ============================================================

df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"] = df[[f"Q7_{i}" for i in range(1,6)]].mean(axis=1)
df["support"] = df[[f"Q16_{i}" for i in range(1,8)]].mean(axis=1)
df["expectation"] = df[[f"Q20_{i}" for i in range(1,5)]].mean(axis=1)

# ============================================================
# 2-1. 상관관계 분석
# ============================================================

print("\n===== Correlation Matrix =====")

corr_vars = df[[
    "expectation",
    "motivation",
    "effect",
    "support",
    "gender",
    "rank_code",
    "career_code"
]]

corr_matrix = corr_vars.corr()

print(corr_matrix.round(3))

def corr_with_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvals = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvals[r][c] = stats.pearsonr(df[r], df[c])[1]
    return pvals

p_matrix = corr_with_pvalues(corr_vars)
print(p_matrix)
# ============================================================
# 3. 위계적 회귀 (HC3 Robust)
# ============================================================

print("\n===== Hierarchical Regression =====")

# Model 1: Controls
m1 = smf.ols(
    "expectation ~ gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 2: + Motivation + Support
m2 = smf.ols(
    "expectation ~ motivation + support + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 3: + Mediator
m3 = smf.ols(
    "expectation ~ motivation + effect + support + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

print(m1.summary())
print(m2.summary())
print(m3.summary())

print("\nR² 변화")
print("ΔR² (M2-M1) =", round(m2.rsquared - m1.rsquared, 3))
print("ΔR² (M3-M2) =", round(m3.rsquared - m2.rsquared, 3))

f2 = (m3.rsquared - m2.rsquared) / (1 - m3.rsquared)
print("f² =", round(f2, 3))

# ============================================================
# 4. VIF
# ============================================================

print("\n===== VIF =====")

X = df[["motivation", "effect", "support",
        "gender", "rank_code", "career_code"]]

X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print(vif_data)

# ============================================================
# 5. 매개효과 (Robust + Sobel)
# ============================================================

print("\n===== Mediation =====")

model_a = smf.ols(
    "effect ~ motivation + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

model_b = smf.ols(
    "expectation ~ motivation + effect + support + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

a = model_a.params["motivation"]
sa = model_a.bse["motivation"]
b = model_b.params["effect"]
sb = model_b.bse["effect"]

sobel = (a*b) / np.sqrt(b**2 * sa**2 + a**2 * sb**2)
p_sobel = 2*(1 - stats.norm.cdf(abs(sobel)))

print(f"Sobel Z = {sobel:.3f}, p = {p_sobel:.5f}")

# ============================================================
# 6. Bootstrap 간접효과
# ============================================================

N_BOOT = 5000
indirect_effects = []

for _ in trange(N_BOOT):
    sample = df.sample(len(df), replace=True)

    a = smf.ols(
        "effect ~ motivation",
        data=sample
    ).fit().params["motivation"]

    b = smf.ols(
        "expectation ~ motivation + effect + support",
        data=sample
    ).fit().params["effect"]

    indirect_effects.append(a * b)

indirect_effects = np.array(indirect_effects)
ci_lower, ci_upper = np.percentile(indirect_effects, [2.5, 97.5])

print("\nBootstrapped Indirect Effect")
print(f"Mean = {indirect_effects.mean():.3f}")
print(f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")

print("\n 매개모형 분석 완료")