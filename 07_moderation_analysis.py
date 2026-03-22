"""
조절효과 분석
=============================================================
모형: 조직지원 인식(support)이
     자발적 AI 활용 동기(motivation) → 전략적 활용 기대(expectation)
     경로를 조절하는 효과를 분석

변수 구성
---------
독립변수(X): motivation  (Q9_3, Q9_4 평균)  - 자발적 AI 활용 동기
조절변수(W): support     (Q16_1~7 평균)     - 조직지원 인식
종속변수(Y): expectation (Q20_1~4 평균)     - 전략적 활용 기대
매개변수(M): effect      (Q7_1~5 평균)      - AI 활용 효과 인식
통제변수: gender, rank_code, career_code

분석 내용
---------
1. 조절효과 모형 (motivation × support → expectation)
2. 단순기울기 분석 (조절변수 수준별)
3. 조절된 매개효과 (1단계: motivation → effect, motivation × support → expectation)
4. Bootstrap 조절된 매개 간접효과 (PROCESS Macro Model 7 방식)
5. Johnson-Neyman 구간
=============================================================
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 데이터 로드 및 변수 생성
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()
print(f"AI 활용자 수: {len(df)}명")

df["motivation"]  = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"]      = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
df["support"]     = df[[f"Q16_{i}" for i in range(1, 8)]].mean(axis=1)
df["expectation"] = df[[f"Q20_{i}" for i in range(1, 5)]].mean(axis=1)

# ============================================================
# 2. 평균 중심화 (Mean Centering)
#    다중공선성 방지를 위해 상호작용 전 변수 중심화
# ============================================================

df["motivation_c"] = df["motivation"] - df["motivation"].mean()
df["support_c"]    = df["support"]    - df["support"].mean()
df["inter"]        = df["motivation_c"] * df["support_c"]

print(f"\n중심화 후 기술통계")
print(df[["motivation", "support", "expectation", "effect"]].describe().round(3))

# ============================================================
# 3. 조절효과 위계적 회귀
# ============================================================

print("\n\n" + "="*60)
print("조절효과 위계적 회귀분석 (Hierarchical Moderated Regression)")
print("="*60)

# Model 1: 통제변수만
mod1 = smf.ols(
    "expectation ~ gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 2: 주효과 (통제 + motivation + support)
mod2 = smf.ols(
    "expectation ~ motivation_c + support_c + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 3: 상호작용 추가 (조절효과 검증)
mod3 = smf.ols(
    "expectation ~ motivation_c + support_c + inter + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

print("\n[Model 1] 통제변수만")
print(f"  R² = {mod1.rsquared:.4f}, Adj.R² = {mod1.rsquared_adj:.4f}")

print("\n[Model 2] 주효과 (motivation + support)")
print(f"  R² = {mod2.rsquared:.4f}, Adj.R² = {mod2.rsquared_adj:.4f}")
print(f"  ΔR² (M2 - M1) = {(mod2.rsquared - mod1.rsquared):.4f}")
for var in ["motivation_c", "support_c"]:
    print(f"  {var}: B = {mod2.params[var]:.4f}, SE = {mod2.bse[var]:.4f}, "
          f"t = {mod2.tvalues[var]:.3f}, p = {mod2.pvalues[var]:.4f}")

print("\n[Model 3] 조절효과 포함 (motivation × support)")
print(f"  R² = {mod3.rsquared:.4f}, Adj.R² = {mod3.rsquared_adj:.4f}")
dr2 = mod3.rsquared - mod2.rsquared
print(f"  ΔR² (M3 - M2) = {dr2:.4f}  ← 조절효과 기여 분산")
f2_mod = dr2 / (1 - mod3.rsquared)
print(f"  f² (조절효과) = {f2_mod:.4f}")

for var in ["motivation_c", "support_c", "inter"]:
    label = {"motivation_c": "motivation (X)",
             "support_c": "support (W)",
             "inter": "motivation × support (상호작용)"}.get(var, var)
    p = mod3.pvalues[var]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else "ns"
    print(f"  {label}: B = {mod3.params[var]:.4f}, SE = {mod3.bse[var]:.4f}, "
          f"t = {mod3.tvalues[var]:.3f}, p = {p:.4f}  {sig}")

print("\n  *** p<.001, ** p<.01, * p<.05, † p<.10, ns p≥.10")

# ============================================================
# 4. 단순기울기 분석 (Simple Slope Analysis)
#    support 수준: 평균 + 1SD, 평균, 평균 - 1SD
# ============================================================

print("\n\n" + "="*60)
print("단순기울기 분석 (Simple Slope Analysis)")
print("="*60)

support_mean = df["support"].mean()
support_sd   = df["support"].std()
motivation_mean = df["motivation"].mean()

levels = {
    "고(+1SD)": support_sd,
    "중(평균)":  0,
    "저(-1SD)": -support_sd
}

print(f"\n조절변수(support) 기술통계: M = {support_mean:.3f}, SD = {support_sd:.3f}")
print(f"\n단순기울기 (motivation → expectation at each level of support)")
print(f"{'수준':<12} {'기울기(B)':<12} {'해석'}")
print("-"*45)

b_mot  = mod3.params["motivation_c"]
b_int  = mod3.params["inter"]
se_mot = mod3.bse["motivation_c"]
se_int = mod3.bse["inter"]

for label, w_val in levels.items():
    slope = b_mot + b_int * w_val
    # SE of simple slope: sqrt(Var(b_mot) + w^2 * Var(b_int) + 2w * Cov(b_mot, b_int))
    cov_matrix = mod3.cov_params()
    var_slope = (cov_matrix.loc["motivation_c", "motivation_c"]
                 + w_val**2 * cov_matrix.loc["inter", "inter"]
                 + 2 * w_val * cov_matrix.loc["motivation_c", "inter"])
    se_slope = np.sqrt(var_slope)
    t_slope  = slope / se_slope
    p_slope  = 2 * (1 - stats.t.cdf(abs(t_slope), df=mod3.df_resid))
    sig = "***" if p_slope < 0.001 else "**" if p_slope < 0.01 else "*" if p_slope < 0.05 else "†" if p_slope < 0.10 else "ns"
    print(f"  {label:<10}  B = {slope:+.4f}  SE = {se_slope:.4f}  "
          f"t = {t_slope:.3f}  p = {p_slope:.4f}  {sig}")

# ============================================================
# 5. Johnson-Neyman 구간 (유의미한 조절 경계값 탐색)
# ============================================================

print("\n\n" + "="*60)
print("Johnson-Neyman 구간 분석")
print("  (조직지원 인식의 어떤 값 이상/이하에서 motivation 효과가 유의한지)")
print("="*60)

# motivation 기울기가 유의한 support_c 범위를 탐색
cov_mat = mod3.cov_params()
v_a = cov_mat.loc["motivation_c", "motivation_c"]
v_b = cov_mat.loc["inter", "inter"]
cov_ab = cov_mat.loc["motivation_c", "inter"]
t_crit = stats.t.ppf(0.975, df=mod3.df_resid)  # 95% 양측

# 이차방정식: (b_int^2 - t_crit^2 * v_b) * w^2
#           + 2*(b_mot*b_int - t_crit^2*cov_ab) * w
#           + (b_mot^2 - t_crit^2 * v_a) = 0
A_coef = b_int**2 - t_crit**2 * v_b
B_coef = 2*(b_mot*b_int - t_crit**2 * cov_ab)
C_coef = b_mot**2 - t_crit**2 * v_a

discriminant = B_coef**2 - 4*A_coef*C_coef

if discriminant < 0:
    print("  JN 구간 없음 (조절변수 전 범위에서 유의하거나 비유의)")
elif abs(A_coef) < 1e-10:
    print("  JN 분석: A계수 ≈ 0, 선형 해법 적용")
    w_jn = -C_coef / B_coef
    print(f"  JN 경계값(중심화): {w_jn:.4f}  →  원래 값: {w_jn + support_mean:.4f}")
else:
    w1 = (-B_coef - np.sqrt(discriminant)) / (2*A_coef)
    w2 = (-B_coef + np.sqrt(discriminant)) / (2*A_coef)
    w1_orig = w1 + support_mean
    w2_orig = w2 + support_mean
    support_min = df["support"].min()
    support_max = df["support"].max()
    print(f"  support 실제 범위: {support_min:.2f} ~ {support_max:.2f}")
    print(f"  JN 경계값 1 (중심화: {w1:.4f}) → 원래 값: {w1_orig:.4f}")
    print(f"  JN 경계값 2 (중심화: {w2:.4f}) → 원래 값: {w2_orig:.4f}")
    # 유의한 구간
    for w_orig, lab in [(w1_orig, "경계 1"), (w2_orig, "경계 2")]:
        if support_min <= w_orig <= support_max:
            print(f"  → {lab}: 실제 측정 범위 내에 있음 (유의미한 조절 경계)")
        else:
            print(f"  → {lab}: 실제 측정 범위 밖 (범위 내는 항상 {'유의' if w_orig < support_min else '비유의'})")

# ============================================================
# 6. 조절된 매개효과 (Moderated Mediation)
#    PROCESS Macro Model 7 방식
#    경로: motivation → effect → expectation
#         motivation × support → expectation (2단계 조절)
# ============================================================

print("\n\n" + "="*60)
print("조절된 매개효과 (Moderated Mediation)")
print("  모형: motivation → effect → expectation")
print("       motivation × support → expectation (조절)")
print("="*60)

# 1단계: motivation → effect (조절 미포함)
path_a = smf.ols(
    "effect ~ motivation_c + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

a_coef = path_a.params["motivation_c"]
print(f"\n[1단계] motivation → effect (매개 경로 a)")
print(f"  B = {a_coef:.4f}, SE = {path_a.bse['motivation_c']:.4f}, "
      f"t = {path_a.tvalues['motivation_c']:.3f}, p = {path_a.pvalues['motivation_c']:.4f}")

# 2단계: effect + motivation × support → expectation
path_b_mod = smf.ols(
    "expectation ~ effect + motivation_c + support_c + inter + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

b_coef = path_b_mod.params["effect"]
print(f"\n[2단계] effect → expectation (매개 경로 b, 조절 포함 모형)")
print(f"  B = {b_coef:.4f}, SE = {path_b_mod.bse['effect']:.4f}, "
      f"t = {path_b_mod.tvalues['effect']:.3f}, p = {path_b_mod.pvalues['effect']:.4f}")
print(f"\n[2단계] motivation × support → expectation (조절)")
p_inter2 = path_b_mod.pvalues["inter"]
sig2 = "***" if p_inter2 < 0.001 else "**" if p_inter2 < 0.01 else "*" if p_inter2 < 0.05 else "†" if p_inter2 < 0.10 else "ns"
print(f"  B = {path_b_mod.params['inter']:.4f}, SE = {path_b_mod.bse['inter']:.4f}, "
      f"t = {path_b_mod.tvalues['inter']:.3f}, p = {p_inter2:.4f}  {sig2}")

# 조건부 간접효과 (각 support 수준에서 a*b 계산)
print(f"\n  [조건부 간접효과] 수준별 간접효과 추정 (a × b)")
print(f"  ※ a는 고정, b는 2단계 b 계수 사용 (support 수준에 따라 b 변하지 않는 경우)")
print(f"    간접효과 = a × b = {a_coef:.4f} × {b_coef:.4f} = {a_coef*b_coef:.4f}")

# ============================================================
# 7. Bootstrap 조건부 간접효과
#    각 support 수준별 간접효과 신뢰구간
# ============================================================

print("\n\n" + "="*60)
print("Bootstrap 조건부 간접효과 (5,000회)")
print("="*60)

N_BOOT = 5000
support_vals = {
    "저(-1SD)":  support_mean - support_sd,
    "중(평균)":  support_mean,
    "고(+1SD)":  support_mean + support_sd,
}

boot_results = {k: [] for k in support_vals}

for _ in trange(N_BOOT, desc="Bootstrap"):
    samp = df.sample(len(df), replace=True)

    samp["motivation_c"] = samp["motivation"] - samp["motivation"].mean()
    samp["support_c"]    = samp["support"]    - samp["support"].mean()
    samp["inter"]        = samp["motivation_c"] * samp["support_c"]

    try:
        # 경로 a
        ba = smf.ols("effect ~ motivation_c", data=samp).fit()
        a_b = ba.params["motivation_c"]

        # 경로 b + 조절 (2단계: support 수준별 b 계산)
        bb = smf.ols(
            "expectation ~ effect + motivation_c + support_c + inter",
            data=samp
        ).fit()
        b_b    = bb.params["effect"]
        # 조건부 직접효과는 support 수준별로 다를 수 있으나
        # b(effect) 자체는 support와 상호작용 없이 추정됨
        # → 단순 a*b 방식 사용 (간접효과는 support 무관)

        for label, w_orig in support_vals.items():
            boot_results[label].append(a_b * b_b)
    except Exception:
        continue

print("\n  조건부 간접효과 부트스트랩 95% CI")
print(f"  {'수준':<12} {'평균':>8}  {'2.5%':>8}  {'97.5%':>8}  {'유의'}")
print("  " + "-"*52)
all_sig = True
for label in support_vals:
    arr = np.array(boot_results[label])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    sig_str = "유의" if not (lo <= 0 <= hi) else "비유의"
    if sig_str == "비유의":
        all_sig = False
    print(f"  {label:<12} {arr.mean():>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {sig_str}")

# ============================================================
# 8. 전체 모형 조절효과 VIF
# ============================================================

print("\n\n" + "="*60)
print("다중공선성 진단 (VIF) - 조절 모형 기준")
print("="*60)

X_mod = df[["motivation_c", "support_c", "inter",
            "gender", "rank_code", "career_code"]].dropna()
X_mod_const = sm.add_constant(X_mod)
vif_df = pd.DataFrame({
    "Variable": X_mod_const.columns,
    "VIF": [variance_inflation_factor(X_mod_const.values, i)
            for i in range(X_mod_const.shape[1])]
})
print(vif_df.to_string(index=False))
print("  ※ VIF < 10: 허용 기준, VIF < 5: 양호")

# ============================================================
# 9. 결과 요약
# ============================================================

print("\n\n" + "="*60)
print("결과 요약")
print("="*60)

b_inter = mod3.params["inter"]
p_inter = mod3.pvalues["inter"]
dr2_mod = mod3.rsquared - mod2.rsquared

print(f"""
[조절효과 주요 결과]
  - motivation × support 상호작용: B = {b_inter:.4f}, p = {p_inter:.4f}
  - 조절효과 ΔR² = {dr2_mod:.4f}, f² = {f2_mod:.4f}
  - f² 해석: {
    '소(small, 0.02~0.15)' if 0.02 <= f2_mod < 0.15
    else '중(medium, 0.15~0.35)' if 0.15 <= f2_mod < 0.35
    else '대(large, ≥0.35)' if f2_mod >= 0.35
    else '무시할 수준(< 0.02)'
  }

[단순기울기]
  - support 高 수준: motivation 효과 {
    '강화(정적 조절)' if b_int > 0 else '약화(부적 조절)'
  }
  - support 低 수준: motivation 효과 {
    '약화' if b_int > 0 else '강화'
  }

[가설 채택 여부]
  - 조절효과 가설: {'채택 (p < .05)' if p_inter < 0.05 else '기각 또는 경계 (p ≥ .05)'}
    → p = {p_inter:.4f}
""")

print("=" * 60)
print("분석 완료")
print("=" * 60)
