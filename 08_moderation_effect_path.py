"""
조절효과 분석 2: effect(AI 활용 효과 인식) × support(조직지원 인식) → expectation
=============================================================
모형: 조직지원 인식(W)이
     AI 활용 효과 인식(M) → 전략적 활용 기대(Y)
     경로를 조절하는 효과를 분석

     ※ PROCESS Macro Model 14 (Moderated Mediation) 방식
        motivation(X) → effect(M) → expectation(Y)
                            ↑             ↑
                         [1단계]     [2단계: W 조절]

변수 구성
---------
독립변수(X): motivation  (Q9_3, Q9_4 평균)  - 자발적 AI 활용 동기
매개변수(M): effect      (Q7_1~5 평균)      - AI 활용 효과 인식
조절변수(W): support     (Q16_1~7 평균)     - 조직지원 인식
종속변수(Y): expectation (Q20_1~4 평균)     - 전략적 활용 기대
통제변수: gender, rank_code, career_code

분석 내용
---------
1. 조절효과 위계적 회귀 (effect × support → expectation)
2. 단순기울기 분석 (support 수준별 effect → expectation)
3. Johnson-Neyman 구간
4. PROCESS Model 14 방식: Bootstrap 조건부 간접효과
   (motivation → effect → expectation, support로 2단계 조절)
5. FloodLight 분석 (선택적)
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

# 평균 중심화
df["motivation_c"] = df["motivation"] - df["motivation"].mean()
df["effect_c"]     = df["effect"]     - df["effect"].mean()
df["support_c"]    = df["support"]    - df["support"].mean()

# 상호작용항: effect × support (2단계 조절)
df["inter_es"]     = df["effect_c"] * df["support_c"]
# 상호작용항: motivation × support (1단계, 비교용)
df["inter_ms"]     = df["motivation_c"] * df["support_c"]

support_mean = df["support"].mean()
support_sd   = df["support"].std()

print(f"\n기술통계")
print(df[["motivation", "effect", "support", "expectation"]].describe().round(3))

# ============================================================
# 2. 조절효과 위계적 회귀
#    종속변수: expectation
#    조절: effect × support
# ============================================================

print("\n\n" + "="*60)
print("【분석 1】 조절효과 위계적 회귀")
print("  effect(M) × support(W) → expectation(Y)")
print("="*60)

# Model 1: 통제변수
es1 = smf.ols(
    "expectation ~ gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 2: 주효과 (effect + support)
es2 = smf.ols(
    "expectation ~ effect_c + support_c + motivation_c + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 3: 상호작용 추가
es3 = smf.ols(
    "expectation ~ effect_c + support_c + inter_es + motivation_c + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

print(f"\n[Model 1] 통제변수만")
print(f"  R² = {es1.rsquared:.4f}, Adj.R² = {es1.rsquared_adj:.4f}")

print(f"\n[Model 2] 주효과 (effect + support + motivation)")
print(f"  R² = {es2.rsquared:.4f}, Adj.R² = {es2.rsquared_adj:.4f}")
print(f"  ΔR² (M2-M1) = {(es2.rsquared - es1.rsquared):.4f}")
for var, lbl in [("effect_c","effect(M)"), ("support_c","support(W)"), ("motivation_c","motivation(X)")]:
    p = es2.pvalues[var]
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "ns"
    print(f"  {lbl}: B={es2.params[var]:+.4f}, SE={es2.bse[var]:.4f}, "
          f"t={es2.tvalues[var]:.3f}, p={p:.4f} {sig}")

print(f"\n[Model 3] 조절효과 포함 (effect × support)")
print(f"  R² = {es3.rsquared:.4f}, Adj.R² = {es3.rsquared_adj:.4f}")
dr2_es = es3.rsquared - es2.rsquared
f2_es  = dr2_es / (1 - es3.rsquared)
print(f"  ΔR² (M3-M2) = {dr2_es:.4f}  ← 조절효과 기여")
print(f"  f² (조절효과) = {f2_es:.4f}")

for var, lbl in [("effect_c","effect(M)"), ("support_c","support(W)"),
                 ("inter_es","effect × support (조절)"), ("motivation_c","motivation(X)")]:
    p = es3.pvalues[var]
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "ns"
    print(f"  {lbl}: B={es3.params[var]:+.4f}, SE={es3.bse[var]:.4f}, "
          f"t={es3.tvalues[var]:.3f}, p={p:.4f} {sig}")

print("\n  *** p<.001, ** p<.01, * p<.05, † p<.10, ns p≥.10")

# ============================================================
# 3. 단순기울기 분석 (Simple Slope Analysis)
#    support 수준: 평균+1SD, 평균, 평균-1SD
# ============================================================

print("\n\n" + "="*60)
print("【분석 2】 단순기울기 분석")
print("  effect → expectation at each level of support")
print("="*60)

b_eff   = es3.params["effect_c"]
b_int_e = es3.params["inter_es"]
cov_es  = es3.cov_params()

levels = {
    "高(+1SD)":  support_sd,
    "中(평균)":   0.0,
    "低(-1SD)":  -support_sd,
}

print(f"\n조절변수(support): M = {support_mean:.3f}, SD = {support_sd:.3f}")
print(f"\n{'수준':<12} {'B':>8} {'SE':>8} {'t':>7} {'p':>8} {'유의'}")
print("-"*52)

for label, w_val in levels.items():
    slope = b_eff + b_int_e * w_val
    var_slope = (cov_es.loc["effect_c","effect_c"]
                 + w_val**2 * cov_es.loc["inter_es","inter_es"]
                 + 2 * w_val * cov_es.loc["effect_c","inter_es"])
    se_slope  = np.sqrt(var_slope)
    t_slope   = slope / se_slope
    p_slope   = 2 * (1 - stats.t.cdf(abs(t_slope), df=es3.df_resid))
    sig = "***" if p_slope<0.001 else "**" if p_slope<0.01 else "*" if p_slope<0.05 else "†" if p_slope<0.10 else "ns"
    print(f"  {label:<10} {slope:>+8.4f} {se_slope:>8.4f} {t_slope:>7.3f} {p_slope:>8.4f}  {sig}")

# ============================================================
# 4. Johnson-Neyman 구간
# ============================================================

print("\n\n" + "="*60)
print("【분석 3】 Johnson-Neyman 구간")
print("  (어떤 support 값에서 effect 효과가 유의/비유의 전환되는지)")
print("="*60)

v_a   = cov_es.loc["effect_c","effect_c"]
v_b   = cov_es.loc["inter_es","inter_es"]
cov_ab = cov_es.loc["effect_c","inter_es"]
t_crit = stats.t.ppf(0.975, df=es3.df_resid)

A_c = b_int_e**2 - t_crit**2 * v_b
B_c = 2*(b_eff*b_int_e - t_crit**2 * cov_ab)
C_c = b_eff**2  - t_crit**2 * v_a

disc = B_c**2 - 4*A_c*C_c
support_min = df["support"].min()
support_max = df["support"].max()

print(f"\n  support 실제 범위: {support_min:.2f} ~ {support_max:.2f}")

if disc < 0:
    print("  JN 구간 없음 (전 범위에서 동일한 유의 상태)")
elif abs(A_c) < 1e-10:
    w_jn = -C_c / B_c
    print(f"  JN 경계: {w_jn + support_mean:.4f}")
else:
    w1 = (-B_c - np.sqrt(disc)) / (2*A_c)
    w2 = (-B_c + np.sqrt(disc)) / (2*A_c)
    for wc, lab in [(w1,"경계 1"),(w2,"경계 2")]:
        wo = wc + support_mean
        in_range = support_min <= wo <= support_max
        print(f"  JN {lab}: support = {wo:.4f}  "
              f"({'측정 범위 내 ✓' if in_range else '측정 범위 밖'})")
        if in_range:
            # 경계 바깥쪽 유의 여부
            slope_lo = b_eff + b_int_e*(support_min - support_mean)
            slope_hi = b_eff + b_int_e*(support_max - support_mean)
            print(f"    → support < {wo:.2f}: effect 영향 {'유의' if slope_lo/abs(slope_lo+1e-10) > 0 else '비유의'} 방향")
            print(f"    → support > {wo:.2f}: effect 영향 방향 전환")

# ============================================================
# 5. PROCESS Model 14 방식
#    Bootstrap 조건부 간접효과
#    경로: motivation(X) → effect(M) → expectation(Y)
#    2단계(b경로)에서 support 조절
# ============================================================

print("\n\n" + "="*60)
print("【분석 4】 조절된 매개효과 Bootstrap")
print("  PROCESS Model 14: motivation→effect→expectation")
print("  (2단계 b경로에서 support가 effect→expectation 조절)")
print("="*60)

support_vals = {
    "低(-1SD)":  support_mean - support_sd,
    "中(평균)":  support_mean,
    "高(+1SD)":  support_mean + support_sd,
}

N_BOOT = 5000
boot_cond = {k: [] for k in support_vals}

for _ in trange(N_BOOT, desc="Bootstrap (Model 14)"):
    samp = df.sample(len(df), replace=True)

    # 중심화 재계산
    samp["motivation_c"] = samp["motivation"] - samp["motivation"].mean()
    samp["effect_c"]     = samp["effect"]     - samp["effect"].mean()
    samp["support_c"]    = samp["support"]    - samp["support"].mean()
    samp["inter_es"]     = samp["effect_c"]   * samp["support_c"]

    try:
        # 경로 a: motivation → effect
        ba = smf.ols("effect_c ~ motivation_c", data=samp).fit()
        a_b = ba.params["motivation_c"]

        # 경로 b (조절): effect + effect×support → expectation
        bb = smf.ols(
            "expectation ~ effect_c + support_c + inter_es + motivation_c",
            data=samp
        ).fit()
        b_eff_b = bb.params["effect_c"]
        b_int_b = bb.params["inter_es"]

        # 조건부 간접효과: a × (b + b_int × W)
        for label, w_orig in support_vals.items():
            w_c = w_orig - samp["support"].mean()  # 부트스트랩 샘플 기준 중심화
            cond_b = b_eff_b + b_int_b * w_c
            boot_cond[label].append(a_b * cond_b)
    except Exception:
        continue

print("\n  조건부 간접효과 95% Bootstrap CI")
print(f"  {'수준':<12} {'평균':>8} {'2.5%':>8} {'97.5%':>8}  {'유의'}")
print("  " + "-"*48)
for label in support_vals:
    arr = np.array(boot_cond[label])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    sig_str = "✓ 유의" if not (lo <= 0 <= hi) else "✗ 비유의"
    print(f"  {label:<12} {arr.mean():>8.4f} {lo:>8.4f} {hi:>8.4f}  {sig_str}")

# 조절된 매개지수 (Index of Moderated Mediation)
print("\n  [조절된 매개지수 (Index of Moderated Mediation)]")
print("  간접효과 차이 = (고 수준 간접) − (저 수준 간접)")

arr_hi = np.array(boot_cond["高(+1SD)"])
arr_lo = np.array(boot_cond["低(-1SD)"])
diff   = arr_hi - arr_lo
lo_d, hi_d = np.percentile(diff, [2.5, 97.5])
print(f"  차이 평균 = {diff.mean():.4f}, 95% CI = [{lo_d:.4f}, {hi_d:.4f}]")
print(f"  → CI에 0 포함 여부: {'포함 (조절된 매개 비유의)' if lo_d<=0<=hi_d else '미포함 (조절된 매개 유의)'}")

# ============================================================
# 6. VIF
# ============================================================

print("\n\n" + "="*60)
print("다중공선성 진단 (VIF) - Model 3 기준")
print("="*60)

X_v = df[["effect_c","support_c","inter_es","motivation_c",
          "gender","rank_code","career_code"]].dropna()
X_v_c = sm.add_constant(X_v)
vif_df = pd.DataFrame({
    "Variable": X_v_c.columns,
    "VIF": [variance_inflation_factor(X_v_c.values, i)
            for i in range(X_v_c.shape[1])]
})
print(vif_df.to_string(index=False))
print("  ※ VIF < 10: 허용 기준")

# ============================================================
# 7. 종합 결과 요약
# ============================================================

b_inter_es = es3.params["inter_es"]
p_inter_es = es3.pvalues["inter_es"]

print("\n\n" + "="*60)
print("종합 결과 요약")
print("="*60)

f2_label = ('소(small, 0.02~0.15)' if 0.02 <= f2_es < 0.15
            else '중(medium, 0.15~0.35)' if 0.15 <= f2_es < 0.35
            else '대(large, ≥0.35)' if f2_es >= 0.35
            else '무시할 수준(< 0.02)')

print(f"""
[effect × support 조절효과]
  상호작용: B = {b_inter_es:+.4f}, p = {p_inter_es:.4f}
  ΔR²     = {dr2_es:.4f}, f² = {f2_es:.4f} ({f2_label})
  → 가설 채택: {'✓ 채택 (p < .05)' if p_inter_es < 0.05 else '✗ 기각 (p ≥ .05)'}

[조절 방향 해석 (부적/정적)]
  상호작용 계수 B = {b_inter_es:+.4f}
  → support 높을수록 effect→expectation 기울기 {
      '증가 (정적 조절: 조직지원이 효과 인식의 영향을 강화)' if b_inter_es > 0
      else '감소 (부적 조절: 조직지원이 효과 인식의 영향을 약화)'
  }

[단순기울기 요약]
  低(-1SD): effect 효과 B = {b_eff + b_int_e*(-support_sd):+.4f}
  中(평균): effect 효과 B = {b_eff:+.4f}
  高(+1SD): effect 효과 B = {b_eff + b_int_e*(support_sd):+.4f}

[조절된 매개효과]
  CI에 0 포함 여부(고-저 차이): {
      '미포함 → 조절된 매개 유의' if lo_d > 0 or hi_d < 0
      else '포함 → 조절된 매개 비유의'
  }
""")

print("="*60)
print("분석 완료")
print("="*60)
