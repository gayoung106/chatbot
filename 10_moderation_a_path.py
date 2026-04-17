"""
조절효과 분석: support × motivation → effect
=============================================================
모형: 조직지원 인식(W)이
     자발적 AI 활용 동기(X) → AI 활용 효과 인식(M)
     경로(a경로)를 조절하는 효과 검증

     ※ PROCESS Macro Model 7 방식 (1단계 a경로 조절)

구조:
  motivation(X) ──[support(W) 조절]──→ effect(M) ──→ expectation(Y)

이 경로가 유의하면:
  - support가 인과 연쇄 내부에 통합되는 구조 완성
  - 조절된 매개효과(Moderated Mediation, Model 7) 검증 가능
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
df["support"]     = df[[f"Q16_{i}" for i in range(1, 7)]].mean(axis=1)  # Q16_7 제외: 개인 관심 문항, motivation과 중첩
df["expectation"] = df[[f"Q20_{i}" for i in range(2, 5)]].mean(axis=1)  # Q20_1 제외: 업무효과와 개념 중첩

# 평균 중심화
df["motivation_c"] = df["motivation"] - df["motivation"].mean()
df["support_c"]    = df["support"]    - df["support"].mean()
df["inter_ms"]     = df["motivation_c"] * df["support_c"]   # 조절항

support_mean = df["support"].mean()
support_sd   = df["support"].std()

# ============================================================
# 2. 위계적 회귀: 종속변수 = effect(M)
#    조절변수 = support(W)
#    독립변수 = motivation(X)
# ============================================================

print("\n\n" + "="*60)
print("【분석】 조절효과 위계적 회귀")
print("  종속변수: effect (AI 활용 효과 인식)")
print("  motivation(X) × support(W) → effect(M)")
print("="*60)

# Model 1: 통제변수
m1 = smf.ols(
    "effect ~ gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 2: 주효과
m2 = smf.ols(
    "effect ~ motivation_c + support_c + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

# Model 3: 상호작용 (조절효과)
m3 = smf.ols(
    "effect ~ motivation_c + support_c + inter_ms + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

def sig_star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else "ns"

print(f"\n[Model 1] 통제변수만")
print(f"  R² = {m1.rsquared:.4f},  Adj.R² = {m1.rsquared_adj:.4f}")

print(f"\n[Model 2] 주효과 (motivation + support)")
print(f"  R² = {m2.rsquared:.4f},  Adj.R² = {m2.rsquared_adj:.4f}")
print(f"  ΔR² (M2-M1) = {m2.rsquared - m1.rsquared:.4f}")
for var, lbl in [("motivation_c","motivation(X)"), ("support_c","support(W)")]:
    p = m2.pvalues[var]
    print(f"  {lbl:<18} B={m2.params[var]:+.4f}, SE={m2.bse[var]:.4f}, "
          f"t={m2.tvalues[var]:.3f}, p={p:.4f} {sig_star(p)}")

print(f"\n[Model 3] 조절효과 포함 (motivation × support)")
print(f"  R² = {m3.rsquared:.4f},  Adj.R² = {m3.rsquared_adj:.4f}")
dr2 = m3.rsquared - m2.rsquared
f2  = dr2 / (1 - m3.rsquared)
print(f"  ΔR² (M3-M2) = {dr2:.4f}  ← 조절효과 기여분산")
print(f"  f²  = {f2:.4f}  ({('소(0.02~0.15)' if 0.02<=f2<0.15 else '중(0.15~0.35)' if 0.15<=f2<0.35 else '대(≥0.35)' if f2>=0.35 else '무시(<0.02)')})")

for var, lbl in [("motivation_c","motivation(X)"), ("support_c","support(W)"),
                 ("inter_ms","motivation×support (조절)")]:
    p = m3.pvalues[var]
    print(f"  {lbl:<28} B={m3.params[var]:+.4f}, SE={m3.bse[var]:.4f}, "
          f"t={m3.tvalues[var]:.3f}, p={p:.4f} {sig_star(p)}")

print("\n  *** p<.001, ** p<.01, * p<.05, † p<.10, ns p≥.10")

# ============================================================
# 3. 단순기울기 분석
# ============================================================

print("\n\n" + "="*60)
print("단순기울기 분석 (Simple Slope Analysis)")
print("  motivation → effect at each level of support")
print("="*60)

b_mot   = m3.params["motivation_c"]
b_int   = m3.params["inter_ms"]
cov_m3  = m3.cov_params()

levels = {"高(+1SD)": support_sd, "中(평균)": 0.0, "低(-1SD)": -support_sd}

print(f"\n  support: M = {support_mean:.3f}, SD = {support_sd:.3f}")
print(f"\n  {'수준':<12} {'B':>8} {'SE':>8} {'t':>7} {'p':>8} {'유의'}")
print("  " + "-"*52)

for label, w_val in levels.items():
    slope    = b_mot + b_int * w_val
    var_s    = (cov_m3.loc["motivation_c","motivation_c"]
                + w_val**2 * cov_m3.loc["inter_ms","inter_ms"]
                + 2 * w_val * cov_m3.loc["motivation_c","inter_ms"])
    se_s     = np.sqrt(var_s)
    t_s      = slope / se_s
    p_s      = 2 * (1 - stats.t.cdf(abs(t_s), df=m3.df_resid))
    print(f"  {label:<12} {slope:>+8.4f} {se_s:>8.4f} {t_s:>7.3f} {p_s:>8.4f}  {sig_star(p_s)}")

# ============================================================
# 4. Johnson-Neyman 구간
# ============================================================

print("\n\n" + "="*60)
print("Johnson-Neyman 구간")
print("  (어떤 support 수준에서 motivation→effect 효과가 전환되는지)")
print("="*60)

v_a    = cov_m3.loc["motivation_c","motivation_c"]
v_b    = cov_m3.loc["inter_ms","inter_ms"]
cov_ab = cov_m3.loc["motivation_c","inter_ms"]
t_crit = stats.t.ppf(0.975, df=m3.df_resid)

A_c = b_int**2 - t_crit**2 * v_b
B_c = 2*(b_mot*b_int - t_crit**2 * cov_ab)
C_c = b_mot**2 - t_crit**2 * v_a
disc = B_c**2 - 4*A_c*C_c

s_min, s_max = df["support"].min(), df["support"].max()
print(f"\n  support 실제 범위: {s_min:.2f} ~ {s_max:.2f}")

if disc < 0:
    print("  JN 구간 없음 (전 범위에서 유의 상태 동일)")
elif abs(A_c) < 1e-10:
    wo = -C_c / B_c + support_mean
    print(f"  JN 경계: {wo:.4f}")
else:
    for wc, lab in [
        ((-B_c - np.sqrt(disc))/(2*A_c), "경계 1"),
        ((-B_c + np.sqrt(disc))/(2*A_c), "경계 2")
    ]:
        wo = wc + support_mean
        in_r = s_min <= wo <= s_max
        print(f"  JN {lab}: support = {wo:.4f}  {'← 측정 범위 내 ✓' if in_r else '(범위 밖)'}")

# ============================================================
# 5. Bootstrap 조절된 매개효과 (PROCESS Model 7)
#    경로: motivation(X) → [support 조절] → effect(M) → expectation(Y)
#    조건부 간접효과: support 수준별 (a×b)
# ============================================================

print("\n\n" + "="*60)
print("Bootstrap 조절된 매개효과 (PROCESS Model 7, 5,000회)")
print("  motivation→[support 조절]→effect→expectation")
print("="*60)

N_BOOT = 5000
support_vals = {
    "低(-1SD)": support_mean - support_sd,
    "中(평균)":  support_mean,
    "高(+1SD)": support_mean + support_sd,
}
boot_cond = {k: [] for k in support_vals}

for _ in trange(N_BOOT, desc="Bootstrap (Model 7)"):
    samp = df.sample(len(df), replace=True)
    samp["motivation_c"] = samp["motivation"] - samp["motivation"].mean()
    samp["support_c"]    = samp["support"]    - samp["support"].mean()
    samp["inter_ms"]     = samp["motivation_c"] * samp["support_c"]

    try:
        # 경로 a (조절): motivation → effect (support 조절)
        ma = smf.ols("effect ~ motivation_c + support_c + inter_ms", data=samp).fit()
        a_b   = ma.params["motivation_c"]
        a_int = ma.params["inter_ms"]

        # 경로 b: effect → expectation
        mb = smf.ols("expectation ~ effect + motivation_c + support_c", data=samp).fit()
        b_b = mb.params["effect"]

        # 조건부 간접효과: [a + a_int×W] × b
        for label, w_orig in support_vals.items():
            w_c = w_orig - samp["support"].mean()
            cond_a = a_b + a_int * w_c
            boot_cond[label].append(cond_a * b_b)
    except Exception:
        continue

print("\n  [조건부 간접효과] support 수준별 95% Bootstrap CI")
print(f"  {'수준':<12} {'평균':>8} {'2.5%':>8} {'97.5%':>8}  {'유의'}")
print("  " + "-"*50)

for label in support_vals:
    arr = np.array(boot_cond[label])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    sig_str = "✓ 유의" if not (lo <= 0 <= hi) else "✗ 비유의"
    print(f"  {label:<12} {arr.mean():>8.4f} {lo:>8.4f} {hi:>8.4f}  {sig_str}")

# 조절된 매개지수 (Index of Moderated Mediation)
arr_hi = np.array(boot_cond["高(+1SD)"])
arr_lo = np.array(boot_cond["低(-1SD)"])
diff   = arr_hi - arr_lo
lo_d, hi_d = np.percentile(diff, [2.5, 97.5])
print(f"\n  [조절된 매개지수] 고-저 수준 간 간접효과 차이")
print(f"  평균 차이 = {diff.mean():.4f}, 95% CI = [{lo_d:.4f}, {hi_d:.4f}]")
imm_sig = not (lo_d <= 0 <= hi_d)
print(f"  → {'✓ 조절된 매개 유의 (CI에 0 미포함)' if imm_sig else '✗ 조절된 매개 비유의 (CI에 0 포함)'}")

# ============================================================
# 6. VIF
# ============================================================

print("\n\n" + "="*60)
print("다중공선성 진단 (VIF)")
print("="*60)

X_v = df[["motivation_c","support_c","inter_ms",
          "gender","rank_code","career_code"]].dropna()
X_vc = sm.add_constant(X_v)
vif_df = pd.DataFrame({
    "Variable": X_vc.columns,
    "VIF": [variance_inflation_factor(X_vc.values, i)
            for i in range(X_vc.shape[1])]
})
print(vif_df.to_string(index=False))

# ============================================================
# 7. 결과 요약
# ============================================================

b_inter = m3.params["inter_ms"]
p_inter = m3.pvalues["inter_ms"]

print(f"\n\n{'='*60}")
print("결과 요약")
print(f"{'='*60}")
print(f"""
[motivation × support → effect  조절효과]
  상호작용: B = {b_inter:+.4f}, p = {p_inter:.4f}  {sig_star(p_inter)}
  ΔR² = {dr2:.4f},  f² = {f2:.4f}

[조절 방향]
  조절계수 B = {b_inter:+.4f}
  → support 높을수록 motivation→effect 기울기가 {'증가 (정적 조절)' if b_inter > 0 else '감소 (부적 조절)'}

[조절된 매개지수 (Index of Moderated Mediation)]
  고-저 차이 95% CI = [{lo_d:.4f}, {hi_d:.4f}]
  → {'✓ 조절된 매개효과 유의: 가설 포함 권고' if imm_sig else '✗ 조절된 매개효과 비유의'}

[가설 채택 기준]
  조절효과(a경로): {'✓ p < .05 → 채택 가능' if p_inter < 0.05 else '조절된 매개효과(IMM): ✓' if imm_sig else '✗ 두 기준 모두 미충족'}
""")
print("="*60)
print("분석 완료")
print("="*60)
