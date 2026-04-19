import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from tqdm import trange
from scipy import stats
from scipy.stats import bootstrap
from statsmodels.stats.outliers_influence import variance_inflation_factor

from result_utils import markdown_output

# ============================================================
# 1. 데이터 로드
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

# ============================================================
# 2. 변수 생성
# ============================================================

df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"] = df[[f"Q7_{i}" for i in range(1,6)]].mean(axis=1)
df["support"] = df[[f"Q16_{i}" for i in range(1,7)]].mean(axis=1)  # Q16_7 제외: 개인 관심 문항, motivation과 중첩
df["expectation"] = df[[f"Q20_{i}" for i in range(2,5)]].mean(axis=1)  # Q20_1 제외: 업무효과와 개념 중첩

# ============================================================
# 2-1. 상관관계 분석
# ============================================================

with markdown_output("04_regression_analysis_ai_users.md") as result_path:
    print("# 04 AI 활용자 회귀 및 매개효과 분석\n")
    print(f"- AI 활용자 수: {len(df)}명\n")

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


    def indirect_effect_stat(sample_indices):
        sample = df.iloc[np.asarray(sample_indices, dtype=int)]
        a_coef = smf.ols(
            "effect ~ motivation + gender + rank_code + career_code",
            data=sample
        ).fit().params["motivation"]
        b_coef = smf.ols(
            "expectation ~ motivation + effect + support + gender + rank_code + career_code",
            data=sample
        ).fit().params["effect"]
        return a_coef * b_coef

    bca_result = bootstrap(
        (np.arange(len(df)),),
        indirect_effect_stat,
        vectorized=False,
        n_resamples=N_BOOT,
        method="BCa",
        confidence_level=0.95,
        random_state=42,
    )
    ci_lower = float(bca_result.confidence_interval.low)
    ci_upper = float(bca_result.confidence_interval.high)
    indirect_point = a * b  # 원본 표본 점 추정치

    print("\nBootstrapped Indirect Effect (BCa)")
    print(f"Point Estimate (a×b) = {indirect_point:.3f}")
    print(f"95% BCa CI = [{ci_lower:.3f}, {ci_upper:.3f}]")

    print("\n 매개모형 분석 완료")

# ============================================================
# 7. 강건성 검증 (Robustness Checks)
# ============================================================

    print("\n\n" + "="*55)
    print("강건성 검증 (Robustness Checks)")
    print("="*55)

# ------------------------------------------------------------
# 7-1. 역인과 모델 (Reverse Causality)
# 가설 방향: motivation → effect → expectation
# 역인과 방향: expectation → effect → motivation
# ------------------------------------------------------------

    print("\n----- 7-1. 역인과 모델 (Reverse Causality) -----")
    print("원래 모델: motivation → effect → expectation")
    print("역인과 모델: expectation → effect → motivation\n")

# 역인과 Model A: effect ~ expectation (매개변수를 종속변수로)
    rev_a = smf.ols(
        "effect ~ expectation + gender + rank_code + career_code",
        data=df
    ).fit(cov_type="HC3")

# 역인과 Model B: motivation ~ expectation + effect (독립변수를 종속변수로)
    rev_b = smf.ols(
        "motivation ~ expectation + effect + support + gender + rank_code + career_code",
        data=df
    ).fit(cov_type="HC3")

    print("[역인과 Model A] effect ~ expectation")
    print(f"  expectation → effect: B = {rev_a.params['expectation']:.3f}, "
          f"p = {rev_a.pvalues['expectation']:.4f}")
    print(f"  R² = {rev_a.rsquared:.3f}")

    print("\n[역인과 Model B] motivation ~ expectation + effect")
    print(f"  expectation → motivation: B = {rev_b.params['expectation']:.3f}, "
          f"p = {rev_b.pvalues['expectation']:.4f}")
    print(f"  effect → motivation:      B = {rev_b.params['effect']:.3f}, "
          f"p = {rev_b.pvalues['effect']:.4f}")
    print(f"  R² = {rev_b.rsquared:.3f}")

# 원래 모델과 비교
    print("\n[원래 모델과 비교]")
    print(f"  원래: motivation → expectation: B = {m2.params['motivation']:.3f}, "
          f"p = {m2.pvalues['motivation']:.4f},  R² = {m2.rsquared:.3f}")
    print(f"  역인과: expectation → motivation: B = {rev_b.params['expectation']:.3f}, "
          f"p = {rev_b.pvalues['expectation']:.4f},  R² = {rev_b.rsquared:.3f}")

# ------------------------------------------------------------
# 7-2. 통제변수 제외 모델
# ------------------------------------------------------------

    print("\n----- 7-2. 통제변수 제외 모델 -----")

    m_no_ctrl = smf.ols(
        "expectation ~ motivation + effect + support",
        data=df
    ).fit(cov_type="HC3")

    print(f"  motivation → expectation: B = {m_no_ctrl.params['motivation']:.3f}, "
          f"p = {m_no_ctrl.pvalues['motivation']:.4f}")
    print(f"  effect → expectation:     B = {m_no_ctrl.params['effect']:.3f}, "
          f"p = {m_no_ctrl.pvalues['effect']:.4f}")
    print(f"  support → expectation:    B = {m_no_ctrl.params['support']:.3f}, "
          f"p = {m_no_ctrl.pvalues['support']:.4f}")
    print(f"  R² = {m_no_ctrl.rsquared:.3f}")
    print(f"\n  [원래 Model 3 비교]")
    print(f"  motivation: B = {m3.params['motivation']:.3f}, p = {m3.pvalues['motivation']:.4f}")
    print(f"  effect:     B = {m3.params['effect']:.3f}, p = {m3.pvalues['effect']:.4f}")
    print(f"  support:    B = {m3.params['support']:.3f}, p = {m3.pvalues['support']:.4f}")
    print(f"  R² = {m3.rsquared:.3f}")

# ------------------------------------------------------------
# 7-3. 강건성 요약
# ------------------------------------------------------------

    print("\n\n" + "="*55)
    print("강건성 검증 요약")
    print("="*55)
    print("\n## 주요 해석\n")
    print(f"- 총효과 모형(M2)에서 `motivation`과 `support`가 유의하면 전략적 활용 기대의 핵심 선행요인으로 해석할 수 있다.")
    print(f"- 직접효과 모형(M3)에서 `effect`가 유의하고 `motivation`의 계수가 약해지면 업무효과 인식의 매개 가능성이 강화된다.")
    print(f"- Sobel 검정과 BCa 부트스트랩 신뢰구간이 모두 간접효과를 지지하면 매개효과 해석이 더 안정적이다.")
    print(f"- 통제변수 제외 및 역인과 비교 결과가 본모형과 크게 충돌하지 않으면 연구모형의 강건성이 확보된 것으로 볼 수 있다.")

print(f"완료: {result_path} 생성")
