"""
CFA (Confirmatory Factor Analysis) + 전략기대 하위차원 EFA
=============================================================
1번: 네 구성 개념에 대한 CFA → CFI, RMSEA, SRMR 보고
5번: Q20 2요인 구조 EFA → 하위차원 실증 확인

실행 방법:
    python cfa_and_subdimension_analysis.py

필요 패키지: pandas, numpy, scipy (기본 설치 확인됨)
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

# ============================================================
# 0. 데이터 로드
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df_ai = df[df["Q3"] == 1].copy()
print(f"AI 활용자 수: {len(df_ai)}명\n")

# ============================================================
# 문항 정의
# ============================================================

cols_motivation = ["Q9_3", "Q9_4"]
cols_effect     = [f"Q7_{i}" for i in range(1, 6)]
cols_support    = [f"Q16_{i}" for i in range(1, 8)]
cols_strategic  = [f"Q20_{i}" for i in range(1, 5)]

construct_dict = {
    "Voluntary Motivation (motivation)":     cols_motivation,
    "Perceived Work Effectiveness (effect)": cols_effect,
    "Perceived Org. Support (support)":      cols_support,
    "Strategic Utilization Expectation (strategic)": cols_strategic,
}

# ============================================================
# CFA 핵심 함수 (단일 요인 모델)
# 추정 방식: ULS (Unweighted Least Squares) — 정규성 가정 불필요
# ============================================================

def single_factor_cfa(data_cols, label, df_sub):
    """
    단일 요인 CFA 수행 및 적합도 지수 계산
    반환: dict(loadings, CFI, RMSEA, SRMR, chi2, df, p)
    """
    data = df_sub[data_cols].dropna().values
    n, p = data.shape

    # 표본 공분산행렬 (S)
    S = np.cov(data, rowvar=False)

    # ── 초기값: PCA 첫 번째 성분 ──
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argmax(eigvals)
    lam_init = eigvecs[:, idx] * np.sqrt(max(eigvals[idx], 0.01))
    err_init  = np.diag(S) - lam_init**2
    err_init  = np.maximum(err_init, 0.01)
    x0 = np.concatenate([lam_init, err_init])

    # ── 목적함수: ULS ──
    def uls_objective(x):
        lam = x[:p]
        err = np.abs(x[p:])           # 오차분산 양수 제약
        Sigma = np.outer(lam, lam) + np.diag(err)
        diff  = S - Sigma
        return 0.5 * np.sum(diff**2)

    result = minimize(
        uls_objective, x0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12}
    )

    lam_hat = result.x[:p]
    err_hat = np.abs(result.x[p:])
    Sigma_hat = np.outer(lam_hat, lam_hat) + np.diag(err_hat)

    # ── 표준화 적재량 ──
    std_loadings = lam_hat / np.sqrt(np.diag(Sigma_hat))

    # ── 카이제곱 (ML 근사) ──
    try:
        sign, logdet_S     = np.linalg.slogdet(S)
        sign2, logdet_Sig  = np.linalg.slogdet(Sigma_hat)
        Sig_inv = np.linalg.inv(Sigma_hat)
        ml_val  = (logdet_Sig - logdet_S + np.trace(Sig_inv @ S) - p)
        chi2_val = (n - 1) * max(ml_val, 0)
    except Exception:
        chi2_val = np.nan

    df_model = p * (p - 1) / 2 - p          # 자유도: 단일요인 모델
    df_model = max(int(df_model), 1)
    p_val    = 1 - chi2.cdf(chi2_val, df_model) if not np.isnan(chi2_val) else np.nan

    # ── CFI ──
    # 기저모델(독립모델): 요인 없이 분산만
    Sigma_null = np.diag(np.diag(S))
    try:
        sign3, logdet_null = np.linalg.slogdet(Sigma_null)
        null_inv = np.linalg.inv(Sigma_null)
        ml_null  = (logdet_null - logdet_S + np.trace(null_inv @ S) - p)
        chi2_null = (n - 1) * max(ml_null, 0)
    except Exception:
        chi2_null = np.nan

    df_null = p * (p - 1) / 2
    if not np.isnan(chi2_val) and not np.isnan(chi2_null) and chi2_null > 0:
        cfi = 1 - max(chi2_val - df_model, 0) / max(chi2_null - df_null, 1e-10)
        cfi = min(max(cfi, 0.0), 1.0)
    else:
        cfi = np.nan

    # ── RMSEA ──
    if not np.isnan(chi2_val) and df_model > 0:
        rmsea = np.sqrt(max((chi2_val - df_model) / (df_model * (n - 1)), 0))
    else:
        rmsea = np.nan

    # ── SRMR ──
    diag_S   = np.sqrt(np.diag(S))
    R_S      = S / np.outer(diag_S, diag_S)
    diag_Sig = np.sqrt(np.diag(Sigma_hat))
    R_Sig    = Sigma_hat / np.outer(diag_Sig, diag_Sig)
    lower_idx = np.tril_indices(p, k=-1)
    srmr = np.sqrt(np.mean((R_S[lower_idx] - R_Sig[lower_idx])**2))

    # ── 출력 ──
    print(f"\n{'='*55}")
    print(f"CFA: {label}")
    print(f"{'='*55}")
    print(f"문항 수: {p},  표본 수: {n},  자유도: {df_model}")
    print(f"\n표준화 요인 적재량:")
    for col, ld in zip(data_cols, std_loadings):
        print(f"  {col}: {ld:.3f}")
    print(f"\n적합도 지수:")
    print(f"  χ²({df_model}) = {chi2_val:.3f},  p = {p_val:.4f}" if not np.isnan(chi2_val) else "  χ²: 계산 불가")
    print(f"  CFI   = {cfi:.3f}"   if not np.isnan(cfi)   else "  CFI: 계산 불가")
    print(f"  RMSEA = {rmsea:.3f}" if not np.isnan(rmsea) else "  RMSEA: 계산 불가")
    print(f"  SRMR  = {srmr:.3f}")

    # 해석 기준 안내
    flags = []
    if not np.isnan(cfi)   and cfi   >= 0.90: flags.append("CFI ✓ (≥.90)")
    if not np.isnan(rmsea) and rmsea <= 0.08: flags.append("RMSEA ✓ (≤.08)")
    if srmr <= 0.08: flags.append("SRMR ✓ (≤.08)")
    print(f"  기준 충족: {', '.join(flags) if flags else '없음 → 모델 수정 검토 필요'}")

    return {
        "label": label,
        "n_items": p,
        "n": n,
        "df": df_model,
        "chi2": round(chi2_val, 3) if not np.isnan(chi2_val) else None,
        "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
        "CFI": round(cfi, 3) if not np.isnan(cfi) else None,
        "RMSEA": round(rmsea, 3) if not np.isnan(rmsea) else None,
        "SRMR": round(srmr, 3),
        "std_loadings": dict(zip(data_cols, np.round(std_loadings, 3)))
    }


# ============================================================
# 1번: 네 구성 개념 CFA 실행
# ============================================================

print("\n" + "#"*55)
print("  1번: CFA — 네 구성 개념 단일요인 모델")
print("#"*55)

cfa_results = []
for label, cols in construct_dict.items():
    res = single_factor_cfa(cols, label, df_ai)
    cfa_results.append(res)

# 요약 테이블
print("\n\n" + "="*55)
print("CFA 적합도 요약 테이블")
print("="*55)
summary_rows = []
for r in cfa_results:
    summary_rows.append({
        "Construct": r["label"],
        "Items": r["n_items"],
        "χ²(df)": f"{r['chi2']}({r['df']})" if r["chi2"] else "N/A",
        "CFI": r["CFI"],
        "RMSEA": r["RMSEA"],
        "SRMR": r["SRMR"],
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# ============================================================
# 5번: Q20 2요인 EFA — 전략기대 하위차원 구조 확인
# ============================================================

print("\n\n" + "#"*55)
print("  5번: 전략기대(Q20) 2요인 EFA — 하위차원 구조")
print("#"*55)

from scipy.linalg import svd

def varimax_rotation(loadings, tol=1e-6, max_iter=1000):
    """Varimax 회전 (직교)"""
    p, k = loadings.shape
    rotation = np.eye(k)
    for _ in range(max_iter):
        old = rotation.copy()
        for i in range(k):
            for j in range(i+1, k):
                x = loadings @ rotation
                u = x[:, i]**2 - x[:, j]**2
                v = 2 * x[:, i] * x[:, j]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)
                num   = D - 2*A*B/p
                denom = C - (A**2 - B**2)/p
                theta = 0.25 * np.arctan2(num, denom)
                c, s  = np.cos(theta), np.sin(theta)
                rot   = np.eye(k)
                rot[i, i] =  c; rot[i, j] = s
                rot[j, i] = -s; rot[j, j] = c
                rotation = rotation @ rot
        if np.max(np.abs(rotation - old)) < tol:
            break
    return loadings @ rotation


def run_efa_2factor(data_cols, label, df_sub):
    data = df_sub[data_cols].dropna().values
    n, p = data.shape
    R = np.corrmat = np.corrcoef(data, rowvar=False)

    # 고유값 분해
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    print(f"\n{'='*55}")
    print(f"EFA 2요인: {label}")
    print(f"{'='*55}")
    print(f"고유값 (Eigenvalues): {np.round(eigvals, 3)}")
    print(f"고유값 >1 기준 요인 수: {np.sum(eigvals > 1)}")

    # 2요인 적재량 행렬 (초기)
    k = 2
    L_init = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0))

    # Varimax 회전
    L_rotated = varimax_rotation(L_init)

    loadings_df = pd.DataFrame(
        L_rotated,
        index=data_cols,
        columns=["Factor1 (지원·확산 기대)", "Factor2 (대체·위임 기대)"]
    )

    print(f"\n회전된 요인 적재량 (Varimax):")
    print(loadings_df.round(3))

    # 공통분산 (communality)
    communality = np.sum(L_rotated**2, axis=1)
    print(f"\n공통분산 (Communality):")
    for col, h2 in zip(data_cols, communality):
        print(f"  {col}: {h2:.3f}")

    # 설명분산
    var_explained = np.sum(L_rotated**2, axis=0) / p * 100
    print(f"\n요인별 설명분산 (%):")
    print(f"  Factor1: {var_explained[0]:.1f}%")
    print(f"  Factor2: {var_explained[1]:.1f}%")
    print(f"  합계:    {sum(var_explained):.1f}%")

    # 하위 척도 신뢰도 (Cronbach's α 근사)
    def cronbach_alpha_simple(cols_sub):
        d = df_sub[cols_sub].dropna()
        k_sub = len(cols_sub)
        item_var = d.var(axis=0, ddof=1).sum()
        total_var = d.sum(axis=1).var(ddof=1)
        return (k_sub / (k_sub - 1)) * (1 - item_var / total_var)

    # Factor1: Q20_1, Q20_2 / Factor2: Q20_3, Q20_4 (이론 기반 분류)
    alpha_f1 = cronbach_alpha_simple(["Q20_1", "Q20_2"])
    alpha_f2 = cronbach_alpha_simple(["Q20_3", "Q20_4"])

    print(f"\n하위 척도 신뢰도 (Cronbach's α):")
    print(f"  Factor1 (Q20_1, Q20_2 — 지원·확산 기대): α = {alpha_f1:.3f}")
    print(f"  Factor2 (Q20_3, Q20_4 — 대체·위임 기대): α = {alpha_f2:.3f}")

    return loadings_df, var_explained, alpha_f1, alpha_f2


efa_loadings, var_exp, a1, a2 = run_efa_2factor(cols_strategic, "전략기대(Q20)", df_ai)

# ============================================================
# 최종 해석 가이드 출력
# ============================================================

print("\n\n" + "#"*55)
print("  논문 작성 가이드")
print("#"*55)

print("""
[1번 CFA 결과 활용]
- CFI ≥ .90, RMSEA ≤ .08, SRMR ≤ .08 이면 단일요인 구조 지지
- 결과를 Table로 추가하고 본문에 아래 문장 삽입:
  "To supplement the EFA results, single-factor CFA was conducted
   for each construct. Fit indices (CFI, RMSEA, SRMR) indicated
   acceptable model fit, supporting the unidimensional structure
   of each construct (see Table X)."

[5번 하위차원 EFA 결과 활용]
- 2요인 구조가 확인되면 Limitations 절 보강:
  "EFA results confirmed a two-factor structure within the
   strategic utilization expectation scale, distinguishing
   between 'supportive diffusion expectations' (Q20_1, Q20_2;
   α = .XXX) and 'delegation/replacement expectations'
   (Q20_3, Q20_4; α = .XXX), supporting the theoretical
   interpretation of the construct's heterogeneity."
""")

print("분석 완료.")