import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import zscore
from tqdm import trange
import sys

from result_utils import markdown_output

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

# ── 변수 생성 ────────────────────────────────────────────
df["motivation"]   = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"]       = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
df["expectation"]  = df[[f"Q20_{i}" for i in range(2, 5)]].mean(axis=1)  # Q20_1 제외: 업무효과와 개념 중첩

# 강건성 비교: Q16_7 포함(full) vs 제외(ex, 본 분석 채택)
# Q16_7("I am interested in using generative AI for my work")은 개인 관심 문항으로
# 조직지원 구성개념 오염 및 motivation과의 중첩으로 인해 본 분석에서 제외함
df["support_full"] = df[[f"Q16_{i}" for i in range(1, 8)]].mean(axis=1)   # Q16_1~7 (강건성 비교용)
df["support_ex"]   = df[[f"Q16_{i}" for i in range(1, 7)]].mean(axis=1)   # Q16_1~6 (본 분석 채택)

df = df.dropna(subset=["motivation", "effect", "expectation",
                        "support_full", "support_ex",
                        "gender", "rank_code", "career_code"])

with markdown_output("13_robustness_q16_7.md") as result_path:
    print("# 13 Q16_7 제외 강건성 검증 결과\n")
    print(f"분석 표본 수: N = {len(df)}\n")

    for col in ["motivation", "effect", "expectation", "support_full", "support_ex"]:
        df[f"z_{col}"] = zscore(df[col], nan_policy="omit")

    def run_models(support_col):
        z_sup = f"z_{support_col}"
        m1 = smf.ols(
            "expectation ~ gender + rank_code + career_code",
            data=df,
        ).fit(cov_type="HC3")
        m2 = smf.ols(
            f"expectation ~ motivation + {support_col} + gender + rank_code + career_code",
            data=df,
        ).fit(cov_type="HC3")
        m3 = smf.ols(
            f"expectation ~ motivation + {support_col} + effect + gender + rank_code + career_code",
            data=df,
        ).fit(cov_type="HC3")
        ma_z = smf.ols(
            f"z_effect ~ z_motivation + {z_sup} + gender + rank_code + career_code",
            data=df,
        ).fit(cov_type="HC3")
        mt_z = smf.ols(
            f"z_expectation ~ z_motivation + {z_sup} + gender + rank_code + career_code",
            data=df,
        ).fit(cov_type="HC3")
        md_z = smf.ols(
            f"z_expectation ~ z_motivation + {z_sup} + z_effect + gender + rank_code + career_code",
            data=df,
        ).fit(cov_type="HC3")
        return m1, m2, m3, ma_z, mt_z, md_z

    def sig(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        if p < 0.1:
            return "†"
        return "ns"

    print("## 대안 모형 (POS = Q16_1 ~ Q16_7, Q16_7 포함 7문항)\n")
    m1f, m2f, m3f, ma_zf, mt_zf, md_zf = run_models("support_full")

    print("### 표준화 계수 비교표 (ß)\n")
    print("| 경로 | AI활용동기(motivation) | 조직지원인식(support) |")
    print("|---|---|---|")
    p_maf, p_saf = ma_zf.pvalues["z_motivation"], ma_zf.pvalues["z_support_full"]
    print(f"| Path A: → 매개변수 | ß={ma_zf.params['z_motivation']:.3f} {sig(p_maf)} | ß={ma_zf.params['z_support_full']:.3f} {sig(p_saf)} |")
    p_mtf, p_stf = mt_zf.pvalues["z_motivation"], mt_zf.pvalues["z_support_full"]
    print(f"| 총효과: → 종속변수 | ß={mt_zf.params['z_motivation']:.3f} {sig(p_mtf)} | ß={mt_zf.params['z_support_full']:.3f} {sig(p_stf)} |")
    p_mdf, p_sdf = md_zf.pvalues["z_motivation"], md_zf.pvalues["z_support_full"]
    print(f"| 직접효과: → 종속변수 | ß={md_zf.params['z_motivation']:.3f} {sig(p_mdf)} | ß={md_zf.params['z_support_full']:.3f} {sig(p_sdf)} |")
    print(f"\nR² Change: M1→M2 = {m2f.rsquared - m1f.rsquared:.3f}, M2→M3 = {m3f.rsquared - m2f.rsquared:.3f}\n")

    print("\n---\n")
    print("## 주모형 (POS = Q16_1 ~ Q16_6, Q16_7 제외 6문항)\n")
    m1e, m2e, m3e, ma_ze, mt_ze, md_ze = run_models("support_ex")

    print("### 표준화 계수 비교표 (ß)\n")
    print("| 경로 | AI활용동기(motivation) | 조직지원인식(support) |")
    print("|---|---|---|")
    p_mae, p_sae = ma_ze.pvalues["z_motivation"], ma_ze.pvalues["z_support_ex"]
    print(f"| Path A: → 매개변수 | ß={ma_ze.params['z_motivation']:.3f} {sig(p_mae)} | ß={ma_ze.params['z_support_ex']:.3f} {sig(p_sae)} |")
    p_mte, p_ste = mt_ze.pvalues["z_motivation"], mt_ze.pvalues["z_support_ex"]
    print(f"| 총효과: → 종속변수 | ß={mt_ze.params['z_motivation']:.3f} {sig(p_mte)} | ß={mt_ze.params['z_support_ex']:.3f} {sig(p_ste)} |")
    p_mde, p_sde = md_ze.pvalues["z_motivation"], md_ze.pvalues["z_support_ex"]
    print(f"| 직접효과: → 종속변수 | ß={md_ze.params['z_motivation']:.3f} {sig(p_mde)} | ß={md_ze.params['z_support_ex']:.3f} {sig(p_sde)} |")
    print(f"\nR² Change: M1→M2 = {m2e.rsquared - m1e.rsquared:.3f}, M2→M3 = {m3e.rsquared - m2e.rsquared:.3f}\n")

    print("\n---\n")
    print("## 핵심 계수 변화 비교 요약 (비표준화 계수)\n")
    print("| 계수 | 대안 모형: Q16_7 포함 (7문항) | 주모형: Q16_7 제외 (6문항) | 변화 |")
    print("|---|---|---|---|")

    ma_full = smf.ols("effect ~ motivation + support_full + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    ma_excl = smf.ols("effect ~ motivation + support_ex + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    md_full = smf.ols("expectation ~ motivation + support_full + effect + gender + rank_code + career_code", data=df).fit(cov_type="HC3")
    md_excl = smf.ols("expectation ~ motivation + support_ex + effect + gender + rank_code + career_code", data=df).fit(cov_type="HC3")

    for label, coef_full, coef_excl, model_full, model_excl in [
        ("motivation → effect", "motivation", "motivation", ma_full, ma_excl),
        ("support → effect", "support_full", "support_ex", ma_full, ma_excl),
        ("motivation → expectation", "motivation", "motivation", md_full, md_excl),
        ("support → expectation", "support_full", "support_ex", md_full, md_excl),
    ]:
        bf = model_full.params[coef_full]
        pf = model_full.pvalues[coef_full]
        be = model_excl.params[coef_excl]
        pe = model_excl.pvalues[coef_excl]
        diff = be - bf
        print(f"| {label} | B={bf:.3f} {sig(pf)} | B={be:.3f} {sig(pe)} | Δ={diff:+.3f} |")

    print("\n> 핵심 추정치의 방향성과 유의성이 두 모형에서 실질적으로 동일하면 강건성이 확보된 것으로 판단합니다.\n")
    print("> († p<.10, * p<.05, ** p<.01, *** p<.001)")
    print("\n## 주요 해석\n")
    print("- Q16_7 포함 여부에 따라 핵심 회귀계수의 방향과 유의성이 거의 유지되면 본 연구 결과는 문항 선택에 과도하게 의존하지 않는다고 볼 수 있다.")
    print("- 반대로 Q16_7을 제외했을 때 조직지원 계수가 더 안정되거나 개념적 중복이 줄어들면 제외 버전이 이론적으로도 더 설득력 있다.")

print(f"완료: {result_path} 생성")
