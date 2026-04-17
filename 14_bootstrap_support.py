import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import bootstrap
from tqdm import trange

sys.stdout = open("bootstrap_support_indirect.md", "w", encoding="utf-8")

print("# 조직지원 인식의 간접효과 Bootstrap 분석\n")

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"] = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
df["support"] = df[[f"Q16_{i}" for i in range(1, 7)]].mean(axis=1)
df["expectation"] = df[[f"Q20_{i}" for i in range(2, 5)]].mean(axis=1)
df = df.dropna(subset=["motivation", "effect", "support", "expectation", "gender", "rank_code", "career_code"])

model_a_sup = smf.ols(
    "effect ~ motivation + support + gender + rank_code + career_code",
    data=df,
).fit(cov_type="HC3")
model_b = smf.ols(
    "expectation ~ motivation + support + effect + gender + rank_code + career_code",
    data=df,
).fit(cov_type="HC3")

a_sup = model_a_sup.params["support"]
sa_sup = model_a_sup.bse["support"]
b = model_b.params["effect"]
sb = model_b.bse["effect"]

sobel_sup = (a_sup * b) / np.sqrt(b**2 * sa_sup**2 + a_sup**2 * sb**2)
p_sobel_sup = 2 * (1 - stats.norm.cdf(abs(sobel_sup)))

print("## Sobel 검정 (조직지원 인식)\n")
print(f"- a (support -> effect): {a_sup:.4f}")
print(f"- b (effect -> expectation): {b:.4f}")
print(f"- Sobel Z = {sobel_sup:.4f}, p = {p_sobel_sup:.4f}\n")

N_BOOT = 5000
print("Bootstrap (조직지원) 진행 중...", file=sys.__stdout__)


def support_indirect_stat(sample_indices):
    sample = df.iloc[np.asarray(sample_indices, dtype=int)]
    a_coef = smf.ols(
        "effect ~ motivation + support + gender + rank_code + career_code",
        data=sample,
    ).fit().params["support"]
    b_coef = smf.ols(
        "expectation ~ motivation + support + effect + gender + rank_code + career_code",
        data=sample,
    ).fit().params["effect"]
    return a_coef * b_coef


bca_sup = bootstrap(
    (np.arange(len(df)),),
    support_indirect_stat,
    vectorized=False,
    n_resamples=N_BOOT,
    method="BCa",
    confidence_level=0.95,
    random_state=42,
)
ci_lo = float(bca_sup.confidence_interval.low)
ci_hi = float(bca_sup.confidence_interval.high)
indirect_sup_point = a_sup * b

print(f"## Bootstrap 결과 (조직지원 인식 간접효과, N={N_BOOT})\n")
print(f"- Point Estimate (a×b): {indirect_sup_point:.4f}")
print(f"- 95% BCa CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

if ci_lo <= 0 <= ci_hi:
    print("- 해석: 신뢰구간에 0이 포함되므로 간접효과는 통계적으로 유의하지 않음")
else:
    print("- 해석: 신뢰구간에 0이 포함되지 않으므로 간접효과는 통계적으로 유의함")

# ============================================================
# AI 활용동기 간접효과 (동적 계산 — 비교표용)
# ============================================================

print("\nBootstrap (AI 활용동기) 진행 중...", file=sys.__stdout__)

model_a_mot = smf.ols(
    "effect ~ motivation + gender + rank_code + career_code", data=df
).fit(cov_type="HC3")
a_mot = model_a_mot.params["motivation"]
sa_mot = model_a_mot.bse["motivation"]
b_mot = model_b.params["effect"]
sb_mot = model_b.bse["effect"]
sobel_mot = (a_mot * b_mot) / np.sqrt(b_mot**2 * sa_mot**2 + a_mot**2 * sb_mot**2)
p_sobel_mot = 2 * (1 - stats.norm.cdf(abs(sobel_mot)))
indirect_mot_point = a_mot * b_mot


def mot_indirect_stat(sample_indices):
    sample = df.iloc[np.asarray(sample_indices, dtype=int)]
    a_coef = smf.ols(
        "effect ~ motivation + gender + rank_code + career_code",
        data=sample,
    ).fit().params["motivation"]
    b_coef = smf.ols(
        "expectation ~ motivation + support + effect + gender + rank_code + career_code",
        data=sample,
    ).fit().params["effect"]
    return a_coef * b_coef


bca_mot = bootstrap(
    (np.arange(len(df)),),
    mot_indirect_stat,
    vectorized=False,
    n_resamples=N_BOOT,
    method="BCa",
    confidence_level=0.95,
    random_state=42,
)
ci_mot_lo = float(bca_mot.confidence_interval.low)
ci_mot_hi = float(bca_mot.confidence_interval.high)

print("\n---\n")
print("## 종합 비교\n")
print("| 항목 | AI 활용동기 | 조직지원 인식 |")
print("| --- | --- | --- |")
print(f"| Sobel Z | {sobel_mot:.3f}*** | {sobel_sup:.3f} (p={p_sobel_sup:.4f}) |")
print(f"| 간접효과 (a×b) | {indirect_mot_point:.3f} | {indirect_sup_point:.3f} |")
print(f"| 95% BCa CI | [{ci_mot_lo:.3f}, {ci_mot_hi:.3f}] | [{ci_lo:.3f}, {ci_hi:.3f}] |")
print(f"| 신뢰구간 내 0 포함 | {'아니오' if not (ci_mot_lo <= 0 <= ci_mot_hi) else '예'} | {'아니오' if not (ci_lo <= 0 <= ci_hi) else '예'} |")

sys.stdout.close()
print("완료: bootstrap_support_indirect.md 생성", file=sys.__stdout__)
