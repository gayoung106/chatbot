import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import bootstrap
from tqdm import trange

from result_utils import markdown_output

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
b = model_b.params["effect"]

N_BOOT = 5000


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

# ============================================================
# AI 활용동기 간접효과 (동적 계산 — 비교표용)
# ============================================================

model_a_mot = smf.ols(
    "effect ~ motivation + gender + rank_code + career_code", data=df
).fit(cov_type="HC3")
a_mot = model_a_mot.params["motivation"]
b_mot = model_b.params["effect"]
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

with markdown_output("14_bootstrap_support.md") as result_path:
    print("# 14 조직지원 인식의 간접효과 Bootstrap 분석\n")
    print(f"- 분석 표본: AI 활용자 {len(df)}명")
    print(f"- Bootstrap resamples: {N_BOOT}\n")
    print(f"## Bootstrap 결과 (조직지원 인식 간접효과, N={N_BOOT})\n")
    print(f"- Path a (support -> effect): {a_sup:.4f}")
    print(f"- Path b (effect -> expectation): {b:.4f}")
    print(f"- Point Estimate (a×b): {indirect_sup_point:.4f}")
    print(f"- 95% BCa CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    if ci_lo <= 0 <= ci_hi:
        print("- 해석: 신뢰구간에 0이 포함되므로 간접효과는 통계적으로 유의하지 않음")
    else:
        print("- 해석: 신뢰구간에 0이 포함되지 않으므로 간접효과는 통계적으로 유의함")

    print("\n---\n")
    print("## 민감도 점검 메모\n")
    print("- 아래 motivation 간접효과 값은 `support`를 path-A 식에서 제외한 단일 IV 민감도 점검치이다.")
    print("- 따라서 병렬매개모형의 정식 보고값과 직접 비교해서는 안 되며, 본문 표와 비교표에는 Script 19의 병렬매개 추정치를 사용해야 한다.\n")
    print("| 항목 | motivation 단일-IV 민감도 점검 |")
    print("| --- | --- |")
    print(f"| 간접효과 (a×b) | {indirect_mot_point:.3f} |")
    print(f"| 95% BCa CI | [{ci_mot_lo:.3f}, {ci_mot_hi:.3f}] |")
    print(f"| 신뢰구간 내 0 포함 | {'아니오' if not (ci_mot_lo <= 0 <= ci_mot_hi) else '예'} |")
    print("\n## 주요 해석\n")
    print("- 본 스크립트는 조직지원 인식이 업무효과 인식을 경유해 전략적 활용 기대에 간접적으로 영향을 주는지 점검한다.")
    print("- BCa 신뢰구간에 0이 포함되지 않으면 조직지원 인식의 간접효과가 통계적으로 유의하다고 해석할 수 있다.")
    print("- motivation 간접효과의 정식 비교값은 병렬매개모형을 사용한 Script 19의 추정치(0.1269, 95% BCa CI [0.0812, 0.1906])를 기준으로 제시해야 한다.")

print(f"완료: {result_path} 생성")
