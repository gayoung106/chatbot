import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from scipy import stats
from tqdm import trange
import sys

sys.stdout = open('bootstrap_support_indirect.md', 'w', encoding='utf-8')

print("# 조직지원 인식 간접효과 Bootstrap 분석\n")

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"]     = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
df["support"]    = df[[f"Q16_{i}" for i in range(1, 8)]].mean(axis=1)
df["expectation"]= df[[f"Q20_{i}" for i in range(1, 5)]].mean(axis=1)
df = df.dropna(subset=["motivation", "effect", "support", "expectation",
                        "gender", "rank_code", "career_code"])

# ── Sobel for support ───────────────────────────────────
model_a_sup = smf.ols(
    "effect ~ motivation + support + gender + rank_code + career_code",
    data=df).fit(cov_type="HC3")
model_b = smf.ols(
    "expectation ~ motivation + support + effect + gender + rank_code + career_code",
    data=df).fit(cov_type="HC3")

a_sup  = model_a_sup.params["support"]
sa_sup = model_a_sup.bse["support"]
b      = model_b.params["effect"]
sb     = model_b.bse["effect"]

sobel_sup = (a_sup * b) / np.sqrt(b**2 * sa_sup**2 + a_sup**2 * sb**2)
p_sobel_sup = 2 * (1 - stats.norm.cdf(abs(sobel_sup)))

print(f"## Sobel 검정 (조직지원 인식)\n")
print(f"- a (support → effect): {a_sup:.4f}")
print(f"- b (effect → expectation): {b:.4f}")
print(f"- Sobel Z = {sobel_sup:.4f}, p = {p_sobel_sup:.4f}\n")

# ── Bootstrap for support ───────────────────────────────
N_BOOT = 5000
indirect_sup = []
print("Bootstrap 진행 중...", file=sys.__stdout__)

for _ in trange(N_BOOT, file=sys.__stdout__):
    s = df.sample(len(df), replace=True)
    a = smf.ols("effect ~ motivation + support", data=s).fit().params["support"]
    b = smf.ols("expectation ~ motivation + support + effect", data=s).fit().params["effect"]
    indirect_sup.append(a * b)

arr = np.array(indirect_sup)
ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])

print(f"## Bootstrap 결과 (조직지원 인식 간접효과, N={N_BOOT})\n")
print(f"- Indirect Effect Mean: {arr.mean():.4f}")
print(f"- 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

if ci_lo <= 0 <= ci_hi:
    print(f"- **해석**: 신뢰구간이 0을 포함하므로 간접효과는 통계적으로 **유의하지 않음** (p > .05)")
else:
    print(f"- **해석**: 신뢰구간이 0을 포함하지 않으므로 간접효과 **유의함** (p < .05)")

print(f"\n---\n")
print(f"## 종합 비교 (표 7 업데이트용)\n")
print(f"| 항목 | AI활용동기 | 조직지원인식 |")
print(f"|---|---|---|")
print(f"| Sobel Z | 4.744*** | {sobel_sup:.3f} (p={p_sobel_sup:.4f}) |")
print(f"| 간접효과 평균 | 0.149 | {arr.mean():.3f} |")
print(f"| 95% CI | [0.097, 0.206] | [{ci_lo:.3f}, {ci_hi:.3f}] |")
print(f"| 신뢰구간 내 0 포함 | 아니요 (유의) | {'아니요 (유의)' if not (ci_lo <= 0 <= ci_hi) else '예 (비유의)'} |")
print(f"| 매개효과 | 부분매개 확인 | {'매개 확인' if not (ci_lo <= 0 <= ci_hi) else '매개 미확인'} |")
