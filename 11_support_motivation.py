"""
support → motivation 경로 검증
=============================================================
목적: 조직지원 인식(support)이 자발적 AI 활용 동기(motivation)의
      선행변수(antecedent)로 기능하는지 확인

     support → motivation → effect → expectation
     (선행-매개-종속 직렬 구조 탐색)
=============================================================
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

df["motivation"]  = df[["Q9_3", "Q9_4"]].mean(axis=1)
df["effect"]      = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
df["support"]     = df[[f"Q16_{i}" for i in range(1, 7)]].mean(axis=1)  # Q16_7 제외: 개인 관심 문항, motivation과 중첩
df["expectation"] = df[[f"Q20_{i}" for i in range(2, 5)]].mean(axis=1)  # Q20_1 제외: 업무효과와 개념 중첩

print("="*55)
print("support → motivation 단순/다중 회귀")
print("="*55)

# ── 모형 1: support → motivation (통제 없이)
m_simple = smf.ols(
    "motivation ~ support",
    data=df
).fit(cov_type="HC3")

# ── 모형 2: support → motivation (통제 포함)
m_full = smf.ols(
    "motivation ~ support + gender + rank_code + career_code",
    data=df
).fit(cov_type="HC3")

def star(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "ns"

print(f"\n[단순 회귀]  motivation ~ support")
p1 = m_simple.pvalues["support"]
print(f"  support: B={m_simple.params['support']:+.4f}, "
      f"SE={m_simple.bse['support']:.4f}, "
      f"t={m_simple.tvalues['support']:.3f}, "
      f"p={p1:.4f}  {star(p1)}")
print(f"  R² = {m_simple.rsquared:.4f}")

print(f"\n[다중 회귀]  motivation ~ support + 통제변수")
p2 = m_full.pvalues["support"]
print(f"  support: B={m_full.params['support']:+.4f}, "
      f"SE={m_full.bse['support']:.4f}, "
      f"t={m_full.tvalues['support']:.3f}, "
      f"p={p2:.4f}  {star(p2)}")
print(f"  R² = {m_full.rsquared:.4f}")

# ── Pearson 상관
r, p_r = stats.pearsonr(df["support"], df["motivation"])
print(f"\n[Pearson 상관]  r = {r:.4f}, p = {p_r:.4f}  {star(p_r)}")

print("\n" + "="*55)
print("결과 해석")
print("="*55)
if p2 < 0.05:
    print(f"""
  ✓ support → motivation 경로 유의 (p={p2:.4f})
    B = {m_full.params['support']:+.4f}

  → 조직지원 인식은 자발적 AI 활용 동기의 선행변수로 기능합니다.
  → 모형 확장 가능:
     support → motivation → effect → expectation
     (직렬 매개 또는 선행변수 포함 순차 모형)

  ⚑ 논문 활용:
    "Perceived Organizational Support was also found to
     significantly predict voluntary AI motivation (B={m_full.params['support']:+.4f},
     p={p2:.4f}), suggesting that organizational context
     may serve as an antecedent to individual motivation,
     consistent with Social Exchange Theory."
""")
else:
    print(f"""
  ✗ support → motivation 경로 비유의 (p={p2:.4f})

  → 조직지원 인식은 자발적 동기의 선행변수로 보기 어렵습니다.
  → 원래 모형(support를 독립적 예측변수로 처리)이 가장 적절합니다.
  → 리뷰어 코멘트에 대한 방어 논거:
    "Organizational support did not predict voluntary motivation
     (p={p2:.4f}), confirming that POS operates as an independent
     predictor of strategic expectations rather than as an
     antecedent to individual motivation."
""")

print("="*55)
print("분석 완료")
print("="*55)
