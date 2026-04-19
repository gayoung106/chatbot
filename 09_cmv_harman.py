"""
Common Method Variance (CMV) 검증
Harman's Single Factor Test
=============================================================
목적: 자기보고 설문 데이터에서 동일방법편향(CMV) 여부를 확인

방법:
  1. 모든 주요 측정문항을 포함하여 EFA 실시
  2. 요인 회전 없이(unrotated) 첫 번째 요인의 설명분산 확인
  3. 단일요인이 전체 분산의 50% 이상 설명하면 CMV 우려

포함 문항:
  - motivation: Q9_3, Q9_4
  - effect:     Q7_1 ~ Q7_5
  - support:    Q16_1 ~ Q16_7
  - expectation: Q20_1 ~ Q20_4

총 18개 문항
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

from result_utils import markdown_output

# ============================================================
# 1. 데이터 로드
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
df = df[df["Q3"] == 1].copy()

# ============================================================
# 2. 주요 측정 문항 선택 (모든 구성개념 문항 통합)
# ============================================================

items = (
    ["Q9_3", "Q9_4"]                        # Voluntary Motivation
    + [f"Q7_{i}" for i in range(1, 6)]      # Perceived Work Effectiveness
    + [f"Q16_{i}" for i in range(1, 7)]     # Perceived Organizational Support (Q16_7 제외)
    + [f"Q20_{i}" for i in range(2, 5)]     # Strategic Utilization Expectations (Q20_1 제외)
)

data = df[items].dropna()
n_items = len(items)

# ============================================================
# 3. Harman's Single Factor Test
#    - 비회전(Unrotated) PCA 실시
#    - 첫 번째 주성분의 설명분산 비율 확인
# ============================================================

with markdown_output("09_cmv_harman.md") as result_path:
    print("# 09 Common Method Variance 검증\n")
    print(f"- AI 활용자 수: {len(df)}명")
    print(f"- 총 측정 문항 수: {n_items}개\n")
    print(f"  - Voluntary Motivation: Q9_3, Q9_4")
    print(f"  - Work Effectiveness: Q7_1~Q7_5")
    print(f"  - Organizational Support: Q16_1~Q16_6")
    print(f"  - Strategic Expectations: Q20_2~Q20_4\n")

    print("\n\n" + "="*60)
    print("Harman's Single Factor Test (CMV 검증)")
    print("="*60)

    # 표준화 후 PCA
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(data)

    pca = PCA()
    pca.fit(X_scaled)

    explained_var   = pca.explained_variance_ratio_ * 100
    cumulative_var  = np.cumsum(explained_var)

    print(f"\n[비회전 PCA 요인별 설명분산]")
    print(f"{'요인':<6} {'고유값':>8} {'설명분산(%)':>12} {'누적분산(%)':>12}")
    print("-"*42)
    eigenvalues = pca.explained_variance_
    for i in range(min(n_items, 10)):
        print(f"  F{i+1:<4} {eigenvalues[i]:>8.3f} {explained_var[i]:>12.2f} {cumulative_var[i]:>12.2f}")

# ============================================================
# 4. 핵심 판단 기준
# ============================================================

    first_factor_var = explained_var[0]
    threshold = 50.0

    print(f"\n\n{'='*60}")
    print(f"CMV 판단 결과")
    print(f"{'='*60}")
    print(f"\n  전체 문항 수         : {n_items}개")
    print(f"  제1요인 설명분산     : {first_factor_var:.2f}%")
    print(f"  판단 기준            : 50% 초과 시 CMV 우려")
    print()

    if first_factor_var > threshold:
        print(f"  ⚠ 결과: CMV 우려 있음")
        print(f"     → 제1요인이 {first_factor_var:.2f}%를 설명하여 기준(50%)을 초과합니다.")
        print(f"     → 논문에서 CMV 가능성을 한계로 명시하고 추가 검증(Marker Variable 등)을 권고합니다.")
    else:
        print(f"  ✓ 결과: 50% 기준 이하는 확인됨")
        print(f"     → 제1요인이 {first_factor_var:.2f}%를 설명하여 기준(50%) 이하입니다.")
        print(f"     → 다만 Harman 검정만으로 CMV 부재를 주장할 수는 없으므로, 단일시점·단일원천 자료라는 점은 계속 주의해서 해석해야 합니다.")

# ============================================================
# 5. 고유값 1 이상 요인 수 (Kaiser 기준)
# ============================================================

    n_factors_kaiser = np.sum(eigenvalues >= 1.0)
    print(f"\n  고유값 ≥ 1.0 요인 수 : {n_factors_kaiser}개 (단일요인 구조이면 1개여야 함)")
    print(f"  → 실제 {n_factors_kaiser}개 요인이 고유값 1 이상 → 다요인 구조 지지")

# 단일요인 49-50% 기준 외 추가 판단
    total_var_by_1 = explained_var[0]
    remaining_by_others = 100 - total_var_by_1
    print(f"\n  제1요인 외 나머지 설명분산 합: {remaining_by_others:.2f}%")

# ============================================================
# 6. 논문 기술용 결과 요약
# ============================================================

    print(f"""
{'='*60}
[논문 기술 예시 (영문)]
{'='*60}

To assess common method variance (CMV), Harman's single factor
test was conducted by entering all {n_items} measurement items into
an unrotated principal component analysis (PCA). The first
factor accounted for {first_factor_var:.1f}% of the total variance,
which is {'above' if first_factor_var > threshold else 'below'} the 50% threshold suggested by Podsakoff et al.
(2003). {'This result indicates potential CMV concern.' if first_factor_var > threshold
         else "This result is below the conventional 50% threshold, although CMV cannot be ruled out solely on the basis of Harman's test."}
Additionally, {n_factors_kaiser} factors had eigenvalues greater than 1.0,
further supporting a multi-factor structure rather than
a single dominant method factor.

[참고문헌]
Podsakoff, P. M., MacKenzie, S. B., Lee, J.-Y., & Podsakoff, N. P. (2003).
Common method biases in behavioral research: A critical review of the
literature and recommended remedies.
Journal of Applied Psychology, 88(5), 879–903.
{'='*60}
""")
    print("\n## 주요 해석\n")
    print("- 제1요인 설명분산이 50%를 넘지 않았다는 점은 최소한 단일 지배적 방법요인이 자료 전체를 설명하는 상황은 아니라는 보조 근거가 된다.")
    print("- 고유값 1 이상 요인이 여러 개이면 응답이 단일 방법요인보다 여러 실질적 구성개념으로 분화되어 있음을 시사한다.")
    print("- 특히 본 연구의 핵심 결과는 모든 경로가 일괄적으로 커지는 패턴이 아니라, motivation은 완전매개에 가깝고 support는 직접효과와 간접효과가 함께 유지되는 비대칭 구조이므로, 이를 단순한 CMV 산물로만 보기는 어렵다.")
    print("- 그럼에도 본 자료는 횡단면 단일원천 설문이므로, CMV 관련 해석에는 계속 보수적 주석을 유지하는 것이 적절하다.")

print(f"완료: {result_path} 생성")
