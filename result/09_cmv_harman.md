# 09 Common Method Variance 검증

- AI 활용자 수: 377명
- 총 측정 문항 수: 16개

  - Voluntary Motivation: Q9_3, Q9_4
  - Work Effectiveness: Q7_1~Q7_5
  - Organizational Support: Q16_1~Q16_6
  - Strategic Expectations: Q20_2~Q20_4



============================================================
Harman's Single Factor Test (CMV 검증)
============================================================

[비회전 PCA 요인별 설명분산]
요인          고유값      설명분산(%)      누적분산(%)
------------------------------------------
  F1       5.793        36.11        36.11
  F2       2.796        17.43        53.54
  F3       1.297         8.08        61.62
  F4       1.243         7.75        69.37
  F5       0.772         4.81        74.19
  F6       0.660         4.12        78.30
  F7       0.613         3.82        82.12
  F8       0.545         3.40        85.52
  F9       0.496         3.09        88.61
  F10      0.382         2.38        90.99


============================================================
CMV 판단 결과
============================================================

  전체 문항 수         : 16개
  제1요인 설명분산     : 36.11%
  판단 기준            : 50% 초과 시 CMV 우려

  ✓ 결과: 50% 기준 이하는 확인됨
     → 제1요인이 36.11%를 설명하여 기준(50%) 이하입니다.
     → 다만 Harman 검정만으로 CMV 부재를 주장할 수는 없으므로, 단일시점·단일원천 자료라는 점은 계속 주의해서 해석해야 합니다.

  고유값 ≥ 1.0 요인 수 : 4개 (단일요인 구조이면 1개여야 함)
  → 실제 4개 요인이 고유값 1 이상 → 다요인 구조 지지

  제1요인 외 나머지 설명분산 합: 63.89%

============================================================
[논문 기술 예시 (영문)]
============================================================

To assess common method variance (CMV), Harman's single factor
test was conducted by entering all 16 measurement items into
an unrotated principal component analysis (PCA). The first
factor accounted for 36.1% of the total variance,
which is below the 50% threshold suggested by Podsakoff et al.
(2003). This result is below the conventional 50% threshold, although CMV cannot be ruled out solely on the basis of Harman's test.
Additionally, 4 factors had eigenvalues greater than 1.0,
further supporting a multi-factor structure rather than
a single dominant method factor.

[참고문헌]
Podsakoff, P. M., MacKenzie, S. B., Lee, J.-Y., & Podsakoff, N. P. (2003).
Common method biases in behavioral research: A critical review of the
literature and recommended remedies.
Journal of Applied Psychology, 88(5), 879–903.
============================================================


## 주요 해석

- 제1요인 설명분산이 50%를 넘지 않았다는 점은 최소한 단일 지배적 방법요인이 자료 전체를 설명하는 상황은 아니라는 보조 근거가 된다.
- 고유값 1 이상 요인이 여러 개이면 응답이 단일 방법요인보다 여러 실질적 구성개념으로 분화되어 있음을 시사한다.
- 특히 본 연구의 핵심 결과는 모든 경로가 일괄적으로 커지는 패턴이 아니라, motivation은 완전매개에 가깝고 support는 직접효과와 간접효과가 함께 유지되는 비대칭 구조이므로, 이를 단순한 CMV 산물로만 보기는 어렵다.
- 그럼에도 본 자료는 횡단면 단일원천 설문이므로, CMV 관련 해석에는 계속 보수적 주석을 유지하는 것이 적절하다.
