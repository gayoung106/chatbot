# 05 측정타당성 분석

- 분석 표본: AI 활용자 377명

===================================
Convergent Validity (AVE & CR)
===================================

[work_effect]
Loadings: [-0.796 -0.785 -0.649 -0.829 -0.815]
AVE = 0.604
CR  = 0.884

[org_support]
Loadings: [-0.781 -0.818 -0.801 -0.809 -0.458 -0.506]
AVE = 0.507
CR  = 0.855

[strategic_expectation]
Loadings: [-0.489 -0.825 -0.703]
AVE = 0.471
CR  = 0.719

[motivation_voluntary]
Loadings: [0.84 0.84]
AVE = 0.706
CR  = 0.827

[motivation_voluntary - Two-item reliability check]
Inter-item correlation = 0.706
Spearman-Brown coefficient = 0.827
Note: motivation_voluntary uses two items, so interpret alpha/CR with caution.

===================================
Discriminant Validity (Fornell-Larcker)
===================================

Correlation Matrix
                       work_effect  ...  motivation_voluntary
work_effect                  1.000  ...                 0.432
org_support                  0.323  ...                 0.211
strategic_expectation        0.456  ...                 0.274
motivation_voluntary         0.432  ...                 1.000

[4 rows x 4 columns]

Sqrt(AVE) Values
work_effect: 0.777
org_support: 0.712
strategic_expectation: 0.687
motivation_voluntary: 0.840

Criterion: sqrt(AVE) should exceed inter-construct correlations.

===================================
HTMT
===================================
work_effect - org_support: 0.393
work_effect - strategic_expectation: 0.594
work_effect - motivation_voluntary: 0.510
org_support - strategic_expectation: 0.561
org_support - motivation_voluntary: 0.274
strategic_expectation - motivation_voluntary: 0.376

Criterion: HTMT < .85

## 주요 해석

- AVE와 CR은 각 구성개념이 자신의 문항을 얼마나 일관되게 설명하는지 보여준다.
- 전략기대(strategic_expectation)의 AVE는 .471로 통상적 기준인 .50에 미달한다는 점을 명시적으로 보고해야 한다.
- 다만 전략기대의 sqrt(AVE)=.687은 다른 구성개념과의 상관보다 크고, HTMT도 모든 쌍에서 .85 미만이므로 판별타당성 측면의 방어 논리는 유지된다.
- Fornell-Larcker 기준에서 각 구성개념의 `sqrt(AVE)`가 상관계수보다 크면 판별타당성이 양호하다고 볼 수 있다.
- HTMT가 .85 이하로 유지되면 서로 다른 구성개념이 과도하게 중첩되지 않았다는 해석이 가능하다.
- 자발적 활용동기는 2문항 척도이므로 적재치가 대칭적으로 나타나는 것은 2문항 제약의 산물일 수 있으며, Spearman-Brown 계수를 함께 보고 보수적으로 해석하는 것이 적절하다.
