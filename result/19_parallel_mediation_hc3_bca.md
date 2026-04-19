# 19 Parallel Mediation with HC3 and BCa Bootstrap

N = 377
Bootstrap resamples = 5000


Mediator model: effect ~ motivation + support + controls
motivation: B = 0.341556, SE(HC3) = 0.056388, p = 0.000000
support: B = 0.208202, SE(HC3) = 0.040217, p = 0.000000
R2 = 0.263267

Outcome model: expectation ~ motivation + support + effect + controls
motivation: B = 0.080082, SE(HC3) = 0.055580, p = 0.149625
support: B = 0.326040, SE(HC3) = 0.056215, p = 0.000000
effect: B = 0.371430, SE(HC3) = 0.065316, p = 0.000000
R2 = 0.311072

Total-effect model: expectation ~ motivation + support + controls
motivation: B = 0.206946, SE(HC3) = 0.057587, p = 0.000326
support: B = 0.403372, SE(HC3) = 0.056032, p = 0.000000
R2 = 0.241411

Effects with 95% BCa CI
direct_motivation: estimate = 0.080082, 95% BCa CI = [-0.024450, 0.185803]
indirect_motivation: estimate = 0.126864, 95% BCa CI = [0.081159, 0.190573]
total_motivation: estimate = 0.206946, 95% BCa CI = [0.092945, 0.315258]
direct_support: estimate = 0.326040, 95% BCa CI = [0.211038, 0.429142]
indirect_support: estimate = 0.077333, 95% BCa CI = [0.046633, 0.121163]
total_support: estimate = 0.403372, 95% BCa CI = [0.289805, 0.508081]

Local f2
mediator_motivation = 0.183365
mediator_support = 0.069188
outcome_direct_motivation = 0.006243
outcome_direct_support = 0.116307
outcome_effect = 0.101114
total_motivation = 0.044806
total_support = 0.172862

## 주요 해석
- 이 모형은 motivation과 support를 동시에 투입해 두 변수의 직접효과, 간접효과, 총효과를 병렬적으로 비교한다.
- BCa 신뢰구간에 0이 포함되지 않는 간접효과는 보다 안정적인 매개효과로 해석할 수 있다.
- `f2`는 각 경로가 설명력에 기여하는 국소 효과크기를 보여주므로, 통계적 유의성과 함께 실질적 크기도 판단할 수 있다.
