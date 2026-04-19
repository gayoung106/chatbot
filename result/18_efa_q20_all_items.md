# 18 Q20 전체 문항 EFA 점검

N = 377

Correlations
       Q20_1  Q20_2  Q20_3  Q20_4
Q20_1  1.000  0.538  0.140  0.137
Q20_2  0.538  1.000  0.403  0.344
Q20_3  0.140  0.403  1.000  0.580
Q20_4  0.137  0.344  0.580  1.000

Reliability
alpha(Q20_1~Q20_4) = 0.691
alpha(Q20_2~Q20_4) = 0.707
Q20_1    0.322
Q20_2    0.561
Q20_3    0.535
Q20_4    0.499
Name: item_total_r, dtype: float64

Factorability
KMO total = 0.598
Q20_1    0.534
Q20_2    0.608
Q20_3    0.604
Q20_4    0.630
Name: KMO, dtype: float64
Bartlett chi2 = 359.661, df = 6

Eigenvalues
Factor1 = 2.090
Factor2 = 1.088
Factor3 = 0.443
Factor4 = 0.380

1-factor PAF
          F1
Q20_1 -0.425
Q20_2 -0.712
Q20_3 -0.661
Q20_4 -0.610
Q20_1    0.181
Q20_2    0.507
Q20_3    0.437
Q20_4    0.373
Name: communality, dtype: float64
RMSR = 0.150

2-factor PAF + varimax
          F1     F2
Q20_1 -0.029 -0.716
Q20_2 -0.337 -0.739
Q20_3 -0.841 -0.162
Q20_4 -0.658 -0.165
Q20_1    0.513
Q20_2    0.659
Q20_3    0.734
Q20_4    0.460
Name: communality, dtype: float64
RMSR = 0.000

## 주요 해석
- Two eigenvalues exceed 1.0, so a 2-factor split is plausible at a heuristic level.
- The rotated 2-factor pattern is Q20_1/Q20_2 versus Q20_3/Q20_4.
- However, Q20_2 cross-loads and each factor has only two indicators, so this is not a strong confirmation of a stable 2-factor construct.
- The existing 3-item expectation scale remains psychometrically cleaner than the 4-item version.
