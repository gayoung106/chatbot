# 10 Marker-Proxy CMV Sensitivity Check

- Sample: AI users only (`Q3 == 1`)
- Method: Lindell-Whitney style marker sensitivity using `Q20_4` as a conservative marker proxy.
- Caution: `Q20_4` is not an ideal pure marker variable; this analysis is reported as a robustness/sensitivity check, not as a definitive CMV correction.

## Zero-order vs. marker-adjusted partial correlations

| x            | y            |   zero-order r |   partial_r_marker |   abs_delta_r |
|:-------------|:-------------|---------------:|-------------------:|--------------:|
| motivation   | effect       |          0.432 |              0.405 |         0.027 |
| motivation   | support_main |          0.095 |              0.052 |         0.043 |
| effect       | support_main |          0.189 |              0.125 |         0.065 |
| motivation   | Q20_1        |          0.499 |              0.487 |         0.012 |
| motivation   | Q20_2        |          0.38  |              0.347 |         0.033 |
| motivation   | Q20_3        |          0.128 |              0.034 |         0.094 |
| support_main | Q20_1        |         -0     |             -0.038 |         0.037 |
| support_main | Q20_2        |          0.144 |              0.06  |         0.084 |
| support_main | Q20_3        |          0.333 |              0.231 |         0.102 |
| effect       | Q20_1        |          0.493 |              0.478 |         0.015 |
| effect       | Q20_2        |          0.501 |              0.448 |         0.053 |
| effect       | Q20_3        |          0.328 |              0.209 |         0.119 |

## Regression coefficient sensitivity to marker inclusion

| DV    | predictor    |   B without marker |   B with marker |   abs_delta_B |
|:------|:-------------|-------------------:|----------------:|--------------:|
| Q20_1 | motivation   |              0.366 |           0.365 |         0.001 |
| Q20_1 | support_main |             -0.087 |          -0.09  |         0.002 |
| Q20_1 | effect       |              0.421 |           0.419 |         0.003 |
| Q20_2 | motivation   |              0.25  |           0.235 |         0.015 |
| Q20_2 | support_main |              0.052 |           0.008 |         0.045 |
| Q20_2 | effect       |              0.525 |           0.474 |         0.051 |
| Q20_3 | motivation   |             -0.025 |          -0.069 |         0.045 |
| Q20_3 | support_main |              0.343 |           0.213 |         0.13  |
| Q20_3 | effect       |              0.428 |           0.28  |         0.147 |

## Interpretation

- Mean absolute correlation change = 0.057
- Max absolute correlation change = 0.119
- Mean absolute coefficient change = 0.049
- Max absolute coefficient change = 0.147
- If marker-adjusted correlations and coefficients remain close to the original estimates, the results are less likely to be dominated by a single common-method artifact.
