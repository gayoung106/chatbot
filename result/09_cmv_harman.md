# 09 Common Method Variance Check

## Input for Harman's single-factor diagnostic

- AI users retained before listwise deletion = 377
- AI users retained after listwise deletion = 377
- Number of items = 15
- Motivation items: Q9_3, Q9_4
- Work-effectiveness items: Q7_1~Q7_5
- Organizational-support items: Q16_1~Q16_4
- Expectancy items: Q20_1~Q20_4
- Q20_4 is supplementary in the main models, but it is retained here because CMV is a measurement-level diagnostic.

## PCA variance summary

| Factor | Eigenvalue | Explained variance (%) | Cumulative variance (%) |
|--------|-----------:|-----------------------:|------------------------:|
| F1 | 5.318 | 35.36 | 35.36 |
| F2 | 2.888 | 19.20 | 54.56 |
| F3 | 1.313 | 8.73 | 63.29 |
| F4 | 1.227 | 8.16 | 71.45 |
| F5 | 0.734 | 4.88 | 76.33 |
| F6 | 0.659 | 4.38 | 80.71 |
| F7 | 0.557 | 3.70 | 84.41 |
| F8 | 0.485 | 3.23 | 87.64 |
| F9 | 0.414 | 2.75 | 90.39 |
| F10 | 0.352 | 2.34 | 92.73 |
| F11 | 0.296 | 1.97 | 94.70 |
| F12 | 0.236 | 1.57 | 96.27 |
| F13 | 0.231 | 1.54 | 97.80 |
| F14 | 0.182 | 1.21 | 99.01 |
| F15 | 0.149 | 0.99 | 100.00 |

## Harman diagnostic result

- First unrotated factor explained variance = 35.36%
- Number of factors with eigenvalue >= 1.00 = 4
- Conventional reference point: CMV concern becomes stronger when the first factor exceeds 50% of total variance.
- Interpretation: the first factor remains below 50%, so the data do not suggest that a single dominant common-method factor drives the observed covariance structure.
- This does not prove the absence of CMV; it only supports the narrower claim that the covariance pattern is not reducible to one dominant response artifact.
