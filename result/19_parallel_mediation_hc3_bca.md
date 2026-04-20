# 19 Parallel Mediation with HC3 and BCa Bootstrap

N (AI users before per-DV listwise) = 377
Bootstrap resamples = 5000
- Main DVs: Q20_1, Q20_2, Q20_3 (item-level; Q20_4 excluded from main analysis)
  - support_main = mean(Q16_1~Q16_4)
  - motivation = mean(Q9_3, Q9_4)

## 0. Inter-item correlation for motivation

- Pearson r(Q9_3, Q9_4) = 0.7056, p = 0.0000
- N (valid pairs) = 377

## Q20_1: 업무효율 개선 기대

- N = 377

### 1. Mediator model
- motivation -> effect: B = 0.3723, p = 0.0000
- support_main -> effect: B = 0.1137, p = 0.0007
- R2 = 0.234

### 2. Total-effect model
- motivation -> Q20_1: B = 0.5231, p = 0.0000
- support_main -> Q20_1: B = -0.0394, p = 0.2870
- R2 = 0.258

### 3. Direct-effect model
- motivation -> Q20_1: B = 0.3663, p = 0.0000
- support_main -> Q20_1: B = -0.0873, p = 0.0110
- effect -> Q20_1: B = 0.4213, p = 0.0000
- R2 = 0.362

### 4. BCa bootstrap effects
- direct_motivation: estimate = 0.3663, 95% BCa CI = [0.2499, 0.4872]
- indirect_motivation: estimate = 0.1569, 95% BCa CI = [0.1085, 0.2256]
- total_motivation: estimate = 0.5231, 95% BCa CI = [0.3913, 0.6454]
- direct_support_main: estimate = -0.0873, 95% BCa CI = [-0.1557, -0.0216]
- indirect_support_main: estimate = 0.0479, 95% BCa CI = [0.0228, 0.0821]
- total_support_main: estimate = -0.0394, 95% BCa CI = [-0.1115, 0.0348]

## Q20_2: 의사결정 지원 기대

- N = 377

### 1. Mediator model
- motivation -> effect: B = 0.3723, p = 0.0000
- support_main -> effect: B = 0.1137, p = 0.0007
- R2 = 0.234

### 2. Total-effect model
- motivation -> Q20_2: B = 0.4452, p = 0.0000
- support_main -> Q20_2: B = 0.1118, p = 0.0221
- R2 = 0.177

### 3. Direct-effect model
- motivation -> Q20_2: B = 0.2499, p = 0.0003
- support_main -> Q20_2: B = 0.0522, p = 0.2699
- effect -> Q20_2: B = 0.5248, p = 0.0000
- R2 = 0.294

### 4. BCa bootstrap effects
- direct_motivation: estimate = 0.2499, 95% BCa CI = [0.1148, 0.3793]
- indirect_motivation: estimate = 0.1954, 95% BCa CI = [0.1375, 0.2734]
- total_motivation: estimate = 0.4452, 95% BCa CI = [0.2963, 0.5758]
- direct_support_main: estimate = 0.0522, 95% BCa CI = [-0.0474, 0.1395]
- indirect_support_main: estimate = 0.0597, 95% BCa CI = [0.0270, 0.1033]
- total_support_main: estimate = 0.1118, 95% BCa CI = [0.0129, 0.2033]

## Q20_3: 반복업무 자동화 기대

- N = 377

### 1. Mediator model
- motivation -> effect: B = 0.3723, p = 0.0000
- support_main -> effect: B = 0.1137, p = 0.0007
- R2 = 0.234

### 2. Total-effect model
- motivation -> Q20_3: B = 0.1345, p = 0.0665
- support_main -> Q20_3: B = 0.3916, p = 0.0000
- R2 = 0.135

### 3. Direct-effect model
- motivation -> Q20_3: B = -0.0247, p = 0.7320
- support_main -> Q20_3: B = 0.3430, p = 0.0000
- effect -> Q20_3: B = 0.4276, p = 0.0000
- R2 = 0.190

### 4. BCa bootstrap effects
- direct_motivation: estimate = -0.0247, 95% BCa CI = [-0.1698, 0.1151]
- indirect_motivation: estimate = 0.1592, 95% BCa CI = [0.0961, 0.2441]
- total_motivation: estimate = 0.1345, 95% BCa CI = [-0.0080, 0.2759]
- direct_support_main: estimate = 0.3430, 95% BCa CI = [0.2221, 0.4555]
- indirect_support_main: estimate = 0.0486, 95% BCa CI = [0.0219, 0.0867]
- total_support_main: estimate = 0.3916, 95% BCa CI = [0.2681, 0.5052]

## Holm-Bonferroni Correction (Total-effect p-values)

### motivation total-effect p-values

| DV | Original p | Adjusted p | Holm threshold | Significant? |
|----|-----------:|-----------:|---------------:|:-------------|
| Q20_1 (업무효율 개선 기대) | 0.0000 | 0.0000 | 0.0167 | Yes |
| Q20_2 (의사결정 지원 기대) | 0.0000 | 0.0000 | 0.0250 | Yes |
| Q20_3 (반복업무 자동화 기대) | 0.0665 | 0.0665 | 0.0500 | No |

### support_main total-effect p-values

| DV | Original p | Adjusted p | Holm threshold | Significant? |
|----|-----------:|-----------:|---------------:|:-------------|
| Q20_1 (업무효율 개선 기대) | 0.2870 | 0.2870 | 0.0500 | No |
| Q20_2 (의사결정 지원 기대) | 0.0221 | 0.0441 | 0.0250 | Yes |
| Q20_3 (반복업무 자동화 기대) | 0.0000 | 0.0000 | 0.0167 | Yes |
