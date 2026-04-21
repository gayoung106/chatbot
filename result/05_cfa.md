# 05 Confirmatory Factor Analysis

- Sample: AI users only (`Q3 == 1`)
- Analysis design of the paper is OLS + bootstrap, not SEM.
- CFA is reported as supplementary measurement evidence, not as the primary basis of analytic validity.
- Excluded from CFA:
  - `support_main`: treated as an observed index, not a latent construct
  - `Q20_1~Q20_4`: treated as item-level outcomes, not a single latent scale

## Supplementary CFA check: effect factor only

| Model                                 |   N |   chi-square |   df | p-value   |   CFI |   TLI |   RMSEA | RMSEA 90% CI   |   SRMR |
|:--------------------------------------|----:|-------------:|-----:|:----------|------:|------:|--------:|:---------------|-------:|
| Supplementary CFA check (effect only) | 377 |      109.646 |    5 | < .001    | 0.901 | 0.802 |   0.236 | [0.199, 0.275] |  0.055 |

## Standardized factor loadings for effect

| Factor   | Item   |   Loading |   Std.Loading | p-value   |
|:---------|:-------|----------:|--------------:|:----------|
| effect   | Q7_1   |     1     |         0.791 | -         |
| effect   | Q7_2   |     0.944 |         0.786 | < .001    |
| effect   | Q7_3   |     0.797 |         0.654 | < .001    |
| effect   | Q7_4   |     1.001 |         0.829 | < .001    |
| effect   | Q7_5   |     1.039 |         0.816 | < .001    |

## Supplementary CFA: full reflective block (motivation + effect)

| Model                               |   N |   chi-square |   df | p-value   |   CFI |   TLI |   RMSEA | RMSEA 90% CI   |   SRMR |
|:------------------------------------|----:|-------------:|-----:|:----------|------:|------:|--------:|:---------------|-------:|
| Two-factor CFA (motivation, effect) | 377 |      125.404 |   13 | < .001    |  0.92 | 0.871 |   0.152 | [0.128, 0.176] |    0.2 |

## Interpretation

- Primary validity evidence in the paper rests on EFA, CR, AVE, Fornell-Larcker, and HTMT criteria.
- The supplementary two-factor CFA showed poor absolute residual fit (`SRMR = 0.200`) and is not used as confirmatory evidence for the full measurement block.
- The one-factor CFA for `effect` is retained only as a limited supplementary check: `CFI = 0.901`, `TLI = 0.802`, `RMSEA = 0.236` with 90% CI [0.199, 0.275], and `SRMR = 0.055`.
- Because CFI/TLI and RMSEA do not meet conventional fit criteria, CFA is not framed as decisive support for unidimensionality; it is reported transparently as a limitation.
- `support_main` remains an observed organizational-context index and is not treated as a latent variable.
