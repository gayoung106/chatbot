# Table 1. Latent Profile Analysis Model Fit Statistics

**Table 1**  
*Latent Profile Analysis: Model Enumeration and Fit Statistics (N = 377)*

| Profiles (*k*) | AIC | BIC | Entropy | Min. Profile *N* | Profile Sizes | Decision |
|:---:|---:|---:|:---:|:---:|:---|:---|
| 2 | 3,019.2 | 3,070.3 | .659 | 178 | [178, 199] | Rejected: entropy < .80; theoretically insufficient differentiation |
| **3** | **−651.6** | **−572.9** | **1.000** | **75** | **[75, 141, 161]** | **Selected: stable across all seeds (BIC *SD* = 0.00); profiles theoretically interpretable** |
| 4 | −1,646.8 | −1,540.6 | 1.000 | 61 | [61, 75, 80, 161] | Rejected: ceiling-response artifact (see note); seed-unstable (BIC *SD* = 304.1) |
| 5 | −2,421.5 | −2,287.8 | .999 | 18 | [18, 57, 70, 91, 141] | Rejected: minimum profile *N* = 18 < criterion of 50 |

*Note.* AIC = Akaike Information Criterion; BIC = Bayesian Information Criterion; Entropy indexes classification certainty (range 0–1; values ≥ .80 indicate adequate separation). Model fit was estimated via Gaussian mixture modeling with diagonal covariance structure (sklearn GaussianMixture, n_init = 20, random_state = 42), which approximates formal latent profile analysis (LPA). Results should be considered exploratory pending cross-validation in dedicated LPA software (e.g., Mplus, R tidyLPA). The k = 3 solution was preferred over k = 4 because the additional profile in k = 4 constitutes a ceiling-response partition artifact: Profile P0 (*n* = 80) consists exclusively of respondents who rated both Q20_1 and Q20_2 at the maximum score (5), and Profile P1 (*n* = 61) consists of respondents who rated Q20_1 = 5 but Q20_2 < 5. This partition reflects response scale ceiling concentration (37.4% of respondents at Q20_1 = 5) rather than a theoretically meaningful latent class distinction. The k = 4 solution is presented as a methodological sensitivity check in the Appendix.

---

## Interpretation Paragraph (manuscript-ready)

Model enumeration proceeded from *k* = 2 to *k* = 5. The two-profile solution was rejected due to inadequate classification entropy (.659), below the conventional threshold of .80. The five-profile solution was rejected because the smallest profile contained only 18 respondents, falling below the recommended minimum of 50 for stable estimation. For the remaining candidates, the three-profile solution (BIC = −572.9, entropy = 1.00) demonstrated perfect reproducibility across ten random initialization seeds (BIC *SD* = 0.00, profile sizes invariant at [75, 141, 161]), and each profile corresponds to a theoretically interpretable AI expectancy configuration within the SDT-EVT framework. The four-profile solution yielded a lower BIC (−1,540.6) but was rendered suspect by severe seed instability (BIC *SD* = 304.1 across raw-scale seeds) and by the finding that its additional profile is a ceiling-response partition rather than a distinct latent class. Accordingly, the three-profile solution is adopted as the primary result, and the four-profile solution is reported in the Appendix as a sensitivity check.

---

*Table type: Model enumeration / fit statistics*  
*For manuscript Section 4.4.1 (Profile solution and selection)*
