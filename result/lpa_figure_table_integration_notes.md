# LPA Figure and Table Manuscript Integration Notes

> **Purpose**: Placement recommendations, example reference sentences, and editorial guidance for integrating LPA outputs into the GIQ manuscript.  
> **Based on**: 39_lpa_integrated_assessment.md (Section 6), GIQ/SSCI manuscript conventions.  
> **Profiles**: k=3 solution (primary); k=4 in Appendix only.

---

## 1. Recommended Section Structure

```
§4 Results
  §4.1  Descriptive Statistics and Correlations
  §4.2  OLS Regression: Direct and Mediated Pathways (main analysis)
  §4.3  BCa Bootstrap Parallel Mediation (main analysis)
  §4.4  Person-Centered Extension: Latent Profile Analysis
      §4.4.1  Profile Enumeration and Selection (→ Table 1)
      §4.4.2  Profile Characteristics (→ Table 2, Figure 1)
      §4.4.3  Demographic and Organizational Differences (→ Table 3)
      §4.4.4  Within-Profile Exploratory OLS (→ narrative only)

§5 Discussion
  §5.x  Automation Skepticism as a Universal Structural Feature
  §5.x  Person-Centered Heterogeneity and Theoretical Implications
  §5.x  Organizational Support as a Within-Profile Moderator

Appendix
  Appendix Figure. k=4 Sensitivity Check (→ Appendix Figure)
```

---

## 2. Figure 1 — Latent Profile Plot

**File**: `figures/lpa_profile_plot.png` / `figures/lpa_profile_plot.pdf`  
**Code**: `code/generate_lpa_profile_plot.py`

### Recommended Placement
- Section 4.4.2, immediately following the opening paragraph describing profile means.
- Insert as a full-width single-column figure (journal-standard width ~86 mm or ~174 mm for double column).

### Results Section Reference Sentences

> "Figure 1 displays the mean expectancy scores across the three LPA-derived profiles. Differentiation across profiles is most pronounced for efficiency expectancy (Q20_1 range: 2.73–5.00) and decision-support expectancy (Q20_2 range: 2.93–4.35), whereas automation expectancy (Q20_3) remains the lowest-rated dimension in all three profiles (range: 2.44–2.84)."

> "As illustrated in Figure 1, all three profiles exhibit a common pattern of substantially lower automation expectations relative to efficiency and decision-support dimensions, suggesting that automation skepticism is a structural feature of the sample's expectancy landscape rather than a characteristic of any specific subgroup."

### Discussion Reference Sentences

> "The profile plot (Figure 1) reinforces the variable-centered finding that intrinsic motivation does not significantly predict automation expectations (Q20_3) at the aggregate level: even among the highest-expectancy profile (Profile 3), automation expectations (M = 2.84) remain well below the scale midpoint relative to efficiency (M = 5.00) and decision-support expectations (M = 4.35)."

> "The substantial gap between Q20_3 and the other two expectancy dimensions—visible across all profiles in Figure 1—suggests that Korean public servants may view AI as a tool for enhancing human judgment and efficiency rather than as a substitute for routine task execution, consistent with institutional norms of public accountability."

---

## 3. Table 1 — Profile Enumeration and Fit Statistics

**File**: `tables/lpa_fit_table.md`  
**Note**: docx version requires python-docx (not installed in current environment). Use the .md version for drafting; typeset in Word/LaTeX for final submission.

### Recommended Placement
- Section 4.4.1, as the first table in the LPA subsection.

### Results Section Reference Sentences

> "Table 1 presents the model fit statistics for k = 2 through k = 5. The two-profile solution was rejected due to inadequate classification entropy (.659). The five-profile solution yielded a minimum profile size of 18, below the recommended threshold of 50. The four-profile solution, while achieving a lower BIC (−1,540.6), was disqualified as the additional profile represents a ceiling-response partition rather than a theoretically distinct latent class (see Appendix Figure and methodological note in Table 1). The three-profile solution (BIC = −572.9, entropy = 1.00) was selected as the primary solution on grounds of stability (BIC SD = 0.00 across ten random seeds), minimum profile size (n_min = 75), and theoretical interpretability."

### Discussion Reference Sentences

> "The selection of k = 3 over k = 4 reflects a substantive rather than purely statistical criterion: the three-profile solution is the most parsimonious structure that provides theoretically meaningful differentiation while remaining reproducible across analytical specifications (Table 1)."

---

## 4. Table 2 — Profile Characteristics

**File**: `tables/lpa_profile_characteristics.md`

### Recommended Placement
- Section 4.4.2, as the primary descriptive table for the LPA results.

### Results Section Reference Sentences

> "Table 2 presents the descriptive characteristics of each latent profile. The three profiles differ substantially on efficiency (Q20_1) and decision-support (Q20_2) expectancies but converge on uniformly low automation expectations (Q20_3 range: 2.44–2.84), consistent with the aggregate-level OLS finding that motivation does not significantly predict Q20_3 (total effect β = n.s.)."

> "Profile 1 (Generally Low Expectancy; n = 75, 19.9%) is characterized by below-midpoint scores on all three expectancy dimensions and the lowest mean motivation (M = 3.21) and effectiveness perception (M = 3.18) in the sample. Profile 2 (Efficiency-Oriented, Automation-Resistant; n = 161, 42.7%) — the modal group — shows moderate-to-high efficiency expectations (M = 4.00) but limited decision-support (M = 3.58) and automation expectations (M = 2.50). Profile 3 (High Efficiency-Decision Expectancy; n = 141, 37.4%) exhibits ceiling-level efficiency expectations (M = 5.00) and near-ceiling decision-support expectations (M = 4.35), accompanied by the highest motivation (M = 4.23) and effectiveness perception (M = 4.09)."

> "Organizational AI support (support_main) displayed negligible variation across profiles (range: 2.79–2.90), in contrast to the systematic monotonic gradients observed for motivation and perceived effectiveness. This pattern suggests that organizational support context does not determine expectancy profile membership (Table 2), although it may modulate expectancy intensity within profiles, as indicated by the within-profile OLS results."

### Discussion Reference Sentences

> "The monotonic gradient in motivation and perceived effectiveness across profiles (Table 2) provides person-centered convergent evidence for the SDT-EVT mediation pathway identified in the primary OLS analysis: intrinsic motivation, channeled through enhanced effectiveness perceptions, appears to be a necessary condition for consolidated positive AI expectations."

> "The comparative invariance of organizational support (support_main) across profiles (Table 2) nuances the OLS finding of a significant support_main → Q20_3 direct effect. The LPA evidence suggests that organizational support functions as a within-profile intensity factor rather than a profile-entry determinant — influencing how strongly expectations are held rather than what type of expectancy structure an individual adopts."

---

## 5. Table 3 — Demographic Differences

**File**: `tables/lpa_demographic_differences.md`

### Recommended Placement
- Section 4.4.3, following the multinomial logistic regression narrative.

### Results Section Reference Sentences

> "Table 3 presents demographic and organizational differences across the three latent profiles. Gender did not differ significantly across profiles (χ²(2) = 3.43, p = .180), though Profile 3 showed a somewhat higher proportion of male respondents (59.6%) relative to Profiles 1 and 2 (52.0% and 49.1%). Administrative rank exhibited a statistically significant association with profile membership (χ²(12) = 22.18, p = .036), though the effect size was small (V = .172) and no clear directional interpretation was consistent across rank levels. Career tenure did not differ significantly across profiles (p = .348)."

> "In contrast to demographic variables, intrinsic motivation (F(2, 374) = 46.83, p < .001, η² = .200) and perceived work effectiveness (F(2, 374) = 49.59, p < .001, η² = .210) showed large and statistically significant differences across profiles, underscoring that motivational and experiential factors — rather than sociodemographic characteristics — are the primary correlates of expectancy profile membership."

### Discussion Reference Sentences

> "The finding that demographic variables (gender, rank, tenure) explain limited variance in profile membership (Table 3) has practical implications: it suggests that AI expectancy profiles are not simply a proxy for career stage or hierarchical position, but rather reflect individual differences in motivation and effectiveness experience that may be more amenable to targeted intervention."

> "The absence of significant support_main differences across profiles (Table 3; F = 0.52, p = .594, η² = .003) is particularly noteworthy given support_main's significant role as a direct predictor of Q20_3 in both the aggregate OLS and the within-profile analyses. This dissociation suggests that the mechanism through which organizational support influences AI adoption expectations operates at the within-person level of expectancy adjustment rather than at the level of expectancy type formation."

---

## 6. Appendix Figure — k=4 Ceiling Artifact

**Files**: `appendix/k4_ceiling_artifact_plot.png` / `appendix/k4_ceiling_artifact_plot.pdf`

### Recommended Placement
- Manuscript Appendix (online supplementary material or printed appendix), labeled "Appendix Figure."
- Cross-reference in main text: "...the k = 4 solution is presented for comparison in the Appendix Figure..."

### Results Section Cross-Reference Sentences

> "As a sensitivity check, we also estimated a k = 4 solution (BIC = −1,540.6, entropy = 1.00). However, robustness testing revealed that the additional profile in k = 4 corresponds to a ceiling-response partition — specifically, the bifurcation of respondents who rated Q20_1 at the maximum score (5) based on whether Q20_2 also reached the ceiling (see Appendix Figure). This partition is mechanically determined by response scale concentration rather than theoretically interpretable latent heterogeneity, and the k = 4 solution proved seed-sensitive (BIC SD = 304.1 across ten random initializations). The k = 3 solution is therefore preferred as the primary result."

### Reviewer-Oriented Justification (for cover letter or response letter)

> "We conducted a sensitivity analysis examining k = 2 through k = 5. The Appendix Figure compares the k = 3 and k = 4 solutions side by side. The four-profile solution was identified as containing a ceiling-response artifact: Profile P0 (n = 80) consists of all respondents who rated Q20_1 = 5 and Q20_2 = 5, while Profile P1 (n = 61) consists of respondents with Q20_1 = 5 and Q20_2 < 5. This bifurcation is an artifact of response ceiling concentration (37.4% at Q20_1 maximum) rather than a substantively distinct latent class. We confirmed this by demonstrating that the k = 4 solution is seed-sensitive (BIC ranging from −489 to −1,133 across ten seeds under raw data), whereas k = 3 is perfectly stable (BIC SD = 0.00). We therefore adopt k = 3 as the primary solution and present k = 4 in the Appendix for methodological transparency."

---

## 7. General Editorial Guidance

### Framing Language for LPA Subsection Opening

> "As a supplementary person-centered extension, we applied latent profile analysis (LPA) to examine whether the AI user sample (*N* = 377) comprises identifiable subgroups with distinct expectancy configurations. This analysis complements the primary OLS regression by revealing individual-level heterogeneity in AI expectancy formation that aggregate estimates cannot capture. We note that the LPA relies on three indicators only, and the GMM-based approximation should be cross-validated in dedicated LPA software prior to confirmatory use; we therefore treat these results as exploratory."

### Framing Language for LPA Subsection Close

> "Taken together, the person-centered analysis converges with and extends the variable-centered findings: motivation and perceived effectiveness differentiate among expectancy profiles (supporting SDT-EVT mediation), organizational support shapes within-profile expectancy intensity without determining profile membership, and automation skepticism is a universal structural feature of Korean public servants' AI expectancy landscape that is not reducible to any single subgroup or predictor pathway."

### Language to Avoid

- "LPA revealed the true latent structure of..." → revise to "LPA identified three reproducible expectancy configurations..."
- "The three classes are clearly distinct..." → revise to "The three profiles are empirically separable..."
- "The high-expectancy class causes..." → revise to "respondents in the high-expectancy profile tend to..."
- "Entropy of 1.00 confirms perfect classification..." → revise to "entropy ≥ .999 indicates near-perfect probabilistic assignment, though this may partly reflect ceiling-response concentration..."

---

## 8. File Index

| Output | File | Manuscript Location |
|:---|:---|:---|
| Figure 1 (PNG) | `figures/lpa_profile_plot.png` | Section 4.4.2 |
| Figure 1 (PDF) | `figures/lpa_profile_plot.pdf` | Section 4.4.2 |
| Figure 1 code | `code/generate_lpa_profile_plot.py` | Supplementary materials |
| Table 1 (Fit) | `tables/lpa_fit_table.md` | Section 4.4.1 |
| Table 2 (Profiles) | `tables/lpa_profile_characteristics.md` | Section 4.4.2 |
| Table 3 (Demographics) | `tables/lpa_demographic_differences.md` | Section 4.4.3 |
| Appendix Figure (PNG) | `appendix/k4_ceiling_artifact_plot.png` | Appendix |
| Appendix Figure (PDF) | `appendix/k4_ceiling_artifact_plot.pdf` | Appendix |
| Appendix Figure code | `code/generate_k4_appendix_plot.py` | Supplementary materials |

---

*Generated: 2026-05-20*  
*Based on: 35_lpa_robustness.md, 36_lpa_sensitivity.md, 39_lpa_integrated_assessment.md*  
*Primary solution: k=3 (BIC=-572.9, entropy=1.00, N_min=75)*
