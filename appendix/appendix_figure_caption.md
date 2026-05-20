# Appendix Figure — Caption and Reviewer Interpretation

## Manuscript-Ready Appendix Figure Caption

**Appendix Figure.** Comparison of *k* = 3 (Primary) and *k* = 4 (Sensitivity Check) LPA Solutions for AI Expectancy Among Korean Public-Sector AI Users (*N* = 377).

*Note.* Panel (a) displays the three-profile solution adopted as the primary result. Panel (b) displays the four-profile solution as a sensitivity check. In panel (b), Profiles P0 and P1 (shown in red) are distinguished exclusively by whether respondents rated Q20_2 (Decision-Support Expectancy) at the response ceiling (score = 5) or below: P0 = {Q20_1 = 5 AND Q20_2 = 5} (*n* = 80); P1 = {Q20_1 = 5 AND Q20_2 < 5} (*n* = 61). This partition constitutes a ceiling-response artifact reflecting scale concentration (37.4% of respondents at Q20_1 maximum) rather than a theoretically meaningful latent class distinction. Robustness testing confirmed that the *k* = 4 solution is seed-sensitive (BIC *SD* = 304.1 across ten random seeds) and yields profiles with *n* < 20 under unstandardized data, whereas the *k* = 3 solution is perfectly stable (BIC *SD* = 0.00) across all tested specifications. See Table 1 for full enumeration statistics.

---

## Reviewer-Oriented Explanation

We conducted a sensitivity analysis examining *k* = 2 through *k* = 5. The Appendix Figure compares the *k* = 3 and *k* = 4 solutions side by side. The four-profile solution was identified as containing a ceiling-response artifact: Profile P0 (*n* = 80) consists of all respondents who rated Q20_1 = 5 and Q20_2 = 5, while Profile P1 (*n* = 61) consists of respondents with Q20_1 = 5 and Q20_2 < 5. This bifurcation is mechanically determined by response ceiling concentration (37.4% at Q20_1 maximum) rather than representing a substantively distinct latent class. We confirmed this interpretation by demonstrating that: (1) the k = 4 solution is seed-sensitive (BIC ranging from -489 to -1,133 across ten seeds under raw data), while k = 3 is perfectly stable (BIC SD = 0.00); (2) removing ceiling cases (n = 223, 59.2% of sample) reduces the viable solution to k = 2 only, confirming that the multi-profile structure beyond k = 3 is driven by ceiling-response heterogeneity; and (3) the k = 4 "additional" profile has no theoretical warrant in SDT-EVT or cognate AI expectancy theories. We therefore adopt k = 3 as the primary solution and present k = 4 here for full methodological transparency.

---

## Files
- PNG: `appendix/k4_ceiling_artifact_plot.png` (300 dpi, ~411 KB)
- PDF: `appendix/k4_ceiling_artifact_plot.pdf` (vector, ~39 KB)
- Code: `code/generate_k4_appendix_plot.py`
