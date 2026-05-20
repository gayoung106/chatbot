# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statistical analysis pipeline for a survey study on generative AI adoption among Korean public servants. Target journal: **Government Information Quarterly (GIQ)**.

- Full sample: N = 1,608
- Analysis sample: AI users only (Q3 == 1), N = 377
- Theoretical framework: SDT (Self-Determination Theory) + EVT (Expectancy-Value Theory)
- Analysis approach: HC3 robust OLS regression + BCa bootstrap parallel mediation

### Research Model (Current Final Design)

```
motivation (Q9_3, Q9_4 mean)
    |
    ├──────────────────────────────► Q20_1 (work efficiency expectation)
    │                                Q20_2 (decision-making expectation)
    │                                Q20_3 (automation expectation)
    ▼
  effect (Q7_1~Q7_5 mean) ────────► Q20_1, Q20_2, Q20_3
    ▲
    │
support_main (Q16_1~Q16_4 mean) ──► Q20_1, Q20_2, Q20_3

Supplementary DV: Q20_4 (job replacement perception)
Controls: gender, rank_code, career_code
```

**Key design decisions:**

- `support_main` is a parallel independent variable (NOT a moderator)
- Q20_1~Q20_3 are item-level outcomes (NOT a composite scale)
- Q20_4 is supplementary only
- `support_main` is treated as an observed index, not a latent construct

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Flow

```
chatbot_input.SAV
    → 01_convert.py
    → chatbot_output.csv
    → 02_preprocessed.py
    → chatbot_output_selected_preprocessed.csv  ← all analysis scripts use this
    → analysis scripts (03~34)
    → result/*.md
```

**Primary data file for all analyses:** `chatbot_output_selected_preprocessed.csv`

**AI user filter:** Always apply `df[df['Q3'] == 1]` before analysis (N = 377)

---

## Key Variables

| Role      | Variable       | Construction      | Notes                                             |
| --------- | -------------- | ----------------- | ------------------------------------------------- |
| IV1       | `motivation`   | mean(Q9_3, Q9_4)  | Intrinsic motivation; inter-item r = .71          |
| IV2       | `support_main` | mean(Q16_1~Q16_4) | Organizational AI support climate; observed index |
| Mediator  | `effect`       | mean(Q7_1~Q7_5)   | Perceived work effectiveness                      |
| DV (main) | `Q20_1`        | single item       | Work efficiency expectation                       |
| DV (main) | `Q20_2`        | single item       | Decision-making support expectation               |
| DV (main) | `Q20_3`        | single item       | Routine task automation expectation               |
| DV (supp) | `Q20_4`        | single item       | Job replacement perception                        |
| Control   | `gender`       | 0=female, 1=male  |                                                   |
| Control   | `rank_code`    | 1~7               | Higher = higher rank                              |
| Control   | `career_code`  | 1~5               |                                                   |

**Excluded variables:**

- Q16_5~Q16_7: excluded from `support_main` (outcome indicators or personal-level content)
- Q9_1, Q9_2: external regulation items, excluded from `motivation`
- Q20 composite: not used as main DV

---

## Analysis Scripts (Current Pipeline)

Scripts are numbered by analysis stage. Run sequentially where dependencies exist.

```bash
python 01_convert.py                    # SPSS .sav → CSV
python 02_preprocessed.py              # cleaning & variable construction
python 03_analysis_ai_group.py         # descriptive stats (AI users only)
python 05_cfa.py                       # supplementary CFA (effect factor only)
python 09_cmv_harman.py                # Harman single-factor CMV check
python 10_cmv_marker_proxy.py          # Lindell-Whitney marker-proxy sensitivity
python 15_compare_ai_users_nonusers.py # Appendix Table A1: user vs non-user comparison
python 16_ai_use_selection_model.py    # selection bias check (logistic)
python 19_parallel_mediation_hc3_bca.py # MAIN ANALYSIS: parallel mediation
python 28_item_level_expectancy_models.py # item-level OLS per Q20_k
python 31_paper_ready_tables.py        # manuscript-ready tables
python 33_supplementary_group_analysis.py # supplementary group analyses
python 34_lpa_analysis.py              # LPA: latent profiles of AI expectancy
```

**Note:** Some scripts in CLAUDE.md legacy version (04, 06~08, 10~14) are from an earlier moderation-based design and are no longer part of the main pipeline.

---

## Regression Approach

All main models use:

- **HC3 heteroskedasticity-robust standard errors** (statsmodels `cov_type='HC3'`)
- **BCa bootstrap** (5,000 resamples) for indirect effects
- **Holm-Bonferroni correction** for multiple DVs (Q20_1~Q20_3)

### Core regression equations

```python
# Mediator model (a-path)
effect ~ motivation + support_main + gender + rank_code + career_code

# Total effect model
Q20_k ~ motivation + support_main + gender + rank_code + career_code

# Direct effect model (b-path + c'-path)
Q20_k ~ motivation + support_main + effect + gender + rank_code + career_code
# where k = 1, 2, 3
```

---

## Output Files

All result files are saved to `result/` directory as `.md` files.

| File                                        | Content                       |
| ------------------------------------------- | ----------------------------- |
| `result/03_analysis_ai_group.md`            | Descriptive statistics        |
| `result/05_cfa.md`                          | Supplementary CFA results     |
| `result/09_cmv_harman.md`                   | CMV Harman test               |
| `result/10_cmv_marker_proxy.md`             | CMV marker-proxy sensitivity  |
| `result/15_compare_ai_users_nonusers.md`    | User vs non-user comparison   |
| `result/16_ai_use_selection_model.md`       | Selection model               |
| `result/19_parallel_mediation_hc3_bca.md`   | **Main mediation results**    |
| `result/28_item_level_expectancy_models.md` | Item-level OLS results        |
| `result/31_paper_ready_tables.md`           | **Manuscript-ready tables**   |
| `result/34_lpa_analysis.md`                 | LPA results (latent profiles) |

---

## Sub-agents (.claude/agents/)

| Agent                          | Role                                   |
| ------------------------------ | -------------------------------------- |
| `supervisor.md`                | Orchestrates multi-agent tasks         |
| `giq_editor.md`                | Manuscript writing and editing         |
| `giq_method_reviewer.md`       | Methods section review                 |
| `giq_theory_reviewer.md`       | Theory section review                  |
| `giq_contribution_reviewer.md` | Contribution and discussion review     |
| `lpa_analyst.md`               | **LPA execution and profile analysis** |

### lpa_analyst responsibilities

- Fit LPA (k=2~5) on Q20_1, Q20_2, Q20_3
- Select optimal k using BIC, entropy, BLRT
- Profile characterization and naming
- Demographic predictors via multinomial logistic regression
- Exploratory within-profile OLS (if N permits)
- Output to `result/34_lpa_analysis.md` and `result/34_lpa_analysis.py`

---

## Important Constraints

1. **Always filter AI users**: `df[df['Q3'] == 1]` before any analysis
2. **Do not treat Q20 as composite**: analyze Q20_1, Q20_2, Q20_3 separately
3. **Do not use Q16_5~Q16_7** in `support_main`
4. **support_main is NOT a moderator**: it is a parallel IV alongside motivation
5. **CFA is supplementary only**: main validity evidence is EFA + CR + AVE + Fornell-Larcker + HTMT
6. **Q20_4 is supplementary only**: do not include in main hypothesis tests
7. **LPA profiles**: minimum N per profile should be ≥ 50; flag if smaller

---

## Manuscript Status

Current draft files in project root:

- `02_manuscript_revision_draft.md` — methods and results draft (Korean)
- `24_model_and_methods_for_manuscript.md` — methods section
- `32_theoretical_integration_sdt_evt.md` — theory section
- `30_research_model_diagram.md` — research model and hypotheses
- `31_paper_ready_tables.md` — all tables

Writing language: **Korean** (to be translated to English for GIQ submission)
