# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statistical analysis pipeline for a survey study on AI chatbot (ChatGPT) adoption behavior. The study uses SPSS survey data (N=1608, AI users subset N=377) to examine voluntary motivation → perceived work effectiveness → strategic expectation, with organizational support as a moderator.

## Setup

```bash
pip install -r requirements.txt
```

A `.venv` directory exists in the repo root.

## Running the Pipeline

Scripts must be run sequentially — each step depends on output from prior steps:

```bash
python 01_convert.py           # SPSS .sav → CSV
python 02_preprocessed.py      # cleaning & preprocessing
python 03_analysis_ai_group.py # descriptive stats (AI vs non-AI)
python 04_regression_analysis_ai_users.py  # main hierarchical regression
python 05_cfa.py               # confirmatory factor analysis (semopy)
python 06_cfa_subdimension.py  # CFA by subdimension
python 07_moderation_analysis.py
python 08_moderation_effect_path.py  # bootstrap mediation (5000 iterations)
python 09_cmv_harman.py        # common method variance check
python 10_moderation_a_path.py
python 11_support_motivation.py
python 12_compare_two_ivs.py
python 13_robustness_q16_7.py
python 14_bootstrap_support.py
```

To test passive motivation mediation: `python test_passive.py`

## Architecture

**Data flow:** `chatbot_input.SAV` → `chatbot_output.csv` → `chatbot_output_selected_preprocessed.csv` → analysis scripts → `.md`/`.txt` result files

**Key variables (column names in preprocessed CSV):**
- IV: `Q9_3`, `Q9_4` (voluntary AI motivation)
- Mediator: `Q7_1`–`Q7_5` (perceived work effectiveness)
- Moderator: `Q16_1`–`Q16_7` (organizational support)
- DV: `Q20_1`–`Q20_4` (strategic expectation)
- Controls: gender, rank, career (demographic)

**Validity checks** (`05_measurement_validity.py`): Cronbach's alpha, EFA, CFA, KMO, Bartlett's test

**Regression approach:** Hierarchical OLS with HC3 heteroskedasticity-robust standard errors (statsmodels); VIF via variance_inflation_factor

**Mediation:** Sobel test + bootstrap resampling (5,000 iterations) for indirect effects

**Moderation:** Interaction terms + simple slopes; Q16 subdimensions tested separately for robustness (scripts 13–14)

**SEM:** semopy used for CFA; factor_analyzer used for EFA

## Output Files

Results are written as `.md` and `.txt` files alongside the scripts. Each analysis script prints results and saves them to a named file (e.g., `test_passive_results.txt`, various `*_results.md`).
