"""
Compute k=3 LPA demographic comparison statistics
Output: exact values for Table 3 (demographic differences across profiles)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy import stats
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "chatbot_output_selected_preprocessed.csv")

df_raw = pd.read_csv(DATA)
df = df_raw[df_raw["Q3"] == 1].copy()
print(f"N (AI users) = {len(df)}")

# Construct composite variables (same as main analysis)
df["motivation"]    = df[["Q9_3","Q9_4"]].mean(axis=1)
df["support_main"]  = df[["Q16_1","Q16_2","Q16_3","Q16_4"]].mean(axis=1)
df["effect"]        = df[["Q7_1","Q7_2","Q7_3","Q7_4","Q7_5"]].mean(axis=1)

indicators = ["Q20_1","Q20_2","Q20_3"]
dfa = df[indicators + ["motivation","support_main","effect",
                        "gender","rank_code","career_code"]].dropna()
print(f"N (complete) = {len(dfa)}")

# ── k=3 GMM (standardized, seed=42, n_init=20) ───────────────────────────────
X = dfa[indicators].values
X_z = (X - X.mean(axis=0)) / X.std(axis=0)
gmm = GaussianMixture(n_components=3, covariance_type="diag",
                      random_state=42, n_init=20)
gmm.fit(X_z)
labels = gmm.predict(X_z)
dfa = dfa.copy()
dfa["profile"] = labels

# Map to consistent ordering: Low < Mid < High by Q20_1 mean
prof_means = dfa.groupby("profile")[indicators[0]].mean().sort_values()
label_map = {old: new for new, old in enumerate(prof_means.index)}
dfa["profile"] = dfa["profile"].map(label_map)

PNAMES = {
    0: "Profile 1:\nGenerally Low",
    1: "Profile 2:\nEfficiency-Oriented,\nAutomation-Resistant",
    2: "Profile 3:\nHigh Efficiency-Decision",
}
# Verify profile assignment
print("\nProfile sizes:")
print(dfa["profile"].value_counts().sort_index())
print("\nProfile means (Q20_1):")
print(dfa.groupby("profile")["Q20_1"].mean().sort_index())

# ── Descriptive stats by profile ──────────────────────────────────────────────
vars_of_interest = indicators + ["motivation","effect","support_main",
                                  "gender","rank_code","career_code"]
print("\n\n=== PROFILE MEANS ===")
desc = dfa.groupby("profile")[vars_of_interest].agg(["mean","std"])
print(desc.to_string())

# ── Chi-square tests (categorical vars) ──────────────────────────────────────
print("\n\n=== CHI-SQUARE TESTS ===")
for var in ["gender","rank_code","career_code"]:
    ct = pd.crosstab(dfa["profile"], dfa[var])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n_total = len(dfa)
    cramers_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))
    print(f"\n{var}: chi2={chi2:.3f}, df={dof}, p={p:.3f}, Cramer's V={cramers_v:.3f}")
    print(ct.to_string())

# ── One-way ANOVA (continuous vars) ──────────────────────────────────────────
print("\n\n=== ONE-WAY ANOVA ===")
for var in ["motivation","support_main","effect"]:
    groups = [grp[var].values for _, grp in dfa.groupby("profile")]
    F, p = stats.f_oneway(*groups)
    # Eta-squared
    grand_mean = dfa[var].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total   = ((dfa[var] - grand_mean)**2).sum()
    eta2 = ss_between / ss_total
    print(f"{var}: F={F:.3f}, p={p:.4f}, eta2={eta2:.3f}")
    for i, grp in dfa.groupby("profile"):
        print(f"  Profile {i}: M={grp[var].mean():.3f}, SD={grp[var].std():.3f}, n={len(grp)}")

# ── Gender breakdown ──────────────────────────────────────────────────────────
print("\n\n=== GENDER BY PROFILE ===")
gt = pd.crosstab(dfa["profile"], dfa["gender"], margins=True)
gtp = pd.crosstab(dfa["profile"], dfa["gender"], normalize="index") * 100
print(gt.to_string())
print("\nPercentages (male=1):")
print(gtp.round(1).to_string())

# ── Career/rank means ─────────────────────────────────────────────────────────
print("\n\n=== CAREER & RANK MEANS ===")
for var in ["rank_code","career_code"]:
    grp = dfa.groupby("profile")[var].agg(["mean","std","count"])
    print(f"\n{var}:")
    print(grp.to_string())

# ── Support main across profiles ─────────────────────────────────────────────
print("\n\n=== SUPPORT_MAIN ACROSS PROFILES ===")
print(dfa.groupby("profile")["support_main"].agg(["mean","std"]).round(3))
