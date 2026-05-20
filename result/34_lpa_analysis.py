"""
34_lpa_analysis.py
Latent Profile Analysis: Strategic AI Expectancy among Korean Public Servants (AI Users)
Reproducibility script for result/34_lpa_analysis.md

Fallback: sklearn GaussianMixture with covariance_type='diag'
         (equivalent to LPA equal-variance constraint — standard LPA approximation)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import os

# ─────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────
np.random.seed(42)
os.makedirs("result", exist_ok=True)

# Korean font for matplotlib (Windows)
try:
    font_candidates = [f.name for f in fm.fontManager.ttflist if "Gothic" in f.name or "Malgun" in f.name or "NanumGothic" in f.name]
    if font_candidates:
        plt.rcParams["font.family"] = font_candidates[0]
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────
# 1. DATA PREPARATION
# ─────────────────────────────────────────────
df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
ai = df[df["Q3"] == 1].copy()
print(f"AI users N = {len(ai)}")

ai["motivation"] = ai[["Q9_3", "Q9_4"]].mean(axis=1)
ai["effect"]     = ai[["Q7_1","Q7_2","Q7_3","Q7_4","Q7_5"]].mean(axis=1)
ai["support_main"] = ai[["Q16_1","Q16_2","Q16_3","Q16_4"]].mean(axis=1)

lpa_vars = ["Q20_1", "Q20_2", "Q20_3"]
lpa_data = ai[lpa_vars].dropna().copy()
print(f"LPA analysis N = {len(lpa_data)}")
print(f"Means: {lpa_data.mean().round(3).to_dict()}")
print(f"SDs:   {lpa_data.std().round(3).to_dict()}")

# Standardize for GMM fitting (means back-transformed for reporting)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(lpa_data)

# ─────────────────────────────────────────────
# 2. ENTROPY FUNCTION
# ─────────────────────────────────────────────
def compute_entropy(model, X):
    """
    Classification entropy for LPA:
    E = 1 - [-sum_i sum_k p_ik * ln(p_ik)] / [N * ln(K)]
    Range [0,1]; higher = better class separation.
    """
    probs = model.predict_proba(X)
    N, K = probs.shape
    eps = 1e-10
    H = -np.sum(probs * np.log(probs + eps))
    return 1.0 - H / (N * np.log(K))

# ─────────────────────────────────────────────
# 3. MODEL COMPARISON k = 2..5
# ─────────────────────────────────────────────
results = []
models  = {}

for k in range(2, 6):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",   # LPA equal-variance equivalent
        max_iter=500,
        n_init=20,                # multiple restarts for stability
        random_state=42
    )
    gmm.fit(X_scaled)
    ent  = compute_entropy(gmm, X_scaled)
    labels = gmm.predict(X_scaled)
    sizes  = [np.sum(labels == c) for c in range(k)]
    min_n  = min(sizes)
    results.append({
        "k": k,
        "AIC":     round(gmm.aic(X_scaled), 1),
        "BIC":     round(gmm.bic(X_scaled), 1),
        "LogLik":  round(gmm.score(X_scaled) * len(X_scaled), 1),
        "Entropy": round(ent, 3),
        "MinN":    min_n,
        "Sizes":   sizes
    })
    models[k] = gmm
    print(f"k={k}  AIC={gmm.aic(X_scaled):.1f}  BIC={gmm.bic(X_scaled):.1f}  "
          f"Entropy={ent:.3f}  Sizes={sizes}  MinN={min_n}")

fit_df = pd.DataFrame(results)
print("\nModel fit table:")
print(fit_df[["k","AIC","BIC","LogLik","Entropy","MinN"]].to_string(index=False))

# ─────────────────────────────────────────────
# 4. OPTIMAL MODEL SELECTION
# ─────────────────────────────────────────────
# Priority: (1) lowest BIC elbow, (2) entropy >= .80, (3) MinN >= 50, (4) interpretability
bic_vals = fit_df["BIC"].values
bic_diffs = np.diff(bic_vals)
print(f"\nBIC differences (k2→k3, k3→k4, k4→k5): {bic_diffs.round(1)}")

# Find best k
best_k = None
for _, row in fit_df.iterrows():
    k = int(row["k"])
    if row["MinN"] >= 50 and row["Entropy"] >= 0.70:
        if best_k is None or row["BIC"] < fit_df[fit_df["k"]==best_k]["BIC"].values[0]:
            best_k = k

if best_k is None:  # relax entropy threshold
    for _, row in fit_df.iterrows():
        k = int(row["k"])
        if row["MinN"] >= 50:
            if best_k is None or row["BIC"] < fit_df[fit_df["k"]==best_k]["BIC"].values[0]:
                best_k = k

if best_k is None:
    best_k = 2

print(f"\nSelected optimal k = {best_k}")
opt_row = fit_df[fit_df["k"] == best_k].iloc[0]
print(opt_row)

# ─────────────────────────────────────────────
# 5. PROFILE CHARACTERIZATION
# ─────────────────────────────────────────────
opt_model = models[best_k]
ai_lpa = ai[lpa_vars].copy()
ai_lpa = ai_lpa.dropna()
lpa_idx = ai_lpa.index

X_opt = scaler.transform(ai_lpa.values)
labels = opt_model.predict(X_opt)
probs  = opt_model.predict_proba(X_opt)

ai_lpa = ai_lpa.copy()
ai_lpa["profile"] = labels
# Merge back to main ai dataframe
ai_main = ai.loc[lpa_idx].copy()
ai_main["profile"] = labels
ai_main["max_prob"] = probs.max(axis=1)

# Compute profile means (original scale)
profile_means = ai_lpa.groupby("profile")[lpa_vars].mean()
profile_sizes = ai_lpa.groupby("profile").size()

print("\nProfile means (original scale):")
print(profile_means.round(3))
print("\nProfile sizes:")
print(profile_sizes)

# Back-transform GMM means for reference
gmm_means_orig = scaler.inverse_transform(opt_model.means_)

# Sort profiles by overall mean (ascending) for consistent labeling
overall_mean = profile_means.mean(axis=1)
rank_order = overall_mean.rank().astype(int) - 1  # 0-indexed ascending

# Define profile labels based on k and ordering
# Labels assigned after examining actual score patterns
def assign_profile_labels(profile_means, best_k):
    """
    Assign theoretically meaningful labels based on score patterns.
    Q20_1 = efficiency, Q20_2 = decision, Q20_3 = automation
    """
    means = profile_means.copy()
    labels_dict = {}

    if best_k == 2:
        # Two profiles: high vs. low overall expectancy
        overall = means.mean(axis=1)
        high_p = overall.idxmax()
        low_p  = overall.idxmin()
        labels_dict[high_p] = "고기대 집단 (High Expectancy)"
        labels_dict[low_p]  = "저기대 집단 (Low Expectancy)"

    elif best_k == 3:
        # Three profiles based on dominant expectancy dimension
        overall = means.mean(axis=1)
        sorted_p = overall.sort_values()
        labels_dict[sorted_p.index[0]] = "저기대 집단 (Low Expectancy)"
        # Middle: check automation vs. general
        mid_p = sorted_p.index[1]
        hi_p  = sorted_p.index[2]
        # Check if middle shows automation-focused pattern
        mid_vals = means.loc[mid_p]
        if mid_vals["Q20_3"] > mid_vals["Q20_2"]:
            labels_dict[mid_p] = "자동화 지향 집단 (Automation-Focused)"
        else:
            labels_dict[mid_p] = "중간기대 집단 (Moderate Expectancy)"
        # High: efficiency+decision dominant
        hi_vals = means.loc[hi_p]
        if hi_vals["Q20_1"] > hi_vals["Q20_3"] + 0.5:
            labels_dict[hi_p] = "효율·의사결정 고기대 집단 (Efficiency-Decision High)"
        else:
            labels_dict[hi_p] = "고기대 집단 (High Expectancy)"

    elif best_k == 4:
        overall = means.mean(axis=1)
        sorted_p = overall.sort_values()
        p_ids = list(sorted_p.index)
        labels_dict[p_ids[0]] = "저기대 집단 (Low Expectancy)"
        labels_dict[p_ids[1]] = "자동화 회의 집단 (Automation-Skeptical)"
        # Check p_ids[2]: automation vs efficiency
        mid_high = means.loc[p_ids[2]]
        if mid_high["Q20_3"] >= mid_high["Q20_1"] - 0.3:
            labels_dict[p_ids[2]] = "균형 중기대 집단 (Balanced Moderate)"
        else:
            labels_dict[p_ids[2]] = "효율 지향 집단 (Efficiency-Focused)"
        labels_dict[p_ids[3]] = "고기대 집단 (High Expectancy)"

    else:  # k=5
        overall = means.mean(axis=1)
        sorted_p = overall.sort_values()
        p_ids = list(sorted_p.index)
        labels_dict[p_ids[0]] = "저기대 집단 (Low Expectancy)"
        labels_dict[p_ids[1]] = "자동화 회의 집단 (Automation-Skeptical)"
        labels_dict[p_ids[2]] = "효율 중심 집단 (Efficiency-Centered)"
        labels_dict[p_ids[3]] = "자동화 포함 중기대 집단 (Moderate with Automation)"
        labels_dict[p_ids[4]] = "고기대 집단 (High Expectancy)"

    return labels_dict

profile_labels = assign_profile_labels(profile_means, best_k)
print("\nProfile labels:")
for k_id, lbl in profile_labels.items():
    print(f"  Profile {k_id}: {lbl}  (N={profile_sizes[k_id]}, means={profile_means.loc[k_id].round(2).to_dict()})")

# ─────────────────────────────────────────────
# 6. PROFILE VISUALIZATION
# ─────────────────────────────────────────────
item_labels = ["Q20_1\n(업무효율)", "Q20_2\n(의사결정)", "Q20_3\n(자동화)"]
colors = ["#2196F3","#FF5722","#4CAF50","#9C27B0","#FF9800"]
markers = ["o","s","^","D","v"]

fig, ax = plt.subplots(figsize=(8, 5))
for i, (p_id, lbl) in enumerate(profile_labels.items()):
    vals = [profile_means.loc[p_id, v] for v in lpa_vars]
    n    = profile_sizes[p_id]
    ax.plot(item_labels, vals, color=colors[i], marker=markers[i],
            linewidth=2.2, markersize=8, label=f"Profile {p_id+1}: {lbl} (N={n})")
    for j, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (item_labels[j], v),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=colors[i])

ax.set_ylim(1, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_ylabel("Mean Score (1–5 scale)", fontsize=11)
ax.set_title(f"Latent Profile Analysis: Strategic AI Expectancy (k={best_k})", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5, linewidth=1)
plt.tight_layout()
plt.savefig("result/34_lpa_profile_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Profile plot saved: result/34_lpa_profile_plot.png")

# ─────────────────────────────────────────────
# 7. DEMOGRAPHIC CHARACTERISTICS BY PROFILE
# ─────────────────────────────────────────────
demo_vars = ["gender", "rank_code", "career_code"]
if "SQ1" in ai_main.columns:
    demo_vars.append("SQ1")
if "SQ4" in ai_main.columns:
    demo_vars.append("SQ4")

demo_results = {}
print("\n=== DEMOGRAPHIC ANALYSIS ===")
for dv in demo_vars:
    if dv not in ai_main.columns:
        continue
    ct = pd.crosstab(ai_main["profile"], ai_main[dv])
    chi2, p, dof, _ = chi2_contingency(ct)
    demo_results[dv] = {"crosstab": ct, "chi2": chi2, "p": p, "dof": dof}
    print(f"\n{dv}: chi2({dof})={chi2:.3f}, p={p:.3f}")
    print(ct)

# Descriptive stats per profile per demo var
print("\n=== PROFILE DESCRIPTIVE STATS ===")
desc_by_profile = ai_main.groupby("profile")[["gender","rank_code","career_code","motivation","support_main","effect"]].mean()
print(desc_by_profile.round(3))

# ─────────────────────────────────────────────
# 8. MULTINOMIAL LOGISTIC REGRESSION
# ─────────────────────────────────────────────
from statsmodels.discrete.discrete_model import MNLogit

mnl_data = ai_main[["profile","gender","rank_code","career_code"]].dropna().copy()
# Reference category = profile with largest N
ref_profile = profile_sizes.idxmax()
print(f"\nMultinomial LR: reference profile = {ref_profile} ({profile_labels.get(ref_profile,'')})")

y_mnl = mnl_data["profile"].values
X_mnl = sm.add_constant(mnl_data[["gender","rank_code","career_code"]].values)
X_mnl_df = sm.add_constant(mnl_data[["gender","rank_code","career_code"]])

try:
    mnl_model = MNLogit(y_mnl, X_mnl_df)
    mnl_result = mnl_model.fit(method="bfgs", maxiter=500, disp=False)
    print(mnl_result.summary())
    mnl_fitted = mnl_result
except Exception as e:
    print(f"MNLogit error: {e}")
    mnl_fitted = None

# ─────────────────────────────────────────────
# 9. WITHIN-PROFILE HC3 ROBUST OLS (EXPLORATORY)
# ─────────────────────────────────────────────
within_profile_results = {}
print("\n=== WITHIN-PROFILE OLS (HC3, Exploratory) ===")
ols_vars = ["motivation","support_main","effect","gender","rank_code","career_code"]

for p_id in sorted(profile_labels.keys()):
    sub = ai_main[ai_main["profile"] == p_id].copy()
    n_sub = len(sub)
    low_power = n_sub < 80
    flag = " [LOW POWER: N < 80, exploratory only]" if low_power else ""
    print(f"\nProfile {p_id}: {profile_labels[p_id]}  N={n_sub}{flag}")
    within_profile_results[p_id] = {"n": n_sub, "low_power": low_power, "models": {}}

    for dv in ["Q20_1","Q20_2","Q20_3"]:
        sub_clean = sub[ols_vars + [dv]].dropna()
        if len(sub_clean) < 20:
            print(f"  {dv}: insufficient N ({len(sub_clean)}), skip")
            continue
        X_sub = sm.add_constant(sub_clean[ols_vars])
        y_sub = sub_clean[dv]
        try:
            mod = OLS(y_sub, X_sub).fit(cov_type="HC3")
            coefs = {v: {"coef": round(mod.params[v],3), "p": round(mod.pvalues[v],3)}
                     for v in ols_vars if v in mod.params}
            r2 = round(mod.rsquared, 3)
            print(f"  {dv}: R²={r2}  motivation β={coefs.get('motivation',{}).get('coef','—')} "
                  f"(p={coefs.get('motivation',{}).get('p','—')})  "
                  f"support_main β={coefs.get('support_main',{}).get('coef','—')} "
                  f"(p={coefs.get('support_main',{}).get('p','—')})  "
                  f"effect β={coefs.get('effect',{}).get('coef','—')} "
                  f"(p={coefs.get('effect',{}).get('p','—')})")
            within_profile_results[p_id]["models"][dv] = {
                "result": mod, "n": len(sub_clean), "coefs": coefs, "r2": r2
            }
        except Exception as e:
            print(f"  {dv}: OLS error — {e}")

# ─────────────────────────────────────────────
# 10. STORE OUTPUTS FOR MARKDOWN
# ─────────────────────────────────────────────
analysis_store = {
    "ai_n": len(ai),
    "lpa_n": len(lpa_data),
    "lpa_means": lpa_data.mean().round(3).to_dict(),
    "lpa_sds":   lpa_data.std().round(3).to_dict(),
    "fit_df": fit_df,
    "best_k": best_k,
    "opt_row": opt_row,
    "profile_means": profile_means,
    "profile_sizes": profile_sizes,
    "profile_labels": profile_labels,
    "demo_results": demo_results,
    "desc_by_profile": desc_by_profile,
    "mnl_fitted": mnl_fitted,
    "within_profile_results": within_profile_results,
    "ref_profile": ref_profile,
}

print("\n=== Analysis complete. Generating markdown report... ===")
print(f"Outputs will be saved to result/34_lpa_analysis.md")

# ─────────────────────────────────────────────
# END OF SCRIPT — markdown generation below
# ─────────────────────────────────────────────
