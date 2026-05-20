"""
Publication-ready LPA Profile Plot (k=3 solution)
GIQ/SSCI journal style — grayscale compatible

Profile data from:
- 36_lpa_sensitivity.md (exact profile means)
- 39_lpa_integrated_assessment.md (k=3 profile characterization)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Output paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── k=3 Profile data (back-transformed from standardized; raw 1–5 scale) ─────
# Source: 36_lpa_sensitivity.md §2.2 + 39_lpa_integrated_assessment.md Table
profiles = {
    "Profile 1: Generally Low\nExpectancy (n = 75, 19.9%)": {
        "means": [2.733, 2.933, 2.440],
        "linestyle": ":",
        "marker": "s",
        "color": "#555555",     # dark gray
        "mfc": "white",
    },
    "Profile 2: Efficiency-Oriented,\nAutomation-Resistant (n = 161, 42.7%)": {
        "means": [4.000, 3.578, 2.503],
        "linestyle": "--",
        "marker": "^",
        "color": "#222222",     # near-black
        "mfc": "#222222",
    },
    "Profile 3: High Efficiency-Decision\nExpectancy (n = 141, 37.4%)": {
        "means": [5.000, 4.355, 2.844],
        "linestyle": "-",
        "marker": "o",
        "color": "#000000",     # black
        "mfc": "#000000",
    },
}

x_labels = [
    "Q20_1\nEfficiency\nExpectancy",
    "Q20_2\nDecision-Support\nExpectancy",
    "Q20_3\nAutomation\nExpectancy",
]
x_pos = [0, 1, 2]

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for label, pdata in profiles.items():
    ax.plot(
        x_pos,
        pdata["means"],
        linestyle=pdata["linestyle"],
        marker=pdata["marker"],
        color=pdata["color"],
        markerfacecolor=pdata["mfc"],
        markeredgecolor=pdata["color"],
        markersize=9,
        linewidth=1.8,
        label=label,
    )
    # annotate each point
    offsets = [(-0.12, 0.10), (-0.12, 0.10), (-0.12, 0.10)]
    for xi, yi, (dx, dy) in zip(x_pos, pdata["means"], offsets):
        ax.annotate(
            f"{yi:.2f}",
            xy=(xi, yi),
            xytext=(xi + dx, yi + dy),
            fontsize=8,
            color=pdata["color"],
            ha="center",
        )

# Reference line at scale midpoint
ax.axhline(3.0, color="gray", linewidth=0.8, linestyle="-.", alpha=0.6,
           label="Scale midpoint (3.0)")

# Shade the automation column to highlight the gap
ax.axvspan(1.55, 2.45, alpha=0.06, color="gray")
ax.text(2, 1.35, "Automation\n(universally low)",
        ha="center", va="bottom", fontsize=7.5, color="gray", style="italic")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_ylabel("Mean Score (1–5 scale)", fontsize=10)
ax.set_xlabel("AI Expectancy Dimension", fontsize=10)
ax.set_ylim(1.0, 5.6)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_xlim(-0.4, 2.4)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=9)

# ── Legend ────────────────────────────────────────────────────────────────────
legend = ax.legend(
    loc="upper right",
    fontsize=8.2,
    frameon=True,
    framealpha=0.9,
    edgecolor="gray",
    title="Latent Profile",
    title_fontsize=8.5,
    handlelength=2.5,
)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Figure 1. Latent Profile Structures of AI Expectancy\n"
    "Among Korean Public-Sector AI Users (N = 377, k = 3)",
    fontsize=10.5,
    pad=12,
    loc="left",
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
png_path = os.path.join(FIG_DIR, "lpa_profile_plot.png")
pdf_path = os.path.join(FIG_DIR, "lpa_profile_plot.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# ── Manuscript caption & interpretation ───────────────────────────────────────
CAPTION = """
Figure Caption (manuscript-ready):
────────────────────────────────────────────────────────────────────────────────
Figure 1. Latent Profile Structures of AI Expectancy Among Korean
Public-Sector AI Users (N = 377).

Note. Three latent profiles were identified via Gaussian mixture modeling with
diagonal covariance (analogous to LPA; k = 3, entropy = 1.00). Profile labels
reflect dominant expectancy patterns: Profile 1 = Generally Low Expectancy
(n = 75, 19.9%); Profile 2 = Efficiency-Oriented, Automation-Resistant
(n = 161, 42.7%); Profile 3 = High Efficiency-Decision Expectancy (n = 141,
37.4%). The shaded region highlights Q20_3 (Automation Expectancy), which
remained the lowest-rated dimension across all three profiles (range: 2.44–2.84),
indicating a universal automation skepticism that transcends profile membership.
Scores are on a 1–5 Likert scale. The dashed horizontal line marks the scale
midpoint (3.0).
────────────────────────────────────────────────────────────────────────────────

Interpretation Paragraph (manuscript-ready):
────────────────────────────────────────────────────────────────────────────────
Figure 1 illustrates the three-profile structure identified in the person-
centered extension. Differentiation across profiles is strongest for efficiency
(Q20_1) and decision-support (Q20_2) expectancies, which span a range of 2.27
and 1.42 points, respectively, across profiles. Automation expectancy (Q20_3),
in contrast, remains uniformly low across all profiles (range: 2.44–2.84),
reinforcing the variable-centered OLS finding that intrinsic motivation does not
significantly predict automation expectations at the aggregate level. This
pattern suggests that automation skepticism represents a structural feature of
Korean public servants' AI expectancy landscape rather than a characteristic
of any particular subgroup.
────────────────────────────────────────────────────────────────────────────────
"""
print(CAPTION)
