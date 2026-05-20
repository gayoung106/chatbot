"""
Appendix Figure — k=4 Ceiling Artifact Sensitivity Check
Supports methodological justification for preferring k=3

Data from: 34_lpa_analysis.md §2.1 + 35_lpa_robustness.md §2.2
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APX_DIR = os.path.join(BASE, "appendix")
os.makedirs(APX_DIR, exist_ok=True)

# ── k=4 Profile data (from 34_lpa_analysis.md §2.1) ─────────────────────────
profiles_k4 = {
    "P0: Q20_1=5 & Q20_2=5\n(n=80, 21.2%) [Ceiling artifact]": {
        "means": [5.000, 5.000, 3.180],
        "linestyle": "-",
        "marker": "o",
        "color": "#000000",
        "mfc": "#000000",
        "lw": 2.0,
    },
    "P1: Q20_1=5 & Q20_2<5\n(n=61, 16.2%) [Ceiling artifact]": {
        "means": [5.000, 3.510, 2.410],
        "linestyle": "--",
        "marker": "s",
        "color": "#000000",
        "mfc": "white",
        "lw": 2.0,
    },
    "P2: Efficiency-Moderate\n(n=161, 42.7%)": {
        "means": [4.000, 3.580, 2.500],
        "linestyle": "-.",
        "marker": "^",
        "color": "#555555",
        "mfc": "#555555",
        "lw": 1.6,
    },
    "P3: Generally Low\n(n=75, 19.9%)": {
        "means": [2.730, 2.930, 2.440],
        "linestyle": ":",
        "marker": "D",
        "color": "#888888",
        "mfc": "white",
        "lw": 1.6,
    },
}

x_labels = [
    "Q20_1\nEfficiency\nExpectancy",
    "Q20_2\nDecision-Support\nExpectancy",
    "Q20_3\nAutomation\nExpectancy",
]
x_pos = [0, 1, 2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── LEFT: k=3 reference ───────────────────────────────────────────────────────
ax_k3 = axes[0]

profiles_k3 = {
    "P1: Generally Low\n(n=75, 19.9%)": {
        "means": [2.733, 2.933, 2.440],
        "ls": ":", "mk": "s", "c": "#888888", "mfc": "white",
    },
    "P2: Efficiency-Oriented,\nAutomation-Resistant (n=161, 42.7%)": {
        "means": [4.000, 3.578, 2.503],
        "ls": "--", "mk": "^", "c": "#444444", "mfc": "#444444",
    },
    "P3: High Efficiency-Decision\n(n=141, 37.4%)": {
        "means": [5.000, 4.355, 2.844],
        "ls": "-", "mk": "o", "c": "#000000", "mfc": "#000000",
    },
}

for label, pd in profiles_k3.items():
    ax_k3.plot(x_pos, pd["means"], linestyle=pd["ls"], marker=pd["mk"],
               color=pd["c"], markerfacecolor=pd["mfc"],
               markeredgecolor=pd["c"], markersize=8, linewidth=1.8,
               label=label)
    for xi, yi in zip(x_pos, pd["means"]):
        ax_k3.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(xi, yi + 0.12),
                       fontsize=7.5, ha="center", color=pd["c"])

ax_k3.axhline(3.0, color="gray", linewidth=0.7, linestyle="-.", alpha=0.5)
ax_k3.set_title("(a) k = 3 Solution (Primary)\nThree substantively distinct profiles",
                fontsize=10, pad=8)
ax_k3.set_xticks(x_pos)
ax_k3.set_xticklabels(x_labels, fontsize=9)
ax_k3.set_ylabel("Mean Score (1–5 scale)", fontsize=9.5)
ax_k3.set_ylim(1.0, 5.8)
ax_k3.set_yticks([1, 2, 3, 4, 5])
ax_k3.set_xlim(-0.4, 2.4)
ax_k3.spines["top"].set_visible(False)
ax_k3.spines["right"].set_visible(False)
ax_k3.legend(fontsize=7.8, loc="lower right", framealpha=0.9, edgecolor="gray")
ax_k3.tick_params(labelsize=8.5)

# Bracket showing that k=3's High profile = k=4's P0+P1
ax_k3.annotate("", xy=(0, 5.60), xytext=(0, 4.90),
               arrowprops=dict(arrowstyle="<->", color="#CC0000", lw=1.5))
ax_k3.text(0.08, 5.25, "k=4 splits\nthis group\n(artifact)",
           fontsize=7, color="#CC0000", style="italic")

# ── RIGHT: k=4 showing artifact ───────────────────────────────────────────────
ax_k4 = axes[1]

for label, pdata in profiles_k4.items():
    is_artifact = "artifact" in label
    ax_k4.plot(
        x_pos, pdata["means"],
        linestyle=pdata["linestyle"],
        marker=pdata["marker"],
        color=pdata["color"] if not is_artifact else "#CC0000",
        markerfacecolor=pdata["mfc"] if not is_artifact else "#CC0000",
        markeredgecolor=pdata["color"] if not is_artifact else "#CC0000",
        markersize=9,
        linewidth=pdata["lw"],
        label=label,
        alpha=1.0 if is_artifact else 0.7,
    )
    col = "#CC0000" if is_artifact else pdata["color"]
    for xi, yi in zip(x_pos, pdata["means"]):
        ax_k4.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(xi, yi + 0.12),
                       fontsize=7.5, ha="center", color=col)

# Highlight the Q20_2=5 split point
ax_k4.axhline(5.0, color="#CC0000", linewidth=0.7, linestyle="--", alpha=0.4)
ax_k4.annotate("Q20_2 ceiling (score = 5)",
               xy=(1, 5.0), xytext=(0.5, 5.35),
               arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.0),
               fontsize=7.5, color="#CC0000")

# Annotation box
ax_k4.text(0.5, 1.55,
           "P0 ∩ P1 split criterion:\nQ20_2 = 5 vs. Q20_2 < 5\n"
           "(response ceiling partition,\nnot a latent construct)",
           fontsize=7.5, color="#CC0000", style="italic",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF0F0",
                     edgecolor="#CC0000", alpha=0.85),
           ha="center")

ax_k4.axhline(3.0, color="gray", linewidth=0.7, linestyle="-.", alpha=0.5)
ax_k4.set_title("(b) k = 4 Solution (Sensitivity Check)\n"
                "P0 & P1 are ceiling-response artifacts (shown in red)",
                fontsize=10, pad=8, color="black")
ax_k4.set_xticks(x_pos)
ax_k4.set_xticklabels(x_labels, fontsize=9)
ax_k4.set_ylabel("Mean Score (1–5 scale)", fontsize=9.5)
ax_k4.set_ylim(1.0, 5.8)
ax_k4.set_yticks([1, 2, 3, 4, 5])
ax_k4.set_xlim(-0.4, 2.4)
ax_k4.spines["top"].set_visible(False)
ax_k4.spines["right"].set_visible(False)
ax_k4.legend(fontsize=7.5, loc="lower right", framealpha=0.9, edgecolor="gray")
ax_k4.tick_params(labelsize=8.5)

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Appendix Figure. Comparison of k=3 (Primary) and k=4 (Sensitivity Check) Solutions\n"
    "LPA of AI Expectancy Among Korean Public-Sector AI Users (N = 377)",
    fontsize=10.5, y=1.01
)

plt.tight_layout()

png_path = os.path.join(APX_DIR, "k4_ceiling_artifact_plot.png")
pdf_path = os.path.join(APX_DIR, "k4_ceiling_artifact_plot.pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

APPENDIX_TEXT = """
Appendix Figure Note (manuscript-ready):
────────────────────────────────────────────────────────────────────────────────
Appendix Figure. k=3 (Primary) and k=4 (Sensitivity Check) LPA Solutions.

Note. Panel (a) displays the three-profile solution selected as the primary
result. Panel (b) displays the four-profile solution for comparison. In the
k=4 solution, Profiles P0 and P1 (shown in red) are distinguished exclusively
by whether respondents rated Q20_2 (Decision-Support Expectancy) at the
response ceiling (score = 5) or below. This partition—where P0 = {Q20_1=5
AND Q20_2=5} (n=80) and P1 = {Q20_1=5 AND Q20_2<5} (n=61)—constitutes a
ceiling-response artifact rather than a theoretically interpretable latent
class. Robustness testing confirmed that the k=4 solution is seed-sensitive
(BIC σ = 304.1 across ten random seeds) and yields profiles with n < 20 under
raw (unstandardized) data, whereas k=3 is perfectly stable (BIC σ = 0.00)
across all tested specifications. These findings support the selection of k=3
as the primary solution.
────────────────────────────────────────────────────────────────────────────────

Reviewer-oriented interpretation:
────────────────────────────────────────────────────────────────────────────────
The k=4 sensitivity analysis was conducted to assess whether an additional
profile provided theoretically meaningful differentiation beyond k=3. We found
that the additional profile in k=4 represents a ceiling-response partition
(cases where both Q20_1 and Q20_2 were rated at the maximum score of 5) rather
than a qualitatively distinct expectancy configuration. Specifically, k=4
Profile P0 (n=80) consists entirely of respondents who scored 5 on both Q20_1
and Q20_2, while Profile P1 (n=61) consists of respondents who scored 5 on
Q20_1 but below 5 on Q20_2. This mechanical bifurcation was further confirmed
by the solution's seed instability (BIC range: −489 to −1,133 across raw-data
seeds) and violation of the minimum profile size criterion (n < 20) under
unstandardized data. We therefore report k=3 as the primary solution and
present k=4 here as a methodological transparency check.
────────────────────────────────────────────────────────────────────────────────
"""
print(APPENDIX_TEXT)
