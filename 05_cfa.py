from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import ncx2
from semopy import Model, calc_stats

from result_utils import markdown_output


AI_FILTER = "Q3 == 1"
MOTIVATION_ITEMS = ["Q9_3", "Q9_4"]
EFFECT_ITEMS = [f"Q7_{i}" for i in range(1, 6)]


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    return "< .001" if float(value) < 0.001 else f"{float(value):.3f}"


def calc_srmr(model: Model, data: pd.DataFrame) -> float:
    implied_cov = model.calc_sigma()[0]
    observed_cov = data.cov().to_numpy()
    implied_std = np.sqrt(np.diag(implied_cov))
    observed_std = np.sqrt(np.diag(observed_cov))
    implied_cor = implied_cov / np.outer(implied_std, implied_std)
    observed_cor = observed_cov / np.outer(observed_std, observed_std)
    tril_idx = np.tril_indices_from(observed_cor)
    return float(np.sqrt(np.mean((observed_cor[tril_idx] - implied_cor[tril_idx]) ** 2)))


def rmsea_ci_90(chi2: float, df: int, n: int) -> tuple[float, float]:
    """Noncentral chi-square 90% CI for RMSEA."""
    if df <= 0 or n <= 1:
        return np.nan, np.nan
    denom = df * (n - 1)

    def cdf(ncp: float) -> float:
        return float(ncx2.cdf(chi2, df, ncp))

    lower = 0.0
    if cdf(0.0) > 0.95:
        hi = max(1.0, chi2)
        while cdf(hi) > 0.95:
            hi *= 2
        lower = brentq(lambda ncp: cdf(ncp) - 0.95, 0.0, hi)

    hi = max(1.0, chi2)
    while cdf(hi) > 0.05:
        hi *= 2
    upper = brentq(lambda ncp: cdf(ncp) - 0.05, 0.0, hi)
    return float(np.sqrt(lower / denom)), float(np.sqrt(upper / denom))


def fit_summary(model_desc: str, data: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = Model(model_desc)
    model.fit(data)
    fit = calc_stats(model)
    est = model.inspect(std_est=True)
    chi2 = float(fit.loc["Value", "chi2"])
    dof = int(fit.loc["Value", "DoF"])
    rmsea_lower, rmsea_upper = rmsea_ci_90(chi2, dof, len(data))
    table = pd.DataFrame(
        [
            {
                "Model": label,
                "N": int(len(data)),
                "chi-square": chi2,
                "df": dof,
                "p-value": format_p(fit.loc["Value", "chi2 p-value"]),
                "CFI": float(fit.loc["Value", "CFI"]),
                "TLI": float(fit.loc["Value", "TLI"]),
                "RMSEA": float(fit.loc["Value", "RMSEA"]),
                "RMSEA 90% CI": f"[{rmsea_lower:.3f}, {rmsea_upper:.3f}]",
                "SRMR": calc_srmr(model, data),
            }
        ]
    )
    return table, est


def main() -> None:
    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df = df.query(AI_FILTER).copy()

    effect_data = df[EFFECT_ITEMS].dropna().copy()
    full_data = df[MOTIVATION_ITEMS + EFFECT_ITEMS].dropna().copy()

    effect_model = "effect =~ Q7_1 + Q7_2 + Q7_3 + Q7_4 + Q7_5"
    full_model = """
    motivation =~ Q9_3 + Q9_4
    effect =~ Q7_1 + Q7_2 + Q7_3 + Q7_4 + Q7_5
    motivation ~~ effect
    """

    effect_fit, effect_est = fit_summary(effect_model, effect_data, "Supplementary CFA check (effect only)")
    full_fit, full_est = fit_summary(full_model, full_data, "Two-factor CFA (motivation, effect)")

    effect_loadings = effect_est[
        (effect_est["op"] == "~")
        & (effect_est["lval"].isin(EFFECT_ITEMS))
        & (effect_est["rval"] == "effect")
    ][["rval", "lval", "Estimate", "Est. Std", "p-value"]].copy()
    effect_loadings.columns = ["Factor", "Item", "Loading", "Std.Loading", "p-value"]
    effect_loadings["p-value"] = effect_loadings["p-value"].map(format_p)

    with markdown_output("05_cfa.md") as result_path:
        print("# 05 Confirmatory Factor Analysis\n")
        print("- Sample: AI users only (`Q3 == 1`)")
        print("- Analysis design of the paper is OLS + bootstrap, not SEM.")
        print("- CFA is reported as supplementary measurement evidence, not as the primary basis of analytic validity.")
        print("- Excluded from CFA:")
        print("  - `support_main`: treated as an observed index, not a latent construct")
        print("  - `Q20_1~Q20_4`: treated as item-level outcomes, not a single latent scale\n")

        print("## Supplementary CFA check: effect factor only\n")
        print(effect_fit.round(3).to_markdown(index=False))
        print()

        print("## Standardized factor loadings for effect\n")
        print(effect_loadings.round({"Loading": 3, "Std.Loading": 3}).to_markdown(index=False))
        print()

        print("## Supplementary CFA: full reflective block (motivation + effect)\n")
        print(full_fit.round(3).to_markdown(index=False))
        print()

        print("## Interpretation\n")
        print("- Primary validity evidence in the paper rests on EFA, CR, AVE, Fornell-Larcker, and HTMT criteria.")
        print(
            f"- The supplementary two-factor CFA showed poor absolute residual fit (`SRMR = {float(full_fit.loc[0, 'SRMR']):.3f}`) and is not used as confirmatory evidence for the full measurement block."
        )
        print(
            f"- The one-factor CFA for `effect` is retained only as a limited supplementary check: `CFI = {float(effect_fit.loc[0, 'CFI']):.3f}`, `TLI = {float(effect_fit.loc[0, 'TLI']):.3f}`, `RMSEA = {float(effect_fit.loc[0, 'RMSEA']):.3f}` with 90% CI {effect_fit.loc[0, 'RMSEA 90% CI']}, and `SRMR = {float(effect_fit.loc[0, 'SRMR']):.3f}`."
        )
        print("- Because CFI/TLI and RMSEA do not meet conventional fit criteria, CFA is not framed as decisive support for unidimensionality; it is reported transparently as a limitation.")
        print("- `support_main` remains an observed organizational-context index and is not treated as a latent variable.")

    print(f"Completed: {result_path}")


if __name__ == "__main__":
    main()
