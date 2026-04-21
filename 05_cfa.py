from __future__ import annotations

import numpy as np
import pandas as pd
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


def fit_summary(model_desc: str, data: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = Model(model_desc)
    model.fit(data)
    fit = calc_stats(model)
    est = model.inspect(std_est=True)
    table = pd.DataFrame(
        [
            {
                "Model": label,
                "N": int(len(data)),
                "chi-square": float(fit.loc["Value", "chi2"]),
                "df": int(fit.loc["Value", "DoF"]),
                "p-value": format_p(fit.loc["Value", "chi2 p-value"]),
                "CFI": float(fit.loc["Value", "CFI"]),
                "TLI": float(fit.loc["Value", "TLI"]),
                "RMSEA": float(fit.loc["Value", "RMSEA"]),
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

    effect_fit, effect_est = fit_summary(effect_model, effect_data, "One-factor CFA (effect only)")
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

        print("## Main CFA: effect factor only\n")
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
            f"- Full CFA fit was suboptimal (CFI = {float(full_fit.loc[0, 'CFI']):.3f}, RMSEA = {float(full_fit.loc[0, 'RMSEA']):.3f}, SRMR = {float(full_fit.loc[0, 'SRMR']):.3f}), attributed in part to the 2-item motivation factor structure."
        )
        print(
            f"- The one-factor CFA for `effect` is reported as the main CFA result for the retained reflective block, but it should still be read cautiously (`CFI = {float(effect_fit.loc[0, 'CFI']):.3f}`, `RMSEA = {float(effect_fit.loc[0, 'RMSEA']):.3f}`, `SRMR = {float(effect_fit.loc[0, 'SRMR']):.3f}`)."
        )
        print("- In low-df models, RMSEA can be inflated and should not be used as the sole basis for rejecting unidimensionality (Kenny et al., 2015).")
        print("- `support_main` remains an observed organizational-context index and is not treated as a latent variable.")

    print(f"Completed: {result_path}")


if __name__ == "__main__":
    main()
