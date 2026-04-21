from __future__ import annotations

import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf

from result_utils import markdown_output


AI_FILTER = "Q3 == 1"
MOTIVATION_ITEMS = ["Q9_3", "Q9_4"]
EFFECT_ITEMS = [f"Q7_{i}" for i in range(1, 6)]
SUPPORT_ITEMS = [f"Q16_{i}" for i in range(1, 5)]
MARKER = "Q20_4"
MAIN_DVS = ["Q20_1", "Q20_2", "Q20_3"]


def main() -> None:
    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df = df.query(AI_FILTER).copy()
    df["motivation"] = df[MOTIVATION_ITEMS].mean(axis=1)
    df["effect"] = df[EFFECT_ITEMS].mean(axis=1)
    df["support_main"] = df[SUPPORT_ITEMS].mean(axis=1)

    corr_rows = []
    pairs = [
        ("motivation", "effect"),
        ("motivation", "support_main"),
        ("effect", "support_main"),
        ("motivation", "Q20_1"),
        ("motivation", "Q20_2"),
        ("motivation", "Q20_3"),
        ("support_main", "Q20_1"),
        ("support_main", "Q20_2"),
        ("support_main", "Q20_3"),
        ("effect", "Q20_1"),
        ("effect", "Q20_2"),
        ("effect", "Q20_3"),
    ]
    for x, y in pairs:
        sub = df[[x, y, MARKER]].dropna()
        zero = pg.corr(sub[x], sub[y], method="pearson").iloc[0]
        partial = pg.partial_corr(data=sub, x=x, y=y, covar=MARKER, method="pearson").iloc[0]
        corr_rows.append(
            {
                "x": x,
                "y": y,
                "zero-order r": float(zero["r"]),
                "partial_r_marker": float(partial["r"]),
                "abs_delta_r": abs(float(partial["r"]) - float(zero["r"])),
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    reg_rows = []
    for dv in MAIN_DVS:
        formula_base = f"{dv} ~ motivation + support_main + effect + gender + rank_code + career_code"
        formula_marker = formula_base + f" + {MARKER}"
        cols = ["motivation", "support_main", "effect", "gender", "rank_code", "career_code", dv, MARKER]
        data = df[cols].dropna().copy()
        base = smf.ols(formula_base, data=data).fit(cov_type="HC3")
        with_marker = smf.ols(formula_marker, data=data).fit(cov_type="HC3")
        for var in ["motivation", "support_main", "effect"]:
            reg_rows.append(
                {
                    "DV": dv,
                    "predictor": var,
                    "B without marker": float(base.params[var]),
                    "B with marker": float(with_marker.params[var]),
                    "abs_delta_B": abs(float(with_marker.params[var]) - float(base.params[var])),
                }
            )
    reg_df = pd.DataFrame(reg_rows)

    with markdown_output("10_cmv_marker_proxy.md") as result_path:
        print("# 10 Marker-Proxy CMV Sensitivity Check\n")
        print("- Sample: AI users only (`Q3 == 1`)")
        print("- Method: Lindell-Whitney style marker sensitivity using `Q20_4` as a conservative marker proxy.")
        print("- Caution: `Q20_4` is not an ideal pure marker variable; this analysis is reported as a robustness/sensitivity check, not as a definitive CMV correction.\n")

        print("## Zero-order vs. marker-adjusted partial correlations\n")
        print(corr_df.round(3).to_markdown(index=False))
        print()

        print("## Regression coefficient sensitivity to marker inclusion\n")
        print(reg_df.round(3).to_markdown(index=False))
        print()

        print("## Interpretation\n")
        print(f"- Mean absolute correlation change = {corr_df['abs_delta_r'].mean():.3f}")
        print(f"- Max absolute correlation change = {corr_df['abs_delta_r'].max():.3f}")
        print(f"- Mean absolute coefficient change = {reg_df['abs_delta_B'].mean():.3f}")
        print(f"- Max absolute coefficient change = {reg_df['abs_delta_B'].max():.3f}")
        print("- If marker-adjusted correlations and coefficients remain close to the original estimates, the results are less likely to be dominated by a single common-method artifact.")

    print(f"Completed: {result_path}")


if __name__ == "__main__":
    main()
