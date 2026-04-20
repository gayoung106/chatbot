import pandas as pd
from scipy.stats import kurtosis, pearsonr, skew

from result_utils import markdown_output


AI_USER_FILTER = "Q3 == 1"
MOTIVATION_ITEMS = ["Q9_3", "Q9_4"]
EFFECT_ITEMS = [f"Q7_{i}" for i in range(1, 6)]
SUPPORT_ITEMS = [f"Q16_{i}" for i in range(1, 5)]
MAIN_DVS = [f"Q20_{i}" for i in range(1, 4)]
SUPPLEMENTARY_DV = "Q20_4"

SCALE_MAP = {
    "motivation": MOTIVATION_ITEMS,
    "effect": EFFECT_ITEMS,
    "support_main": SUPPORT_ITEMS,
    "strategic_expectancy_main": MAIN_DVS,
}


def summarize_scale(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    data = df[columns].dropna()
    pairs = []
    if len(columns) >= 2:
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                r, _ = pearsonr(data[columns[i]], data[columns[j]])
                pairs.append(r)

    row_mean = data.mean(axis=1) if not data.empty else pd.Series(dtype=float)
    return {
        "n": len(data),
        "mean": float(row_mean.mean()) if not row_mean.empty else float("nan"),
        "sd": float(row_mean.std()) if not row_mean.empty else float("nan"),
        "min": float(row_mean.min()) if not row_mean.empty else float("nan"),
        "max": float(row_mean.max()) if not row_mean.empty else float("nan"),
        "avg_inter_item_r": float(sum(pairs) / len(pairs)) if pairs else float("nan"),
    }


def main() -> None:
    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df_ai = df.query(AI_USER_FILTER).copy()
    df_non = df.query("Q3 == 0").copy()

    df_ai["motivation"] = df_ai[MOTIVATION_ITEMS].mean(axis=1)
    df_ai["effect"] = df_ai[EFFECT_ITEMS].mean(axis=1)
    df_ai["support_main"] = df_ai[SUPPORT_ITEMS].mean(axis=1)
    df_ai["strategic_expectancy_main"] = df_ai[MAIN_DVS].mean(axis=1)

    all_item_cols = MOTIVATION_ITEMS + EFFECT_ITEMS + SUPPORT_ITEMS + MAIN_DVS + [SUPPLEMENTARY_DV]

    with markdown_output("03_analysis_ai_group.md") as result_path:
        print("# 03 Descriptive Analysis for AI Users\n")
        print("## Sample overview\n")
        print(f"- Full sample N = {len(df)}")
        print(f"- AI users N = {len(df_ai)}")
        print(f"- Non-users N = {len(df_non)}\n")

        print("## Scale construction used in the main study\n")
        print("- motivation = mean(Q9_3, Q9_4)")
        print("- effect = mean(Q7_1~Q7_5)")
        print("- support_main = mean(Q16_1~Q16_4)")
        print("- main strategic expectancy outcomes = Q20_1, Q20_2, Q20_3")
        print("- supplementary outcome = Q20_4")
        print("- Q16_5~Q16_7 are excluded from the main organizational-support index because they blend organizational and personal-level evaluation content.\n")

        print("## Construct-level descriptives and inter-item correlations\n")
        print("| Construct | Items | N | Mean | SD | Min | Max | Average inter-item r |")
        print("|-----------|-------|--:|-----:|---:|----:|----:|---------------------:|")
        for label, columns in SCALE_MAP.items():
            stats = summarize_scale(df_ai, columns)
            print(
                f"| {label} | {', '.join(columns)} | {stats['n']} | {stats['mean']:.3f} | "
                f"{stats['sd']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['avg_inter_item_r']:.3f} |"
            )
        print()

        print("## Item-level descriptives for AI users\n")
        item_desc = pd.DataFrame(index=all_item_cols)
        for col in all_item_cols:
            item_desc.loc[col, "mean"] = df_ai[col].mean()
            item_desc.loc[col, "sd"] = df_ai[col].std()
            item_desc.loc[col, "min"] = df_ai[col].min()
            item_desc.loc[col, "max"] = df_ai[col].max()
            item_desc.loc[col, "skew"] = skew(df_ai[col].dropna())
            item_desc.loc[col, "kurtosis"] = kurtosis(df_ai[col].dropna())
        print(item_desc.round(3).to_markdown())
        print()

        print("## Main-study interpretation\n")
        print("- motivation is represented with a two-item observed index, so average inter-item correlation is more informative than coefficient alpha.")
        print("- effect and support_main show stable descriptive distributions among AI users.")
        print("- Q20_1~Q20_3 can be compared item-by-item without forcing them into a single composite.")
        print("- Q20_4 is retained only as a supplementary threat-perception outcome, not as part of the main strategic expectancy block.")

    print(f"Completed: {result_path}")


if __name__ == "__main__":
    main()
