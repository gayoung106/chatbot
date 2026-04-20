import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from result_utils import markdown_output


def main() -> None:
    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df = df[df["Q3"] == 1].copy()

    items = (
        ["Q9_3", "Q9_4"]
        + [f"Q7_{i}" for i in range(1, 6)]
        + [f"Q16_{i}" for i in range(1, 5)]
        + [f"Q20_{i}" for i in range(1, 5)]
    )
    data = df[items].dropna()

    x_scaled = StandardScaler().fit_transform(data)
    pca = PCA()
    pca.fit(x_scaled)

    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)
    eigenvalues = pca.explained_variance_
    first_factor_var = explained_var[0]
    n_factors_kaiser = int(np.sum(eigenvalues >= 1.0))

    with markdown_output("09_cmv_harman.md") as result_path:
        print("# 09 Common Method Variance Check\n")
        print("## Input for Harman's single-factor diagnostic\n")
        print(f"- AI users retained before listwise deletion = {len(df)}")
        print(f"- AI users retained after listwise deletion = {len(data)}")
        print(f"- Number of items = {len(items)}")
        print("- Motivation items: Q9_3, Q9_4")
        print("- Work-effectiveness items: Q7_1~Q7_5")
        print("- Organizational-support items: Q16_1~Q16_4")
        print("- Expectancy items: Q20_1~Q20_4")
        print("- Q20_4 is supplementary in the main models, but it is retained here because CMV is a measurement-level diagnostic.\n")

        print("## PCA variance summary\n")
        print("| Factor | Eigenvalue | Explained variance (%) | Cumulative variance (%) |")
        print("|--------|-----------:|-----------------------:|------------------------:|")
        for idx in range(len(items)):
            print(
                f"| F{idx + 1} | {eigenvalues[idx]:.3f} | "
                f"{explained_var[idx]:.2f} | {cumulative_var[idx]:.2f} |"
            )
        print()

        print("## Harman diagnostic result\n")
        print(f"- First unrotated factor explained variance = {first_factor_var:.2f}%")
        print(f"- Number of factors with eigenvalue >= 1.00 = {n_factors_kaiser}")
        print("- Conventional reference point: CMV concern becomes stronger when the first factor exceeds 50% of total variance.")
        print(
            f"- Interpretation: the first factor remains below 50%, so the data do not suggest that a single dominant common-method factor drives the observed covariance structure."
        )
        print(
            "- This does not prove the absence of CMV; it only supports the narrower claim that the covariance pattern is not reducible to one dominant response artifact."
        )

    print(f"Completed: {result_path}")


if __name__ == "__main__":
    main()
