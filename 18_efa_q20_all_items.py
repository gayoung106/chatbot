import site

site.addsitedir("/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages")

import numpy as np
import pandas as pd

from result_utils import markdown_output


Q20_COLS = [f"Q20_{i}" for i in range(1, 5)]


def cronbach_alpha(matrix: np.ndarray) -> float:
    k = matrix.shape[1]
    item_vars = matrix.var(axis=0, ddof=1)
    total_var = matrix.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - item_vars.sum() / total_var)


def item_total_correlations(matrix: np.ndarray, cols: list[str]) -> pd.Series:
    values = {}
    for i, col in enumerate(cols):
        total = np.delete(matrix, i, axis=1).mean(axis=1)
        values[col] = np.corrcoef(matrix[:, i], total)[0, 1]
    return pd.Series(values)


def kmo(corr: np.ndarray) -> tuple[np.ndarray, float]:
    inv_corr = np.linalg.inv(corr)
    partial = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    np.fill_diagonal(partial, 0)

    r2 = corr.copy() ** 2
    np.fill_diagonal(r2, 0)
    p2 = partial ** 2

    kmo_total = r2.sum() / (r2.sum() + p2.sum())
    kmo_items = r2.sum(axis=0) / (r2.sum(axis=0) + p2.sum(axis=0))
    return kmo_items, kmo_total


def bartlett_sphericity(corr: np.ndarray, n: int) -> tuple[float, float]:
    p = corr.shape[0]
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(corr))
    df = p * (p - 1) / 2
    return chi2, df


def smc(corr: np.ndarray) -> np.ndarray:
    inv_corr = np.linalg.inv(corr)
    return 1 - 1 / np.diag(inv_corr)


def principal_axis_factoring(
    corr: np.ndarray,
    n_factors: int,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    communalities = smc(corr)

    for _ in range(max_iter):
        reduced = corr.copy()
        np.fill_diagonal(reduced, communalities)

        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        retained = np.clip(eigenvalues[:n_factors], 0, None)
        loadings = eigenvectors[:, :n_factors] * np.sqrt(retained)
        updated = np.sum(loadings ** 2, axis=1)

        if np.max(np.abs(updated - communalities)) < tol:
            return loadings, updated

        communalities = updated

    return loadings, communalities


def varimax(
    loadings: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    n_rows, n_cols = loadings.shape
    rotation = np.eye(n_cols)
    criterion = 0.0

    for _ in range(max_iter):
        previous = criterion
        rotated = loadings @ rotation
        u, s, vh = np.linalg.svd(
            loadings.T
            @ (
                rotated ** 3
                - (gamma / n_rows)
                * rotated
                @ np.diag(np.diag(rotated.T @ rotated))
            )
        )
        rotation = u @ vh
        criterion = np.sum(s)

        if previous != 0 and criterion / previous < 1 + tol:
            break

    return loadings @ rotation


def rmsr(corr: np.ndarray, loadings: np.ndarray) -> float:
    reproduced = loadings @ loadings.T
    residual = corr - reproduced
    off_diag = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.sqrt(np.mean(residual[off_diag] ** 2)))


def main() -> None:
    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df_ai = df.loc[df["Q3"] == 1, Q20_COLS].dropna()

    x = df_ai.to_numpy(dtype=float)
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    corr = np.corrcoef(z, rowvar=False)

    with markdown_output("18_efa_q20_all_items.md") as result_path:
        print("# 18 Q20 전체 문항 EFA 점검\n")
        print(f"N = {len(df_ai)}")

        print("\nCorrelations")
        print(pd.DataFrame(corr, index=Q20_COLS, columns=Q20_COLS).round(3))

        print("\nReliability")
        print(f"alpha(Q20_1~Q20_4) = {cronbach_alpha(x):.3f}")
        print(f"alpha(Q20_2~Q20_4) = {cronbach_alpha(x[:, 1:]):.3f}")
        print(item_total_correlations(x, Q20_COLS).round(3).rename("item_total_r"))

        kmo_items, kmo_total = kmo(corr)
        chi2, df_b = bartlett_sphericity(corr, len(df_ai))

        print("\nFactorability")
        print(f"KMO total = {kmo_total:.3f}")
        print(pd.Series(kmo_items, index=Q20_COLS).round(3).rename("KMO"))
        print(f"Bartlett chi2 = {chi2:.3f}, df = {df_b:.0f}")

        eigenvalues = np.linalg.eigvalsh(corr)[::-1]
        print("\nEigenvalues")
        for i, value in enumerate(eigenvalues, start=1):
            print(f"Factor{i} = {value:.3f}")

        one_factor, one_h2 = principal_axis_factoring(corr, n_factors=1)
        print("\n1-factor PAF")
        print(pd.DataFrame(one_factor, index=Q20_COLS, columns=["F1"]).round(3))
        print(pd.Series(one_h2, index=Q20_COLS).round(3).rename("communality"))
        print(f"RMSR = {rmsr(corr, one_factor):.3f}")

        two_factor, two_h2 = principal_axis_factoring(corr, n_factors=2)
        two_factor = varimax(two_factor)
        print("\n2-factor PAF + varimax")
        print(pd.DataFrame(two_factor, index=Q20_COLS, columns=["F1", "F2"]).round(3))
        print(pd.Series(two_h2, index=Q20_COLS).round(3).rename("communality"))
        print(f"RMSR = {rmsr(corr, two_factor):.3f}")

        print("\n## 주요 해석")
        print("- Two eigenvalues exceed 1.0, so a 2-factor split is plausible at a heuristic level.")
        print("- The rotated 2-factor pattern is Q20_1/Q20_2 versus Q20_3/Q20_4.")
        print("- However, Q20_2 cross-loads and each factor has only two indicators, so this is not a strong confirmation of a stable 2-factor construct.")
        print("- The existing 3-item expectation scale remains psychometrically cleaner than the 4-item version.")

    print(f"완료: {result_path} 생성")


if __name__ == "__main__":
    main()
