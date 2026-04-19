import site

site.addsitedir("/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages")
site.addsitedir("/Users/song-gayeong/Library/Python/3.11/lib/python/site-packages")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

from result_utils import markdown_output


N_BOOT = 5000
RNG_SEED = 42


def ols_coef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def bca_interval(
    theta_hat: float,
    theta_boot: np.ndarray,
    theta_jack: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    z_low = norm.ppf(alpha / 2)
    z_high = norm.ppf(1 - alpha / 2)

    prop_less = np.mean(theta_boot < theta_hat)
    prop_less = min(max(prop_less, 1 / (2 * len(theta_boot))), 1 - 1 / (2 * len(theta_boot)))
    z0 = norm.ppf(prop_less)

    jack_mean = theta_jack.mean()
    numerator = np.sum((jack_mean - theta_jack) ** 3)
    denominator = 6 * (np.sum((jack_mean - theta_jack) ** 2) ** 1.5)
    acceleration = 0.0 if denominator == 0 else numerator / denominator

    adj_low = norm.cdf(z0 + (z0 + z_low) / (1 - acceleration * (z0 + z_low)))
    adj_high = norm.cdf(z0 + (z0 + z_high) / (1 - acceleration * (z0 + z_high)))

    return float(np.quantile(theta_boot, adj_low)), float(np.quantile(theta_boot, adj_high))


def f2(full_r2: float, reduced_r2: float) -> float:
    return float((full_r2 - reduced_r2) / (1 - full_r2))


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df = df[df["Q3"] == 1].copy()

    df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
    df["effect"] = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
    df["support"] = df[[f"Q16_{i}" for i in range(1, 7)]].mean(axis=1)
    df["expectation"] = df[[f"Q20_{i}" for i in range(2, 5)]].mean(axis=1)

    required = [
        "motivation",
        "support",
        "effect",
        "expectation",
        "gender",
        "rank_code",
        "career_code",
    ]
    df = df.dropna(subset=required).copy()

    n = len(df)
    full_idx = np.arange(n)

    motivation = df["motivation"].to_numpy()
    support = df["support"].to_numpy()
    effect = df["effect"].to_numpy()
    expectation = df["expectation"].to_numpy()
    gender = df["gender"].to_numpy()
    rank = df["rank_code"].to_numpy()
    career = df["career_code"].to_numpy()

    x_mediator = np.column_stack([np.ones(n), motivation, support, gender, rank, career])
    x_total = np.column_stack([np.ones(n), motivation, support, gender, rank, career])
    x_outcome = np.column_stack([np.ones(n), motivation, support, effect, gender, rank, career])

    def effect_vector(indices: np.ndarray) -> np.ndarray:
        beta_m = ols_coef(x_mediator[indices], effect[indices])
        beta_y = ols_coef(x_outcome[indices], expectation[indices])
        beta_t = ols_coef(x_total[indices], expectation[indices])

        return np.array(
            [
                beta_y[1],               # direct motivation
                beta_m[1] * beta_y[3],   # indirect motivation
                beta_t[1],               # total motivation
                beta_y[2],               # direct support
                beta_m[2] * beta_y[3],   # indirect support
                beta_t[2],               # total support
            ]
        )

    effect_names = [
        "direct_motivation",
        "indirect_motivation",
        "total_motivation",
        "direct_support",
        "indirect_support",
        "total_support",
    ]

    observed = effect_vector(full_idx)

    boot = np.empty((N_BOOT, len(effect_names)))
    for i in range(N_BOOT):
        sample_idx = rng.integers(0, n, n)
        boot[i] = effect_vector(sample_idx)

    jackknife = np.empty((n, len(effect_names)))
    for i in range(n):
        jackknife[i] = effect_vector(np.delete(full_idx, i))

    ci = {}
    for j, name in enumerate(effect_names):
        ci[name] = bca_interval(observed[j], boot[:, j], jackknife[:, j])

    mediator_model = smf.ols(
        "effect ~ motivation + support + gender + rank_code + career_code",
        data=df,
    ).fit(cov_type="HC3")
    outcome_model = smf.ols(
        "expectation ~ motivation + support + effect + gender + rank_code + career_code",
        data=df,
    ).fit(cov_type="HC3")
    total_model = smf.ols(
        "expectation ~ motivation + support + gender + rank_code + career_code",
        data=df,
    ).fit(cov_type="HC3")

    mediator_mot_reduced = smf.ols(
        "effect ~ support + gender + rank_code + career_code",
        data=df,
    ).fit()
    mediator_sup_reduced = smf.ols(
        "effect ~ motivation + gender + rank_code + career_code",
        data=df,
    ).fit()

    outcome_mot_reduced = smf.ols(
        "expectation ~ support + effect + gender + rank_code + career_code",
        data=df,
    ).fit()
    outcome_sup_reduced = smf.ols(
        "expectation ~ motivation + effect + gender + rank_code + career_code",
        data=df,
    ).fit()
    outcome_eff_reduced = smf.ols(
        "expectation ~ motivation + support + gender + rank_code + career_code",
        data=df,
    ).fit()

    total_mot_reduced = smf.ols(
        "expectation ~ support + gender + rank_code + career_code",
        data=df,
    ).fit()
    total_sup_reduced = smf.ols(
        "expectation ~ motivation + gender + rank_code + career_code",
        data=df,
    ).fit()

    with markdown_output("19_parallel_mediation_hc3_bca.md") as result_path:
        print("# 19 Parallel Mediation with HC3 and BCa Bootstrap\n")
        print(f"N = {n}")
        print(f"Bootstrap resamples = {N_BOOT}\n")

        print("\nMediator model: effect ~ motivation + support + controls")
        for var in ["motivation", "support"]:
            print(
                f"{var}: B = {mediator_model.params[var]:.6f}, "
                f"SE(HC3) = {mediator_model.bse[var]:.6f}, "
                f"p = {mediator_model.pvalues[var]:.6f}"
            )
        print(f"R2 = {mediator_model.rsquared:.6f}")

        print("\nOutcome model: expectation ~ motivation + support + effect + controls")
        for var in ["motivation", "support", "effect"]:
            print(
                f"{var}: B = {outcome_model.params[var]:.6f}, "
                f"SE(HC3) = {outcome_model.bse[var]:.6f}, "
                f"p = {outcome_model.pvalues[var]:.6f}"
            )
        print(f"R2 = {outcome_model.rsquared:.6f}")

        print("\nTotal-effect model: expectation ~ motivation + support + controls")
        for var in ["motivation", "support"]:
            print(
                f"{var}: B = {total_model.params[var]:.6f}, "
                f"SE(HC3) = {total_model.bse[var]:.6f}, "
                f"p = {total_model.pvalues[var]:.6f}"
            )
        print(f"R2 = {total_model.rsquared:.6f}")

        print("\nEffects with 95% BCa CI")
        for name, estimate in zip(effect_names, observed):
            low, high = ci[name]
            print(f"{name}: estimate = {estimate:.6f}, 95% BCa CI = [{low:.6f}, {high:.6f}]")

        print("\nLocal f2")
        print(f"mediator_motivation = {f2(mediator_model.rsquared, mediator_mot_reduced.rsquared):.6f}")
        print(f"mediator_support = {f2(mediator_model.rsquared, mediator_sup_reduced.rsquared):.6f}")
        print(f"outcome_direct_motivation = {f2(outcome_model.rsquared, outcome_mot_reduced.rsquared):.6f}")
        print(f"outcome_direct_support = {f2(outcome_model.rsquared, outcome_sup_reduced.rsquared):.6f}")
        print(f"outcome_effect = {f2(outcome_model.rsquared, outcome_eff_reduced.rsquared):.6f}")
        print(f"total_motivation = {f2(total_model.rsquared, total_mot_reduced.rsquared):.6f}")
        print(f"total_support = {f2(total_model.rsquared, total_sup_reduced.rsquared):.6f}")

        print("\n## 주요 해석")
        print("- 이 모형은 motivation과 support를 동시에 투입해 두 변수의 직접효과, 간접효과, 총효과를 병렬적으로 비교한다.")
        print("- BCa 신뢰구간에 0이 포함되지 않는 간접효과는 보다 안정적인 매개효과로 해석할 수 있다.")
        print("- `f2`는 각 경로가 설명력에 기여하는 국소 효과크기를 보여주므로, 통계적 유의성과 함께 실질적 크기도 판단할 수 있다.")

    print(f"완료: {result_path} 생성")


if __name__ == "__main__":
    main()
