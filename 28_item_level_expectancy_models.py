import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

from result_utils import markdown_output


N_BOOT = 5000
RNG_SEED = 42
MAIN_DVS = ["Q20_1", "Q20_2", "Q20_3"]


def ols_coef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def bca_interval(theta_hat: float, theta_boot: np.ndarray, theta_jack: np.ndarray, alpha: float = 0.05):
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


def analyze_outcome(df: pd.DataFrame, dv: str, rng: np.random.Generator):
    data = df.dropna(
        subset=["motivation", "support_main", "effect", dv, "gender", "rank_code", "career_code"]
    ).copy()

    mediator_model = smf.ols(
        "effect ~ motivation + support_main + gender + rank_code + career_code",
        data=data,
    ).fit(cov_type="HC3")
    total_model = smf.ols(
        f"{dv} ~ motivation + support_main + gender + rank_code + career_code",
        data=data,
    ).fit(cov_type="HC3")
    direct_model = smf.ols(
        f"{dv} ~ motivation + support_main + effect + gender + rank_code + career_code",
        data=data,
    ).fit(cov_type="HC3")

    n = len(data)
    full_idx = np.arange(n)
    motivation = data["motivation"].to_numpy()
    support = data["support_main"].to_numpy()
    effect = data["effect"].to_numpy()
    outcome = data[dv].to_numpy()
    gender = data["gender"].to_numpy()
    rank = data["rank_code"].to_numpy()
    career = data["career_code"].to_numpy()

    x_mediator = np.column_stack([np.ones(n), motivation, support, gender, rank, career])
    x_total = np.column_stack([np.ones(n), motivation, support, gender, rank, career])
    x_outcome = np.column_stack([np.ones(n), motivation, support, effect, gender, rank, career])

    def effect_vector(indices: np.ndarray) -> np.ndarray:
        beta_m = ols_coef(x_mediator[indices], effect[indices])
        beta_y = ols_coef(x_outcome[indices], outcome[indices])
        beta_t = ols_coef(x_total[indices], outcome[indices])
        return np.array(
            [
                beta_y[1],
                beta_m[1] * beta_y[3],
                beta_t[1],
                beta_y[2],
                beta_m[2] * beta_y[3],
                beta_t[2],
            ]
        )

    observed = effect_vector(full_idx)
    boot = np.empty((N_BOOT, 6))
    for i in range(N_BOOT):
        sample_idx = rng.integers(0, n, n)
        boot[i] = effect_vector(sample_idx)

    jackknife = np.empty((n, 6))
    for i in range(n):
        jackknife[i] = effect_vector(np.delete(full_idx, i))

    effect_names = [
        "direct_motivation",
        "indirect_motivation",
        "total_motivation",
        "direct_support_main",
        "indirect_support_main",
        "total_support_main",
    ]
    cis = {}
    for idx, name in enumerate(effect_names):
        cis[name] = bca_interval(observed[idx], boot[:, idx], jackknife[:, idx])

    return data, mediator_model, total_model, direct_model, observed, cis


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df = df[df["Q3"] == 1].copy()
    df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
    df["effect"] = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
    df["support_main"] = df[[f"Q16_{i}" for i in range(1, 5)]].mean(axis=1)

    labels = {
        "Q20_1": "업무효율 개선 기대",
        "Q20_2": "의사결정 지원 기대",
        "Q20_3": "반복업무 자동화 기대",
        "Q20_4": "일자리 대체 인식",
    }

    with markdown_output("28_item_level_expectancy_models.md") as result_path:
        print("# 28 Item-Level Expectancy Models\n")
        print("- Main analysis: Q20_1, Q20_2, Q20_3 as item-level strategic expectancy outcomes.")
        print("- Supplementary analysis: Q20_4 job replacement perception.")
        print("- Predictor strategy")
        print("  - motivation = mean(Q9_3, Q9_4)")
        print("  - support_main = mean(Q16_1~Q16_4)")
        print("  - effect = mean(Q7_1~Q7_5)\n")

        for dv in MAIN_DVS:
            data, mediator_model, total_model, direct_model, observed, cis = analyze_outcome(df, dv, rng)
            print(f"## {dv}: {labels[dv]}\n")
            print(f"- N = {len(data)}\n")
            print("### Total-effect model")
            print(f"- motivation: B = {total_model.params['motivation']:.4f}, p = {total_model.pvalues['motivation']:.4f}")
            print(f"- support_main: B = {total_model.params['support_main']:.4f}, p = {total_model.pvalues['support_main']:.4f}")
            print(f"- R2 = {total_model.rsquared:.3f}\n")
            print("### Direct-effect model")
            print(f"- motivation: B = {direct_model.params['motivation']:.4f}, p = {direct_model.pvalues['motivation']:.4f}")
            print(f"- support_main: B = {direct_model.params['support_main']:.4f}, p = {direct_model.pvalues['support_main']:.4f}")
            print(f"- effect: B = {direct_model.params['effect']:.4f}, p = {direct_model.pvalues['effect']:.4f}")
            print(f"- R2 = {direct_model.rsquared:.3f}\n")
            print("### BCa bootstrap effects")
            for idx, name in enumerate(cis):
                low, high = cis[name]
                print(f"- {name}: estimate = {observed[idx]:.4f}, 95% BCa CI = [{low:.4f}, {high:.4f}]")
            print()

        print("## Cross-item interpretation\n")
        print("- Q20_1 shows the strongest motivation-driven pattern.")
        print("- Q20_2 is also mainly explained by motivation and effect, with support_main operating mostly through the indirect path.")
        print("- Q20_3 is the clearest support_main-driven strategic expectancy outcome.")
        print("- Therefore Q20_1~Q20_3 should be treated as separate strategic expectancy outcomes rather than a single averaged scale.\n")

        supp_data, _, supp_total, supp_direct, supp_observed, supp_cis = analyze_outcome(df, "Q20_4", rng)
        print("## Supplementary: Q20_4 Job Replacement Perception\n")
        print("- Q20_4 is reported separately because it reflects replacement/threat perception rather than positive strategic expectancy.")
        print(f"- N = {len(supp_data)}\n")
        print("### Total-effect model")
        print(f"- motivation: B = {supp_total.params['motivation']:.4f}, p = {supp_total.pvalues['motivation']:.4f}")
        print(f"- support_main: B = {supp_total.params['support_main']:.4f}, p = {supp_total.pvalues['support_main']:.4f}")
        print(f"- R2 = {supp_total.rsquared:.3f}\n")
        print("### Direct-effect model")
        print(f"- motivation: B = {supp_direct.params['motivation']:.4f}, p = {supp_direct.pvalues['motivation']:.4f}")
        print(f"- support_main: B = {supp_direct.params['support_main']:.4f}, p = {supp_direct.pvalues['support_main']:.4f}")
        print(f"- effect: B = {supp_direct.params['effect']:.4f}, p = {supp_direct.pvalues['effect']:.4f}")
        print(f"- R2 = {supp_direct.rsquared:.3f}\n")
        print("### BCa bootstrap effects")
        for idx, name in enumerate(supp_cis):
            low, high = supp_cis[name]
            print(f"- {name}: estimate = {supp_observed[idx]:.4f}, 95% BCa CI = [{low:.4f}, {high:.4f}]")

    print(f"완료: {result_path} 생성")


if __name__ == "__main__":
    main()
