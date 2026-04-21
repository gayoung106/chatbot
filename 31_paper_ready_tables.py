from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import norm, pearsonr
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from result_utils import markdown_output


AI_FILTER = "Q3 == 1"
RNG_SEED = 42
N_BOOT = 5000

MOTIVATION_ITEMS = ["Q9_3", "Q9_4"]
EFFECT_ITEMS = [f"Q7_{i}" for i in range(1, 6)]
SUPPORT_MAIN_ITEMS = [f"Q16_{i}" for i in range(1, 5)]
SUPPORT_FULL_ITEMS = [f"Q16_{i}" for i in range(1, 8)]
MAIN_DVS = [f"Q20_{i}" for i in range(1, 4)]
ALL_DVS = [f"Q20_{i}" for i in range(1, 5)]
CONTROLS = ["gender", "rank_code", "career_code"]

DV_LABELS = {
    "Q20_1": "업무효율 개선 기대",
    "Q20_2": "의사결정 지원 기대",
    "Q20_3": "반복업무 자동화 기대",
    "Q20_4": "일자리 대체 인식",
}

CONSTRUCT_SPECS = {
    "motivation": MOTIVATION_ITEMS,
    "effect": EFFECT_ITEMS,
    "support_main": SUPPORT_MAIN_ITEMS,
}


@dataclass
class MediationResult:
    n: int
    mediator_model: sm.regression.linear_model.RegressionResultsWrapper
    total_model: sm.regression.linear_model.RegressionResultsWrapper
    direct_model: sm.regression.linear_model.RegressionResultsWrapper
    observed: dict[str, float]
    cis: dict[str, tuple[float, float]]


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.3f}"


def fmt_b_p(b: float, p: float) -> str:
    return f"{b:.3f}{significance_stars(p)} ({fmt_p(p)})"


def fmt_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def fmt_optional_p(p: float | str) -> str:
    if isinstance(p, str):
        return p
    if pd.isna(p):
        return ""
    return fmt_p(float(p))


def avg_inter_item_r(df: pd.DataFrame, columns: list[str]) -> float:
    data = df[columns].dropna()
    rs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r, _ = pearsonr(data[columns[i]], data[columns[j]])
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def one_factor_cr_ave(df: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    data = df[columns].dropna()
    z = (data - data.mean()) / data.std(ddof=0)
    fa = FactorAnalyzer(n_factors=1, method="minres", rotation=None)
    fa.fit(z)
    loadings = np.abs(fa.loadings_.flatten())
    error = 1 - loadings**2
    cr = (loadings.sum() ** 2) / ((loadings.sum() ** 2) + error.sum())
    ave = np.mean(loadings**2)
    return float(cr), float(ave)


def load_data() -> pd.DataFrame:
    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df = df.query(AI_FILTER).copy()
    df["motivation"] = df[MOTIVATION_ITEMS].mean(axis=1)
    df["effect"] = df[EFFECT_ITEMS].mean(axis=1)
    df["support_main"] = df[SUPPORT_MAIN_ITEMS].mean(axis=1)
    df["strategic_expectancy_main"] = df[MAIN_DVS].mean(axis=1)
    return df


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for construct, items in CONSTRUCT_SPECS.items():
        score = df[items].mean(axis=1)
        rows.append(
            {
                "구성개념": construct,
                "문항": ", ".join(items),
                "N": int(df[items].dropna().shape[0]),
                "평균": score.mean(),
                "표준편차": score.std(),
                "최소값": score.min(),
                "최대값": score.max(),
            }
        )
    return pd.DataFrame(rows)


def reliability_validity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for construct, items in CONSTRUCT_SPECS.items():
        if construct in {"strategic_expectancy_main", "support_main"}:
            continue
        alpha, _ = pg.cronbach_alpha(data=df[items].dropna())
        avg_r = avg_inter_item_r(df, items)
        cr, ave = one_factor_cr_ave(df, items)
        rows.append(
            {
                "구성개념": construct,
                "문항수": len(items),
                "Cronbach α": alpha,
                "평균 문항간 상관": avg_r,
                "복합신뢰도(CR)": cr,
                "AVE": ave,
                "α >= .70": "충족" if alpha >= 0.70 else "미달",
                "CR >= .70": "충족" if cr >= 0.70 else "미달",
                "AVE >= .50": "충족" if ave >= 0.50 else "미달",
            }
        )
    return pd.DataFrame(rows)


def observed_index_reliability_note(df: pd.DataFrame) -> dict[str, float]:
    support_alpha, _ = pg.cronbach_alpha(data=df[SUPPORT_MAIN_ITEMS].dropna())
    return {
        "support_alpha": float(support_alpha),
        "support_avg_r": avg_inter_item_r(df, SUPPORT_MAIN_ITEMS),
    }


def reflective_validity_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reflective = {
        "motivation": MOTIVATION_ITEMS,
        "effect": EFFECT_ITEMS,
    }
    ave_map = {name: one_factor_cr_ave(df, items)[1] for name, items in reflective.items()}

    fl = pd.DataFrame(index=reflective.keys(), columns=reflective.keys(), dtype=object)
    for row_name, row_items in reflective.items():
        for col_name, col_items in reflective.items():
            if row_name == col_name:
                fl.loc[row_name, col_name] = round(float(np.sqrt(ave_map[row_name])), 3)
            else:
                row_score = df[row_items].mean(axis=1)
                col_score = df[col_items].mean(axis=1)
                valid = pd.concat([row_score, col_score], axis=1).dropna()
                r, _ = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                fl.loc[row_name, col_name] = round(float(r), 3)

    hetero = []
    for a in MOTIVATION_ITEMS:
        for b in EFFECT_ITEMS:
            hetero.append(abs(pearsonr(df[[a, b]].dropna()[a], df[[a, b]].dropna()[b])[0]))
    mono_m = [abs(pearsonr(df[[MOTIVATION_ITEMS[0], MOTIVATION_ITEMS[1]]].dropna()[MOTIVATION_ITEMS[0]], df[[MOTIVATION_ITEMS[0], MOTIVATION_ITEMS[1]]].dropna()[MOTIVATION_ITEMS[1]])[0])]
    mono_e = []
    for i in range(len(EFFECT_ITEMS)):
        for j in range(i + 1, len(EFFECT_ITEMS)):
            pair = df[[EFFECT_ITEMS[i], EFFECT_ITEMS[j]]].dropna()
            mono_e.append(abs(pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])[0]))
    htmt = float(np.mean(hetero) / np.sqrt(np.mean(mono_m) * np.mean(mono_e)))
    htmt_df = pd.DataFrame([{"Construct pair": "motivation-effect", "HTMT": round(htmt, 3), "Criterion (< .85)": "충족" if htmt < 0.85 else "미달"}])
    return fl, htmt_df


def integrated_efa(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    efa_items = MOTIVATION_ITEMS + EFFECT_ITEMS + SUPPORT_FULL_ITEMS
    data = df[efa_items].dropna()
    corr = data.corr()
    eigenvalues = np.sort(np.linalg.eigvalsh(corr.to_numpy()))[::-1]
    n_factors = int(np.sum(eigenvalues >= 1.0))
    chi2, bartlett_p = calculate_bartlett_sphericity(data)
    kmo_per_item, kmo_model = calculate_kmo(data)

    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="minres")
    fa.fit(data)
    loadings = pd.DataFrame(
        fa.loadings_,
        index=efa_items,
        columns=[f"요인{i}" for i in range(1, n_factors + 1)],
    )
    loadings["공통성"] = fa.get_communalities()
    loadings["주요요인"] = loadings[[f"요인{i}" for i in range(1, n_factors + 1)]].abs().idxmax(axis=1)

    summary = pd.DataFrame(
        [
            {"지표": "EFA 투입 문항 수", "값": len(efa_items)},
            {"지표": "분석 표본수", "값": len(data)},
            {"지표": "KMO", "값": round(float(kmo_model), 3)},
            {"지표": "Bartlett χ²", "값": round(float(chi2), 3)},
            {"지표": "Bartlett p", "값": "< .001" if bartlett_p < 0.001 else round(float(bartlett_p), 3)},
            {"지표": "고유값 1 이상 요인 수", "값": n_factors},
            {"지표": "1요인 설명분산(%)", "값": round(float(eigenvalues[0] / eigenvalues.sum() * 100), 2)},
            {"지표": "문항별 KMO 최소값", "값": round(float(np.min(kmo_per_item)), 3)},
        ]
    )
    return summary, loadings.reset_index(names="문항")


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    vars_for_corr = ["motivation", "effect", "support_main"] + ALL_DVS
    labels = {
        "motivation": "1. motivation",
        "effect": "2. effect",
        "support_main": "3. support_main",
        "Q20_1": "4. Q20_1",
        "Q20_2": "5. Q20_2",
        "Q20_3": "6. Q20_3",
        "Q20_4": "7. Q20_4",
    }

    out = pd.DataFrame(index=[labels[v] for v in vars_for_corr])
    means = []
    sds = []
    for col in vars_for_corr:
        means.append(df[col].mean())
        sds.append(df[col].std())

    for i, row_var in enumerate(vars_for_corr):
        row = []
        for j, col_var in enumerate(vars_for_corr):
            if i == j:
                row.append("1.000")
            elif i > j:
                sub = df[[row_var, col_var]].dropna()
                r, p = pearsonr(sub[row_var], sub[col_var])
                row.append(f"{r:.3f}{significance_stars(p)}")
            else:
                row.append("")
        out[labels[row_var]] = row
    out.insert(0, "평균", np.round(means, 3))
    out.insert(1, "표준편차", np.round(sds, 3))
    out.columns = ["평균", "표준편차"] + [labels[v] for v in vars_for_corr]
    return out


def standardized_beta_from_model(model: sm.regression.linear_model.RegressionResultsWrapper, data: pd.DataFrame, dv: str, predictor: str) -> float:
    y_sd = data[dv].std(ddof=0)
    x_sd = data[predictor].std(ddof=0)
    if y_sd == 0 or x_sd == 0:
        return float("nan")
    return float(model.params[predictor] * x_sd / y_sd)


def hierarchical_models(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    if dv == "effect":
        formulas = [
            "effect ~ gender + rank_code + career_code",
            "effect ~ gender + rank_code + career_code + motivation + support_main",
        ]
        predictors = ["motivation", "support_main"]
    else:
        formulas = [
            f"{dv} ~ gender + rank_code + career_code",
            f"{dv} ~ gender + rank_code + career_code + motivation + support_main",
            f"{dv} ~ gender + rank_code + career_code + motivation + support_main + effect",
        ]
        predictors = ["motivation", "support_main", "effect"]

    models_nonrobust = []
    models_hc3 = []
    for formula in formulas:
        cols = [c.strip() for c in formula.replace("~", "+").split("+")]
        data = df[cols].dropna()
        models_nonrobust.append(smf.ols(formula, data=data).fit())
        models_hc3.append(smf.ols(formula, data=data).fit(cov_type="HC3"))

    rows = []
    for idx, model in enumerate(models_hc3):
        delta_r2 = np.nan if idx == 0 else model.rsquared - models_hc3[idx - 1].rsquared
        delta_p = ""
        if idx > 0:
            nested = anova_lm(models_nonrobust[idx - 1], models_nonrobust[idx])
            delta_p = fmt_p(float(nested["Pr(>F)"].iloc[1]))
        formula = formulas[idx]
        cols = [c.strip() for c in formula.replace("~", "+").split("+")]
        model_data = df[cols].dropna()
        if idx == 0:
            rows.append(
                {
                    "종속변수": dv,
                    "단계": "모형 1",
                    "예측변수": "Controls only",
                    "N": int(model.nobs),
                    "R²": model.rsquared,
                    "ΔR²": delta_r2,
                    "ΔR² p": delta_p,
                    "B": np.nan,
                    "SE": np.nan,
                    "β": np.nan,
                    "95% CI": "",
                    "p": np.nan,
                    "Holm family": "",
                    "Holm adjusted p": np.nan,
                    "Holm critical p": np.nan,
                    "Holm 유의": "",
                }
            )
        for predictor in ["motivation", "support_main", "effect"]:
            if predictor in model.params.index:
                ci_low, ci_high = model.conf_int().loc[predictor]
                rows.append(
                    {
                        "종속변수": dv,
                        "단계": f"모형 {idx + 1}",
                        "예측변수": predictor,
                        "N": int(model.nobs),
                        "R²": model.rsquared,
                        "ΔR²": delta_r2,
                        "ΔR² p": delta_p,
                        "B": float(model.params[predictor]),
                        "SE": float(model.bse[predictor]),
                        "β": standardized_beta_from_model(model, model_data, dv, predictor),
                        "95% CI": fmt_ci(float(ci_low), float(ci_high)),
                        "p": float(model.pvalues[predictor]),
                        "Holm family": "",
                        "Holm adjusted p": np.nan,
                        "Holm critical p": np.nan,
                        "Holm 유의": "",
                    }
                )
    return pd.DataFrame(rows)


def add_holm_to_hierarchical_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    combined = pd.concat(
        [table.assign(_table_key=key, _row_id=table.index) for key, table in tables.items()],
        ignore_index=True,
    )
    families = [
        (
            "Q20 총효과 모형(모형2, m=6)",
            (combined["종속변수"].isin(MAIN_DVS))
            & (combined["단계"] == "모형 2")
            & (combined["예측변수"].isin(["motivation", "support_main"])),
        ),
        (
            "Q20 직접효과 모형(모형3, m=9)",
            (combined["종속변수"].isin(MAIN_DVS))
            & (combined["단계"] == "모형 3")
            & (combined["예측변수"].isin(["motivation", "support_main", "effect"])),
        ),
    ]

    for family_name, mask in families:
        family = combined.loc[mask].copy().sort_values("p", kind="mergesort")
        m = len(family)
        if m == 0:
            continue
        raw_p = family["p"].to_numpy(dtype=float)
        adjusted = np.maximum.accumulate(raw_p * np.arange(m, 0, -1))
        adjusted = np.clip(adjusted, 0, 1)
        reject = []
        still_rejecting = True
        for rank, p_value in enumerate(raw_p, start=1):
            critical = 0.05 / (m - rank + 1)
            current_reject = bool(still_rejecting and p_value <= critical)
            reject.append(current_reject)
            if not current_reject:
                still_rejecting = False
            idx = family.index[rank - 1]
            combined.loc[idx, "Holm family"] = family_name
            combined.loc[idx, "Holm adjusted p"] = adjusted[rank - 1]
            combined.loc[idx, "Holm critical p"] = critical
            combined.loc[idx, "Holm 유의"] = "유의" if current_reject else "비유의"

    output = {}
    for key, table in tables.items():
        restored = combined[combined["_table_key"] == key].drop(columns=["_table_key", "_row_id"]).copy()
        for col in ["p", "Holm adjusted p", "Holm critical p"]:
            restored[col] = restored[col].map(fmt_optional_p)
        output[key] = restored
    return output


def vif_table(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    formula = (
        "effect ~ motivation + support_main + gender + rank_code + career_code"
        if dv == "effect"
        else f"{dv} ~ motivation + support_main + effect + gender + rank_code + career_code"
    )
    cols = [c.strip() for c in formula.replace("~", "+").split("+")]
    data = df[cols].dropna()
    y, x = smf.ols(formula, data=data).fit().model.endog, smf.ols(formula, data=data).fit().model.exog
    names = smf.ols(formula, data=data).fit().model.exog_names
    rows = []
    for idx, name in enumerate(names):
        if name == "Intercept":
            continue
        vif = variance_inflation_factor(x, idx)
        rows.append({"종속변수": dv, "예측변수": name, "Tolerance": 1 / vif, "VIF": vif})
    return pd.DataFrame(rows)


def ols_coef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def bca_interval(theta_hat: float, theta_boot: np.ndarray, theta_jack: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
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


def mediation_analysis(df: pd.DataFrame, dv: str) -> MediationResult:
    data = df.dropna(subset=["motivation", "support_main", "effect", dv] + CONTROLS).copy()
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
    idx = np.arange(n)
    motivation = data["motivation"].to_numpy()
    support = data["support_main"].to_numpy()
    effect = data["effect"].to_numpy()
    outcome = data[dv].to_numpy()
    gender = data["gender"].to_numpy()
    rank = data["rank_code"].to_numpy()
    career = data["career_code"].to_numpy()

    x_m = np.column_stack([np.ones(n), motivation, support, gender, rank, career])
    x_t = np.column_stack([np.ones(n), motivation, support, gender, rank, career])
    x_y = np.column_stack([np.ones(n), motivation, support, effect, gender, rank, career])

    def effect_vector(sample_idx: np.ndarray) -> np.ndarray:
        beta_m = ols_coef(x_m[sample_idx], effect[sample_idx])
        beta_t = ols_coef(x_t[sample_idx], outcome[sample_idx])
        beta_y = ols_coef(x_y[sample_idx], outcome[sample_idx])
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

    rng = np.random.default_rng(RNG_SEED)
    observed_vec = effect_vector(idx)
    boot = np.empty((N_BOOT, 6))
    for i in range(N_BOOT):
        sample_idx = rng.integers(0, n, n)
        boot[i] = effect_vector(sample_idx)

    jack = np.empty((n, 6))
    for i in range(n):
        jack[i] = effect_vector(np.delete(idx, i))

    names = [
        "direct_motivation",
        "indirect_motivation",
        "total_motivation",
        "direct_support_main",
        "indirect_support_main",
        "total_support_main",
    ]
    observed = {name: float(observed_vec[i]) for i, name in enumerate(names)}
    cis = {name: bca_interval(observed_vec[i], boot[:, i], jack[:, i]) for i, name in enumerate(names)}
    return MediationResult(
        n=n,
        mediator_model=mediator_model,
        total_model=total_model,
        direct_model=direct_model,
        observed=observed,
        cis=cis,
    )


def mediation_table(result_map: dict[str, MediationResult]) -> pd.DataFrame:
    rows = []
    order = [
        "direct_motivation",
        "indirect_motivation",
        "total_motivation",
        "direct_support_main",
        "indirect_support_main",
        "total_support_main",
    ]
    for dv, result in result_map.items():
        for effect_name in order:
            low, high = result.cis[effect_name]
            p_value = np.nan
            if effect_name == "direct_motivation":
                p_value = float(result.direct_model.pvalues["motivation"])
            elif effect_name == "total_motivation":
                p_value = float(result.total_model.pvalues["motivation"])
            elif effect_name == "direct_support_main":
                p_value = float(result.direct_model.pvalues["support_main"])
            elif effect_name == "total_support_main":
                p_value = float(result.total_model.pvalues["support_main"])
            rows.append(
                {
                    "종속변수": DV_LABELS[dv],
                    "효과": effect_name,
                    "추정치": result.observed[effect_name],
                    "95% BCa CI": fmt_ci(low, high),
                    "p(HC3)": p_value,
                    "Holm adjusted p": np.nan,
                    "Holm critical p": np.nan,
                    "Holm 유의": "",
                    "BCa CI 판단": "유의" if low * high > 0 else "비유의",
                }
            )
    out = pd.DataFrame(rows)
    mask = out["효과"].isin(["total_motivation", "total_support_main"])
    family = out.loc[mask].copy().sort_values("p(HC3)", kind="mergesort")
    m = len(family)
    raw_p = family["p(HC3)"].to_numpy(dtype=float)
    adjusted = np.maximum.accumulate(raw_p * np.arange(m, 0, -1))
    adjusted = np.clip(adjusted, 0, 1)
    still_rejecting = True
    for rank, p_value in enumerate(raw_p, start=1):
        critical = 0.05 / (m - rank + 1)
        current_reject = bool(still_rejecting and p_value <= critical)
        if not current_reject:
            still_rejecting = False
        idx = family.index[rank - 1]
        out.loc[idx, "Holm adjusted p"] = adjusted[rank - 1]
        out.loc[idx, "Holm critical p"] = critical
        out.loc[idx, "Holm 유의"] = "유의" if current_reject else "비유의"
    for col in ["p(HC3)", "Holm adjusted p", "Holm critical p"]:
        out[col] = out[col].map(fmt_optional_p)
    return out


def standardized_beta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("effect", "effect ~ motivation + support_main + gender + rank_code + career_code"),
        ("Q20_1", "Q20_1 ~ motivation + support_main + effect + gender + rank_code + career_code"),
        ("Q20_2", "Q20_2 ~ motivation + support_main + effect + gender + rank_code + career_code"),
        ("Q20_3", "Q20_3 ~ motivation + support_main + effect + gender + rank_code + career_code"),
    ]
    for dv, formula in specs:
        cols = [c.strip() for c in formula.replace("~", "+").split("+")]
        data = df[cols].dropna().copy()
        for col in cols:
            data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=0)
        model = smf.ols(formula, data=data).fit(cov_type="HC3")
        beta_m = float(model.params["motivation"])
        beta_s = float(model.params["support_main"])
        stronger = "motivation" if abs(beta_m) > abs(beta_s) else "support_main"
        rows.append(
            {
                "종속변수": "effect" if dv == "effect" else DV_LABELS[dv],
                "표준화 β(motivation)": beta_m,
                "표준화 β(support_main)": beta_s,
                "표준화 β(effect)": float(model.params["effect"]) if "effect" in model.params.index and dv != "effect" else np.nan,
                "더 큰 독립변수 영향력": stronger,
            }
        )
    return pd.DataFrame(rows)


def mancova_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = MAIN_DVS + ["motivation", "support_main", "effect"] + CONTROLS
    data = df[cols].dropna().copy()
    model = MANOVA.from_formula(
        "Q20_1 + Q20_2 + Q20_3 ~ motivation + support_main + effect + gender + rank_code + career_code",
        data=data,
    )
    result = model.mv_test()
    rows = []
    for predictor in ["motivation", "support_main", "effect", "gender", "rank_code", "career_code"]:
        stat = result.results[predictor]["stat"].loc["Pillai's trace"]
        rows.append(
            {
                "효과": predictor,
                "Pillai's trace": float(stat["Value"]),
                "Num df": float(stat["Num DF"]),
                "Den df": float(stat["Den DF"]),
                "F": float(stat["F Value"]),
                "p": fmt_p(float(stat["Pr > F"])),
                "해석": "유의" if float(stat["Pr > F"]) < 0.05 else "비유의",
            }
        )
    return pd.DataFrame(rows)


def supplementary_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmv_items = MOTIVATION_ITEMS + EFFECT_ITEMS + SUPPORT_MAIN_ITEMS + ALL_DVS
    cmv_data = df[cmv_items].dropna()
    fa = FactorAnalyzer(rotation=None)
    fa.fit(cmv_data)
    eigenvalues, _ = fa.get_eigenvalues()
    first_var = float(eigenvalues[0] / eigenvalues.sum() * 100)
    harman = pd.DataFrame(
        [
            {
                "분석": "Harman 단일요인 검정",
                "표본수": len(cmv_data),
                "문항수": len(cmv_items),
                "제1요인 설명분산(%)": round(first_var, 2),
                "고유값 1 이상 요인수": int(np.sum(eigenvalues >= 1.0)),
                "판정": "CMV 우려 낮음" if first_var < 50 else "추가 점검 필요",
            }
        ]
    )

    q20_4 = mediation_analysis(df, "Q20_4")
    supp = pd.DataFrame(
        [
            {
                "종속변수": DV_LABELS["Q20_4"],
                "총효과 B(motivation)": q20_4.total_model.params["motivation"],
                "총효과 p(motivation)": q20_4.total_model.pvalues["motivation"],
                "총효과 B(support_main)": q20_4.total_model.params["support_main"],
                "총효과 p(support_main)": q20_4.total_model.pvalues["support_main"],
                "직접효과 B(motivation)": q20_4.direct_model.params["motivation"],
                "직접효과 p(motivation)": q20_4.direct_model.pvalues["motivation"],
                "직접효과 B(support_main)": q20_4.direct_model.params["support_main"],
                "직접효과 p(support_main)": q20_4.direct_model.pvalues["support_main"],
                "매개효과(motivation)": fmt_ci(*q20_4.cis["indirect_motivation"]),
                "매개효과(support_main)": fmt_ci(*q20_4.cis["indirect_support_main"]),
            }
        ]
    )
    return harman, supp


def marker_proxy_cmv_summary(df: pd.DataFrame) -> pd.DataFrame:
    marker = "Q20_4"
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
    corr_deltas = []
    for x, y in pairs:
        sub = df[[x, y, marker]].dropna()
        zero = pg.corr(sub[x], sub[y], method="pearson").iloc[0]["r"]
        partial = pg.partial_corr(data=sub, x=x, y=y, covar=marker, method="pearson").iloc[0]["r"]
        corr_deltas.append(abs(float(partial) - float(zero)))

    coef_deltas = []
    for dv in MAIN_DVS:
        base_formula = f"{dv} ~ motivation + support_main + effect + gender + rank_code + career_code"
        marker_formula = base_formula + f" + {marker}"
        cols = ["motivation", "support_main", "effect", "gender", "rank_code", "career_code", dv, marker]
        data = df[cols].dropna().copy()
        base = smf.ols(base_formula, data=data).fit(cov_type="HC3")
        marked = smf.ols(marker_formula, data=data).fit(cov_type="HC3")
        for predictor in ["motivation", "support_main", "effect"]:
            coef_deltas.append(abs(float(marked.params[predictor]) - float(base.params[predictor])))

    return pd.DataFrame(
        [
            {
                "진단": "Marker-proxy/Lindell-Whitney sensitivity",
                "marker proxy": marker,
                "평균 abs_delta_r": float(np.mean(corr_deltas)),
                "최대 abs_delta_r": float(np.max(corr_deltas)),
                "평균 abs_delta_B": float(np.mean(coef_deltas)),
                "최대 abs_delta_B": float(np.max(coef_deltas)),
                "해석": "계수 변화 제한적",
            }
        ]
    )


def appendix_item_selection_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q16_rows = [
        {
            "영역": "Q16",
            "문항": "Q16_1",
            "역할": "조직의 AI 강조",
            "최종 포함 여부": "포함",
            "근거": "조직 차원의 AI 활용 강조를 직접 반영하므로 support_main의 핵심 구성요소에 해당함.",
        },
        {
            "영역": "Q16",
            "문항": "Q16_2",
            "역할": "조직의 AI 지원",
            "최종 포함 여부": "포함",
            "근거": "조직의 제도적·실무적 지원을 직접 반영하므로 support_main의 핵심 구성요소에 해당함.",
        },
        {
            "영역": "Q16",
            "문항": "Q16_3",
            "역할": "구성원의 AI 수용 분위기",
            "최종 포함 여부": "포함",
            "근거": "조직 내 AI 친화적 태도와 분위기를 반영하므로 조직지원 인식의 맥락적 요소에 해당함.",
        },
        {
            "영역": "Q16",
            "문항": "Q16_4",
            "역할": "구성원의 실제 AI 사용 분위기",
            "최종 포함 여부": "포함",
            "근거": "조직 내 실제 사용 확산 정도를 반영하므로 support_main의 관측지수 구성에 적절함.",
        },
        {
            "영역": "Q16",
            "문항": "Q16_5",
            "역할": "AI 도입 결과에 대한 평가",
            "최종 포함 여부": "제외",
            "근거": "조직지원의 선행조건이라기보다 AI 도입의 결과지표 성격이 강해 구성개념의 원인-결과 구분을 흐릴 수 있음.",
        },
        {
            "영역": "Q16",
            "문항": "Q16_6",
            "역할": "성과 또는 변화에 대한 평가",
            "최종 포함 여부": "제외",
            "근거": "조직지원 그 자체보다 도입 이후의 평가적 판단에 가까워 내용타당성과 구성개념 순도를 약화시킬 수 있음.",
        },
        {
            "영역": "Q16",
            "문항": "Q16_7",
            "역할": "개인 역량개발 인식",
            "최종 포함 여부": "제외",
            "근거": "조직 수준 인식이 아니라 개인 역량개발 인식에 가까워 support_main과 개념수준이 다르며, 통합 EFA에서도 분리되는 패턴을 보임.",
        },
    ]

    q20_rows = [
        {
            "영역": "Q20",
            "문항": "Q20_1",
            "역할": "업무효율 개선 기대",
            "최종 포함 여부": "메인 포함",
            "근거": "생성형 AI의 긍정적 전략기대를 직접 반영하는 핵심 결과항목으로 메인 종속변수에 포함함.",
        },
        {
            "영역": "Q20",
            "문항": "Q20_2",
            "역할": "의사결정 지원 기대",
            "최종 포함 여부": "메인 포함",
            "근거": "생성형 AI의 활용가치를 의사결정 지원 측면에서 반영하는 핵심 결과항목으로 메인 종속변수에 포함함.",
        },
        {
            "영역": "Q20",
            "문항": "Q20_3",
            "역할": "반복업무 자동화 기대",
            "최종 포함 여부": "메인 포함",
            "근거": "생성형 AI의 자동화 기대를 반영하는 독립적 결과항목으로 메인 종속변수에 포함함.",
        },
        {
            "영역": "Q20",
            "문항": "Q20_4",
            "역할": "일자리 대체 인식",
            "최종 포함 여부": "보조 포함",
            "근거": "긍정적 전략기대라기보다 위협·대체 인식에 가까워 Q20_1~Q20_3과 동일 차원의 합성척도로 보지 않고 supplementary outcome으로 분리함.",
        },
    ]

    return pd.DataFrame(q16_rows), pd.DataFrame(q20_rows)


def print_df(df: pd.DataFrame, decimals: int = 3) -> None:
    out = df.copy()
    if "ΔR² p" in out.columns:
        out["ΔR² p"] = out["ΔR² p"].map(
            lambda x: "" if pd.isna(x) or x == "" else (fmt_p(float(x)) if isinstance(x, (int, float, np.floating)) else str(x))
        )
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else round(float(x), decimals))
    print(out.to_markdown(index=False))
    print()


def main() -> None:
    df = load_data()

    desc = descriptive_table(df)
    rv = reliability_validity_table(df)
    index_reliability = observed_index_reliability_note(df)
    fl_df, htmt_df = reflective_validity_tables(df)
    efa_summary, efa_loadings = integrated_efa(df)
    corr = correlation_table(df)
    hier_effect = hierarchical_models(df, "effect")
    hier_q20_1 = hierarchical_models(df, "Q20_1")
    hier_q20_2 = hierarchical_models(df, "Q20_2")
    hier_q20_3 = hierarchical_models(df, "Q20_3")
    hier_tables = add_holm_to_hierarchical_tables(
        {
            "effect": hier_effect,
            "Q20_1": hier_q20_1,
            "Q20_2": hier_q20_2,
            "Q20_3": hier_q20_3,
        }
    )
    hier_effect = hier_tables["effect"]
    hier_q20_1 = hier_tables["Q20_1"]
    hier_q20_2 = hier_tables["Q20_2"]
    hier_q20_3 = hier_tables["Q20_3"]
    vif = pd.concat([vif_table(df, dv) for dv in ["effect", "Q20_1", "Q20_2", "Q20_3"]], ignore_index=True)
    mediation_results = {dv: mediation_analysis(df, dv) for dv in MAIN_DVS}
    mediation_df = mediation_table(mediation_results)
    beta_df = standardized_beta_table(df)
    mancova_df = mancova_table(df)
    harman_df, q20_4_df = supplementary_table(df)
    marker_cmv_df = marker_proxy_cmv_summary(df)
    q16_appendix_df, q20_appendix_df = appendix_item_selection_tables(df)

    with markdown_output("31_paper_ready_tables.md") as result_path:
        print("# 논문용 통합 결과표\n")
        print(f"- 분석대상: 생성형 AI 사용 경험자 {len(df)}명")
        print("- 연구모형: motivation, support_main(Q16_1~Q16_4 평균) -> effect -> Q20_1~Q20_3")
        print("- 통제변수: gender, rank_code, career_code\n")

        print("## 표 1. 기술통계표\n")
        print_df(desc)
        print("주: 기술통계는 본 연구의 메인 분석에 실제 투입된 구성개념만 제시하였다. 전략적 기대는 합성지수로 처리하지 않고 Q20_1~Q20_3을 개별 결과항목으로 분석하였다.\n")

        print("## 표 2. 구성개념 신뢰도 및 통합 탐색적 요인분석 개요\n")
        print_df(efa_summary)

        print("## 표 3. 통합 탐색적 요인분석 결과(회전된 요인적재량)\n")
        print_df(efa_loadings)
        print("주: 통합 EFA는 motivation(2문항), effect(5문항), 조직지원 관련 문항(Q16_1~Q16_7)을 함께 투입하였다. Q16_5~Q16_7의 적재 패턴이 Q16_1~Q16_4와 구분되는지를 확인하기 위한 목적이다.\n")

        print("## 표 4. 구성개념별 신뢰도 및 타당도 검증 결과\n")
        print_df(rv)
        print(f"주: CR과 AVE는 각 구성개념에 대해 1요인 모형을 적용해 산출한 보조적 지표다. motivation은 2문항 척도이므로 평균 문항간 상관을 함께 해석하는 것이 적절하다. `support_main`은 잠재변수보다 조직 맥락을 나타내는 observed index로 다루므로 CR/AVE 보고 대상에서 제외하였다. 다만 Q16_1~Q16_4로 구성한 `support_main` 관측지수의 내적 일관성은 Cronbach α = {index_reliability['support_alpha']:.3f}, 평균 문항간 상관 = {index_reliability['support_avg_r']:.3f}로 확인하였다. `strategic_expectancy_main(Q20_1~Q20_3 평균)`은 메인 분석에 사용하지 않았으므로 신뢰도·타당도 표에서는 제외하였다. 보조적 CFA 점검에서 `effect` 5문항 단독 모형은 `CFI = .901`, `TLI = .802`, `RMSEA = .236`, `RMSEA 90% CI = [.199, .275]`, `SRMR = .055`로 나타났다. CFI/TLI 및 RMSEA가 관행적 기준을 충족하지 않으므로, CFA는 단일차원성의 결정적 근거가 아니라 제한적 보조 점검 및 한계로 보고한다.\n")

        print("## 표 5. Fornell-Larcker 판별타당도 행렬\n")
        print(fl_df.to_markdown())
        print()
        print("주: 대각선은 각 reflective construct의 AVE 제곱근이며, 비대각선은 구성개념 간 상관계수다. `sqrt(AVE)`가 상관계수보다 크면 판별타당도 기준을 충족한 것으로 본다.\n")

        print("## 표 6. HTMT 비율\n")
        print_df(htmt_df)
        print("주: Henseler et al. (2015)에 따라 `HTMT < .85`를 보수적 기준으로 적용하였다.\n")

        print("## 표 7. 상관관계 분석 결과\n")
        print(corr.to_markdown())
        print()
        print("주: 하삼각 행렬에 Pearson 상관계수를 제시하였다. `* p < .05`, `** p < .01`, `*** p < .001`.\n")

        print("## 표 8. 위계적 회귀분석 결과(effect)\n")
        print_df(hier_effect)
        print("## 표 9. 위계적 회귀분석 결과(Q20_1: 업무효율 개선 기대)\n")
        print_df(hier_q20_1)
        print("## 표 10. 위계적 회귀분석 결과(Q20_2: 의사결정 지원 기대)\n")
        print_df(hier_q20_2)
        print("## 표 11. 위계적 회귀분석 결과(Q20_3: 반복업무 자동화 기대)\n")
        print_df(hier_q20_3)
        print("주: 모형 1은 통제변수(gender, rank_code, career_code)만 포함한 baseline block이며, B/SE/β는 주요 예측변수 중심 표기 원칙에 따라 생략하였다. B, SE, p, 95% CI는 HC3 robust standard errors 기준이다. β는 동일 모형 내 표준화 계수다. Holm-Bonferroni 보정의 동일 족(family)은 동일한 연구질문과 동일한 추론 단계에 속한 Q20_1~Q20_3의 주요 계수군으로 정의하였다. 따라서 모형 2의 `motivation` 및 `support_main` 총효과 6개 검정은 `Q20 총효과 모형(모형2, m=6)`으로, 모형 3의 `motivation`, `support_main`, `effect` 직접효과 9개 검정은 `Q20 직접효과 모형(모형3, m=9)`으로 각각 분리 보정하였다. `Holm critical p`는 각 순위별 보정 후 임계값이며, `Holm 유의`는 순차 Holm 절차를 적용한 최종 유의 여부다. `effect` 매개변수 모형은 Q20 결과변수군에 대한 다중가설 검정이 아니므로 Holm 보정 대상이 아니다. `ΔR² p`는 중첩모형 비교의 보조지표다. H3/H4의 직접효과 가설은 매개변수 `effect`를 포함한 최종모형(모형 3)의 해당 직접경로에 대한 `Holm 유의`를 기준으로 판정하며, 모형 2는 총효과 참고값으로 제시한다. H5/H6의 매개효과는 간접효과의 BCa CI 기준으로 별도 판단한다.\n")

        print("## 표 12. 다중공선성 진단 결과\n")
        print_df(vif)
        print("주: 일반적으로 VIF < 10, Tolerance > .10이면 다중공선성 문제는 크지 않다고 본다.\n")

        print("## 표 13. 매개효과 검증 결과(BCa bootstrap 5,000회)\n")
        print_df(mediation_df)
        print("주: `BCa CI 판단`은 95% BCa 신뢰구간이 0을 포함하는지 여부에 따른 판단이다. Holm-Bonferroni 보정은 총효과 p값에만 적용하였고, 총효과의 다중비교 보정 후 판단은 `Holm 유의` 열을 기준으로 해석한다. 간접효과는 p값 및 Holm 보정 대상이 아니며, BCa bootstrap 신뢰구간을 기준으로 별도 판단하였다. 두 기준은 서로 다른 추론 프레임워크이므로 동일 기준으로 해석하지 않는다(Zhao et al., 2010). 특히 Q20_1에서 `support_main`은 총효과는 비유의하나, 직접효과는 유의한 음(-)의 값, 간접효과는 유의한 양(+)의 값으로 나타나 경쟁적 경로 구조를 보인다. Q20_3에서 `motivation`의 패턴은 사후적으로 `indirect-only mediation`으로 읽을 수 있으나, 총효과와 직접효과가 모두 비유의하고 suppression 가능성을 배제할 수 없으므로 탐색적 해석에 그쳐야 한다.\n")

        print("## 표 14. 독립변수 간 영향력 비교(표준화 계수 기준)\n")
        print_df(beta_df)
        print("주: 표준화 β는 동일 모형 내 영향력 크기 비교를 위한 값이다. `더 큰 독립변수 영향력`은 |β| 기준으로 motivation과 support_main 중 더 큰 값을 표시하였다.\n")

        print("## 표 15. 추가 분석 1: 다중 종속변수 보조 MANCOVA 결과\n")
        print_df(mancova_df)
        print("주: Q20_1~Q20_3 간 상관은 r = .140~.538로 나타나 종속변수 간 공분산을 완전히 무시하기 어렵다. 이에 따라 `Q20_1 + Q20_2 + Q20_3 ~ motivation + support_main + effect + gender + rank_code + career_code` 모형의 보조 MANCOVA를 수행하였다. Pillai's trace 기준에서 motivation, support_main, effect의 다변량 효과가 모두 유의하여, 주요 예측변수가 세 결과항목을 결합한 다변량 결과공간에서도 설명력을 가진다는 점을 확인하였다. 다만 연구가설은 Q20_1~Q20_3 각각의 내용적으로 구분된 전략적 기대에 대한 방향과 매개경로를 검증하므로, 최종 해석은 HC3 OLS와 Holm-Bonferroni 보정 결과를 기준으로 제시한다.\n")

        print("## 표 16. 추가 분석 2: 공통방법편의 진단(Harman 단일요인 검정)\n")
        print_df(harman_df, decimals=2)

        print("## 표 17. 추가 분석 3: Marker-proxy 공통방법편의 민감도 분석\n")
        print_df(marker_cmv_df)
        print("주: 전용 marker variable이 없는 자료 구조를 고려하여 `Q20_4`를 보수적 marker proxy로 사용한 Lindell-Whitney식 민감도 분석이다. `Q20_4`는 완전한 순수 marker가 아니므로, 이 결과는 확정적 CMB 보정보다 추가 강건성 점검으로 해석한다.\n")

        print("## 표 18. 추가 분석 4: 보조 종속변수(Q20_4: 일자리 대체 인식) 결과\n")
        print_df(q20_4_df)

        print("## 표 19. 논문 서술용 핵심 요약\n")
        summary_rows = [
            {
                "주제": "매개변수(effect)",
                "핵심 결과": "motivation과 support_main 모두 effect를 유의하게 정(+)적으로 예측했다.",
            },
            {
                "주제": "Q20_1",
                "핵심 결과": "업무효율 개선 기대에서 support_main은 총효과가 비유의였으나, 직접효과는 음(-), 간접효과는 양(+)의 방향으로 각각 유의하여 직접경로와 간접경로가 상반되는 경쟁적 경로 구조를 보였다.",
            },
            {
                "주제": "Q20_2",
                "핵심 결과": "의사결정 지원 기대는 motivation의 직접·간접효과가 모두 유의했고, support_main은 주로 effect를 통한 간접경로에서 작동했다.",
            },
            {
                "주제": "Q20_3",
                "핵심 결과": "반복업무 자동화 기대는 support_main의 직접효과가 가장 강했고, motivation의 간접경로는 유의했지만 총효과와 직접효과는 비유의하여 indirect-only mediation 또는 suppression 가능성을 포함하는 탐색적 패턴으로 해석할 필요가 있다.",
            },
        ]
        print(pd.DataFrame(summary_rows).to_markdown(index=False))
        print()

        print("# 부록\n")
        print("## 부록표 A2. Q16 문항 선정 및 제외 근거\n")
        print_df(q16_appendix_df)
        print("주: 최종 조직지원 인식 변수인 `support_main`은 Q16_1~Q16_4 평균으로 구성하였다.\n")

        print("## 부록표 A3. Q20 문항의 분석상 위치와 선정 근거\n")
        print_df(q20_appendix_df)
        print("주: Q20_1~Q20_3은 메인 종속변수이며, Q20_4는 보조 종속변수로 분리하여 분석하였다.\n")

    print(f"Completed: {result_path}")


if __name__ == "__main__":
    main()
