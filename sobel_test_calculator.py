import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import norm
from result_utils import markdown_output

def calc_sobel(a, sa, b, sb):
    sobel_z = (a * b) / np.sqrt((b**2 * sa**2) + (a**2 * sb**2))
    p_value = 2 * (1 - norm.cdf(np.abs(sobel_z)))
    return sobel_z, p_value

def main():
    df = pd.read_csv("chatbot_output_selected_preprocessed.csv")
    df = df[df["Q3"] == 1].copy()
    df["motivation"] = df[["Q9_3", "Q9_4"]].mean(axis=1)
    df["effect"] = df[[f"Q7_{i}" for i in range(1, 6)]].mean(axis=1)
    df["support_main"] = df[[f"Q16_{i}" for i in range(1, 5)]].mean(axis=1)
    
    DVS = ["Q20_1", "Q20_2", "Q20_3"]
    DV_LABELS = {
        "Q20_1": "업무효율 개선 기대",
        "Q20_2": "의사결정 지원 기대",
        "Q20_3": "반복업무 자동화 기대",
    }
    
    with markdown_output("sobel_test_results.md") as result_path:
        print("# Sobel Test Results for Mediation Paths\n")
        
        for dv in DVS:
            data = df.dropna(
                subset=["motivation", "support_main", "effect", dv, "gender", "rank_code", "career_code"]
            ).copy()
            
            mediator_model = smf.ols(
                "effect ~ motivation + support_main + gender + rank_code + career_code",
                data=data,
            ).fit(cov_type="HC3")
            
            direct_model = smf.ols(
                f"{dv} ~ motivation + support_main + effect + gender + rank_code + career_code",
                data=data,
            ).fit(cov_type="HC3")
            
            # For Motivation -> Effect -> DV
            a_mot = mediator_model.params['motivation']
            sa_mot = mediator_model.bse['motivation']
            
            b = direct_model.params['effect']
            sb = direct_model.bse['effect']
            
            sobel_z_mot, p_mot = calc_sobel(a_mot, sa_mot, b, sb)
            
            # For Support -> Effect -> DV
            a_sup = mediator_model.params['support_main']
            sa_sup = mediator_model.bse['support_main']
            
            sobel_z_sup, p_sup = calc_sobel(a_sup, sa_sup, b, sb)
            
            print(f"## {dv}: {DV_LABELS[dv]}\n")
            print("### IV: motivation -> M: effect -> DV")
            print(f"- a = {a_mot:.4f} (SE = {sa_mot:.4f})")
            print(f"- b = {b:.4f} (SE = {sb:.4f})")
            print(f"- Sobel Z = {sobel_z_mot:.4f}")
            print(f"- p-value = {p_mot:.4e}\n")
            
            print("### IV: support_main -> M: effect -> DV")
            print(f"- a = {a_sup:.4f} (SE = {sa_sup:.4f})")
            print(f"- b = {b:.4f} (SE = {sb:.4f})")
            print(f"- Sobel Z = {sobel_z_sup:.4f}")
            print(f"- p-value = {p_sup:.4e}\n")

    print(f"Sobel test results saved to: {result_path}")

if __name__ == "__main__":
    main()
