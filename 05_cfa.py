import pandas as pd
from semopy import Model, calc_stats

# ============================================================
# 1. 데이터 로드
# ============================================================

df = pd.read_csv("chatbot_output_selected_preprocessed.csv")

# AI 활용자만
df_ai = df[df["Q3"] == 1].copy()

# ============================================================
# 2. 문항 정의
# ============================================================

cols_9_voluntary = ["Q9_3", "Q9_4"]
cols_7 = [f"Q7_{i}" for i in range(1, 6)]
cols_16 = [f"Q16_{i}" for i in range(1, 8)]
cols_20 = [f"Q20_{i}" for i in range(1, 5)]

efa_cols_refined = (
    cols_9_voluntary +
    cols_7 +
    cols_16 +
    cols_20
)

df_cfa = df_ai[efa_cols_refined].dropna()

print(f"CFA 유효표본 수: {len(df_cfa)}")

# ============================================================
# 3. CFA 모델 정의
# ============================================================

model_desc = """
Voluntary =~ Q9_3 + Q9_4
WorkEffect =~ Q7_1 + Q7_2 + Q7_3 + Q7_4 + Q7_5
OrgSupport =~ Q16_1 + Q16_2 + Q16_3 + Q16_4 + Q16_5 + Q16_6 + Q16_7
Strategic =~ Q20_1 + Q20_2 + Q20_3 + Q20_4
"""

model = Model(model_desc)
model.fit(df_cfa)

# ============================================================
# 4. 적합도 지표 출력
# ============================================================

stats = calc_stats(model)

print("\n===== Fit Indices =====")
print("Chi-square:", stats['chi2'].values[0])
print("df:", stats['DoF'].values[0])
print("CFI:", stats['CFI'].values[0])
print("TLI:", stats['TLI'].values[0])
print("RMSEA:", stats['RMSEA'].values[0])
print("SRMR: Not natively provided by semopy")

# ============================================================
# 5. 요인적재량 출력
# ============================================================

estimates = model.inspect()
loadings = estimates[estimates['op'] == '~']

print("\n===== Factor Loadings =====")
print(loadings[['lval', 'rval', 'Estimate']])