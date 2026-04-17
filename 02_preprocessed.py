import numpy as np
import pandas as pd

# ============================================================
# Build the analysis-ready dataset used across the project.
# ============================================================

df = pd.read_csv("chatbot_output.csv")

cols_to_keep = (
    ["SQ1", "SQ4", "Q3"]
    + [f"Q4_{i}" for i in range(1, 10)]
    + [f"Q7_{i}" for i in range(1, 6)]
    + [f"Q9_{i}" for i in range(3, 5)]
    + [f"Q16_{i}" for i in range(1, 8)]
    + [f"Q20_{i}" for i in range(1, 5)]
    + ["Q21", "Q22", "Q23"]
)
df = df[cols_to_keep]

likert_5 = {
    "매우 그렇다": 5,
    "약간 그렇다": 4,
    "보통이다": 3,
    "별로 그렇지 않다": 2,
    "전혀 그렇지 않다": 1,
    "매우 동의한다": 5,
    "동의한다": 4,
    "동의하지 않는다": 2,
    "전혀 동의하지 않는다": 1,
    "다소 동의하지 않는다": 2,
    "약간 동의한다": 4,
}
yn_map = {"예": 1, "아니오": 0}

df["Q3"] = df["Q3"].map(yn_map)

q4_cols = [f"Q4_{i}" for i in range(1, 10)]
df[q4_cols] = df[q4_cols].replace(np.nan, 0)
for col in q4_cols:
    df[col] = df[col].apply(lambda x: 1 if str(x).strip() in ["1", "예", "True", "O", "o", "Y"] else 0)
df["ai_task_count"] = df[q4_cols].sum(axis=1)

likert_cols = [col for col in df.columns if col.startswith(("Q7_", "Q9_", "Q16_", "Q20_"))]
for col in likert_cols:
    df[col] = df[col].map(likert_5).astype("float")

df["Q21"] = df["Q21"].astype(str).str.strip()
df["Q22"] = df["Q22"].astype(str).str.strip()
df["Q23"] = df["Q23"].astype(str).str.strip()

gender_map = {"여성": 1, "남성": 0, "여자": 1, "남자": 0}
rank_map = {"9급": 1, "8급": 2, "7급": 3, "6급": 4, "5급": 5, "4급": 6, "3급 이상": 7}
career_map = {
    "5년 이하": 1,
    "6-10년": 2,
    "11-20년": 3,
    "21-30년": 4,
    "31년 이상": 5,
    "6~10년": 2,
    "11~20년": 3,
    "21~30년": 4,
}

df["gender"] = df["Q21"].map(gender_map)
df["rank_code"] = df["Q22"].map(rank_map)
df["career_code"] = df["Q23"].map(career_map)

df[q4_cols] = df[q4_cols].fillna(0)

drop_cols = [*q4_cols, "Q21", "Q22", "Q23"]
df = df.drop(columns=drop_cols, errors="ignore")

df.to_csv("chatbot_output_selected_preprocessed.csv", index=False, encoding="utf-8-sig")

print("Preprocessing complete.")
print(f"Total responses: {len(df)}")
print(df[["gender", "rank_code", "career_code", "ai_task_count"]].head())


def freq_percent(series, name, total_n):
    freq = series.value_counts().sort_index()
    percent = round(freq / total_n * 100, 1)
    result = pd.DataFrame({"N": freq, "%": percent})
    print("\n====================================")
    print(f"{name} distribution (N={total_n})")
    print("====================================")
    print(result)
    return result


print("\n\n###############################")
print("Full sample characteristics")
print("###############################")

freq_percent(df["gender"], "gender", len(df))
freq_percent(df["rank_code"], "rank", len(df))
freq_percent(df["career_code"], "career", len(df))
freq_percent(df["SQ1"], "age_raw", len(df))
freq_percent(df["SQ4"], "organization_raw", len(df))
