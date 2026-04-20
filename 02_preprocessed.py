import numpy as np
import pandas as pd

from result_utils import markdown_output


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
    "보통이다.1": 3,
    "동의하지 않는다": 2,
    "전혀 동의하지 않는다": 1,
    "다소 동의하지 않는다": 2,
    "약간 동의한다": 4,
}
yn_map = {"예": 1, "아니오": 0}

df["Q3"] = df["Q3"].map(yn_map)

q4_cols = [f"Q4_{i}" for i in range(1, 10)]
for col in q4_cols:
    # Multi-response checkbox items are stored as the selected label text; any non-null entry means checked.
    df[col] = df[col].notna().astype(int)
df["ai_task_count"] = df[q4_cols].sum(axis=1)

likert_cols = [col for col in df.columns if col.startswith(("Q7_", "Q9_", "Q16_", "Q20_"))]
for col in likert_cols:
    df[col] = df[col].map(likert_5).astype("float")

df["Q21"] = df["Q21"].astype(str).str.strip()
df["Q22"] = df["Q22"].astype(str).str.strip()
df["Q23"] = df["Q23"].astype(str).str.strip()

gender_map = {"남성": 1, "여성": 0, "남자": 1, "여자": 0}
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

drop_cols = [*q4_cols, "Q21", "Q22", "Q23"]
df = df.drop(columns=drop_cols, errors="ignore")

df.to_csv("chatbot_output_selected_preprocessed.csv", index=False, encoding="utf-8-sig")


def freq_percent(series, name, total_n):
    freq = series.value_counts().sort_index()
    percent = round(freq / total_n * 100, 1)
    result = pd.DataFrame({"N": freq, "%": percent})
    print("\n====================================")
    print(f"{name} distribution (N={total_n})")
    print("====================================")
    print(result)
    return result


with markdown_output("02_preprocessed.md") as result_path:
    print("# 02 Preprocessing Summary\n")
    print("Built the analysis-ready dataset and checked the main control variables.\n")
    print("## Dataset output\n")
    print("- Output file: `chatbot_output_selected_preprocessed.csv`")
    print(f"- Total respondents: {len(df)}")
    print(f"- AI users: {int(df['Q3'].fillna(0).sum())}")
    print(f"- AI non-users: {int((df['Q3'] == 0).sum())}\n")

    print("## Preview of key derived variables")
    print("```text")
    print(df[["gender", "rank_code", "career_code", "ai_task_count"]].head())
    print("```\n")

    print("## Overall sample characteristics")
    freq_percent(df["gender"], "gender", len(df))
    freq_percent(df["rank_code"], "rank", len(df))
    freq_percent(df["career_code"], "career", len(df))
    freq_percent(df["SQ1"], "age_raw", len(df))
    freq_percent(df["SQ4"], "organization_raw", len(df))

    print("\n## Key note")
    print("- `ai_task_count` is computed as the number of checked Q4 task-use categories.")
    print("- The preprocessed dataset is now ready for all downstream scripts.")

print(f"완료: {result_path} 생성")
