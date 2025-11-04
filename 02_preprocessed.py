import pandas as pd
import numpy as np

# ===========================================
# 1️ CSV 로드
# ===========================================
df = pd.read_csv("chatbot_output.csv")

# ===========================================
# 2️ 연구에 필요한 컬럼만 선택
# ===========================================
cols_to_keep = (
    ["Q3"] +
    [f"Q4_{i}" for i in range(1, 10)] +           # 복수응답 (업무유형)
    [f"Q7_{i}" for i in range(1, 6)] +            # 업무효과
    [f"Q9_{i}" for i in range(1, 5)] +            # 활용동기
    [f"Q16_{i}" for i in range(1, 8)] +           # 조직지원
    [f"Q20_{i}" for i in range(1, 5)] +           # 전략기대
    ["Q21", "Q22", "Q23"]                         # 통제변수
)
df = df[cols_to_keep]

# ===========================================
# 3️ 공통 변환 사전
# ===========================================
likert_5 = {
    "매우 그렇다": 5, "약간 그렇다": 4, "보통이다": 3,
    "별로 그렇지 않다": 2, "전혀 그렇지 않다": 1,
    "매우 동의한다": 5, "동의한다": 4, "보통이다": 3,
    "동의하지 않는다": 2, "전혀 동의하지 않는다": 1
}
yn_map = {"예": 1, "아니오": 0}

# ===========================================
# 4️ 문항3 (AI 경험) 변환
# ===========================================
df["Q3"] = df["Q3"].map(yn_map)

# ===========================================
# 5️ 문항4 (복수응답 업무유형) 0/1 변환 + 총합 변수 생성
# ===========================================
q4_cols = [f"Q4_{i}" for i in range(1, 10)]
df[q4_cols] = df[q4_cols].replace(np.nan, 0)

for col in q4_cols:
    df[col] = df[col].apply(lambda x: 1 if str(x).strip() in ["1", "예", "True", "O", "o", "✓"] else 0)

#  파생 변수: 선택된 항목 개수 (AI 활용 업무 개수)
df["ai_task_count"] = df[q4_cols].sum(axis=1)

# ===========================================
# 6️ Likert형 문항 변환 (7, 9, 16, 20)
# ===========================================
likert_cols = [col for col in df.columns if col.startswith(("Q7_", "Q9_", "Q16_", "Q20_"))]
for col in likert_cols:
    df[col] = df[col].map(likert_5)

# ===========================================
# 7️ 통제변수 숫자화 (성별, 직급, 근무연수)
# ===========================================
df["Q21"] = df["Q21"].astype(str).str.strip()
df["Q22"] = df["Q22"].astype(str).str.strip()
df["Q23"] = df["Q23"].astype(str).str.strip()

gender_map = {"남성": 1, "여성": 0, "남자": 1, "여자": 0}
rank_map = {"9급": 1, "8급": 2, "7급": 3, "6급": 4, "5급": 5, "4급": 6, "3급 이상": 7}
career_map = {
    "5년 이하": 1, "6-10년": 2, "11-20년": 3, "21-30년": 4, "31년 이상": 5,
    "6~10년": 2, "11~20년": 3, "21~30년": 4
}

df["gender"] = df["Q21"].map(gender_map)
df["rank_code"] = df["Q22"].map(rank_map)
df["career_code"] = df["Q23"].map(career_map)

# ===========================================
# 8️ 결측값 처리
# ===========================================
df = df.fillna(0)

# ===========================================
# 9️ AI 활용 경험자 필터링
# ===========================================
df_ai_users = df[df["Q3"] == 1].copy()

# ===========================================
#  불필요한 원본 컬럼 제거 (한글 텍스트 포함된 컬럼)
# ===========================================
drop_cols = [*q4_cols, "Q21", "Q22", "Q23"]
df = df.drop(columns=drop_cols, errors="ignore")
df_ai_users = df_ai_users.drop(columns=drop_cols, errors="ignore")

# ===========================================
# ⑪ CSV 저장 (utf-8-sig로 한글 포함)
# ===========================================
df.to_csv("chatbot_output_selected_preprocessed.csv", index=False, encoding="utf-8-sig")
df_ai_users.to_csv("chatbot_output_ai_users_selected.csv", index=False, encoding="utf-8-sig")

# ===========================================
# ⑫ 결과 확인
# ===========================================
print(" 전처리 완료!")
print(f"전체 응답자 수: {len(df)}명, AI 경험자 수: {len(df_ai_users)}명")
print("통제변수 및 Q4 처리 예시:")
print(df[["gender", "rank_code", "career_code", "ai_task_count"]].head())
