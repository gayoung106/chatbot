import pandas as pd
import numpy as np

# 1️⃣ CSV 로드
df = pd.read_csv("chatbot_output.csv")
print(df.columns.tolist())