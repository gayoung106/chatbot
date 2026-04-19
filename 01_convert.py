import pandas as pd

# 변환할 파일 경로 입력
input_path = "chatbot_input.SAV"      # 원본 SPSS 파일 (같은 폴더)
output_path = "chatbot_output.csv"    # 변환 후 저장할 CSV 파일

# .sav 파일 읽기
df = pd.read_spss(input_path)

# CSV 파일로 저장 (UTF-8 인코딩)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(" 변환 완료! CSV 파일이 생성되었습니다:", output_path)
