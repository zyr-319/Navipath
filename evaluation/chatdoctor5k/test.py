import csv
csv.field_size_limit(10**8)

# 设置你想检查的CSV路径
input_file = 'output_NER_chatgpt_myall.csv'

# 设置阈值（默认csv模块限制是131072字符）
FIELD_LIMIT = 131072

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)

    row_num = 1  # 从1开始（跳过header）
    for row in reader:
        row_num += 1
        for i, cell in enumerate(row):
            if len(cell) > FIELD_LIMIT:
                print(f"🚨 第 {row_num} 行，第 {i+1} 列 超出限制（{len(cell)} 字符）")
                if len(row) > 0:
                    print(f"➡️ 对应第1列内容: {row[0]}")
                print("-" * 80)
