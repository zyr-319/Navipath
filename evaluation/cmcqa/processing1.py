import pandas as pd

# 文件路径
file1_path = '1.csv'
file2_path = '2.csv'
file3_path = '3.csv'

# 读取文件
df1 = pd.read_csv(file1_path, encoding='utf-8-sig')
df2 = pd.read_csv(file2_path, encoding='utf-8-sig')
df3 = pd.read_csv(file3_path, encoding='utf-8-sig')

# 分数列
score_cols = df1.columns[3:6]

# 提取前三列：Question, Label, Mindmap
base_cols = df1.columns[:3]

# 转为数值以便求和
df1_scores = df1[score_cols].apply(pd.to_numeric, errors='coerce')
df2_scores = df2[score_cols].apply(pd.to_numeric, errors='coerce')
df3_scores = df3[score_cols].apply(pd.to_numeric, errors='coerce')

# 总分
df1_total = df1_scores.sum(axis=1)
df2_total = df2_scores.sum(axis=1)
df3_total = df3_scores.sum(axis=1)

# 保存结果
best_rows = []
worst_rows = []

# 修正行的收集方式，确保只保留前三列（内容）
best_rows = []
worst_rows = []

for i in range(len(df1)):
    rows = [df1.iloc[i, :3].values, df2.iloc[i, :3].values, df3.iloc[i, :3].values]
    totals = [df1_total[i], df2_total[i], df3_total[i]]
    best_idx = totals.index(max(totals))
    worst_idx = totals.index(min(totals))
    best_rows.append(rows[best_idx])
    worst_rows.append(rows[worst_idx])

# 重新构造 DataFrame，明确指定列
best_df = pd.DataFrame(best_rows, columns=['Question', 'Label', 'Navipath'])
worst_df = pd.DataFrame(worst_rows, columns=['Question', 'Label', 'Mindmap'])

# 重命名第三列
best_df.columns = ['Question', 'Label', 'Navipath']
worst_df.columns = ['Question', 'Label', 'Mindmap']

# 保存
best_df.to_csv('output_navipath_all_processing.csv', index=False, encoding='utf-8-sig')
worst_df.to_csv('output_mindmap_processing.csv', index=False, encoding='utf-8-sig')

print("✅ 处理完成：output_navipath_all_processing.csv 的第3列为 Navipath，output_mindmap_processing.csv 的为 Mindmap")
