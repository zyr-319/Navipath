import pandas as pd

# 读取文件
csv1 = pd.read_csv('3.csv', encoding='utf-8-sig')
csv2 = pd.read_csv('1.csv', encoding='utf-8-sig')

# 获取第一列内容
questions1 = set(csv1.iloc[:, 0].astype(str))
questions2 = set(csv2.iloc[:, 0].astype(str))

# 找出 csv2 中有但 csv1 中没有的
missing_questions = questions2 - questions1

# 查找这些 question 在 csv2 中的行号及内容
missing_rows = csv2[csv2.iloc[:, 0].astype(str).isin(missing_questions)].copy()
missing_rows['行号_in_csv2'] = missing_rows.index

# 仅保留行号和第一列内容
result = missing_rows[['行号_in_csv2', missing_rows.columns[0]]]

# 输出结果
result.to_csv('missing_from_csv1.csv', index=False, encoding='utf-8-sig')
print(f"✅ 共找到 {len(result)} 条在 csv2 中但不在 csv1 中的问题，已保存为 missing_from_csv1.csv")
