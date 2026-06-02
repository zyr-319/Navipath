import csv
csv.field_size_limit(10**8)
from evaluate import load
from tqdm import tqdm

# 初始化 BERTScore
bertscore = load("bertscore")

# 文件路径
input_file = 'output_navipath_processing.csv'
output_file = 'output_navipath_processing_acc.csv'

# input_file = 'output_mindmap_processing.csv'
# output_file = 'output_mindmap_processing_acc.csv'

# 累计指标
total_precision = 0
total_recall = 0
total_f1 = 0
num_rows = 0

# 打开文件进行处理
with open(input_file, 'r', newline="", encoding='utf-8-sig') as f_input, open(output_file, 'w', newline='', encoding='utf-8-sig') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    # 读取并扩展表头
    header = next(reader)
    header.extend(["precision", "recall", "f1"])
    writer.writerow(header)

    # 使用 tqdm 添加进度条
    for row in tqdm(reader, desc="Calculating BERTScore", ncols=100):
        # ✅ 若第三列为空或不存在，跳过该行，不终止处理
        if len(row) < 3 or not row[2].strip():
            print("⚠️ 跳过一行（第三列为空）：", row)
            continue

        question = row[0]
        label = row[1]
        output_text = row[2]

        # 准备输入 BERTScore 的内容
        references = [label.strip()]
        predictions = [output_text.strip()]

        # 计算 BERTScore
        results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")

        # 解析分数
        precision = results["precision"][0]
        recall = results["recall"][0]
        f1_score = results["f1"][0]

        # 累加
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score
        num_rows += 1

        # 写入结果
        row.extend([precision, recall, f1_score])
        writer.writerow(row)

    # 写入平均值
    avg_precision = total_precision / num_rows if num_rows > 0 else 0
    avg_recall = total_recall / num_rows if num_rows > 0 else 0
    avg_f1 = total_f1 / num_rows if num_rows > 0 else 0

    writer.writerow(["Averages", "", "", avg_precision, avg_recall, avg_f1])
    print(f"✅ 共处理 {num_rows} 条记录，平均 Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
