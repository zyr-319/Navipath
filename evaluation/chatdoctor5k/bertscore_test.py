from evaluate import load
import csv

# Initialize BERTScore
bertscore = load("bertscore")

# Input and output file paths
input_file = '333.csv'
output_file = 'output_bert_score_test.csv'

# Variables to accumulate precision, recall, and F1 score
total_precision = 0
total_recall = 0
total_f1 = 0
num_rows = 0

# Maximum number of rows to process
max_rows_to_process = 10

# Open input and output files
with open(input_file, 'r', newline="") as f_input, open(output_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    # Read the header from the input file and prepare the output file header
    header = next(reader)
    header.extend(["output_precision", "output_recall", "output_f1"])
    writer.writerow(header)

    # Initialize a counter to keep track of processed rows
    row_count = 0

    # Process each row in the input file
    for row in reader:
        if row_count >= max_rows_to_process:
            break

        question = row[0]
        label = row[1]
        output_text = row[2]  # Assuming the output text is in the third column

        # The reference text (correct answers) and the model's output
        references = [label.strip()]
        predictions = [output_text.strip()]

        # Compute BERTScore
        results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")

        # Extract precision, recall, and F1 score
        precision = results["precision"][0]
        recall = results["recall"][0]
        f1_score = results["f1"][0]

        # Accumulate precision, recall, and F1 score
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score
        num_rows += 1

        # Append BERTScore results to the row
        row.extend([precision, recall, f1_score])

        # Write the modified row to the output file
        writer.writerow(row)

        # Increment the row counter
        row_count += 1

    # Calculate average precision, recall, and F1 score
    avg_precision = total_precision / num_rows if num_rows > 0 else 0
    avg_recall = total_recall / num_rows if num_rows > 0 else 0
    avg_f1 = total_f1 / num_rows if num_rows > 0 else 0

    # Write the average values as the last row
    writer.writerow(["Averages", "", "", avg_precision, avg_recall, avg_f1])
