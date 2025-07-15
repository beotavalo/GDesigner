import json
dataset_path = r"datasets\gsm8k\gsm8k.jsonl"

count = 0
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1

print(f"Number of questions in the dataset: {count}")
