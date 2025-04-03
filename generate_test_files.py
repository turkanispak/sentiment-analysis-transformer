# generate_test_files.py

import pandas as pd
import os

# Set correct path to test.csv
csv_path = r"C:\Users\turka\OneDrive\Desktop\DI725_Assignment1\data\customer_service\test.csv"
output_dir = "data_test"

os.makedirs(output_dir, exist_ok=True)

# Read the test.csv file
df = pd.read_csv(csv_path)

# Ensure required columns exist
assert 'conversation' in df.columns and 'customer_sentiment' in df.columns, "Missing required columns in test.csv"

# Save test.txt and labels_test.txt
with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as text_file, \
     open(os.path.join(output_dir, "labels_test.txt"), "w", encoding="utf-8") as label_file:
    for _, row in df.iterrows():
        text = row['conversation'].strip().replace('\n', ' ')
        label = row['customer_sentiment'].strip().lower()
        text_file.write(text + "\n")
        label_file.write(label + "\n")

print("test.txt and labels_test.txt successfully generated in 'data_test/'")
