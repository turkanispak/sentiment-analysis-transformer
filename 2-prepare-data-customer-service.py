import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import tiktoken  # pip install tiktoken  # For nanoGPT encoding

# ========== Logging Setup ==========
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

data_dir = os.path.join("data", "customer_service")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(data_dir, f"data_prep_log_{timestamp}.txt")
sys.stdout = Logger(log_path)

# ========== Settings ==========
MAX_CHARS = 1024
DELIMITER = "\n### SENTIMENT: "
VAL_RATIO = 0.1
RANDOM_STATE = 42

# ========== Load and Upsample ==========
train_path = os.path.join(data_dir, "train.csv")
df = pd.read_csv(train_path)
df.columns = df.columns.str.strip()

print("\n[INFO] Original class distribution:")
print(df['customer_sentiment'].value_counts())

max_class_size = df['customer_sentiment'].value_counts().max() # Upsample all classes to the max class size
df_upsampled = pd.concat([
    resample(df[df['customer_sentiment'] == label],
             replace=True,
             n_samples=max_class_size,
             random_state=RANDOM_STATE)
    for label in df['customer_sentiment'].unique()
])
df_upsampled = df_upsampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\n[INFO] After upsampling:")
print(df_upsampled['customer_sentiment'].value_counts())

# ========== Format Text Samples ==========
def format_sample(row):
    convo = str(row['conversation']).strip().replace('\n', ' ')[:MAX_CHARS]
    label = row['customer_sentiment'].strip().lower()
    return f"{convo}{DELIMITER}{label}\n"

df_upsampled['formatted'] = df_upsampled.apply(format_sample, axis=1)
all_texts = df_upsampled['formatted'].tolist()

# ========== Train/Val Split ==========
train_texts, val_texts = train_test_split(
    all_texts,
    test_size=VAL_RATIO,
    stratify=df_upsampled['customer_sentiment'],
    random_state=RANDOM_STATE
)

# ========== Save GPT-2 Text Files ==========
with open(os.path.join(data_dir, "train.txt"), "w", encoding="utf-8") as f:
    f.writelines(train_texts)

with open(os.path.join(data_dir, "val.txt"), "w", encoding="utf-8") as f:
    f.writelines(val_texts)

print(f"\n[INFO] Saved {len(train_texts)} GPT-2 training samples.")
print(f"[INFO] Saved {len(val_texts)} GPT-2 validation samples.")

# ========== Save nanoGPT Binary Files ==========
enc = tiktoken.get_encoding("gpt2")

def encode_and_save(input_txt, output_bin):
    with open(input_txt, "r", encoding="utf-8") as f:
        data = f.read()
    tokens = enc.encode(data)
    np.array(tokens, dtype=np.uint16).tofile(output_bin)
    print(f"[INFO] Encoded {input_txt} → {output_bin} ({len(tokens)} tokens)")

encode_and_save(os.path.join(data_dir, "train.txt"), os.path.join(data_dir, "train.bin"))
encode_and_save(os.path.join(data_dir, "val.txt"), os.path.join(data_dir, "val.bin"))

# ========== Save Test Files ==========
test_path = os.path.join(data_dir, "test.csv")
df_test = pd.read_csv(test_path)
df_test.columns = df_test.columns.str.strip()

assert 'conversation' in df_test.columns and 'customer_sentiment' in df_test.columns, \
    "[ERROR] Missing required columns in test.csv"

with open(os.path.join(data_dir, "test.txt"), "w", encoding="utf-8") as text_file, \
     open(os.path.join(data_dir, "labels_test.txt"), "w", encoding="utf-8") as label_file:
    for _, row in df_test.iterrows():
        text = str(row['conversation']).strip().replace('\n', ' ')
        label = row['customer_sentiment'].strip().lower()
        text_file.write(text + "\n")
        label_file.write(label + "\n")

print("\n[INFO] Saved test.txt and labels_test.txt for test set.")

# ========== Wrap Up ==========
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
print(f"[DONE] Data preparation complete. Log saved to {log_path}")
