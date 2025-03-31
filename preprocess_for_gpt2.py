import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os

# === Settings ===
MAX_CHARS = 1024
DELIMITER = "\n### SENTIMENT: "
VAL_RATIO = 0.1
RANDOM_STATE = 42

# === Load Data ===
data_path = os.path.join("data", "customer_service")
df = pd.read_csv(os.path.join(data_path, "train.csv"))
df.columns = df.columns.str.strip()

# === Upsample minority class (positive) ===
print("\nOriginal class distribution:")
print(df['customer_sentiment'].value_counts())

max_class_size = df['customer_sentiment'].value_counts().max()
upsampled_df = pd.concat([
    resample(df[df['customer_sentiment'] == label],
             replace=True,
             n_samples=max_class_size,
             random_state=RANDOM_STATE)
    for label in df['customer_sentiment'].unique()
])

upsampled_df = upsampled_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nAfter upsampling:")
print(upsampled_df['customer_sentiment'].value_counts())

# === Preprocess function ===
def format_sample(row):
    convo = str(row['conversation']).strip()[:MAX_CHARS]
    label = row['customer_sentiment']
    return f"{convo}{DELIMITER}{label}\n"

# === Create preprocessed text list ===
upsampled_df['formatted'] = upsampled_df.apply(format_sample, axis=1)
all_texts = upsampled_df['formatted'].tolist()

# === Split Train/Val ===
train_texts, val_texts = train_test_split(
    all_texts, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=upsampled_df['customer_sentiment']
)

# === Save to text files ===
os.makedirs("data_gpt2", exist_ok=True)

with open("data_gpt2/train.txt", "w", encoding="utf-8") as f:
    f.writelines(train_texts)

with open("data_gpt2/val.txt", "w", encoding="utf-8") as f:
    f.writelines(val_texts)

# === Check output ===
print(f"\nSaved {len(train_texts)} training samples and {len(val_texts)} validation samples.")
print(f"First few examples:\n---\n{train_texts[0][:300]}...\n---")
