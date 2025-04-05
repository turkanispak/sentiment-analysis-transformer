import os
import sys
import re
import json
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from transformers import GPT2Tokenizer
from model import GPTConfig, GPT

# ========== SETTINGS ==========
model_dir = 'out-gpt2-finetune' # out-gpt2-finetune out-customer
ckpt_path = os.path.join(model_dir, 'ckpt.pt')
test_text_path = os.path.join("data", "customer_service", "test.txt")
test_labels_path = os.path.join("data", "customer_service", "labels_test.txt")

output_json_path = os.path.join(model_dir, f"results_{os.path.basename(model_dir)}.json")
output_log_path = os.path.join(model_dir, f"log_{os.path.basename(model_dir)}.txt")
output_plot_path = os.path.join(model_dir, f"confusion_matrix_{os.path.basename(model_dir)}.png")

valid_labels = ["positive", "negative", "neutral"]

# ========== LOGGING ==========
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

sys.stdout = Logger(output_log_path)

# ========== LOAD MODEL ==========
print(f"\n[{datetime.now()}] Loading model from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# ========== LOAD TOKENIZER ==========
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ========== LOAD TEST DATA ==========
with open(test_text_path, "r", encoding="utf-8") as f:
    test_texts = [line.strip() for line in f.readlines()]
with open(test_labels_path, "r", encoding="utf-8") as f:
    test_labels = [line.strip().lower() for line in f.readlines()]

assert len(test_texts) == len(test_labels), "Mismatch between test data and labels"

# ========== SENTIMENT EXTRACTOR ==========
def extract_sentiment(text):
    text = text.lower()
    for label in valid_labels:
        if re.search(rf"\b{label}\b", text):
            return label
    return "unknown"

# ========== INFERENCE ==========
print(f"\n[{datetime.now()}] Running inference on {len(test_texts)} samples...")
predictions = []

for idx, (text, true_label) in enumerate(zip(test_texts, test_labels)):
    prompt = text + "\n### SENTIMENT:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.5,
            top_k=1,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    after_prompt = decoded.split("### SENTIMENT:")[-1].strip()
    predicted = extract_sentiment(after_prompt)
    predictions.append(predicted)

    # === Per-sample print ===
    print("\n------------------------------")
    print(f"[Sample {idx + 1}]")
    print(f"Conversation:\n{text[:300]}")  # Optional truncate
    print(f"Actual Sentiment:    {true_label}")
    print(f"Model Raw Output:\n{decoded.strip()}")
    print(f"Extracted Prediction: {predicted}")


# ========== FILTER OUT UNKNOWN ==========
filtered_true = []
filtered_pred = []

for true_label, pred_label in zip(test_labels, predictions):
    if true_label in valid_labels and pred_label in valid_labels:
        filtered_true.append(true_label)
        filtered_pred.append(pred_label)

print(f"[INFO] Kept {len(filtered_true)} valid samples after removing 'unknown'")

# ========== METRICS ==========
report = classification_report(
    filtered_true, filtered_pred, labels=valid_labels, output_dict=True, zero_division=0
)
conf_matrix = confusion_matrix(filtered_true, filtered_pred, labels=valid_labels)

# Extra global metrics
accuracy = accuracy_score(filtered_true, filtered_pred)
macro_f1 = f1_score(filtered_true, filtered_pred, average='macro', zero_division=0)
micro_f1 = f1_score(filtered_true, filtered_pred, average='micro', zero_division=0)
macro_precision = precision_score(filtered_true, filtered_pred, average='macro', zero_division=0)
macro_recall = recall_score(filtered_true, filtered_pred, average='macro', zero_division=0)

# ========== PACKAGE RESULTS ==========
results = {
    "classification_report": report,
    "confusion_matrix": {
        "labels": valid_labels,
        "matrix": conf_matrix.tolist()
    },
    "overall_metrics": {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    }
}

# ========== PRINT TO CONSOLE (Human-readable) ==========
print("\n===== CLASSIFICATION REPORT =====")
for label in valid_labels:
    metrics = report[label]
    print(f"{label.upper()}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, Support={metrics['support']}")

print("\n===== OVERALL METRICS =====")
for k, v in results['overall_metrics'].items():
    print(f"{k}: {v:.4f}")

print("\n===== CONFUSION MATRIX =====")
print(f"Labels: {valid_labels}")
print(np.array(conf_matrix))

# ========== SAVE RESULTS ==========
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nJSON results saved to: {output_json_path}")

# ========== SAVE CONFUSION MATRIX PLOT ==========
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=valid_labels, yticklabels=valid_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(output_plot_path)
plt.close()

print(f"Confusion matrix plot saved to: {output_plot_path}")

# ========== GROUPED BAR PLOT FOR CLASS METRICS ==========

barplot_path = os.path.join(model_dir, f"metrics_barplot_{os.path.basename(model_dir)}.png")

# Prepare data
metric_names = ["precision", "recall", "f1-score"]
plot_data = []

for cls in valid_labels:
    for metric in metric_names:
        plot_data.append({
            "Class": cls.capitalize(),
            "Metric": metric.capitalize(),
            "Score": report[cls][metric]
        })

df_plot = pd.DataFrame(plot_data)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(data=df_plot, x="Metric", y="Score", hue="Class")
plt.ylim(0, 1)
plt.title("Per-Class Evaluation Metrics", pad=20)
plt.ylabel("Score")
plt.tight_layout()

# Add value labels
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

# Save plot
plt.savefig(barplot_path)
plt.close()

print(f"Grouped metrics bar plot saved to: {barplot_path}")

# ========== WRAP UP ==========
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
print(f"\nConsole output saved to: {output_log_path}")
