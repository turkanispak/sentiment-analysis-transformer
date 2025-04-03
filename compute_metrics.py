# compute_metrics.py
# VERSION 1
import os
import torch
import json
from transformers import GPT2Tokenizer
from sklearn.metrics import classification_report, confusion_matrix
from model import GPTConfig, GPT  # This is nanoGPT's custom model class
import numpy as np

# === SETTINGS === #
model_dir = 'out-gpt2-finetune'  # or another model out-gpt2-finetune - out-customer
ckpt_path = os.path.join(model_dir, 'ckpt.pt')
test_text_path = 'data_test/test.txt'
test_labels_path = 'data_test/labels_test.txt'
results_save_path = f"results_{os.path.basename(model_dir)}.json"

# === LOAD TOKENIZER === #
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# === LOAD MODEL === #
print(f"Loading nanoGPT model from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === LOAD TEST DATA === #
with open(test_text_path, "r", encoding="utf-8") as f:
    test_texts = [line.strip() for line in f.readlines()]
with open(test_labels_path, "r", encoding="utf-8") as f:
    test_labels = [line.strip().lower() for line in f.readlines()]

assert len(test_texts) == len(test_labels), "Mismatch between test data and labels"

# === PREDICT === #
predictions = []
print(f"Running inference on {len(test_texts)} samples...")
for i, text in enumerate(test_texts):
    prompt = text + "\n### SENTIMENT:"
    enc = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    idx = enc[0]

    with torch.no_grad():
        out = model.generate(idx.unsqueeze(0), max_new_tokens=5)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    pred = decoded.split("### SENTIMENT:")[-1].strip().split()[0].lower()
    predictions.append(pred)


# === CLEANUP === #
valid_labels = ["positive", "negative", "neutral"]
predictions_clean = [p if p in valid_labels else "unknown" for p in predictions]
test_labels_clean = [l if l in valid_labels else "unknown" for l in test_labels]

# === METRICS === #
report = classification_report(test_labels_clean, predictions_clean, output_dict=True)
conf_matrix = confusion_matrix(test_labels_clean, predictions_clean, labels=valid_labels + ["unknown"])

# === SAVE === #
results = {
    "classification_report": report,
    "confusion_matrix": {
        "labels": valid_labels + ["unknown"],
        "matrix": conf_matrix.tolist()
    }
}

with open(results_save_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Evaluation complete. Results saved to: {results_save_path}")




# compute_metrics.py
"""
# VERSION 2
import os
import torch
import json
import re
from transformers import GPT2Tokenizer
from sklearn.metrics import classification_report, confusion_matrix
from model import GPTConfig, GPT
import numpy as np

# === SETTINGS === #
model_dir = 'out-customer' # or another model out-gpt2-finetune - out-customer
ckpt_path = os.path.join(model_dir, 'ckpt.pt')
test_text_path = 'data_test/test.txt'
test_labels_path = 'data_test/labels_test.txt'
results_save_path = f"results_{os.path.basename(model_dir)}_smart.json"

# === LOAD TOKENIZER === #
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# === LOAD MODEL === #
print(f"Loading nanoGPT model from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === LOAD TEST DATA === #
with open(test_text_path, "r", encoding="utf-8") as f:
    test_texts = [line.strip() for line in f.readlines()]
with open(test_labels_path, "r", encoding="utf-8") as f:
    test_labels = [line.strip().lower() for line in f.readlines()]

assert len(test_texts) == len(test_labels), "Mismatch between test data and labels"

valid_labels = ["positive", "negative", "neutral"]

# === SENTIMENT EXTRACTOR === #
def extract_sentiment(text):
    text = text.lower()
    for label in valid_labels:
        if re.search(rf"\b{label}\b", text):
            return label
    return "unknown"

# === INFERENCE === #
predictions = []
print(f"Running inference on {len(test_texts)} samples...")

for text in test_texts:
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

# === CLEANUP === #
test_labels_clean = [label if label in valid_labels else "unknown" for label in test_labels]
predictions_clean = [p if p in valid_labels else "unknown" for p in predictions]

# === METRICS === #
report = classification_report(test_labels_clean, predictions_clean, output_dict=True)
conf_matrix = confusion_matrix(test_labels_clean, predictions_clean, labels=valid_labels + ["unknown"])

# === SAVE === #
results = {
    "classification_report": report,
    "confusion_matrix": {
        "labels": valid_labels + ["unknown"],
        "matrix": conf_matrix.tolist()
    }
}

with open(results_save_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Evaluation complete. Results saved to: {results_save_path}")
"""