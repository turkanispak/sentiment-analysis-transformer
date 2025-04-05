# sentiment-analysis-transformer
Assignment 1 Transformer-based Sentiment Analysis for DI725 – Transformers and Attention-Based Deep Networks

---

```markdown
# Transformer-based Sentiment Classification for Customer Service Dialogues

This project explores sentiment classification using transformer-based models: a **nanoGPT model trained from scratch**, and a **GPT-2 model fine-tuned** with pretrained weights. The task is to classify customer support dialogues into **positive**, **neutral**, or **negative** sentiment using text-generation-based prompting and supervised learning.

> 📄 Project report available in this repo: `DI725_Assignment1_Report.pdf`

---

## 📁 Project Structure

```
├── 0-check-cuda-and-wandb-integration.py   # Setup and environment check
├── 1-eda-customer-service.py               # Exploratory Data Analysis
├── 2-prepare-data-customer-service.py      # Data preprocessing for nanoGPT & GPT-2
├── 3-compute_metrics.py                    # Evaluation script for both models
├── train_customer.py                       # nanoGPT training config
├── finetune_customer.py                    # GPT-2 fine-tuning config (HuggingFace)
├── DI725_Assignment1_Report.pdf            # Final report (IEEE format)
└── data/                                   # Data files
```

---

## Setup Instructions

### 1. Create your environment
```bash
conda create -n sentiment-transformers python=3.10
conda activate sentiment-transformers
```

> Make sure to install: `torch`, `transformers`, `wandb`, `scikit-learn`, `matplotlib`, `pandas`.

---

## Phase 1: Environment Check

```bash
python 0-check-cuda-and-wandb-integration.py
```

This checks if:
- CUDA is available
- WandB is properly logged in
- PyTorch is correctly installed

---

## Phase 2: Exploratory Data Analysis

```bash
python 1-eda-customer-service.py
```

Outputs:
- Class distributions
- Sentence lengths and word counts per class
- Matplotlib plots with potential imbalance warnings

---

## Phase 3: Data Preparation

```bash
python 2-prepare-data-customer-service.py
```

This creates:
- `train.txt`, `val.txt`, and `test.txt` for **GPT-2**
- `train.bin`, `val.bin` for **nanoGPT**

---

## Phase 4: Train nanoGPT from Scratch

```bash
python train.py train_customer.py
```

🛠️ Key Configs:
- Block size: 1024
- Model: 6 layers, 6 heads, 384-dim embedding
- Max iters: 2000
- Data: `data/customer_service/train.bin`


---

## Phase 5: Fine-tune GPT-2

```bash
python finetune_customer.py
```

🛠️ Uses HuggingFace GPT-2 with:
- Custom text files
- 3-class sentiment classifier head (via prompt)
- Fine-tuning with temperature decoding

---

🔗 [View WandB Project Dashboard](https://wandb.ai/turkanispak-middle-east-technical-university/transformer-sentiment-analysis)

## Phase 6: Evaluation

```bash
python 3-compute_metrics.py
```

Runs inference using both models (based on `model_dir` flag) and outputs:
- Accuracy
- Macro & Weighted F1-scores
- Confusion Matrix (plotted and saved)
- JSON log of results

---

## Results Summary

| Model       | Accuracy | Macro F1 | Positive F1 | Neutral F1 | Negative F1 |
|-------------|----------|----------|-------------|------------|-------------|
| GPT-2       | 36.67%   | 0.227    | 0.471       | 0.438      | 0.000       |
| nanoGPT     | 36.67%   | 0.227    | 0.000       | 0.462      | 0.471       |

---

## 📄 Report & Deliverables

- Project report: `DI725_Assignment1_Report.pdf`
- Clean and modular Python code
- Public WandB experiment tracking
- GitHub version control with history

---

## Cleanup & `.gitignore`

Excluded:
- Model checkpoints (`out-customer/`, `out-gpt2-finetune/`)
- `wandb/` logs
- `.pyc`, `.DS_Store`, etc.

---

## Key Takeaways

- Pretrained weights are **invaluable** in low-data regimes.
- Even simple nanoGPT models can learn **structure**, but generalization remains a challenge.

---

```
