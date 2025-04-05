import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
import sys
from datetime import datetime

# Logger class to capture stdout
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # needed for compatibility
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# Setup log file and redirect stdout
log_dir = os.path.join(os.getcwd(), "eda_outputs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"eda_summary_{timestamp}.txt")
logger = Logger(log_file_path)
sys.stdout = logger

# Set seaborn style
sns.set(style="whitegrid")

# Define paths
data_path = os.path.join("data", "customer_service")
output_path = os.path.join(os.getcwd(), "eda_outputs")
os.makedirs(output_path, exist_ok=True)

# Load data
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

# Strip column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Overview
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain columns:", list(train_df.columns))
print("\nTrain sample:")
print(train_df.head())

# Check for missing values
print("\nMissing values in train set:\n", train_df.isnull().sum())
print("\nMissing values in test set:\n", test_df.isnull().sum())

# Basic stats
print("\nTrain descriptive stats:")
print(train_df.describe(include='all'))

# Check class balance
plt.figure(figsize=(6, 4))
sns.countplot(x='customer_sentiment', data=train_df)
plt.title("Class Distribution in Training Set")
plt.xlabel("Customer Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "class_distribution.png"))
plt.show()

class_counts = train_df['customer_sentiment'].value_counts(normalize=True)
print("\nClass proportions:\n", class_counts)
if class_counts.min() < 0.4:
    print("\n[Warning] Potential class imbalance detected!")

# Add text length and word count columns
train_df['text_length'] = train_df['conversation'].astype(str).apply(len)
train_df['word_count'] = train_df['conversation'].astype(str).apply(lambda x: len(x.split()))

# Text length distribution
plt.figure(figsize=(6, 4))
sns.histplot(train_df['text_length'], bins=30, kde=True)
plt.title("Character Count Distribution")
plt.xlabel("Text Length (Characters)")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "char_length_distribution.png"))
plt.show()

# Word count distribution
plt.figure(figsize=(6, 4))
sns.histplot(train_df['word_count'], bins=30, kde=True, color='orange')
plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "word_count_distribution.png"))
plt.show()

# Average lengths per class
print("\nAverage character length per class:")
print(train_df.groupby('customer_sentiment')['text_length'].mean())
print("\nAverage word count per class:")
print(train_df.groupby('customer_sentiment')['word_count'].mean())

# Boxplot by class
plt.figure(figsize=(6, 4))
sns.boxplot(x='customer_sentiment', y='text_length', data=train_df)
plt.title("Text Length by Customer Sentiment")
plt.xlabel("Customer Sentiment")
plt.ylabel("Character Length")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "text_length_by_class.png"))
plt.show()

# Word cloud per class
for label in train_df['customer_sentiment'].unique():
    text = ' '.join(train_df[train_df['customer_sentiment'] == label]['conversation'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Customer Sentiment = {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"wordcloud_sentiment_{label}.png"))
    plt.show()

# Restore original stdout and close log file
sys.stdout = logger.terminal
logger.close()
print(f"\nEDA summary saved to: {log_file_path}")
