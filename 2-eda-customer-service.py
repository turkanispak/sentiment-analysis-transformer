import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud

# Set seaborn style
sns.set(style="whitegrid")

# Define path
data_path = os.path.join("DI725", "assignment_1", "data", "customer_service")

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
plt.figure(figsize=(6,4))
sns.countplot(x='customer_sentiment', data=train_df)
plt.title("Class Distribution in Training Set")
plt.xlabel("Customer Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

class_counts = train_df['customer_sentiment'].value_counts(normalize=True)
print("\nClass proportions:\n", class_counts)
if class_counts.min() < 0.4:
    print("\n[Warning] Potential class imbalance detected!")

# Add text length and word count columns
train_df['text_length'] = train_df['conversation'].astype(str).apply(len)
train_df['word_count'] = train_df['conversation'].astype(str).apply(lambda x: len(x.split()))

# Text length distribution
plt.figure(figsize=(6,4))
sns.histplot(train_df['text_length'], bins=30, kde=True)
plt.title("Character Count Distribution")
plt.xlabel("Text Length (Characters)")
plt.tight_layout()
plt.savefig("char_length_distribution.png")
plt.show()

# Word count distribution
plt.figure(figsize=(6,4))
sns.histplot(train_df['word_count'], bins=30, kde=True, color='orange')
plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.tight_layout()
plt.savefig("word_count_distribution.png")
plt.show()

# Average lengths per class
print("\nAverage character length per class:")
print(train_df.groupby('customer_sentiment')['text_length'].mean())
print("\nAverage word count per class:")
print(train_df.groupby('customer_sentiment')['word_count'].mean())

# Boxplot by class
plt.figure(figsize=(6,4))
sns.boxplot(x='customer_sentiment', y='text_length', data=train_df)
plt.title("Text Length by Customer Sentiment")
plt.xlabel("Customer Sentiment")
plt.ylabel("Character Length")
plt.tight_layout()
plt.savefig("text_length_by_class.png")
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
    plt.savefig(f"wordcloud_sentiment_{label}.png")
    plt.show()
